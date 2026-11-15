from __future__ import annotations

from datetime import datetime

import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PretrainedConfig,
)
from transformers.utils import logging as hf_logging

from aimlabs.hyperparameters import HyperParameters
from aimlabs.utils import get_logger

# Width of the pretrained head's output when a trainable hidden layer is stacked
# on top. These overrides are kept for checkpoint compatibility; any model not
# listed falls back to the first matching attribute in `_DIMENSION_FIELDS`.
_MODEL_DIMENSION_FIELD: dict[str, str] = {
    "albert-base-v2": "intermediate_size",
    "bert-base-cased": "intermediate_size",
    "bert-base-uncased": "intermediate_size",
    "distilbert-base-cased": "dim",
    "distilbert-base-uncased": "dim",
    "distilroberta-base": "intermediate_size",
    "roberta-base": "intermediate_size",
}
_DIMENSION_FIELDS = ("hidden_size", "dim", "d_model", "n_embd")

# Config attributes controlling classifier dropout, in order of preference.
_DROPOUT_FIELDS = (
    "classifier_dropout",
    "classifier_dropout_prob",
    "seq_classif_dropout",
    "hidden_dropout_prob",
)


def get_dimensions(config: PretrainedConfig) -> int:
    """Return the output width used for the pretrained classification head."""
    field = _MODEL_DIMENSION_FIELD.get(config.name_or_path or "")
    if field is not None and getattr(config, field, None):
        return int(getattr(config, field))
    for field in _DIMENSION_FIELDS:
        if value := getattr(config, field, None):
            return int(value)
    raise ValueError(
        f"Could not infer hidden dimension for {config.name_or_path!r}; "
        f"none of {_DIMENSION_FIELDS} are set in its config."
    )


def get_dropout_name(config: PretrainedConfig) -> str:
    """Return the name of the config attribute controlling classifier dropout."""
    for field in _DROPOUT_FIELDS:
        if hasattr(config, field):
            return field
    return "hidden_dropout_prob"


class Model(nn.Module):
    """
    A pretrained Hugging Face sequence classifier with an optional trainable
    hidden layer stacked on top.

    ### Args
        - `hyperparameters`: HyperParameters used to initialize the model.
        - `label2id`: mapping of label names to class indices. Defaults to
          `{"0": 0, "1": 1, ...}`.
    """

    def __init__(
        self,
        hyperparameters: HyperParameters,
        label2id: dict[str, int] | None = None,
    ) -> None:
        super().__init__()
        self.hyperparameters = hyperparameters
        self._hp = hyperparameters
        self.label2id = label2id or {str(i): i for i in range(self._hp.num_classes)}
        if len(self.label2id) != self._hp.num_classes:
            raise ValueError(
                f"label2id has {len(self.label2id)} labels but num_classes is "
                f"{self._hp.num_classes}"
            )
        self.id2label = {v: k for k, v in self.label2id.items()}
        self.version = datetime.now().strftime(f"{self._hp.version}.%Y%m%d%H%M%S")
        self.logger = get_logger(self.__class__.__name__)
        self.loss_fn = nn.CrossEntropyLoss()
        self._first_step = True

        verbosity = hf_logging.get_verbosity()
        hf_logging.set_verbosity_error()
        try:
            config = AutoConfig.from_pretrained(self._hp.model)
            if self._hp.num_hidden > 0:
                self.inner_dims = get_dimensions(config)
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    self._hp.model,
                    num_labels=self.inner_dims,
                    # the classification head is always (re)initialized for our
                    # label count, even if the checkpoint ships with one
                    ignore_mismatched_sizes=True,
                )
                self.outer_layer: nn.Module = nn.Sequential(
                    nn.Dropout(self._hp.dropout),
                    nn.Linear(self.inner_dims, self._hp.num_hidden),
                    nn.BatchNorm1d(self._hp.num_hidden),
                    nn.Linear(self._hp.num_hidden, self._hp.num_classes),
                )
            else:
                self.inner_dims = 0
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    self._hp.model,
                    num_labels=self._hp.num_classes,
                    id2label=self.id2label,
                    label2id=self.label2id,
                    ignore_mismatched_sizes=True,
                    **{get_dropout_name(config): self._hp.dropout},
                )
                self.outer_layer = nn.Identity()
            self.tokenizer = AutoTokenizer.from_pretrained(self._hp.model)
        finally:
            hf_logging.set_verbosity(verbosity)

        self.freeze()
        self.init_weights()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def init_weights(self) -> None:
        """Initialize the outer (non-pretrained) linear layers."""
        for module in self.outer_layer.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=self._hp.init_std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def freeze(self) -> None:
        """Freeze pretrained parameters according to the hyperparameters."""
        params = None
        if self._hp.hard_freeze:
            params = self.model.parameters()
        elif self._hp.freeze:
            params = self.model.base_model.parameters()
        for param in params or []:
            param.requires_grad = False

    def preprocess(self, messages: list[str]) -> dict[str, torch.Tensor]:
        """
        Tokenize a batch of messages.

        ### Args
            - `messages`: a list (batch) of messages to be preprocessed.

        ### Returns
            - A dictionary with `input_ids` and `attention_mask` tensors.
        """
        encoded = self.tokenizer(
            messages,
            return_tensors="pt",
            truncation=True,
            max_length=self._hp.max_len,
            add_special_tokens=True,
            padding="max_length",
        )
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
        }

    def forward_raw(self, messages: list[str]) -> torch.Tensor:
        """Tokenize `messages` and return logits of shape (batch, num_classes)."""
        inputs = {k: v.to(self.device) for k, v in self.preprocess(messages).items()}
        return self.forward(**inputs)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        ### Args
            - `input_ids`: token indices
            - `attention_mask`: attention masks
            - `labels`: ignored; accepted so dataset batches can be unpacked

        ### Returns
            - Tensor of shape (batch_size, num_classes) containing logits
        """
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return self.outer_layer(output.logits)

    def train_step(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute the training loss for a single batch."""
        logits = self.forward(batch["input_ids"], batch["attention_mask"])
        if self._first_step:
            self._first_step = False
            self.logger.debug(
                "shapes", _output=list(logits.shape), labels=list(batch["labels"].shape)
            )
        return self.loss_fn(logits, batch["labels"])
