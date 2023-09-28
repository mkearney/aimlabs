from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from aimlabs.hyperparameters import HyperParameters
from aimlabs.utils import get_logger
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    logging,
)

# create dict containing name of intermediate or hidden output dimension
# for each model based on the default configuration file
model_dimensions_name: Dict[str, str] = defaultdict(lambda: "hidden_size")
model_dimensions_name.update(
    {
        "albert-base-v2": "intermediate_size",
        "bert-base-cased": "intermediate_size",
        "bert-base-uncased": "intermediate_size",
        "distilbert-base-cased": "dim",
        "distilbert-base-uncased": "dim",
        "distilroberta-base": "intermediate_size",
        "roberta-base": "intermediate_size",
    }
)


model_dropout_name: Dict[str, str] = defaultdict(lambda: "hidden_dropout_prob")
model_dropout_name.update(
    {
        "albert-base-v2": "classifier_dropout_prob",
        "bert-base-cased": "classifier_dropout",
        "bert-base-uncased": "classifier_dropout",
        "distilbert-base-cased": "seq_classif_dropout",
        "distilbert-base-uncased": "seq_classif_dropout",
        "distilroberta-base": "classifier_dropout",
        "roberta-base": "classifier_dropout",
    }
)


def get_dimensions(config: AutoConfig) -> int:
    field = model_dimensions_name[config.__dict__["_name_or_path"]]
    if field == "hidden_size" and "intermediate_size" in model_dimensions_name.__dict__:
        field = "intermediate_size"
    return config.__dict__[field]


def get_dropout_name(config: AutoConfig) -> str:
    return model_dropout_name[config.__dict__["_name_or_path"]]


class Model(nn.Module):
    """
    Initialize a pretrained torch module model

    ### Args
        - `hyperparameters` HyperParameters used to initialize the model.
    """

    def __init__(
        self,
        hyperparameters: HyperParameters,
        label2id: Optional[Dict[str, int]] = None,
    ):
        super(Model, self).__init__()
        # model settings
        self.hyperparameters = hyperparameters
        self._hp = self.hyperparameters
        self.label2id = label2id or {str(i): i for i in range(self._hp.num_classes)}
        self.id2label = {v: k for k, v in self.label2id.items()}
        self.version = datetime.now().strftime(f"{self._hp.version}.%Y%m%d%H%M%S")
        self.logger = get_logger(self.__class__.__name__)
        self.loss_fn = nn.CrossEntropyLoss()
        self.first_step = True
        logging.set_verbosity_error()

        # model architecture
        config = AutoConfig.from_pretrained(self._hp.model)
        if self._hp.num_hidden > 0:
            self.inner_dims = get_dimensions(config)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self._hp.model,
                num_labels=self.inner_dims,
                max_length=self._hp.max_len,
                output_hidden_states=False,
            )
            self.outer_layer = nn.Sequential(
                nn.Dropout(self._hp.dropout),
                nn.Linear(self.inner_dims, self._hp.num_hidden),
                nn.BatchNorm1d(self._hp.num_hidden),
                nn.Linear(self._hp.num_hidden, self._hp.num_classes),
            )
        else:
            self.inner_dims = 0
            dropout_arg = {get_dropout_name(config): self._hp.dropout}
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self._hp.model,
                num_labels=self._hp.num_classes,
                max_length=self._hp.max_len,
                output_hidden_states=False,
                **dropout_arg,
            )
            self.outer_layer = nn.Identity()

        self.tokenizer = AutoTokenizer.from_pretrained(
            self._hp.model, **self.model.config.__dict__
        )
        logging.set_verbosity_warning()
        self.freeze()
        self.init_weights()

    def init_weights(self):
        if self.inner_dims > 0:
            for module in self.outer_layer:  # type: ignore
                for name, param in module.named_parameters():
                    if "linear" in name:
                        if param.dim() > 1:
                            param.data.normal_(mean=0.0, std=self._hp.init_std)
                        else:
                            param.data.zero_()

    def freeze(self):
        """Freeze base model parameters"""
        if self._hp.freeze:
            for param in self.model.base_model.parameters():  # type: ignore
                param.requires_grad = False
        if self._hp.hard_freeze:
            for param in self.model.parameters():  # type: ignore
                param.requires_grad = False

    def preprocess(self, messages: List[str]) -> Dict[str, torch.Tensor]:
        """
        Preprocess messages

        ### Args
            - `messages` A list (batch) of messages to be preprocessed.

        ### Returns
            - A dictionary of tokenization results.
        """
        return self.tokenizer(
            messages,
            return_tensors="pt",
            truncation=True,
            max_length=self._hp.max_len,
            add_special_tokens=True,
            padding="max_length",
        )  # type: ignore

    def forward_raw(self, messages: List[str]) -> torch.Tensor:
        """
        Forward pass of neural network

        ### Args
            - `messages` A list (batch) of messages to be processed

        ### Returns
            - Tensor of shape (batch_size, num_classes) containing output logits
        """
        inputs = self.preprocess(messages)
        return self.forward(**inputs)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of neural network

        ### Args
            - `input_ids` Indices
            - `attention_mask` Masks
            - `labels` Labels, optional

        ### Returns
            - Tensor of shape (batch_size, num_classes) containing output logits
        """
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return self.outer_layer(output.logits)

    def train_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        output = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        logits = self.outer_layer(output.logits)
        if self.first_step:
            self.first_step = False
            self.logger.info(
                "shapes", _output=list(logits.shape), labels=list(batch["labels"].shape)
            )
        loss = self.loss_fn(logits, batch["labels"])
        return loss
