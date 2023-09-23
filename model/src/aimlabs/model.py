from datetime import datetime
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from aimlabs.hyperparameters import HyperParameters
from transformers import AutoModelForSequenceClassification, AutoTokenizer, logging


class Model(nn.Module):
    """
    Initialize a pretrained torch module model

    ### Args
        - `hyperparameters` HyperParameters used to initialize the model.
        - `label_map` A map from target label to target index.
    """

    def __init__(
        self,
        hyperparameters: HyperParameters,
        label_map: Optional[Dict[str, int]] = None,
    ):
        super(Model, self).__init__()
        # model settings
        self.version = datetime.now().strftime(
            f"{hyperparameters.version}.%Y%m%d%H%M%S"
        )
        if label_map:
            self.label_map = label_map
        else:
            self.label_map: Dict[str, int] = {
                str(i): i for i in range(hyperparameters.num_classes)
            }
        self.hyperparameters = hyperparameters
        self.max_len = hyperparameters.max_len

        # model architecture
        logging.set_verbosity_error()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            hyperparameters.model,
            num_labels=hyperparameters.num_classes,
            max_length=hyperparameters.max_len,
            output_hidden_states=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            hyperparameters.model, **self.model.config.__dict__
        )
        logging.set_verbosity_warning()
        if hyperparameters.freeze:
            self.freeze()
        self.fc = nn.Linear(
            self.model.config.dim * 2,
            self.hyperparameters.num_classes,
        )
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)

    def init_weights(self, modules):
        for param in modules.parameters():  # type: ignore
            if param.dim() > 1:
                param.data.normal_(mean=0.0, std=self.hyperparameters.init_std)
            else:
                param.data.zero_()

    def freeze(self):
        """Freeze base model parameters"""
        for param in self.model.base_model.parameters():  # type: ignore
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
            max_length=self.max_len,
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
        targets: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of neural network

        ### Args
            - `input_ids` Indices
            - `attention_mask` Masks
            - `targets` Labels, optional

        ### Returns
            - Tensor of shape (batch_size, num_classes) containing output logits
        """
        outputs = self.model(
            input_ids=input_ids, attention_mask=attention_mask
        ).hidden_states[-1]
        max_pooled = self.max_pool(outputs.permute(0, 2, 1)).squeeze(-1)
        avg_pooled = self.avg_pool(outputs.permute(0, 2, 1)).squeeze(-1)
        pooled = torch.cat((max_pooled, avg_pooled), dim=1)
        return self.fc(pooled)
