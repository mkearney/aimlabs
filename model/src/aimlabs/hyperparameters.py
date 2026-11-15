from typing import Literal

from pydantic import BaseModel, Field, model_validator


class HyperParameters(BaseModel):
    """
    Hyperparameters for a fine-tuned text classification model.

    ### Attributes
        - `batch_size`: batch size for training
        - `best_metric`: metric used for checkpoint selection and early stopping
        - `dropout`: dropout rate
        - `early_stopping_patience`: number of epochs without improvement
        - `eps`: epsilon for optimizer numerical stability
        - `freeze`: whether to freeze the pretrained base model
        - `gamma`: factor by which the learning rate is reduced on plateau
        - `hard_freeze`: whether to freeze *all* pretrained parameters (incl. head)
        - `init_std`: standard deviation used to initialize the outer layers
        - `lr_patience`: patience for the learning rate scheduler
        - `lr`: learning rate
        - `max_len`: maximum length (in tokens) of a sequence
        - `model`: name or path of the pretrained Hugging Face model
        - `name`: model name (used for the output directory)
        - `num_classes`: number of classes
        - `num_epochs`: maximum number of epochs
        - `num_hidden`: width of the trainable hidden layer (0 disables it)
        - `num_steps`: training steps per epoch (<= 0 means the full dataset)
        - `save`: whether to save the model
        - `seed`: random seed (None leaves RNGs unseeded)
        - `test_batch_size`: test batch size (< 0 means `batch_size`)
        - `test_max_steps`: max test steps (-1 means all, < -1 means `num_steps`)
        - `valid_batch_size`: validation batch size (< 0 means `batch_size`)
        - `valid_max_steps`: max validation steps (-1 means all,
          < -1 means `num_steps`)
        - `version`: model version prefix
    """

    batch_size: int = Field(default=16, gt=0)
    best_metric: Literal[
        "loss", "val_loss", "val_acc", "val_f1", "val_pr", "val_rc"
    ] = "val_loss"
    dropout: float = Field(default=0.15, ge=0.0, lt=1.0)
    early_stopping_patience: int = Field(default=5, ge=0)
    eps: float = Field(default=1e-10, gt=0.0)
    freeze: bool = True
    gamma: float = Field(default=0.8, gt=0.0, le=1.0)
    hard_freeze: bool = False
    init_std: float = Field(default=0.015, gt=0.0)
    lr_patience: int = Field(default=0, ge=0)
    lr: float = Field(default=5e-5, gt=0.0)
    max_len: int = Field(default=32, gt=0)
    model: str = "distilbert-base-uncased"
    name: str = "nlpmodel"
    num_classes: int = Field(default=2, ge=2)
    num_epochs: int = Field(default=32, gt=0)
    num_hidden: int = Field(default=1536, ge=0)
    num_steps: int = 16
    save: bool = False
    seed: int | None = None
    test_batch_size: int = -1
    test_max_steps: int = -2
    valid_batch_size: int = -1
    valid_max_steps: int = -2
    version: str = "0.1.0"

    @model_validator(mode="after")
    def _resolve_defaults(self) -> "HyperParameters":
        if self.test_max_steps < -1:
            self.test_max_steps = self.num_steps
        if self.valid_max_steps < -1:
            self.valid_max_steps = self.num_steps
        if self.test_batch_size <= 0:
            self.test_batch_size = self.batch_size
        if self.valid_batch_size <= 0:
            self.valid_batch_size = self.batch_size
        return self
