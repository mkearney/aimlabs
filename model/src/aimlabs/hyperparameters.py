from pydantic import BaseModel


class HyperParameters(BaseModel):
    """
    Hyperparameters for language model

    ### Attributes
        - `batch_size`: batch size for training
        - `best_metric`: metric to use for early stopping
        - `dropout`: dropout rate
        - `early_stopping_patience`: number of epochs without improvement
        - `eps`: epsilon for numerical stability
        - `freeze`: whether to freeze layers
        - `gamma`: discount factor
        - `hard_freeze`: whether to hard freeze layers
        - `init_std`: variance of initialization
        - `limit_test_steps`: whether to apply num_steps to test set
        - `lr_patience`: patience for learning rate scheduler
        - `lr`: learning rate
        - `max_len`: maximum length of a sequence
        - `model`: model to use
        - `name`: model name
        - `num_classes`: number of classes
        - `num_epochs`: number of epochs
        - `num_hidden`: number of trainable hidden dimensions
        - `num_steps`: number of steps
        - `save`: whether to save model
        - `version`: model version
    """

    batch_size: int = 16
    best_metric: str = "val_loss"
    dropout: float = 0.15
    early_stopping_patience: int = 5
    eps: float = 1e-10
    freeze: bool = True
    gamma: float = 0.8
    hard_freeze: bool = False
    init_std: float = 0.015
    limit_test_steps: bool = False
    lr_patience: int = 0
    lr: float = 5e-5
    max_len: int = 32
    model: str = "distilbert-base-uncased"
    name: str = "nlpmodel"
    num_classes: int = 2
    num_epochs: int = 32
    num_hidden: int = 2304
    num_steps: int = 16
    save: bool = False
    version: str = "0.1.0"
