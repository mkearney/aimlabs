from pydantic import BaseModel, model_validator


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
        - `test_max_steps`: max number of test iterations to complete.
        - `valid_max_steps`: max number of valid iterations to complete
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
    lr_patience: int = 0
    lr: float = 5e-5
    max_len: int = 32
    model: str = "distilbert-base-uncased"
    name: str = "nlpmodel"
    num_classes: int = 2
    num_epochs: int = 32
    num_hidden: int = 1536
    num_steps: int = 16
    save: bool = False
    test_batch_size: int = -1
    test_max_steps: int = -2
    valid_batch_size: int = -1
    valid_max_steps: int = -2
    version: str = "0.1.0"

    @model_validator(mode="after")
    @classmethod
    def check_max_steps(cls, data):
        if data.test_max_steps < -1:
            data.test_max_steps = data.num_steps
        if data.valid_max_steps < -1:
            data.valid_max_steps = data.num_steps
        if data.test_batch_size < 0:
            data.test_batch_size = data.batch_size
        if data.valid_batch_size < 0:
            data.valid_batch_size = data.batch_size
        return data
