from pydantic import BaseModel


class HyperParameters(BaseModel):
    """
    Hyperparameters for language model

    ### Attributes
        - `batch_size`: batch size for training
        - `best_metric`: metric to use for early stopping
        - `dropout`: dropout rate
        - `early_stopping_patience`: number of epochs without improvement
        - `freeze`: whether to freeze layers
        - `gamma`: discount factor
        - `init_std`: variance of initialization
        - `lr_patience`: patience for learning rate scheduler
        - `lr`: learning rate
        - `max_len`: maximum length of a sequence
        - `model`: model to use
        - `name`: model name
        - `num_classes`: number of classes
        - `num_epochs`: number of epochs
        - `num_output_dims`: number of dimensions in output model
        - `num_steps`: number of steps
        - `save_model`: whether to save model
        - `version`: model version
    """

    batch_size: int = 32
    best_metric: str = "val_loss"
    dropout: float = 0.2
    early_stopping_patience: int = 5
    freeze: bool = True
    gamma: float = 0.8
    init_std: float = 1.5
    lr_patience: int = 0
    lr: float = 2e-04
    max_len: int = 16
    model: str = "distilbert-base-uncased"
    name: str = "nlpmodel"
    num_classes: int = 2
    num_epochs: int = 32
    num_output_dims: int = 16
    num_steps: int = 16
    save_model: bool = False
    version: str = "0.1.0"
