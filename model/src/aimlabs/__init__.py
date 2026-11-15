"""Fine-tune and serve transformer-based text classification models."""

from importlib.metadata import PackageNotFoundError, version

from aimlabs.data import InputsDataset
from aimlabs.hyperparameters import HyperParameters
from aimlabs.model import Model
from aimlabs.predictor import Prediction, Predictions, Predictor, load_model
from aimlabs.save import ModelSaver

try:
    __version__ = version("aimlabs")
except PackageNotFoundError:  # pragma: no cover - running from a source tree
    __version__ = "0.0.0"

__all__ = [
    "HyperParameters",
    "InputsDataset",
    "Model",
    "ModelSaver",
    "Prediction",
    "Predictions",
    "Predictor",
    "__version__",
    "load_model",
]
