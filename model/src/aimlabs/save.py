from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import structlog
import torch

from aimlabs.model import Model
from aimlabs.utils import get_logger


class ModelSaver:
    """
    Saves a model and its training artifacts to `<path>/<version>/`.

    ### Args
        - `path`: parent directory to save the model in
        - `version`: version of the model (used as the subdirectory name)
        - `logger`: structlog logger
    """

    def __init__(
        self,
        path: str | Path,
        version: str,
        logger: structlog.BoundLogger | None = None,
    ) -> None:
        self.path = Path(path).joinpath(version)
        self.logger = logger or get_logger(self.__class__.__name__)

    def _write_json(self, name: str, data: Any) -> Path:
        self.path.mkdir(parents=True, exist_ok=True)
        path = self.path.joinpath(name)
        path.write_text(json.dumps(data, indent=2, default=str))
        self.logger.info("_save_", **{path.stem: str(path)})
        return path

    def save_state_dict(self, model: Model) -> Path:
        self.path.mkdir(parents=True, exist_ok=True)
        path = self.path.joinpath("state_dict.pt")
        torch.save(model.state_dict(), path)
        self.logger.info("_save_", state_dict=str(path))
        return path

    def save_hyperparameters(self, model: Model) -> Path:
        return self._write_json(
            "hyperparameters.json", model.hyperparameters.model_dump(mode="json")
        )

    def save_config(self, model: Model) -> Path:
        return self._write_json("config.json", model.model.config.to_dict())

    def save_label2id(self, model: Model) -> Path:
        return self._write_json("label2id.json", model.label2id)

    def save_metrics(self, metrics: dict[str, Any]) -> Path:
        return self._write_json("metrics.json", metrics)

    def save(self, model: Model, metrics: dict[str, Any]) -> None:
        if not model.hyperparameters.save:
            self.logger.info("nosave")
            return
        self.save_state_dict(model)
        self.save_hyperparameters(model)
        self.save_config(model)
        self.save_metrics(metrics)
        self.save_label2id(model)
