from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from pathlib import Path

import torch
from pydantic import BaseModel

from aimlabs.hyperparameters import HyperParameters
from aimlabs.model import Model


class Prediction(BaseModel):
    """A single prediction: input text, predicted label, class probabilities."""

    text: str
    label: str
    proba: list[float]


class Predictions(Sequence[Prediction]):
    """A sequence of `Prediction`s with a column-oriented `to_dict()` view."""

    def __init__(self, predictions: list[Prediction]) -> None:
        self.predictions = predictions

    def to_dict(self) -> dict[str, list]:
        return {
            "text": [p.text for p in self.predictions],
            "label": [p.label for p in self.predictions],
            "proba": [p.proba for p in self.predictions],
        }

    def __getitem__(self, idx):  # type: ignore[override]
        return self.predictions[idx]

    def __len__(self) -> int:
        return len(self.predictions)

    def __iter__(self) -> Iterator[Prediction]:
        return iter(self.predictions)

    def __repr__(self) -> str:
        return f"Predictions({self.to_dict()})"

    __str__ = __repr__


def batched(items: list[str], size: int) -> Iterator[list[str]]:
    """Yield successive chunks of `items` of length `size`."""
    if size <= 0:
        raise ValueError(f"batch size must be positive, got {size}")
    for i in range(0, len(items), size):
        yield items[i : i + size]


class Predictor:
    """
    Load a saved model directory and classify messages.

    ### Args
        - `path`: directory written by `ModelSaver`
        - `device`: torch device to run on (defaults to cpu)

    ### Examples
    ```python
    predictor = Predictor("models/imdbsentiment/0.1.0.20240101000000")
    predictor.predict(["what a great film", "utterly boring"])
    ```
    """

    def __init__(
        self, path: str | Path, device: str | torch.device | None = None
    ) -> None:
        self.model = load_model(Path(path), map_location=device)

    def predict(self, messages: str | list[str], batch_size: int = 50) -> Predictions:
        msgs = [messages] if isinstance(messages, str) else list(messages)
        return self.predict_batch(msgs, batch_size)

    @staticmethod
    def batch(msgs: list[str], bs: int) -> Iterator[list[str]]:
        return batched(msgs, bs)

    @torch.inference_mode()
    def predict_batch(self, msgs: list[str], bs: int) -> Predictions:
        if not msgs:
            return Predictions([])
        logits = torch.cat([self.model.forward_raw(b) for b in batched(msgs, bs)])
        probas = torch.softmax(logits, dim=1).cpu()
        labels = [self.model.id2label[i] for i in probas.argmax(1).tolist()]
        return Predictions(
            [
                Prediction(text=text, label=label, proba=proba)
                for text, label, proba in zip(
                    msgs, labels, probas.tolist(), strict=True
                )
            ]
        )


def load_model(
    path: str | Path, map_location: str | torch.device | None = None
) -> Model:
    """Rebuild a `Model` from a directory written by `ModelSaver`."""
    path = Path(path)
    hp = HyperParameters.model_validate_json(
        path.joinpath("hyperparameters.json").read_text()
    )
    label2id_path = path.joinpath("label2id.json")
    label2id = (
        json.loads(label2id_path.read_text()) if label2id_path.is_file() else None
    )
    model = Model(hyperparameters=hp, label2id=label2id)
    state_dict = torch.load(
        path.joinpath("state_dict.pt"),
        map_location=map_location or "cpu",
        weights_only=True,
    )
    model.load_state_dict(state_dict)
    model.to(map_location or "cpu")
    model.eval()
    return model
