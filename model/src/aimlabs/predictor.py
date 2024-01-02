import json
from pathlib import Path
from typing import Generator, List, Union

import torch
from aimlabs.hyperparameters import HyperParameters
from pydantic import BaseModel

from aimlabs.model import Model


class Prediction(BaseModel):
    text: str
    label: str
    proba: List[float]


class Predictions:
    def __init__(self, predictions: List[Prediction]):
        self.predictions = predictions

    def __dict__(self):
        return {
            "text": [prediction.text for prediction in self.predictions],
            "label": [prediction.label for prediction in self.predictions],
            "proba": [prediction.proba for prediction in self.predictions],
        }

    def __str__(self):
        return str(self.__dict__())

    def __repr__(self):
        return self.__str__()

    def __iter__(self):
        return iter(self.predictions)


class Predictor:
    def __init__(self, path: str):
        self.model = load_model(Path(path))

    def predict(
        self,
        messages: Union[str, List[str]],
        batch_size: int = 50,
    ) -> Predictions:
        return self.predict_batch(
            [messages] if isinstance(messages, str) else messages,
            batch_size,
        )

    def batch(
        self,
        msgs: List[str],
        bs: int,
    ) -> Generator[List[str], List[str], None]:
        for i in range(0, len(msgs), bs):
            yield msgs[i : i + bs]  # noqa

    def predict_batch(
        self,
        msgs: List[str],
        bs: int,
    ) -> Predictions:
        batches = [
            self.model(**self.model.preprocess(b)) for b in self.batch(msgs, bs)
        ]
        probas = torch.stack([row for batch in batches for row in batch])
        probas = torch.softmax(probas, 1)
        labels = [self.model.id2label[i] for i in probas.argmax(1).tolist()]
        return Predictions(
            predictions=[
                Prediction(text=text, label=label, proba=proba)
                for text, label, proba in zip(msgs, labels, probas.tolist())
            ],
        )


def load_model(path: Path) -> Model:
    with path.joinpath("hyperparameters.json").open("r") as f:
        hp = HyperParameters(**json.load(f))
    if path.joinpath("label2id.json").is_file():
        with path.joinpath("label2id.json").open("r") as f:
            label2id = json.load(f)
    else:
        label2id = None
    model = Model(hyperparameters=hp, label2id=label2id)
    state_dict = torch.load(path.joinpath("state_dict.pt"))
    model.load_state_dict(state_dict)
    model.eval()
    return model
