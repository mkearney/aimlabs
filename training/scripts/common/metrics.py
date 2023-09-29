import json
from typing import Dict, List, Union

from pydantic import BaseModel
from torchmetrics import Accuracy, F1Score, Precision, Recall

from aimlabs.model import Model


class FitStatistics(BaseModel):
    """
    Fit metrics for classification.

    ### Attributes
        - `acc`: accuracy
        - `f1`: f1 score
        - `pr`: precision
        - `rc`: recall
    """

    acc: float
    f1: float
    pr: float
    rc: float


class Fit:
    """
    A fit estimator class that when called returns FitStatistics object.

    ### Args
        - `num_classes`: number of classes
        - `task`: task type
        - `average`: how to average the metrics, i.e., "micro" or "macro"

    ### Examples
    ```python
    import torch

    # set number of classes and initialize estimator
    num_classes = 4
    fit = Fit(num_classes=num_classes)

    # create example tensors
    output = torch.randn(8, num_classes)
    labels = torch.randint(0, num_classes, (8,))

    # apply estimator
    fit(output, labels)
    """

    def __init__(
        self,
        num_classes: int,
        task: str = "multiclass",
        average: str = "macro",
    ):
        self.num_classes = num_classes
        self.average = average
        self.task = task
        self.accuracy = Accuracy(
            task=self.task,  # type: ignore
            num_classes=num_classes,
            average=self.average,  # type: ignore
        )
        self.f1_score = F1Score(
            task=self.task,  # type: ignore
            num_classes=num_classes,
            average=self.average,  # type: ignore
        )
        self.precision = Precision(
            task=self.task,  # type: ignore
            num_classes=num_classes,
            average=self.average,  # type: ignore
        )
        self.recall = Recall(
            task=self.task,  # type: ignore
            num_classes=num_classes,
            average=self.average,  # type: ignore
        )

    def __call__(self, output, target) -> FitStatistics:
        return FitStatistics(
            acc=self.accuracy(output, target),
            f1=self.f1_score(output, target),
            pr=self.precision(output, target),
            rc=self.recall(output, target),
        )


class EpochMetrics(BaseModel):
    """
    Metrics for a given epoch.

    ### Attributes
        - `epoch`: epoch number
        - `loss`: training loss
        - `val_loss`: validation loss
        - `val_acc`: validation accuracy
        - `val_f1`: validation F1 score
        - `val_pr`: validation precision
        - `val_rc`: validation recall
    """

    epoch: int
    loss: float
    val_loss: float
    val_acc: float
    val_f1: float
    val_pr: float
    val_rc: float


class Metrics:
    metrics: List[EpochMetrics] = []

    def collect(self) -> Dict[str, Union[List[int], List[float]]]:
        return {
            "epoch": self.epoch(),
            "loss": self.loss(),
            "val_loss": self.val_loss(),
            "val_acc": self.val_acc(),
            "val_f1": self.val_f1(),
            "val_pr": self.val_pr(),
            "val_rc": self.val_rc(),
        }

    def epoch(self) -> List[int]:
        return [m.epoch for m in self.metrics]

    def loss(self) -> List[float]:
        return [m.loss for m in self.metrics]

    def val_loss(self) -> List[float]:
        return [m.val_loss for m in self.metrics]

    def val_acc(self) -> List[float]:
        return [m.val_acc for m in self.metrics]

    def val_f1(self) -> List[float]:
        return [m.val_f1 for m in self.metrics]

    def val_pr(self) -> List[float]:
        return [m.val_pr for m in self.metrics]

    def val_rc(self) -> List[float]:
        return [m.val_rc for m in self.metrics]

    def append(self, epoch: int, metrics: Dict[str, Union[int, float]]) -> None:
        epoch_metrics = EpochMetrics(epoch=epoch, **metrics)
        self.metrics.append(epoch_metrics)

    def save(
        self,
        output_dir: str,
        version: str = "",
    ) -> str:
        version = "" if version == "" else f"-{version}"
        save_as = f"{output_dir}/metrics{version}.json"
        data = self.collect()
        with open(save_as, "w") as f:
            json.dump(data, f, indent=2)
        return save_as


class BestMetric:
    def __init__(self, metric: str = "val_loss"):
        self.metric = metric
        self.mode = self.get_mode(metric)
        self.epoch = 0
        self.value = self.get_value(metric)
        self.alt_metric = self.get_alt_metric(metric)
        self.alt_value = self.get_value(self.alt_metric)
        self.state_dict = {}
        self.early_stop_counter = 0
        self.trn_loss = float("inf")

    def get_mode(self, metric: str) -> str:
        if metric in ["loss", "val_loss"]:
            return "min"
        return "max"

    def get_value(self, metric: str) -> float:
        if self.mode == "max":
            return -float("inf")
        return float("inf")

    def get_alt_metric(self, metric: str) -> str:
        if metric in ["loss", "val_loss"]:
            return "val_acc"
        return "val_loss"

    def get_alt_value(self, metric: str) -> float:
        if self.mode == "max":
            return -float("inf")
        return float("inf")

    def is_improvement(self, new: float) -> bool:
        if self.mode == "max":
            return new > self.value
        return new < self.value

    def __call__(self, epoch: int, metrics: Dict[str, float], model: Model):
        new_value = metrics[self.metric]
        new_alt_value = metrics[self.alt_metric]
        if self.is_improvement(new_value):
            self.epoch = epoch
            self.value = new_value
            self.alt_value = new_alt_value
            self.epoch = epoch
            self.state_dict = model.state_dict()
            self.early_stop_counter = 0
            self.trn_loss = metrics["loss"]
        else:
            self.early_stop_counter += 1
