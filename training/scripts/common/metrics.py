from __future__ import annotations

import json
from pathlib import Path

import torch
from pydantic import BaseModel
from torchmetrics import Accuracy, F1Score, MetricCollection, Precision, Recall

LOWER_IS_BETTER = frozenset({"loss", "val_loss"})


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
    Streaming classification metrics. Call `update()` for every batch and
    `compute()` once at the end, so metrics are computed over the whole
    dataset rather than averaged across batches.

    ### Args
        - `num_classes`: number of classes
        - `average`: how to average the metrics, i.e., "micro" or "macro"
        - `device`: device the predictions live on

    ### Examples
    ```python
    import torch

    fit = Fit(num_classes=4)
    fit.update(torch.randn(8, 4), torch.randint(0, 4, (8,)))
    fit.compute()
    ```
    """

    def __init__(
        self,
        num_classes: int,
        average: str = "macro",
        device: torch.device | str = "cpu",
    ) -> None:
        kwargs = {"task": "multiclass", "num_classes": num_classes, "average": average}
        self.collection = MetricCollection(
            {
                "acc": Accuracy(**kwargs),
                "f1": F1Score(**kwargs),
                "pr": Precision(**kwargs),
                "rc": Recall(**kwargs),
            }
        ).to(device)

    def update(self, output: torch.Tensor, target: torch.Tensor) -> None:
        self.collection.update(output, target)

    def compute(self) -> FitStatistics:
        values = self.collection.compute()
        return FitStatistics(**{k: float(v) for k, v in values.items()})

    def reset(self) -> None:
        self.collection.reset()

    def __call__(self, output: torch.Tensor, target: torch.Tensor) -> FitStatistics:
        """Compute metrics for a single batch (does not accumulate)."""
        self.reset()
        self.update(output, target)
        stats = self.compute()
        self.reset()
        return stats


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
    """History of per-epoch metrics."""

    def __init__(self) -> None:
        self.metrics: list[EpochMetrics] = []

    def append(self, epoch: int, metrics: dict[str, float]) -> None:
        self.metrics.append(EpochMetrics(epoch=epoch, **metrics))

    def collect(self) -> dict[str, list[float]]:
        """Column-oriented view: `{"epoch": [...], "loss": [...], ...}`."""
        return {
            field: [getattr(m, field) for m in self.metrics]
            for field in EpochMetrics.model_fields
        }

    def save(self, output_dir: str | Path, version: str = "") -> Path:
        suffix = f"-{version}" if version else ""
        path = Path(output_dir) / f"metrics{suffix}.json"
        path.write_text(json.dumps(self.collect(), indent=2))
        return path

    def __len__(self) -> int:
        return len(self.metrics)


class BestMetric:
    """
    Tracks the best value of a metric, keeps a snapshot of the corresponding
    model weights, and counts epochs without improvement for early stopping.
    """

    def __init__(self, metric: str = "val_loss") -> None:
        self.metric = metric
        self.mode = "min" if metric in LOWER_IS_BETTER else "max"
        self.alt_metric = "val_acc" if metric in LOWER_IS_BETTER else "val_loss"
        self.epoch = 0
        self.value = float("inf") if self.mode == "min" else -float("inf")
        self.alt_value = float("nan")
        self.state_dict: dict[str, torch.Tensor] = {}
        self.early_stop_counter = 0
        self.trn_loss = float("inf")

    def is_improvement(self, new: float) -> bool:
        return new < self.value if self.mode == "min" else new > self.value

    def __call__(
        self, epoch: int, metrics: dict[str, float], model: torch.nn.Module
    ) -> bool:
        """Record an epoch's metrics. Returns True if it was an improvement."""
        new_value = float(metrics[self.metric])
        if not self.is_improvement(new_value):
            self.early_stop_counter += 1
            return False
        self.epoch = epoch
        self.value = new_value
        self.alt_value = float(metrics[self.alt_metric])
        # `state_dict()` returns references to live parameters; snapshot them
        # (on cpu) so later optimizer steps don't overwrite the best weights.
        self.state_dict = {
            k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()
        }
        self.early_stop_counter = 0
        self.trn_loss = float(metrics["loss"])
        return True
