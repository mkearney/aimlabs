from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from aimlabs.model import Model
from common.metrics import Fit


def move_to(batch: dict[str, torch.Tensor], device: torch.device) -> dict:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


@torch.inference_mode()
def evaluate(
    model: Model,
    dataloader: DataLoader,
    criterion: torch.nn.Module,
    train_loss: float,
    max_steps: int = -1,
) -> dict[str, float]:
    """
    Evaluate a model on (up to `max_steps` batches of) a dataloader.

    ### Args
        - `model`: model
        - `dataloader`: dataloader yielding `InputsDataset` batches
        - `criterion`: loss function
        - `train_loss`: training loss to report alongside the eval metrics
        - `max_steps`: maximum number of batches; <= 0 means all

    ### Returns
        - dict with `loss`, `val_loss`, `val_acc`, `val_f1`, `val_pr`, `val_rc`
    """
    device = model.device
    fit = Fit(num_classes=model.hyperparameters.num_classes, device=device)
    model.eval()
    total_loss, n = 0.0, 0
    for i, batch in enumerate(dataloader):
        if 0 < max_steps <= i:
            break
        batch = move_to(batch, device)
        labels = batch["labels"].long()
        outputs = model(**batch)
        total_loss += criterion(outputs, labels).item() * len(labels)
        n += len(labels)
        fit.update(outputs, labels)
    if n == 0:
        raise ValueError("evaluation dataloader produced no batches")
    stats = fit.compute()
    return {
        "loss": float(train_loss),
        "val_loss": total_loss / n,
        "val_acc": stats.acc,
        "val_f1": stats.f1,
        "val_pr": stats.pr,
        "val_rc": stats.rc,
    }
