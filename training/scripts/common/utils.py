from __future__ import annotations

import json
from argparse import Namespace
from collections.abc import Iterator
from datetime import timedelta
from pathlib import Path
from typing import Any

import torch

from aimlabs.hyperparameters import HyperParameters


def batch_messages(messages: list[str], batch_size: int) -> Iterator[list[str]]:
    """Split a list of messages into batches of `batch_size`."""
    for i in range(0, len(messages), batch_size):
        yield messages[i : i + batch_size]


def save_hypers(params: dict[str, Any], output_dir: str | Path, version: str) -> Path:
    """Save hyperparameters to `<output_dir>/hyperparameters[-<version>].json`."""
    suffix = f"-{version}" if version else ""
    path = Path(output_dir) / f"hyperparameters{suffix}.json"
    path.write_text(json.dumps(params, indent=2))
    return path


def get_hyperparameters_from_args(args: Namespace) -> HyperParameters:
    """Build HyperParameters from parsed CLI args, ignoring unset (None) values."""
    kwargs = {
        k: v
        for k, v in vars(args).items()
        if k in HyperParameters.model_fields and v is not None
    }
    return HyperParameters(**kwargs)


def model_size(model: torch.nn.Module) -> dict[str, str]:
    """Return the number of (trainable) parameters and memory footprint."""
    tensors = [*model.parameters(), *model.buffers()]
    parameters = sum(t.numel() for t in tensors)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    memory = sum(t.nelement() * t.element_size() for t in tensors) / 1024**2
    return {
        "parameters": f"{parameters:,}",
        "trainable": f"{trainable:,}",
        "memory": f"{memory:,.1f}MB",
    }


def format_duration(duration: timedelta) -> dict[str, str]:
    seconds = duration.total_seconds()
    if seconds > 3600:
        return {"hours": f"{seconds / 3600:,.2f}"}
    if seconds > 120:
        return {"minutes": f"{seconds / 60:,.2f}"}
    return {"seconds": f"{seconds:,.2f}"}


def _fmt(metrics: dict[str, float], keys: dict[str, str]) -> dict[str, str]:
    return {alias: f"{metrics[key]:.4f}" for alias, key in keys.items()}


def log_metrics(epoch: int, epoch_lr: float, metrics: dict[str, float], logger) -> None:
    fields = _fmt(
        metrics,
        {
            "_trn": "loss",
            "_val": "val_loss",
            "acc": "val_acc",
            "f1": "val_f1",
            "pr": "val_pr",
            "rc": "val_rc",
        },
    )
    logger.info(f"{epoch:^6d}".replace(" ", "_"), __lr=f"{epoch_lr:.5f}", **fields)


def log_test_metrics(metrics: dict[str, float], logger) -> None:
    fields = _fmt(
        metrics,
        {
            "_tst": "val_loss",
            "acc": "val_acc",
            "f1": "val_f1",
            "pr": "val_pr",
            "rc": "val_rc",
        },
    )
    logger.info("_test_", **fields)
