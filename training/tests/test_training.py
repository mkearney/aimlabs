import json

import polars as pl
import pytest
import torch
import train

from common.metrics import BestMetric, Fit, Metrics
from common.utils import get_hyperparameters_from_args


def test_fit_matches_manual_accuracy():
    fit = Fit(num_classes=3)
    output = torch.tensor([[5.0, 0, 0], [0, 5.0, 0], [0, 0, 5.0], [5.0, 0, 0]])
    target = torch.tensor([0, 1, 2, 1])
    fit.update(output[:2], target[:2])
    fit.update(output[2:], target[2:])
    stats = fit.compute()
    assert 0 < stats.acc < 1
    assert fit(output[:3], target[:3]).acc == pytest.approx(1.0)


def test_metrics_are_per_instance():
    a, b = Metrics(), Metrics()
    a.append(
        0,
        {k: 0.1 for k in ["loss", "val_loss", "val_acc", "val_f1", "val_pr", "val_rc"]},
    )
    assert len(a) == 1 and len(b) == 0
    assert a.collect()["epoch"] == [0]


def test_best_metric_snapshots_weights():
    model = torch.nn.Linear(2, 1)
    best = BestMetric("val_loss")
    assert best(0, {"val_loss": 1.0, "val_acc": 0.5, "loss": 1.0}, model)
    snapshot = best.state_dict["weight"].clone()
    with torch.no_grad():
        model.weight.add_(1.0)
    assert torch.equal(best.state_dict["weight"], snapshot)
    assert not best(1, {"val_loss": 2.0, "val_acc": 0.9, "loss": 1.0}, model)
    assert best.early_stop_counter == 1 and best.epoch == 0


def test_best_metric_max_mode():
    best = BestMetric("val_acc")
    assert best.mode == "max" and best.alt_metric == "val_loss"


def test_cli_maps_to_hyperparameters(tmp_path):
    args = train.parse_args(
        ["--input-dir", str(tmp_path), "--lr", "0.01", "--no-freeze", "--seed", "3"]
    )
    hp = get_hyperparameters_from_args(args)
    assert hp.lr == 0.01 and hp.freeze is False and hp.seed == 3
    assert hp.batch_size == 16  # untouched default


def test_end_to_end_training(tiny_model_dir, tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    rows = {"text": ["a b c", "d e f", "g h i", "j a b"] * 6, "label": ["x", "y"] * 12}
    for split in train.SPLITS:
        pl.DataFrame(rows).write_parquet(data / f"{split}.parquet")

    out = tmp_path / "models"
    args = train.parse_args(
        [
            "--input-dir", str(data),
            "--output-dir", str(out),
            "--model", str(tiny_model_dir),
            "--name", "tiny",
            "--max-len", "8",
            "--num-hidden", "8",
            "--num-epochs", "2",
            "--batch-size", "4",
            "--seed", "0",
            "--save",
        ]
    )  # fmt: skip
    train.main(args)

    (version_dir,) = (out / "tiny").iterdir()
    metrics = json.loads((version_dir / "metrics.json").read_text())
    assert metrics["epoch"] == [0, 1]
    assert set(metrics["test"]) >= {"val_loss", "val_acc"}
    assert json.loads((version_dir / "label2id.json").read_text()) == {"x": 0, "y": 1}
