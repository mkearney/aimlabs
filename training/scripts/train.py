"""
Fine-tune an `aimlabs.Model` on a text classification dataset.

The input directory must contain `train`, `valid`, and `test` splits as
`.parquet`, `.csv`, or `.jsonl` files, each with `text` and `label` columns.

Run `python scripts/train.py --help` for all options.
"""

from __future__ import annotations

from argparse import (
    ArgumentParser,
    BooleanOptionalAction,
    Namespace,
    RawDescriptionHelpFormatter,
)
from datetime import datetime
from pathlib import Path

import polars as pl
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from aimlabs.data import InputsDataset
from aimlabs.hyperparameters import HyperParameters
from aimlabs.model import Model
from aimlabs.save import ModelSaver
from aimlabs.utils import get_device, get_logger, seed_everything
from common.evaluate import evaluate, move_to
from common.metrics import BestMetric, Metrics
from common.utils import (
    format_duration,
    get_hyperparameters_from_args,
    log_metrics,
    log_test_metrics,
    model_size,
)

SPLITS = ("train", "valid", "test")
READERS = {".parquet": pl.read_parquet, ".csv": pl.read_csv, ".jsonl": pl.read_ndjson}


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description=__doc__, formatter_class=RawDescriptionHelpFormatter
    )
    io = parser.add_argument_group("data and output")
    io.add_argument(
        "--input-dir",
        "--data-dir",
        type=Path,
        required=True,
        help="directory containing train/valid/test splits",
    )
    io.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models"),
        help="models are saved to <output-dir>/<name>/<version> (default: models)",
    )
    io.add_argument(
        "--fraction",
        type=float,
        default=1.0,
        help="train on a random fraction of the data (useful for development)",
    )
    io.add_argument("--device", type=str, help="torch device (default: auto-detect)")

    # Hyperparameters: unset flags fall back to `HyperParameters` defaults.
    hp = parser.add_argument_group("hyperparameters")
    for name, field in HyperParameters.model_fields.items():
        flag = f"--{name.replace('_', '-')}"
        help_ = f"(default: {field.default})"
        if field.annotation is bool:
            hp.add_argument(flag, action=BooleanOptionalAction, help=help_)
        else:
            kind = {int: int, float: float, int | None: int}.get(field.annotation, str)
            hp.add_argument(flag, type=kind, help=help_)
    return parser


def parse_args(argv: list[str] | None = None) -> Namespace:
    return get_parser().parse_args(argv)


def read_split(input_dir: Path, split: str) -> pl.DataFrame:
    for suffix, reader in READERS.items():
        path = input_dir / f"{split}{suffix}"
        if path.is_file():
            df = reader(path)
            missing = {"text", "label"} - set(df.columns)
            if missing:
                raise ValueError(f"{path} is missing column(s): {sorted(missing)}")
            return df.select(
                pl.col("text").cast(pl.Utf8), pl.col("label").cast(pl.Utf8)
            )
    raise FileNotFoundError(
        f"no {split} split found in {input_dir} (looked for {split}"
        f"{{{','.join(READERS)}}})"
    )


def make_dataset(
    model: Model, df: pl.DataFrame, label2id: dict[str, int]
) -> InputsDataset:
    unknown = set(df["label"].unique().to_list()) - set(label2id)
    if unknown:
        raise ValueError(f"labels not present in the training split: {unknown}")
    inputs = model.preprocess(df["text"].to_list())
    labels = torch.tensor([label2id[x] for x in df["label"].to_list()])
    return InputsDataset(inputs["input_ids"], inputs["attention_mask"], labels)


def main(args: Namespace) -> None:
    hp = get_hyperparameters_from_args(args)
    logger = get_logger(hp.name, version=hp.version)
    start_time = datetime.now()
    logger.info("_init_", time=start_time.strftime("%Y-%m-%d %H:%M:%S"))
    if hp.seed is not None:
        seed_everything(hp.seed)
    device = get_device(args.device)
    logger.info("device", device=str(device))

    # data
    train_df, valid_df, test_df = (read_split(args.input_dir, s) for s in SPLITS)
    labels = sorted(train_df["label"].unique().to_list())
    label2id = {label: idx for idx, label in enumerate(labels)}
    if len(label2id) != hp.num_classes:
        logger.warning(
            "num_classes does not match the training labels; using the latter",
            num_classes=hp.num_classes,
            labels=len(label2id),
        )
        hp = hp.model_copy(update={"num_classes": len(label2id)})

    if args.fraction < 1.0:
        train_df = train_df.sample(fraction=args.fraction, shuffle=True, seed=hp.seed)
        eval_fraction = (args.fraction + 1) / 2
        valid_df = valid_df.sample(fraction=eval_fraction, shuffle=True, seed=hp.seed)
        test_df = test_df.sample(fraction=eval_fraction, shuffle=True, seed=hp.seed)

    logger.info("_nobs_", train=len(train_df), valid=len(valid_df), test=len(test_df))
    for k, v in hp.model_dump().items():
        logger.info("__hp__", **{k: v})

    model = Model(hyperparameters=hp, label2id=label2id)
    logger.info("_size_", **model_size(model))
    model.to(device)

    pin = device.type == "cuda"
    train_dataloader = DataLoader(
        make_dataset(model, train_df, label2id),
        batch_size=hp.batch_size,
        shuffle=True,
        pin_memory=pin,
        # BatchNorm in the outer layer can't handle a trailing batch of one
        drop_last=hp.num_hidden > 0,
    )
    valid_dataloader = DataLoader(
        make_dataset(model, valid_df, label2id),
        batch_size=hp.valid_batch_size,
        shuffle=True,
        pin_memory=pin,
    )
    test_dataloader = DataLoader(
        make_dataset(model, test_df, label2id),
        batch_size=hp.test_batch_size,
        shuffle=False,
        pin_memory=pin,
    )

    # training objects
    saver = ModelSaver(args.output_dir / hp.name, version=model.version, logger=logger)
    metrics = Metrics()
    best_metric = BestMetric(metric=hp.best_metric)
    optimizer = AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=hp.lr, eps=hp.eps
    )
    lr_scheduler = ReduceLROnPlateau(
        optimizer, mode=best_metric.mode, factor=hp.gamma, patience=hp.lr_patience
    )
    criterion = nn.CrossEntropyLoss()

    # train loop
    try:
        for epoch in range(hp.num_epochs):
            epoch_lr = optimizer.param_groups[0]["lr"]
            model.train()
            total_loss, steps = 0.0, 0
            for batch in train_dataloader:
                if 0 < hp.num_steps <= steps:
                    break
                loss = model.train_step(move_to(batch, device))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                total_loss += loss.item()
                steps += 1

            epoch_metrics = evaluate(
                model=model,
                dataloader=valid_dataloader,
                criterion=criterion,
                train_loss=total_loss / max(steps, 1),
                max_steps=hp.valid_max_steps,
            )
            metrics.append(epoch=epoch, metrics=epoch_metrics)
            log_metrics(epoch, epoch_lr, epoch_metrics, logger)

            best_metric(epoch=epoch, metrics=epoch_metrics, model=model)
            lr_scheduler.step(epoch_metrics[best_metric.metric])
            if best_metric.early_stop_counter >= hp.early_stopping_patience:
                logger.info("_stop_", early_stopping=best_metric.early_stop_counter)
                break
    except KeyboardInterrupt:
        logger.info("__keyboard_interrupt__")

    end_time = datetime.now()
    logger.info("_end__", time=end_time.strftime("%Y-%m-%d %H:%M:%S"))
    logger.info("_dur__", **format_duration(end_time - start_time))

    if not best_metric.state_dict:
        logger.warning("no completed epochs; nothing to evaluate or save")
        return

    logger.info(
        "_best_",
        _epoch=best_metric.epoch,
        _metric=best_metric.metric,
        _value=f"{best_metric.value:.4f}",
        alt=best_metric.alt_metric,
        value=f"{best_metric.alt_value:.4f}",
    )

    # restore the best weights, evaluate on the test set, then save
    model.load_state_dict(best_metric.state_dict)
    model.eval()
    try:
        test_metrics = evaluate(
            model=model,
            dataloader=test_dataloader,
            criterion=criterion,
            train_loss=best_metric.trn_loss,
            max_steps=hp.test_max_steps,
        )
        log_test_metrics(test_metrics, logger)
    except KeyboardInterrupt:
        logger.info("__keyboard_interrupt__")
        test_metrics = None

    model.to("cpu")
    saver.save(
        model,
        {
            **metrics.collect(),
            "best_epoch": best_metric.epoch,
            "test": test_metrics,
        },
    )


if __name__ == "__main__":
    main(parse_args())
