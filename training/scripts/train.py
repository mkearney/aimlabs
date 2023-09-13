from argparse import ArgumentParser, Namespace
from datetime import datetime
from pathlib import Path

import polars as pl
import torch
import torch.nn as nn
import torch.optim as optim
from aimlabs.save import ModelSaver
from aimlabs.utils import get_logger
from common.evaluate import evaluate
from common.metrics import BestMetric, Fit, Metrics
from common.utils import (
    get_hyperparameters_from_args,
    log_metrics,
    model_size,
    save_hypers,
)
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from aimlabs.data import InputsDataset
from aimlabs.model import Model


def get_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--best-metric", type=str)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--early-stopping-patience", type=int)
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--gamma", type=float)
    parser.add_argument("--lr-patience", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--data-dir", type=str)
    parser.add_argument("--max-len", type=int)
    parser.add_argument("--model", type=str)
    parser.add_argument("--name", type=str)
    parser.add_argument("--num-classes", type=int)
    parser.add_argument("--num-dims", type=int)
    parser.add_argument("--num-epochs", type=int)
    parser.add_argument("--num-layers", type=int)
    parser.add_argument("--num-steps", type=int)
    parser.add_argument("--version", type=str)
    return parser


def parse_args() -> Namespace:
    parser = get_parser()
    return parser.parse_args()


def main(args: Namespace):
    logger = get_logger(args.version, args.name)
    start_time = datetime.now()
    logger.info("_init_", time=start_time.strftime("%Y-%m-%d %H:%M:%S"))

    # data
    data_dir = Path(args.data_dir)
    train_df = pl.read_parquet(data_dir.joinpath("train.parquet"))
    valid_df = pl.read_parquet(data_dir.joinpath("valid.parquet"))
    test_df = pl.read_parquet(data_dir.joinpath("test.parquet"))
    label_map = {
        label: idx for idx, label in enumerate(train_df["label"].unique().sort())
    }
    train_df = train_df.with_columns(target=pl.col("label").map_dict(label_map))

    if args.fraction < 1.0:
        train_df = train_df.sample(fraction=args.fraction, shuffle=True)
        valid_df = valid_df.sample(fraction=(args.fraction + 1) / 2, shuffle=True)
        test_df = test_df.sample(fraction=(args.fraction + 1) / 2, shuffle=True)

    # data sizes, hypers, and training components
    logger.info("_nobs_", train=train_df.shape[0])
    logger.info("_nobs_", valid=valid_df.shape[0])
    logger.info("_nobs_", test=test_df.shape[0])

    hp = get_hyperparameters_from_args(args)
    for k, v in hp.__dict__.items():
        logger.info("__hp__", **{k: v})

    model = Model(hyperparameters=hp)
    logger.info("_mdsz_", **model_size(model))
    optimizer = optim.AdamW(
        model.parameters(),  # type: ignore
        lr=hp.lr,
        eps=1e-8,
    )  # type: ignore
    lr_scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min" if "loss" in hp.best_metric else "max",
        factor=hp.gamma,
        patience=hp.lr_patience,
        verbose=False,
    )
    criterion = nn.CrossEntropyLoss()

    # preprocess data
    train_inputs = model.preprocess(train_df["text"].to_list())
    valid_inputs = model.preprocess(valid_df["text"].to_list())
    test_inputs = model.preprocess(test_df["text"].to_list())
    train_targets = torch.tensor(
        train_df["label"].map_dict(label_map).to_list(), dtype=torch.int64
    )
    valid_targets = torch.tensor(
        valid_df["label"].map_dict(label_map).to_list(), dtype=torch.int64
    )
    test_targets = torch.tensor(
        test_df["label"].map_dict(label_map).to_list(), dtype=torch.int64
    )
    # create datasets
    train_data = InputsDataset(
        input_ids=train_inputs["input_ids"],  # type: ignore
        attention_mask=train_inputs["attention_mask"],  # type: ignore
        targets=train_targets,
    )
    valid_data = InputsDataset(
        input_ids=valid_inputs["input_ids"],  # type: ignore
        attention_mask=valid_inputs["attention_mask"],  # type: ignore
        targets=valid_targets,
    )
    test_data = InputsDataset(
        input_ids=test_inputs["input_ids"],  # type: ignore
        attention_mask=test_inputs["attention_mask"],  # type: ignore
        targets=test_targets,
    )
    train_dataloader = DataLoader(train_data, batch_size=hp.batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_data, batch_size=hp.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=hp.batch_size, shuffle=True)
    fit = Fit(num_classes=hp.num_classes)

    # train objects
    output_dir = Path("/Users/mwk/models/").joinpath(hp.name)
    output_dir.mkdir(parents=True, exist_ok=True)
    saver = ModelSaver(
        str(output_dir), version=model.version, logger=logger
    )  # type: ignore
    metrics = Metrics()
    best_metric = BestMetric(metric=hp.best_metric)

    # train loop
    try:
        for epoch in range(hp.num_epochs):
            epoch_lr = optimizer.param_groups[0]["lr"]

            # training steps
            trn_epoch_loss = []
            model.train()  # type: ignore
            for i, data in enumerate(train_dataloader):
                outputs = model(**data)
                loss = criterion(outputs, data["targets"].long())
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                trn_epoch_loss.append(loss.item())
                if i == hp.num_steps:
                    break
            trn_epoch_loss_stat = sum(trn_epoch_loss) / len(trn_epoch_loss)

            # validation steps
            epoch_metrics = evaluate(
                model=model,
                dataloader=valid_dataloader,
                criterion=criterion,
                train_loss=trn_epoch_loss_stat,
            )
            metrics.append(epoch=epoch, metrics=epoch_metrics)
            log_metrics(epoch, epoch_lr, epoch_metrics, logger)

            # track best metric
            best_metric(epoch=epoch, metrics=epoch_metrics, model=model)
            lr_scheduler.step(epoch_metrics[best_metric.metric])
            if best_metric.early_stop_counter >= hp.early_stopping_patience:
                logger.info(
                    "__early_stopping__", es_counter=best_metric.early_stop_counter
                )
                break
    except KeyboardInterrupt:
        logger.info("__keyboard_interrupt__")

    # training duration
    end_time = datetime.now()
    logger.info("__end__", time=end_time.strftime("%Y-%m-%d %H:%M:%S"))
    duration = end_time - start_time
    if duration.total_seconds() <= 120:
        logger.info("duration", seconds=f"{duration.total_seconds():,.2f}")
    else:
        logger.info("duration", minutes=f"{duration.total_seconds()/60:,.2f}")

    # best epoch logging
    logger.info(
        "best_metric",
        _epoch=best_metric.epoch,
        _metric=best_metric.metric,
        _value=f"{best_metric.value:.4f}",
        alt=best_metric.alt_metric,
        value=f"{best_metric.alt_value:.4f}",
    )

    # save model
    model.load_state_dict(best_metric.state_dict)  # type: ignore
    model.eval()  # type: ignore
    saver.save(model, metrics.__dict__)

    # test set
    try:
        with torch.no_grad():
            test_loss, acc, f1s, prs, rcs = [], [], [], [], []
            for i, data in enumerate(test_dataloader):
                outputs = model(**data)  # type: ignore
                loss = criterion(outputs, data["targets"].long())
                fit_metrics = fit(outputs, data["targets"])
                acc.append(fit_metrics.acc)
                f1s.append(fit_metrics.f1)
                prs.append(fit_metrics.pr)
                rcs.append(fit_metrics.rc)
                test_loss.append(loss.item())
                if i == hp.num_steps:
                    break
            denom = len(acc)
            tacc = sum(acc) / denom
            tf1 = sum(f1s) / denom
            tlss = sum(test_loss) / denom
            tpr = sum(prs) / denom
            trc = sum(rcs) / denom
            logger.info(
                "test",
                loss=f"{tlss:.4f}",
                acc=f"{tacc:.4f}",
                f1=f"{tf1:.4f}",
                pr=f"{tpr:.4f}",
                rc=f"{trc:.4f}",
            )
    except KeyboardInterrupt:
        logger.info("__keyboard_interrupt__")

    # save metadata
    saved_as = metrics.save("/Users/mwk/models/meta", model.version)
    hyperparameters_path = save_hypers(
        params=model.hyperparameters.__dict__,
        output_dir="/Users/mwk/models/meta",
        version=model.version,  # type: ignore
    )
    logger.info("__metadata__", hyperparameters=hyperparameters_path)
    logger.info("__metadata__", metrics=saved_as)


if __name__ == "__main__":
    args = parse_args()
    main(args)
