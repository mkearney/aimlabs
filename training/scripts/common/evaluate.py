import torch
from common.metrics import Fit


def evaluate(
    model,
    dataloader,
    criterion,
    train_loss: float,
    max_steps: int,
):
    """
    Epoch evaluation for model training

    ### Args
        - `model`: model
        - `dataloader`: dataloader
        - `criterion`: loss function
        - `train_loss`: training loss

    ### Returns
        - `metrics`: dict of metrics
    """
    losses, accs, f1s, prs, rcs = [], [], [], [], []
    imax = max_steps if max_steps > 0 else len(dataloader)
    fit = Fit(num_classes=model.hyperparameters.num_classes)
    model.eval()  # type: ignore
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            outputs = model(**data)  # type: ignore
            loss = criterion(outputs, data["labels"].long())
            losses.append(loss)
            fit_metrics = fit(outputs, data["labels"])
            accs.append(fit_metrics.acc)
            f1s.append(fit_metrics.f1)
            prs.append(fit_metrics.pr)
            rcs.append(fit_metrics.rc)
            if i == imax:
                break
        # calculate means
        denom = len(losses)
        val_loss = torch.stack(losses).mean()
        val_acc = sum(accs) / denom
        val_f1 = sum(f1s) / denom
        val_pr = sum(prs) / denom
        val_rc = sum(rcs) / denom
        metrics = {
            "loss": train_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_f1": val_f1,
            "val_pr": val_pr,
            "val_rc": val_rc,
        }
        return metrics
