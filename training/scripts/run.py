from subprocess import Popen

from aimlabs.hyperparameters import HyperParameters


class Conductor:
    cmd: str = "training/.venv/bin/python training/scripts/train.py"
    defaults: HyperParameters = HyperParameters()

    def as_arg(self, k, v):
        k = k.replace("_", "-")
        return f"--{k} {v}"

    def run(self, command):
        p = Popen(command, shell=True)
        p.communicate()

    def args(self, **kwargs):
        return [
            self.as_arg(k, v) for k, v in HyperParameters(**kwargs).__dict__.items()
        ]

    def __call__(self, data_dir: str, fraction: float = 1.0, *args, **kwargs):
        args = self.args(**kwargs)
        args += ["--data-dir", data_dir, "--fraction", str(fraction)]
        call = self.cmd + " \\\n    " + " \\\n    ".join(args)
        self.run(call)


conductor = Conductor()

conductor(
    "/Users/mwk/data/imdb",
    batch_size=64,
    best_metric="val_acc",
    dropout=0.1,
    lr=5e-5,
    early_stopping_patience=16,
    gamma=0.5,
    max_len=80,
    model="distilbert-base-uncased",
    name="imdbsentiment",
    num_classes=2,
    num_epochs=64,
    num_steps=8,
)
