# A.I. Message Labels

Training pipelines for deep learning text classification models

## Installation

The following make command will install the required dependencies along with
the `aimlabs` package, which is found in the base `model/` directory of this
repository.

```bash
make develop
```

## Testing

```bash1
make test
```

## Usage

Example of fine-tuning or transfer learning via `distilbert-base-uncased`
on the task of sentiment classification using the IMDB reviews dataset.

```bash
.venv/bin/python ./scripts/train.py \
    --data-dir /Users/mwk/data/imdb \
    --batch-size 16 \
    --best-metric val_loss \
    --dropout 0.15 \
    --early_stopping_patience 6 \
    --eps 1e-10 \
    --no-freeze \
    --init-std 0.015 \
    --lr 5e-5 \
    --max-len 512 \
    --model distilbert-base-uncased \
    --name imdbsentiment \
    --num-classes 2 \
    --num-epochs 32 \
    --num-hidden 768 \
    --num-steps 128 \
    --save \
    --test-max-steps 9999
```

A screenshot of the logging output from the example command above. This
shows a successful replication of the IMDB classification benchmark
(~93% accuracy) achieved by the distilbert-base-uncased model. (see
paperswithcode for a full list of peer-reviewed benchmarks)

![](../tools/readme/training-screenshot.png)