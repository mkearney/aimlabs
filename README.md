# A.I. Message Labels

Packaging and pipelines for deep learning text classification models

## Installation

```bash
make develop
```

## Testing

```bash
make test
```

## Usage

```bash
training/.venv/bin/python training/scripts/train.py \
    --data-dir /Users/mwk/data/imdb \
    --batch-size 64 \
    --best-metric val_acc \
    --dropout 0.2 \
    --early-stopping-patience 3 \
    --gamma 0.5 \
    --lr 0.0001 \
    --max-len 100 \
    --model roberta-base \
    --name imdbsentiment \
    --num-classes 2 \
    --num-epochs 16 \
    --num-steps 8
```

## Training

![](tools/readme/training-screenshot.png)

## Testing

![](tools/readme/pytest-screenshot.png)
