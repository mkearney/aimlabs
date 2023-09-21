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
    --dropout 0.1 \
    --fraction 0.5 \
    --freeze \
    --lr 0.0002 \
    --max-len 32 \
    --model distilbert-base-uncased \
    --name imdbsentiment \
    --num-classes 2 \
    --num-epochs 16 \
    --num-steps 16
```

## Training

![](tools/readme/training-screenshot.png)

## Testing

![](tools/readme/pytest-screenshot.png)
