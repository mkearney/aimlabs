# A.I. Message Labels

Training pipelines for deep learning text classification models

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
.venv/bin/python ./scripts/train.py \
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


## Models

There are a variety of models that can be fine tuned.

| pretrained_model                          |   parameters |   size_mb |
|-------------------------------------------|-------------:|----------:|
| albert-base-v2                            |   11,692,290 |      44.6 |
| albert-large-v2                           |   17,716,052 |      67.6 |
| bert-base-cased                           |  109,490,954 |     417.7 |
| bert-base-uncased                         |  109,490,954 |     417.7 |
| distilroberta-base                        |   82,127,118 |     313.3 |
| distilbert-base-cased                     |   65,797,606 |     251.0 |
| distilbert-base-multilingual-cased        |  135,332,874 |     516.3 |
| distilbert-base-uncased                   |   66,961,674 |     255.4 |
| distilbert-base-uncased-distilled-squad   |   66,961,674 |     255.4 |
| distilgpt2                                |   82,118,922 |     313.3 |
| microsoft/deberta-base                    |  139,200,522 |     531.0 |
| roberta-base                              |  124,654,350 |     475.5 |
| roberta-base-openai-detector              |  124,654,350 |     475.5 |
| squeezebert/squeezebert-mnli-headless     |   51,102,474 |     194.9 |
