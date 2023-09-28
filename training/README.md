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
    --batch-size 32 \
    --best-metric val_loss \
    --dropout 0.3 \
    --early_stopping_patience 6 \
    --eps 1e-9 \
    --no-freeze \
    --init-std 0.02 \
    --lr 5e-5 \
    --max-len 128 \
    --model distilbert-base-uncased \
    --name imdbsentiment \
    --num-classes 2 \
    --num-epochs 32 \
    --num-hidden 200 \
    --num-steps 32
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
