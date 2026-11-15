# aimlabs

Fine-tune and serve transformer-based text classification models.

`aimlabs` wraps a pretrained Hugging Face sequence classifier (BERT,
DistilBERT, RoBERTa, …) with an optional trainable hidden layer, plus helpers
for preprocessing, saving, and inference.

## Installation

```bash
pip install aimlabs
```

## Usage

```python
from aimlabs import HyperParameters, Model, ModelSaver, Predictor

hp = HyperParameters(model="distilbert-base-uncased", num_classes=2, save=True)
model = Model(hp, label2id={"neg": 0, "pos": 1})

# ... train (see the training/ pipelines in the repository) ...

saver = ModelSaver("models/sentiment", version=model.version)
saver.save(model, metrics={})

predictor = Predictor(saver.path)
predictor.predict("what a great film").to_dict()
```

A saved model directory contains `state_dict.pt`, `hyperparameters.json`,
`label2id.json`, `config.json`, and `metrics.json`. Weights are loaded with
`torch.load(..., weights_only=True)`.

## Architecture

- `num_hidden > 0` (default): the pretrained model's classification head
  projects to its hidden width, followed by dropout → linear(`num_hidden`) →
  batch norm → linear(`num_classes`).
- `num_hidden = 0`: the pretrained classification head outputs `num_classes`
  logits directly, with `dropout` applied as the head's dropout.
- `freeze=True` freezes the pretrained base model; `hard_freeze=True` freezes
  every pretrained parameter, leaving only the outer layers trainable.

## Models

Any Hugging Face sequence-classification checkpoint should work. Some
examples with their parameter counts and sizes:

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
