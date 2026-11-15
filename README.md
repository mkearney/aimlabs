# A.I. Message Labels

[![CI](https://github.com/mkearney/aimlabs/actions/workflows/ci.yml/badge.svg)](https://github.com/mkearney/aimlabs/actions/workflows/ci.yml)

Packaging and pipelines for fine-tuning transformer models (BERT, DistilBERT,
RoBERTa, …) on text classification tasks.

## Repository contents

| Path         | What it is                                                            |
|--------------|-----------------------------------------------------------------------|
| `model/`     | The `aimlabs` Python package: model spec, preprocessing, saving, and serving |
| `training/`  | Training pipelines built on `aimlabs`                                 |
| `tools/`     | Miscellaneous documentation and other resources                       |

The repository is a [uv](https://docs.astral.sh/uv/) workspace, so one command
sets up the library, the training dependencies, and the dev tooling.

## Quickstart

```bash
# install uv: https://docs.astral.sh/uv/getting-started/installation/
make develop        # or: uv sync --all-packages

# train (see training/README.md for the data format and all options)
uv run python training/scripts/train.py --input-dir ~/data/imdb --save
```

```python
from aimlabs import Predictor

predictor = Predictor("models/nlpmodel/0.1.0.20240101000000")
for p in predictor.predict(["what a great film", "utterly boring"]):
    print(p.label, p.proba)
```

## Development

```bash
make check          # ruff lint + format check, mypy, pytest
make format         # auto-fix lint issues and format
```

The test-suite builds a tiny randomly-initialized BERT on the fly, so it runs
in about a second and needs no network access.

Releases: bump `version` in `model/pyproject.toml`, then publish a GitHub
release. The `Publish` workflow builds the package and uploads it to PyPI via
trusted publishing.
