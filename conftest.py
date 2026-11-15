"""Shared fixtures. A tiny randomly-initialized BERT is written to disk so the
test-suite never needs network access to the Hugging Face Hub."""

from pathlib import Path

import pytest
from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast

from aimlabs.hyperparameters import HyperParameters

VOCAB = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *"abcdefghij"]


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("tiny-bert")
    vocab_file = path / "vocab.txt"
    vocab_file.write_text("\n".join(VOCAB) + "\n")
    BertTokenizerFast(vocab_file=str(vocab_file)).save_pretrained(path)
    config = BertConfig(
        vocab_size=len(VOCAB),
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=64,
    )
    BertForSequenceClassification(config).save_pretrained(path)
    return path


@pytest.fixture
def hp_factory(tiny_model_dir: Path):
    def make(**kwargs) -> HyperParameters:
        defaults = {"model": str(tiny_model_dir), "max_len": 8, "num_hidden": 8}
        return HyperParameters(**{**defaults, **kwargs})

    return make
