import pytest
import torch
from transformers import BertConfig

from aimlabs.model import Model, get_dimensions, get_dropout_name


def test_get_dimensions_fallback():
    assert get_dimensions(BertConfig(hidden_size=24, num_attention_heads=2)) == 24


def test_get_dimensions_override():
    config = BertConfig(intermediate_size=99)
    config.name_or_path = "bert-base-uncased"
    assert get_dimensions(config) == 99


def test_get_dropout_name():
    assert get_dropout_name(BertConfig()) == "classifier_dropout"


@pytest.mark.parametrize("num_hidden", [0, 8])
def test_forward_shapes(hp_factory, num_hidden):
    model = Model(hp_factory(num_hidden=num_hidden, num_classes=3)).eval()
    logits = model.forward_raw(["abc", "de f", "j"])
    assert logits.shape == (3, 3)


def test_preprocess_pads_and_truncates(hp_factory):
    model = Model(hp_factory(max_len=6))
    inputs = model.preprocess(["a", "a b c d e f g h"])
    assert inputs["input_ids"].shape == (2, 6)
    assert inputs["attention_mask"].sum(1).tolist() == [3, 6]


def test_freeze(hp_factory):
    model = Model(hp_factory(freeze=True))
    assert not any(p.requires_grad for p in model.model.base_model.parameters())
    assert all(p.requires_grad for p in model.outer_layer.parameters())

    model = Model(hp_factory(freeze=False))
    assert all(p.requires_grad for p in model.parameters())

    model = Model(hp_factory(hard_freeze=True))
    assert not any(p.requires_grad for p in model.model.parameters())


def test_train_step_backprop(hp_factory):
    model = Model(hp_factory()).train()
    batch = model.preprocess(["ab", "cd", "ef"])
    batch["labels"] = torch.tensor([0, 1, 0])
    loss = model.train_step(batch)
    loss.backward()
    assert loss.item() > 0
    assert model.outer_layer[-1].weight.grad is not None


def test_label2id_must_match_num_classes(hp_factory):
    with pytest.raises(ValueError, match="num_classes"):
        Model(hp_factory(num_classes=2), label2id={"a": 0, "b": 1, "c": 2})
