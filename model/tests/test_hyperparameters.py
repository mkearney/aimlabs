import pytest
from pydantic import ValidationError

from aimlabs.hyperparameters import HyperParameters


def test_defaults_resolve_from_training_settings():
    hp = HyperParameters(batch_size=8, num_steps=10)
    assert hp.valid_batch_size == 8
    assert hp.test_batch_size == 8
    assert hp.valid_max_steps == 10
    assert hp.test_max_steps == 10


def test_explicit_values_are_kept():
    hp = HyperParameters(valid_batch_size=4, test_max_steps=-1)
    assert hp.valid_batch_size == 4
    assert hp.test_max_steps == -1


def test_json_round_trip():
    hp = HyperParameters(name="x", lr=1e-3, seed=7)
    assert HyperParameters.model_validate_json(hp.model_dump_json()) == hp


@pytest.mark.parametrize(
    "kwargs",
    [{"batch_size": 0}, {"dropout": 1.0}, {"num_classes": 1}, {"best_metric": "auc"}],
)
def test_invalid_values_rejected(kwargs):
    with pytest.raises(ValidationError):
        HyperParameters(**kwargs)
