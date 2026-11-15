import json

import pytest

from aimlabs.model import Model
from aimlabs.predictor import Predictor, batched
from aimlabs.save import ModelSaver


def test_batched():
    assert [len(b) for b in batched(["a"] * 45, 20)] == [20, 20, 5]
    assert list(Predictor.batch(["a"] * 100, 20)) == [["a"] * 20] * 5
    with pytest.raises(ValueError):
        list(batched(["a"], 0))


def test_save_and_predict_round_trip(hp_factory, tmp_path):
    model = Model(hp_factory(save=True), label2id={"neg": 0, "pos": 1}).eval()
    saver = ModelSaver(tmp_path, version=model.version)
    saver.save(model, metrics={"epoch": [0], "val_loss": [0.5]})

    saved = {p.name for p in saver.path.iterdir()}
    assert saved == {
        "state_dict.pt",
        "hyperparameters.json",
        "config.json",
        "metrics.json",
        "label2id.json",
    }
    assert json.loads((saver.path / "metrics.json").read_text())["val_loss"] == [0.5]

    predictor = Predictor(saver.path)
    preds = predictor.predict(["abc", "def", "ghi"], batch_size=2)
    assert len(preds) == 3
    assert all(p.label in {"neg", "pos"} for p in preds)
    assert all(abs(sum(p.proba) - 1) < 1e-5 for p in preds)
    assert preds.to_dict()["text"] == ["abc", "def", "ghi"]

    expected = model.forward_raw(["abc"]).softmax(1)[0].tolist()
    assert preds[0].proba == pytest.approx(expected, abs=1e-5)

    single = predictor.predict("abc")
    assert len(single) == 1
    assert len(predictor.predict([])) == 0


def test_saver_respects_save_flag(hp_factory, tmp_path):
    model = Model(hp_factory(save=False))
    saver = ModelSaver(tmp_path, version=model.version)
    saver.save(model, metrics={})
    assert not saver.path.exists()
