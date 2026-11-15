import pytest
import torch

from aimlabs.data import InputsDataset


def test_dataset_items():
    ds = InputsDataset(torch.ones(3, 4), torch.zeros(3, 4), torch.tensor([0, 1, 0]))
    assert len(ds) == 3
    item = ds[1]
    assert set(item) == {"input_ids", "attention_mask", "labels"}
    assert item["labels"].item() == 1


def test_dataset_length_mismatch():
    with pytest.raises(ValueError, match="same length"):
        InputsDataset(torch.ones(3, 4), torch.zeros(2, 4), torch.tensor([0, 1, 0]))
