import torch
from torch.utils.data import Dataset


class InputsDataset(Dataset):
    """
    Dataset of tokenized messages and their labels.

    ### Attributes
        - `input_ids`: tensor of token indices, shape `(n, max_len)`
        - `attention_mask`: tensor of attention masks, shape `(n, max_len)`
        - `labels`: tensor of target indices, shape `(n,)`
    """

    def __init__(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> None:
        if not len(input_ids) == len(attention_mask) == len(labels):
            raise ValueError(
                "input_ids, attention_mask, and labels must have the same length; "
                f"got {len(input_ids)}, {len(attention_mask)}, {len(labels)}"
            )
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }
