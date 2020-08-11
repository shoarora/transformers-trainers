from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
import os
import torch


class LineByLineTextDataset(Dataset):
    """ Based on huggingface/transformers v3.0.0 implementation.
    This will be superseded by a framework-agnostic approach
    soon.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int):
        assert os.path.isfile(file_path)
        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)

        with open(file_path, encoding="utf-8") as f:
            self.lines = [
                line
                for line in f.read().splitlines()
                if (len(line) > 0 and not line.isspace())
            ]
        self.tokenizer = tokenizer
        self.block_size = block_size

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, i) -> torch.Tensor:
        return self.tokenizer.encode(
            self.lines[i],
            add_special_tokens=True,
            truncation=True,
            max_length=self.block_size,
            return_tensors="pt",
        ).squeeze()
