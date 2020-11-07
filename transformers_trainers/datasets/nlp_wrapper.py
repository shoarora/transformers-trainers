import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

import datasets


class NlpWrapperDataset(Dataset):
    """ Based on huggingface/transformers v3.0.0 implementation.
    This will be superseded by a framework-agnostic approach
    soon.
    """

    def __init__(
        self,
        dataset: datasets.Dataset,
        tokenizer: PreTrainedTokenizer,
        column: str,
        block_size: int,
        pretokenize: bool = False,
    ):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.column = column
        self.block_size = block_size
        self.pretokenize = pretokenize

        if pretokenize:
            self.dataset = self.dataset.map(
                lambda example: tokenizer.batch_encode_plus(
                    example[column], truncation=True, max_length=self.block_size
                ),
                batched=True,
            )
            self.dataset.set_format(type="torch", columns=["input_ids"])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i) -> torch.Tensor:
        if self.pretokenize:
            return self.dataset[i]["input_ids"]
        else:
            return self.tokenizer.encode(
                self.dataset[i][self.column],
                add_special_tokens=True,
                truncation=True,
                max_length=self.block_size,
                return_tensors="pt",
            ).squeeze()
