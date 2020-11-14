from typing import Union

import torch
from tokenizers import Encoding, Tokenizer
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, PreTrainedTokenizerBase

import datasets


class HuggingfaceDataset(Dataset):
    def __init__(
        self,
        dataset: datasets.Dataset,
        tokenizer: Union[Tokenizer, PreTrainedTokenizer],
        column: str,
        block_size: int,
        pretokenize: bool = False,
    ):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.column = column
        self.block_size = block_size
        self.pretokenize = pretokenize

        if isinstance(tokenizer, (PreTrainedTokenizer, PreTrainedTokenizerBase)):
            self.tokenizer_type = "transformers"
        elif isinstance(tokenizer, Tokenizer):
            self.tokenizer_type = "tokenizers"
            tokenizer.enable_padding(length=block_size)
            tokenizer.enable_truncation(max_length=block_size)

        if pretokenize:
            if self.tokenizer_type == "transformers":
                self.dataset = self.dataset.map(
                    lambda x: tokenizer.batch_encode_plus(
                        x[column], truncation=True, max_length=block_size
                    ),
                    batched=True,
                )
            elif self.tokenizer_type == "tokenizers":
                self.dataset = self.dataset.map(lambda x: encoding_to_dict(tokenizer.encode(x)))
            self.dataset.set_format(
                type="torch",
                columns=["input_ids", "attention_mask", "token_type_ids"],
            )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i) -> torch.Tensor:
        if self.pretokenize:
            return self.dataset[i]
        else:
            if self.tokenizer_type == "transformers":
                return self.tokenizer.encode_plus(
                    self.dataset[i][self.column],
                    add_special_tokens=True,
                    padding=True,
                    truncation=True,
                    max_length=self.block_size,
                    return_tensors="pt",
                )
            if self.tokenizer_type == "tokenizers":
                return encoding_to_dict(
                    self.tokenizer.encode(self.dataset[i][self.column])
                )


def encoding_to_dict(encoding: Encoding):
    return {
        "input_ids": encoding.ids,
        "attention_mask": encoding.attention_mask,
        "token_type_ids": encoding.type_ids,
    }


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
