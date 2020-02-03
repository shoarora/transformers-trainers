import torch
from polytune.utils import mask_tokens
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import ConcatDataset, Dataset


class Collater():
    def __init__(self,
                 mlm=True,
                 mlm_prob=0.15,
                 pad_token_id=None,
                 mask_token_id=None,
                 vocab_size=None,
                 cls_token_id=None,
                 max_seq_len=128):
        self.mlm = mlm
        self.mlm_prob = mlm_prob
        self.pad_token_id = pad_token_id
        self.mask_token_id = mask_token_id
        self.vocab_size = vocab_size
        self.cls_token_id = None
        self.max_seq_len = max_seq_len

    def __call__(self, examples):
        # TODO do we need other attributes like attention_mask?
        # TODO how to make this backward compatible with the old tokenizers?
        inputs, special_tokens_masks = zip(*examples)
        # inputs = self._pad_sequence(inputs)
        # special_tokens_masks = self._pad_sequence(special_tokens_masks)

        if self.mlm:
            inputs, labels = mask_tokens(inputs, special_tokens_masks,
                                         self.pad_token_id, self.mask_token_id,
                                         self.vocab_size, self.mlm_prob)
            return inputs, labels
        else:
            return inputs, inputs

    def _pad_sequence(self, inputs):
        if self.pad_token_id is None:
            inputs = pad_sequence(inputs, batch_first=True)
        else:
            inputs = pad_sequence(inputs,
                                  batch_first=True,
                                  padding_value=self.pad_token_id)
        return inputs


def create_concat_dataset(tokenizer, paths):
    datasets = [LineByLineDataset(tokenizer, p) for p in paths]
    dataset = ConcatDataset(datasets)
    return dataset


class LineByLineDataset(Dataset):
    def __init__(self, tokenizer, path):
        self.tokenizer = tokenizer
        self.path = path
        with open(path) as f:
            self.len = len(f.readlines())
        self.lines = None

    def __len__(self):
        return self.len

    def __getitem__(self, i):
        if not self.lines:
            with open(self.path) as f:
                self.lines = f.readlines()
        item = self.lines[i]

        out = self.tokenizer.encode(item)
        ids = out.ids
        special_tokens_mask = out.special_tokens_mask
        if i == len(self.lines):
            self.lines = None

        return torch.tensor(ids), torch.tensor(special_tokens_mask,
                                               dtype=torch.bool)
