import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import ConcatDataset, Dataset


class Collater():
    def __init__(self, pad_token_id=None, cls_token_id=None):
        self.pad_token_id = pad_token_id
        self.cls_token_id = None

        # TODO do batch_encode?

    def __call__(self, examples):
        inputs, special_tokens_masks = zip(*examples)
        inputs = self._pad_sequence(inputs)
        special_tokens_masks = self._pad_sequence(special_tokens_masks)
        return inputs, special_tokens_masks

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
