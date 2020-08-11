import torch
from transformers_trainers.utils import mask_tokens
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import ConcatDataset, Dataset


class LineByLineCollater():
    def __init__(self,
                 tokenizer,
                 mlm=True,
                 mlm_prob=0.15,
                 pad_token_id=None,
                 mask_token_id=None,
                 vocab_size=None,
                 cls_token_id=None):
        self.tokenizer = tokenizer
        self.mlm = mlm
        self.mlm_prob = mlm_prob
        self.pad_token_id = pad_token_id
        self.mask_token_id = mask_token_id
        self.vocab_size = vocab_size
        self.cls_token_id = None

    def __call__(self, examples):
        # TODO how to make this backward compatible with the old tokenizers?
        encodings = self.tokenizer.encode_batch(examples)
        ids = []
        attention_masks = []
        special_tokens_masks = []
        for e in encodings:
            ids.append(e.ids)
            attention_masks.append(e.attention_mask)
            special_tokens_masks.append(e.special_tokens_mask)

        inputs = torch.tensor(ids, dtype=torch.long)
        attention_masks = torch.tensor(attention_masks,
                                       dtype=torch.long)
        special_tokens_masks = torch.tensor(special_tokens_masks,
                                            dtype=torch.bool)

        if self.mlm:
            inputs, labels = mask_tokens(inputs, special_tokens_masks,
                                         self.pad_token_id, self.mask_token_id,
                                         self.vocab_size, self.mlm_prob)
            return inputs, labels, attention_masks
        else:
            return inputs, inputs, attention_masks

    def _pad_sequence(self, inputs):
        if self.pad_token_id is None:
            inputs = pad_sequence(inputs, batch_first=True)
        else:
            inputs = pad_sequence(inputs,
                                  batch_first=True,
                                  padding_value=self.pad_token_id)
        return inputs


def create_line_by_line_dataset(tokenizer, paths):
    datasets = [LineByLineDataset(tokenizer, p) for p in paths]
    dataset = ConcatDataset(datasets)
    return dataset


class LineByLineDataset(Dataset):
    def __init__(self, tokenizer, path):
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
        if i == len(self.lines):
            self.lines = None

        return item
