import torch
from lmtuners.utils import mask_tokens
from torch.utils.data import ConcatDataset, Dataset


class PreTokenizedFileDataset(Dataset):
    def __init__(self, path):
        self.path = path

        data = torch.load(path)
        self.ids = data['ids']
        self.attention_masks = data['attention_masks']
        self.special_tokens_masks = data['special_tokens_masks']

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        return self.ids[i], self.attention_masks[i], self.special_tokens_masks[i]


def create_pretokenized_dataset(paths):
    datasets = [PreTokenizedFileDataset(p) for p in paths]
    dataset = ConcatDataset(datasets)
    return dataset


class PreTokenizedCollater(object):
    def __init__(self,
                 mlm=True,
                 mlm_prob=0.15,
                 pad_token_id=None,
                 mask_token_id=None,
                 vocab_size=None,
                 cls_token_id=None):
        self.mlm = mlm
        self.mlm_prob = mlm_prob
        self.pad_token_id = pad_token_id
        self.mask_token_id = mask_token_id
        self.vocab_size = vocab_size
        self.cls_token_id = None

    def __call__(self, examples):
        inputs, attention_masks, special_tokens_masks = zip(*examples)
        inputs = torch.stack(inputs).long()
        attention_masks = torch.stack(attention_masks).long()
        special_tokens_masks = torch.stack(special_tokens_masks)

        if self.mlm:
            inputs, labels = mask_tokens(inputs, special_tokens_masks,
                                         self.pad_token_id, self.mask_token_id,
                                         self.vocab_size, self.mlm_prob)
            return inputs, labels, attention_masks
        else:
            return inputs, inputs, attention_masks
