import os
import random
from shutil import rmtree

import fire
import torch
from tokenizers import BertWordPieceTokenizer
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence


def tokenize_and_cache_data(data_dir,
                            output_dir,
                            tokenizer=None,
                            tokenizer_path=None,
                            n_sentences=0,
                            use_overflow=False,
                            two_segments=True,
                            delete_existing=False,
                            max_length=512):

    if not tokenizer:
        tokenizer = BertWordPieceTokenizer(tokenizer_path)

    tokenizer.enable_truncation(max_length=max_length)

    num_tokens = 0
    num_examples = 0

    if delete_existing:
        rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    pbar = tqdm(os.listdir(data_dir))
    for path in pbar:
        result = process_one_file(data_dir, path, tokenizer, output_dir,
                                  n_sentences, use_overflow, two_segments)
        num_examples += result['num_examples']
        num_tokens += result['num_tokens']

        pbar.set_description(
            f"{num_tokens} tokens, {num_examples} examples"
        )


def process_one_file(data_dir, path, tokenizer, output_dir, n_sentences, use_overflow, two_segments):
    ids = []
    attention_masks = []
    special_tokens_masks = []
    token_type_ids = []
    num_tokens = 0
    num_examples = 0

    output_file = os.path.join(output_dir, f"{path}.pt")

    if os.path.exists(output_file):
        tokens = torch.load(output_file)
        num_examples = len(tokens['ids'])
        num_tokens = int(torch.sum(tokens['attention_masks']).numpy())
        return {'num_examples': num_examples, 'num_tokens': num_tokens}

    def add_example(encoded):
        ids.append(encoded.ids)
        attention_masks.append(encoded.attention_mask)
        special_tokens_masks.append(encoded.special_tokens_mask)
        token_type_ids.append([0] * len(encoded.ids))

    with open(os.path.join(data_dir, path)) as f:
        try:
            texts = [line.strip() for line in f.readlines() if line.strip()]
        except Exception as e:
            print(e)
            print('skipping', path)
            return {'num_examples': 0, 'num_tokens': 0}

        if n_sentences == 0:
            pass
        else:
            text = ' '.join(texts)
            texts = [f"{sent}." for sent in text.split('.')]
            if n_sentences > 1:
                sentences = list(texts)
                texts = []
                for i in range(0, len(sentences), n_sentences):
                    texts.append(' '.join(sentences[i:i + n_sentences]))

    encoded_batch = tokenizer.encode_batch(texts)
    for tokenized in encoded_batch:
        add_example(tokenized)
        num_examples += 1
        num_tokens += sum(tokenized.attention_mask)

        if tokenized.overflowing and use_overflow:
            for example in tokenized.overflowing:
                add_example(example)
                num_examples += 1
                num_tokens += sum(example.attention_mask)

    if two_segments:
        new_ids = []
        new_attention_masks = []
        new_special_tokens_masks = []
        new_token_type_ids = []
        indices = list(range(len(ids)))
        random.shuffle(indices)
        indices = [(indices[i], indices[i+1]) for i in range(0, len(indices)-1, 2)]
        for i, j in indices:
            _ids = ids[i] + ids[j][1:]
            _attention_mask = attention_masks[i] + attention_masks[j][1:]
            _special_tokens_mask = special_tokens_masks[i] + special_tokens_masks[j][1:]
            _token_type_ids = ([0] * len(ids[i])) + ([1] * len(ids[j][1:]))
            new_ids.append(_ids)
            new_attention_masks.append(_attention_mask)
            new_special_tokens_masks.append(_special_tokens_mask)
            new_token_type_ids.append(_token_type_ids)
        ids = new_ids
        attention_masks = new_attention_masks
        special_tokens_masks = new_special_tokens_masks
        token_type_ids = new_token_type_ids

    ids = [torch.tensor(i, dtype=torch.int32) for i in ids]
    attention_masks = [torch.tensor(i, dtype=torch.bool) for i in attention_masks]
    special_tokens_masks = [torch.tensor(i, dtype=torch.bool) for i in special_tokens_masks]
    token_type_ids = [torch.tensor(i, dtype=torch.int8) for i in token_type_ids]

    torch.save(
        {
            'ids': pad_sequence(ids, batch_first=True, padding_value=tokenizer.token_to_id('[PAD]')),
            'attention_masks': pad_sequence(attention_masks, batch_first=True, padding_value=0),
            'special_tokens_masks': pad_sequence(special_tokens_masks, batch_first=True, padding_value=1),
            'token_type_ids': pad_sequence(token_type_ids, batch_first=True, padding_value=1)
        }, output_file)

    return {'num_tokens': num_tokens, 'num_examples': num_examples}


if __name__ == '__main__':
    fire.Fire(tokenize_and_cache_data)
