import fire
import os
import torch
from tokenizers import BertWordPieceTokenizer
from tqdm import tqdm
from shutil import rmtree


def load_and_cache_examples(data_dir,
                            output_dir,
                            tokenizer=None,
                            tokenizer_path=None,
                            n_sentences=0,
                            delete_existing=False,
                            max_length=512):

    if not tokenizer:
        tokenizer = BertWordPieceTokenizer(tokenizer_path)

    tokenizer.enable_truncation(max_length=max_length)
    tokenizer.enable_padding(max_length=max_length)

    num_tokens = 0
    num_examples = 0

    if delete_existing:
        rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    pbar = tqdm(os.listdir(data_dir))
    for path in pbar:
        result = process_one_file(data_dir, path, tokenizer, output_dir,
                                  n_sentences)
        num_examples += result['num_examples']
        num_tokens += result['num_tokens']

        pbar.set_description(
            f"{num_tokens} tokens, {num_examples} examples, {num_tokens/(num_examples*max_length)} non-pad tokens"
        )


def process_one_file(data_dir, path, tokenizer, output_dir, n_sentences):
    ids = []
    attention_masks = []
    special_tokens_masks = []
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

        if tokenized.overflowing:
            for example in tokenized.overflowing:
                add_example(example)
                num_examples += 1
                num_tokens += sum(example.attention_mask)

    torch.save(
        {
            'ids':
            torch.tensor(ids, dtype=torch.int8),
            'attention_masks':
            torch.tensor(attention_masks, dtype=torch.bool),
            'special_tokens_masks':
            torch.tensor(special_tokens_masks, dtype=torch.bool)
        }, output_file)

    return {'num_tokens': num_tokens, 'num_examples': num_examples}


if __name__ == '__main__':
    fire.Fire(load_and_cache_examples)
