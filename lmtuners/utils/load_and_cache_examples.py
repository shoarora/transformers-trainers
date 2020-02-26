import fire
import os
import torch
from tokenizers import BertWordPieceTokenizer
from tqdm import tqdm


def load_and_cache_examples(data_dir,
                            output_dir,
                            tokenizer=None,
                            tokenizer_path=None,
                            line_by_line=False,
                            max_length=512):

    if not tokenizer:
        tokenizer = BertWordPieceTokenizer(tokenizer_path)

    tokenizer.enable_truncation(max_length=max_length)
    tokenizer.enable_padding(max_length=max_length)

    num_tokens = 0
    num_examples = 0

    os.makedirs(output_dir, exist_ok=True)

    pbar = tqdm(os.listdir(data_dir))
    for path in pbar:
        result = process_one_file(data_dir, path, line_by_line, tokenizer, output_dir)
        num_examples += result['num_examples']
        num_tokens += result['num_tokens']

        pbar.set_description(f"{num_tokens} tokens, {num_examples} examples.")


def process_one_file(data_dir, path, line_by_line, tokenizer, output_dir):
    ids = []
    attention_masks = []
    special_tokens_masks = []
    num_tokens = 0
    num_examples = 0

    def add_example(encoded):
        ids.append(encoded.ids)
        attention_masks.append(encoded.attention_mask)
        special_tokens_masks.append(encoded.special_tokens_mask)

    with open(os.path.join(data_dir, path)) as f:
        texts = [line.strip() for line in f.readlines() if line.strip()]
        if not line_by_line:
            texts = [' '.join(texts)]

    for text in texts:
        tokenized = tokenizer.encode(text)
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
            torch.tensor(ids, dtype=torch.long),
            'attention_masks':
            torch.tensor(attention_masks, dtype=torch.long),
            'special_tokens_masks':
            torch.tensor(special_tokens_masks, dtype=torch.long)
        }, os.path.join(output_dir, f"{path}.pt"))

    return {
        'num_tokens': num_tokens,
        'num_examples': num_examples
    }


if __name__ == '__main__':
    fire.Fire(load_and_cache_examples)
