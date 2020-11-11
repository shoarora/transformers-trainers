import os
import tempfile

import datasets
import fire
import tokenizers


def create_tokenizer(dataset_name, dataset_version, dataset_column, tokenizer_type, output_path, vocab_size=30000):

    dataset = datasets.load_dataset(dataset_name, dataset_version, split=datasets.Split.TRAIN)
    dataset.set_format(columns=[dataset_column])

    dataset_paths = write_dataset_to_tempfile(dataset, dataset_column)

    if tokenizer_type == "bpe":
        tokenizer = tokenizers.ByteLevelBPETokenizer()
    elif tokenizer_type == "wordpiece":
        tokenizer = tokenizers.BertWordPieceTokenizer()
    elif tokenizer_type == "unigram":
        tokenizer = tokenizers.SentencePieceUnigramTokenizer()
    else:
        raise Exception(f"Tokenizer type {tokenizer_type} unsupported.")

    tokenizer.train(files=dataset_paths, vocab_size=vocab_size)
    tokenizer.save(output_path)
    print("saved tokenizer to", output_path)


def write_dataset_to_tempfile(dataset: datasets.Dataset, dataset_column: str, num_shards=32):
    dirname = tempfile.mkdtemp()

    file_paths = [os.path.join(dirname, f"{i}.txt") for i in range(num_shards)]
    file_handles = [open(path, "w") for path in file_paths]

    def write_line(x):
        i = len(x[dataset_column]) % num_shards
        file_handles[i].write(x[dataset_column] + "\n")

    dataset.map(write_line)

    for f in file_handles:
        f.close()

    return file_paths


if __name__ == "__main__":
    fire.Fire(create_tokenizer)
