import tempfile

import datasets
import fire
import tokenizers


def create_tokenizer(dataset_name, dataset_version, dataset_column, tokenizer_type, output_path, vocab_size=30000):

    dataset = datasets.load_dataset(dataset_name, dataset_version, split=datasets.Split.TRAIN)
    dataset.set_format(columns=[dataset_column])

    filename = tempfile.mkstemp()[1]
    f = open(filename, "w")
    dataset.map(lambda x: f.write(x[dataset_column] + "\n"))
    f.close()

    if tokenizer_type == "bpe":
        tokenizer = tokenizers.ByteLevelBPETokenizer()
    elif tokenizer_type == "wordpiece":
        tokenizer = tokenizers.BertWordPieceTokenizer()
    elif tokenizer_type == "unigram":
        tokenizer = tokenizers.SentencePieceUnigramTokenizer()
    else:
        raise Exception(f"Tokenizer type {tokenizer_type} unsupported.")

    tokenizer.train(files=filename, vocab_size=vocab_size)
    tokenizer.save(output_path)


if __name__ == "__main__":
    fire.Fire(create_tokenizer)
