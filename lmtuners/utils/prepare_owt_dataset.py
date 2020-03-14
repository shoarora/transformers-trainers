import fire
import os
import tarfile


class ExampleWriter:
    def __init__(self, tokenizer_path, name):
        self.tokenizer = BertWordPieceTokenizer(tokenizer_path)

        self.examples = []

    def add_example_file(path):
        with open(path) as f:
            text = f.read()
        examples = self.tokenizer.encode(text)

        self.examples.append(examples)

    def write_examples(output_path):
        

def main(data_dir, tokenizer_path):
    filenames = os.listdir(data_dir)
    for filename in filenames:
        with tarfile.open(os.path.join(data_dir, filename)) as f:
            f.extractall(tmp_dir)
        extracted_files = os.listdir(tmp_dir)
        random.shuffle(extracted_files)
        for textfile in extracted_files:

