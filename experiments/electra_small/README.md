# Training an unofficial electra-small model

ELECTRA comes from [this paper](https://openreview.net/pdf?id=r1xMH1BtvB),
where they use a discriminative language modelling task to preptrain transformers.
This experiment tries to approximately replicate their `electra-small` model.

A few changes I made:
1. using an ALBERT model for the generator model (for the smaller GPU footprint)
2. using embedding size = hidden size = 256 (huggingface BERT doesn't let you set them independently).

## Data Prep

First, download the wikipedia dump

```sh
wget https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2
```

Download the wiki extractor script and run it

```sh
wget https://raw.githubusercontent.com/attardi/wikiextractor/master/WikiExtractor.py
python WikiExtractor.py enwiki-latest-pages-articles.xml.bz2

mkdir data
python move_files.py text data
```

Download the [Gutenberg Dataset](https://web.eecs.umich.edu/~lahiri/gutenberg_dataset.html).

```sh
unzip Gutenberg.zip
mv Gutenberg/txt/* data
```

Download the vocab file:
```sh
wget https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt
```

## Run training
Make sure this repo is available on your pythonpath.

If you're running this on a cluster and want to upload your checkpoints as it trains,
there's an example `checkpoint_fn` implemented.

## Evaluation
These create `transformers`-compatible models, so you can evaluate using their `run_glue.py` script.

| Method                   |# params | MNLI | MRPC | QNLI | QQP  | RTE  | SST-2 | STS-B |
|--------------------------|---------|------|------|------|------|------|-------|-------|
| BERT-base                |  110M   | 83.5 | 89.5 | 91.2 | 89.8 | 71.1 | 91.5  | 88.9  |
| DistillBERT              |   66M   | 79.0 | 87.5 | 85.3 | 84.9 | 59.9 | 90.7  | 81.2  |
| GPT                      |   117M  | 88.1 | 75.7 | 85.3 | 88.5 | 56.0 | 91.3  | 80.0  |
| ELECTRA-small (official) |   14M   | 79.7 | 83.7 | 87.7 | 88.0 | 60.8 | 89.1  | 80.3  |
| ELECTRA-small (ours)     |   17M   | ???? | ???? | ???? | ???? | ???? | ????  | ????  |