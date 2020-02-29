# Discriminative LM (small) Experiments

This directory contains code for training:

1. a baseline BERT-small model
2. an unofficial ELECTRA-small model (with a couple changes)
3. an attempt at an ALECTRA-small (an ALBERT model trained in the ELECTRA scheme)

ELECTRA comes from [this paper](https://openreview.net/pdf?id=r1xMH1BtvB),
where they use a discriminative language modelling task to preptrain transformers.

A few changes I made to ELECTRA:
1. using an ALBERT model for the generator model
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

Pre-tokenize the data:
```sh
python -m lmtuners.utils.tokenize_and_cache_data data/ data_tokenized_128/ --tokenizer_path bert-base-uncased-vocab.txt --max_length=128 --n_sentences=40
```

Split the datatset into train/val/test
```sh
python -m lmtuners.utils.create_dataset_splits data_tokenized_128
```

Install the requirements for experiments
```sh
pip install -r experiments/disc_lm_small/requirements.txt

# NOTE, it currently installs my personal branch of pytorch-lightning due
# to a bug fix surrounding passing dataloaders to trainer.fit() directly
# on TPU/DDP.
```

## Training the models

### BERT-small

### ELECTRA-small

### ALECTRA-small

## Results