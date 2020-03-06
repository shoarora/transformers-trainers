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

### GLUE

#### Evaluation Method
- Following BERT, disregard results on WNLI
- For RTE and STS, start from an intermedate checkpoint from MNLI
- For RTE do this with `lr=2e-5`

- accuracy is used for SST2, MNLI, QNLI, and RTE
- F1 and accuracy are used for MRPC and QQP
- pearson and spearman are used for STS-B
- matthew's is used for CoLA
- average result on MNLI-m and MNLI-mm; the results on MRPC and QQP are reported with
    the average of F1 and accuracy; the result reported on
    STS-B is the average of the Pearson correlation and Spearman correlation.

#### Metrics
All reported results are on GLUE dev _except_ ELECTRA-small (official)
which only reported task-by-task scores for the test set.

| Model                          |# params | MNLI | MRPC | QNLI | QQP  | RTE  | SST-2 | STS-B |
|--------------------------------|---------|------|------|------|------|------|-------|-------|
| BERT-base                      |  110M   | 83.5 | 89.5 | 91.2 | 89.8 | 71.1 | 91.5  | 88.9  |
| DistillBERT                    |   66M   | 79.0 | 87.5 | 85.3 | 84.9 | 59.9 | 90.7  | 81.2  |
| GPT                            |   117M  | 88.1 | 75.7 | 85.3 | 88.5 | 56.0 | 91.3  | 80.0  |
| ELECTRA-small (official test)  |   14M   | 79.7 | 83.7 | 87.7 | 88.0 | 60.8 | 89.1  | 80.3  |
| ELECTRA-small (ours)           |   17M   | ???? | ???? | ???? | ???? | ???? | ????  | ????  |
| ALECTRA-small (ours)           |   ??M   | ???? | ???? | ???? | ???? | ???? | ????  | ????  |