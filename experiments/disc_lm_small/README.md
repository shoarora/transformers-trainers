# Discriminative LM (small) Experiments

This directory contains code for training:

1. a baseline BERT-small model
2. an unofficial ELECTRA-small model (with a couple changes)
3. an attempt at an ALECTRA-small (an ALBERT model trained in the ELECTRA scheme)

ELECTRA comes from [this paper](https://openreview.net/pdf?id=r1xMH1BtvB),
where they use a discriminative LM task to pretrain transformers.

## Using the models
```python
from transformers import BertForSequenceClassification, AlbertForSequenceClassification, BertTokenizer

# Both models use the bert-base-uncased tokenizer and vocab.
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
electra = BertForSequenceClassification.from_pretrained('shoarora/electra-small-owt')
alectra = AlbertForSequenceClassification.from_pretrained('shoarora/alectra-small-owt')
```

## Data Prep
Download and unpack the [OpenWebText corpus](https://skylion007.github.io/OpenWebTextCorpus/)

Download the vocab file:
```sh
wget https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt
```

Pre-tokenize the data:
We create training data using BERT's normal two-segment data points (with \[SEP\] tokens between).
Our data preprocessing is more crude than the official implementation: we create length-64 samples and concatenate two together.
The original is more intelligent about processing lines and packing in examples.
```sh
python -m lmtuners.utils.tokenize_and_cache_data data/ data_tokenized_128/ --tokenizer_path bert-base-uncased-vocab.txt --max_length=64
```

Split the datatset into train/val/test
```sh
python -m lmtuners.utils.create_dataset_splits data_tokenized_128
```

Install the requirements for experiments
```sh
pip install -r experiments/disc_lm_small/requirements.txt
```

## Training the models

### ELECTRA-small

The electra generator is functionally equivalent to `BertForMaskedLM` and the discriminator is `BertForTokenClassification`.
One difference is that the huggingface Bert implementation doesn't make a distinction between embedding size and hidden size,
which electra does.  Instead of using `embedding_size=128` and `hidden_size=256` as in the official implementation,
we use `embedding_size=hidden_size=256` for convenience.  This results in a slight increase in model size.

Like the paper, we tie the embedding weights between the generator and discriminator.

The electra implementation also ties the decoder weights in the generator, which `BertForMaskedLM` doesn't do by default.

### ALECTRA-small

The discriminative-LM/replaced-token-detection task generalizes to any `*ForMaskedLM` and `*ForTokenClassification` models.  So we trained an ALBERT-small type model in this setting.

We tried three variants:
- use a BERT generator with ALBERT discriminator
- use ALBERT for both
- use ALBERT for both, and tie encoder weights

Mixing BERT and ALBERT didn't perform.  We tied the embeddings, but since ALBERT
projects the embeddings up, the actual embeddings fed into the transformer end up differing
between the two.

We thought one nice feature of ALBERT sharing layer weights, is that an ALBERT generartor and
discriminator could share encoder weights.  We found this to hurt performance at this scale.
It's already difficult to get 4M parameters to learn one task, so trying to hand it two
didn't help me.

From a hierarchical-multitask perspective, it could arguably make more sense to have
the discriminator be the auxiliary/low-level task and have the generator be the primary/complex task,
and share encoder weights.  This doesn't make sense with this model size though, so we don't explore
the possibility.

The reported model uses an ALBERT model for both generator and discriminator.
We tie the input embeddings together.  We see no notable gains by tying the embedding
projection layers, so we leave those separate.  We find the generator to be sensitive
to initialization: some runs perform significantly better with the same configuration.


Notes:
 - When we started dev on this model, `AlbertForTokenClassification` was not yet available, so I
    produced one in this repo.
 - Sentencepiece was also not available in `tokenizers`, so we decieded to keep using the same
    `bert-base-uncased` tokenization.


## Results

I trained these models with max-seq-length 128, similar to the official quickstart models
in the [github repo](https://github.com/google-research/electra), so it's not fit for tasks
that require longer lengths, like SQuAD.  So we'll limit our evaluation to just GLUE.

### GLUE

#### Evaluation Method
- Following BERT, disregard results on WNLI
- For RTE and STS, start from an intermedate checkpoint from MNLI
- average result on MNLI-m and MNLI-mm
- metrics are Spearman correlation for STS, Matthews correlation for CoLA, and accuracy for the
other GLUE tasks
- dev score reported as median of 5 runs
- test set evaluated using single best model by dev score (5 runs)

ELECTRA:
-  For RTE use `lr=2e-5`
-  hyperparameters taken from the appendix in Clark 2020.
-  it would be very possible to produce better scores by doing a more rigorous hyperparameter search.

ALECTRA:
- Performed a hyperparameter search and ended up with the following parameters:
<!-- TODO -->


#### GLUE Dev results
| Model                    | # Params | CoLA | SST | MRPC | STS  | QQP  | MNLI | QNLI | RTE |
| ---                      | ---      | ---  | --- | ---  | ---  | ---  | ---  | ---  | --- |
| ELECTRA-Small++          | 14M      | 57.0 | 91. | 88.0 | 87.5 | 89.0 | 81.3 | 88.4 | 66.7|
| ELECTRA-Small-OWT        | 14M      | 56.8 | 88.3| 87.4 | 86.8 | 88.3 | 78.9 | 87.9 | 68.5|
| ELECTRA-Small-OWT (ours) | 17M      | 56.3 | 88.4| 75.0 | 86.1 | 89.1 | 77.9 | 83.0 | 67.1|
| ALECTRA-Small-OWT (ours) |  4M      | 50.6 | 89.1| 86.3 | 87.2 | 89.1 | 78.2 | 85.9 | 69.6|

- Table initialized from [ELECTRA github repo](https://github.com/google-research/electra)

#### GLUE Test results
| Model                    | # Params | CoLA | SST | MRPC | STS  | QQP  | MNLI | QNLI | RTE |
| ---                      | ---      | ---  | --- | ---  | ---  | ---  | ---  | ---  | --- |
| BERT-Base                | 110M     | 52.1 | 93.5| 84.8 | 85.9 | 89.2 | 84.6 | 90.5 | 66.4|
| GPT                      | 117M     | 45.4 | 91.3| 75.7 | 80.0 | 88.5 | 82.1 | 88.1 | 56.0|
| ELECTRA-Small++          | 14M      | 57.0 | 91.2| 88.0 | 87.5 | 89.0 | 81.3 | 88.4 | 66.7|
| ELECTRA-Small-OWT (ours) | 17M      | 57.4 | 89.3| 76.2 | 81.9 | 87.5 | 78.1 | 82.4 | 68.1|
| ALECTRA-Small-OWT (ours) |  4M      | 43.9 | 87.9| 82.1 | 82.0 | 87.6 | 77.9 | 85.8 | 67.5|

- Table initialized from [original ELECTRA paper](https://openreview.net/pdf?id=r1xMH1BtvB)
