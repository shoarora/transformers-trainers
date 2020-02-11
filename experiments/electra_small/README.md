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

Preprocess the data

```sh
python preprocess_data_files.py data data_processed
```


Download the tokenizer vocab:

```sh
wget https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt
>>>>>>> Download vocab file
```
