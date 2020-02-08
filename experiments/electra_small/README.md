## Data Prep

First, download the wikipedia dump

```sh
wget https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2
```

Download the wiki extractor script and run it

```sh
wget https://raw.githubusercontent.com/attardi/wikiextractor/master/WikiExtractor.py
python --input enwiki-latest-pages-articles.xml.bz2
```

Download the [Gutenberg Dataset](https://web.eecs.umich.edu/~lahiri/gutenberg_dataset.html).

```sh
unzip Gutenberg.zip
mv gutenberg/txt/* data
