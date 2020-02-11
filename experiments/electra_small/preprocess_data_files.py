import os
import fire
from random import randint
from deco import synchronized, concurrent


@concurrent(processes=8)
def process_one_file(src_path, dst_path, p, i, min_word, max_word):
    with open(os.path.join(src_path, p)) as f:
        try:
            text = f.readlines()
        except UnicodeDecodeError:
            print('skipping', p)
            return
        text = [line.strip() for line in text if line]
        text = ' '.join(text)
        text = text.split(' ')

    with open(os.path.join(dst_path, p), 'w') as f:
        while text:
            length = randint(min_word, max_word)
            subtext = text[:length]
            f.write(' '.join(subtext) + '\n')
            text = text[length:]

    if i % 100 == 0:
        print(i, 'done')


@synchronized
def preprocess(src_path, dst_path, min_word=5, max_word=128):
    paths = os.listdir(src_path)
    print(len(paths), 'paths to process')
    if not os.path.exists(dst_path):
        os.makedirs(dst_path, exist_ok=True)
    for i, p in enumerate(paths):
        process_one_file(src_path, dst_path, p, i, min_word, max_word)


def main(src_path, dst_path):
    preprocess(src_path, dst_path)


if __name__ == '__main__':
    fire.Fire(main)
