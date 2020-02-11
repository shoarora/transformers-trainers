import os
import shutil

import fire


def main(src_dir, dst_dir):
    for root, dirs, files in os.walk(src_dir, topdown=False):
        for file in files:
            try:
                shutil.move(os.path.join(root, file), os.path.join(dst_dir, f"{root.replace('/', '')}-{file}"))
            except OSError:
                pass


if __name__ == '__main__':
    fire.Fire(main)
