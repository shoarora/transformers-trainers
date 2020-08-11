"""Split a directory of files into train/val/test subdirectories."""
import os

import fire


def create_dataset_splits(dir_path,
                          train_split=0.8,
                          val_split=0.1,
                          test_split=None):
    """Split a directory of text files into subdirectories.

    Args:
        dir_path (str): Path to directory of files.
        train_split (float): Fraction of files to move to train split.
        val_split (float): Fraction of files to move to val split.
        test_split (float): Fraction of files to move to test split. (Defaults to remainder.)
    """
    filenames = os.listdir(dir_path)

    n = len(filenames)
    idx_train = int(n * train_split)
    idx_val = idx_train + int(n * val_split)

    train_files = filenames[:idx_train]
    val_files = filenames[idx_train:idx_val]

    if test_split:
        idx_test = idx_val + int(n * test_split)
        test_files = filenames[idx_val, idx_test]
    else:
        test_files = filenames[idx_val:]

    create_split_dir(dir_path, 'train', train_files)
    create_split_dir(dir_path, 'val', val_files)
    create_split_dir(dir_path, 'test', test_files)


def create_split_dir(dir_path, name, files):
    """Put list of files into new subdirectory with the specified name.

    Args:
        dir_path (str): Path to files.
        name (str): Name of subdirectory to create within dir_path.
        files ([str]): List of filenames to move from dir_path into dir_path/name.

    Returns:
        type: Description of returned object.

    """
    new_dir = os.path.join(dir_path, name)
    os.mkdir(new_dir)
    move_files(dir_path, new_dir, files)


def move_files(src, dst, filenames):
    """Move a list of files from src to dst."""
    for filename in filenames:
        os.rename(os.path.join(src, filename), os.path.join(dst, filename))


if __name__ == '__main__':
    fire.Fire(create_dataset_splits)
