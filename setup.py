import io
import os
import re

from setuptools import find_packages, setup
import transformers_trainers


def read(filename):
    filename = os.path.join(os.path.dirname(__file__), filename)
    text_type = type(u"")
    with io.open(filename, mode="r", encoding='utf-8') as fd:
        return re.sub(text_type(r':[a-z]+:`~?(.*?)`'), text_type(r'``\1``'),
                      fd.read())


setup(
    name="transformers_trainers",
    version=transformers_trainers.__version__,
    url="htts://github.com/shoarora/transformers-trainers",
    license='MIT',
    author="Sho Arora",
    author_email="shoarora@cs.stanford.edu",
    description="Tools for training pytorch language models",
    long_description=read("README.md"),
    packages=find_packages(exclude=('tests', )),
    install_requires=[
        'transformers>=3.0.0',
        'fire',
        'pytorch-lightning>=0.8.5',
        'pytorch-lamb @ git+ssh://git@github.com/cybertronai/pytorch-lamb.git'
    ],
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.6',
    ],
)
