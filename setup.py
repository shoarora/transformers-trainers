import io
import os
import re

from setuptools import find_packages, setup


def read(filename):
    filename = os.path.join(os.path.dirname(__file__), filename)
    text_type = type(u"")
    with io.open(filename, mode="r", encoding='utf-8') as fd:
        return re.sub(text_type(r':[a-z]+:`~?(.*?)`'), text_type(r'``\1``'),
                      fd.read())


setup(
    name="polytune",
    version="0.1.0",
    url="htts://github.com/shoarora/polytune",
    license='MIT',
    author="Sho Arora",
    author_email="shoarora@cs.stanford.edu",
    description="Tools for training pytorch language models",
    long_description=read("README.md"),
    packages=find_packages(exclude=('tests', )),
    install_requires=[
        'transformers==2.4.0',
        'tokenizers==0.2.1',
        'fire==0.2.1',
        'pytorch-lightning==0.6.0',
        'pytorch-lamb @ git+ssh://git@github.com/cybertronai/pytorch-lamb.git'
    ],
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.6',
    ],
)
