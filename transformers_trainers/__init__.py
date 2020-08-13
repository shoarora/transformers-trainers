"""transformers_trainers - Tools for training pytorch language models"""
from .datasets import LineByLineTextDataset
from .lightning_modules import ElectraTrainer, ElectraTrainerConfig


__version__ = "0.2.0"
__author__ = "Sho Arora <shoarora@cs.stanford.edu>"
__all__ = [LineByLineTextDataset, ElectraTrainer, ElectraTrainerConfig]
