import os

import hydra
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    ElectraConfig,
    ElectraForMaskedLM,
    ElectraForTokenClassification,
)

from transformers_trainers import ElectraTrainer, ElectraTrainerConfig
from transformers_trainers.datasets import NlpWrapperDataset
from transformers_trainers.callbacks import HFModelSaveCallback
import nlp

CONFIG_PATH = "config/config.yaml"
CWD = os.path.dirname(os.path.abspath(__file__))


@hydra.main(
    config_path=CONFIG_PATH, strict=False,
)
def train(cfg):
    cfg.model.generator_name = os.path.join(
        CWD, "model_configs", cfg.model.generator_name + ".json"
    )
    cfg.model.discriminator_name = os.path.join(
        CWD, "model_configs", cfg.model.discriminator_name + ".json"
    )
    print(cfg.model.generator_name)
    print(cfg.model.discriminator_name)
    g_config = ElectraConfig.from_pretrained(cfg.model.generator_name)
    d_config = ElectraConfig.from_pretrained(cfg.model.discriminator_name)
    generator = ElectraForMaskedLM(g_config)
    discriminator = ElectraForTokenClassification(d_config)

    # d_config.save_pretrained(cfg.model.tokenizer_path)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.tokenizer_path, use_fast=True)

    train_cfg = ElectraTrainerConfig(**cfg.model.training)
    lightning_module = ElectraTrainer(generator, discriminator, tokenizer, train_cfg)

    train_loader, val_loader = get_dataloaders(tokenizer, cfg.data)

    callbacks = [HFModelSaveCallback()]

    logger = get_logger(cfg.logger)

    trainer = pl.Trainer(callbacks=callbacks, logger=logger, **cfg.trainer)
    trainer.fit(lightning_module, train_loader, val_loader)


def get_dataloaders(tokenizer, cfg):
    dataset = nlp.load_dataset(
        cfg.dataset_name, version=cfg.dataset_version, split=nlp.Split.TRAIN
    )
    print(dataset.features)
    dataset.set_format(columns=[cfg.column])

    dataset = NlpWrapperDataset(dataset, tokenizer, cfg.column, cfg.block_size)

    collater = DataCollatorForLanguageModeling(
        tokenizer, mlm=True, mlm_probability=cfg.mlm_probability
    )
    train_loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        collate_fn=collater,
        num_workers=cfg.num_workers,
    )

    return train_loader, train_loader


def get_logger(cfg):
    if cfg.type == "wandb":
        logger = pl.loggers.WandbLogger(**cfg.args)
    elif cfg.type == "comet":
        logger = pl.loggers.CometLogger(**cfg.args)
    else:
        logger = pl.loggers.TensorBoardLogger()
    return logger


if __name__ == "__main__":
    train()
