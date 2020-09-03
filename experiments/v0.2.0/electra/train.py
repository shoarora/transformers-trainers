import os

import hydra
import nlp
import pytorch_lightning as pl
import wandb
from torch.utils.data import DataLoader
from transformers import (AutoTokenizer, DataCollatorForLanguageModeling,
                          ElectraConfig, ElectraForMaskedLM,
                          ElectraForTokenClassification)

from transformers_trainers import ElectraTrainer, ElectraTrainerConfig
from transformers_trainers.callbacks import HFModelSaveCallback
from transformers_trainers.datasets import NlpWrapperDataset

CONFIG_PATH = "config/config.yaml"
CWD = os.path.dirname(os.path.abspath(__file__))


@hydra.main(
    config_path=CONFIG_PATH, strict=False,
)
def train(cfg):
    print(cfg.pretty())
    cfg.model.generator_name = os.path.join(
        CWD, "model_configs", cfg.model.generator_name + ".json"
    )
    cfg.model.discriminator_name = os.path.join(
        CWD, "model_configs", cfg.model.discriminator_name + ".json"
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.tokenizer_path, use_fast=True)
    g_config = ElectraConfig.from_pretrained(cfg.model.generator_name)
    d_config = ElectraConfig.from_pretrained(cfg.model.discriminator_name)

    # g_config.vocab_size = tokenizer.vocab_size
    # d_config.vocab_size = tokenizer.vocab_size

    generator = ElectraForMaskedLM(g_config)
    discriminator = ElectraForTokenClassification(d_config)

    train_cfg = ElectraTrainerConfig(**cfg.model.training)
    lightning_module = ElectraTrainer(generator, discriminator, tokenizer, train_cfg)

    train_loader, val_loader = get_dataloaders(tokenizer, cfg.data)

    callbacks = [
        # HFModelSaveCallback()
    ]
    if cfg.logger.type == "wandb":
        callbacks.append(WandbCheckpointCallback())

    logger, ckpt_path = get_logger_and_ckpt_path(cfg.logger)

    trainer = pl.Trainer(callbacks=callbacks, logger=logger, resume_from_checkpoint=ckpt_path, **cfg.trainer)
    trainer.fit(lightning_module, train_loader, val_loader)


def get_dataloaders(tokenizer, cfg):
    dataset = nlp.load_dataset(
        cfg.dataset_path, cfg.dataset_version, split=nlp.Split.TRAIN
    )
    print(dataset.features)
    dataset.set_format(columns=[cfg.column])

    dataset = NlpWrapperDataset(dataset, tokenizer, cfg.column, cfg.block_size, cfg.pretokenize)

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


def get_logger_and_ckpt_path(cfg):
    if cfg.type == "wandb":
        ckpt_path = restore_wandb_experiment(**cfg.args)
        logger = pl.loggers.WandbLogger(**cfg.args)
    elif cfg.type == "comet":
        ckpt_path = None
        logger = pl.loggers.CometLogger(**cfg.args)
    else:
        ckpt_path = None
        logger = pl.loggers.TensorBoardLogger()
    return logger, ckpt_path


def restore_wandb_experiment(project=None, entity=None, epoch=None, version=None, **kwargs):

    api = wandb.Api()
    run_path = f"{entity}/{project}/{version}"
    try:
        run = api.run(run_path)
    except wandb.apis.CommError:
        return None

    if epoch is None:
        epoch = run.summary["epoch"]

    ckpt_path = f"{project}/{version}/checkpoints/epoch={epoch}.ckpt"

    # download checkpoints dir
    restored = wandb.restore(ckpt_path, run_path=run_path)
    print("Restored checkpoint:", ckpt_path, restored.name)
    return restored.name


class WandbCheckpointCallback(pl.Callback):
    def __init__(self):
        super().__init__()
        print("initalized", type(self))
        self.logged_artifact_paths = []

    @pl.utilities.rank_zero_only
    def on_validation_end(self, trainer, pl_module):

        save_dir = trainer.checkpoint_callback.dirpath
        local_checkpoints = os.listdir(save_dir)

        print("val end", save_dir, local_checkpoints)

        for path in local_checkpoints:
            filepath = os.path.join(save_dir, path)
            if filepath not in self.logged_artifact_paths:
                print("saving", filepath)
                wandb.save(filepath)
                self.logged_artifact_paths.append(filepath)


if __name__ == "__main__":
    train()
