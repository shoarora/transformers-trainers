import os

import datasets
import hydra
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoModelForMaskedLM,
    AutoModelForTokenClassification,
    DataCollatorForLanguageModeling,
    PreTrainedTokenizerFast,
)
from transformers_trainers import ElectraTrainer, ElectraTrainerConfig
from transformers_trainers.datasets import HuggingfaceDataset
from transformers_trainers.utils.logging import (
    WandbCheckpointCallback,
    get_logger_and_ckpt_path,
)

CONFIG_PATH = "config/config.yaml"
CWD = os.path.dirname(os.path.abspath(__file__))


tokenizer = PreTrainedTokenizerFast(
    tokenizer_file="tokenizers/wikipedia-wordpiece",
    pad_token="[PAD]",
    unk_token="[UNK]",
    cls_token="[CLS]",
    sep_token="[SEP]",
    mask_token="[MASK]",
)


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

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=hydra.utils.to_absolute_path(cfg.model.tokenizer_path),
        pad_token="[PAD]",
        unk_token="[UNK]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    )
    # tokenizer = Tokenizer.from_file(hydra.utils.to_absolute_path(cfg.model.tokenizer_path))
    g_config = AutoConfig.from_pretrained(cfg.model.generator_name)
    d_config = AutoConfig.from_pretrained(cfg.model.discriminator_name)

    g_config.vocab_size = tokenizer.vocab_size
    d_config.vocab_size = tokenizer.vocab_size

    generator = AutoModelForMaskedLM.from_config(g_config)
    discriminator = AutoModelForTokenClassification.from_config(d_config)

    train_cfg = ElectraTrainerConfig(**cfg.model.training)
    lightning_module = ElectraTrainer(generator, discriminator, tokenizer, train_cfg)

    train_loader, val_loader = get_dataloaders(tokenizer, cfg.data)

    callbacks = [
        # HFModelSaveCallback()
    ]
    if cfg.logger.type == "wandb":
        callbacks.append(WandbCheckpointCallback())

    logger, ckpt_path = get_logger_and_ckpt_path(cfg.logger)

    trainer = pl.Trainer(
        callbacks=callbacks,
        logger=logger,
        resume_from_checkpoint=ckpt_path,
        **cfg.trainer
    )
    trainer.fit(lightning_module, train_loader, val_loader)


def get_dataloaders(tokenizer, cfg):
    dataset = datasets.load_dataset(
        cfg.dataset_path, cfg.dataset_version, split=datasets.Split.TRAIN
    )
    print(dataset.features)
    dataset.set_format(columns=[cfg.column])

    if cfg.debug:
        dataset = dataset.shard(num_shards=1000, index=0)

    dataset = HuggingfaceDataset(
        dataset, tokenizer, cfg.column, cfg.block_size, cfg.pretokenize
    )

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


if __name__ == "__main__":
    train()
