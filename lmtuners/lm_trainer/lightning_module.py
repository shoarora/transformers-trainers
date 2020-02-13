"""Pytorch lightning module for language modelling."""
import logging
import os
from argparse import Namespace

import pytorch_lightning as pl
import torch
from lmtuners.data import Collater, create_concat_dataset
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup

logger = logging.getLogger(__name__)


class LMTrainingModuleConfig(Namespace):
    """Config class LMTrainingModule."""
    def __init__(
            self,
            data_path,
            mlm=True,
            mlm_prob=0.15,
            max_seq_len=128,
            save_path=None,
            weight_decay=0.0,
            learning_rate=5e-5,
            epsilon=1e-8,
            warmup_steps=0,
            batch_size=32,
            num_workers=0,
            shuffle=True,
            use_linecache=False,
            accumulate_grad_batches=1,
    ):
        super().__init__(data_path=data_path,
                         mlm=mlm,
                         max_seq_len=max_seq_len,
                         mlm_prob=mlm_prob,
                         save_path=save_path,
                         weight_decay=weight_decay,
                         learning_rate=learning_rate,
                         epsilon=epsilon,
                         warmup_steps=warmup_steps,
                         batch_size=batch_size,
                         num_workers=num_workers,
                         shuffle=shuffle,
                         use_linecache=use_linecache,
                         accumulate_grad_batches=accumulate_grad_batches)


class LMTrainingModule(pl.LightningModule):
    def __init__(self, model, tokenizer, config, checkpoint_fn=None):
        super().__init__()
        self.config = config
        self.hparams = config
        self.checkpoint_fn = checkpoint_fn

        self.tokenizer = tokenizer

        self.pad_token_id = self.tokenizer.token_to_id("[PAD]")
        self.mask_token_id = self.tokenizer.token_to_id("[MASK]")
        self.vocab_size = self.model.config.vocab_size

        self.tokenizer.enable_padding(pad_id=self.pad_token_id,
                                      max_length=config.max_seq_len)
        self.tokenizer.enable_truncation(max_length=config.max_seq_len,
                                         strategy='only_first')

        self.model = model

    def forward(self, inputs, labels, attention_mask):
        if self.config.mlm:
            outputs = self.model(inputs,
                                 masked_lm_labels=labels,
                                 attention_mask=attention_mask)
        else:
            outputs = self.model(inputs,
                                 labels=labels,
                                 attention_mask=attention_mask)
        return outputs

    def training_step(self, batch, batch_idx):
        inputs, labels, attention_mask = batch
        outputs = self.forward(inputs, labels, attention_mask)
        loss = outputs[0]
        perplexity = torch.exp(loss)
        self._log_and_step_lr()
        tensorboard_logs = {'train/loss': loss, 'train/perplexity': perplexity}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        inputs, labels, attention_mask = batch
        outputs = self.forward(inputs, labels, attention_mask)
        loss = outputs[0]

        return {'val_loss': loss}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()

        perplexity = torch.exp(avg_loss)

        output_dir = os.path.join(self.config.save_path,
                                  f"{self.current_epoch}-{self.global_step}")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model_to_save = (self.model.module
                         if hasattr(self.model, "module") else self.model)
        model_to_save.save_pretrained(output_dir)

        if hasattr(self.tokenizer, 'save_pretrained'):
            self.tokenizer.save_pretrained(output_dir)
        else:
            self.tokenizer.save(output_dir, 'tokenizer')
        if self.checkpoint_fn:
            self.checkpoint_fn(self)

        tensorboard_logs = {'val/loss': avg_loss, 'val/perplexity': perplexity}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay":
                self.config.weight_decay,
            },
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay":
                0.0
            },
        ]

        t_total = len(self.train_dataloader()) // self.config.batch_size
        t_total = t_total // self.config.accumulate_grad_batches

        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.config.learning_rate,
                          eps=self.config.epsilon)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=t_total)

        return [optimizer], [scheduler]

    def _log_and_step_lr(self):
        """Logs learning rate to tensorboard.
        """
        # get LR schedulers from the pytorch-lightning trainer object.
        scheduler = self.trainer.lr_schedulers[0]
        scheduler.step()
        for i, lr in enumerate(scheduler.get_lr()):
            # add the scalar to the test_tube Experiment object.
            self.logger.experiment.add_scalar(f'lr_{i}', lr, self.global_step)

    @property
    def is_distributed(self):
        if hasattr(self, 'trainer') and self.trainer.distributed_backend:
            return 'ddp' in self.trainer.distributed_backend
        return False

    def get_dataloader(self, path):
        paths = [os.path.join(path, name) for name in os.listdir(path)]
        dataset = create_concat_dataset(self.tokenizer, paths, use_linecache=self.config.use_linecache)

        if hasattr(self, 'is_distributed') and self.is_distributed:
            dist_sampler = torch.utils.data.distributed.DistributedSampler(
                dataset)
        else:
            dist_sampler = None

        collater = Collater(self.tokenizer,
                            mlm=self.config.mlm,
                            mlm_prob=self.config.mlm_prob,
                            pad_token_id=self.pad_token_id,
                            mask_token_id=self.mask_token_id,
                            vocab_size=self.vocab_size)

        return DataLoader(dataset,
                          batch_size=self.config.batch_size,
                          num_workers=self.config.num_workers,
                          collate_fn=collater,
                          sampler=dist_sampler,
                          shuffle=self.config.shuffle)

    @pl.data_loader
    def train_dataloader(self):
        path = os.path.join(self.config.data_path, 'train')
        return self.get_dataloader(path)

    @pl.data_loader
    def val_dataloader(self):
        path = os.path.join(self.config.data_path, 'val')
        return self.get_dataloader(path)

    @pl.data_loader
    def test_dataloader(self):
        path = os.path.join(self.config.data_path, 'test')
        return self.get_dataloader(path)
