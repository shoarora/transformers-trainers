"""Pytorch lightning module for Discriminative LM task from ELECTRA.

https://openreview.net/forum?id=r1xMH1BtvB
"""
import logging
import os
from argparse import Namespace

import numpy as np

import pytorch_lightning as pl
import torch
from lmtuners.data import Collater, create_concat_dataset
from pytorch_lamb import Lamb
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

logger = logging.getLogger(__name__)


class DiscLMTrainingModuleConfig(Namespace):
    """Config class for DiscLMTrainingModule."""
    def __init__(self,
                 data_path,
                 d_loss_weight=50,
                 mlm=True,
                 mlm_prob=0.15,
                 save_path=None,
                 max_seq_len=128,
                 weight_decay=0.0,
                 learning_rate=5e-5,
                 epsilon=1e-8,
                 warmup_steps=0,
                 batch_size=32,
                 num_workers=0,
                 shuffle=True,
                 accumulate_grad_batches=1):
        super().__init__(data_path=data_path,
                         d_loss_weight=d_loss_weight,
                         mlm=mlm,
                         mlm_prob=mlm_prob,
                         max_seq_len=max_seq_len,
                         save_path=save_path,
                         weight_decay=weight_decay,
                         learning_rate=learning_rate,
                         epsilon=epsilon,
                         warmup_steps=warmup_steps,
                         batch_size=batch_size,
                         num_workers=num_workers,
                         shuffle=shuffle,
                         accumulate_grad_batches=accumulate_grad_batches)


class DiscLMTrainingModule(pl.LightningModule):
    def __init__(self, generator, discriminator, tokenizer, config, checkpoint_fn=None, ddp_fn=None):
        super().__init__()

        self.config = config
        self.hparams = config
        self.checkpoint_fn = checkpoint_fn
        self.ddp_fn = ddp_fn
        if ddp_fn:
            logger.warning('ddp_fn functionality is not implemented yet.')

        print('set hparams:', self.hparams)

        self.tokenizer = tokenizer

        # get special tokens.
        self.pad_token_id = self.tokenizer.token_to_id("[PAD]")
        self.mask_token_id = self.tokenizer.token_to_id("[MASK]")
        self.vocab_size = generator.config.vocab_size

        # configure max length and padding.
        self.tokenizer.enable_padding(pad_id=self.pad_token_id,
                                      max_length=config.max_seq_len)
        self.tokenizer.enable_truncation(max_length=config.max_seq_len,
                                         strategy='only_first')

        self.generator = generator
        self.discriminator = discriminator

    def forward(self, inputs, labels, attention_mask):
        # copy the variables for use with discriminator.
        d_inputs, d_labels = inputs.clone(), labels.clone()

        # run masked LM.
        g_out = self.generator(inputs,
                               masked_lm_labels=labels,
                               attention_mask=attention_mask)

        # get predictions for masked LM.
        preds = torch.argmax(g_out[1], dim=-1)

        # labels have a -100 value to mask out loss from unchanged tokens.
        mask = labels.eq(-100)

        # replace the masked out tokens of the input with the generator predictions.
        d_inputs[mask] = preds[mask]

        # turn mask into new target labels.  1 (True) for corrupted, 0 otherwise.
        # if the prediction was correct, mark it as uncorrupted.
        correct_preds = preds == labels
        d_labels[correct_preds] = False
        d_labels = mask.long()

        # run token classification, predict whether each token was corrupted.
        d_out = self.discriminator(d_inputs,
                                   labels=d_labels,
                                   attention_mask=attention_mask)
        return g_out, d_out, d_labels

    def training_step(self, batch, batch_idx):
        inputs, labels, attention_mask = batch
        g_out, d_out, d_labels = self.forward(inputs, labels, attention_mask)

        g_loss = g_out[0]
        d_loss = d_out[0]

        scores = d_out[1]
        preds = torch.argmax(scores, dim=-1)
        acc = torch.sum(preds == d_labels).item() / np.prod(d_labels.shape)
        acc = torch.tensor(acc)

        # weight the discriminator loss.
        total_loss = g_loss + (self.config.d_loss_weight * d_loss)

        self._log_and_step_lr()

        tensorboard_logs = {
            'train/loss': total_loss,
            'train/d_loss': d_loss,
            'train/g_loss': g_loss,
            'train/d_acc': acc
        }
        return {'loss': total_loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        inputs, labels, attention_mask = batch
        g_out, d_out, d_labels = self.forward(inputs, labels, attention_mask)

        g_loss = g_out[0]
        d_loss = d_out[0]

        scores = d_out[1]
        preds = torch.argmax(scores, dim=-1)
        acc = torch.sum(preds == d_labels).item() / np.prod(d_labels.shape)
        acc = torch.tensor(acc)

        # weight the discriminator loss.
        total_loss = g_loss + (self.config.d_loss_weight * d_loss)
        return {
            'val_loss': total_loss,
            'val_d_loss': d_loss,
            'val_g_loss': g_loss,
            'val_d_acc': acc
        }

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_d_loss = torch.stack([x['val_d_loss'] for x in outputs]).mean()
        avg_g_loss = torch.stack([x['val_g_loss'] for x in outputs]).mean()
        avg_d_acc = torch.stack([x['val_d_acc'] for x in outputs]).mean()

        perplexity = torch.exp(torch.tensor(avg_g_loss))

        self._save_model(self.generator, 'generator')
        self._save_model(self.discriminator, 'discriminator')

        output_dir = os.path.join(self.config.save_path,
                                  f"{self.current_epoch}-{self.global_step}")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if hasattr(self.tokenizer, 'save_pretrained'):
            self.tokenizer.save_pretrained(output_dir)
        else:
            self.tokenizer.save(output_dir, 'tokenizer')

        if self.checkpoint_fn:
            self.checkpoint_fn(self)

        tensorboard_logs = {
            'val/loss': avg_loss,
            'val/d_loss': avg_d_loss,
            'val/g_loss': avg_g_loss,
            'val/perplexity': perplexity,
            'val/d_acc': avg_d_acc
        }
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

    def _save_model(self, model, name):
        output_dir = os.path.join(self.config.save_path, name,
                                  f"{self.current_epoch}-{self.global_step}")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        model_to_save = (model.module if hasattr(model, "module") else model)
        model_to_save.save_pretrained(output_dir)

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in self.generator.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ] + [
                    p for n, p in self.discriminator.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay":
                self.config.weight_decay,
            },
            {
                "params": [
                    p for n, p in self.generator.named_parameters()
                    if any(nd in n for nd in no_decay)
                ] + [
                    p for n, p in self.discriminator.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay":
                0.0
            },
        ]

        t_total = len(self.train_dataloader()) // self.config.batch_size
        t_total = t_total // self.config.accumulate_grad_batches

        optimizer = Lamb(optimizer_grouped_parameters, lr=self.config.learning_rate, eps=self.config.epsilon)

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
        dataset = create_concat_dataset(self.tokenizer, paths)

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
