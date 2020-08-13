"""Pytorch lightning module for Discriminative LM task from ELECTRA.

https://openreview.net/forum?id=r1xMH1BtvB
"""
import logging
from dataclasses import dataclass

import pytorch_lightning as pl
import torch
from pytorch_lamb import Lamb
from argparse import Namespace

from transformers_trainers.utils import get_lr_schedule

logger = logging.getLogger(__name__)


@dataclass
class ElectraTrainerConfig(Namespace):
    d_loss_weight: float = 50
    weight_decay: float = 0.01
    learning_rate: float = 5e-4
    epsilon: float = 1e-6
    schedule_type: str = "linear"
    warmup_steps: int = 100000
    total_steps: int = 1e6


class ElectraTrainer(pl.LightningModule):
    def __init__(self, generator, discriminator, tokenizer, config):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.tokenizer = tokenizer
        self.config = config
        self.hparams = config

        # join embeddings
        self.generator.base_model.embeddings = self.discriminator.base_model.embeddings

    def forward(self, inputs, labels):
        # copy the variables for use with discriminator.
        d_inputs = inputs.clone()

        # run masked LM.
        g_out = self.generator(inputs, masked_lm_labels=labels,)

        # get samples from masked LM.
        sample_probs = torch.softmax(g_out[1], dim=-1)
        sample_probs = sample_probs.view(-1, self.generator.config.vocab_size)

        sampled_tokens = torch.multinomial(sample_probs, 1).view(-1)
        sampled_tokens = sampled_tokens.view(d_inputs.shape[0], -1)

        # labels have a -100 value to mask out loss from unchanged tokens.
        mask = labels.ne(-100)

        # replace the masked out tokens of the input with the generator predictions.
        d_inputs[mask] = sampled_tokens[mask]

        # turn mask into new target labels.  1 (True) for corrupted, 0 otherwise.
        # if the prediction was correct, mark it as uncorrupted.
        correct_preds = sampled_tokens == labels
        d_labels = mask.long()
        d_labels[correct_preds] = 0

        # run token classification, predict whether each token was corrupted.
        d_out = self.discriminator(d_inputs, labels=d_labels,)

        g_loss = g_out[0]
        d_loss = d_out[0]
        g_scores = g_out[1]
        d_scores = d_out[1]
        return g_loss, d_loss, g_scores, d_scores, d_labels

    def training_step(self, batch, batch_idx):
        inputs, labels, = batch["input_ids"], batch["labels"]
        g_loss, d_loss, g_scores, d_scores, d_labels = self.forward(inputs, labels,)

        g_preds = torch.argmax(g_scores, dim=-1)
        correct_preds = (g_preds == labels)[labels.ne(-100)]
        g_acc = torch.sum(correct_preds).float() / correct_preds.numel()

        d_preds = torch.argmax(d_scores, dim=-1)
        # d_preds = d_scores
        correct_d_preds = (d_preds == d_labels)[inputs.ne(self.tokenizer.pad_token_id)]
        d_acc = torch.sum(correct_d_preds).float() / correct_d_preds.numel()

        # weight the discriminator loss.
        total_loss = g_loss + (self.config.d_loss_weight * d_loss)

        metrics = {
            "train/loss": total_loss,
            "train/d_loss": d_loss,
            "train/g_loss": g_loss,
            "train/g_acc": g_acc,
            "train/d_acc": d_acc,
        }

        metrics = self.add_training_metrics(metrics, total_loss.device)
        return {"loss": total_loss, "log": metrics}

    def validation_step(self, batch, batch_idx):
        inputs, labels, = batch["input_ids"], batch["labels"]
        g_loss, d_loss, g_scores, d_scores, d_labels = self.forward(inputs, labels,)

        g_preds = torch.argmax(g_scores, dim=-1)
        correct_preds = (g_preds == labels)[labels.ne(-100)]
        g_acc = torch.sum(correct_preds).float() / correct_preds.numel()

        d_preds = torch.argmax(d_scores, dim=-1)
        # d_preds = d_scores
        correct_d_preds = (d_preds == d_labels)[inputs.ne(self.tokenizer.pad_token_id)]
        d_acc = torch.sum(correct_d_preds).float() / correct_d_preds.numel()

        # weight the discriminator loss.
        total_loss = g_loss + (self.config.d_loss_weight * d_loss)
        return {
            "val_loss": total_loss,
            "val_d_loss": d_loss,
            "val_g_loss": g_loss,
            "val_g_acc": g_acc,
            "val_d_acc": d_acc,
        }

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_d_loss = torch.stack([x["val_d_loss"] for x in outputs]).mean()
        avg_g_loss = torch.stack([x["val_g_loss"] for x in outputs]).mean()
        avg_g_acc = torch.stack([x["val_g_acc"] for x in outputs]).mean()
        avg_d_acc = torch.stack([x["val_d_acc"] for x in outputs]).mean()

        perplexity = torch.exp(avg_g_loss)

        metrics = {
            "val_loss": avg_loss,
            "val/loss": avg_loss,
            "val/d_loss": avg_d_loss,
            "val/g_loss": avg_g_loss,
            "val/perplexity": perplexity,
            "val/g_acc": avg_g_acc,
            "val/d_acc": avg_d_acc,
        }
        return {"avg_val_loss": avg_loss, "log": metrics}

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.config.weight_decay,
            },
        ]

        optimizer = Lamb(
            optimizer_grouped_parameters,
            lr=self.config.learning_rate,
            eps=self.config.epsilon,
        )

        scheduler = get_lr_schedule(
            optimizer,
            self.config.schedule_type,
            self.config.warmup_steps,
            self.config.total_steps,
        )

        scheduler_config = {"scheduler": scheduler, "interval": "step"}

        return [optimizer], [scheduler_config]

    def add_training_metrics(self, metrics, device):
        """Store auxiliary training metrics.

        Args:
            metrics (dict): dict of metrics.
            device (torch.device): device the metric tensors are allocated to.
        """
        # HACK need to cast these to tensors
        metrics["epoch"] = torch.tensor(self.current_epoch).float().to(device)

        # get LR schedulers from the pytorch-lightning trainer object.
        scheduler = self.trainer.lr_schedulers[0]["scheduler"]

        # tie LR stepping to global step.
        for i, lr in enumerate(scheduler.get_last_lr()):
            metrics[f"lr_{i}"] = torch.tensor(lr).to(device)

        return metrics
