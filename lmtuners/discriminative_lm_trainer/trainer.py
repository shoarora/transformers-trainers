"""Custom Trainer abstraction for Discriminative LM task from ELECTRA.

https://openreview.net/forum?id=r1xMH1BtvB
"""
import logging
import os

import pytorch_lightning as pl

from .lightning_module import DiscLMTrainingModule, DiscLMTrainingModuleConfig

logger = logging.getLogger(__name__)


class DiscLMTrainer:
    """Custom trainer class for Discriminative LM task.

    Args:
        generator (*ForMaskedLM): generator language model for training the discriminator.
        discriminator (*ForTokenClassification): Discriminator LM, the one that's being trained.
        tokenizer (BertWordPieceTokenizer): Tokenizer from huggingface/tokenizers.
        data_path (str): Path to a data directory, with train/val/test subdirectories of files.
        mlm (bool): Whether to perform masked LM for the generator.
        mlm_prob (float): Masking probability.
        save_path (str): Path to save models and tokenizers.
        weight_decay (float): Weight decay.
        learning_rate (float): Optimizer learning rate.
        epsilon (float): Optimizer epsilon.
        warmup_steps (int): Num warmup steps for linear schedule.
        batch_size (int): Number of data points per batch.
        num_workers (int): NUmber of data loader workers.  (note: each process takes a lot memory)
        shuffle (bool): Whether to shuffle data points.
        ===============================================
        Pytorch-Lightning Trainer Args.  See their docs
        ===============================================
        accumulate_grad_batches (int): Description of parameter `accumulate_grad_batches`.
        gpus (type): Description of parameter `gpus`.
        distributed_backend (type): Description of parameter `distributed_backend`.
        max_nb_epochs (type): Description of parameter `max_nb_epochs`.
        fast_dev_run (type): Description of parameter `fast_dev_run`.
        use_amp (type): Description of parameter `use_amp`.
        amp_level (type): Description of parameter `amp_level`.
        val_check_interval (type): Description of parameter `val_check_interval`.
        ===============================================
        Custom hooks
        ===============================================
        checkpoint_fn (callable): Function to be called after checkpoints saved.
        ddp_fn (callable): Not Implemented.

    Attributes:
        training_module (pl.LightningModule): the LightningModule that gets trained.
        trainer (pl.Trainer): The underlying trainer.
        generator
        discriminator
        tokenizer

    """

    def __init__(self,
                 generator,
                 discriminator,
                 tokenizer,
                 data_path,
                 mlm=True,
                 mlm_prob=0.15,
                 save_path=None,
                 weight_decay=0.0,
                 learning_rate=5e-5,
                 epsilon=1e-8,
                 warmup_steps=0,
                 batch_size=32,
                 num_workers=0,
                 shuffle=True,
                 accumulate_grad_batches=1,
                 gpus=1,
                 distributed_backend=None,
                 max_nb_epochs=50,
                 fast_dev_run=False,
                 use_amp=False,
                 amp_level='O2',
                 val_check_interval=0.25,
                 checkpoint_fn=None,
                 ddp_fn=None):
        self.generator = generator
        self.discriminator = discriminator
        self.tokenizer = tokenizer

        assert generator.config.vocab_size == discriminator.config.vocab_size

        logging.debug('checking data_path contents')
        self.check_data_path(data_path)

        logging.debug('creating module config')
        config = DiscLMTrainingModuleConfig(data_path=data_path,
                                            mlm=mlm,
                                            mlm_prob=mlm_prob,
                                            save_path=save_path,
                                            weight_decay=weight_decay,
                                            learning_rate=learning_rate,
                                            epsilon=epsilon,
                                            warmup_steps=warmup_steps,
                                            batch_size=batch_size,
                                            num_workers=num_workers,
                                            shuffle=shuffle,
                                            accumulate_grad_batches=accumulate_grad_batches)

        logging.debug('creating training module')
        self.training_module = DiscLMTrainingModule(generator, discriminator,
                                                    tokenizer, config, checkpoint_fn=checkpoint_fn, ddp_fn=ddp_fn)
        logging.debug('creating pytorch-lightning trainer')
        self.trainer = pl.Trainer(
            accumulate_grad_batches=accumulate_grad_batches,
            gpus=gpus,
            distributed_backend=distributed_backend,
            max_nb_epochs=max_nb_epochs,
            fast_dev_run=fast_dev_run,
            use_amp=use_amp,
            amp_level=amp_level,
            val_check_interval=val_check_interval)

    def fit(self):
        logging.info(f"generator: {self.generator}")
        logging.info(f"discriminator: {self.discriminator}")
        return self.trainer.fit(self.training_module)

    def check_data_path(self, data_path):
        expected_data_dirs = ['train', 'val', 'test']
        data_dir_contents = os.listdir(data_path)
        for expected in expected_data_dirs:
            assert expected in data_dir_contents
            assert os.path.isdir(os.path.join(data_path, expected))
