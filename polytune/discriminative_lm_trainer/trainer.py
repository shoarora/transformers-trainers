import logging
import os

import pytorch_lightning as pl

from .lightning_module import DiscLMTrainingModule, DiscLMTrainingModuleConfig

logger = logging.getLogger(__name__)


class DiscLMTrainer:
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
                 epsion=1e-8,
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
                 val_check_interval=0.25):
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
                                            epsion=epsion,
                                            warmup_steps=warmup_steps,
                                            batch_size=batch_size,
                                            num_workers=num_workers,
                                            shuffle=shuffle,
                                            accumulate_grad_batches=accumulate_grad_batches)

        logging.debug('creating training module')
        self.training_module = DiscLMTrainingModule(generator, discriminator,
                                                    tokenizer, config)
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
