import pytorch_lightning as pl

from .lightning_module import LMTrainingModule, LMTrainingModuleConfig


class LMTrainer:
    def __init__(self,
                 model,
                 tokenizer,
                 data_path,
                 mlm=True,
                 mlm_prob=0.15,
                 save_path=None,
                 weight_decay=0.0,
                 learning_rate=5e-5,
                 adam_epsilon=1e-8,
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
                 amp_level='O2'):
        self.model = model
        self.tokenizer = tokenizer

        config = LMTrainingModuleConfig(data_path=data_path,
                                        mlm=mlm,
                                        mlm_prob=mlm_prob,
                                        save_path=save_path,
                                        weight_decay=weight_decay,
                                        learning_rate=learning_rate,
                                        adam_epsilon=adam_epsilon,
                                        warmup_steps=warmup_steps,
                                        batch_size=batch_size,
                                        num_workers=num_workers,
                                        shuffle=shuffle,
                                        accumulate_grad_batches=accumulate_grad_batches)

        self.training_module = LMTrainingModule(model, tokenizer, config)
        self.trainer = pl.Trainer(
            accumulate_grad_batches=accumulate_grad_batches,
            gpus=gpus,
            distributed_backend=distributed_backend,
            max_nb_epochs=max_nb_epochs,
            fast_dev_run=fast_dev_run,
            use_amp=use_amp,
            amp_level=amp_level)

    def fit(self):
        return self.trainer.fit(self.training_module)
