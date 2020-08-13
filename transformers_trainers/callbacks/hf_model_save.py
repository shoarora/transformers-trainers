import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only


class HFModelSaveCallback(pl.Callback):

    @rank_zero_only
    def on_validation_end(self, trainer, pl_module):
        save_name = f"electra-discriminator-{trainer.global_step}"
        pl_module.discriminator.save_pretrained(save_name)

        if self.polyaxon:
            experiment = pl_module.logger.experiment
            experiment.log_artifacts(save_name)
