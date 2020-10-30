import pytorch_lightning as pl
import logging

logger = logging.getLogger(__name__)


class ElectraLayerSharingCallback(pl.Callback):
    def __init__(self, pl_module, starting_num_layers=3, layer_add_rate=5e4):
        self.layer_add_rate = layer_add_rate
        self.starting_num_layers = starting_num_layers
        self.last_update = 0

        self.cur_layers = starting_num_layers
        self.assign_layers(pl_module.generator, pl_module.discriminator, self.cur_layers)
        self.do_update = True

    def on_validation_end(self, trainer, pl_module):
        if trainer.global_step > self.last_update + self.layer_add_rate and self.do_update:
            self.assign_layers(pl_module.generator, pl_module.discriminator, self.cur_layers + 1)
            self.cur_layers += 1
            self.last_update += self.layer_add_rate

        self.do_update = self.cur_layers < len(pl_module.discriminator.base_model.encoder.layer)

    def assign_layers(self, g, d, num_layers):
        g.base_model.encoder.layer = d.base_model.encoder.layer[:num_layers]
        logger.info(f"Setting generator to use the first {num_layers} layers of the discriminator")
