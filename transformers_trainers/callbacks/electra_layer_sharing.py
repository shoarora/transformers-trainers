import pytorch_lightning as pl


class ElectraLayerSharingCallback(pl.Callback):
    def __init__(self, pl_module, starting_num_layers=3, layer_add_rate=10e5):
        self.layer_add_rate = layer_add_rate
        self.starting_num_layers = starting_num_layers
        self.last_update = 0

        self.cur_layers = starting_num_layers
        pl_module.generator.base_model.layers = pl_module.discriminator.base_model.layers[
            self.cur_layers
        ]

    def on_validation_end(self, trainer, pl_module):
        if trainer.global_step > self.last_update + self.layer_add_rate:
            pl_module.generator.base_model.layers = pl_module.discriminator.base_model.layers[
                self.cur_layers + 1
            ]
            self.cur_layers += 1
            self.last_update += self.layer_add_rate
