import os

import pytorch_lightning as pl
import wandb


def get_logger_and_ckpt_path(cfg):
    if cfg.type == "wandb":
        ckpt_path = restore_wandb_experiment(**cfg.args)
        logger = pl.loggers.WandbLogger(**cfg.args)
    elif cfg.type == "comet":
        ckpt_path = None
        logger = pl.loggers.CometLogger(**cfg.args)
    else:
        ckpt_path = None
        logger = pl.loggers.TensorBoardLogger()
    return logger, ckpt_path


def restore_wandb_experiment(
    project=None, entity=None, epoch=None, version=None, **kwargs
):
    api = wandb.Api()
    run_path = f"{entity}/{project}/{version}"
    try:
        run = api.run(run_path)
    except Exception as e:
        print("Wandb experiemtn not found:", e)
        return None

    if epoch is None:
        epoch = run.summary["epoch"]

    ckpt_path = f"{project}/{version}/checkpoints/epoch={epoch}.ckpt"

    # download checkpoints dir
    restored = wandb.restore(ckpt_path, run_path=run_path)
    print("Restored checkpoint:", ckpt_path, restored.name)
    return restored.name


class WandbCheckpointCallback(pl.Callback):
    def __init__(self):
        super().__init__()
        print("initalized", type(self))
        self.logged_artifact_paths = []

    def on_validation_end(self, trainer, pl_module):

        save_dir = trainer.checkpoint_callback.dirpath
        local_checkpoints = os.listdir(save_dir)

        print("val end", save_dir, local_checkpoints)

        for path in local_checkpoints:
            filepath = os.path.join(save_dir, path)
            if filepath not in self.logged_artifact_paths:
                print("saving", filepath)
                wandb.save(filepath)
                self.logged_artifact_paths.append(filepath)
