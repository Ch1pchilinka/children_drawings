"""Train models on Children Drawings dataset using PyTorch Lightning + Hydra.

Usage:
    python train.py
    python train.py training.epochs=50 data.batch_size=32
    python train.py training.resume=checkpoints/last.ckpt
"""

import hydra
import pytorch_lightning as pl
from model import BaselineModel, MultiHeadEfficientNet
from omegaconf import DictConfig
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from data import ChildrenDrawingsDataModule


@hydra.main(config_path="conf", config_name="config", version_base=None)
def train(cfg: DictConfig) -> None:
    pl.seed_everything(cfg.training.seed, workers=True)

    datamodule = ChildrenDrawingsDataModule(
        data_root=cfg.data.data_root,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
    )

    if cfg.model.name == "baseline":
        model = BaselineModel(
            lr=cfg.training.lr,
            weight_decay=cfg.training.weight_decay,
            epochs=cfg.training.epochs,
        )
    elif cfg.model.name == "efficientnet_b3":
        model = MultiHeadEfficientNet(
            lr=cfg.training.lr,
            weight_decay=cfg.training.weight_decay,
            epochs=cfg.training.epochs,
            freeze_below_index=cfg.model.freeze_below_index,
        )
    else:
        raise ValueError(f"Unknown model: {cfg.model.name}")

    callbacks = [
        ModelCheckpoint(
            dirpath=cfg.training.checkpoint_dir,
            filename="best",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
        ),
        ModelCheckpoint(
            dirpath=cfg.training.checkpoint_dir,
            filename="last",
            save_top_k=1,
            every_n_epochs=1,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    trainer = pl.Trainer(
        max_epochs=cfg.training.epochs,
        precision=cfg.training.precision,
        callbacks=callbacks,
        logger=TensorBoardLogger(cfg.training.log_dir),
        log_every_n_steps=cfg.training.log_every_n_steps,
        deterministic=False,
    )

    trainer.fit(
        model,
        datamodule=datamodule,
        ckpt_path=cfg.training.resume,
    )


if __name__ == "__main__":
    train()
