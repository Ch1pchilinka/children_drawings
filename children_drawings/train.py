from pathlib import Path

import hydra
import mlflow
import pytorch_lightning as pl
from dvc.exceptions import DvcException
from dvc.repo import Repo
from loggers.resolver import get_logger
from model import MultiHeadEfficientNet
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import MLFlowLogger
from utils import ensure_data

from data import ChildrenDrawingsDataModule


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def train(cfg: DictConfig):
    ensure_data(cfg.data.data_root, train)

    pl.seed_everything(cfg.training.seed)

    datamodule = ChildrenDrawingsDataModule(
        data_root=cfg.data.data_root,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
    )

    model = MultiHeadEfficientNet(
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
        epochs=cfg.training.epochs,
        age_loss_weight=cfg.training.age_loss_weight,
        freeze_below_index=cfg.model.freeze_below_index,
    )

    callbacks = [
        ModelCheckpoint(
            dirpath="artifacts/checkpoints",
            filename="best",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            mode="min",
        ),
        LearningRateMonitor(
            logging_interval="epoch",
        ),
    ]

    trainer = pl.Trainer(
        enable_checkpointing=cfg.training.enable_checkpointing,
        max_epochs=cfg.training.epochs,
        precision=cfg.training.precision,
        logger=get_logger(cfg),
        callbacks=callbacks,
        log_every_n_steps=cfg.training.log_every_n_steps,
    )

    trainer.fit(
        model,
        datamodule=datamodule,
    )


if __name__ == "__main__":
    train()
