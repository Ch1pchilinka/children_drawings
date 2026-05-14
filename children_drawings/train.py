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

from data import ChildrenDrawingsDataModule


def ensure_data(data_root: str):
    """Проверяет наличие данных и при необходимости выполняет dvc pull."""
    data_path = Path(data_root, "train")
    if not data_path.exists() or not any(data_path.iterdir()):
        print("Данные не найдены. Загружаем из облака...")
        try:
            Repo(".").pull()
            print("Данные успешно загружены.")
        except DvcException as e:
            raise RuntimeError(f"Ошибка загрузки данных через DVC: {e}")
    else:
        print("Данные уже существуют.")


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def train(cfg: DictConfig):
    ensure_data(cfg.data.data_root)

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
            dirpath="checkpoints",
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
