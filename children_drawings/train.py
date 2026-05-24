import subprocess

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import MLFlowLogger

from .data import ChildrenDrawingsDataModule
from .loggers.resolver import get_logger
from .model import MultiHeadEfficientNet
from .utils import REPO_ROOT, ensure_data, resolve_repo_path


def _resolve_git_commit_id() -> str | None:
    try:
        commit_id = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
        ).strip()
    except (subprocess.SubprocessError, OSError):
        return None
    return commit_id or None


def train(cfg: DictConfig):
    ensure_data(cfg.data.data_root, "train")
    ensure_data(cfg.data.data_root, "validation")

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
        pretrained=cfg.model.pretrained,
    )

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            mode="min",
        ),
        LearningRateMonitor(
            logging_interval="epoch",
        ),
    ]

    if cfg.training.enable_checkpointing:
        callbacks.insert(
            0,
            ModelCheckpoint(
                dirpath=str(
                    resolve_repo_path(cfg.paths.artifacts_path) / "checkpoints"
                ),
                filename="best",
                monitor="val_loss",
                mode="min",
                save_top_k=1,
            ),
        )

    logger = get_logger(cfg)

    git_commit_id = _resolve_git_commit_id()
    if isinstance(logger, MLFlowLogger) and logger.run_id is not None and git_commit_id:
        logger.experiment.log_param(logger.run_id, "git_commit_id", git_commit_id)

    trainer = pl.Trainer(
        enable_checkpointing=cfg.training.enable_checkpointing,
        max_epochs=cfg.training.epochs,
        precision=cfg.training.precision,
        accelerator=cfg.model.device,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=cfg.training.log_every_n_steps,
    )

    trainer.fit(
        model,
        datamodule=datamodule,
    )

    return trainer


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    train(cfg)


if __name__ == "__main__":
    main()
