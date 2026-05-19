"""Resolve the configured logger from Hydra config."""

from omegaconf import DictConfig
from pytorch_lightning.loggers.logger import Logger


def get_logger(cfg: DictConfig) -> Logger:
    """Instantiate the logger specified by ``cfg.logger.type``.

    Reads ``cfg.logger.type`` and forwards the matching sub-config to the
    corresponding factory function.

    Args:
        cfg: Full Hydra config. Must contain a ``logger`` key with at least
            ``type`` and a matching sub-key holding the logger hyperparameters.

    Returns:
        A configured PyTorch Lightning logger instance.

    Raises:
        ValueError: If ``cfg.logger.type`` is not one of the supported values.
    """
    logger_type = cfg.logger.type

    if logger_type == "mlflow":
        from .mlflow_logger import build_logger

        return build_logger(
            tracking_uri=cfg.logger.mlflow.tracking_uri,
            experiment_name=cfg.logger.mlflow.experiment_name,
            run_name=cfg.logger.mlflow.run_name,
            save_dir=cfg.logger.mlflow.save_dir,
        )

    raise ValueError(f"Unknown logger type: '{logger_type}'. Choose one of: mlflow.")
