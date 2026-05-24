import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig

from .data import ChildrenDrawingsDataModule
from .loggers.resolver import get_logger
from .model import load_model_from_checkpoint
from .utils import ensure_data, resolve_repo_path


def evaluate(cfg: DictConfig):
    ensure_data(cfg.data.data_root, "validation")

    checkpoint_path = resolve_repo_path(cfg.inference.checkpoint)
    model = load_model_from_checkpoint(
        str(checkpoint_path),
        architecture=cfg.model.architecture,
        map_location="cpu",
    )

    datamodule = ChildrenDrawingsDataModule(
        data_root=cfg.data.data_root,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
    )

    trainer = pl.Trainer(
        accelerator=cfg.model.device,
        precision=cfg.training.precision,
        logger=get_logger(cfg),
    )
    return trainer.validate(
        model, datamodule=datamodule, ckpt_path=str(checkpoint_path)
    )


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    metrics = evaluate(cfg)
    print(metrics)


if __name__ == "__main__":
    main()
