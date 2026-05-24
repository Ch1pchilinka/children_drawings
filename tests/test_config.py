from hydra import compose, initialize_config_dir

from children_drawings.utils import REPO_ROOT


def test_hydra_config_loads():
    with initialize_config_dir(
        config_dir=str(REPO_ROOT / "conf"),
        version_base=None,
    ):
        cfg = compose(config_name="config")

    assert cfg.data.batch_size > 0
    assert cfg.model.opset >= 18
    assert cfg.model.architecture == "efficientnet_b3"
    assert cfg.logger.type == "mlflow"
