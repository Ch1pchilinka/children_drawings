"""MLflow logger factory."""

from urllib.parse import urlparse

import requests
from pytorch_lightning.loggers import MLFlowLogger


def _is_http_tracking_uri(tracking_uri: str) -> bool:
    parsed = urlparse(tracking_uri)
    return parsed.scheme in {"http", "https"}


def _assert_tracking_server_available(tracking_uri: str):
    if not _is_http_tracking_uri(tracking_uri):
        return

    base_uri = tracking_uri.rstrip("/")
    health_urls = (
        f"{base_uri}/health",
        f"{base_uri}/version",
    )

    for health_url in health_urls:
        try:
            response = requests.get(
                health_url,
                timeout=3.0,
            )
        except requests.RequestException:
            continue

        if response.status_code == 200:
            return

    raise RuntimeError(
        "MLflow tracking server is unavailable "
        f"at '{tracking_uri}'. Start it with "
        "`docker compose up mlflow` or override "
        "`logger.mlflow.tracking_uri` / `MLFLOW_TRACKING_URI`."
    )


def build_logger(
    tracking_uri: str, experiment_name: str, run_name: str | None, save_dir: str
) -> MLFlowLogger:
    """Instantiate an MLFlowLogger from config parameters.

    Args:
        tracking_uri: MLflow tracking server URI
            (e.g. ``"http://localhost:5000"`` or a local path).
        experiment_name: Name of the MLflow experiment to log into.
        run_name: Display name for this run. ``None`` lets MLflow generate one.

    Returns:
        Configured :class:`MLFlowLogger`.
    """
    _assert_tracking_server_available(tracking_uri)

    return MLFlowLogger(
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
        run_name=run_name,
        save_dir=save_dir,
    )
