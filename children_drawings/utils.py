from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

REPO_ROOT = Path(__file__).resolve().parents[1]

DATA_DVC_TARGETS = {
    "train": "data/train.dvc",
    "validation": "data/validation.dvc",
    "batch": "data/batch.dvc",
}

ONNX_DVC_TARGET = "artifacts/onnx_models.dvc"


def resolve_repo_path(path_like: str | Path) -> Path:
    """Resolve config paths against the repository root."""
    path = Path(path_like).expanduser()
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def _import_dvc():
    try:
        from dvc.exceptions import DvcException
        from dvc.repo import Repo
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "DVC is not installed. Install it with `uv sync --extra dvc`."
        ) from exc
    return DvcException, Repo


def _pull_dvc_target(target: str, remote: str | None = None):
    _, Repo = _import_dvc()
    repo = Repo(str(REPO_ROOT))
    pull_kwargs: dict[str, object] = {"targets": [target]}
    if remote is not None:
        pull_kwargs["remote"] = remote
    repo.pull(**pull_kwargs)


def ensure_data(
    data_root: str,
    mode: str,
    remote: str = "r2-storage",
):
    """Проверяет наличие сплита и при необходимости выполняет dvc pull target."""
    DvcException, Repo = _import_dvc()

    data_path = resolve_repo_path(Path(data_root, mode))
    if not data_path.exists() or not any(data_path.iterdir()):
        print("Данные не найдены. Загружаем из облака...")
        try:
            target = DATA_DVC_TARGETS.get(mode)
            if target is not None:
                _pull_dvc_target(target, remote=remote)
            else:
                Repo(str(REPO_ROOT)).pull(remote=remote)
            print("Данные успешно загружены.")
        except DvcException as e:
            raise RuntimeError(f"Ошибка загрузки данных через DVC: {e}") from e
    else:
        print("Данные уже существуют.")


def ensure_onnx_artifacts(
    onnx_model_path: str | Path = "artifacts/onnx_models/children_drawings.onnx",
    remote: str = "r2-models",
):
    """Проверяет наличие ONNX-артефактов и при необходимости выполняет dvc pull."""
    DvcException, _ = _import_dvc()

    onnx_path = resolve_repo_path(onnx_model_path)
    onnx_data_path = resolve_repo_path(f"{onnx_model_path}.data")

    if onnx_path.exists() and onnx_data_path.exists():
        return

    print("ONNX артефакты не найдены. Загружаем из DVC...")
    try:
        _pull_dvc_target(ONNX_DVC_TARGET, remote=remote)
    except DvcException as e:
        raise RuntimeError(f"Ошибка загрузки ONNX через DVC: {e}") from e

    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model still missing after pull: {onnx_path}")


def load_images_as_tensor_batch(
    image_paths: list[Path],
) -> tuple["torch.Tensor", list[str]]:
    """Load images from disk and stack them into a BCHW tensor batch."""
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "PyTorch is not installed. Install it with `uv sync --extra train`."
        ) from exc

    from .preprocessing import load_images_as_numpy_batch

    batch, names = load_images_as_numpy_batch(image_paths)
    return torch.from_numpy(batch), names
