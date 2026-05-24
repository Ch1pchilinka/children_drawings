from pathlib import Path

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from dvc.exceptions import DvcException
from dvc.repo import Repo
from PIL import Image

NUM_CLASSES = 4

CLASS_NAMES = [
    "house",
    "tree",
    "man",
    "woman",
]

GENDER_NAMES = [
    "male",
    "female",
]

IMAGE_SIZE = 300

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

CATEGORY_MAP = {
    "집": 0,
    "나무": 1,
    "남자사람": 2,
    "여자사람": 3,
}

GENDER_MAP = {
    "남": 0,
    "여": 1,
}

TRAIN_TRANSFORMS = A.Compose(
    [
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Rotate(limit=15, p=0.3),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ]
)

VAL_TRANSFORMS = A.Compose(
    [
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ]
)

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


def preprocess_pil_image(image: Image.Image) -> torch.Tensor:
    """Convert a PIL image to a normalized BCHW tensor."""
    image_array = np.array(image.convert("RGB"))
    tensor = VAL_TRANSFORMS(image=image_array)["image"]
    return tensor.unsqueeze(0).float()


def preprocess_image(path: str | Path) -> torch.Tensor:
    with Image.open(path) as image:
        return preprocess_pil_image(image)


def _pull_dvc_target(target: str, remote: str | None = None):
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
            raise RuntimeError(f"Ошибка загрузки данных через DVC: {e}")
    else:
        print("Данные уже существуют.")


def ensure_onnx_artifacts(
    onnx_model_path: str | Path = "artifacts/onnx_models/children_drawings.onnx",
    remote: str = "r2-models",
):
    """Проверяет наличие ONNX-артефактов и при необходимости выполняет dvc pull."""
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
) -> tuple[torch.Tensor, list[str]]:
    """Load images from disk and stack them into a BCHW tensor batch."""
    tensors: list[torch.Tensor] = []
    names: list[str] = []
    for image_path in image_paths:
        tensor = preprocess_image(image_path)
        # Убираем лишнее измерение [1, C, H, W] -> [C, H, W]
        tensor = tensor.squeeze(0)
        tensors.append(tensor)
        names.append(image_path.name)
    if not tensors:
        raise ValueError("No images were loaded for benchmarking.")
    return torch.stack(tensors, dim=0), names
