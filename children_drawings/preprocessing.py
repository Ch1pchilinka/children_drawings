from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
from PIL import Image

from .constants import IMAGE_PATTERNS, IMAGE_SIZE, MEAN, STD
from .utils import resolve_repo_path


def preprocess_pil_image(image: Image.Image) -> np.ndarray:
    """Convert a PIL image to a normalized BCHW float32 numpy batch."""
    image_array = np.asarray(
        image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE)),
        dtype=np.float32,
    )
    image_array /= 255.0
    image_array -= np.asarray(MEAN, dtype=np.float32)
    image_array /= np.asarray(STD, dtype=np.float32)
    nchw = np.transpose(image_array, (2, 0, 1))
    return np.expand_dims(nchw, axis=0).astype(np.float32, copy=False)


def preprocess_image(path: str | Path) -> np.ndarray:
    with Image.open(path) as image:
        return preprocess_pil_image(image)


def collect_image_paths(path: str | Path) -> list[Path]:
    image_path = resolve_repo_path(path)
    if image_path.is_file():
        return [image_path]
    if not image_path.exists():
        return []

    paths: list[Path] = []
    for pattern in IMAGE_PATTERNS:
        paths.extend(Path(p) for p in glob.glob(str(image_path / pattern)))
    return sorted(paths)


def load_images_as_numpy_batch(
    image_paths: list[Path],
) -> tuple[np.ndarray, list[str]]:
    """Load images from disk and stack them into a BCHW numpy batch."""
    batches: list[np.ndarray] = []
    names: list[str] = []
    for image_path in image_paths:
        batches.append(preprocess_image(image_path))
        names.append(image_path.name)

    if not batches:
        raise ValueError("No images were loaded for inference.")

    return np.concatenate(batches, axis=0), names
