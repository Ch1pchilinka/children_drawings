from __future__ import annotations

import json
from pathlib import Path

import hydra
import onnxruntime as ort
from omegaconf import DictConfig

from .constants import OUTPUT_NAMES
from .prediction import decode_numpy_outputs
from .preprocessing import collect_image_paths, load_images_as_numpy_batch
from .utils import ensure_data, ensure_onnx_artifacts, resolve_repo_path


def _ensure_image_paths(images: str | Path) -> list[Path]:
    image_root = resolve_repo_path(images)
    image_paths = collect_image_paths(image_root)
    if image_paths:
        return image_paths

    if not image_root.exists() and image_root.name in {"train", "validation", "batch"}:
        ensure_data(str(image_root.parent), image_root.name)
        image_paths = collect_image_paths(image_root)
        if image_paths:
            return image_paths

    raise FileNotFoundError(f"No images found in {image_root}")


def infer_onnx(cfg: DictConfig):
    onnx_path = resolve_repo_path(cfg.inference.onnx_model)
    if not onnx_path.exists():
        ensure_onnx_artifacts(cfg.inference.onnx_model)

    image_paths = _ensure_image_paths(cfg.inference.images)
    batch, names = load_images_as_numpy_batch(image_paths)

    session = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )
    raw_outputs = session.run(list(OUTPUT_NAMES), {"input": batch})
    predictions = decode_numpy_outputs(*raw_outputs)

    for prediction, image_name in zip(predictions, names, strict=True):
        prediction["image"] = image_name
        print(json.dumps(prediction, ensure_ascii=False, indent=2))


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    infer_onnx(cfg)


if __name__ == "__main__":
    main()
