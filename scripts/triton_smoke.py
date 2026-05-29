"""Triton smoke test for the children_drawings model."""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import numpy as np
import tritonclient.http as triton_http

from children_drawings.constants import OUTPUT_NAMES
from children_drawings.prediction import decode_numpy_outputs
from children_drawings.preprocessing import load_images_as_numpy_batch
from children_drawings.utils import resolve_repo_path

IMAGE_PATTERNS = ("*.jpg", "*.jpeg", "*.png")


def _normalize_triton_url(url: str) -> str:
    cleaned = url.strip().rstrip("/")
    cleaned = cleaned.removeprefix("http://")
    cleaned = cleaned.removeprefix("https://")
    return cleaned


def _collect_images(images_dir: str | Path) -> list[Path]:
    root = resolve_repo_path(images_dir)
    image_paths: list[Path] = []
    for pattern in IMAGE_PATTERNS:
        image_paths.extend(Path(path) for path in glob.glob(str(root / pattern)))
    return sorted(image_paths)


def run(
    images_dir: str = "data/batch",
    triton_url: str | None = None,
    model_name: str = "children_drawings",
):
    url = _normalize_triton_url(triton_url or os.getenv("TRITON_URL", "localhost:8100"))

    image_paths = _collect_images(images_dir)
    if not image_paths:
        raise FileNotFoundError(f"No images found in {resolve_repo_path(images_dir)}")

    batch, names = load_images_as_numpy_batch(image_paths)
    batch = batch.astype(np.float32, copy=False)

    client = triton_http.InferenceServerClient(url=url, verbose=False)
    infer_input = triton_http.InferInput("input", list(batch.shape), "FP32")
    infer_input.set_data_from_numpy(batch, binary_data=True)
    outputs = [triton_http.InferRequestedOutput(name) for name in OUTPUT_NAMES]

    result = client.infer(model_name=model_name, inputs=[infer_input], outputs=outputs)
    decoded = decode_numpy_outputs(*(result.as_numpy(name) for name in OUTPUT_NAMES))

    for prediction, image_name in zip(decoded, names, strict=True):
        prediction["image"] = image_name
        print(json.dumps(prediction, ensure_ascii=False))


if __name__ == "__main__":
    run()
