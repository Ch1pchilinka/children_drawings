import glob
import json
import os
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

from .model import EFFICIENTNET_B3_ARCH, MultiHeadEfficientNet
from .prediction import decode_torch_outputs
from .preprocessing import preprocess_image
from .utils import ensure_data, resolve_repo_path

IMAGE_PATTERNS = ("*.jpg", "*.jpeg", "*.png")


def collect_image_paths(path: str | Path) -> list[Path]:
    image_path = resolve_repo_path(path)
    if not image_path.exists():
        split_name = image_path.name
        ensure_data(str(image_path.parent), split_name)

    if image_path.is_file():
        return [image_path]

    paths: list[Path] = []
    for pattern in IMAGE_PATTERNS:
        paths.extend(Path(p) for p in glob.glob(os.path.join(str(image_path), pattern)))
    return sorted(paths)


def infer(cfg: DictConfig):
    if cfg.model.architecture != EFFICIENTNET_B3_ARCH:
        raise ValueError(
            "Inference is available only for the main model "
            f"('{EFFICIENTNET_B3_ARCH}'). "
            "Baseline ResNet-18 is for train/eval comparison only."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = resolve_repo_path(cfg.inference.checkpoint)

    model = MultiHeadEfficientNet.load_from_checkpoint(
        str(checkpoint_path),
        map_location=device,
    )

    model.eval()

    image_paths = collect_image_paths(cfg.inference.images)
    if not image_paths:
        raise FileNotFoundError(f"No images found in {cfg.inference.images}")

    for image_path in image_paths:
        image = torch.from_numpy(preprocess_image(image_path)).to(device)

        with torch.no_grad():
            outputs = model(image)
            result = decode_torch_outputs(outputs)[0]
            result["image"] = image_path.name

        print(json.dumps(result, indent=2))


@hydra.main(
    config_path="../conf",
    config_name="config",
    version_base=None,
)
def main(cfg: DictConfig):
    infer(cfg)


if __name__ == "__main__":
    main()
