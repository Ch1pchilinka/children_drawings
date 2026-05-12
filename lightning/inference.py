"""Inference script for children drawing classification.

Usage:
    python infer.py checkpoint=checkpoints/best.ckpt image=path/to/drawing.jpg
"""

import json
from pathlib import Path

import hydra
import numpy as np
import torch
from model import BaselineModel, MultiHeadEfficientNet
from omegaconf import DictConfig
from PIL import Image
from utils import CLASS_NAMES, IMAGE_SIZE, MEAN, STD

GENDER_NAMES = ["male", "female"]


def preprocess_image(image_path: str) -> torch.Tensor:
    """Загружает и предобрабатывает изображение."""
    image = Image.open(image_path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    image = np.array(image) / 255.0
    image = (image - np.array(MEAN)) / np.array(STD)
    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
    return tensor


@hydra.main(config_path="conf", config_name="infer_config", version_base=None)
def infer(cfg: DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_class = (
        BaselineModel if cfg.model.name == "baseline" else MultiHeadEfficientNet
    )
    model = model_class.load_from_checkpoint(cfg.checkpoint, map_location=device)
    model.eval()

    image_tensor = preprocess_image(cfg.image).to(device)

    with torch.no_grad():
        if cfg.model.name == "baseline":
            logits = model(image_tensor)
            probs = logits.softmax(dim=-1).squeeze(0).cpu().numpy()
            pred_class = int(probs.argmax())
            result = {
                "class": CLASS_NAMES[pred_class],
                "confidence": float(probs[pred_class]),
            }
        else:
            outputs = model(image_tensor)
            probs_cat = outputs["category"].softmax(dim=-1).squeeze(0).cpu().numpy()
            pred_class = int(probs_cat.argmax())
            age = int(outputs["age"].item())
            # Исправлено: argmax для multiclass gender
            probs_gen = outputs["gender"].softmax(dim=-1).squeeze(0).cpu().numpy()
            gender_id = int(probs_gen.argmax())
            gender = GENDER_NAMES[gender_id]
            result = {
                "class": CLASS_NAMES[pred_class],
                "confidence": float(probs_cat[pred_class]),
                "age": age,
                "gender": gender,
            }

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    infer()
