import glob
import json
import os
from pathlib import Path

import hydra
import numpy as np
import torch
from dvc.exceptions import DvcException
from dvc.repo import Repo
from model import MultiHeadEfficientNet
from omegaconf import DictConfig
from PIL import Image
from utils import (
    CLASS_NAMES,
    GENDER_NAMES,
    IMAGE_SIZE,
    MEAN,
    STD,
)


def preprocess_image(path):

    image = Image.open(path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))

    image = np.array(image) / 255.0

    image = (image - np.array(MEAN)) / np.array(STD)

    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()

    return tensor


def ensure_data(data_root: str):
    """Проверяет наличие данных и при необходимости выполняет dvc pull."""
    data_path = Path(data_root, "validation")
    if not data_path.exists() or not any(data_path.iterdir()):
        print("Данные не найдены. Загружаем из облака...")
        try:
            Repo(".").pull()
            print("Данные успешно загружены.")
        except DvcException as e:
            raise RuntimeError(f"Ошибка загрузки данных через DVC: {e}")
    else:
        print("Данные уже существуют.")


@hydra.main(
    config_path="../conf",
    config_name="config",
    version_base=None,
)
def infer(cfg: DictConfig):
    ensure_data(cfg.data.data_root)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MultiHeadEfficientNet.load_from_checkpoint(
        cfg.inference.checkpoint,
        map_location=device,
    )

    model.eval()

    for image in glob.glob(os.path.join(cfg.inference.images, "*.[pj][np]g")):
        image = preprocess_image(image).to(device)

        with torch.no_grad():
            outputs = model(image)

            probs = outputs["category"].softmax(-1)

            category_id = probs.argmax().item()

            gender_id = outputs["gender"].softmax(-1).argmax().item()

            result = {
                "class": CLASS_NAMES[category_id],
                "confidence": probs[0, category_id].item(),
                "age": int(outputs["age"].item()),
                "gender": GENDER_NAMES[gender_id],
            }

        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    infer()
