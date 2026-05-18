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
    ensure_data,
    preprocess_image,
)


@hydra.main(
    config_path="../conf",
    config_name="config",
    version_base=None,
)
def infer(cfg: DictConfig):
    ensure_data(cfg.data.data_root, "validation")

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
