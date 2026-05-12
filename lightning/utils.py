"""Shared utility functions."""

import albumentations as A
from albumentations.pytorch import ToTensorV2

# Константы
NUM_CLASSES = 4  # дом, дерево, мужчина, женщина
CLASS_NAMES = ["house", "tree", "man", "woman"]
IMAGE_SIZE = 300
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def build_transforms(split: str) -> A.Compose:
    """Строит пайплайн аугментаций для заданного сплита."""
    shared = [A.Resize(IMAGE_SIZE, IMAGE_SIZE)]
    augment = (
        [
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Rotate(limit=15, p=0.3),
        ]
        if split == "train"
        else []
    )
    normalize = [
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ]
    return A.Compose(shared + augment + normalize)
