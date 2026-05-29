import albumentations as A
from albumentations.pytorch import ToTensorV2

from .constants import IMAGE_SIZE, MEAN, STD

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
