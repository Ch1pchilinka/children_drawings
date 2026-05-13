import albumentations as A
from albumentations.pytorch import ToTensorV2

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
