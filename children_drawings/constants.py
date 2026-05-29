"""Shared constants for labels, preprocessing and model IO."""

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

IMAGE_PATTERNS = ("*.jpg", "*.jpeg", "*.png")
OUTPUT_NAMES = ("category", "age", "gender")
