"""Children Drawings dataset and LightningDataModule (local saved splits).

Ожидаемая структура после выполнения download.py:
    data_root/
        train/
            dataset_info.json
            state.json
            data-*.arrow
        validation/
            dataset_info.json
            state.json
            data-*.arrow

Реальные поля датасета:
    - image: PIL Image (RGB)
    - category: str ("집", "나무", "남자사람", "여자사람")
    - age: str ("6", "7", ...)
    - gender: str ("남", "여")  -> male, female
    - id: str (игнорируем)
"""

from pathlib import Path
from typing import Optional

import albumentations as A
import numpy as np
import pytorch_lightning as pl
from datasets import load_from_disk
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from utils import build_transforms

# Маппинги из корейских строк
CATEGORY_MAP = {
    "집": 0,  # house
    "나무": 1,  # tree
    "남자사람": 2,  # man
    "여자사람": 3,  # woman
}

GENDER_MAP = {
    "남": 0,  # male
    "여": 1,  # female
}


class ChildrenDrawingsDataset(Dataset):
    """Dataset-обёртка над загруженным с диска Hugging Face датасетом.

    Args:
        hf_dataset: объект Dataset (split), загруженный через load_from_disk.
        transforms: Albumentations-пайплайн.
    """

    def __init__(self, hf_dataset, transforms: A.Compose) -> None:
        self.dataset = hf_dataset
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        example = self.dataset[idx]

        # Изображение
        image = example["image"]
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        image = image.convert("RGB")

        # Парсим метки из корейских строк
        category_str = example["category"]
        if category_str not in CATEGORY_MAP:
            raise ValueError(f"Unknown category: {category_str}")
        label = CATEGORY_MAP[category_str]

        # Возраст (хранится как строка) - ВАЖНО: преобразуем в float32
        age = np.float32(example["age"])

        # Пол
        gender_str = example["gender"]
        if gender_str not in GENDER_MAP:
            raise ValueError(f"Unknown gender: {gender_str}")
        gender = GENDER_MAP[gender_str]

        # Альбументации ожидают numpy array
        image_np = np.array(image)
        transformed = self.transforms(image=image_np)
        image_tensor = transformed["image"].float()  # float32

        # ЯВНО ПРИВОДИМ ВСЁ К float32/long
        import torch

        return {
            "image": image_tensor,
            "category": torch.tensor(label, dtype=torch.long),
            "age": torch.tensor(age, dtype=torch.float32),
            "gender": torch.tensor(gender, dtype=torch.long),
        }


class ChildrenDrawingsDataModule(pl.LightningDataModule):
    """LightningDataModule, загружающий предварительно сохранённые сплиты.

    Args:
        data_root: корневая папка, куда был скачан датасет (содержит train/ validation/).
        batch_size: размер батча.
        num_workers: число процессов DataLoader.
    """

    def __init__(
        self,
        data_root: str,
        batch_size: int = 64,
        num_workers: int = 4,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage: str | None = None) -> None:
        data_root = Path(self.hparams.data_root)

        if stage in ("fit", None):
            train_path = data_root / "train"
            val_path = data_root / "validation"

            if not train_path.exists():
                raise FileNotFoundError(
                    f"Training split not found at {train_path}. Run download.py first."
                )
            if not val_path.exists():
                raise FileNotFoundError(
                    f"Validation split not found at {val_path}. Run download.py first."
                )

            print(f"Loading training data from {train_path}")
            self.train_ds = ChildrenDrawingsDataset(
                load_from_disk(str(train_path)),
                build_transforms("train"),
            )

            print(f"Loading validation data from {val_path}")
            self.val_ds = ChildrenDrawingsDataset(
                load_from_disk(str(val_path)),
                build_transforms("val"),
            )

        if stage in ("test", None):
            test_path = data_root / "test"
            if test_path.exists():
                print(f"Loading test data from {test_path}")
                self.test_ds = ChildrenDrawingsDataset(
                    load_from_disk(str(test_path)),
                    build_transforms("val"),
                )
            else:
                self.test_ds = None

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        if self.test_ds is None:
            raise RuntimeError("Test split not found.")
        return DataLoader(
            self.test_ds,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
        )
