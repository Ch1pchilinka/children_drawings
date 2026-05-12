from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from datasets import load_from_disk
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from utils import (
    CATEGORY_MAP,
    GENDER_MAP,
    TRAIN_TRANSFORMS,
    VAL_TRANSFORMS,
)


class ChildrenDrawingsDataset(Dataset):
    def __init__(self, dataset, transforms):
        self.dataset = dataset
        self.transforms = transforms

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image = sample["image"]
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        image = np.array(image.convert("RGB"))
        image = self.transforms(image=image)["image"]
        return {
            "image": image,
            "category": torch.tensor(
                CATEGORY_MAP[sample["category"]],
                dtype=torch.long,
            ),
            "age": torch.tensor(
                float(sample["age"]),
                dtype=torch.float32,
            ),
            "gender": torch.tensor(
                GENDER_MAP[sample["gender"]],
                dtype=torch.long,
            ),
        }


class ChildrenDrawingsDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_root: str,
        batch_size: int = 16,
        num_workers: int = 1,
    ):
        super().__init__()
        self.data_root = Path(data_root)
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.train_ds = ChildrenDrawingsDataset(
            load_from_disk(str(self.data_root / "train")),
            TRAIN_TRANSFORMS,
        )
        self.val_ds = ChildrenDrawingsDataset(
            load_from_disk(str(self.data_root / "validation")),
            VAL_TRANSFORMS,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
