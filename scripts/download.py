"""Download Children_Drawings dataset from Hugging Face.

Dataset:
https://huggingface.co/datasets/ironDong/Children_Drawings

Usage:
    uv run python scripts/download.py info
    uv run python scripts/download.py download_data
    uv run python scripts/download.py download_data --destination_path ./data
    uv run python scripts/download.py download_data --split train
"""

from __future__ import annotations

from pathlib import Path

import fire
from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import dataset_info
from omegaconf import OmegaConf

DATASET_NAME = "ironDong/Children_Drawings"


class ChildrenDrawingsDownloader:
    def __init__(self):
        load_dotenv()
        self.cfg = OmegaConf.load("./conf/secret/default.yaml")

    def info(self):
        info = dataset_info(
            DATASET_NAME,
            token=self.cfg.token,
        )

        print(f"\nDataset: {info.id}")

        if info.description:
            print("\nDescription:")
            print(info.description[:500])

    def download_data(
        self,
        destination_path: str = "./data",
        split: str | None = None,
    ):
        """Download dataset to local disk.

        Args:
            destination_path: Directory where dataset will be stored.
            split: Optional dataset split (e.g. 'train').
                   If None, downloads all splits.
        """
        destination = Path(destination_path)
        destination.mkdir(parents=True, exist_ok=True)

        print(f"Downloading dataset '{DATASET_NAME}'...")

        if split is not None:
            dataset = load_dataset(DATASET_NAME, split=split, token=self.cfg.token)

            split_path = destination / split
            dataset.save_to_disk(str(split_path))

            print(f"Saved split '{split}' to '{split_path}'.")
        else:
            dataset = load_dataset(DATASET_NAME, token=self.cfg.token)

            for split_name, split_data in dataset.items():
                split_path = destination / split_name

                print(f"Saving split '{split_name}'...")
                split_data.save_to_disk(str(split_path))

                print(f"Saved to '{split_path}'.")

        print("Download complete.")


if __name__ == "__main__":
    fire.Fire(ChildrenDrawingsDownloader)
