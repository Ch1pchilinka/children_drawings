"""Download Children_Drawings dataset from Hugging Face.

Dataset:
https://huggingface.co/datasets/ironDong/Children_Drawings

Usage:
    uv run python scripts/download.py info
    uv run python scripts/download.py download
    uv run python scripts/download.py download --destination_path ./data
    uv run python scripts/download.py download --split train
"""

from __future__ import annotations

import os
from getpass import getpass
from pathlib import Path

import fire
from datasets import load_dataset
from huggingface_hub import dataset_info

DATASET_NAME = "ironDong/Children_Drawings"


class ChildrenDrawingsDownloader:
    """CLI for downloading Children_Drawings dataset from Hugging Face."""

    def info(self):
        """Print dataset information."""
        print("Hugging Face token required for gated dataset")
        print("Get token at: https://huggingface.co/settings/tokens")
        token = getpass("Enter your HF token: ")
        info = dataset_info(
            DATASET_NAME,
            token=token,
        )

        print(f"\nDataset: {info.id}")

        if info.description:
            print("\nDescription:")
            print(info.description[:500])

    def download(
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
            print("Hugging Face token required for gated dataset")
            print("Get token at: https://huggingface.co/settings/tokens")
            token = getpass("Enter your HF token: ")
            dataset = load_dataset(DATASET_NAME, split=split, token=token)

            split_path = destination / split
            dataset.save_to_disk(str(split_path))

            print(f"Saved split '{split}' to '{split_path}'.")
        else:
            print("Hugging Face token required for gated dataset")
            print("Get token at: https://huggingface.co/settings/tokens")
            token = getpass("Enter your HF token: ")
            dataset = load_dataset(DATASET_NAME, token=token)

            for split_name, split_data in dataset.items():
                split_path = destination / split_name

                print(f"Saving split '{split_name}'...")
                split_data.save_to_disk(str(split_path))

                print(f"Saved to '{split_path}'.")

        print("Download complete.")


if __name__ == "__main__":
    fire.Fire(ChildrenDrawingsDownloader)
