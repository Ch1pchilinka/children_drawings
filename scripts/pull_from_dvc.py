"""Utility CLI for pulling datasets and TensorRT model artifacts from DVC."""

from __future__ import annotations

import fire

from children_drawings.utils import ensure_data, ensure_onnx_artifacts


class DvcPullCLI:
    def data(self, split: str = "train"):
        """Pull one dataset split tracked by DVC: train / validation / batch."""
        ensure_data(data_root="data", mode=split)

    def onnx(self):
        """Pull ONNX artifacts from DVC model remote."""
        ensure_onnx_artifacts()


if __name__ == "__main__":
    fire.Fire(DvcPullCLI)
