from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .constants import CLASS_NAMES, GENDER_NAMES

if TYPE_CHECKING:
    import torch


def decode_numpy_outputs(
    category_logits: np.ndarray,
    age_values: np.ndarray,
    gender_logits: np.ndarray,
) -> list[dict[str, float | int | str]]:
    category_probs = _softmax(category_logits, axis=-1)
    gender_probs = _softmax(gender_logits, axis=-1)
    category_ids = category_probs.argmax(axis=-1)
    gender_ids = gender_probs.argmax(axis=-1)
    ages = np.rint(age_values).astype(int)

    return [
        {
            "class": CLASS_NAMES[int(category_id)],
            "confidence": float(category_probs[index, category_id]),
            "age": int(ages[index]),
            "gender": GENDER_NAMES[int(gender_ids[index])],
            "gender_confidence": float(gender_probs[index, gender_ids[index]]),
        }
        for index, category_id in enumerate(category_ids)
    ]


def decode_torch_outputs(outputs: dict[str, "torch.Tensor"]):
    return decode_numpy_outputs(
        outputs["category"].detach().cpu().numpy(),
        outputs["age"].detach().cpu().numpy(),
        outputs["gender"].detach().cpu().numpy(),
    )


def _softmax(values: np.ndarray, axis: int) -> np.ndarray:
    shifted = values - np.max(values, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)
