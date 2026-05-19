from __future__ import annotations

import base64
import os
from functools import lru_cache
from io import BytesIO
from pathlib import Path

import numpy as np
import onnxruntime as ort
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel

from .prediction import decode_numpy_outputs
from .utils import preprocess_pil_image, resolve_repo_path

DEFAULT_ONNX_PATH = "artifacts/onnx_models/children_drawings.onnx"

app = FastAPI(title="Children Drawings Inference API")


class Base64ImageRequest(BaseModel):
    image_base64: str


def _onnx_model_path() -> Path:
    return resolve_repo_path(
        os.getenv("CHILDREN_DRAWINGS_ONNX_PATH", DEFAULT_ONNX_PATH)
    )


@lru_cache(maxsize=1)
def _get_session() -> ort.InferenceSession:
    model_path = _onnx_model_path()
    if not model_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {model_path}")
    return ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])


def _predict_image(image: Image.Image) -> dict[str, float | int | str]:
    tensor = preprocess_pil_image(image).numpy().astype(np.float32)
    session = _get_session()
    output_names = ["category", "age", "gender"]
    outputs = dict(
        zip(
            output_names,
            session.run(output_names, {"input": tensor}),
            strict=True,
        )
    )
    return decode_numpy_outputs(
        outputs["category"],
        outputs["age"],
        outputs["gender"],
    )[0]


@app.get("/health")
def health():
    model_path = _onnx_model_path()
    return {
        "status": "ok",
        "model_path": str(model_path),
        "model_exists": model_path.exists(),
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        content = await file.read()
        image = Image.open(BytesIO(content))
        prediction = _predict_image(image)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}") from exc

    prediction["image"] = file.filename or "uploaded"
    return prediction


@app.post("/predict/base64")
def predict_base64(payload: Base64ImageRequest):
    try:
        image_bytes = base64.b64decode(payload.image_base64)
        image = Image.open(BytesIO(image_bytes))
        return _predict_image(image)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail=f"Invalid image payload: {exc}"
        ) from exc


def main():
    uvicorn.run(
        "children_drawings.api:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=os.getenv("RELOAD", "false").lower() == "true",
    )


if __name__ == "__main__":
    main()
