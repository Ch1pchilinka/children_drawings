from __future__ import annotations

import asyncio
import base64
import os
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path

import numpy as np
import requests
import tritonclient.http as triton_http
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image
from pydantic import BaseModel
from tritonclient.utils import InferenceServerException

from .constants import OUTPUT_NAMES
from .prediction import decode_numpy_outputs
from .preprocessing import preprocess_pil_image

DEFAULT_TRITON_URL = "localhost:8100"
DEFAULT_TRITON_MODEL_NAME = "children_drawings"
WEB_ROOT = Path(__file__).resolve().parent / "web"
INDEX_HTML = WEB_ROOT / "index.html"

app = FastAPI(title="Children Drawings Triton Inference API")
_TRITON_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="triton-http")


class Base64ImageRequest(BaseModel):
    image_base64: str


def _normalize_triton_url(url: str) -> str:
    cleaned = url.strip().rstrip("/")
    cleaned = cleaned.removeprefix("http://")
    cleaned = cleaned.removeprefix("https://")
    return cleaned


def _triton_url() -> str:
    return _normalize_triton_url(os.getenv("TRITON_URL", DEFAULT_TRITON_URL))


def _triton_model_name() -> str:
    return os.getenv("TRITON_MODEL_NAME", DEFAULT_TRITON_MODEL_NAME)


def _new_triton_client() -> triton_http.InferenceServerClient:
    return triton_http.InferenceServerClient(
        url=_triton_url(),
        verbose=False,
    )


def _status_message(prediction: dict[str, float | int | str]) -> str:
    class_name = str(prediction["class"])
    age = int(prediction["age"])
    gender = str(prediction["gender"])

    class_messages = {
        "house": "Отличный выбор сюжета: рисунок похож на дом.",
        "tree": "Рисунок похож на дерево, линии выглядят уверенно.",
        "man": "Похоже на мужскую фигуру, модель распознала основные признаки.",
        "woman": "Похоже на женскую фигуру, модель распознала основные признаки.",
    }
    gender_messages = {
        "male": "Пол по рисунку: мужской.",
        "female": "Пол по рисунку: женский.",
    }

    class_message = class_messages.get(class_name, "Класс рисунка определён.")
    gender_message = gender_messages.get(gender, "Пол определён.")
    return f"{class_message} Прогноз возраста: {age}. {gender_message}"


def _prepare_batch(images: list[Image.Image]) -> np.ndarray:
    if not images:
        raise ValueError("No images were provided.")

    batches = [preprocess_pil_image(image) for image in images]
    return np.concatenate(batches, axis=0).astype(np.float32, copy=False)


def _predict_batch_sync(
    images: list[Image.Image],
) -> list[dict[str, float | int | str]]:
    batch = _prepare_batch(images)
    infer_input = triton_http.InferInput("input", list(batch.shape), "FP32")
    infer_input.set_data_from_numpy(batch, binary_data=True)

    outputs = [triton_http.InferRequestedOutput(name) for name in OUTPUT_NAMES]
    client = _new_triton_client()
    result = client.infer(
        model_name=_triton_model_name(),
        inputs=[infer_input],
        outputs=outputs,
    )

    raw_outputs = [result.as_numpy(name) for name in OUTPUT_NAMES]
    if any(output is None for output in raw_outputs):
        raise RuntimeError("Triton returned incomplete outputs.")

    predictions = decode_numpy_outputs(*raw_outputs)
    for prediction in predictions:
        prediction["message"] = _status_message(prediction)
    return predictions


async def _predict_batch(
    images: list[Image.Image],
) -> list[dict[str, float | int | str]]:
    # Triton HTTP client uses gevent internals; keep all calls on one dedicated thread.
    loop = asyncio.get_running_loop()
    for attempt_index in range(2):
        try:
            return await loop.run_in_executor(
                _TRITON_EXECUTOR,
                _predict_batch_sync,
                images,
            )
        except Exception as exc:
            error_text = str(exc)
            retryable = (
                "Cannot switch to a different thread" in error_text
                or "This operation would block forever" in error_text
            )
            is_last_attempt = attempt_index == 1
            if is_last_attempt or not retryable:
                raise
            await asyncio.sleep(0.05)

    raise RuntimeError("Triton inference failed after retry.")


def _decode_base64_image(image_base64: str) -> Image.Image:
    encoded = image_base64
    if "," in encoded:
        encoded = encoded.split(",", maxsplit=1)[1]
    image_bytes = base64.b64decode(encoded)
    image = Image.open(BytesIO(image_bytes))
    image.load()
    return image


def _safe_open_image(content: bytes) -> Image.Image:
    image = Image.open(BytesIO(content))
    image.load()
    return image


@app.get("/", response_class=FileResponse)
def index():
    if not INDEX_HTML.exists():
        raise HTTPException(status_code=500, detail=f"UI page not found: {INDEX_HTML}")
    return FileResponse(str(INDEX_HTML))


def _health_sync():
    server_live = False
    model_ready = False
    error: str | None = None

    try:
        triton_http_url = f"http://{_triton_url()}"
        live_response = requests.get(
            f"{triton_http_url}/v2/health/live",
            timeout=1.0,
        )
        model_response = requests.get(
            f"{triton_http_url}/v2/models/{_triton_model_name()}/ready",
            timeout=1.0,
        )
        server_live = live_response.status_code == 200
        model_ready = model_response.status_code == 200
    except Exception as exc:
        error = str(exc)

    return {
        "status": "ok" if server_live and model_ready else "degraded",
        "triton_url": _triton_url(),
        "model_name": _triton_model_name(),
        "server_live": server_live,
        "model_ready": model_ready,
        "error": error,
    }


@app.get("/health")
async def health():
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_TRITON_EXECUTOR, _health_sync)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        content = await file.read()
        image = _safe_open_image(content)
        prediction = (await _predict_batch([image]))[0]
    except InferenceServerException as exc:
        raise HTTPException(
            status_code=503, detail=f"Triton unavailable: {exc}"
        ) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}") from exc

    prediction["image"] = file.filename or "uploaded"
    return prediction


@app.post("/predict/batch")
async def predict_batch(files: list[UploadFile] = File(...)):
    try:
        opened_images: list[Image.Image] = []
        image_names: list[str] = []
        for file in files:
            content = await file.read()
            opened_images.append(_safe_open_image(content))
            image_names.append(file.filename or "uploaded")
        predictions = await _predict_batch(opened_images)
    except InferenceServerException as exc:
        raise HTTPException(
            status_code=503, detail=f"Triton unavailable: {exc}"
        ) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid images: {exc}") from exc

    for prediction, image_name in zip(predictions, image_names, strict=True):
        prediction["image"] = image_name
    return {"count": len(predictions), "predictions": predictions}


@app.post("/predict/base64")
async def predict_base64(payload: Base64ImageRequest):
    try:
        image = _decode_base64_image(payload.image_base64)
        return (await _predict_batch([image]))[0]
    except InferenceServerException as exc:
        raise HTTPException(
            status_code=503, detail=f"Triton unavailable: {exc}"
        ) from exc
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
