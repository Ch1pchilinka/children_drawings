import glob
import os
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from model import MultiHeadEfficientNet
from omegaconf import DictConfig
from utils import ensure_data, load_images_as_tensor_batch


def export_cityscapes_to_onnx(cfg: DictConfig) -> tuple[Path, int, float]:
    """Export the configured children_drawings checkpoint to ONNX and validate parity."""
    ensure_data(cfg.data.data_root, "batch")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiHeadEfficientNet.load_from_checkpoint(
        cfg.inference.checkpoint,
        map_location=device,
    )
    batch, _ = load_images_as_tensor_batch(
        glob.glob(os.path.join(cfg.export.batch_path, "*.[pj][np]g"))
    )
    batch = batch.to(device)
    onnx_path = cfg.export.onnx_path
    dynamic_axes = None
    if cfg.model.dynamic_axes:
        dynamic_axes = {
            "input": {0: "batch"},
            "category": {0: "batch"},
            "age": {0: "batch"},
            "gender": {0: "batch"},
        }

    with torch.inference_mode():
        torch.onnx.export(
            model,
            batch,
            str(onnx_path),
            input_names=["input"],
            output_names=["category", "age", "gender"],
            dynamic_axes=dynamic_axes,
            opset_version=int(cfg.model.opset),
            do_constant_folding=True,
        )
        torch_output = model(batch).detach().cpu().numpy()

    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)

    ort_session = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )
    ort_output = ort_session.run(None, {"input": batch.detach().cpu().numpy()})[0]
    max_abs_diff = float(np.max(np.abs(torch_output - ort_output)))

    torch.testing.assert_close(
        torch.from_numpy(ort_output),
        torch.from_numpy(torch_output),
        atol=float(cfg.export.parity_atol),
        rtol=float(cfg.export.parity_rtol),
    )
    return onnx_path, len(onnx_model.graph.node), max_abs_diff
