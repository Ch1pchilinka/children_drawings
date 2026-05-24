import glob
import os
from pathlib import Path

import hydra
import numpy as np
import onnx
import onnxruntime as ort
import torch
from omegaconf import DictConfig

from .model import MultiHeadEfficientNet
from .utils import ensure_data, load_images_as_tensor_batch, resolve_repo_path

OUTPUT_NAMES = ("category", "age", "gender")
CLASSIFICATION_OUTPUTS = ("category", "gender")


class _OnnxMultiHeadWrapper(torch.nn.Module):
    """Return model heads as a stable tuple for ONNX export."""

    def __init__(self, model: MultiHeadEfficientNet):
        super().__init__()
        self.model = model

    def forward(self, x):
        outputs = self.model(x)
        return tuple(outputs[name] for name in OUTPUT_NAMES)


def _to_numpy_outputs(outputs: tuple[torch.Tensor, ...]) -> dict[str, np.ndarray]:
    return {
        name: output.detach().cpu().numpy()
        for name, output in zip(OUTPUT_NAMES, outputs, strict=True)
    }


def _output_parity_cfg(cfg: DictConfig, output_name: str) -> DictConfig | dict:
    parity_cfg = cfg.export.get("parity", {})
    return parity_cfg.get(output_name, {})


def _assert_classification_parity(
    output_name: str,
    torch_output: np.ndarray,
    ort_output: np.ndarray,
    cfg: DictConfig,
) -> float:
    if torch_output.shape != ort_output.shape:
        raise AssertionError(
            f"ONNX {output_name} parity check failed: "
            f"shape mismatch, PyTorch={torch_output.shape}, ONNX={ort_output.shape}"
        )

    parity_cfg = _output_parity_cfg(cfg, output_name)
    max_probability_diff = float(parity_cfg.get("max_probability_diff", 1e-3))
    torch_probs = torch.softmax(torch.from_numpy(torch_output), dim=-1).numpy()
    ort_probs = torch.softmax(torch.from_numpy(ort_output), dim=-1).numpy()
    probability_diff = np.abs(torch_probs - ort_probs)
    max_prob_diff = float(np.max(probability_diff))

    if max_prob_diff > max_probability_diff:
        raise AssertionError(
            f"ONNX probability parity check failed for output '{output_name}': "
            f"max_prob_diff={max_prob_diff:.6e} > "
            f"{max_probability_diff:.6e}"
        )

    check_argmax = bool(parity_cfg.get("check_argmax", True))
    if check_argmax:
        torch_pred = torch_output.argmax(axis=-1)
        ort_pred = ort_output.argmax(axis=-1)
        if not np.array_equal(torch_pred, ort_pred):
            mismatch_count = int(np.count_nonzero(torch_pred != ort_pred))
            raise AssertionError(
                f"ONNX class parity check failed for output '{output_name}': "
                f"{mismatch_count}/{torch_pred.size} argmax values differ"
            )

    return max_prob_diff


def _assert_age_parity(
    torch_output: np.ndarray,
    ort_output: np.ndarray,
    cfg: DictConfig,
) -> float:
    if torch_output.shape != ort_output.shape:
        raise AssertionError(
            "ONNX age parity check failed: "
            f"shape mismatch, PyTorch={torch_output.shape}, ONNX={ort_output.shape}"
        )

    parity_cfg = _output_parity_cfg(cfg, "age")
    max_abs_error = float(parity_cfg.get("max_abs_error", 1e-2))
    abs_error = np.abs(torch_output - ort_output)
    max_abs_diff = float(np.max(abs_error))

    if max_abs_diff > max_abs_error:
        mean_abs_diff = float(np.mean(abs_error))
        raise AssertionError(
            "ONNX age parity check failed: "
            f"max_abs_diff={max_abs_diff:.6e} > {max_abs_error:.6e}, "
            f"mean_abs_diff={mean_abs_diff:.6e}"
        )

    return max_abs_diff


def _assert_output_parity(
    output_name: str,
    torch_output: np.ndarray,
    ort_output: np.ndarray,
    cfg: DictConfig,
) -> float:
    if output_name in CLASSIFICATION_OUTPUTS:
        return _assert_classification_parity(
            output_name,
            torch_output,
            ort_output,
            cfg,
        )
    if output_name == "age":
        return _assert_age_parity(torch_output, ort_output, cfg)
    raise ValueError(f"Unknown ONNX output: {output_name}")


def export_to_onnx(cfg: DictConfig) -> tuple[Path, int, float]:
    """Export the configured children_drawings checkpoint to ONNX and validate parity."""
    ensure_data(cfg.data.data_root, "batch")

    device = torch.device("cpu")
    checkpoint_path = resolve_repo_path(cfg.inference.checkpoint)
    model = MultiHeadEfficientNet.load_from_checkpoint(
        str(checkpoint_path),
        map_location=device,
    ).to(device)
    model.eval()

    export_model = _OnnxMultiHeadWrapper(model).to(device)
    export_model.eval()

    image_paths = sorted(
        Path(p)
        for p in glob.glob(
            os.path.join(str(resolve_repo_path(cfg.export.batch_path)), "*.[pj][np]g")
        )
    )
    batch, _ = load_images_as_tensor_batch(image_paths)
    batch = batch.to(device)

    onnx_path = resolve_repo_path(cfg.export.onnx_path)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

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
            export_model,
            (batch,),
            str(onnx_path),
            input_names=["input"],
            output_names=list(OUTPUT_NAMES),
            dynamic_axes=dynamic_axes,
            dynamo=True,
            opset_version=int(cfg.model.opset),
            do_constant_folding=True,
        )
        torch_outputs = _to_numpy_outputs(export_model(batch))

    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)

    ort_session = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )
    ort_outputs = dict(
        zip(
            OUTPUT_NAMES,
            ort_session.run(
                list(OUTPUT_NAMES), {"input": batch.detach().cpu().numpy()}
            ),
            strict=True,
        )
    )

    return (
        onnx_path,
        len(onnx_model.graph.node),
        max(
            _assert_output_parity(
                output_name,
                torch_outputs[output_name],
                ort_outputs[output_name],
                cfg,
            )
            for output_name in OUTPUT_NAMES
        ),
    )


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    onnx_path, num_nodes, max_parity_diff = export_to_onnx(cfg)
    print(f"Saved ONNX model to: {onnx_path}")
    print(f"ONNX graph nodes: {num_nodes}")
    print(f"Max parity diff vs PyTorch: {max_parity_diff:.6e}")


if __name__ == "__main__":
    main()
