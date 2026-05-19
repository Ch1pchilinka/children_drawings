import numpy as np
import onnx
import onnxruntime as ort
import torch
from omegaconf import OmegaConf

from children_drawings.exporting import OUTPUT_NAMES, _assert_output_parity


class TinyMultiHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.category = torch.nn.Linear(3, 4)
        self.age = torch.nn.Linear(3, 1)
        self.gender = torch.nn.Linear(3, 2)

    def forward(self, inputs):
        return (
            self.category(inputs),
            self.age(inputs).squeeze(1),
            self.gender(inputs),
        )


def test_export_parity_helpers():
    cfg = OmegaConf.create(
        {
            "export": {
                "parity": {
                    "category": {"max_probability_diff": 1e-3, "check_argmax": True},
                    "age": {"max_abs_error": 1e-3},
                    "gender": {"max_probability_diff": 1e-3, "check_argmax": True},
                }
            }
        }
    )
    logits = np.array([[4.0, 1.0, 0.0, -1.0]], dtype=np.float32)
    gender = np.array([[0.1, 0.9]], dtype=np.float32)
    age = np.array([9.0], dtype=np.float32)

    assert _assert_output_parity("category", logits, logits.copy(), cfg) == 0.0
    assert _assert_output_parity("gender", gender, gender.copy(), cfg) == 0.0
    assert _assert_output_parity("age", age, age.copy(), cfg) == 0.0


def test_onnxruntime_inference_smoke(tmp_path):
    model = TinyMultiHead().eval()
    inputs = torch.randn(2, 3)
    onnx_path = tmp_path / "tiny.onnx"

    with torch.no_grad():
        torch_outputs = model(inputs)
        torch.onnx.export(
            model,
            (inputs,),
            str(onnx_path),
            input_names=["input"],
            output_names=list(OUTPUT_NAMES),
            opset_version=18,
        )

    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_outputs = session.run(list(OUTPUT_NAMES), {"input": inputs.numpy()})

    for expected, actual in zip(torch_outputs, ort_outputs, strict=True):
        np.testing.assert_allclose(
            expected.detach().numpy(), actual, rtol=1e-5, atol=1e-5
        )
