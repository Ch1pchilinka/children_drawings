import torch

from children_drawings.model import (
    MultiHeadEfficientNet,
    MultiHeadResNet18Baseline,
)


def test_model_forward_smoke():
    model = MultiHeadEfficientNet(
        pretrained=False,
        freeze_below_index=-1,
    )
    model.eval()

    with torch.no_grad():
        outputs = model(torch.randn(2, 3, 64, 64))

    assert outputs["category"].shape == (2, 4)
    assert outputs["age"].shape == (2,)
    assert outputs["gender"].shape == (2, 2)


def test_resnet18_baseline_forward_smoke():
    model = MultiHeadResNet18Baseline(pretrained=False)
    model.eval()

    with torch.no_grad():
        outputs = model(torch.randn(2, 3, 64, 64))

    assert outputs["category"].shape == (2, 4)
    assert outputs["age"].shape == (2,)
    assert outputs["gender"].shape == (2, 2)
