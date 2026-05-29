import pytorch_lightning as pl
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics.classification import Accuracy, F1Score
from torchmetrics.regression import MeanAbsoluteError
from torchvision.models import (
    EfficientNet_B3_Weights,
    ResNet18_Weights,
    efficientnet_b3,
    resnet18,
)

from .constants import NUM_CLASSES

EFFICIENTNET_B3_ARCH = "efficientnet_b3"
RESNET18_BASELINE_ARCH = "resnet18_baseline"
SUPPORTED_MODEL_ARCHITECTURES = (
    EFFICIENTNET_B3_ARCH,
    RESNET18_BASELINE_ARCH,
)


class _BaseMultiHeadModel(pl.LightningModule):
    def __init__(
        self,
        backbone: nn.Module,
        in_features: int,
        lr=1e-3,
        weight_decay=1e-4,
        epochs=30,
        age_loss_weight=0.01,
        architecture: str = EFFICIENTNET_B3_ARCH,
    ):
        super().__init__()

        self.save_hyperparameters(
            ignore=["backbone"],
        )

        self.backbone = backbone

        self.head_category = nn.Linear(
            in_features,
            NUM_CLASSES,
        )

        self.head_age = nn.Linear(
            in_features,
            1,
        )

        self.head_gender = nn.Linear(
            in_features,
            2,
        )

        self.ce = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()

        self.train_category_acc = Accuracy(
            task="multiclass",
            num_classes=NUM_CLASSES,
        )

        self.val_category_acc = Accuracy(
            task="multiclass",
            num_classes=NUM_CLASSES,
        )

        self.train_category_f1 = F1Score(
            task="multiclass",
            num_classes=NUM_CLASSES,
            average="macro",
        )

        self.val_category_f1 = F1Score(
            task="multiclass",
            num_classes=NUM_CLASSES,
            average="macro",
        )

        self.train_mae = MeanAbsoluteError()
        self.val_mae = MeanAbsoluteError()

        self.train_gender_acc = Accuracy(
            task="multiclass",
            num_classes=2,
        )

        self.val_gender_acc = Accuracy(
            task="multiclass",
            num_classes=2,
        )

        self.train_gender_f1 = F1Score(
            task="multiclass",
            num_classes=2,
            average="macro",
        )

        self.val_gender_f1 = F1Score(
            task="multiclass",
            num_classes=2,
            average="macro",
        )

    def forward(self, x):

        features = self.backbone(x)

        return {
            "category": self.head_category(features),
            "age": self.head_age(features).squeeze(1),
            "gender": self.head_gender(features),
        }

    def shared_step(self, batch, stage):

        outputs = self(batch["image"])

        loss_cat = self.ce(
            outputs["category"],
            batch["category"],
        )

        loss_age = self.mse(
            outputs["age"],
            batch["age"],
        )

        loss_gender = self.ce(
            outputs["gender"],
            batch["gender"],
        )

        loss = loss_cat + self.hparams.age_loss_weight * loss_age + loss_gender

        category_acc_metric = (
            self.train_category_acc if stage == "train" else self.val_category_acc
        )

        category_f1_metric = (
            self.train_category_f1 if stage == "train" else self.val_category_f1
        )

        mae_metric = self.train_mae if stage == "train" else self.val_mae

        gender_acc_metric = (
            self.train_gender_acc if stage == "train" else self.val_gender_acc
        )

        gender_f1_metric = (
            self.train_gender_f1 if stage == "train" else self.val_gender_f1
        )

        self.log_dict(
            {
                f"{stage}_loss": loss,
                f"{stage}_category_loss": loss_cat,
                f"{stage}_age_loss": loss_age,
                f"{stage}_gender_loss": loss_gender,
                f"{stage}_category_acc": category_acc_metric(
                    outputs["category"],
                    batch["category"],
                ),
                f"{stage}_category_f1": category_f1_metric(
                    outputs["category"],
                    batch["category"],
                ),
                f"{stage}_age_mae": mae_metric(
                    outputs["age"],
                    batch["age"],
                ),
                f"{stage}_gender_acc": gender_acc_metric(
                    outputs["gender"],
                    batch["gender"],
                ),
                f"{stage}_gender_f1": gender_f1_metric(
                    outputs["gender"],
                    batch["gender"],
                ),
            },
            prog_bar=True,
            on_epoch=True,
            on_step=stage == "train",
        )

        return loss

    def training_step(self, batch, batch_idx):

        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):

        return self.shared_step(batch, "val")

    def configure_optimizers(self):

        optimizer = AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.epochs,
            eta_min=self.hparams.lr * 0.01,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }


class MultiHeadEfficientNet(_BaseMultiHeadModel):
    def __init__(
        self,
        lr=1e-3,
        weight_decay=1e-4,
        epochs=30,
        age_loss_weight=0.01,
        freeze_below_index=300,
        pretrained=True,
        architecture: str = EFFICIENTNET_B3_ARCH,
    ):
        if architecture != EFFICIENTNET_B3_ARCH:
            raise ValueError(
                f"Unexpected architecture for MultiHeadEfficientNet: '{architecture}'"
            )

        backbone = efficientnet_b3(
            weights=EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None,
        )

        for idx, (_, param) in enumerate(backbone.named_parameters()):
            param.requires_grad = idx > freeze_below_index

        backbone.classifier = nn.Identity()

        super().__init__(
            backbone=backbone,
            in_features=1536,
            lr=lr,
            weight_decay=weight_decay,
            epochs=epochs,
            age_loss_weight=age_loss_weight,
            architecture=architecture,
        )
        self.save_hyperparameters(
            {
                "freeze_below_index": freeze_below_index,
                "pretrained": pretrained,
            }
        )


class MultiHeadResNet18Baseline(_BaseMultiHeadModel):
    def __init__(
        self,
        lr=1e-3,
        weight_decay=1e-4,
        epochs=30,
        age_loss_weight=0.01,
        pretrained=True,
        architecture: str = RESNET18_BASELINE_ARCH,
    ):
        if architecture != RESNET18_BASELINE_ARCH:
            raise ValueError(
                f"Unexpected architecture for MultiHeadResNet18Baseline: "
                f"'{architecture}'"
            )

        backbone = resnet18(
            weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None,
        )
        for param in backbone.parameters():
            param.requires_grad = False
        backbone.fc = nn.Identity()

        super().__init__(
            backbone=backbone,
            in_features=512,
            lr=lr,
            weight_decay=weight_decay,
            epochs=epochs,
            age_loss_weight=age_loss_weight,
            architecture=architecture,
        )
        self.save_hyperparameters({"pretrained": pretrained})


def build_model(
    architecture: str,
    lr=1e-3,
    weight_decay=1e-4,
    epochs=30,
    age_loss_weight=0.01,
    freeze_below_index=300,
    pretrained=True,
) -> pl.LightningModule:
    if architecture == EFFICIENTNET_B3_ARCH:
        return MultiHeadEfficientNet(
            lr=lr,
            weight_decay=weight_decay,
            epochs=epochs,
            age_loss_weight=age_loss_weight,
            freeze_below_index=freeze_below_index,
            pretrained=pretrained,
            architecture=architecture,
        )

    if architecture == RESNET18_BASELINE_ARCH:
        return MultiHeadResNet18Baseline(
            lr=lr,
            weight_decay=weight_decay,
            epochs=epochs,
            age_loss_weight=age_loss_weight,
            pretrained=pretrained,
            architecture=architecture,
        )

    supported = ", ".join(SUPPORTED_MODEL_ARCHITECTURES)
    raise ValueError(f"Unknown architecture '{architecture}'. Supported: {supported}")


def load_model_from_checkpoint(
    checkpoint_path: str,
    architecture: str,
    map_location="cpu",
) -> pl.LightningModule:
    if architecture == EFFICIENTNET_B3_ARCH:
        return MultiHeadEfficientNet.load_from_checkpoint(
            checkpoint_path,
            map_location=map_location,
            architecture=architecture,
        )

    if architecture == RESNET18_BASELINE_ARCH:
        return MultiHeadResNet18Baseline.load_from_checkpoint(
            checkpoint_path,
            map_location=map_location,
            architecture=architecture,
        )

    supported = ", ".join(SUPPORTED_MODEL_ARCHITECTURES)
    raise ValueError(f"Unknown architecture '{architecture}'. Supported: {supported}")
