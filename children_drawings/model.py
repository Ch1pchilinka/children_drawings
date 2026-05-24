import pytorch_lightning as pl
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics.classification import Accuracy, F1Score
from torchmetrics.regression import MeanAbsoluteError
from torchvision.models import (
    EfficientNet_B3_Weights,
    efficientnet_b3,
)

from .utils import NUM_CLASSES


class MultiHeadEfficientNet(pl.LightningModule):
    def __init__(
        self,
        lr=1e-3,
        weight_decay=1e-4,
        epochs=30,
        age_loss_weight=0.01,
        freeze_below_index=300,
        pretrained=True,
    ):
        super().__init__()

        self.save_hyperparameters()

        backbone = efficientnet_b3(
            weights=EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None,
        )

        for idx, (_, param) in enumerate(backbone.named_parameters()):
            param.requires_grad = idx > freeze_below_index

        backbone.classifier = nn.Identity()

        self.backbone = backbone

        in_features = 1536

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
