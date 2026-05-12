"""PyTorch Lightning modules for children drawings classification."""

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics import Accuracy, F1Score, MeanAbsoluteError
from utils import NUM_CLASSES


class BaselineModel(pl.LightningModule):
    """Базовая модель: ResNet-18 с замороженными весами, обучается только голова."""

    def __init__(
        self,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        epochs: int = 30,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        from torchvision.models import ResNet18_Weights, resnet18

        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        for param in backbone.parameters():
            param.requires_grad = False
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.fc = nn.Linear(512, NUM_CLASSES)
        self.criterion = nn.CrossEntropyLoss()

        # Метрики
        self.train_acc = Accuracy(task="multiclass", num_classes=NUM_CLASSES)
        self.train_f1 = F1Score(
            task="multiclass", num_classes=NUM_CLASSES, average="macro"
        )
        self.val_acc = Accuracy(task="multiclass", num_classes=NUM_CLASSES)
        self.val_f1 = F1Score(
            task="multiclass", num_classes=NUM_CLASSES, average="macro"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x).flatten(1)
        return self.fc(features)

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        logits = self(batch["image"])
        loss = self.criterion(logits, batch["category"])
        preds = logits.softmax(dim=-1)

        self.train_acc.update(preds, batch["category"])
        self.train_f1.update(preds, batch["category"])

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_end(self) -> None:
        self.log("train_acc", self.train_acc.compute(), prog_bar=True)
        self.log("train_f1", self.train_f1.compute(), prog_bar=True)
        self.train_acc.reset()
        self.train_f1.reset()

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        logits = self(batch["image"])
        loss = self.criterion(logits, batch["category"])
        preds = logits.softmax(dim=-1)

        self.val_acc.update(preds, batch["category"])
        self.val_f1.update(preds, batch["category"])

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        self.log("val_acc", self.val_acc.compute(), prog_bar=True)
        self.log("val_f1", self.val_f1.compute(), prog_bar=True)
        self.val_acc.reset()
        self.val_f1.reset()

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
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }


class MultiHeadEfficientNet(pl.LightningModule):
    """Основная модель: EfficientNet-B3 с дообучением верхних слоёв и тремя головами."""

    def __init__(
        self,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        epochs: int = 30,
        freeze_below_index: int = 300,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        from torchvision.models import EfficientNet_B3_Weights, efficientnet_b3

        backbone = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)

        # Заморозка слоёв с индексом <= freeze_below_index
        for idx, (name, param) in enumerate(backbone.named_parameters()):
            if idx <= freeze_below_index:
                param.requires_grad = False
            else:
                param.requires_grad = True

        backbone.classifier = nn.Identity()
        self.backbone = backbone
        in_features = 1536

        # Три головы
        self.head_category = nn.Linear(in_features, NUM_CLASSES)
        self.head_age = nn.Linear(in_features, 1)
        self.head_gender = nn.Linear(in_features, 2)

        # Функции потерь
        self.criterion_category = nn.CrossEntropyLoss()
        self.criterion_age = nn.MSELoss()
        self.criterion_gender = nn.CrossEntropyLoss()

        # Метрики
        self.train_cat_acc = Accuracy(task="multiclass", num_classes=NUM_CLASSES)
        self.train_cat_f1 = F1Score(
            task="multiclass", num_classes=NUM_CLASSES, average="macro"
        )
        self.train_age_mae = MeanAbsoluteError()
        # ИСПРАВЛЕНО: multiclass для gender (2 класса)
        self.train_gen_acc = Accuracy(task="multiclass", num_classes=2)

        self.val_cat_acc = Accuracy(task="multiclass", num_classes=NUM_CLASSES)
        self.val_cat_f1 = F1Score(
            task="multiclass", num_classes=NUM_CLASSES, average="macro"
        )
        self.val_age_mae = MeanAbsoluteError()
        self.val_gen_acc = Accuracy(task="multiclass", num_classes=2)

    def forward(self, x: torch.Tensor) -> dict:
        features = self.backbone(x)
        return {
            "category": self.head_category(features),
            "age": self.head_age(features).squeeze(-1),
            "gender": self.head_gender(features),
        }

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        outputs = self(batch["image"])

        loss_cat = self.criterion_category(outputs["category"], batch["category"])
        loss_age = self.criterion_age(outputs["age"], batch["age"])
        loss_gen = self.criterion_gender(outputs["gender"], batch["gender"])
        loss = loss_cat + 0.01 * loss_age + loss_gen

        # Обновляем метрики отдельно
        self.train_cat_acc.update(
            outputs["category"].softmax(dim=-1), batch["category"]
        )
        self.train_cat_f1.update(outputs["category"].softmax(dim=-1), batch["category"])
        self.train_age_mae.update(outputs["age"], batch["age"])
        # ИСПРАВЛЕНО: передаём softmax для multiclass
        self.train_gen_acc.update(outputs["gender"].softmax(dim=-1), batch["gender"])

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_cat_loss", loss_cat, on_step=False, on_epoch=True)
        self.log("train_age_loss", loss_age, on_step=False, on_epoch=True)
        self.log("train_gen_loss", loss_gen, on_step=False, on_epoch=True)

        return loss

    def on_train_epoch_end(self) -> None:
        self.log("train_cat_acc", self.train_cat_acc.compute(), prog_bar=True)
        self.log("train_cat_f1", self.train_cat_f1.compute())
        self.log("train_age_mae", self.train_age_mae.compute(), prog_bar=True)
        self.log("train_gen_acc", self.train_gen_acc.compute())

        self.train_cat_acc.reset()
        self.train_cat_f1.reset()
        self.train_age_mae.reset()
        self.train_gen_acc.reset()

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        outputs = self(batch["image"])

        loss_cat = self.criterion_category(outputs["category"], batch["category"])
        loss_age = self.criterion_age(outputs["age"], batch["age"])
        loss_gen = self.criterion_gender(outputs["gender"], batch["gender"])
        loss = loss_cat + 0.01 * loss_age + loss_gen

        # Обновляем метрики отдельно
        self.val_cat_acc.update(outputs["category"].softmax(dim=-1), batch["category"])
        self.val_cat_f1.update(outputs["category"].softmax(dim=-1), batch["category"])
        self.val_age_mae.update(outputs["age"], batch["age"])
        self.val_gen_acc.update(outputs["gender"].softmax(dim=-1), batch["gender"])

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_cat_loss", loss_cat, on_step=False, on_epoch=True)

    def on_validation_epoch_end(self) -> None:
        self.log("val_cat_acc", self.val_cat_acc.compute(), prog_bar=True)
        self.log("val_cat_f1", self.val_cat_f1.compute(), prog_bar=True)
        self.log("val_age_mae", self.val_age_mae.compute(), prog_bar=True)
        self.log("val_gen_acc", self.val_gen_acc.compute(), prog_bar=True)

        self.val_cat_acc.reset()
        self.val_cat_f1.reset()
        self.val_age_mae.reset()
        self.val_gen_acc.reset()

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
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }
