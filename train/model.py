from __future__ import annotations

from typing import Any

import lightning as L
import torch
from transformers import AutoModelForSequenceClassification


class BertTextClassifier(L.LightningModule):
    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        learning_rate: float = 3e-5,
        num_labels: int = 2,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
        )
        self._val_predictions: list[torch.Tensor] = []
        self._val_labels: list[torch.Tensor] = []

    def forward(self, **batch: torch.Tensor) -> Any:
        return self.model(**batch)

    def _shared_step(self, batch: dict[str, torch.Tensor], stage: str) -> Any:
        output = self(**batch)
        batch_size = int(batch["labels"].size(0))
        self.log(f"{stage}_loss", output.loss, prog_bar=True, on_epoch=True, batch_size=batch_size)
        predictions = output.logits.argmax(dim=-1)
        accuracy = (predictions == batch["labels"]).float().mean()
        self.log(
            f"{stage}_acc",
            accuracy,
            prog_bar=stage == "val",
            on_epoch=True,
            batch_size=batch_size,
        )
        return output

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train").loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        output = self._shared_step(batch, "val")
        self._val_predictions.append(output.logits.argmax(dim=-1).detach())
        self._val_labels.append(batch["labels"].detach())
        return output.loss

    def on_validation_epoch_end(self) -> None:
        if not self._val_labels:
            return
        try:
            from .helpers import metrics_dict
        except ImportError:
            from train.helpers import metrics_dict

        labels = torch.cat(self._val_labels).cpu().tolist()
        predictions = torch.cat(self._val_predictions).cpu().tolist()
        metrics = metrics_dict(labels, predictions)
        for key in ("accuracy", "precision", "recall", "f1"):
            self.log(
                f"val_{key}",
                float(metrics[key]),
                prog_bar=key in {"accuracy", "f1"},
                on_epoch=True,
                on_step=False,
            )
        self._val_labels.clear()
        self._val_predictions.clear()

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "test").loss

    def predict_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> dict[str, torch.Tensor]:
        labels = batch.pop("labels")
        output = self(**batch)
        return {"predictions": output.logits.argmax(dim=-1), "labels": labels}

    def configure_optimizers(self) -> dict[str, Any]:
        decay: list[torch.nn.Parameter] = []
        no_decay: list[torch.nn.Parameter] = []
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            if parameter.ndim == 1 or name.endswith(".bias"):
                no_decay.append(parameter)
            else:
                decay.append(parameter)
        optimizer = torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": self.hparams.weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=self.hparams.learning_rate,
        )
        total_steps = max(int(self.trainer.estimated_stepping_batches), 1)
        warmup_steps = max(1, int(total_steps * self.hparams.warmup_ratio))

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return step / warmup_steps
            return max(0.0, (total_steps - step) / max(total_steps - warmup_steps, 1))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
