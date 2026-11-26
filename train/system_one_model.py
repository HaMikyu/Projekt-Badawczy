from __future__ import annotations

from typing import Any

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel


HUMAN_OPTION = (
    "label=human. Written by a person. Look for uneven rhythm, concrete details, "
    "personal stance, imperfect flow, citations or local facts, student voice."
)
AI_OPTION = (
    "label=ai. Written by an LLM. Look for formulaic template, evenly polished prose, "
    "generic thesis, empty transitions, little lived detail, template conclusions."
)
OPTIONS = [HUMAN_OPTION, AI_OPTION]


class SystemOneOptionScorer(L.LightningModule):
    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.06,
        class_weights: list[float] | None = None,
        label_smoothing: float = 0.05,
        temperature: float = 1.0,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["pretrained"])
        config = AutoConfig.from_pretrained(model_name)
        if pretrained:
            # transformers 5 laduje wagi w typie checkpointu (DeBERTa-v3 = fp16 -> NaN w AdamW); trenujemy w fp32 + AMP
            self.encoder = AutoModel.from_pretrained(model_name, dtype=torch.float32)
        else:
            # aplikacja: wszystkie wagi sa w system_one.pt, wiec bez pobierania wag bazowych
            self.encoder = AutoModel.from_config(config, dtype=torch.float32)
        self.encoder.train()
        self.dropout = nn.Dropout(0.1)
        self.score_head = nn.Linear(config.hidden_size, 1)
        self._class_weights = class_weights
        self.temperature = float(temperature)
        self._val_logits: list[torch.Tensor] = []
        self._val_labels: list[torch.Tensor] = []

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.train(mode)
        return self

    def score_pairs(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0]
        cls = self.dropout(cls)
        return self.score_head(cls).squeeze(-1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        batch = input_ids.size(0)
        flat_ids = input_ids.reshape(batch * 2, -1)
        flat_mask = attention_mask.reshape(batch * 2, -1)
        scores = self.score_pairs(flat_ids, flat_mask).reshape(batch, 2)
        return scores / max(self.temperature, 1e-4)

    def _loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        weight = None
        if self._class_weights is not None:
            weight = torch.tensor(self._class_weights, dtype=logits.dtype, device=logits.device)
        return F.cross_entropy(
            logits,
            labels,
            weight=weight,
            label_smoothing=float(self.hparams.label_smoothing),
        )

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        logits = self(batch["input_ids"], batch["attention_mask"])
        loss = self._loss(logits, batch["labels"])
        preds = logits.argmax(dim=-1)
        acc = (preds == batch["labels"]).float().mean()
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, batch_size=labels_n(batch))
        self.log("train_acc", acc, on_epoch=True, batch_size=labels_n(batch))
        return loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        logits = self(batch["input_ids"], batch["attention_mask"])
        loss = self._loss(logits, batch["labels"])
        self._val_logits.append(logits.detach())
        self._val_labels.append(batch["labels"].detach())
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, batch_size=labels_n(batch))
        return loss

    def on_validation_epoch_end(self) -> None:
        if not self._val_labels:
            return
        try:
            from .helpers import metrics_dict
        except ImportError:
            from train.helpers import metrics_dict

        logits = torch.cat(self._val_logits)
        labels = torch.cat(self._val_labels)
        preds = logits.argmax(dim=-1)
        metrics = metrics_dict(labels.cpu().tolist(), preds.cpu().tolist())
        probs = torch.softmax(logits, dim=-1)[:, 1]
        cm = torch.tensor(metrics["confusion_matrix"])
        tn, fp, fn, tp = cm.reshape(-1).tolist()
        tpr = tp / max(tp + fn, 1)
        tnr = tn / max(tn + fp, 1)
        balanced = 0.5 * (tpr + tnr)
        self.log("val_f1", float(metrics["f1"]), prog_bar=True, on_epoch=True)
        self.log("val_acc", float(metrics["accuracy"]), prog_bar=True, on_epoch=True)
        self.log("val_balanced", float(balanced), prog_bar=True, on_epoch=True)
        self.log("val_mean_p_ai", float(probs.mean()), on_epoch=True)
        self._val_logits.clear()
        self._val_labels.clear()

    def configure_optimizers(self) -> dict[str, Any]:
        decay, no_decay = [], []
        for name, parameter in self.named_parameters():
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


def labels_n(batch: dict[str, torch.Tensor]) -> int:
    return int(batch["labels"].size(0))
