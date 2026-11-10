from pathlib import Path
import json
import sys
import time

import lightning as L
import numpy as np
import pandas as pd
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train.helpers import metrics_dict
from train.model import BertTextClassifier

DATA_DIR = ROOT / "data" / "prepared"
RUN_DIR = ROOT / "train" / "runs" / "distilbert-v3"
MODEL_NAME = "distilbert-base-uncased"
SEED = 42
BATCH_SIZE = 32
LR = 2e-5
EPOCHS = 5
MAX_LENGTH = 256
PATIENCE = 2
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1


class TextDataset(Dataset):
    def __init__(self, frame, tokenizer, max_length):
        enc = tokenizer(
            frame["text"].tolist(),
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        self.encodings = enc
        self.labels = torch.tensor(frame["generated"].tolist(), dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


@torch.no_grad()
def predict_proba(model, loader, device):
    model.eval()
    model.to(device)
    probs, labels = [], []
    for batch in loader:
        labels.extend(batch["labels"].tolist())
        batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        logits = model(**batch).logits
        p = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()
        probs.append(p)
    return np.concatenate(probs), np.asarray(labels)


def best_threshold(probs, labels):
    best_t, best_f1, best_metrics = 0.5, -1.0, None
    for t in np.linspace(0.2, 0.8, 61):
        preds = (probs >= t).astype(int)
        m = metrics_dict(labels, preds)
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_t = float(t)
            best_metrics = m
    return best_t, best_metrics


def eval_probs(probs, labels, frame, name, threshold):
    preds = (probs >= threshold).astype(int)
    metrics = metrics_dict(labels, preds)
    metrics["name"] = name
    metrics["threshold"] = threshold
    if "source" in frame.columns:
        by_source = {}
        frame = frame.reset_index(drop=True)
        for source, idx in frame.groupby("source").groups.items():
            idx = list(idx)
            by_source[source] = metrics_dict([labels[i] for i in idx], [preds[i] for i in idx])
        metrics["by_source"] = by_source
    print(
        name,
        json.dumps({k: metrics[k] for k in ("accuracy", "precision", "recall", "f1", "n", "threshold", "confusion_matrix")}),
    )
    return metrics


def main():
    L.seed_everything(SEED, workers=True)
    torch.set_float32_matmul_precision("high")
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(DATA_DIR / "train.csv")
    val_df = pd.read_csv(DATA_DIR / "val.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    ood_df = pd.read_csv(DATA_DIR / "ood.csv")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_ds = TextDataset(train_df, tokenizer, MAX_LENGTH)
    val_ds = TextDataset(val_df, tokenizer, MAX_LENGTH)
    test_ds = TextDataset(test_df, tokenizer, MAX_LENGTH)
    ood_ds = TextDataset(ood_df, tokenizer, MAX_LENGTH)

    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=pin, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, pin_memory=pin)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, pin_memory=pin)
    ood_loader = DataLoader(ood_ds, batch_size=BATCH_SIZE, pin_memory=pin)

    model = BertTextClassifier(MODEL_NAME, LR, weight_decay=WEIGHT_DECAY, warmup_ratio=WARMUP_RATIO)
    ckpt = ModelCheckpoint(
        dirpath=RUN_DIR / "checkpoints",
        monitor="val_f1",
        mode="max",
        save_top_k=1,
        filename="best",
    )
    precision = "16-mixed" if torch.cuda.is_available() else "32-true"
    trainer = L.Trainer(
        accelerator="auto",
        devices=1,
        precision=precision,
        max_epochs=EPOCHS,
        logger=False,
        num_sanity_val_steps=0,
        gradient_clip_val=1.0,
        callbacks=[ckpt, EarlyStopping(monitor="val_f1", patience=PATIENCE, mode="max")],
    )

    started = time.time()
    trainer.fit(model, train_loader, val_loader)
    train_seconds = time.time() - started

    best = BertTextClassifier.load_from_checkpoint(ckpt.best_model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    val_probs, val_labels = predict_proba(best, val_loader, device)
    threshold, val_tuned = best_threshold(val_probs, val_labels)
    test_probs, test_labels = predict_proba(best, test_loader, device)
    ood_probs, ood_labels = predict_proba(best, ood_loader, device)

    results = {
        "model_name": MODEL_NAME,
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "learning_rate": LR,
        "max_length": MAX_LENGTH,
        "precision": precision,
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
        "ood_rows": len(ood_df),
        "train_seconds": train_seconds,
        "checkpoint": str(ckpt.best_model_path),
        "threshold": threshold,
        "val_tuned": val_tuned,
        "test": eval_probs(test_probs, test_labels, test_df, "test_iid", threshold),
        "ood": eval_probs(ood_probs, ood_labels, ood_df, "ood_ghostbuster", threshold),
        "ood_default_05": eval_probs(ood_probs, ood_labels, ood_df, "ood_ghostbuster_t0.5", 0.5),
    }
    (RUN_DIR / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("saved", RUN_DIR / "metrics.json")
    print("train_seconds", round(train_seconds, 1), "threshold", threshold)


if __name__ == "__main__":
    main()
