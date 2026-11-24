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
from train.system_one_model import OPTIONS, SystemOneOptionScorer

DATA_DIR = ROOT / "data" / "prepared"
RUN_DIR = ROOT / "train" / "runs" / "system-one-v3"
MODEL_NAME = "distilbert-base-uncased"
SEED = 42
BATCH_SIZE = 12
LR = 1.5e-5
EPOCHS = 4
MAX_LENGTH = 384
PATIENCE = 2
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.06
CLASS_WEIGHTS = [2.8, 1.0]
LABEL_SMOOTHING = 0.03


class OptionRankDataset(Dataset):
    def __init__(self, frame, tokenizer, max_length):
        texts = frame["text"].astype(str).tolist()
        labels = frame["generated"].astype(int).tolist()
        pair_a = tokenizer(
            texts,
            [OPTIONS[0]] * len(texts),
            padding="max_length",
            truncation="only_first",
            max_length=max_length,
            return_tensors="pt",
        )
        pair_b = tokenizer(
            texts,
            [OPTIONS[1]] * len(texts),
            padding="max_length",
            truncation="only_first",
            max_length=max_length,
            return_tensors="pt",
        )
        self.input_ids = torch.stack([pair_a["input_ids"], pair_b["input_ids"]], dim=1)
        self.attention_mask = torch.stack([pair_a["attention_mask"], pair_b["attention_mask"]], dim=1)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }


@torch.no_grad()
def predict_logits(model, loader, device):
    model.eval()
    model.to(device)
    logits_all, labels = [], []
    for batch in loader:
        labels.extend(batch["labels"].tolist())
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        logits = model(ids, mask)
        logits_all.append(logits.detach().cpu())
    return torch.cat(logits_all).numpy(), np.asarray(labels)


def fit_temperature(logits, labels):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.tensor(logits, dtype=torch.float32, device=device)
    y = torch.tensor(labels, dtype=torch.long, device=device)
    log_t = torch.nn.Parameter(torch.zeros((), device=device))
    opt = torch.optim.LBFGS([log_t], lr=0.5, max_iter=50)

    def closure():
        opt.zero_grad()
        t = torch.exp(log_t).clamp(0.05, 10.0)
        loss = torch.nn.functional.cross_entropy(x / t, y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(torch.exp(log_t).clamp(0.05, 10.0).item())


def best_threshold(probs, labels):
    best_t, best_score, best_metrics = 0.5, -1.0, None
    for t in np.linspace(0.2, 0.85, 66):
        preds = (probs >= t).astype(int)
        m = metrics_dict(labels, preds)
        cm = np.asarray(m["confusion_matrix"])
        tn, fp, fn, tp = cm.ravel()
        tpr = tp / max(tp + fn, 1)
        tnr = tn / max(tn + fp, 1)
        if tpr < 0.94:
            continue
        score = 0.45 * tpr + 0.55 * tnr
        if score > best_score:
            best_score = score
            best_t = float(t)
            best_metrics = m
            best_metrics["balanced_acc"] = float(0.5 * (tpr + tnr))
            best_metrics["tpr"] = float(tpr)
            best_metrics["tnr"] = float(tnr)
    if best_metrics is None:
        for t in np.linspace(0.2, 0.8, 61):
            preds = (probs >= t).astype(int)
            m = metrics_dict(labels, preds)
            cm = np.asarray(m["confusion_matrix"])
            tn, fp, fn, tp = cm.ravel()
            tpr = tp / max(tp + fn, 1)
            tnr = tn / max(tn + fp, 1)
            score = 0.5 * (tpr + tnr)
            if score > best_score:
                best_score = score
                best_t = float(t)
                best_metrics = m
                best_metrics["balanced_acc"] = float(score)
    return best_t, best_metrics


def eval_probs(probs, labels, frame, name, threshold):
    preds = (probs >= threshold).astype(int)
    metrics = metrics_dict(labels, preds)
    cm = np.asarray(metrics["confusion_matrix"])
    tn, fp, fn, tp = cm.ravel()
    metrics["tpr"] = float(tp / max(tp + fn, 1))
    metrics["tnr"] = float(tn / max(tn + fp, 1))
    metrics["balanced_acc"] = float(0.5 * (metrics["tpr"] + metrics["tnr"]))
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
        json.dumps(
            {
                k: metrics[k]
                for k in (
                    "accuracy",
                    "precision",
                    "recall",
                    "f1",
                    "balanced_acc",
                    "tpr",
                    "tnr",
                    "n",
                    "threshold",
                    "confusion_matrix",
                )
            }
        ),
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
    train_ds = OptionRankDataset(train_df, tokenizer, MAX_LENGTH)
    val_ds = OptionRankDataset(val_df, tokenizer, MAX_LENGTH)
    test_ds = OptionRankDataset(test_df, tokenizer, MAX_LENGTH)
    ood_ds = OptionRankDataset(ood_df, tokenizer, MAX_LENGTH)

    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=pin, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, pin_memory=pin)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, pin_memory=pin)
    ood_loader = DataLoader(ood_ds, batch_size=BATCH_SIZE, pin_memory=pin)

    model = SystemOneOptionScorer(
        MODEL_NAME,
        LR,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        class_weights=CLASS_WEIGHTS,
        label_smoothing=LABEL_SMOOTHING,
    )
    ckpt = ModelCheckpoint(
        dirpath=RUN_DIR / "checkpoints",
        monitor="val_balanced",
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
        callbacks=[ckpt, EarlyStopping(monitor="val_balanced", patience=PATIENCE, mode="max")],
    )

    started = time.time()
    trainer.fit(model, train_loader, val_loader)
    train_seconds = time.time() - started

    best = SystemOneOptionScorer.load_from_checkpoint(ckpt.best_model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    val_logits, val_labels = predict_logits(best, val_loader, device)
    temperature = fit_temperature(val_logits, val_labels)
    best.temperature = temperature

    val_probs = torch.softmax(torch.tensor(val_logits) / temperature, dim=-1)[:, 1].numpy()
    threshold, val_tuned = best_threshold(val_probs, val_labels)

    test_logits, test_labels = predict_logits(best, test_loader, device)
    ood_logits, ood_labels = predict_logits(best, ood_loader, device)
    test_probs = torch.softmax(torch.tensor(test_logits) / temperature, dim=-1)[:, 1].numpy()
    ood_probs = torch.softmax(torch.tensor(ood_logits) / temperature, dim=-1)[:, 1].numpy()

    results = {
        "model_name": MODEL_NAME,
        "architecture": "system_one_option_scorer",
        "options": OPTIONS,
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "learning_rate": LR,
        "max_length": MAX_LENGTH,
        "class_weights": CLASS_WEIGHTS,
        "label_smoothing": LABEL_SMOOTHING,
        "temperature": temperature,
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
        "ood_t06": eval_probs(ood_probs, ood_labels, ood_df, "ood_ghostbuster_t0.6", 0.6),
        "ood_t07": eval_probs(ood_probs, ood_labels, ood_df, "ood_ghostbuster_t0.7", 0.7),
    }
    (RUN_DIR / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    meta = {
        "model_name": MODEL_NAME,
        "architecture": "system_one_option_scorer",
        "options": OPTIONS,
        "temperature": temperature,
        "threshold": threshold,
        "max_length": MAX_LENGTH,
        "checkpoint": str(ckpt.best_model_path),
    }
    (RUN_DIR / "export_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("saved", RUN_DIR / "metrics.json")
    print("train_seconds", round(train_seconds, 1), "T", round(temperature, 3), "thr", threshold)


if __name__ == "__main__":
    main()
