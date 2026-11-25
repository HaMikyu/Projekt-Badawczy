"""Douczenie System One v4 od wag v3 (train/GPU_TRAINING.md, punkt 3).

    python train/train_system_one_v4.py a   # bez okien: pelne teksty v3 + IvyPanda, lr 1e-5
    python train/train_system_one_v4.py b   # okna 1-3 fragmentow, lr 1e-5
    python train/train_system_one_v4.py c   # okna 1-3 fragmentow, lr 2e-5

Kazdy wariant: train/runs/system-one-v4<x>/ (metrics.json, export_meta.json) oraz
gotowy do oceny katalog export/ (system_one.pt + tokenizer) dla train/eval_dev.py.
models/system-one-final/ nie jest ruszany.
"""
from pathlib import Path
import json
import sys
import time

import lightning as L
import pandas as pd
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train.system_one_model import OPTIONS, SystemOneOptionScorer  # noqa: E402
from train.train_system_one import (  # noqa: E402
    CLASS_WEIGHTS,
    LABEL_SMOOTHING,
    MAX_LENGTH,
    MODEL_NAME,
    SEED,
    WARMUP_RATIO,
    WEIGHT_DECAY,
    OptionRankDataset,
    best_threshold,
    eval_probs,
    fit_temperature,
    predict_logits,
)

V3_BUNDLE = ROOT / "models" / "system-one-final" / "system_one.pt"
VARIANTS = {
    "a": {"data": "prepared_v4_full", "lr": 1e-5, "note": "bez okien, + IvyPanda"},
    "b": {"data": "prepared_v4", "lr": 1e-5, "note": "okna, + IvyPanda"},
    "c": {"data": "prepared_v4", "lr": 2e-5, "note": "okna, + IvyPanda, lr 2e-5"},
}
BATCH_SIZE = 16
EPOCHS = 3
PATIENCE = 1


def export(model, temperature, threshold, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    model = model.cpu().eval()
    model.temperature = float(temperature)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "hparams": dict(model.hparams),
            "temperature": float(temperature),
            "options": OPTIONS,
            "threshold": float(threshold),
            "max_length": MAX_LENGTH,
            "model_name": MODEL_NAME,
        },
        out_dir / "system_one.pt",
    )
    AutoTokenizer.from_pretrained(MODEL_NAME).save_pretrained(out_dir)


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "b"
    cfg = VARIANTS[name]
    data_dir = ROOT / "data" / cfg["data"]
    run_dir = ROOT / "train" / "runs" / f"system-one-v4{name}"
    run_dir.mkdir(parents=True, exist_ok=True)

    L.seed_everything(SEED, workers=True)
    torch.set_float32_matmul_precision("high")

    frames = {s: pd.read_csv(data_dir / f"{s}.csv") for s in ("train", "val", "test", "ood")}
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    t0 = time.time()
    ds = {s: OptionRankDataset(f, tokenizer, MAX_LENGTH) for s, f in frames.items()}
    print(f"[{name}] tokenizacja {time.time() - t0:.0f} s, rozmiary:", {s: len(d) for s, d in ds.items()}, flush=True)

    pin = torch.cuda.is_available()
    loaders = {
        s: DataLoader(d, batch_size=BATCH_SIZE, shuffle=(s == "train"), pin_memory=pin, num_workers=0)
        for s, d in ds.items()
    }

    model = SystemOneOptionScorer(
        MODEL_NAME,
        cfg["lr"],
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        class_weights=CLASS_WEIGHTS,
        label_smoothing=LABEL_SMOOTHING,
    )
    bundle = torch.load(V3_BUNDLE, map_location="cpu", weights_only=False)
    model.load_state_dict(bundle["state_dict"])

    ckpt = ModelCheckpoint(dirpath=run_dir / "checkpoints", monitor="val_balanced", mode="max",
                           save_top_k=1, filename="best")
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
    trainer.fit(model, loaders["train"], loaders["val"])
    train_seconds = time.time() - started

    best = SystemOneOptionScorer.load_from_checkpoint(ckpt.best_model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    val_logits, val_labels = predict_logits(best, loaders["val"], device)
    temperature = fit_temperature(val_logits, val_labels)
    best.temperature = temperature
    val_probs = torch.softmax(torch.tensor(val_logits) / temperature, dim=-1)[:, 1].numpy()
    threshold, val_tuned = best_threshold(val_probs, val_labels)

    def probs(split):
        logits, labels = predict_logits(best, loaders[split], device)
        return torch.softmax(torch.tensor(logits) / temperature, dim=-1)[:, 1].numpy(), labels

    test_p, test_y = probs("test")
    ood_p, ood_y = probs("ood")
    results = {
        "variant": name,
        **cfg,
        "init": "models/system-one-final (v3)",
        "batch_size": BATCH_SIZE,
        "max_epochs": EPOCHS,
        "epochs_run": int(trainer.current_epoch),
        "best_epoch_ckpt": str(ckpt.best_model_path),
        "max_length": MAX_LENGTH,
        "class_weights": CLASS_WEIGHTS,
        "temperature": temperature,
        "precision": precision,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "rows": {s: len(d) for s, d in ds.items()},
        "train_seconds": train_seconds,
        "threshold": threshold,
        "val_tuned": val_tuned,
        "test": eval_probs(test_p, test_y, frames["test"], "test_iid", threshold),
        "test_t05": eval_probs(test_p, test_y, frames["test"], "test_iid_t0.5", 0.5),
        "ood": eval_probs(ood_p, ood_y, frames["ood"], "ood_ghostbuster", threshold),
        "ood_default_05": eval_probs(ood_p, ood_y, frames["ood"], "ood_ghostbuster_t0.5", 0.5),
    }
    (run_dir / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    meta = {"model_name": MODEL_NAME, "architecture": "system_one_option_scorer", "options": OPTIONS,
            "temperature": temperature, "threshold": threshold, "max_length": MAX_LENGTH,
            "checkpoint": str(ckpt.best_model_path), "variant": name}
    (run_dir / "export_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    export(best, temperature, threshold, run_dir / "export")
    print(f"[{name}] gotowe: {train_seconds / 60:.1f} min, T={temperature:.3f}, thr={threshold:.2f}", flush=True)


if __name__ == "__main__":
    main()
