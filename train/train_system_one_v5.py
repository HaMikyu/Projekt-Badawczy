"""Trening System One v5 na data/prepared_v5 (tylko eseje, wiecej generatorow).

    python train/train_system_one_v5.py a   # DistilBERT od wag v4, lr 1e-5
    python train/train_system_one_v5.py d   # DeBERTa-v3-base od zera (wstepnie wytrenowany), lr 2e-5, bf16

Wynik: train/runs/system-one-v5<x>/ (metrics.json, progress.log) i export/ (system_one.pt + tokenizer)
w formacie aplikacji. Wybor wariantu tylko na walidacji, dev i Ghostbuster OOD; testy esejowe
(data/prepared_v5_tests) i finalne uruchamia osobno train/eval_essay_suite.py.
"""
from pathlib import Path
import json
import sys
import time

import lightning as L
import pandas as pd
import torch
from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train.system_one_model import OPTIONS, SystemOneOptionScorer  # noqa: E402
from train.train_system_one import (  # noqa: E402
    CLASS_WEIGHTS, LABEL_SMOOTHING, MAX_LENGTH, SEED, WARMUP_RATIO, WEIGHT_DECAY,
    OptionRankDataset, best_threshold, eval_probs, fit_temperature, predict_logits,
)

DATA = ROOT / "data" / "prepared_v5"
VARIANTS = {
    "a": {"model_name": "distilbert-base-uncased", "init": "models/system-one-v4/system_one.pt",
          "lr": 1e-5, "batch": 16, "accumulate": 1, "epochs": 2, "precision": "16-mixed"},
    "d": {"model_name": "microsoft/deberta-v3-base", "init": None,
          "lr": 2e-5, "batch": 8, "accumulate": 2, "epochs": 1, "precision": "bf16-mixed"},
}
PATIENCE = 1


class Progress(Callback):
    """Postep do pliku (Lightning nie pokazuje paska poza terminalem)."""

    def __init__(self, path, every=500):
        self.path, self.every, self.t0 = path, every, None

    def _write(self, line):
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def on_train_start(self, trainer, pl_module):
        self.t0 = time.time()
        self._write(f"start: {trainer.estimated_stepping_batches} krokow optymalizatora")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = trainer.global_step
        if step and step % self.every == 0 and batch_idx % trainer.accumulate_grad_batches == 0:
            total = trainer.estimated_stepping_batches
            el = time.time() - self.t0
            eta = el / step * (total - step)
            loss = float(outputs["loss"]) if isinstance(outputs, dict) else float(outputs)
            if loss != loss:  # NaN - nie marnujemy godzin GPU
                self._write("loss NaN - przerywam trening"); trainer.should_stop = True
            self._write(f"epoka {trainer.current_epoch} krok {step}/{total} loss {loss:.4f} "
                        f"uplynelo {el / 60:.1f} min, zostalo ~{eta / 60:.0f} min")

    def on_validation_end(self, trainer, pl_module):
        m = trainer.callback_metrics.get("val_balanced")
        if m is not None:
            self._write(f"walidacja po epoce {trainer.current_epoch}: val_balanced {float(m):.4f}")


def export(model, tokenizer_name, temperature, threshold, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    model = model.cpu().float().eval()
    model.temperature = float(temperature)
    torch.save({"state_dict": model.state_dict(), "hparams": dict(model.hparams), "temperature": float(temperature),
                "options": OPTIONS, "threshold": float(threshold), "max_length": MAX_LENGTH,
                "model_name": tokenizer_name}, out_dir / "system_one.pt")
    AutoTokenizer.from_pretrained(tokenizer_name).save_pretrained(out_dir)


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "a"
    cfg = VARIANTS[name]
    run_dir = ROOT / "train" / "runs" / f"system-one-v5{name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "progress.log").unlink(missing_ok=True)

    L.seed_everything(SEED, workers=True)
    torch.set_float32_matmul_precision("high")
    frames = {s: pd.read_csv(DATA / f"{s}.csv") for s in ("train", "val", "test", "ood")}
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    t0 = time.time()
    ds = {s: OptionRankDataset(f, tokenizer, MAX_LENGTH) for s, f in frames.items()}
    print(f"[{name}] tokenizacja {time.time() - t0:.0f} s, rozmiary:", {s: len(d) for s, d in ds.items()}, flush=True)

    pin = torch.cuda.is_available()
    loaders = {s: DataLoader(d, batch_size=cfg["batch"] if s == "train" else cfg["batch"] * 2,
                             shuffle=(s == "train"), pin_memory=pin, num_workers=0) for s, d in ds.items()}

    model = SystemOneOptionScorer(cfg["model_name"], cfg["lr"], weight_decay=WEIGHT_DECAY,
                                  warmup_ratio=WARMUP_RATIO, class_weights=CLASS_WEIGHTS,
                                  label_smoothing=LABEL_SMOOTHING)
    if cfg["init"]:
        model.load_state_dict(torch.load(ROOT / cfg["init"], map_location="cpu", weights_only=False)["state_dict"])

    ckpt = ModelCheckpoint(dirpath=run_dir / "checkpoints", monitor="val_balanced", mode="max",
                           save_top_k=1, filename="best")
    precision = cfg["precision"] if torch.cuda.is_available() else "32-true"
    trainer = L.Trainer(accelerator="auto", devices=1, precision=precision, max_epochs=cfg["epochs"],
                        accumulate_grad_batches=cfg["accumulate"], logger=False, num_sanity_val_steps=0,
                        gradient_clip_val=1.0, enable_progress_bar=False,
                        callbacks=[ckpt, EarlyStopping(monitor="val_balanced", patience=PATIENCE, mode="max"),
                                   Progress(run_dir / "progress.log")])
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
        "variant": name, **cfg, "epochs_run": int(trainer.current_epoch), "best_ckpt": str(ckpt.best_model_path),
        "max_length": MAX_LENGTH, "class_weights": CLASS_WEIGHTS, "temperature": temperature, "precision": precision,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "rows": {s: len(d) for s, d in ds.items()}, "train_seconds": train_seconds,
        "threshold": threshold, "val_tuned": val_tuned,
        "test": eval_probs(test_p, test_y, frames["test"], "test_iid", threshold),
        "test_t05": eval_probs(test_p, test_y, frames["test"], "test_iid_t0.5", 0.5),
        "ood_default_05": eval_probs(ood_p, ood_y, frames["ood"], "ood_ghostbuster_t0.5", 0.5),
    }
    (run_dir / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    export(best, cfg["model_name"], temperature, threshold, run_dir / "export")
    print(f"[{name}] gotowe: {train_seconds / 60:.1f} min, T={temperature:.3f}, thr={threshold:.2f}", flush=True)


if __name__ == "__main__":
    main()
