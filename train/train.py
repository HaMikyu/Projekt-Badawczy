from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import lightning as L
import torch
from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint
from transformers import AutoTokenizer

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from train.data import (
        create_dataloaders,
        deduplicate_frame,
        load_dataframe,
        split_dataframe,
        split_overlap,
        split_summary,
    )
    from train.evaluate import (
        calculate_metrics,
        length_only_auc,
        metrics_by_length_bucket,
        save_metrics,
    )
    from train.model import BertTextClassifier
else:
    from .data import (
        create_dataloaders,
        deduplicate_frame,
        load_dataframe,
        split_dataframe,
        split_overlap,
        split_summary,
    )
    from .evaluate import (
        calculate_metrics,
        length_only_auc,
        metrics_by_length_bucket,
        save_metrics,
    )
    from .model import BertTextClassifier


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "Training_Essay_Data.csv"
DEFAULT_RUNS = Path(__file__).resolve().parent / "runs"
REPORTED_METRICS = (
    "train_loss",
    "train_acc",
    "val_loss",
    "val_acc",
    "val_accuracy",
    "val_precision",
    "val_recall",
    "val_f1",
)


def resolve_precision(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return "bf16-mixed"
    return "32-true"


def monitor_mode(monitor: str) -> str:
    return "max" if any(token in monitor for token in ("f1", "acc", "precision", "recall")) else "min"


def display_path(path: Path | str, base: Path = ROOT) -> str:
    """Return a portable path for JSON artifacts.

    Paths inside the repository are stored relative to its root with POSIX
    separators, so `config.json`/`metrics.json` stay readable on other machines.
    External paths remain absolute as a last resort.
    """
    resolved = Path(path).expanduser().resolve()
    try:
        return resolved.relative_to(base.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def collect_metrics(trainer: L.Trainer) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for key, value in trainer.callback_metrics.items():
        if key not in REPORTED_METRICS:
            continue
        metrics[key] = float(value.detach().cpu()) if torch.is_tensor(value) else float(value)
    for key in ("accuracy", "precision", "recall", "f1"):
        value = metrics.get(f"val_{key}")
        if value is not None:
            metrics[key] = value
    return metrics


class DashboardCallback(Callback):
    def __init__(self, run_dir: Path, max_epochs: int) -> None:
        self.run_dir = run_dir
        self.max_epochs = max_epochs
        self.started = time.time()
        self.history_path = run_dir / "history.jsonl"
        self.progress_path = run_dir / "progress.json"
        self.last_metrics: dict[str, float] = {}
        self.best_val: dict[str, Any] | None = None
        self.completed_epoch = 0

    def _write(self, payload: dict[str, Any]) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        payload["updated_at"] = time.time()
        self.progress_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        with self.history_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload) + "\n")

    def on_train_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self._write({"status": "training", "epoch": 0, "max_epochs": self.max_epochs, "metrics": {}})

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if batch_idx % 25 != 0:
            return
        batches = max(int(trainer.num_training_batches), 1)
        completed = trainer.current_epoch + (batch_idx + 1) / batches
        elapsed = max(time.time() - self.started, 0.001)
        self._write({
            "status": "training",
            "epoch": trainer.current_epoch + 1,
            "max_epochs": self.max_epochs,
            "progress": min(completed / self.max_epochs, 1.0),
            "eta_seconds": max(0.0, (self.max_epochs - completed) * elapsed / completed),
            "metrics": self.last_metrics,
        })

    def on_validation_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        metrics = collect_metrics(trainer)
        self.last_metrics = metrics
        epoch = trainer.current_epoch + 1
        self.completed_epoch = epoch
        val_f1 = metrics.get("val_f1")
        if val_f1 is not None and (self.best_val is None or val_f1 > self.best_val["val_f1"]):
            self.best_val = {
                "epoch": epoch,
                **{
                    key: metrics[key]
                    for key in ("val_loss", "val_f1", "val_accuracy", "val_precision", "val_recall")
                    if key in metrics
                },
            }
        elapsed = max(time.time() - self.started, 0.001)
        self._write({
            "status": "training",
            "epoch": epoch,
            "max_epochs": self.max_epochs,
            "progress": min(epoch / self.max_epochs, 1.0),
            "eta_seconds": max(0.0, (self.max_epochs - epoch) * elapsed / epoch),
            "metrics": metrics,
        })

    def on_train_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        epoch = self.completed_epoch or min(trainer.current_epoch + 1, self.max_epochs)
        self._write({
            "status": "done",
            "epoch": epoch,
            "max_epochs": self.max_epochs,
            "progress": min(epoch / self.max_epochs, 1.0),
            "eta_seconds": 0,
            "metrics": self.last_metrics,
            "best_val": self.best_val,
        })


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune a transformer for AI text detection")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model-name", default="distilbert-base-uncased")
    parser.add_argument("--run-name", default="distilbert-base-uncased")
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--monitor", default="val_f1", choices=["val_f1", "val_accuracy", "val_loss"])
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--gradient-clip-val", type=float, default=1.0)
    parser.add_argument("--min-words", type=int, default=5)
    parser.add_argument(
        "--precision",
        default="auto",
        choices=["auto", "bf16-mixed", "16-mixed", "32-true"],
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-test", action="store_true", help="Do not touch the test split (used by tuning)")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, object]:
    L.seed_everything(args.seed, workers=True)
    run_dir = args.runs_dir / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    precision = resolve_precision(args.precision)
    mode = monitor_mode(args.monitor)
    (run_dir / "progress.json").write_text(
        json.dumps({
            "status": "preparing",
            "epoch": 0,
            "max_epochs": args.epochs,
            "progress": 0,
            "eta_seconds": None,
            "metrics": {},
            "updated_at": time.time(),
        }, indent=2),
        encoding="utf-8",
    )
    raw_frame = load_dataframe(args.data, args.limit, args.seed)
    frame, cleaning = deduplicate_frame(raw_frame, args.min_words)
    splits = split_dataframe(frame, args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    loaders = create_dataloaders(
        splits,
        tokenizer,
        args.batch_size,
        args.max_length,
        pin_memory=torch.cuda.is_available(),
        num_workers=args.num_workers,
    )
    model = BertTextClassifier(
        args.model_name,
        args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
    )
    checkpoint = ModelCheckpoint(
        dirpath=run_dir / "checkpoints",
        monitor=args.monitor,
        mode=mode,
        save_top_k=1,
        filename="best",
    )
    dashboard = DashboardCallback(run_dir, args.epochs)
    trainer = L.Trainer(
        accelerator="auto",
        devices=1,
        precision=precision,
        max_epochs=args.epochs,
        logger=False,
        num_sanity_val_steps=0,
        log_every_n_steps=50,
        gradient_clip_val=args.gradient_clip_val,
        gradient_clip_algorithm="norm",
        callbacks=[
            checkpoint,
            EarlyStopping(monitor=args.monitor, patience=args.patience, mode=mode),
            dashboard,
        ],
    )
    metadata = {
        "model_name": args.model_name,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "max_length": args.max_length,
        "patience": args.patience,
        "monitor": args.monitor,
        "mixed_precision": precision,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "gradient_clip_val": args.gradient_clip_val,
        "min_words": args.min_words,
        "dataset": display_path(args.data),
        "cleaning": cleaning,
        "leakage": split_overlap(splits.train, splits.validation, splits.test),
        "splits": split_summary(splits),
    }
    (run_dir / "config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    trainer.fit(model, loaders["train"], loaders["validation"])
    best_checkpoint = checkpoint.best_model_path
    result: dict[str, object] = {
        **metadata,
        "checkpoint": display_path(best_checkpoint) if best_checkpoint else None,
        "best_val": dashboard.best_val,
    }
    if not args.skip_test:
        outputs = trainer.predict(model, dataloaders=loaders["test"])
        labels: list[int] = []
        predictions: list[int] = []
        for batch in outputs:
            labels.extend(int(value) for value in batch["labels"])
            predictions.extend(int(value) for value in batch["predictions"])
        metrics = calculate_metrics(labels, predictions)
        word_lengths = splits.test["text"].str.split().str.len().tolist()
        result.update(metrics)
        result["length_buckets"] = metrics_by_length_bucket(labels, predictions, word_lengths)
        result["length_only_auc"] = length_only_auc(labels, word_lengths)
    save_metrics(result, run_dir / "metrics.json")
    return result


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2))
