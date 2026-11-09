from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score


def calculate_metrics(labels: Iterable[int], predictions: Iterable[int]) -> dict[str, object]:
    y_true = np.asarray(list(labels))
    y_pred = np.asarray(list(predictions))
    if len(y_true) == 0 or len(y_true) != len(y_pred):
        raise ValueError("Labels and predictions must be non-empty and have equal length")
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }


def predictions_from_batches(batches: Iterable[dict[str, object]]) -> tuple[list[int], list[int]]:
    predictions: list[int] = []
    labels: list[int] = []
    for batch in batches:
        predictions.extend(int(value) for value in batch["predictions"])
        labels.extend(int(value) for value in batch["labels"])
    return labels, predictions


def metrics_by_length_bucket(
    labels: Iterable[int],
    predictions: Iterable[int],
    lengths: Iterable[int],
    bins: int = 4,
) -> list[dict[str, object]]:
    """Metrics per text-length quartile, to expose dependence on text length."""
    frame = pd.DataFrame({
        "label": list(labels),
        "prediction": list(predictions),
        "length": list(lengths),
    })
    if frame.empty or frame["length"].nunique() < 2:
        return []
    frame["bucket"] = pd.qcut(frame["length"], q=min(bins, frame["length"].nunique()), duplicates="drop")
    rows: list[dict[str, object]] = []
    for interval, group in frame.groupby("bucket", observed=True):
        metrics = calculate_metrics(group["label"], group["prediction"])
        rows.append({
            "bucket": str(interval),
            "rows": int(len(group)),
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
        })
    return rows


def length_only_auc(labels: Iterable[int], lengths: Iterable[int]) -> float | None:
    """AUC of a trivial `shorter text -> AI` classifier. >0.5 means length is informative."""
    y_true = np.asarray(list(labels))
    values = np.asarray([float(value) for value in lengths])
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, -values))


def save_metrics(metrics: dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
