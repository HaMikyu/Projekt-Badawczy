"""Ewaluacja sciezki statystycznej poza domena (Ghostbuster OOD).

Porownuje dwa warianty regresji logistycznej na tym samym zbiorze OOD,
na ktorym mierzony jest distilbert-v3:
  1. model z run_experiment.py (trening tylko na Kaggle, runs/model.joblib),
  2. ta sama regresja trenowana na mieszance zrodel z data/prepared/train.csv
     (te same dane co distilbert-v3), oceniana na test.csv (IID) i ood.csv.

Wymaga: python statistical/run_experiment.py oraz python train/build_data.py.
Wynik: statistical/runs/ood_results.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train.evaluate import calculate_metrics  # noqa: E402
from statistical.features import FEATURE_NAMES, extract_features_df  # noqa: E402
from statistical.model import StatisticalLogisticClassifier  # noqa: E402

PREPARED = ROOT / "data" / "prepared"
RUNS = Path(__file__).resolve().parent / "runs"


def load_features(name: str) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    df = pd.read_csv(PREPARED / f"{name}.csv")
    start = time.time()
    X = extract_features_df(df["text"].astype(str).tolist())[FEATURE_NAMES]
    print(f"[{name}] {len(df)} tekstow, cechy w {time.time() - start:.1f}s")
    return X, df["generated"].astype(int), df


def evaluate(model, X: pd.DataFrame, y: pd.Series) -> dict:
    pred = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    metrics = calculate_metrics(y.values, pred)
    metrics["roc_auc"] = float(roc_auc_score(y.values, proba))
    return metrics


def accuracy_per_source(model, X: pd.DataFrame, df: pd.DataFrame) -> dict[str, float]:
    frame = df.assign(pred=model.predict(X))
    return {
        src: round(float((g["pred"] == g["generated"]).mean()), 4)
        for src, g in frame.groupby("source")
    }


def run() -> dict:
    X_ood, y_ood, df_ood = load_features("ood")

    kaggle_model = joblib.load(RUNS / "model.joblib")

    X_train, y_train, _ = load_features("train")
    X_test, y_test, _ = load_features("test")
    mix_model = StatisticalLogisticClassifier(C=1.0, random_state=42)
    mix_model.fit(X_train, y_train.values)

    results = {
        "kaggle_lr_on_ood": evaluate(kaggle_model, X_ood, y_ood),
        "mix_lr_on_iid_test": evaluate(mix_model, X_test, y_test),
        "mix_lr_on_ood": evaluate(mix_model, X_ood, y_ood),
        "ood_accuracy_per_source": {
            "kaggle_lr": accuracy_per_source(kaggle_model, X_ood, df_ood),
            "mix_lr": accuracy_per_source(mix_model, X_ood, df_ood),
        },
        "mix_lr_weights": mix_model.get_feature_importances().to_dict(orient="records"),
    }

    RUNS.mkdir(parents=True, exist_ok=True)
    out = RUNS / "ood_results.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    for key in ("kaggle_lr_on_ood", "mix_lr_on_iid_test", "mix_lr_on_ood"):
        m = results[key]
        print(f"{key:20} acc={m['accuracy']:.4f} f1={m['f1']:.4f} auc={m['roc_auc']:.4f}")
    print(f"Zapisano: {out}")
    return results


if __name__ == "__main__":
    run()
