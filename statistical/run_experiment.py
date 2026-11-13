"""Glowny skrypt eksperymentu sciezki statystycznej (Michal Niechcialkowski).

Uruchamia pelny rygorystyczny proces badawczy:
1. Wczytanie i deduplikacja danych dokladnie jak w sciezce ML (ten sam seed=42, podzial 80/10/10)
2. Ekstrakcja wektorow cech stylometrycznych (sentence length, burstiness, TTR, stopwords, etc.)
3. Trening modeli:
   - Model A: Progi / reguly decyzyjne (DecisionTree depth=2)
   - Model B: Regresja Logistyczna ze standaryzacja cech
4. Obliczenie metryk: Accuracy, Precision, Recall, F1, Macierz pomylek, ROC-AUC
5. Analiza wrazliwosci na dlugosc tekstu (kwartyle Q1-Q4)
6. Zapis raportu i wynikow do statistical/WYNIKI.md oraz statistical/results.json
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train.data import deduplicate_frame, load_dataframe, split_dataframe
from train.evaluate import calculate_metrics, length_only_auc, metrics_by_length_bucket
from statistical.features import FEATURE_NAMES, extract_features_df
from statistical.model import RuleBasedThresholdClassifier, StatisticalLogisticClassifier

DATA_PATH = ROOT / "data" / "Training_Essay_Data.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "runs"
CACHE_DIR = Path(__file__).resolve().parent / "cache"


def get_cached_features(split_name: str, df: pd.DataFrame) -> pd.DataFrame:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"features_{split_name}_{len(df)}.csv"
    if cache_file.exists():
        print(f"[{split_name}] Wczytywanie wyekstrahowanych cech z cache: {cache_file.name}")
        return pd.read_csv(cache_file)
    print(f"[{split_name}] Ekstrakcja cech stylometrycznych dla {len(df)} tekstow...")
    feats = extract_features_df(df["text"].tolist())
    feats.to_csv(cache_file, index=False)
    return feats


def run() -> dict[str, object]:
    print("=== SCIEZKA STATYSTYCZNA: START EKSPERYMENTU ===")
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Nie znaleziono zbioru: {DATA_PATH}")

    # 1. Spojne dane i podzial
    print("1. Wczytywanie i deduplikacja danych...")
    raw_df = load_dataframe(DATA_PATH)
    clean_df, dedup_report = deduplicate_frame(raw_df, min_words=5)
    splits = split_dataframe(clean_df, seed=42, validation_ratio=0.1, test_ratio=0.1)

    print(f"   Wiersze: raw={len(raw_df)}, clean={len(clean_df)}")
    print(f"   Podzial: train={len(splits.train)}, val={len(splits.validation)}, test={len(splits.test)}")

    # 2. Ekstrakcja cech
    X_train = get_cached_features("train", splits.train)
    y_train = splits.train["generated"].values

    X_val = get_cached_features("val", splits.validation)
    y_val = splits.validation["generated"].values

    X_test = get_cached_features("test", splits.test)
    y_test = splits.test["generated"].values

    # Dlugosci w slowach do analizy wrazliwosci
    test_word_lengths = splits.test["text"].str.split().str.len().tolist()

    # Srednie wartosci cech dla klas Human vs AI w train
    stats_rows = []
    for f in FEATURE_NAMES:
        h_val = float(X_train.loc[y_train == 0, f].mean())
        a_val = float(X_train.loc[y_train == 1, f].mean())
        stats_rows.append({"feature": f, "human_mean": h_val, "ai_mean": a_val, "diff": a_val - h_val})
    stats_df = pd.DataFrame(stats_rows)

    # 3. Model A: Progi decyzyjne
    print("\n2. Trening Modelu A (Progi / Reguly Decyzyjne)...")
    clf_thresh = RuleBasedThresholdClassifier(max_depth=2, random_state=42)
    clf_thresh.fit(X_train, y_train)
    rules_text = clf_thresh.export_rules()
    y_pred_thresh = clf_thresh.predict(X_test)
    metrics_thresh = calculate_metrics(y_test, y_pred_thresh)
    print("   Zasady progowe:")
    for line in rules_text.strip().split("\n"):
        print(f"     {line}")
    print(f"   Wyniki progow (test): Acc={metrics_thresh['accuracy']:.4f}, F1={metrics_thresh['f1']:.4f}")

    # 4. Model B: Regresja Logistyczna
    print("\n3. Trening Modelu B (Regresja Logistyczna ze standaryzacja)...")
    clf_lr = StatisticalLogisticClassifier(C=1.0, random_state=42)
    clf_lr.fit(X_train, y_train)

    y_pred_lr = clf_lr.predict(X_test)
    y_proba_lr = clf_lr.predict_proba(X_test)[:, 1]
    metrics_lr = calculate_metrics(y_test, y_pred_lr)
    auc_lr = float(roc_auc_score(y_test, y_proba_lr))
    metrics_lr["roc_auc"] = auc_lr

    print(f"   Wyniki Regresji Logistycznej (test):")
    print(f"     Accuracy:  {metrics_lr['accuracy']:.5f}")
    print(f"     Precision: {metrics_lr['precision']:.5f}")
    print(f"     Recall:    {metrics_lr['recall']:.5f}")
    print(f"     F1-score:  {metrics_lr['f1']:.5f}")
    print(f"     ROC-AUC:   {auc_lr:.5f}")
    print(f"     Confusion Matrix: {metrics_lr['confusion_matrix']}")

    # Wagi cech
    weights_df = clf_lr.get_feature_importances()
    print("\n4. Wagi cech w regresji logistycznej:")
    for _, row in weights_df.iterrows():
        print(f"     {row['feature']:24} waga={row['weight']:+8.4f} -> faworyzuje {row['direction']}")

    # 5. Analiza wrazliwosci na dlugosc tekstu
    print("\n5. Analiza wrazliwosci na dlugosc tekstu...")
    buckets_lr = metrics_by_length_bucket(y_test, y_pred_lr, test_word_lengths, bins=4)
    length_auc = length_only_auc(y_test, test_word_lengths)

    # 6. Zapis do plikow
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    import joblib
    joblib.dump(clf_lr, OUTPUT_DIR / "model.joblib")
    print(f"Model zapisany do {OUTPUT_DIR / 'model.joblib'}")

    results = {
        "dataset_report": dedup_report,
        "feature_means": stats_rows,
        "threshold_model": {
            "rules": rules_text,
            "metrics": metrics_thresh,
        },
        "logistic_regression": {
            "metrics": metrics_lr,
            "feature_weights": weights_df.to_dict(orient="records"),
            "length_buckets": buckets_lr,
            "length_only_auc": length_auc,
        },
    }

    with open(OUTPUT_DIR / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nWyniki zapisane do {OUTPUT_DIR / 'results.json'}")
    return results


if __name__ == "__main__":
    run()

