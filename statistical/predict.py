"""Szybki skrypt do predykcji na wlasnym tekscie przy uzyciu wyuczonego modelu statystycznego.

Przyklad uzycia:
  python statistical/predict.py "Tutaj wklej tekst eseju..."
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from statistical.features import extract_stylometric_features
from statistical.model import StatisticalLogisticClassifier

MODEL_PATH = ROOT / "statistical" / "runs" / "model.joblib"


def main() -> None:
    parser = argparse.ArgumentParser(description="Klasyfikuj tekst (Human vs AI) sciezka statystyczna.")
    parser.add_argument("text", nargs="?", type=str, help="Tekst do sklasyfikowania")
    parser.add_argument("--file", type=Path, help="Sciezka do pliku tekstowego")
    args = parser.parse_args()

    if args.file:
        text = args.file.read_text(encoding="utf-8")
    elif args.text:
        text = args.text
    else:
        text = sys.stdin.read()

    text = text.strip()
    if not text:
        print("Blad: podano pusty tekst.")
        sys.exit(1)

    if not MODEL_PATH.exists():
        print("Model nie zostal jeszcze wytrenowany. Uruchom najpierw: python statistical/run_experiment.py")
        sys.exit(1)

    model: StatisticalLogisticClassifier = joblib.load(MODEL_PATH)

    feats_dict = extract_stylometric_features(text)
    feats_df = pd.DataFrame([feats_dict])

    pred = model.predict(feats_df)[0]
    proba = model.predict_proba(feats_df)[0]

    label = "AI" if pred == 1 else "HUMAN"
    confidence = proba[pred] * 100

    print("=== WYNIK DETEKCJI STATYSTYCZNEJ ===")
    print(f"Decyzja:         {label} (pewnosc: {confidence:.1f}%)")
    print(f"Prawdopodobienstwo AI:    {proba[1] * 100:.1f}%")
    print(f"Prawdopodobienstwo Human: {proba[0] * 100:.1f}%\n")
    print("Wybrane cechy tekstu:")
    print(f"  - Liczba slow:           {len(text.split())}")
    print(f"  - Srednia dlugosc zdan:  {feats_dict['avg_sentence_len']:.1f} slow")
    print(f"  - Burstiness (CV):       {feats_dict['burstiness_cv']:.3f}")
    print(f"  - Type-Token Ratio:      {feats_dict['ttr']:.3f}")
    print(f"  - Srednia dlugosc slowa: {feats_dict['avg_word_len']:.2f} znakow")
    print(f"  - Udzial stop-words:     {feats_dict['stopword_ratio'] * 100:.1f}%")


if __name__ == "__main__":
    main()
