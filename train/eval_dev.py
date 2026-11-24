"""Ocena kandydata System One na zbiorze dev (data/dev/dev.csv) tym samym kodem co aplikacja.

    python train/eval_dev.py                          # model z models/system-one-final
    python train/eval_dev.py models/system-one-v4     # inny katalog z system_one.pt

Wypisuje odsetek ludzkich esejow oznaczonych jako AI i odsetek wykrytych AI
dla kilku progow sredniej z fragmentow oraz rozbicie na zrodla.
NIE uzywa essays_2021 (to finalny test).
"""
from pathlib import Path
import json
import sys
import time

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
sys.path.insert(0, str(ROOT))

from helpers import MAX_CHUNKS, score_system_one, split_chunks  # noqa: E402
from train.predict_system_one import load_detector  # noqa: E402

DEV = ROOT / "data" / "dev" / "dev.csv"


def main():
    model_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "models" / "system-one-final"
    model, tok, meta, device = load_detector(model_dir)
    so = {"model": model, "tokenizer": tok, "meta": meta, "device": device}
    dev = pd.read_csv(DEV)
    t0 = time.time()
    means = []
    for text in dev["text"].astype(str):
        chunks = split_chunks(text, tok)[:MAX_CHUNKS] or [text]
        probs = score_system_one(chunks, so)
        means.append(sum(probs) / len(probs))
    dev["p_ai"] = means
    human = dev[dev.generated == 0]
    ai = dev[dev.generated == 1]
    report = {"model_dir": str(model_dir), "n_human": len(human), "n_ai": len(ai),
              "seconds": round(time.time() - t0), "thresholds": {}}
    for th in (0.3, 0.5, 0.6, 0.7):
        report["thresholds"][str(th)] = {
            "human_as_ai": round(float((human.p_ai >= th).mean()), 4),
            "ai_detected": round(float((ai.p_ai >= th).mean()), 4),
        }
    report["by_source_at_0.5"] = {
        src: round(float(((g.p_ai >= 0.5) == (g.generated == 1)).mean()), 4)
        for src, g in dev.groupby("source")
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
