from pathlib import Path
import json
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train.helpers import Detector, metrics_dict

OOD = ROOT / "data" / "prepared" / "ood.csv"
OLD = ROOT / "models" / "distilbert-final"
NEW_CKPT_DIR = ROOT / "train" / "runs" / "distilbert-v3"
OUT = NEW_CKPT_DIR / "compare_ood.json"


def main():
    ood = pd.read_csv(OOD)
    texts = ood["text"].tolist()
    labels = ood["generated"].tolist()

    old = Detector(OLD, max_length=256)
    old_preds = old.predict(texts, batch_size=16)
    old_m = metrics_dict(labels, old_preds)
    old_m["name"] = "baseline_kaggle_only"

    # v3 is evaluated in train_v2 already; also score exported later
    print("baseline_on_ghostbuster", json.dumps(old_m, indent=2))
    OUT.write_text(json.dumps({"baseline_on_ghostbuster_ood": old_m}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
