from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train.helpers import Detector, load_hc3_jsonl, load_kaggle_csv, metrics_dict, sample_balanced, dedupe_exact


MODEL_DIR = ROOT / "models" / "distilbert-final"
KAGGLE = ROOT / "data" / "Training_Essay_Data.csv"
HC3 = ROOT / "data" / "hc3" / "all.jsonl"
OUT = ROOT / "train" / "runs" / "ood_baseline" / "metrics.json"
SEED = 42
N_PER_CLASS = 800
MAX_LENGTH = 256
BATCH = 8


def eval_frame(detector, frame, name):
    preds = detector.predict(frame["text"].tolist(), batch_size=BATCH)
    metrics = metrics_dict(frame["generated"].tolist(), preds)
    metrics["name"] = name
    metrics["sources"] = frame["source"].value_counts().to_dict() if "source" in frame.columns else {}
    print(name, json.dumps({k: metrics[k] for k in ("accuracy", "precision", "recall", "f1", "n", "confusion_matrix")}))
    return metrics


def main():
    detector = Detector(MODEL_DIR, max_length=MAX_LENGTH)
    results = []

    kaggle = dedupe_exact(load_kaggle_csv(KAGGLE))
    kaggle_sample = sample_balanced(kaggle, N_PER_CLASS, SEED)
    results.append(eval_frame(detector, kaggle_sample, "kaggle_essays_sample"))

    hc3 = dedupe_exact(load_hc3_jsonl(HC3))
    hc3_sample = sample_balanced(hc3, N_PER_CLASS, SEED)
    results.append(eval_frame(detector, hc3_sample, "hc3_ood"))

    for source, group in hc3.groupby("source"):
        if len(group[group.generated == 0]) < 50 or len(group[group.generated == 1]) < 50:
            continue
        part = sample_balanced(group, min(200, len(group) // 2), SEED)
        results.append(eval_frame(detector, part, f"hc3_domain_{source}"))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("saved", OUT)


if __name__ == "__main__":
    main()
