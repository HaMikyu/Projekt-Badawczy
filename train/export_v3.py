from pathlib import Path
import json
import sys

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train.model import BertTextClassifier

RUN_DIR = ROOT / "train" / "runs" / "distilbert-v3"
CKPT = RUN_DIR / "checkpoints" / "best.ckpt"
OUT = ROOT / "models" / "distilbert-final"
MODEL_NAME = "distilbert-base-uncased"
LABELS = {0: "human", 1: "ai"}


def main():
    model = BertTextClassifier.load_from_checkpoint(str(CKPT), map_location="cpu")
    model.eval()
    model.model.config.id2label = dict(LABELS)
    model.model.config.label2id = {v: k for k, v in LABELS.items()}
    OUT.mkdir(parents=True, exist_ok=True)
    model.model.save_pretrained(OUT, safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.save_pretrained(OUT)
    meta = {
        "source_run": "train/runs/distilbert-v3",
        "model_name": MODEL_NAME,
        "threshold": 0.5,
        "labels": LABELS,
    }
    (OUT / "export_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(OUT), **meta}, indent=2))


if __name__ == "__main__":
    main()
