from pathlib import Path
import json
import sys

import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train.system_one_model import OPTIONS, SystemOneOptionScorer

MODEL_DIR = ROOT / "models" / "system-one-final"


def load_detector(model_dir: Path = MODEL_DIR):
    bundle = torch.load(model_dir / "system_one.pt", map_location="cpu", weights_only=False)
    hp = {
        k: bundle["hparams"][k]
        for k in (
            "model_name",
            "learning_rate",
            "weight_decay",
            "warmup_ratio",
            "class_weights",
            "label_smoothing",
            "temperature",
        )
        if k in bundle["hparams"]
    }
    if (model_dir / "config.json").exists():  # konfiguracja enkodera zapisana przy modelu -> dziala offline
        hp["model_name"] = str(model_dir)
    model = SystemOneOptionScorer(**hp, pretrained=False)
    model.load_state_dict(bundle["state_dict"])  # wagi fp16 sa rzutowane do fp32 modelu
    model.temperature = float(bundle.get("temperature", 1.0))
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    meta = {
        "options": bundle.get("options", OPTIONS),
        "threshold": float(bundle.get("threshold", 0.5)),
        "max_length": int(bundle.get("max_length", 384)),
    }
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return model, tokenizer, meta, device


@torch.no_grad()
def score_texts(model, tokenizer, texts, meta, device):
    options = meta["options"]
    max_length = meta["max_length"]
    pair_a = tokenizer(
        texts,
        [options[0]] * len(texts),
        padding="max_length",
        truncation="only_first",
        max_length=max_length,
        return_tensors="pt",
    )
    pair_b = tokenizer(
        texts,
        [options[1]] * len(texts),
        padding="max_length",
        truncation="only_first",
        max_length=max_length,
        return_tensors="pt",
    )
    input_ids = torch.stack([pair_a["input_ids"], pair_b["input_ids"]], dim=1).to(device)
    attention_mask = torch.stack([pair_a["attention_mask"], pair_b["attention_mask"]], dim=1).to(device)
    logits = model(input_ids, attention_mask)
    probs = torch.softmax(logits, dim=-1)
    return probs[:, 1].detach().cpu().tolist()


def main():
    if len(sys.argv) < 2:
        path = ROOT / "naturalny_essay.txt"
        text = path.read_text(encoding="utf-8")
    else:
        arg = sys.argv[1]
        p = Path(arg)
        text = p.read_text(encoding="utf-8") if p.is_file() else arg

    model, tokenizer, meta, device = load_detector()
    p_ai = score_texts(model, tokenizer, [text], meta, device)[0]
    label = "ai" if p_ai >= meta["threshold"] else "human"
    print(json.dumps({"label": label, "proba_ai": p_ai, "threshold": meta["threshold"]}, indent=2))


if __name__ == "__main__":
    main()
