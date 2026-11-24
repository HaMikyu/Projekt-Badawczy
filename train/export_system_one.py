from pathlib import Path
import json
import shutil
import sys

import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train.system_one_model import SystemOneOptionScorer

RUN_DIR = ROOT / "train" / "runs" / "system-one-v3"
OUT = ROOT / "models" / "system-one-final"


def main():
    meta = json.loads((RUN_DIR / "export_meta.json").read_text(encoding="utf-8"))
    ckpt = Path(meta["checkpoint"])
    model = SystemOneOptionScorer.load_from_checkpoint(str(ckpt), map_location="cpu")
    model.eval()
    model.temperature = float(meta["temperature"])
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "hparams": dict(model.hparams),
            "temperature": model.temperature,
            "options": meta["options"],
            "threshold": meta["threshold"],
            "max_length": meta["max_length"],
            "model_name": meta["model_name"],
        },
        OUT / "system_one.pt",
    )
    tokenizer = AutoTokenizer.from_pretrained(meta["model_name"])
    tokenizer.save_pretrained(OUT)
    (OUT / "export_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(OUT), **meta}, indent=2))


if __name__ == "__main__":
    main()
