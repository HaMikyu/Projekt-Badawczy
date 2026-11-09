from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from train.model import BertTextClassifier
    from train.train import DEFAULT_RUNS, ROOT, display_path
else:
    from .model import BertTextClassifier
    from .train import DEFAULT_RUNS, ROOT, display_path


DEFAULT_MODELS = ROOT / "models"
LABELS = {0: "human", 1: "ai"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export a Lightning checkpoint to a portable Hugging Face model directory"
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=DEFAULT_RUNS / "distilbert-final",
        help="Run directory containing config.json and checkpoints/best.ckpt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Destination directory (default: models/<run-name>)",
    )
    parser.add_argument(
        "--half",
        action="store_true",
        help="Store weights as float16 (smaller, GPU inference only)",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    config_path = run_dir / "config.json"
    checkpoint_path = run_dir / "checkpoints" / "best.ckpt"
    if not config_path.is_file():
        raise SystemExit(f"Missing run config: {display_path(config_path)}")
    if not checkpoint_path.is_file():
        raise SystemExit(f"Missing checkpoint: {display_path(checkpoint_path)}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    model = BertTextClassifier.load_from_checkpoint(
        str(checkpoint_path),
        map_location="cpu",
    )
    model.eval()
    model.model.config.id2label = dict(LABELS)
    model.model.config.label2id = {label: index for index, label in LABELS.items()}
    if args.half:
        model.model.half()

    output_dir = (args.output_dir or DEFAULT_MODELS / run_dir.name).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model.model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    tokenizer.save_pretrained(output_dir)

    files = sorted(
        display_path(path)
        for path in output_dir.iterdir()
        if path.is_file()
    )
    print(json.dumps({
        "source_run": display_path(run_dir),
        "output_dir": display_path(output_dir),
        "model_name": config["model_name"],
        "dtype": "float16" if args.half else "float32",
        "files": files,
    }, indent=2))


if __name__ == "__main__":
    main()
