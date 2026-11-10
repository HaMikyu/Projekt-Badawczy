from pathlib import Path
import json
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train.helpers import (
    dedupe_exact,
    length_match,
    load_ghostbuster,
    load_hc3_jsonl,
    load_kaggle_csv,
    load_pile_sample,
    stratified_split,
)

KAGGLE = ROOT / "data" / "Training_Essay_Data.csv"
HC3 = ROOT / "data" / "hc3" / "all.jsonl"
GB_TRAIN = ROOT / "data" / "ghostbuster" / "data" / "train-00000-of-00001.parquet"
GB_TEST = ROOT / "data" / "ghostbuster" / "data" / "test-00000-of-00001.parquet"
PILE_HUMAN = ROOT / "data" / "pile" / "data" / "train-00000-of-00007-bc5952582e004d67.parquet"
PILE_AI = ROOT / "data" / "pile" / "data" / "train-00005-of-00007-3dce5e05ddbad789.parquet"
OUT_DIR = ROOT / "data" / "prepared"
SEED = 42


def main():
    kaggle = load_kaggle_csv(KAGGLE)
    kaggle["source"] = "kaggle_essays"
    hc3 = load_hc3_jsonl(HC3, min_words=30)
    pile = load_pile_sample(PILE_HUMAN, PILE_AI, n_per_class=7000, seed=SEED)
    ghost = load_ghostbuster(GB_TRAIN, GB_TEST)

    train_pool = pd.concat([kaggle, hc3, pile], ignore_index=True)
    train_pool = dedupe_exact(train_pool, min_words=20)
    train_pool = length_match(train_pool, seed=SEED)

    ood = dedupe_exact(ghost, min_words=30)
    ood = length_match(ood, seed=SEED)

    train, val, test = stratified_split(train_pool, seed=SEED, val_ratio=0.1, test_ratio=0.1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train.to_csv(OUT_DIR / "train.csv", index=False)
    val.to_csv(OUT_DIR / "val.csv", index=False)
    test.to_csv(OUT_DIR / "test.csv", index=False)
    ood.to_csv(OUT_DIR / "ood.csv", index=False)

    summary = {
        "train": len(train),
        "val": len(val),
        "test": len(test),
        "ood": len(ood),
        "train_ai": int((train.generated == 1).sum()),
        "train_human": int((train.generated == 0).sum()),
        "ood_sources": sorted(ood["source"].unique().tolist()),
        "train_sources": train["source"].value_counts().to_dict(),
        "ood_balance": ood["generated"].value_counts().to_dict(),
        "note": "OOD = Ghostbuster (Claude/GPT variants), never seen in training",
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
