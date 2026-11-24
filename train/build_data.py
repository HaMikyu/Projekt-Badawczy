from pathlib import Path
import json
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train.helpers import (
    dedupe_exact,
    length_match,
    load_argugpt,
    load_daigt_v3,
    load_ghostbuster,
    load_kaggle_csv,
    load_pile_sample,
    stratified_split,
    word_count,
)

KAGGLE = ROOT / "data" / "Training_Essay_Data.csv"
DAIGT = ROOT / "data" / "daigt_v3" / "train.parquet"
ARGUGPT = ROOT / "data" / "argugpt" / "all.parquet"
GB_TRAIN = ROOT / "data" / "ghostbuster" / "data" / "train-00000-of-00001.parquet"
GB_TEST = ROOT / "data" / "ghostbuster" / "data" / "test-00000-of-00001.parquet"
PILE_HUMAN = ROOT / "data" / "pile" / "data" / "train-00000-of-00007-bc5952582e004d67.parquet"
PILE_AI = ROOT / "data" / "pile" / "data" / "train-00005-of-00007-3dce5e05ddbad789.parquet"
OUT_DIR = ROOT / "data" / "prepared"
SEED = 42


def formal_human_from_pile(n: int = 4000) -> pd.DataFrame:
    pile = load_pile_sample(PILE_HUMAN, PILE_AI, n_per_class=n, min_words=120, max_words=650, seed=SEED)
    human = pile[pile["generated"] == 0].copy()
    human["source"] = "pile_formal_human"
    return human[["text", "generated", "source"]].reset_index(drop=True)


def main():
    kaggle = load_kaggle_csv(KAGGLE)
    kaggle["source"] = "kaggle_essays"
    daigt = load_daigt_v3(DAIGT, min_words=50)
    argugpt = load_argugpt(ARGUGPT, min_words=50)
    gb_train = load_ghostbuster(GB_TRAIN, None)
    gb_test = load_ghostbuster(GB_TEST, None)

    hard_human = gb_train[gb_train["generated"] == 0]
    hard_human = pd.concat([hard_human] * 10, ignore_index=True)
    formal = formal_human_from_pile(5000)

    train_pool = pd.concat([kaggle, daigt, argugpt, gb_train, hard_human, formal], ignore_index=True)
    train_pool = train_pool[train_pool["text"].map(word_count) >= 40]
    train_pool = dedupe_exact(train_pool, min_words=40)
    train_pool = length_match(train_pool, seed=SEED, bins=14)

    ood = dedupe_exact(gb_test, min_words=30)
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
        "note": "essays + GB train humans x10 + pile formal humans; OOD = GB test",
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
