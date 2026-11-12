from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import PreTrainedTokenizerBase
except ImportError:
    torch = None
    DataLoader = None
    Dataset = object
    PreTrainedTokenizerBase = Any


LABEL_NAMES = ("human", "ai")


def normalize_text(series: pd.Series) -> pd.Series:
    """Lowercase and collapse whitespace so duplicates are comparable across splits."""
    return series.astype(str).str.lower().str.replace(r"\s+", " ", regex=True).str.strip()


def deduplicate_frame(
    frame: pd.DataFrame,
    min_words: int = 5,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Remove noisy rows and exact duplicates before splitting.

    Duplicate texts are removed from the *whole* dataset first, so the same text
    can never appear in both train and validation/test. Identical texts with
    conflicting labels are dropped because their label is ambiguous.
    """
    working = frame.copy()
    working["_normalized"] = normalize_text(working["text"])
    working["_words"] = working["text"].str.split().str.len()
    raw_rows = len(working)

    short = working["_words"] < min_words
    working = working[~short]

    label_counts = working.groupby("_normalized")["generated"].nunique()
    conflicting_keys = set(label_counts[label_counts > 1].index)
    ambiguous = working["_normalized"].isin(conflicting_keys)
    working = working[~ambiguous]

    duplicate_groups = int((working["_normalized"].value_counts() > 1).sum())
    duplicated = working.duplicated(subset="_normalized", keep="first")
    working = working[~duplicated]

    report = {
        "raw_rows": raw_rows,
        "dropped_short_rows": int(short.sum()),
        "dropped_conflicting_label_rows": int(ambiguous.sum()),
        "dropped_duplicate_rows": int(duplicated.sum()),
        "duplicate_groups_removed": duplicate_groups,
        "clean_rows": len(working),
    }
    clean = working.drop(columns=["_normalized", "_words"]).reset_index(drop=True)
    return clean, report


def split_overlap(train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame) -> dict[str, int]:
    """Count identical normalized texts shared between splits."""
    train_keys = set(normalize_text(train["text"]))
    validation_keys = set(normalize_text(validation["text"]))
    test_keys = set(normalize_text(test["text"]))
    return {
        "train_validation": len(train_keys & validation_keys),
        "train_test": len(train_keys & test_keys),
        "validation_test": len(validation_keys & test_keys),
    }


class TokenizedTextDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
    ) -> None:
        encoded = tokenizer(
            frame["text"].tolist(),
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        self.encodings = encoded
        self.labels = torch.tensor(frame["generated"].tolist(), dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = {key: value[index] for key, value in self.encodings.items()}
        item["labels"] = self.labels[index]
        return item


@dataclass(frozen=True)
class DataSplits:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def load_dataframe(path: Path, limit: int | None = None, seed: int = 42) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")
    frame = pd.read_csv(path).dropna(subset=["text", "generated"]).copy()
    required = {"text", "generated"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")
    frame["text"] = frame["text"].astype(str).str.strip()
    frame["generated"] = frame["generated"].astype(int)
    frame = frame[frame["text"].str.len() > 0]
    invalid = set(frame["generated"].unique()).difference({0, 1})
    if invalid:
        raise ValueError(f"Labels must be 0 or 1, got: {sorted(invalid)}")
    if limit is not None:
        if limit < 2:
            raise ValueError("limit must be at least 2 when provided")
        sampled = []
        for _, group in frame.groupby("generated", sort=True):
            count = max(1, round(limit * len(group) / len(frame)))
            sampled.append(group.sample(n=min(count, len(group)), random_state=seed))
        frame = pd.concat(sampled).sample(frac=1.0, random_state=seed).head(limit)
    return frame.reset_index(drop=True)


def split_dataframe(
    frame: pd.DataFrame,
    seed: int,
    validation_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> DataSplits:
    if validation_ratio <= 0 or test_ratio <= 0 or validation_ratio + test_ratio >= 1:
        raise ValueError("validation_ratio and test_ratio must be positive and sum below 1")
    shuffled = frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    test_count = max(1, int(len(shuffled) * test_ratio))
    validation_count = max(1, int(len(shuffled) * validation_ratio))
    test_parts = []
    validation_parts = []
    train_parts = []
    for _, group in shuffled.groupby("generated", sort=True):
        group = group.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        group_test = max(1, round(len(group) * test_ratio))
        group_validation = max(1, round(len(group) * validation_ratio))
        test_parts.append(group.iloc[:group_test])
        validation_parts.append(group.iloc[group_test : group_test + group_validation])
        train_parts.append(group.iloc[group_test + group_validation :])
    test = pd.concat(test_parts, ignore_index=True).sample(frac=1.0, random_state=seed)
    validation = pd.concat(validation_parts, ignore_index=True).sample(frac=1.0, random_state=seed)
    train = pd.concat(train_parts, ignore_index=True).sample(frac=1.0, random_state=seed)
    if train.empty:
        raise ValueError("Dataset is too small for the requested split")
    splits = DataSplits(
        train.reset_index(drop=True),
        validation.reset_index(drop=True),
        test.reset_index(drop=True),
    )
    overlaps = split_overlap(splits.train, splits.validation, splits.test)
    if any(overlaps.values()):
        raise ValueError(
            "Identical texts found in more than one split "
            f"({overlaps}). Run deduplicate_frame before split_dataframe."
        )
    return splits


def create_dataloaders(
    splits: DataSplits,
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int,
    max_length: int,
    pin_memory: bool = False,
    num_workers: int = 0,
) -> dict[str, DataLoader[dict[str, torch.Tensor]]]:
    datasets = {
        "train": TokenizedTextDataset(splits.train, tokenizer, max_length),
        "validation": TokenizedTextDataset(splits.validation, tokenizer, max_length),
        "test": TokenizedTextDataset(splits.test, tokenizer, max_length),
    }
    common = {"batch_size": batch_size, "num_workers": num_workers, "pin_memory": pin_memory}
    return {
        "train": DataLoader(datasets["train"], shuffle=True, **common),
        "validation": DataLoader(datasets["validation"], **common),
        "test": DataLoader(datasets["test"], **common),
    }


def split_summary(splits: DataSplits) -> dict[str, Any]:
    return {
        name: {"rows": len(frame), "human": int((frame["generated"] == 0).sum()), "ai": int((frame["generated"] == 1).sum())}
        for name, frame in (("train", splits.train), ("validation", splits.validation), ("test", splits.test))
    }
