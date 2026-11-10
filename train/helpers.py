from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def normalize_text(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().str.replace(r"\s+", " ", regex=True).str.strip()


def word_count(text: str) -> int:
    return len(str(text).split())


def metrics_dict(y_true, y_pred) -> dict:
    y_true = np.asarray(list(y_true))
    y_pred = np.asarray(list(y_pred))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "n": int(len(y_true)),
    }


def load_kaggle_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path).dropna(subset=["text", "generated"]).copy()
    frame["text"] = frame["text"].astype(str).str.strip()
    frame["generated"] = frame["generated"].astype(int)
    frame = frame[frame["text"].str.len() > 0]
    frame = frame[frame["generated"].isin([0, 1])]
    frame["source"] = "kaggle_essays"
    return frame.reset_index(drop=True)


def load_ghostbuster(train_path: Path, test_path: Path | None = None) -> pd.DataFrame:
    frames = [pd.read_parquet(train_path)]
    if test_path is not None and Path(test_path).is_file():
        frames.append(pd.read_parquet(test_path))
    frame = pd.concat(frames, ignore_index=True)
    # ClassLabel: claude,gpt,gpt_prompt1,gpt_prompt2,gpt_semantic,gpt_writing,human
    label_names = {
        0: "claude",
        1: "gpt",
        2: "gpt_prompt1",
        3: "gpt_prompt2",
        4: "gpt_semantic",
        5: "gpt_writing",
        6: "human",
    }
    frame = frame.dropna(subset=["text", "label"]).copy()
    frame["text"] = frame["text"].astype(str).str.strip()
    frame["label_name"] = frame["label"].map(label_names).fillna("unknown")
    frame["generated"] = (frame["label"] != 6).astype(int)
    frame["source"] = "ghostbuster_" + frame["label_name"].astype(str)
    return frame[["text", "generated", "source"]].reset_index(drop=True)


def load_pile_sample(
    human_path: Path,
    ai_path: Path,
    n_per_class: int = 6000,
    min_words: int = 50,
    max_words: int = 700,
    seed: int = 42,
) -> pd.DataFrame:
    human = pd.read_parquet(human_path)
    ai = pd.read_parquet(ai_path)
    human = human[human["source"] == "human"].copy()
    ai = ai[ai["source"] == "ai"].copy()
    for frame in (human, ai):
        frame["text"] = frame["text"].astype(str).str.strip()
        frame["_words"] = frame["text"].map(word_count)
        frame.query(f"{min_words} <= _words <= {max_words}", inplace=True)
    human = human.sample(n=min(n_per_class, len(human)), random_state=seed)
    ai = ai.sample(n=min(n_per_class, len(ai)), random_state=seed)
    human["generated"] = 0
    ai["generated"] = 1
    human["source"] = "pile_human"
    ai["source"] = "pile_ai"
    out = pd.concat([human[["text", "generated", "source"]], ai[["text", "generated", "source"]]])
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def load_hc3_jsonl(path: Path, min_words: int = 30) -> pd.DataFrame:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            source = item.get("source", "hc3")
            for answer in item.get("human_answers") or []:
                text = str(answer).strip()
                if word_count(text) >= min_words:
                    rows.append({"text": text, "generated": 0, "source": f"hc3_{source}"})
            for answer in item.get("chatgpt_answers") or []:
                text = str(answer).strip()
                if word_count(text) >= min_words:
                    rows.append({"text": text, "generated": 1, "source": f"hc3_{source}"})
    return pd.DataFrame(rows)


def dedupe_exact(frame: pd.DataFrame, min_words: int = 5) -> pd.DataFrame:
    working = frame.copy()
    working["_norm"] = normalize_text(working["text"])
    working["_words"] = working["text"].map(word_count)
    working = working[working["_words"] >= min_words]
    conflict = working.groupby("_norm")["generated"].nunique()
    bad = set(conflict[conflict > 1].index)
    working = working[~working["_norm"].isin(bad)]
    working = working.drop_duplicates("_norm", keep="first")
    return working.drop(columns=["_norm", "_words"]).reset_index(drop=True)


def length_match(frame: pd.DataFrame, seed: int = 42, bins: int = 10) -> pd.DataFrame:
    working = frame.copy()
    working["_words"] = working["text"].map(word_count)
    human = working[working["generated"] == 0]
    ai = working[working["generated"] == 1]
    if human.empty or ai.empty:
        return frame.reset_index(drop=True)
    edges = np.unique(np.quantile(working["_words"], np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        n = min(len(human), len(ai))
        return pd.concat([
            human.sample(n=n, random_state=seed),
            ai.sample(n=n, random_state=seed),
        ]).drop(columns=["_words"]).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    working["_bin"] = pd.cut(working["_words"], bins=edges, include_lowest=True, duplicates="drop")
    parts = []
    rng = np.random.default_rng(seed)
    for bucket, group in working.groupby("_bin", observed=True):
        h = group[group["generated"] == 0]
        a = group[group["generated"] == 1]
        n = min(len(h), len(a))
        if n == 0:
            continue
        h_idx = rng.choice(h.index.to_numpy(), size=n, replace=False)
        a_idx = rng.choice(a.index.to_numpy(), size=n, replace=False)
        parts.append(working.loc[h_idx])
        parts.append(working.loc[a_idx])
    if not parts:
        n = min(len(human), len(ai))
        out = pd.concat([
            human.sample(n=n, random_state=seed),
            ai.sample(n=n, random_state=seed),
        ])
    else:
        out = pd.concat(parts)
    return out.drop(columns=["_words", "_bin"]).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def stratified_split(frame: pd.DataFrame, seed: int, val_ratio: float = 0.1, test_ratio: float = 0.1):
    train_parts, val_parts, test_parts = [], [], []
    for _, group in frame.groupby("generated", sort=True):
        group = group.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        n_test = max(1, int(round(len(group) * test_ratio)))
        n_val = max(1, int(round(len(group) * val_ratio)))
        test_parts.append(group.iloc[:n_test])
        val_parts.append(group.iloc[n_test:n_test + n_val])
        train_parts.append(group.iloc[n_test + n_val:])
    train = pd.concat(train_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    val = pd.concat(val_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    test = pd.concat(test_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return train, val, test


def sample_balanced(frame: pd.DataFrame, n_per_class: int, seed: int = 42) -> pd.DataFrame:
    parts = []
    for label in (0, 1):
        group = frame[frame["generated"] == label]
        take = min(n_per_class, len(group))
        parts.append(group.sample(n=take, random_state=seed))
    return pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)


class Detector:
    def __init__(self, model_dir: Path, max_length: int = 256, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict_proba(self, texts: list[str], batch_size: int = 16) -> np.ndarray:
        probs = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            logits = self.model(**enc).logits
            p = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            probs.append(p)
        return np.concatenate(probs) if probs else np.array([])

    def predict(self, texts: list[str], batch_size: int = 16, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(texts, batch_size) >= threshold).astype(int)
