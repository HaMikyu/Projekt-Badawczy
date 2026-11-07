import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

KAGGLE_ID = "sunilthite/llm-detect-ai-generated-text-dataset"
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
CSV_NAME = "Training_Essay_Data.csv"

FEATURE_NAMES = [
    "avg_sentence_len",
    "std_sentence_len",
    "avg_word_len",
    "type_token_ratio",
    "punct_ratio",
    "upper_ratio",
    "digit_ratio",
    "stopword_ratio",
    "comma_per_sentence",
    "exclaim_question_ratio",
]

STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "is", "are", "was", "were", "be", "been", "being", "have", "has",
    "had", "do", "does", "did", "will", "would", "could", "should", "may",
    "might", "must", "shall", "can", "this", "that", "these", "those", "i",
    "you", "he", "she", "it", "we", "they", "my", "your", "his", "her",
    "its", "our", "their", "with", "from", "by", "as", "not", "no", "if",
}


def _sentences(text):
    parts = re.split(r"[.!?]+", text)
    return [p.strip() for p in parts if p.strip()]


def _words(text):
    return re.findall(r"[A-Za-z']+", text.lower())


def extract_features(text):
    text = text.strip()
    if not text:
        return np.zeros(len(FEATURE_NAMES), dtype=np.float32)

    sents = _sentences(text)
    words = _words(text)
    n_chars = max(len(text), 1)
    n_words = max(len(words), 1)
    n_sents = max(len(sents), 1)

    sent_lens = [len(_words(s)) for s in sents] or [0]
    avg_sent = float(np.mean(sent_lens))
    std_sent = float(np.std(sent_lens)) if len(sent_lens) > 1 else 0.0
    avg_word = float(np.mean([len(w) for w in words])) if words else 0.0
    ttr = len(set(words)) / n_words
    punct = sum(1 for c in text if c in ".,;:!?\"'()-") / n_chars
    upper = sum(1 for c in text if c.isupper()) / n_chars
    digit = sum(1 for c in text if c.isdigit()) / n_chars
    stop_r = sum(1 for w in words if w in STOPWORDS) / n_words
    commas = text.count(",") / n_sents
    eq = sum(1 for c in text if c in "!?") / n_chars

    feats = np.array(
        [avg_sent, std_sent, avg_word, ttr, punct, upper, digit, stop_r, commas, eq],
        dtype=np.float32,
    )
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    return feats


def ensure_kaggle_csv():
    dest = DATA_DIR / CSV_NAME
    if dest.exists():
        return dest
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    import kagglehub
    cache_dir = Path(kagglehub.dataset_download(KAGGLE_ID))
    src = next(cache_dir.rglob(CSV_NAME))
    shutil.copy2(src, dest)
    print(f"pobrano kaggle dataset -> {dest}")
    return dest


def load_kaggle_data(seed=42):
    csv_path = ensure_kaggle_csv()
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["text", "generated"]).copy()
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 0]
    df["generated"] = df["generated"].astype(int)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    texts = df["text"].tolist()
    labels = df["generated"].tolist()
    return texts, labels


if __name__ == "__main__":
    csv_path = ensure_kaggle_csv()
    full = pd.read_csv(csv_path)
    print("helpers smoke test")
    print(f"csv: {csv_path}")
    print(f"rows: {len(full)}")
    print(full["generated"].value_counts().to_string())
    print()

    human_ex = full[full["generated"] == 0]["text"].iloc[0]
    ai_ex = full[full["generated"] == 1]["text"].iloc[0]
    print("=== PRZYKLAD HUMAN (generated=0) ===")
    print(human_ex[:500].replace("\n", " "))
    print("...")
    print()
    print("=== PRZYKLAD AI (generated=1) ===")
    print(ai_ex[:500].replace("\n", " "))
    print("...")
    print()

    texts, labels = load_kaggle_data(seed=42)
    print(f"loaded: {len(texts)} human={labels.count(0)} ai={labels.count(1)}")
