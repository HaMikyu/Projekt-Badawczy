import json
import os
import re
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# DETEKTOR_ROOT: katalog z models/ i statistical/runs/ (ustawia go app/desktop.py w spakowanym EXE)
ROOT = Path(os.environ.get("DETEKTOR_ROOT", Path(__file__).resolve().parents[1]))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train.predict_system_one import load_detector
from statistical.features import (
    FEATURE_NAMES,
    extract_stylometric_features,
    split_sentences,
    tokenize_words,
)

STAT_MODEL_PATH = ROOT / "statistical" / "runs" / "model.joblib"
STAT_RESULTS_PATH = ROOT / "statistical" / "runs" / "results.json"
ML_MODEL_DIR = ROOT / "models" / "distilbert-final"
SYSTEM_ONE_DIR = Path(os.environ.get("DETEKTOR_SYSTEM_ONE_DIR", ROOT / "models" / "system-one-final"))

# Ustawienia System One dobrane na zbiorze deweloperskim (data/dev, IvyPanda),
# NIE na essays_2021 - ten folder jest wylacznie finalnym testem.
CHUNK_TOKENS = 320      # docelowa dlugosc fragmentu (tokeny, wyrownane do zdan)
MAX_CHUNKS = 24         # limit fragmentow na tekst
AI_THRESHOLD = 0.50     # srednie p_ai >= prog -> AI (dev: 3,4% ludzi jako AI, 84% AI wykryte)
HUMAN_THRESHOLD = 0.30  # srednie p_ai < prog -> CZLOWIEK, pomiedzy -> NIEPEWNE
_SENT_RE = re.compile(r"(?<=[.!?])\s+")

FEATURE_LABELS = {
    "avg_sentence_len": "Srednia dlugosc zdania (slowa)",
    "std_sentence_len": "Odchylenie dlugosci zdan",
    "burstiness_cv": "Burstiness (CV)",
    "burstiness_diff": "Burstiness (roznica sasiednich zdan)",
    "ttr": "Type-Token Ratio (TTR)",
    "root_ttr": "Indeks Guirauda (Root-TTR)",
    "stopword_ratio": "Udzial stop-words",
    "avg_word_len": "Srednia dlugosc slowa (znaki)",
    "punct_ratio": "Udzial interpunkcji",
    "comma_per_sentence": "Przecinki na zdanie",
    "exclaim_question_ratio": "Udzial ! oraz ?",
    "uppercase_ratio": "Udzial wielkich liter",
    "digit_ratio": "Udzial cyfr",
}


def load_reference_means():
    if not STAT_RESULTS_PATH.is_file():
        return {}
    data = json.loads(STAT_RESULTS_PATH.read_text(encoding="utf-8"))
    means = {}
    for row in data.get("feature_means", []):
        means[row["feature"]] = {
            "human": float(row["human_mean"]),
            "ai": float(row["ai_mean"]),
        }
    return means


def load_models():
    warnings.filterwarnings("ignore")
    stat_model = None
    if STAT_MODEL_PATH.is_file():
        stat_model = joblib.load(STAT_MODEL_PATH)
        clf = stat_model.pipeline.named_steps["clf"]
        if not hasattr(clf, "multi_class"):
            clf.multi_class = "auto"

    tokenizer = None
    ml_model = None
    if ML_MODEL_DIR.is_dir():
        tokenizer = AutoTokenizer.from_pretrained(str(ML_MODEL_DIR))
        ml_model = AutoModelForSequenceClassification.from_pretrained(str(ML_MODEL_DIR))
        ml_model.eval()

    system_one = None
    if (SYSTEM_ONE_DIR / "system_one.pt").is_file():
        so_model, so_tok, so_meta, so_dev = load_detector(SYSTEM_ONE_DIR)
        system_one = {"model": so_model, "tokenizer": so_tok, "meta": so_meta, "device": so_dev}

    return {
        "system_one": system_one,
        "stat_model": stat_model,
        "tokenizer": tokenizer,
        "ml_model": ml_model,
        "reference_means": load_reference_means(),
    }


def predict_statistical(text, stat_model):
    feats = extract_stylometric_features(text)
    if stat_model is None:
        return None, feats, []

    df = pd.DataFrame([feats], columns=FEATURE_NAMES)
    proba = stat_model.predict_proba(df)[0]
    human_prob = float(proba[0])
    ai_prob = float(proba[1])
    pred = 1 if ai_prob >= 0.5 else 0

    scaler = stat_model.pipeline.named_steps["scaler"]
    clf = stat_model.pipeline.named_steps["clf"]
    scaled = scaler.transform(df.values)[0]
    weights = clf.coef_[0]

    impacts = []
    for i, name in enumerate(FEATURE_NAMES):
        impact = float(scaled[i] * weights[i])
        impacts.append({
            "feature": name,
            "label": FEATURE_LABELS.get(name, name),
            "value": float(feats[name]),
            "impact": impact,
            "direction": "AI" if impact >= 0 else "HUMAN",
        })
    impacts.sort(key=lambda item: abs(item["impact"]), reverse=True)

    result = {
        "label": "AI" if pred == 1 else "HUMAN",
        "ai_percent": round(ai_prob * 100, 2),
        "human_percent": round(human_prob * 100, 2),
        "confidence": round(max(ai_prob, human_prob) * 100, 2),
    }
    return result, feats, impacts


def count_tokens(tokenizer, text):
    ids = tokenizer.encode(text, add_special_tokens=True, truncation=True, max_length=100000)
    return len(ids)


def predict_ml(text, tokenizer, ml_model):
    if tokenizer is None or ml_model is None:
        return None

    token_count = count_tokens(tokenizer, text)
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=256,
        return_token_type_ids=False,
    )
    with torch.no_grad():
        logits = ml_model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0].tolist()

    human_prob = float(probs[0])
    ai_prob = float(probs[1])
    pred = 1 if ai_prob >= 0.5 else 0

    return {
        "label": "AI" if pred == 1 else "HUMAN",
        "ai_percent": round(ai_prob * 100, 2),
        "human_percent": round(human_prob * 100, 2),
        "confidence": round(max(ai_prob, human_prob) * 100, 2),
        "token_count": token_count,
        "truncated": token_count > 256,
    }


def split_chunks(text, tokenizer, target=CHUNK_TOKENS):
    """Dzieli tekst na fragmenty ~target tokenow, nie tnac zdan."""
    sents = [s for p in text.splitlines() for s in _SENT_RE.split(p.strip()) if s.strip()]
    chunks, cur, n = [], [], 0
    for s in sents:
        k = len(tokenizer.encode(s, add_special_tokens=False))
        if cur and n + k > target:
            chunks.append(" ".join(cur))
            cur, n = [], 0
        cur.append(s)
        n += k
    if cur:
        if chunks and n < target * 0.4:
            chunks[-1] += " " + " ".join(cur)
        else:
            chunks.append(" ".join(cur))
    return chunks


@torch.no_grad()
def score_system_one(texts, so):
    model, tok, meta = so["model"], so["tokenizer"], so["meta"]
    options, max_len = meta["options"], meta["max_length"]
    out = []
    for i in range(0, len(texts), 8):
        batch = texts[i:i + 8]
        encs = [tok(batch, [o] * len(batch), truncation="only_first", max_length=max_len) for o in options]
        width = max(len(x) for e in encs for x in e["input_ids"])
        ids = torch.zeros(len(batch), 2, width, dtype=torch.long)
        mask = torch.zeros(len(batch), 2, width, dtype=torch.long)
        for j, e in enumerate(encs):
            for r, x in enumerate(e["input_ids"]):
                ids[r, j, :len(x)] = torch.tensor(x)
                mask[r, j, :len(x)] = 1
        logits = model(ids.to(so["device"]), mask.to(so["device"]))
        out += torch.softmax(logits, dim=-1)[:, 1].cpu().tolist()
    return out


def verdict(p_ai):
    if p_ai >= AI_THRESHOLD:
        return "AI"
    if p_ai < HUMAN_THRESHOLD:
        return "HUMAN"
    return "UNSURE"


def predict_system_one(text, so):
    if so is None:
        return None
    chunks = split_chunks(text, so["tokenizer"])[:MAX_CHUNKS] or [text]
    probs = score_system_one(chunks, so)
    p_ai = float(sum(probs) / len(probs))
    label = verdict(p_ai)
    return {
        "label": label,
        "ai_percent": round(p_ai * 100, 2),
        "human_percent": round((1 - p_ai) * 100, 2),
        "confidence": round(max(p_ai, 1 - p_ai) * 100, 2),
        "chunks_total": len(chunks),
        "chunks_ai": int(sum(p >= 0.5 for p in probs)),
        "segments": [
            {"index": i + 1, "text": c, "ai_percent": round(p * 100, 1),
             "label": "AI" if p >= 0.5 else "HUMAN"}
            for i, (c, p) in enumerate(zip(chunks, probs))
        ],
    }


def compute_basic_stats(text):
    clean = text.strip()
    sentences = split_sentences(clean)
    words = tokenize_words(clean)
    paragraphs = [p.strip() for p in clean.splitlines() if p.strip()]
    sentence_lengths = [len(tokenize_words(s)) for s in sentences]
    sentence_lengths = [n for n in sentence_lengths if n > 0]

    word_count = len(words)
    unique_words = len(set(words))
    char_count = len(clean)
    char_no_spaces = len("".join(clean.split()))
    reading_seconds = round(word_count / 3.5)

    return {
        "characters": char_count,
        "characters_no_spaces": char_no_spaces,
        "words": word_count,
        "unique_words": unique_words,
        "sentences": len(sentence_lengths),
        "paragraphs": len(paragraphs),
        "reading_time_sec": reading_seconds,
        "min_sentence_len": int(min(sentence_lengths)) if sentence_lengths else 0,
        "max_sentence_len": int(max(sentence_lengths)) if sentence_lengths else 0,
        "median_sentence_len": round(float(np.median(sentence_lengths)), 1) if sentence_lengths else 0.0,
        "sentence_lengths": sentence_lengths[:80],
    }


def build_feature_table(feats, reference_means):
    rows = []
    for name in FEATURE_NAMES:
        val = float(feats[name])
        ref = reference_means.get(name, {})
        human_mean = ref.get("human")
        ai_mean = ref.get("ai")
        closer = None
        if human_mean is not None and ai_mean is not None:
            closer = "AI" if abs(val - ai_mean) < abs(val - human_mean) else "HUMAN"
        rows.append({
            "feature": name,
            "label": FEATURE_LABELS.get(name, name),
            "value": val,
            "human_mean": human_mean,
            "ai_mean": ai_mean,
            "closer_to": closer,
        })
    return rows


def analyze_text(text, models):
    clean = text.strip()
    basic = compute_basic_stats(clean)
    stat_result, feats, impacts = predict_statistical(clean, models["stat_model"])
    ml_result = predict_ml(clean, models["tokenizer"], models["ml_model"])

    so_result = predict_system_one(clean, models.get("system_one"))

    if so_result is not None:
        combined = {k: so_result[k] for k in ("label", "ai_percent", "human_percent", "confidence")}
        combined["source"] = "System One (srednia z fragmentow)"
    else:
        ai_scores = [r["ai_percent"] for r in (ml_result, stat_result) if r is not None]
        combined_ai = round(sum(ai_scores) / len(ai_scores), 2) if ai_scores else 0.0
        combined = {
            "label": verdict(combined_ai / 100),
            "ai_percent": combined_ai,
            "human_percent": round(100.0 - combined_ai, 2),
            "confidence": round(max(combined_ai, 100.0 - combined_ai), 2),
            "source": "srednia modeli pomocniczych (brak System One)",
        }

    warnings_list = []
    if basic["words"] < 150:
        warnings_list.append("Tekst ma mniej niz 150 slow - werdykt opiera sie na jednym krotkim fragmencie i jest mniej pewny.")
    if so_result and so_result["chunks_total"] > 1 and 0 < so_result["chunks_ai"] < so_result["chunks_total"]:
        warnings_list.append(
            f"Fragmenty oceniane jako AI: {so_result['chunks_ai']} z {so_result['chunks_total']}. "
            "Pojedyncze fragmenty (zwlaszcza wstep i zakonczenie) bywaja oznaczane jako AI takze w tekstach ludzkich - liczy sie srednia."
        )
    if basic["words"] < 50:
        warnings_list.append("Tekst jest krotki (ponizej 50 slow) - statystyki i predykcja moga byc mniej stabilne.")
    if ml_result and ml_result["truncated"]:
        warnings_list.append(
            f"Pomocniczy DistilBERT widzi tylko pierwsze 256 tokenow (tekst ma {ml_result['token_count']})."
        )

    return {
        "combined": combined,
        "system_one": so_result,
        "ml": ml_result,
        "statistical": stat_result,
        "basic_stats": basic,
        "features": build_feature_table(feats, models["reference_means"]),
        "impacts": impacts[:8],
        "warnings": warnings_list,
    }
