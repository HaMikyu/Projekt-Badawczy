"""Ekstrakcja cech stylometrycznych tekstu dla detekcji AI vs Human.

Modul implementuje cechy zalecane w stylometrii i analizie autorstwa:
1. Srednia i odchylenie standardowe dlugosci zdan (sentence length)
2. Burstiness (wspolczynnik zmiennosci dlugosci zdan oraz srednia roznica sasiednich zdan)
3. Type-Token Ratio (TTR) oraz Root-TTR (Guiraud R)
4. Udzial stop-words (slow funkcyjnych)
5. Srednia dlugosc slowa
6. Wskazniki interpunkcji (przecinki na zdanie, interpunkcja ogolna, wykrzykniki/pytajniki)
7. Udzial wielkich liter i cyfr
"""

from __future__ import annotations

import re
from typing import Sequence

import numpy as np
import pandas as pd


STOPWORDS_EN = frozenset({
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can", "can't", "cannot", "could",
    "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down",
    "during", "each", "few", "for", "from", "further", "had", "hadn't", "has",
    "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her",
    "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's",
    "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it",
    "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my",
    "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other",
    "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shan't",
    "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then",
    "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've",
    "this", "those", "through", "to", "too", "under", "until", "up", "very", "was",
    "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren't", "what",
    "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's",
    "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd",
    "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves",
})

FEATURE_NAMES = [
    "avg_sentence_len",
    "std_sentence_len",
    "burstiness_cv",
    "burstiness_diff",
    "ttr",
    "root_ttr",
    "stopword_ratio",
    "avg_word_len",
    "punct_ratio",
    "comma_per_sentence",
    "exclaim_question_ratio",
    "uppercase_ratio",
    "digit_ratio",
]

_SENTENCE_SPLIT_REGEX = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")
_WORD_REGEX = re.compile(r"\b[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?\b")


def split_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    parts = _SENTENCE_SPLIT_REGEX.split(text)
    if len(parts) <= 1:
        # Fallback jesli brak wielkich liter po kropce
        parts = re.split(r"[.!?]+\s*", text)
    sentences = [p.strip() for p in parts if p.strip()]
    return sentences or [text]


def tokenize_words(text: str) -> list[str]:
    return [w.lower() for w in _WORD_REGEX.findall(text)]


def extract_stylometric_features(text: str) -> dict[str, float]:
    """Ekstrahuje slownik cech stylometrycznych z pojedynczego tekstu."""
    text_clean = text.strip()
    if not text_clean:
        return {name: 0.0 for name in FEATURE_NAMES}

    sentences = split_sentences(text_clean)
    words = tokenize_words(text_clean)

    n_chars = max(len(text_clean), 1)
    n_words = max(len(words), 1)
    n_sents = max(len(sentences), 1)

    # 1. Dlugosc zdan w slowach
    sentence_word_counts = [len(tokenize_words(s)) for s in sentences]
    sentence_word_counts = [c for c in sentence_word_counts if c > 0] or [n_words]
    
    avg_sentence_len = float(np.mean(sentence_word_counts))
    std_sentence_len = float(np.std(sentence_word_counts))

    # 2. Burstiness:
    # a) CV (coefficient of variation = std / mean)
    burstiness_cv = float(std_sentence_len / avg_sentence_len) if avg_sentence_len > 0 else 0.0
    # b) srednia roznica bezwzgledna pomiedzy kolejnymi zdaniami
    if len(sentence_word_counts) > 1:
        diffs = [
            abs(sentence_word_counts[i] - sentence_word_counts[i - 1])
            for i in range(1, len(sentence_word_counts))
        ]
        burstiness_diff = float(np.mean(diffs))
    else:
        burstiness_diff = 0.0

    # 3. Bogactwo slownikowe (TTR i Root-TTR)
    unique_words_count = len(set(words))
    ttr = float(unique_words_count / n_words)
    root_ttr = float(unique_words_count / np.sqrt(n_words))

    # 4. Udzial stop-words
    stopwords_count = sum(1 for w in words if w in STOPWORDS_EN)
    stopword_ratio = float(stopwords_count / n_words)

    # 5. Srednia dlugosc slowa
    word_lengths = [len(w) for w in words]
    avg_word_len = float(np.mean(word_lengths)) if word_lengths else 0.0

    # 6. Wskazniki interpunkcji
    punct_count = sum(1 for c in text_clean if c in ".,;:!?\"'()-[]{}")
    punct_ratio = float(punct_count / n_chars)
    comma_count = text_clean.count(",")
    comma_per_sentence = float(comma_count / n_sents)
    exclaim_question_count = sum(1 for c in text_clean if c in "!?")
    exclaim_question_ratio = float(exclaim_question_count / n_chars)

    # 7. Wielkie litery i cyfry
    uppercase_count = sum(1 for c in text_clean if c.isupper())
    uppercase_ratio = float(uppercase_count / n_chars)
    digit_count = sum(1 for c in text_clean if c.isdigit())
    digit_ratio = float(digit_count / n_chars)

    return {
        "avg_sentence_len": avg_sentence_len,
        "std_sentence_len": std_sentence_len,
        "burstiness_cv": burstiness_cv,
        "burstiness_diff": burstiness_diff,
        "ttr": ttr,
        "root_ttr": root_ttr,
        "stopword_ratio": stopword_ratio,
        "avg_word_len": avg_word_len,
        "punct_ratio": punct_ratio,
        "comma_per_sentence": comma_per_sentence,
        "exclaim_question_ratio": exclaim_question_ratio,
        "uppercase_ratio": uppercase_ratio,
        "digit_ratio": digit_ratio,
    }


def extract_features_df(texts: Sequence[str]) -> pd.DataFrame:
    """Ekstrahuje cechy stylometryczne dla calej sekwencji tekstow do DataFrame."""
    rows = [extract_stylometric_features(t) for t in texts]
    return pd.DataFrame(rows, columns=FEATURE_NAMES)

