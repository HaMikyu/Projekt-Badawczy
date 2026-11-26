"""Dane do System One v5: tylko eseje, wiecej generatorow, uczciwy test na nieznanych rodzinach.

Baza: data/prepared_v4_full (v3 + IvyPanda, juz oczyszczone z dev / essays_2021 / Ghostbustera).
Nowe zrodla (data/external/, gitignore):
  - artnitolog/llm-generated-texts, podzbior "essay": 1000 tematow IvyPanda x 7 modeli + esej ludzki.
      Trening:   GPT-4 Turbo, GPT-4o, Llama 3 70B, Command R+.
      WYLACZONE z treningu (test nieznanych rodzin): Claude 3 Opus, YandexGPT 3 Pro, GigaChat Pro.
      Tematy dzielone 70/15/15 - testowe tematy nie wystepuja w treningu dla zadnego modelu.
  - chiikabu-labs/LLMEssay-100K (3 z 10 shardow): eseje AI z 2026 r., ~1000 slow, bez markdownu.
  - ramensoft/daigt_v4: eseje PERSUADE od Llam dostrojonych do stylu uczniow (trudne pozytywy).
  - dodatkowe ludzkie eseje IvyPanda (nieuzyte w v4) dobrane dlugoscia do nowych esejow AI.

Kontrola przecieku (> 20 wspolnych 8-gramow lub identyczny tekst) wzgledem:
dev, essays_2021, Ghostbuster (train+test), benchmarkow zewnetrznych (WasItAI, TextSight),
oraz kazdy nowy tekst testowy wzgledem calego treningu i walidacji.

Wynik:
  data/prepared_v5_full/{train,val,test,ood}.csv    pelne teksty
  data/prepared_v5/{train,val,test,ood}.csv         okna 1-3 fragmentow (jak w aplikacji)
  data/prepared_v5_tests/*.csv                      testy na poziomie eseju (ocena kodem aplikacji)
"""
from collections import Counter, defaultdict
from pathlib import Path
import json
import random
import re
import sys

import pandas as pd
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "app"))

from train.build_data_v4 import (  # noqa: E402
    DEV, FINAL_TEST_DIR, IVY_HUMAN, MIN_WORDS, SEED, norm, strip_ai, title_of, windows,
)
from train.build_dev_set import shingles, strip_human  # noqa: E402
from train.helpers import normalize_text, word_count  # noqa: E402

V4_FULL = ROOT / "data" / "prepared_v4_full"
EXT = ROOT / "data" / "external"
OUT_FULL = ROOT / "data" / "prepared_v5_full"
OUT_WIN = ROOT / "data" / "prepared_v5"
OUT_TESTS = ROOT / "data" / "prepared_v5_tests"

ART_TRAIN = ["GPT4 Turbo 2024-04-09", "GPT4 Omni", "Llama3 70B", "Command R+"]
ART_HELDOUT = ["Claude 3 Opus", "YandexGPT 3 Pro", "GigaChat Pro"]
LLMESSAY_N = {"train": 6000, "val": 750, "test": 1000}
DAIGT4_FRAC = {"train": 0.8, "val": 0.1, "test": 0.1}
MAX_SHARED = 20
_MD_HEAD = re.compile(r"^\s*#{1,6}\s+")
_MD_BULLET = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")
_MD_EMPH = re.compile(r"(\*\*|__|\*(?=\S)|(?<=\S)\*)")


def clean_md(text):
    """Usuwa markdown (naglowki, wypunktowania, pogrubienia) i linie 'Title:'. Aplikacja dostaje zwykly tekst."""
    out = []
    for line in strip_ai(str(text)).splitlines():
        if _MD_HEAD.match(line):
            continue
        line = _MD_BULLET.sub("", line)
        line = _MD_EMPH.sub("", line).strip()
        if line in ("---", "***"):
            continue
        out.append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()


def split_ids(ids, seed, fracs=(0.7, 0.15, 0.15)):
    ids = sorted(ids)
    random.Random(seed).shuffle(ids)
    n = len(ids)
    a, b = int(n * fracs[0]), int(n * (fracs[0] + fracs[1]))
    return {"train": set(ids[:a]), "val": set(ids[a:b]), "test": set(ids[b:])}


def load_artnitolog():
    a = pd.read_parquet(EXT / "artnitolog" / "train.parquet")
    a = a[a.dataset_name == "essay"].copy()
    parts = split_ids(a["id"].unique(), SEED)
    rows = []
    for rec in a.to_dict("records"):
        split = next(s for s, ids in parts.items() if rec["id"] in ids)
        rows.append({"text": strip_human(rec["human"]), "generated": 0, "source": "artnitolog_human",
                     "split": split, "family": "human", "group": f"art{rec['id']}"})
        for m in ART_TRAIN + ART_HELDOUT:
            text = rec[m]
            if not isinstance(text, str) or not text.strip():
                continue
            heldout = m in ART_HELDOUT
            if heldout and split != "test":
                continue   # rodziny wylaczone: tylko tematy testowe, nigdy w treningu
            rows.append({"text": clean_md(text), "generated": 1,
                         "source": "artnitolog_" + re.sub(r"[^a-z0-9]+", "_", m.lower()).strip("_"),
                         "split": "test_heldout" if heldout else split, "family": m, "group": f"art{rec['id']}"})
    return pd.DataFrame(rows)


def load_llmessay():
    d = pd.concat([pd.read_parquet(p) for p in sorted((EXT / "llmessay100k").glob("*.parquet"))],
                  ignore_index=True).sample(frac=1.0, random_state=SEED)
    d["text"] = d["body"].map(clean_md)
    d = d[d["text"].map(word_count) >= MIN_WORDS].drop_duplicates("topic")
    rows, start = [], 0
    for split, n in LLMESSAY_N.items():
        part = d.iloc[start:start + n]
        start += n
        rows += [{"text": t, "generated": 1, "source": "llmessay100k", "split": split,
                  "family": "llmessay", "group": f"llm{i}"} for i, t in zip(part["id"], part["text"])]
    return pd.DataFrame(rows)


def load_daigt4():
    d = pd.read_csv(EXT / "daigt_v4" / "daigt_magic_generations.csv").dropna(subset=["text"])
    d = d.drop_duplicates("text").sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    n = len(d)
    a, b = int(n * DAIGT4_FRAC["train"]), int(n * (DAIGT4_FRAC["train"] + DAIGT4_FRAC["val"]))
    split = ["train"] * a + ["val"] * (b - a) + ["test"] * (n - b)
    return pd.DataFrame({"text": d["text"].map(clean_md), "generated": 1, "source": "daigt_v4_llama_ft",
                         "split": split, "family": "llama_ft", "group": [f"dg{i}" for i in range(n)]})


def extra_ivypanda(ai_new, used_keys, banned_titles, n_needed):
    """Ludzkie eseje IvyPanda nieuzyte w v4, w przedzialach dlugosci nowych esejow AI."""
    h = pd.read_parquet(IVY_HUMAN)
    h["title_n"] = h["TEXT"].map(title_of)
    h = h[~h["title_n"].isin(banned_titles)].copy()
    h["text"] = h["TEXT"].map(strip_human)
    h = h[~normalize_text(h["text"]).isin(used_keys)]
    h["w"] = h["text"].map(word_count)
    h = h[h["w"] >= MIN_WORDS].drop_duplicates("text").drop_duplicates("title_n")
    w_ai = ai_new["text"].map(word_count)
    edges = sorted(set(w_ai.quantile([i / 14 for i in range(15)]).round().astype(int)))
    edges[0], edges[-1] = 0, 10 ** 9
    need = pd.cut(w_ai, edges, include_lowest=True).value_counts(normalize=True)
    h["bin"] = pd.cut(h["w"], edges, include_lowest=True)
    picked = [h[h["bin"] == b].sample(frac=1.0, random_state=SEED).head(int(round(frac * n_needed * 1.2)))
              for b, frac in need.items()]
    h = pd.concat(picked).sample(frac=1.0, random_state=SEED)
    parts = split_ids(h["title_n"].unique(), SEED + 1, (0.8, 0.1, 0.1))
    split = h["title_n"].map(lambda t: next(s for s, ids in parts.items() if t in ids))
    return pd.DataFrame({"text": h["text"].values, "generated": 0, "source": "ivypanda_human_v5",
                         "split": split.values, "family": "human", "group": h["title_n"].values})


class ShingleIndex:
    """Odwrotny indeks 8-gramow; max liczba wspolnych 8-gramow z dowolnym tekstem indeksu."""

    def __init__(self, texts):
        self.idx = defaultdict(list)
        for i, t in enumerate(texts):
            for s in shingles(t):
                self.idx[s].append(i)

    def max_overlap(self, text):
        c = Counter()
        for s in shingles(text):
            for i in self.idx.get(s, ()):
                c[i] += 1
        return max(c.values(), default=0)


def external_texts():
    w = pd.read_parquet(EXT / "wasitai" / "benchmark.parquet")
    texts = list(w["text"].dropna())
    for f in (EXT / "textsight").glob("*.jsonl"):
        texts += [json.loads(line)["text"] for line in f.open(encoding="utf-8")]
    return texts


def main():
    base = {s: pd.read_csv(V4_FULL / f"{s}.csv") for s in ("train", "val", "test")}
    ood = pd.read_csv(V4_FULL / "ood.csv")
    used_keys = set().union(*(set(normalize_text(df["text"])) for df in base.values()))

    dev = pd.read_csv(DEV)
    final_titles = set(pd.read_csv(FINAL_TEST_DIR / "index.csv")["title"].map(norm))
    ivy = pd.read_parquet(IVY_HUMAN, columns=["TEXT"])
    dev_h = set(dev.loc[dev.generated == 0, "text"].map(norm))
    dev_titles = set(ivy.loc[ivy["TEXT"].map(lambda t: norm(strip_human(t))).isin(dev_h), "TEXT"].map(title_of))

    art, llm, dg = load_artnitolog(), load_llmessay(), load_daigt4()
    ai_new = pd.concat([art[art.generated == 1], llm, dg], ignore_index=True)
    n_ai_trainable = int((ai_new.split != "test_heldout").sum())
    n_h_new = int((art.generated == 0).sum())
    hum = extra_ivypanda(ai_new[ai_new.split != "test_heldout"], used_keys, dev_titles | final_titles,
                         max(n_ai_trainable - n_h_new, 0))
    pool = pd.concat([art, llm, dg, hum], ignore_index=True)
    pool = pool[pool["text"].map(word_count) >= MIN_WORDS].reset_index(drop=True)
    print("pula przed filtrami:", pool.groupby(["split", "source"]).size().to_dict(), flush=True)

    # --- filtry przecieku do zbiorow ewaluacyjnych
    print("indeksowanie 8-gramow (dev, essays_2021, Ghostbuster, benchmarki)...", flush=True)
    gb = pd.concat([pd.read_parquet(ROOT / "data" / "ghostbuster" / "data" / f)
                    for f in ("train-00000-of-00001.parquet", "test-00000-of-00001.parquet")])
    guards = {
        "dev": ShingleIndex(dev["text"].astype(str)),
        "essays_2021": ShingleIndex([p.read_text(encoding="utf-8") for p in FINAL_TEST_DIR.glob("*.txt")]),
        "ghostbuster": ShingleIndex(gb["text"].astype(str)),
        "external": ShingleIndex(external_texts()),
    }
    removed = {}
    keep = pd.Series(True, index=pool.index)
    for name, g in guards.items():
        hit = pool["text"].map(lambda t: g.max_overlap(t) > MAX_SHARED)
        removed[name] = pool.loc[hit & keep, "source"].value_counts().to_dict()
        keep &= ~hit
    hit = normalize_text(pool["text"]).isin(used_keys) | normalize_text(pool["text"]).duplicated()
    removed["identical_to_v4_or_dup"] = pool.loc[hit & keep, "source"].value_counts().to_dict()
    keep &= ~hit
    pool = pool[keep].reset_index(drop=True)
    print("usuniete przez filtry:", removed, flush=True)

    # --- nowe teksty testowe nie moga pokrywac sie z treningiem/walidacja (baza v4 + nowe)
    is_test = pool.split.isin(["test", "test_heldout"])
    trainval_texts = pd.concat([base["train"].text, base["val"].text, pool.loc[~is_test, "text"]])
    tidx = ShingleIndex(pool.loc[is_test, "text"])
    worst = Counter()
    test_ids = list(pool.index[is_test])
    for t in trainval_texts.astype(str):
        c = Counter()
        for s in shingles(t):
            for i in tidx.idx.get(s, ()):
                c[i] += 1
        for i, k in c.items():
            worst[i] = max(worst[i], k)
    leaked = [test_ids[i] for i, k in worst.items() if k > MAX_SHARED]
    removed["test_vs_trainval"] = pool.loc[leaked, "source"].value_counts().to_dict()
    pool = pool.drop(index=leaked).reset_index(drop=True)
    print("testowe usuniete (pokrycie z treningiem):", removed["test_vs_trainval"], flush=True)

    cols = ["text", "generated", "source"]
    full = {s: pd.concat([base[s], pool.loc[pool.split == s, cols]], ignore_index=True)
            .sample(frac=1.0, random_state=SEED).reset_index(drop=True) for s in ("train", "val", "test")}

    # --- testy na poziomie eseju (kod aplikacji); tylko nowe zrodla, nigdy w treningu
    OUT_TESTS.mkdir(parents=True, exist_ok=True)
    t = pool[pool.split == "test"]
    tests = {
        "heldout_families": pd.concat([pool[pool.split == "test_heldout"], t[t.source == "artnitolog_human"]]),
        "artnitolog_train_families": t[t.source.str.startswith("artnitolog")],
        "llmessay": pd.concat([t[t.source == "llmessay100k"], t[t.source == "ivypanda_human_v5"]]),
        "daigt_v4": t[t.source == "daigt_v4_llama_ft"],
    }
    for name, df in tests.items():
        df[["text", "generated", "source", "family"]].to_csv(OUT_TESTS / f"{name}.csv", index=False)

    tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    win = {s: windows(full[s], tok, SEED + i).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
           for i, s in enumerate(("train", "val", "test"))}
    for out, data in ((OUT_FULL, full), (OUT_WIN, win)):
        out.mkdir(parents=True, exist_ok=True)
        for s, df in data.items():
            df.to_csv(out / f"{s}.csv", index=False)
        ood.to_csv(out / "ood.csv", index=False)

    keys = {s: set(normalize_text(full[s]["text"])) for s in full}
    summary = {
        "new_by_split_source": {f"{s}|{src}": int(n) for (s, src), n in pool.groupby(["split", "source"]).size().items()},
        "full": {s: {"n": len(full[s]), "ai": int(full[s].generated.sum())} for s in full},
        "windows": {s: {"n": len(win[s]), "ai": int(win[s].generated.sum())} for s in win},
        "essay_tests": {k: {"n": len(v), "ai": int(v.generated.sum()),
                            "families": v.family.value_counts().to_dict()} for k, v in tests.items()},
        "removed": removed,
        "leak_check": {"train&val": len(keys["train"] & keys["val"]), "train&test": len(keys["train"] & keys["test"])},
        "heldout_families": ART_HELDOUT,
    }
    for out in (OUT_FULL, OUT_WIN, OUT_TESTS):
        (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
