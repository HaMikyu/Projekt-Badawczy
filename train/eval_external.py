"""Zewnetrzny test System One na gotowych benchmarkach detekcji AI (listopad 2026).

Zrodla (pobrane do data/external/, gitignore):
  - WasItAIGenerated AI-Text Detector Benchmark 2026-09
    (huggingface.co/datasets/wasitaigeneratedcom/ai-detector-benchmark-2026-09):
    teksty ludzkie z otwartych licencji, AI z ~40 modeli 2025-2026 (GPT-5/6,
    Claude 4/5, Gemini 3, DeepSeek V4, Qwen, Grok ...), parafrazy i humanizery,
    plus werdykty ZeroGPT / GPTZero / Pangram / tropa-3 na tych samych tekstach.
  - TextSight AI text detection benchmark 2026
    (huggingface.co/datasets/textsightai/ai-text-detection-benchmark-2026):
    teksty ludzkie (Wikipedia sprzed 2022, PMC autorow spoza krajow anglojezycznych,
    literatura) i 56 tekstow AI z 2026 r.

Zaden z tych zbiorow nie byl uzywany do treningu ani strojenia. Teksty majace
> 20 wspolnych 8-gramow z treningiem/walidacja v4 sa usuwane. Ocena tym samym
kodem co aplikacja (fragmenty, srednia p_ai, progi 0,5 / 0,3). Bez strojenia
czegokolwiek na wynikach.

    python train/eval_external.py v3=train/runs/system-one-v3/export v4=models/system-one-v4
"""
from collections import Counter, defaultdict
from pathlib import Path
import json
import re
import sys
import time

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "app"))

from helpers import predict_system_one  # noqa: E402
from train.predict_system_one import load_detector  # noqa: E402

EXT = ROOT / "data" / "external"
TRAIN_SETS = [ROOT / "data" / "prepared_v4_full" / f"{s}.csv" for s in ("train", "val")]
OUT_DIR = ROOT / "train" / "runs" / "external"
OUT_JSON = ROOT / "train" / "external_test_results.json"
MIN_WORDS = 80
NGRAM = 8
MAX_SHARED = 20
_WORD = re.compile(r"[a-z0-9']+")


def load_wasitai():
    d = pd.read_parquet(EXT / "wasitai" / "benchmark.parquet")
    d = d[d.text.notna() & (d.lang == "en") & d.label.isin(["human", "ai", "humanized"])].copy()
    kind = d.group.where(~d.group.str.startswith("humanized"), "humanized")
    kind = kind.where(~kind.str.startswith("human/"), "human")
    return pd.DataFrame({
        "dataset": "wasitai",
        "id": d.id,
        "kind": kind,                      # human / ai_raw / ai_paraphrase / humanized
        "group": d.group,
        "generator": d.generator.fillna(""),
        "text": d.text,
        "y": (d.label != "human").astype(int),
        "zerogpt": d.zerogpt_label,
        "gptzero": d.gptzero_label,
        "pangram": d.pangram_label,
        "tropa3": d.tropa3_display,
    })


def load_textsight():
    rows = []
    for f in sorted((EXT / "textsight").glob("*.jsonl")):
        for line in f.open(encoding="utf-8"):
            r = json.loads(line)
            ai = r["label"] == "ai"
            rows.append({
                "dataset": "textsight",
                "id": r["id"],
                "kind": "ai_raw" if ai else "human",
                "group": f"ai/{r.get('genre')}" if ai else f"human/{f.stem.replace('human_', '')}",
                "generator": r.get("source", "") if ai else "",
                "text": r["text"],
                "y": int(ai),
            })
    return pd.DataFrame(rows)


def shingles(text):
    w = _WORD.findall(str(text).lower())
    return {hash(tuple(w[i:i + NGRAM])) for i in range(len(w) - NGRAM + 1)}


def leak_filter(df):
    """Usuwa teksty z > MAX_SHARED wspolnymi 8-gramami z dowolnym tekstem treningu/walidacji v4."""
    index = defaultdict(list)
    for i, t in enumerate(df.text):
        for s in shingles(t):
            index[s].append(i)
    worst = [0] * len(df)
    for path in TRAIN_SETS:
        for t in pd.read_csv(path, usecols=["text"]).text:
            hits = Counter()
            for s in shingles(t):
                for i in index.get(s, ()):
                    hits[i] += 1
            for i, c in hits.items():
                worst[i] = max(worst[i], c)
    df = df.assign(max_train_overlap=worst)
    return df[df.max_train_overlap <= MAX_SHARED].copy(), df[df.max_train_overlap > MAX_SHARED]


def score(df, model_dir):
    model, tok, meta, dev = load_detector(Path(model_dir))
    so = {"model": model, "tokenizer": tok, "meta": meta, "device": dev}
    out = []
    t0 = time.time()
    for i, t in enumerate(df.text, 1):
        r = predict_system_one(t, so)
        out.append((r["ai_percent"] / 100, r["label"]))
        if i % 500 == 0:
            print(f"  {i}/{len(df)} ({time.time() - t0:.0f} s)", flush=True)
    return [p for p, _ in out], [lab for _, lab in out]


def rates(sub, col):
    n = len(sub)
    if n == 0:
        return None
    v = sub[col].value_counts()
    return {"n": n, "AI": round(v.get("AI", 0) / n, 4), "UNSURE": round(v.get("UNSURE", 0) / n, 4),
            "HUMAN": round(v.get("HUMAN", 0) / n, 4)}


def commercial(sub):
    """Odsetek werdyktow 'AI' detektorow komercyjnych na tych samych tekstach (gdzie ocenione)."""
    res = {}
    for col in ("gptzero", "pangram", "zerogpt"):
        s = sub[col].dropna() if col in sub else pd.Series(dtype=object)
        s = s[s != ""]
        if len(s):
            res[col] = {"n": int(len(s)), "AI": round(float((s == "ai").mean()), 4)}
    if "tropa3" in sub and sub.tropa3.notna().any():
        s = sub.tropa3.dropna()
        res["tropa3_ge90"] = {"n": int(len(s)), "AI": round(float((s >= 90).mean()), 4)}
    return res


def main():
    models = dict(a.split("=", 1) for a in sys.argv[1:]) or {
        "v3": "train/runs/system-one-v3/export", "v4": "models/system-one-v4"}
    df = pd.concat([load_wasitai(), load_textsight()], ignore_index=True)
    df["words"] = df.text.str.split().str.len()
    short = int((df.words < MIN_WORDS).sum())
    df = df[df.words >= MIN_WORDS].reset_index(drop=True)
    t0 = time.time()
    df, leaked = leak_filter(df)
    print(f"krotkie (<{MIN_WORDS} slow) usuniete: {short}; przeciek do treningu usuniety: {len(leaked)} "
          f"({leaked.group.value_counts().to_dict()}); kontrola {time.time() - t0:.0f} s", flush=True)
    df = df.reset_index(drop=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, path in models.items():
        print(f"[{name}] {path}", flush=True)
        df[f"p_{name}"], df[f"v_{name}"] = score(df, ROOT / path)
    df.drop(columns=["text"]).to_csv(OUT_DIR / "external_scores.csv", index=False)

    summary = {"n_total": len(df), "removed_short": short, "removed_leak": len(leaked),
               "min_words": MIN_WORDS, "models": models, "by_kind": {}, "by_group": {}, "by_vendor_ai_raw": {}}
    for (ds, kind), sub in df.groupby(["dataset", "kind"]):
        key = f"{ds}/{kind}"
        summary["by_kind"][key] = {m: rates(sub, f"v_{m}") for m in models}
        summary["by_kind"][key]["commercial"] = commercial(sub)
    for (ds, group), sub in df.groupby(["dataset", "group"]):
        summary["by_group"][f"{ds}/{group}"] = {m: rates(sub, f"v_{m}") for m in models}
    raw = df[df.kind == "ai_raw"].copy()
    raw["vendor"] = raw.generator.str.extract(r"^([a-z]+)", expand=False).fillna("?")
    for vendor, sub in raw.groupby("vendor"):
        summary["by_vendor_ai_raw"][vendor] = {m: rates(sub, f"v_{m}") for m in models}
    OUT_JSON.write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")

    for key, val in summary["by_kind"].items():
        line = " | ".join(f"{m}: AI {val[m]['AI']:.1%} UNS {val[m]['UNSURE']:.1%}" for m in models)
        print(f"{key:28s} n={val[next(iter(models))]['n']:4d} | {line} | {val['commercial']}")


if __name__ == "__main__":
    main()
