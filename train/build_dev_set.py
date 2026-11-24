"""Zbior deweloperski do strojenia detektora (NIE finalny test).

Ludzkie: losowe eseje IvyPanda (qwedsacf/ivypanda-essays, zrzut z 01.2023).
AI: eseje na te same tematy IvyPanda (artfultom/ivypanda-llm-generated-essays:
ChatGPT 4.1, DeepSeek V3.2, Llama 3, Mistral 7B).

essays_2021/ (StudyCorgi) to finalny zbior testowy - wszystko, co sie z nim
pokrywa (tytul albo 8-gramy), jest tu usuwane, zeby nie przeciekl do strojenia.
"""
from pathlib import Path
import json
import random
import re
import time
import urllib.parse
import urllib.request

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "dev" / "dev.csv"
FINAL_TEST_DIR = ROOT / "essays_2021"
API = "https://datasets-server.huggingface.co/rows"
SEED = 7
N_HUMAN = 300
N_AI = 300


def fetch(dataset, config, offset, length):
    q = urllib.parse.urlencode({"dataset": dataset, "config": config, "split": "train",
                                "offset": offset, "length": length})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(f"{API}?{q}", timeout=60) as r:
                return [x["row"] for x in json.load(r)["rows"]]
        except Exception:
            time.sleep(2 + attempt * 3)
    return []


def strip_human(text):
    lines = [l.rstrip() for l in str(text).strip().splitlines()]
    if lines and len(lines[0].split()) < 20 and not lines[0].endswith("."):
        lines = lines[1:]  # tytul
    out = []
    for l in lines:
        if re.match(r"^\s*(references|works cited|bibliography|reference list)\s*$", l, re.I):
            break
        out.append(l)
    return "\n".join(out).strip()


def shingles(text, n=8):
    w = re.findall(r"[a-z0-9']+", text.lower())
    return {" ".join(w[i:i + n]) for i in range(0, max(0, len(w) - n + 1))}


def main():
    rng = random.Random(SEED)
    final_texts = [p.read_text(encoding="utf-8") for p in FINAL_TEST_DIR.glob("*.txt")]
    final_sh = set().union(*(shingles(t) for t in final_texts))
    final_titles = set(pd.read_csv(FINAL_TEST_DIR / "index.csv")["title"].str.lower().str.strip())

    rows = []
    for off in rng.sample(range(0, 128000, 10), N_HUMAN // 5):
        for r in fetch("qwedsacf/ivypanda-essays", "default", off, 5):
            rows.append({"text": strip_human(r["TEXT"]), "generated": 0, "source": "ivypanda_human",
                         "title": str(r["TEXT"]).strip().splitlines()[0]})
    for cfg, n in [("one prompt", int(N_AI * 0.75)), ("two prompts", N_AI - int(N_AI * 0.75))]:
        total = 22492 if cfg == "one prompt" else 20000
        for off in rng.sample(range(0, total - 5), n // 3):
            for r in fetch("artfultom/ivypanda-llm-generated-essays", cfg, off, 3)[:3]:
                rows.append({"text": str(r["text"]).strip(), "generated": 1,
                             "source": "ivypanda_ai_" + str(r.get("model", "?")).split("-")[0].split(".")[0],
                             "title": str(r.get("title", ""))})

    df = pd.DataFrame(rows).drop_duplicates("text")
    df = df[df["text"].str.split().str.len() >= 150]
    overlap = df.apply(lambda r: r["title"].lower().strip() in final_titles
                       or len(shingles(r["text"]) & final_sh) > 20, axis=1)
    print("usunieto (pokrycie z essays_2021):", int(overlap.sum()))
    df = df[~overlap]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df[["text", "generated", "source"]].to_csv(OUT, index=False)
    print(df.groupby("source").size().to_string())


if __name__ == "__main__":
    main()
