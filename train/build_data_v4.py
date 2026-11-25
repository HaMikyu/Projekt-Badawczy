"""Dane do System One v4: pula v3 + eseje IvyPanda (ludzkie i AI na te same tematy).

Wynik:
  data/prepared_v4_full/  - pelne teksty (wariant bez okien; model widzi pierwsze 384 tokeny)
  data/prepared_v4/       - okna 1-3 fragmentow na esej, ten sam podzial na fragmenty co aplikacja
  ood.csv w obu katalogach = data/prepared/ood.csv (Ghostbuster test, nietkniety)

Zasady (train/GPU_TRAINING.md):
- dev (data/dev/dev.csv) nie przecieka: wykluczone sa jego teksty i tematy (tytuly),
- essays_2021 (finalny test) nie przecieka: wykluczone tytuly i teksty z >20 wspolnymi 8-gramami,
- podzialy v3 (data/prepared/{train,val,test}.csv) zostaja bez zmian, bo v4 startuje z wag v3:
  ponowne losowanie wpusciloby test v3 do treningu,
- nowe dane IvyPanda dzielone 80/10/10 po tematach (seed 42), klasy wyrownane dlugoscia.
"""
from pathlib import Path
import glob
import json
import random
import re
import sys

import pandas as pd
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "app"))

from helpers import CHUNK_TOKENS  # noqa: E402  (app/helpers.py)
from train.build_dev_set import shingles, strip_human  # noqa: E402
from train.helpers import length_match, normalize_text, word_count  # noqa: E402

V3_DIR = ROOT / "data" / "prepared"
DEV = ROOT / "data" / "dev" / "dev.csv"
FINAL_TEST_DIR = ROOT / "essays_2021"
IVY_HUMAN = ROOT / "data" / "ivypanda" / "ivypanda.parquet"
IVY_AI_GLOB = str(ROOT / "data" / "ivypanda_llm" / "*.csv")
OUT_FULL = ROOT / "data" / "prepared_v4_full"
OUT_WIN = ROOT / "data" / "prepared_v4"
SEED = 42
AI_PER_MODEL = 3000
MIN_WORDS = 150
MAX_WINDOWS = 3
_SENT_RE = re.compile(r"(?<=[.!?])\s+")
_TITLE_LINE = re.compile(r"^\s*\**\s*title\s*:", re.I)


def norm(t):
    return re.sub(r"\s+", " ", str(t)).strip().lower()


def title_of(text):
    return norm(str(text).strip().splitlines()[0])


def strip_ai(text):
    lines = str(text).strip().splitlines()
    if lines and _TITLE_LINE.match(lines[0]):
        lines = lines[1:]
    return "\n".join(lines).strip()


def model_family(name):
    return str(name).split("-")[0].split(".")[0].lower()


def load_ai():
    parts = []
    for f in sorted(glob.glob(IVY_AI_GLOB)):
        d = pd.read_csv(f, usecols=["title", "text", "model"])
        parts.append(d)
    ai = pd.concat(parts, ignore_index=True).dropna(subset=["text", "title"])
    ai["family"] = ai["model"].map(model_family)
    ai["title_n"] = ai["title"].map(norm)
    return ai


def excluded_titles(human, ai):
    dev = pd.read_csv(DEV)
    dev_h = set(dev.loc[dev.generated == 0, "text"].map(norm))
    dev_a = set(dev.loc[dev.generated == 1, "text"].map(norm))
    h_strip = human["TEXT"].map(lambda t: norm(strip_human(t)))
    dev_titles = set(human.loc[h_strip.isin(dev_h), "title_n"])
    dev_titles |= set(ai.loc[ai["text"].map(norm).isin(dev_a), "title_n"])
    final_titles = set(pd.read_csv(FINAL_TEST_DIR / "index.csv")["title"].map(norm))
    return dev_titles, final_titles, dev_h | dev_a


def grouped_split(frame, seed):
    titles = sorted(frame["title_n"].unique())
    random.Random(seed).shuffle(titles)
    n = len(titles)
    test_t = set(titles[: int(n * 0.1)])
    val_t = set(titles[int(n * 0.1): int(n * 0.2)])
    split = frame["title_n"].map(lambda t: "test" if t in test_t else "val" if t in val_t else "train")
    return {s: frame[split == s].reset_index(drop=True) for s in ("train", "val", "test")}


def split_sentences_chunks(text, tok, target=CHUNK_TOKENS):
    """Ten sam algorytm co app/helpers.split_chunks, z dlugosciami zdan liczonymi wsadowo."""
    sents = [s for p in str(text).splitlines() for s in _SENT_RE.split(p.strip()) if s.strip()]
    if not sents:
        return []
    lens = [len(x) for x in tok(sents, add_special_tokens=False)["input_ids"]]
    chunks, cur, n = [], [], 0
    for s, k in zip(sents, lens):
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


def windows(frame, tok, seed):
    """1-3 okna na esej: zawsze poczatek, dalej losowy srodek i koniec. Etykieta okna = etykieta eseju."""
    rng = random.Random(seed)
    rows = []
    for rec in frame.itertuples(index=False):
        chunks = split_sentences_chunks(rec.text, tok) or [rec.text]
        k = rng.randint(1, MAX_WINDOWS)
        picks = [0]
        if len(chunks) > 1 and k >= 2:
            picks.append(len(chunks) - 1 if len(chunks) == 2 else rng.randrange(1, len(chunks) - 1))
        if len(chunks) > 2 and k >= 3:
            picks.append(len(chunks) - 1)
        for i in picks:
            rows.append({"text": chunks[i], "generated": rec.generated, "source": rec.source,
                         "window": "start" if i == 0 else "end" if i == len(chunks) - 1 else "middle"})
    return pd.DataFrame(rows)


def main():
    human = pd.read_parquet(IVY_HUMAN)
    human["title_n"] = human["TEXT"].map(title_of)
    ai = load_ai()
    dev_titles, final_titles, dev_texts = excluded_titles(human, ai)
    banned = dev_titles | final_titles
    print(f"tematy dev: {len(dev_titles)}, tytuly essays_2021: {len(final_titles)}")

    # --- AI: rowno na kazda rodzine modeli, bez tematow dev / finalnego testu
    ai = ai[~ai["title_n"].isin(banned)].copy()
    ai["text"] = ai["text"].map(strip_ai)
    ai = ai[~ai["text"].map(norm).isin(dev_texts)]
    ai = ai[ai["text"].map(word_count) >= MIN_WORDS].drop_duplicates("text")
    ai = pd.concat([g.sample(n=min(len(g), AI_PER_MODEL), random_state=SEED)
                    for _, g in ai.groupby("family")], ignore_index=True)
    ai = pd.DataFrame({"text": ai["text"], "generated": 1,
                       "source": "ivypanda_ai_" + ai["family"], "title_n": ai["title_n"]})

    # --- ludzie: w kazdym przedziale dlugosci tyle, ile esejow AI (z nadwyzka na filtry),
    #     najpierw eseje na te same tematy co wybrane AI, potem losowe z calej puli 128 tys.
    human = human[~human["title_n"].isin(banned)].copy()
    human["text"] = human["TEXT"].map(strip_human)
    human = human[~human["text"].map(norm).isin(dev_texts)]
    human["w"] = human["text"].map(word_count)
    human = human[human["w"] >= MIN_WORDS].drop_duplicates("text")
    ai["w"] = ai["text"].map(word_count)
    edges = sorted(set(ai["w"].quantile([i / 14 for i in range(15)]).round().astype(int)))
    edges[0], edges[-1] = 0, 10 ** 9
    ai["bin"] = pd.cut(ai["w"], edges, include_lowest=True)
    human["bin"] = pd.cut(human["w"], edges, include_lowest=True)
    ai_topics = set(ai["title_n"])
    human["same_topic"] = human["title_n"].isin(ai_topics)
    picked = []
    for b, n_ai in ai["bin"].value_counts().items():
        g = human[human["bin"] == b].sample(frac=1.0, random_state=SEED)
        g = g.sort_values("same_topic", ascending=False, kind="stable").drop_duplicates("title_n")
        picked.append(g.head(int(n_ai * 1.15)))
    human = pd.concat(picked)
    print("ludzkie na tematy esejow AI:", int(human["same_topic"].sum()), "z", len(human))
    human = pd.DataFrame({"text": human["text"], "generated": 0, "source": "ivypanda_human",
                          "title_n": human["title_n"]})
    ai = ai[["text", "generated", "source", "title_n"]]

    pool = pd.concat([human, ai], ignore_index=True)

    # --- bez przecieku do testow: essays_2021 (finalny) i Ghostbuster (OOD; jego ludzkie eseje
    #     pochodza z IvyPandy!) - wyrzuc wszystko, co dzieli z nimi >20 osmiogramow,
    #     oraz teksty identyczne z jakimkolwiek splitem v3
    final_sh = set().union(*(shingles(p.read_text(encoding="utf-8")) for p in FINAL_TEST_DIR.glob("*.txt")))
    gb = pd.concat([pd.read_parquet(ROOT / "data" / "ghostbuster" / "data" / f)
                    for f in ("train-00000-of-00001.parquet", "test-00000-of-00001.parquet")])
    gb_sh = set().union(*(shingles(t) for t in gb["text"].astype(str)))
    v3_keys = set().union(*(set(normalize_text(pd.read_csv(V3_DIR / f"{s}.csv")["text"]))
                            for s in ("train", "val", "test", "ood")))
    # dev: same tematy i identyczne teksty to za malo - ten sam esej bywa pod innym tytulem
    dev_by_text = [shingles(t) for t in pd.read_csv(DEV)["text"].astype(str)]
    dev_index = {}
    for i, s in enumerate(dev_by_text):
        for g in s:
            dev_index.setdefault(g, []).append(i)

    def max_dev_overlap(s):
        counts = {}
        for g in s:
            for i in dev_index.get(g, ()):
                counts[i] = counts.get(i, 0) + 1
        return max(counts.values(), default=0)

    sh = pool["text"].map(shingles)
    hit_final = sh.map(lambda s: len(s & final_sh) > 20)
    hit_gb = sh.map(lambda s: len(s & gb_sh) > 20)
    hit_dev = sh.map(lambda s: max_dev_overlap(s) > 20)
    hit_v3 = normalize_text(pool["text"]).isin(v3_keys)
    overlap = hit_final | hit_gb | hit_dev | hit_v3
    print("usuniete: essays_2021", int(hit_final.sum()), "| ghostbuster", int(hit_gb.sum()),
          "| dev", int(hit_dev.sum()), "| identyczne z v3", int(hit_v3.sum()))
    pool = pool[~overlap].reset_index(drop=True)

    pool = length_match(pool, seed=SEED, bins=14)
    ivy = grouped_split(pool, SEED)

    v3 = {s: pd.read_csv(V3_DIR / f"{s}.csv") for s in ("train", "val", "test")}
    ood = pd.read_csv(V3_DIR / "ood.csv")
    full = {s: pd.concat([v3[s], ivy[s][["text", "generated", "source"]]], ignore_index=True)
            .sample(frac=1.0, random_state=SEED).reset_index(drop=True) for s in v3}
    # pula v3 tez zawiera prawie-duplikaty esejow z dev (dev zbudowano po v3): usun je z train i val
    dev_leak_v3 = {}
    for s in ("train", "val"):
        hit = full[s]["text"].map(lambda t: max_dev_overlap(shingles(t)) > 20)
        dev_leak_v3[s] = full[s].loc[hit, "source"].value_counts().to_dict()
        full[s] = full[s][~hit].reset_index(drop=True)
    print("usuniete z train/val (pokrycie z dev):", dev_leak_v3)

    # bez przecieku: zadna nowa probka nie pokrywa sie z dev, ood ani miedzy splitami
    keys = {s: set(normalize_text(full[s]["text"])) for s in full}
    leaks = {
        "train&val": len(keys["train"] & keys["val"]),
        "train&test": len(keys["train"] & keys["test"]),
        "train&dev": len(keys["train"] & set(normalize_text(pd.read_csv(DEV)["text"]))),
        "train&ood": len(keys["train"] & set(normalize_text(ood["text"]))),
        "ivy_titles train&test": len(set(ivy["train"]["title_n"]) & set(ivy["test"]["title_n"])),
    }
    print("kontrola przecieku:", leaks)

    tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    win = {s: windows(full[s], tok, SEED + i).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
           for i, s in enumerate(("train", "val", "test"))}

    for out, data in ((OUT_FULL, full), (OUT_WIN, win)):
        out.mkdir(parents=True, exist_ok=True)
        for s, df in data.items():
            df.to_csv(out / f"{s}.csv", index=False)
        ood.to_csv(out / "ood.csv", index=False)

    summary = {
        "ivypanda_after_length_match": pool["source"].value_counts().to_dict(),
        "ivypanda_split": {s: len(ivy[s]) for s in ivy},
        "full": {s: {"n": len(full[s]), "ai": int(full[s].generated.sum())} for s in full},
        "windows": {s: {"n": len(win[s]), "ai": int(win[s].generated.sum()),
                        "by_window": win[s]["window"].value_counts().to_dict()} for s in win},
        "ood": len(ood),
        "leak_check": leaks,
        "excluded": {"dev_titles": len(dev_titles), "final_titles": len(final_titles),
                     "final_shingle_overlap": int(hit_final.sum()),
                     "ghostbuster_shingle_overlap": int(hit_gb.sum()),
                     "dev_shingle_overlap": int(hit_dev.sum()),
                     "dev_overlap_removed_from_train_val": dev_leak_v3,
                     "identical_to_v3_splits": int(hit_v3.sum())},
        "note": "v3 splits kept; IvyPanda added (title-grouped split); AI 'Title:' line stripped; "
                "windows = app split_chunks (320 tokens), 1-3 per essay: start + middle + end",
    }
    for out in (OUT_FULL, OUT_WIN):
        (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
