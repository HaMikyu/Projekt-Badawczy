"""Testy esejowe System One kodem aplikacji (fragmenty ~320 tokenow, srednia p_ai, progi 0,5 / 0,3).

Zestaw (nic z tego nie bylo w treningu ani walidacji; patrz train/build_data_v5.py):
  heldout_families           Claude 3 Opus, YandexGPT 3 Pro, GigaChat Pro - rodziny calkowicie
                             wylaczone z treningu v5, tematy testowe + ludzkie eseje na te tematy
  artnitolog_train_families  GPT-4 Turbo, GPT-4o, Llama 3 70B, Command R+ na tematach testowych
  llmessay                   LLMEssay-100K (2026) + ludzkie eseje IvyPanda o podobnej dlugosci
  daigt_v4                   Llamy dostrojone do stylu uczniow (PERSUADE)
  dev                        data/dev (strojenie progow aplikacji)
  essays_2021                300 ludzkich esejow StudyCorgi (finalny test; tylko z --final)

    python train/eval_essay_suite.py v4=models/system-one-v4 v5a=train/runs/system-one-v5a/export [--final]
"""
from pathlib import Path
import csv
import json
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "app"))

from helpers import predict_system_one  # noqa: E402
from train.predict_system_one import load_detector  # noqa: E402

TESTS = ROOT / "data" / "prepared_v5_tests"
OUT = ROOT / "train" / "runs" / "essay_suite"


def load_sets(final):
    sets = {p.stem: pd.read_csv(p) for p in sorted(TESTS.glob("*.csv"))}
    dev = pd.read_csv(ROOT / "data" / "dev" / "dev.csv")
    sets["dev"] = pd.DataFrame({"text": dev.text, "generated": dev.generated, "source": dev.source,
                                "family": dev.source})
    if final:
        rows = list(csv.DictReader(open(ROOT / "essays_2021" / "index.csv", encoding="utf-8")))
        sets["essays_2021"] = pd.DataFrame({
            "text": [(ROOT / "essays_2021" / r["filename"]).read_text(encoding="utf-8") for r in rows],
            "generated": 0, "source": "studycorgi_2021", "family": "human"})
    return sets


def summarize(df, p, v):
    df = df.assign(p=p, v=v)
    out = {"n": len(df)}
    h, a = df[df.generated == 0], df[df.generated == 1]
    if len(h):
        out["human_as_ai"] = round(float((h.v == "AI").mean()), 4)
        out["human_unsure"] = round(float((h.v == "UNSURE").mean()), 4)
        out["n_human"] = len(h)
    if len(a):
        out["ai_detected"] = round(float((a.v == "AI").mean()), 4)
        out["ai_as_human"] = round(float((a.v == "HUMAN").mean()), 4)
        out["n_ai"] = len(a)
        out["ai_detected_by_family"] = {f: round(float((g.v == "AI").mean()), 4) for f, g in a.groupby("family")}
    if len(h) and len(a):
        out["auc"] = round(float(roc_auc_score(df.generated, df.p)), 4)
    return out


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    final = "--final" in sys.argv
    models = dict(a.split("=", 1) for a in args)
    sets = load_sets(final)
    OUT.mkdir(parents=True, exist_ok=True)
    report = {"final_included": final, "models": models, "results": {}}
    for name, path in models.items():
        model, tok, meta, dev = load_detector(ROOT / path)
        so = {"model": model, "tokenizer": tok, "meta": meta, "device": dev}
        report["results"][name] = {}
        for sname, df in sets.items():
            t0 = time.time()
            res = [predict_system_one(str(t), so) for t in df.text]
            p = np.array([r["ai_percent"] / 100 for r in res])
            v = [r["label"] for r in res]
            df.assign(p=p, v=v).drop(columns=["text"]).to_csv(OUT / f"{name}__{sname}.csv", index=False)
            s = summarize(df, p, v)
            report["results"][name][sname] = s
            print(f"[{name}] {sname:26s} {json.dumps({k: v for k, v in s.items() if k != 'ai_detected_by_family'})}"
                  f" ({time.time() - t0:.0f} s)", flush=True)
            if "ai_detected_by_family" in s:
                print(f"      rodziny: {s['ai_detected_by_family']}", flush=True)
        del model
    tag = "_".join(models)
    (OUT / f"report_{tag}{'_final' if final else ''}.json").write_text(json.dumps(report, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
