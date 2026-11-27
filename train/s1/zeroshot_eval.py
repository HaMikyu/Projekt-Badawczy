"""Faza 0: Kev bez douczania (zero-shot) na data/dev i data/prepared_v5_tests, cale eseje w jednym przejsciu.

Jedno pytanie Noul (ai_generated, kryteria jak w train/jev_client.py), max_state 8192 tokenow (dluzsze eseje
przycinane do poczatku - zapisane w kolumnie truncated). Wyniki per esej: train/runs/s1_spike/zeroshot_<run>/<zbior>.csv
(wznawialne), podsumowanie: summary.json z AUC, falszywymi alarmami i wykrywalnoscia przy progu 0,5 oraz przy
progu dobranym na dev (maks. zbalansowana trafnosc) - obok wynikow System One v5 (report_v5d.json).

    .venv-s1\\Scripts\\python.exe train\\s1\\zeroshot_eval.py [run] [--limit N]
"""
from pathlib import Path
import json
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from kev.api import SystemOneRequest, to_record  # noqa: E402
from train.s1.kev4bit import load_kev, memory_report  # noqa: E402
from train.s1.smoke_kev4bit import ESSAY_Q  # noqa: E402

MAX_STATE = 8192
SETS = ["dev", "heldout_families", "artnitolog_train_families", "daigt_v4", "llmessay"]


def load_set(name):
    if name == "dev":
        return pd.read_csv(ROOT / "data" / "dev" / "dev.csv")
    return pd.read_csv(ROOT / "data" / "prepared_v5_tests" / f"{name}.csv")


def score_set(tok, model, df, out_csv):
    done = pd.read_csv(out_csv) if out_csv.exists() else pd.DataFrame(columns=["i", "p", "state_tokens", "truncated", "sec"])
    have = set(done.i.astype(int))
    rows = done.to_dict("records")
    t_start = time.time()
    qs = {"ai_generated": ESSAY_Q["ai_generated"]}
    for i, text in enumerate(df.text):
        if i in have:
            continue
        rec, _ = to_record(SystemOneRequest(state={"genre": "student_essay", "text": str(text)}, questions=qs))
        t = time.time()
        for max_state in (MAX_STATE, 4096, 2048):   # brak pamieci przy najdluzszych esejach -> krotszy stan (zapisany)
            enc = model.encode(tok, rec, max_state=max_state, max_branch=max_state + 1024)
            try:
                p = float(model.probs(enc)[0][1])
                break
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
        else:
            raise RuntimeError(f"esej {i}: brak pamieci nawet przy 2048 tokenach stanu")
        rows.append({"i": i, "p": p, "state_tokens": enc["state_tokens"], "truncated": enc["state_truncated"],
                     "max_state": max_state, "sec": round(time.time() - t, 3)})
        if len(rows) % 50 == 0:
            pd.DataFrame(rows).to_csv(out_csv, index=False)
            print(f"  {len(rows)}/{len(df)} ({time.time() - t_start:.0f} s)", flush=True)
    res = pd.DataFrame(rows).sort_values("i")
    res.to_csv(out_csv, index=False)
    return res


def metrics(y, p, thr):
    y, p = np.asarray(y), np.asarray(p)
    out = {"n": int(len(y)), "threshold": round(float(thr), 4)}
    h, a = p[y == 0], p[y == 1]
    if len(h):
        out["human_as_ai"] = round(float((h >= thr).mean()), 4); out["n_human"] = int(len(h))
    if len(a):
        out["ai_detected"] = round(float((a >= thr).mean()), 4); out["n_ai"] = int(len(a))
    if len(h) and len(a):
        out["auc"] = round(float(roc_auc_score(y, p)), 4)
    return out


def dev_threshold(y, p):
    """Prog z dev: maksymalna zbalansowana trafnosc (TPR + TNR) / 2."""
    y, p = np.asarray(y), np.asarray(p)
    best, thr = -1, 0.5
    for t in np.unique(p):
        bal = ((p[y == 1] >= t).mean() + (p[y == 0] < t).mean()) / 2
        if bal > best:
            best, thr = bal, t
    return float(thr), float(best)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    run = args[0] if args else "jaredpalmer/kev-4b"
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
    out = ROOT / "train" / "runs" / "s1_spike" / f"zeroshot_{run.split('/')[-1]}"
    out.mkdir(parents=True, exist_ok=True)
    tok, model = load_kev(run)
    print(memory_report("po wczytaniu"), flush=True)
    scores = {}
    for name in SETS:
        df = load_set(name)
        if limit:
            df = df.groupby("generated", group_keys=False).apply(lambda g: g.head(limit))
            df = df.reset_index(drop=True)
        t0 = time.time()
        res = score_set(tok, model, df, out / f"{name}.csv")
        scores[name] = (df.generated.values[res.i.astype(int).values], res.p.values, res)
        print(f"[{name}] {len(res)} esejow, {time.time() - t0:.0f} s, sr. {res.sec.mean():.2f} s/esej, "
              f"przyciete {int(res.truncated.sum())}", flush=True)
    y_dev, p_dev, _ = scores["dev"]
    thr, bal = dev_threshold(y_dev, p_dev)
    v5 = json.loads((ROOT / "train" / "runs" / "essay_suite" / "report_v5d.json").read_text(encoding="utf-8"))["results"]["v5d"]
    summary = {"run": run, "max_state": MAX_STATE, "dev_threshold": round(thr, 4), "dev_balanced_acc": round(bal, 4),
               "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2), "sets": {}}
    for name, (y, p, res) in scores.items():
        summary["sets"][name] = {"at_0.5": metrics(y, p, 0.5), "at_dev_threshold": metrics(y, p, thr),
                                 "sec_per_essay": round(float(res.sec.mean()), 3),
                                 "truncated": int(res.truncated.sum()),
                                 "v5_app_at_0.5": {k: v for k, v in v5.get(name, {}).items() if k != "ai_detected_by_family"}}
        print(name, json.dumps(summary["sets"][name], ensure_ascii=False), flush=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
