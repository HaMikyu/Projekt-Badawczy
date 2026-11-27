"""Faza 0: pamiec i tempo kroku treningu QLoRA Kev-4B na 8 GB (w przod + wstecz + AdamW).

    .venv-s1\\Scripts\\python.exe train\\s1\\spike_train.py [run] [dlugosci, np. 512,1024,2048,3072]

Kazdy rekord: esej (przyciety do N tokenow) + 1 lub 2 pytania (Noul ai_generated, Choice origin), etykieta znana.
Postac wierszowa (backbone hybrydowy): kazde pytanie = osobny wiersz stan + galaz, wiec 2 pytania = 2x stan.
"""
from pathlib import Path
import json
import sys
import time

import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from kev.api import SystemOneRequest, to_record  # noqa: E402
from train.s1.kev4bit import load_kev, memory_report  # noqa: E402
from train.s1.smoke_kev4bit import ESSAY_Q  # noqa: E402


def records(tok, texts, labels, n_tokens, questions):
    out = []
    for t, y in zip(texts, labels):
        ids = tok(t, add_special_tokens=False).input_ids[: n_tokens - 1]
        text = tok.decode(ids)
        qs = {k: ESSAY_Q[k] for k in questions}
        rec, _ = to_record(SystemOneRequest(state={"genre": "student_essay", "text": text}, questions=qs))
        for q in rec["questions"]:
            q["label"] = int(y)   # noul: [no, yes]; origin: [human, ai] -> AI = 1 w obu
        out.append(rec)
    return out


def step(model, tok, opt, recs):
    encs = [model.encode(tok, r, max_state=8192, max_branch=8192 + 1024) for r in recs]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model.forward_batch(encs)
    loss = sum(F.cross_entropy(z.float()[None], torch.tensor([q["label"]], device="cuda"))
               for zs, r in zip(logits, recs) for z, q in zip(zs, r["questions"]))
    loss = loss / sum(len(r["questions"]) for r in recs)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 1.0)
    opt.step(); opt.zero_grad(set_to_none=True)
    return float(loss), sum(len(e["ids"]) + (len(r["questions"]) - 1) * e["state_tokens"] for e, r in zip(encs, recs))


def main():
    run = sys.argv[1] if len(sys.argv) > 1 else "jaredpalmer/kev-4b"
    lengths = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "512,1024,2048,3072").split(",")]
    import bitsandbytes as bnb
    tok, model = load_kev(run, train=True)
    params = model.trainable_parameters()
    print(f"trenowalne parametry: {sum(p.numel() for p in params) / 1e6:.1f} mln", memory_report("po wczytaniu"), flush=True)
    opt = bnb.optim.PagedAdamW8bit(params, lr=2e-5, weight_decay=0.01)
    dev = pd.read_csv(ROOT / "data" / "dev" / "dev.csv")
    long = dev[dev.text.str.len() > 15000].sample(8, random_state=0)
    results = []
    for n in lengths:
        for qs in (["ai_generated"], ["ai_generated", "origin"]):
            recs = records(tok, long.text.tolist()[:2], long.generated.tolist()[:2], n, qs)
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            try:
                step(model, tok, opt, recs[:1])           # rozgrzewka
                torch.cuda.synchronize(); t = time.time(); toks = 0
                for r in recs:
                    _, k = step(model, tok, opt, [r]); toks += k
                torch.cuda.synchronize(); dt = (time.time() - t) / len(recs)
                peak = torch.cuda.max_memory_allocated() / 2**30
                res = {"state_tokens": n, "questions": len(qs), "s_per_record": round(dt, 2),
                       "tokens_per_s": round(toks / len(recs) / dt), "peak_gb": round(peak, 2)}
            except torch.cuda.OutOfMemoryError:
                opt.zero_grad(set_to_none=True); torch.cuda.empty_cache()
                res = {"state_tokens": n, "questions": len(qs), "oom": True}
            print(json.dumps(res), flush=True); results.append(res)
    out = ROOT / "train" / "runs" / "s1_spike"; out.mkdir(parents=True, exist_ok=True)
    (out / f"train_{run.split('/')[-1]}.json").write_text(json.dumps(results, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
