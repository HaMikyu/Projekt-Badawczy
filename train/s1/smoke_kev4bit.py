"""Faza 0: czy Kev-4B w NF4 dziala na 8 GB i ile kosztuje jedno zadanie System One.

    .venv-s1\\Scripts\\python.exe train\\s1\\smoke_kev4bit.py [run]
"""
from pathlib import Path
import sys
import time

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from kev.api import SystemOneRequest, to_answers, to_record  # noqa: E402
from train.s1.kev4bit import load_kev, memory_report  # noqa: E402

ESSAY_Q = {
    "ai_generated": {
        "type": "noul",
        "instructions": "Was this student essay written by an AI model rather than by a human student?",
        "criteria": {
            "true": "generic thesis, formulaic paragraph template, evenly polished sentences, few concrete personal details, "
                    "repeated transitions like furthermore/moreover",
            "false": "uneven sentence rhythm, specific examples or local details, occasional awkward phrasing, "
                     "identifiable student voice",
        },
    },
    "origin": {
        "type": "choice",
        "instructions": "Who most likely produced this essay?",
        "criteria": {"human": "a human student wrote it", "ai": "an AI language model generated it"},
    },
}


def main():
    run = sys.argv[1] if len(sys.argv) > 1 else "jaredpalmer/kev-4b"
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    tok, model = load_kev(run)
    print(f"wczytanie {time.time() - t0:.0f} s", memory_report("po wczytaniu"), flush=True)
    dev = pd.read_csv(ROOT / "data" / "dev" / "dev.csv")
    sample = pd.concat([dev[dev.generated == 0].head(3), dev[dev.generated == 1].head(3)])
    for _, r in sample.iterrows():
        req = SystemOneRequest(state={"genre": "student_essay", "text": r.text}, questions=ESSAY_Q)
        rec, meta = to_record(req)
        enc = model.encode(tok, rec, max_state=8192, max_branch=8192 + 1024)
        torch.cuda.synchronize(); t = time.time()
        probs = model.probs(enc)
        torch.cuda.synchronize()
        ans = to_answers([p.tolist() for p in probs], meta)
        print(f"y={r.generated} tokeny stanu={enc['state_tokens']:5d} czas={time.time() - t:5.2f} s "
              f"noul={ans['ai_generated']['noul']:.3f} origin={ans['origin']['probabilities']}", flush=True)
    print(memory_report("szczyt"), flush=True)


if __name__ == "__main__":
    main()
