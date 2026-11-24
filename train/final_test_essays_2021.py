"""FINALNY test detektora na essays_2021 (300 ludzkich esejow StudyCorgi z 2021 r.).

Ten zbior NIE jest uzywany do treningu ani strojenia (progi i rozmiar fragmentow
dobrano na data/dev). Uruchamia dokladnie ten sam kod co aplikacja (app/helpers.py).

    python train/final_test_essays_2021.py
"""
from pathlib import Path
import csv
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from helpers import analyze_text, load_models  # noqa: E402

OUT = ROOT / "train" / "final_test_essays_2021.json"


def main():
    models = load_models()
    rows = list(csv.DictReader(open(ROOT / "essays_2021" / "index.csv", encoding="utf-8")))
    results = []
    t0 = time.time()
    for i, r in enumerate(rows, 1):
        text = (ROOT / "essays_2021" / r["filename"]).read_text(encoding="utf-8")
        res = analyze_text(text, models)
        so = res["system_one"]
        results.append({
            "file": r["filename"],
            "words": int(r["words"]),
            "verdict": res["combined"]["label"],
            "ai_percent": res["combined"]["ai_percent"],
            "chunks_ai": so["chunks_ai"] if so else None,
            "chunks_total": so["chunks_total"] if so else None,
            "distilbert_label": res["ml"]["label"] if res["ml"] else None,
            "stat_label": res["statistical"]["label"] if res["statistical"] else None,
        })
        if i % 50 == 0:
            print(f"{i}/{len(rows)} ({time.time() - t0:.0f} s)", flush=True)

    n = len(results)

    def share(key, value):
        return round(sum(r[key] == value for r in results) / n, 4)

    summary = {
        "n_human_essays": n,
        "system_one": {
            "HUMAN (poprawnie)": share("verdict", "HUMAN"),
            "UNSURE": share("verdict", "UNSURE"),
            "AI (falszywy alarm)": share("verdict", "AI"),
        },
        "distilbert_pomocniczy_falszywe_AI": share("distilbert_label", "AI"),
        "statystyczny_pomocniczy_falszywe_AI": share("stat_label", "AI"),
    }
    OUT.write_text(json.dumps({"summary": summary, "essays": results}, indent=1, ensure_ascii=False),
                   encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
