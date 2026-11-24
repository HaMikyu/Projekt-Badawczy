import csv
import re
from pathlib import Path

from helpers import PROMO_PHRASES


ROOT = Path(__file__).resolve().parents[1]
ESSAYS_DIR = ROOT / "essays_2021"
OUT_CSV = ESSAYS_DIR / "essays_2021.csv"


def is_promo_paragraph(text):
    low = text.lower()
    return any(phrase in low for phrase in PROMO_PHRASES)


def clean_body(text):
    parts = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    while parts and is_promo_paragraph(parts[0]):
        parts = parts[1:]

    cleaned = []
    for part in parts:
        low = part.lower().strip()
        if low.startswith(("works cited", "references", "reference list", "bibliography")):
            break
        if is_promo_paragraph(part):
            continue
        cleaned.append(re.sub(r"[ \t]+", " ", part).strip())
    return "\n\n".join(cleaned)


def main():
    rows = []
    files = sorted(ESSAYS_DIR.glob("*.txt"))
    for path in files:
        raw = path.read_text(encoding="utf-8")
        body = clean_body(raw)
        if len(body.split()) < 30:
            continue
        path.write_text(body + "\n", encoding="utf-8")
        rows.append({
            "text": body,
            "generated": 0,
            "source_file": path.name,
        })

    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "generated", "source_file"])
        writer.writeheader()
        writer.writerows(rows)

    words = [len(r["text"].split()) for r in rows]
    print(f"Zapisano {len(rows)} esejow")
    print(f"CSV: {OUT_CSV}")
    print(f"slowa: mean={sum(words)/len(words):.1f} min={min(words)} max={max(words)}")


if __name__ == "__main__":
    main()
