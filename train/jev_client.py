from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any


DEFAULT_URL = "https://api.typesafe.ai/v1/systemone"
DO_URL = "https://inference.do-ai.run/v1/systemone"
DEFAULT_MODEL = "typesafe-jev-1.13.0"

ESSAY_CRITERIA = {
    "true": (
        "generic thesis, formulaic paragraph template, evenly polished sentences, "
        "few concrete personal details, repeated transitions like furthermore/moreover"
    ),
    "false": (
        "uneven sentence rhythm, specific examples or local details, "
        "occasional awkward phrasing, identifiable student voice"
    ),
}


def _api_key() -> str | None:
    for name in ("TYPESAFE_API_KEY", "MODEL_ACCESS_KEY", "DO_API_TOKEN"):
        value = os.environ.get(name)
        if value:
            return value
    return None


def _endpoint() -> str:
    if os.environ.get("MODEL_ACCESS_KEY") or os.environ.get("DO_API_TOKEN"):
        return os.environ.get("SYSTEMONE_URL", DO_URL)
    return os.environ.get("SYSTEMONE_URL", DEFAULT_URL)


def available() -> bool:
    return _api_key() is not None


def score_ai_essay(text: str, timeout: float = 20.0) -> dict[str, Any]:
    key = _api_key()
    if not key:
        raise RuntimeError("brak TYPESAFE_API_KEY / MODEL_ACCESS_KEY")

    payload = {
        "model": os.environ.get("SYSTEMONE_MODEL", DEFAULT_MODEL),
        "state": {"genre": "student_essay", "text": text},
        "questions": {
            "ai_generated": {
                "type": "noul",
                "instructions": (
                    "Was this student essay written by an AI rather than a human student?"
                ),
                "criteria": ESSAY_CRITERIA,
            }
        },
    }
    request = urllib.request.Request(
        _endpoint(),
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Jev HTTP {exc.code}: {detail[:400]}") from exc

    answers = body.get("answers") or {}
    noul = answers.get("ai_generated", {}).get("noul")
    if noul is None:
        raise RuntimeError(f"dziwna odpowiedz Jev: {json.dumps(body)[:400]}")
    return {
        "proba_ai": float(noul),
        "raw": body,
        "backend": "jev_system_one",
    }


if __name__ == "__main__":
    if not available():
        print(json.dumps({"available": False, "hint": "ustaw TYPESAFE_API_KEY"}, indent=2))
    else:
        sample = "I think smoking should be banned because it hurts people."
        print(json.dumps(score_ai_essay(sample), indent=2)[:1000])
