# Wyniki ML — detekcja tekstu AI (eseje)

## Werdykt

Lokalny **System One option scorer** (DistilBERT, kontrakt Jev: `state + options -> wybór`)
na mieszance esejów bije stary DistilBERT v3 na Ghostbuster OOD i mocno obcina fałszywe AI.

| model | OOD acc | human TNR | AI TPR |
|---|---|---|---|
| DistilBERT v3 (stary, Kaggle+HC3+Pile) | 0,838 | 0,705 | 0,970 |
| System One v1 (same essays, bez GB train) | 0,503 | 0,007 | 0,999 |
| System One v2 (esseje + GB train) | 0,931 | 0,863 | 1,000 |
| **System One v3 (final)** | **0,946** | **0,900** | **0,993** |

## Co to jest System One

Nie chatbot. Dla eseju scoruje dwie opcje:

- `human` (kryteria: nierówny rytm, szczegóły, głos studenta)
- `ai` (kryteria: szablon, wygładzony styl, generyczna teza)

Softmax po score'ach = skalibrowane `p_ai` (+ temperature scaling).

## Dane (essay-first)

Trening: Kaggle essays + DAIGT v3 + ArguGPT + Ghostbuster **train**
(humans z GB x10) + formalne human z Pile jako twarde negatywy.
OOD: Ghostbuster **test** (nigdy w treningu), length-matched.

| split | n |
|---|---|
| train | 52 526 |
| val / test | 6 566 |
| OOD GB test | 598 |

## Wyniki System One v3

### Test IID

| metryka | wartość |
|---|---|
| accuracy | **0,994** |
| precision | 0,989 |
| recall | 0,998 |
| F1 | **0,994** |
| TNR (human) | 0,989 |

### OOD Ghostbuster test

| metryka | wartość |
|---|---|
| accuracy | **0,946** |
| precision | 0,908 |
| recall | 0,993 |
| F1 | **0,949** |
| TNR (human) | **0,900** |
| threshold | 0,83 |
| temperature | 0,74 |

Macierz OOD (`[[TN, FP], [FN, TP]]`):

```text
[[269, 30],
 [  2, 297]]
```

Czas treningu v3: ~37 min (RTX 3070 Ti, batch 12, max_length 384, 4 epoki).

## Odpalenie

```powershell
.\.venv\Scripts\python.exe train\build_data.py
.\.venv\Scripts\python.exe train\train_system_one.py
.\.venv\Scripts\python.exe train\export_system_one.py
.\.venv\Scripts\python.exe train\predict_system_one.py naturalny_essay.txt
```

Artefakty:

- wagi: `models/system-one-final/`
- metryki: `train/runs/system-one-v3/metrics.json`
- dane: `data/prepared/` (gitignore)

Opcjonalnie prawdziwy Jev API: `train/jev_client.py` (wymaga `TYPESAFE_API_KEY`).
