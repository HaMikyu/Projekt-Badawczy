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

## Aplikacja: ocena fragmentami + finalny test na esejach spoza treningu

System One widzi tylko 384 tokeny, czyli zwykle sam wstęp eseju. Aplikacja
(`app/helpers.py`) dzieli więc tekst na fragmenty ~320 tokenów (bez cięcia zdań),
ocenia każdy osobno i uśrednia `p_ai`:
**AI ≥ 0,5 · NIEPEWNE 0,3–0,5 · CZŁOWIEK < 0,3**.

Ustawienia dobrano na zbiorze dev (`train/build_dev_set.py`: 290 ludzkich esejów
IvyPanda + 282 eseje AI na te same tematy z ChatGPT 4.1 / DeepSeek / Llama 3 / Mistral
+ 16 esejów od Claude):

| wariant (dev) | ludzkie → AI | AI wykryte |
|---|---|---|
| tylko pierwszy fragment | 10,0% | 87,6% |
| **średnia z fragmentów, próg 0,5** | **3,4%** | 84,2% |
| średnia z fragmentów, próg 0,7 | 2,4% | 77,2% |

Finalny test, uruchomiony raz po zamrożeniu ustawień: `essays_2021/`, 300 ludzkich
esejów StudyCorgi z 2021 r. (zbiór nigdy nie używany do treningu ani strojenia),
skrypt `train/final_test_essays_2021.py`:

| detektor | człowiek | niepewne | AI (fałszywy alarm) |
|---|---|---|---|
| poprzednia aplikacja (średnia DistilBERT baseline + statystyczny) | – | – | 93,3% |
| **System One, fragmenty** | **80,3%** | 12,7% | **7,0%** |

Plan douczenia modelu na GPU, który ma zmniejszyć te 7%: [`GPU_TRAINING.md`](./GPU_TRAINING.md).
