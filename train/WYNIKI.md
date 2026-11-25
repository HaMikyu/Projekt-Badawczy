# Wyniki ML — detekcja tekstu AI (eseje)

## Werdykt

Lokalny **System One option scorer** (DistilBERT, kontrakt Jev: `state + options -> wybór`)
na mieszance esejów bije stary DistilBERT v3 na Ghostbuster OOD i mocno obcina fałszywe AI.

| model | OOD acc | human TNR | AI TPR |
|---|---|---|---|
| DistilBERT v3 (stary, Kaggle+HC3+Pile) | 0,838 | 0,705 | 0,970 |
| System One v1 (same essays, bez GB train) | 0,503 | 0,007 | 0,999 |
| System One v2 (esseje + GB train) | 0,931 | 0,863 | 1,000 |
| System One v3 | 0,946 | 0,900 | 0,993 |
| **System One v4 (final, od 25.11.2026)** | **0,985** | **0,993** | 0,977 |

v4 to v3 douczony na +23 tys. esejów IvyPanda (ludzkie + AI) we fragmentach
jak w aplikacji — szczegóły w sekcji [System One v4](#system-one-v4-douczenie-na-gpu-25112026).

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
Wykonany — patrz niżej.

## System One v4: douczenie na GPU (25.11.2026)

Wykonane wg [`GPU_TRAINING.md`](./GPU_TRAINING.md), gałąź `system-one-v4`,
RTX 3070 Ti 8 GB, start z wag v3.

### Dane (`train/build_data_v4.py`)

- v3 (bez zmian w podziale) + **11 644 ludzkich esejów IvyPanda i 11 644 esejów AI**
  z tego samego zbioru LLM (GPT 2951, DeepSeek 2938, Llama 2882, Mistral 2873),
  ludzkie dobierane do długości AI w koszykach długości, podział po tytułach.
- Wykluczone: tytuły i teksty zbioru dev, tytuły `essays_2021` oraz wszystko, co
  ma > 20 wspólnych 8-gramów z dev, `essays_2021` lub Ghostbusterem
  (**ludzkie eseje Ghostbustera pochodzą z IvyPanda** — 873 teksty usunięte,
  żeby OOD zostało OOD).
- Wariant „okna”: z każdego eseju 1–3 fragmenty ~320 tokenów tym samym
  algorytmem co `app/helpers.py` (początek + losowy środek + koniec).
- Rozmiar: 105 830 / 13 192 / 13 019 (okna), 71 244 / 8 898 / 8 797 (pełne teksty).

Uwaga: przy budowie znalazłem bliskie duplikaty dev w puli v3
(`pile_formal_human`, maks. 613 wspólnych 8-gramów) — usunięte z train/val v4.
Wynik v3 na dev mógł więc być lekko zawyżony.

### Warianty (wybór tylko na dev + Ghostbuster OOD)

| wariant | dane | lr | epoki | czas | dev ludzkie→AI @0,5 | dev AI wykryte @0,5 | OOD acc | OOD TNR | OOD TPR |
|---|---|---|---|---|---|---|---|---|---|
| v3 (punkt odniesienia) | — | — | — | — | 2,71% | 84,2% | 0,945 | 0,896 | 0,993 |
| a | pełne teksty | 1e-5 | 2 | 23 min | 0,0% | 97,5% | **0,987** | 0,983 | 0,990 |
| **b (wybrany)** | **okna** | **1e-5** | 2 | 35 min | **0,0%** | **100%** | 0,985 | **0,993** | 0,977 |
| c | okna | 2e-5 | 2 | 35 min | 0,0% | 100% | 0,980 | 0,993 | 0,967 |

(v3 na dev policzony na obecnej wersji `data/dev`: 295 ludzkich + 279 AI.)

Wszystkie trzy spełniają kryteria akceptacji. Wybrany **b**: najmniej fałszywych
alarmów na ludzkich tekstach OOD (2/299 vs 5/299 dla a) przy perfekcyjnym dev
nawet przy progu 0,7 (0% / 99,6%). Każdy generator na dev: 100% (v3: Mistral 72,9%,
Llama 74,1%). Macierz OOD b: `[[297, 2], [7, 292]]`. IID (okna): acc 0,994, F1 0,994.

### Finalny test `essays_2021` (uruchomiony raz, po wyborze wariantu)

| detektor | człowiek | niepewne | AI (fałszywy alarm) |
|---|---|---|---|
| System One v3, fragmenty | 80,3% | 12,7% | 7,0% |
| **System One v4, fragmenty** | **99,3%** | 0,7% | **0,0%** |

v3 przeliczony na tej samej lokalnej kopii esejów daje identyczne liczby co
pierwotny test, więc porównanie jest bezpośrednie. Dwa eseje v4 ocenił jako
NIEPEWNE (średnie p_ai 0,48 i 0,49), mediana p_ai wynosi 0,0005.

**Ograniczenia, których ten wynik nie znosi:**

- `essays_2021` zawiera wyłącznie teksty ludzkie, więc mierzy tylko fałszywe alarmy.
  Wykrywalność AI pochodzi z dev i Ghostbustera.
- AI w dev pochodzi z tej samej rodziny danych LLM co nowy trening (te same
  generatory, inne tematy), więc 100% na dev jest pomiarem bliskim domenie.
  Bardziej miarodajne jest Ghostbuster OOD: 97,7% AI wykryte (v3: 99,3%, czyli lekki spadek).
- StudyCorgi i IvyPanda to podobne serwisy z esejami. Model nauczył się tego stylu
  jako ludzkiego, więc na innym typie ludzkiego tekstu wynik może być gorszy.

### Test zewnętrzny na gotowych benchmarkach (nowe generatory, różne gatunki)

`train/eval_external.py`, uruchomiony raz, bez strojenia progów. Wyniki w `train/external_test_results.json`.

- **WasItAIGenerated Benchmark 2026-09** (HF `wasitaigeneratedcom/ai-detector-benchmark-2026-09`).
  Zawiera teksty AI z ok. 40 modeli z lat 2025–2026 (GPT-5/6, Claude 4/5, Gemini 3, DeepSeek V4, Qwen, Grok…),
  parafrazy, teksty po 6 humanizerach komercyjnych i teksty ludzkie (mowy, dokumentacje,
  teksty uczniów, artykuły) oraz werdykty GPTZero / Pangram / ZeroGPT na tych samych tekstach.
- **TextSight 2026** (HF `textsightai/ai-text-detection-benchmark-2026`): ludzkie teksty z Wikipedii sprzed 2022,
  z PMC (autorzy spoza krajów anglojęzycznych) i z literatury oraz 56 tekstów AI (akademickie i encyklopedyczne).
- Tylko język angielski, teksty od 80 słów. Usunięto 101 tekstów, które mają ponad 20 wspólnych 8-gramów z treningiem v4
  (głównie eseje PERSUADE, które już były w danych Kaggle). Zostało 2066 tekstów.

| grupa | n | v3 → AI | **v4 → AI** | v4 → niepewne |
|---|---|---|---|---|
| WasItAI: teksty ludzkie (fałszywy alarm) | 256 | 17,6% | **1,9%** | 0,4% |
| TextSight: teksty ludzkie (fałszywy alarm) | 635 | 20,5% | **8,5%** | 1,3% |
| – w tym Wikipedia | 155 | 63% | 30% | 2% |
| WasItAI: surowe AI (wykryte) | 671 | 44,7% | **35,6%** | 3,7% |
| WasItAI: AI parafrazowane | 44 | 56,8% | 45,5% | 0% |
| WasItAI: AI po humanizerze | 404 | 33,9% | 8,7% | 9,9% |
| TextSight: AI akademickie | 29 | 7% | **69%** | 14% |
| TextSight: AI encyklopedyczne | 27 | 100% | 100% | 0% |

AUC (ludzkie vs surowe AI, liczone z p_ai):

| zbiór | v3 | v4 |
|---|---|---|
| WasItAI | 0,654 | 0,743 |
| TextSight | 0,633 | 0,960 |

Te same teksty ocenione przez detektory komercyjne (dane benchmarku):

| podzbiór | GPTZero | Pangram | v4 |
|---|---|---|---|
| surowe AI, podzbiór GPTZero (n=188) | 60% | – | 36% |
| surowe AI, podzbiór Pangram (n=143) | – | 42% | 36% |
| po humanizerze, podzbiór GPTZero (n=278) | 24% | – | 9% |
| ludzkie, podzbiór GPTZero/Pangram (n≈42) | 0% | 0% | 7% |

**Wniosek.** v4 rzadko fałszywie oskarża ludzi także poza esejami, ale słabo wykrywa
teksty AI z nowych modeli w innych gatunkach niż esej, np. abstrakty, newsy, recenzje i przepisy.
Wykrywa ich 36%, mniej niż v3 (45%) i GPTZero (60%). v4 lepiej porządkuje teksty (AUC wyższe niż v3),
ale próg 0,5 dobrany na esejach jest poza tą domeną bardzo ostrożny. Progu nie zmieniamy pod ten test.
Humanizery skutecznie ukrywają AI przed v4 (9%). Teksty AI z benchmarku nie są esejami,
więc test nie odpowiada wprost na pytanie o eseje z nowych modeli; to nadal brakujący pomiar.

Odtworzenie:

```powershell
.\.venv\Scripts\python.exe train\build_data_v4.py
bash train/run_v4_variants.sh a b c     # trening + eval_dev, bez finalnego testu
$env:DETEKTOR_SYSTEM_ONE_DIR="models\system-one-v4"; .\.venv\Scripts\python.exe train\final_test_essays_2021.py
```

Metryki: `train/runs/system-one-v4{a,b,c}/` (gitignore), wagi: `models/system-one-v4/`
= `models/system-one-final/`.
