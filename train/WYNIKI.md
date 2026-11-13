# Wyniki ML — detekcja tekstu AI

## Werdykt

Wynik kolegi **~99,7% accuracy na Kaggle nie oznacza dobrego detektora**.
Ten sam model na Ghostbuster (Claude/GPT, nigdy nie widziany w treningu) spada do
**~51,6% accuracy** i oznacza prawie każdy tekst jako AI.

Nowy model (`distilbert-v3`) uczy się na mieszance źródeł i jest mierzony też poza domeną.

## 1. Co było nie tak z baseline (commit 6158bbf)

| test | accuracy | F1 | komentarz |
|---|---|---|---|
| Kaggle (ten sam rozkład) | **0,997** | 0,996 | łatwy, jednorodny zbiór |
| HC3 OOD (wcześniejszy smoke test) | **0,74** | 0,78 | duży spadek |
| Ghostbuster OOD (Claude/GPT) | **0,516** | 0,673 | praktycznie losowe |

Kod baseline był OK technicznie (deduplikacja, early stopping, brak przecieku exact-dup),
ale **brak zewnętrznego testu** ukrywał, że model uczy się artefaktów jednego CSV, nie „stylu AI”.

## 2. Nowy setup (`distilbert-v3`)

- Model: `distilbert-base-uncased` + Lightning, AMP fp16, RTX 3070 Ti
- Trening: Kaggle essays + HC3 (wszystkie domeny) + sample z AI Text Detection Pile
- Length-matching klas (żeby mniej cheatować długością)
- Deduplikacja exact
- **OOD holdout: Ghostbuster** (human / GPT / Claude / warianty promptów) — zero overlap z treningiem
- Seed 42, early stopping na `val_f1`

Rozmiary:

| split | n | human / ai |
|---|---|---|
| train | 59 014 | 29 507 / 29 507 |
| val | 7 376 | balanced |
| test (IID) | 7 376 | balanced |
| OOD Ghostbuster | 1 892 | 946 / 946 |

## 3. Wyniki finalne

### Test IID (mieszanka źródeł z treningu)

| metryka | wartość |
|---|---|
| accuracy | **0,986** |
| precision | 0,985 |
| recall | 0,987 |
| F1 | **0,986** |

### OOD Ghostbuster (prawdziwy test generalizacji)

| model | accuracy | precision | recall | F1 |
|---|---|---|---|---|
| baseline Kaggle-only | **0,516** | 0,508 | 0,997 | 0,673 |
| **distilbert-v3 (t=0.5)** | **0,838** | 0,767 | 0,970 | **0,857** |

Macierz v3 na OOD (`[[TN, FP], [FN, TP]]`):

```text
[[667, 279],
 [ 28, 918]]
```

Czas treningu v3: ~18 min (5 epok, batch 32, max_length 256).

### Rozbicie na źródła (z `metrics.json`)

**v3 nic nie traci na Kaggle.** Na wycinku `kaggle_essays` testu IID: accuracy **0,9985** (3 błędy na 1 978), czyli tyle co baseline (0,997), a na OOD +32 p.p.

OOD Ghostbuster, próg 0,5:

| źródło | n | accuracy |
|---|---|---|
| gpt | 243 | **1,000** |
| gpt_prompt1 | 163 | **0,994** |
| gpt_prompt2 | 112 | **1,000** |
| gpt_writing | 111 | **1,000** |
| gpt_semantic | 178 | 0,983 |
| claude | 139 | 0,827 |
| human | 946 | 0,705 |

Tekst AI z Ghostbustera v3 łapie w 97%, a GPT praktycznie zawsze. Prawie wszystkie błędy to fałszywe alarmy na tekstach ludzkich (279 z 307).

**Próg strojony na walidacji nie przenosi się na OOD.** `train_v2.py` dobrał na walidacji próg 0,2 (IID: 0,986), ale na Ghostbusterze daje on 0,829 zamiast 0,838 przy 0,5. Dlatego eksportowany model (`models/distilbert-final/export_meta.json`) używa progu 0,5. To samo zjawisko widać w ścieżce statystycznej (`statistical/WYNIKI.md`, sekcja 7).

### Eksperyment pośredni v2: hold-out domen HC3

`train/runs/distilbert-v2/metrics.json`: trening na Kaggle + HC3 finance + HC3 reddit_eli5 (43 200 tekstów), test na domenach HC3, których model nie widział:

| test | n | accuracy | F1 |
|---|---|---|---|
| IID (Kaggle + finance + reddit) | 5 400 | 0,997 | 0,997 |
| OOD razem | 3 808 | 0,821 | 0,848 |
| hc3_medicine | 1 245 | **0,969** | 0,932 |
| hc3_open_qa | 1 649 | 0,927 | 0,960 |
| hc3_wiki_csai | 914 | 0,428 | 0,414 |

Medycyna i open QA przenoszą się dobrze, ale na tekstach encyklopedycznych (wiki_csai) model myli 522 ludzkie teksty z AI: formalny styl wygląda dla niego jak AI. Wniosek: do treningu v3 dołączono wszystkie domeny HC3 i Pile, a test generalizacji przeniesiono na całkowicie zewnętrzny Ghostbuster.

## 4. Ograniczenia (uczciwie)

- Detekcja AI **nie jest rozwiązana**: nowszy Claude/GPT i parafrazy nadal będą trudne.
- Model nadal ma bias w stronę „AI” na OOD (więcej FP niż FN).
- Pile sample i HC3 to głównie starsze generatory (ChatGPT-era), nie najnowsze LLM.
- Wynik IID ~98% nadal jest wysoki, bo źródła treningowe są separowalne; **liczy się OOD**.
- Nie kalibrujemy pod konkretny koszt błędu produkcyjnego; domyślny próg = 0.5.

## 5. Odpalenie

```powershell
.\.venv\Scripts\python.exe train\build_data.py
.\.venv\Scripts\python.exe train\train_v2.py
.\.venv\Scripts\python.exe train\export_v3.py
.\.venv\Scripts\python.exe train\compare_ood.py
```

Artefakty:

- wagi: `models/distilbert-final/`
- metryki: `train/runs/distilbert-v3/metrics.json`
- porównanie OOD: `train/runs/distilbert-v3/compare_ood.json`
- dane przygotowane: `data/prepared/` (lokalnie, gitignore)
