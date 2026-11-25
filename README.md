# Rozdysponowanie zadań

Projekt badawczy: detekcja tekstu generowanego przez AI

**Termin oddania: 13 listopada 2026 (piątek), 23:59**

## Michał Niechciałkowski — ścieżka statystyczna

- cechy stylometryczne tekstu (m.in. długość zdań, burstiness, TTR, udział stop-words)
- decyzja klasyfikacyjna na podstawie cech / prosty model na samych cechach (np. progi, regresja logistyczna)
- metryki: accuracy, precision, recall, F1
- analiza wrażliwości na długość tekstu
- krótki opis ograniczeń metody statystycznej

## Dominik Zabielski — ścieżka machine learning

- model transformerowy do klasyfikacji human vs AI (preferowane: DistilBERT / BERT, albo własny mały transformer)
- implementacja w PyTorch Lightning (fine-tuning albo trening od zera przy własnym transformerze)
- trening, walidacja i strojenie hiperparametrów (lr, batch size, liczba epok, max długość sekwencji)
- metryki: accuracy, precision, recall, F1
- porównanie wyniku ML z wynikiem ścieżki statystycznej (ten sam seed i podział danych)
- krótki opis ograniczeń metody ML (koszt, overfitting, zależność od domeny / długości tekstu)
- bez SVM / klasycznego TF-IDF jako głównego modelu

## Stan prac

| Ścieżka | Kod | Wyniki | Ograniczenia |
|---|---|---|---|
| Statystyczna | `statistical/` | [`statistical/WYNIKI.md`](statistical/WYNIKI.md) | [`statistical/stat-ograniczenia.md`](statistical/stat-ograniczenia.md) |
| Machine learning | `train/` | [`train/WYNIKI.md`](train/WYNIKI.md) | [`train/ml-ograniczenia.md`](train/ml-ograniczenia.md) |

Najważniejsze liczby (accuracy):

| Model | Test IID | Ghostbuster (OOD) |
|---|---|---|
| Statystyczny — regresja logistyczna, 13 cech, trening Kaggle | 0,907 | 0,659 |
| DistilBERT — trening Kaggle (baseline) | 0,997 | 0,516 |
| DistilBERT v3 — mieszanka źródeł | 0,986* | 0,838 |
| System One v3 (lokalny Jev-style, eseje) | 0,994 | 0,946 |
| **System One v4** (v3 + 23 tys. esejów IvyPanda, fragmenty) | **0,994** | **0,985** |

\* test IID mieszanki źródeł, nie samego Kaggle.

### Finalny test na ludzkich esejach spoza treningu (`essays_2021/`)

`essays_2021/` to 300 ludzkich esejów StudyCorgi z 2021 r. (sprzed ChatGPT).
Służą **wyłącznie** jako finalny test i nie są używane do treningu ani strojenia.
Rozmiar fragmentów i progi dobrano na osobnym zbiorze dev
(`train/build_dev_set.py`: eseje IvyPanda ludzkie + eseje AI na te same tematy).

| Detektor | Ludzki esej → CZŁOWIEK | NIEPEWNE | Ludzki esej → AI (fałszywy alarm) |
|---|---|---|---|
| Stara aplikacja (średnia DistilBERT-baseline + statystyczny) | — | — | 93% |
| DistilBERT baseline sam | — | — | 96% |
| Statystyczny sam | — | — | 18% |
| System One v3, średnia z fragmentów | 80% | 13% | 7% |
| **System One v4, średnia z fragmentów (aplikacja teraz)** | **99,3%** | 0,7% | **0%** |

Na zbiorze dev v4 wykrywa 100% esejów AI przy 0% fałszywych alarmów (v3: 84% / 2,7%),
a na Ghostbuster OOD 97,7% AI przy 2 fałszywych alarmach na 299 ludzkich tekstów.
`essays_2021` zawiera tylko ludzkie eseje, więc mierzy wyłącznie fałszywe alarmy.
Dev AI pochodzi z tej samej rodziny generatorów co trening v4.
Szczegóły i ograniczenia: [`train/WYNIKI.md`](train/WYNIKI.md#system-one-v4-douczenie-na-gpu-25112026).

Test zewnętrzny (gotowe benchmarki WasItAIGenerated 2026-09 i TextSight 2026: ok. 40 nowych modeli i różne gatunki,
`train/eval_external.py`) potwierdza niski odsetek fałszywych alarmów v4 (1,9–8,5% vs 17,6–20,5% dla v3).
Pokazuje też słabość: poza esejami v4 wykrywa tylko 36% surowych tekstów nowych modeli (GPTZero: 60%)
i 9% tekstów po humanizerach.

```powershell
python train\build_dev_set.py          # zbior dev (data/dev, gitignore)
python train\final_test_essays_2021.py # finalny test -> train\final_test_essays_2021.json
python train\eval_dev.py               # ocena modelu na zbiorze dev
```

Douczanie System One na GPU (instrukcja dla agenta): [`train/GPU_TRAINING.md`](train/GPU_TRAINING.md),
wykonane w v4 (`train/build_data_v4.py`, `train/train_system_one_v4.py`, `train/run_v4_variants.sh`).

Predykcja System One:

```powershell
.\.venv\Scripts\python.exe train\predict_system_one.py "tekst eseju"
```


Uruchomienie ścieżki statystycznej:

```powershell
.\.venv\Scripts\python.exe statistical\run_experiment.py
.\.venv\Scripts\python.exe statistical\eval_ood.py
.\.venv\Scripts\python.exe statistical\predict.py "tekst do sprawdzenia"
```

Dokumentacja LaTeX: `Dokumentacja/` (kompilacja: `compile.sh`), rozdział 6 opisuje wyniki obu ścieżek.
