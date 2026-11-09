# Wyniki ścieżki ML — detekcja tekstu AI

Dokument zbiera finalne wyniki eksperymentu transformerowego. Szczegółowy opis
ograniczeń znajduje się w [`ml-ograniczenia.md`](./ml-ograniczenia.md), a pełna
dokumentacja komponentów w `docs/train-components/` (folder roboczy, ignorowany
przez Git).

## 1. Model i dane

- Model: **`distilbert-base-uncased`** (67 mln parametrów), fine-tuning całego
  modelu + głowica klasyfikacyjna `human` / `ai`.
- Framework: **PyTorch Lightning** (`BertTextClassifier`).
- Seed: **42**, jeden podział train/validation/test.
- Źródło: `data/Training_Essay_Data.csv` (Kaggle), angielskie eseje.

Czyszczenie przed podziałem:

| etap | liczba wierszy |
|---|---|
| dane wejściowe | 29 145 |
| usunięte duplikaty dokładne | 1 846 |
| usunięte teksty < 5 słów | 3 |
| **po czyszczeniu** | **27 296** |

Podział po czyszczeniu: **21 838 / 2 729 / 2 729** (train / validation / test).
Liczba wspólnych tekstów między splitami: **0** (kontrola `split_overlap`).

## 2. Strojenie hiperparametrów

Uruchomienie: `python train/tune.py --execute`. Test nie jest dotykany podczas
prób (`--skip-test`), wybór następuje po `val_f1`.

| próba | learning rate | batch size | max_length | best val F1 | epoka | czas |
|---|---|---|---|---|---|---|
| 1 | 2e-5 | 32 | 256 | **0,99821** | 4 | 186 s |
| 2 | 3e-5 | 32 | 256 | 0,99821 | 3 | 195 s |
| 3 | 4e-5 | 64 | 256 | 0,99821 | 3 | 212 s |
| 4 | 2e-5 | 32 | 128 | 0,99599 | 4 | 157 s |

Wybrana konfiguracja: **lr = 2e-5, batch_size = 32, max_length = 256**.
Liczba epok jest sterowana early stoppingiem: próby mają limit 4 epok, finalny
trening — 20 epok z `patience = 5` na `val_f1`.

## 3. Wynik finalny na zbiorze testowym

| metryka | wartość |
|---|---|
| accuracy | **0,99707** |
| precision | **1,00000** |
| recall | **0,99284** |
| F1 | **0,99641** |

Macierz pomyłek (human / ai):

```text
[[1611,    0],
 [   8, 1110]]
```

8 błędów na 2 729 próbek, wszystkie w stronę `ai → human`; zero fałszywych
alarmów. Najlepszy `val_f1 = 0,99911` w epoce 7. Trening zatrzymał się po
epoce 12 (5 epok bez poprawy), czas: **8,4 min**.

Krzywa jest stabilna: najgorszy epizod to epoka 3 (`val_f1 = 0,9933`,
`val_loss = 0,0172`). Nie występuje skok do `val_f1 = 0,957` / `val_loss = 0,124`,
który pojawiał się przed dodaniem gradient clippingu i obniżeniem LR.

## 4. Zależność od długości tekstu

| kwartyl długości (słowa) | próbki | accuracy | precision | recall | F1 |
|---|---|---|---|---|---|
| Q1 (61–251) | 692 | 0,99855 | 1,00000 | 0,99746 | 0,99873 |
| Q2 (251–345) | 678 | 0,99410 | 1,00000 | 0,98892 | 0,99443 |
| Q3 (345–464) | 677 | 0,99557 | 1,00000 | 0,98790 | 0,99391 |
| Q4 (464–1436) | 682 | 1,00000 | 1,00000 | 1,00000 | 1,00000 |

Klasyfikator oparty **wyłącznie na liczbie słów** osiąga `AUC = 0,691`
(krótszy tekst → AI). Oznacza to, że długość niesie realną informację i model
może częściowo uczyć się formatu, a nie wyłącznie stylu.

## 5. Ograniczenia

- **Koszt:** fine-tuning wymaga GPU. Na RX 9070 XT (ROCm, bf16) ok. 40 s/epokę,
  pełny trening 8,4 min. Starszy BERT-base we fp32 potrzebował ok. 9 min/epokę.
- **Overfitting:** `train_loss` spada do ~1,3e-6, podczas gdy `val_loss`
  pozostaje ~0,006. Early stopping wybiera wczesny checkpoint.
- **Domena:** wynik dotyczy angielskich esejów z jednego zbioru. Brak
  zewnętrznego testu — nie znamy spadku jakości na innych językach, modelach LLM
  ani gatunkach tekstu.
- **Długość tekstu:** patrz sekcja 4; to confound, nie można twierdzić, że model
  wykrywa wyłącznie „styl AI”.
- **Near-duplikaty:** usunięto tylko dokładne duplikaty; parafrazy mogą nadal
  zawyżać wynik.
- **Nierównowaga klas:** ok. 60% human / 40% AI; raportujemy precision, recall
  i F1, a nie tylko accuracy.
- **Szum etykiet i kalibracja:** etykiety nie były weryfikowane ręcznie,
  raportowany jest argmax bez kalibracji progu.
- **Brak porównania ze ścieżką statystyczną:** wymaga wyników na tym samym
  oczyszczonym zbiorze i tym samym podziale; do uzupełnienia po stronie ścieżki
  statystycznej.

## 6. Artefakty i odtwarzanie

- Wagi finalnego modelu: `models/distilbert-final/` (Hugging Face
  `safetensors`, ok. 255 MB, etykiety `human` / `ai`).
- Checkpoint Lightning: `train/runs/distilbert-final/checkpoints/best.ckpt`
  (folder runów jest ignorowany przez Git).
- Metryki i konfiguracja: `train/runs/distilbert-final/metrics.json`,
  `config.json`, `history.jsonl`.

```powershell
python train/tune.py --execute
python train/train.py --run-name distilbert-final
python train/export_model.py
python train/dashboard/server.py
```
