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

| Model | Test Kaggle (IID) | Ghostbuster (OOD) |
|---|---|---|
| Statystyczny — regresja logistyczna, 13 cech, trening Kaggle | 0,907 | 0,659 |
| DistilBERT — trening Kaggle (baseline) | 0,997 | 0,516 |
| DistilBERT v3 — trening na mieszance źródeł | 0,986* | **0,838** |

\* test IID mieszanki źródeł, nie samego Kaggle.

Uruchomienie ścieżki statystycznej:

```powershell
.\.venv\Scripts\python.exe statistical\run_experiment.py
.\.venv\Scripts\python.exe statistical\eval_ood.py
.\.venv\Scripts\python.exe statistical\predict.py "tekst do sprawdzenia"
```

Dokumentacja LaTeX: `Dokumentacja/` (kompilacja: `compile.sh`), rozdział 6 opisuje wyniki obu ścieżek.
