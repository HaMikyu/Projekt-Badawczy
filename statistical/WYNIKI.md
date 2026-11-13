# Wyniki ścieżki statystycznej — detekcja tekstu AI

Dokument zbiera finalne wyniki eksperymentu stylometrycznego (ścieżka statystyczna, Michał Niechciałkowski). Szczegółowy opis ograniczeń znajduje się w [`stat-ograniczenia.md`](./stat-ograniczenia.md), a kod implementacji w `statistical/`.

---

## 1. Zbiór danych i metodologia

Eksperyment przeprowadzono na **dokładnie tym samym zbiorze danych** i **identycznym podziale** co ścieżka machine learning Dominika Zabielskiego (`train/WYNIKI.md`), co gwarantuje pełną poprawność metodologiczną porównania:

- **Źródło:** `data/Training_Essay_Data.csv` (Kaggle), angielskie eseje.
- **Czyszczenie przed podziałem:** usunięcie dokładnych duplikatów (1 862 wiersze) oraz tekstów poniżej 5 słów (3 wiersze). Z 29 145 wierszy wejściowych pozostało **27 280** unikalnych tekstów.
- **Seed i podział:** Seed **42**, podział 80% / 10% / 10%:
  - **Train:** 21 824 próbki
  - **Validation:** 2 728 próbek
  - **Test:** 2 728 próbek (odłożony zbiór testowy, nietknięty podczas doboru parametrów)
- **Overlap:** Liczba wspólnych znormalizowanych tekstów między splitami wynosi **0**.

---

## 2. Wyekstrahowane cechy stylometryczne

Dla każdego tekstu obliczono 13 cech stylometrycznych. Średnie wartości dla klas `Human` (0) i `AI` (1) na zbiorze treningowym:

| Cecha | Średnia Human | Średnia AI | Różnica (AI - Human) | Interpretacja |
|---|---|---|---|---|
| `avg_sentence_len` | 25,94 | 22,99 | -2,95 | Ludzie piszą średnio dłuższe zdania |
| `std_sentence_len` | 13,03 | 8,44 | -4,59 | U ludzi znacznie większy rozrzut długości zdań |
| `burstiness_cv` | 0,491 | 0,353 | -0,138 | **U ludzi wyższa porywczość (zmienność) zdań** |
| `burstiness_diff` | 15,33 | 9,00 | -6,33 | **U ludzi wyższa skokowość między kolejnymi zdaniami** |
| `ttr` | 0,447 | 0,556 | +0,109 | Wyższy surowy TTR u AI (związany ze średnią długością eseju) |
| `root_ttr` | 8,74 | 9,44 | +0,71 | Indeks Guirauda (skorygowany TTR) |
| `stopword_ratio` | 0,500 | 0,438 | -0,062 | Nieco wyższy udział słów funkcyjnych u ludzi |
| `avg_word_len` | 4,42 | 5,11 | +0,69 | AI używa średnio dłuższych wyrazów |
| `punct_ratio` | 0,020 | 0,018 | -0,002 | Zbliżony poziom interpunkcji ogólnej |
| `comma_per_sentence` | 0,827 | 0,990 | +0,163 | AI wstawia więcej przecinków na zdanie |
| `exclaim_question_ratio` | 0,00041 | 0,00026 | -0,00015 | Ludzie częściej stosują wykrzykniki i pytajniki |
| `uppercase_ratio` | 0,0172 | 0,0112 | -0,0060 | Ludzie częściej używają wielkich liter (nazwy własne, skróty) |
| `digit_ratio` | 0,00286 | 0,00066 | -0,00220 | Ludzie częściej wplatają cyfry |

---

## 3. Wyniki klasyfikacji na zbiorze testowym

Przetestowano dwa warianty decyzyjne oparte wyłącznie na wyekstrahowanych cechach:

1. **Model A (Progi / Reguły decyzyjne):**
   Wygenerowane drzewo progowe o głębokości 2:
   - Jeśli `avg_word_len <= 4.85`:
     - Jeśli `burstiness_cv <= 0.32` $\to$ **AI**
     - Jeśli `burstiness_cv > 0.32` $\to$ **Human**
   - Jeśli `avg_word_len > 4.85` $\to$ **AI**

2. **Model B (Regresja Logistyczna ze standaryzacją StandardScaler):**
   W pełni znormalizowany model liniowy z wagami cech i wyjściem probabilistycznym.

### Zestawienie metryk na zbiorze testowym:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| **Model A (Progi decyzyjne)** | 0,85337 | 0,79187 | 0,87120 | 0,82964 | — |
| **Model B (Regresja Logistyczna)** | **0,90689** | **0,86986** | **0,90877** | **0,88889** | **0,96836** |

Macierz pomyłek Regresji Logistycznej:
```text
[[1458,  152],   <- Human (TN: 1458, FP: 152)
 [ 102, 1016]]   <- AI    (FN: 102,  TP: 1016)
```

Z 2 728 próbek testowych model statystyczny poprawnie sklasyfikował **2 474 teksty** (90,7% skuteczności).

---

## 4. Wagi cech w Regresji Logistycznej (Interpretowalność)

Znormalizowane wagi modelu liniowego pozwalają dokładnie wskazać, które sygnały stylometryczne decydują o werdykcie:

| Cecha stylometryczna | Waga znormalizowana | Kierunek decyzji |
|---|---|---|
| `avg_word_len` | **+3,0771** | $\to$ **AI** (bardziej złożone słownictwo) |
| `std_sentence_len` | **+3,0479** | $\to$ **AI** |
| `avg_sentence_len` | **-2,7443** | $\to$ **Human** (dłuższe zdania) |
| `burstiness_diff` | **-2,5636** | $\to$ **Human** (skokowa zmienność długości sąsiednich zdań) |
| `burstiness_cv` | **-1,7203** | $\to$ **Human** (wysoka zmienność rytmu) |
| `ttr` | **+1,4714** | $\to$ **AI** |
| `uppercase_ratio` | **-1,4529** | $\to$ **Human** |
| `root_ttr` | **-1,3612** | $\to$ **Human** |
| `comma_per_sentence` | **+0,9393** | $\to$ **AI** |
| `digit_ratio` | **-0,5773** | $\to$ **Human** |
| `stopword_ratio` | **+0,3080** | $\to$ **AI** |
| `exclaim_question_ratio` | **+0,2206** | $\to$ **AI** |
| `punct_ratio` | **+0,0730** | $\to$ **AI** |

---

## 5. Analiza wrażliwości na długość tekstu

Podział odłożonego zbioru testowego na kwartyle długości słownej potwierdza silną wrażliwość metody statystycznej na objętość analizowanego materiału:

| Kwartyl długości (słowa) | Liczba próbek | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|---|
| **Q1 (50–256)** | 685 | 0,84088 | 0,83846 | 0,87668 | 0,85714 |
| **Q2 (256–345)** | 680 | 0,89559 | 0,90000 | 0,90736 | 0,90366 |
| **Q3 (345–453)** | 681 | 0,93098 | 0,87755 | 0,95911 | 0,91652 |
| **Q4 (453–1650)** | 682 | **0,96041** | 0,85965 | 0,89908 | 0,87892 |

### Wnioski z analizy wrażliwości:
1. **Wyraźna zależność od objętości tekstu:** Dokładność klasyfikatora statystycznego rośnie monotonicznie wraz z długością: od **84,1%** dla najkrótszych esejów (Q1) do **96,0%** dla najdłuższych (Q4).
2. Na krótkich tekstach estymatory wariancji zdań (`burstiness`) oraz bogactwa słownikowego (`TTR`) cechują się dużą wariancją próbkową, co powoduje ponad dwukrotnie wyższy odsetek błędów (15,9% błędów w Q1 vs 4,0% w Q4).
3. Samodzielny klasyfikator bazujący wyłącznie na liczbie słów osiąga `AUC = 0,686` (zbieżnie z wynikiem 0,691 raportowanym w ścieżce ML), potwierdzając rolę długości jako zmiennej zakłócającej (*confounder*).

---

## 6. Porównanie ścieżki statystycznej ze ścieżką Machine Learning

Zestawienie bezpośrednie obu ścieżek na tym samym zbiorze testowym (2 728 próbek, `seed=42`):

| Metryka / Cecha | Ścieżka Statystyczna (Regresja Logistyczna) | Ścieżka ML (DistilBERT Fine-tuning) |
|---|---|---|
| **Accuracy** | 0,90689 | **0,99707** |
| **Precision** | 0,86986 | **1,00000** |
| **Recall** | 0,90877 | **0,99284** |
| **F1-Score** | 0,88889 | **0,99641** |
| **Liczba błędów (na 2 728)** | 254 | **8** |
| **Czas ekstrakcji / treningu** | **~25 sekund (CPU)** | ~8,4 min (GPU RX 9070 XT) |
| **Liczba parametrów** | **14 wag** | 67 mln parametrów |
| **Interpretowalność** | Pełna (znana waga każdej cechy) | Ograniczona (czarna skrzynka) |


> Ta tabela porównuje z modelem DistilBERT trenowanym **tylko na Kaggle** (baseline z commita `6158bbf`). Aktualny model ML (`distilbert-v3`) jest trenowany na mieszance źródeł — porównanie z nim jest w sekcji 7.

---

## 7. Test poza domeną (Ghostbuster OOD)

Wynik na Kaggle jest zawyżony, bo zbiór jest jednorodny. Dlatego model statystyczny oceniono też na **tym samym zbiorze OOD co `distilbert-v3`**: Ghostbuster, 1 892 teksty (946 human / 946 Claude + GPT), nigdy nieobecny w treningu.

Sprawdzono dwa warianty regresji logistycznej:
- **Kaggle** — model z `runs/model.joblib` (sekcje 3–6),
- **mieszanka** — ta sama regresja trenowana na `data/prepared/train.csv` (te same 59 014 tekstów co `distilbert-v3`: Kaggle + HC3 + AI Text Detection Pile, length-matching).

| Model | Trening | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| Statystyczny (reg. log.) | Kaggle | 0,659 | 0,600 | 0,952 | 0,736 | 0,838 |
| Statystyczny (reg. log.) | mieszanka | 0,520 | 0,510 | 0,988 | 0,673 | 0,868 |
| DistilBERT baseline | Kaggle | 0,516 | 0,508 | 0,997 | 0,673 | — |
| **DistilBERT v3** | mieszanka | **0,838** | **0,767** | 0,970 | **0,857** | — |

Statystyczny na mieszance, test IID (7 376 tekstów): accuracy **0,747**, F1 0,748, ROC-AUC 0,819 (DistilBERT v3: 0,986).

Accuracy na OOD w podziale na źródło:

| Źródło Ghostbuster | Stat. (Kaggle) | Stat. (mieszanka) |
|---|---|---|
| human | **0,366** | 0,052 |
| claude | 0,964 | 0,986 |
| gpt | 0,975 | 1,000 |
| gpt_prompt1 | 0,945 | 0,982 |
| gpt_prompt2 | 0,893 | 0,964 |
| gpt_semantic | 0,961 | 0,994 |
| gpt_writing | 0,946 | 0,991 |

### Wnioski

1. **Poza domeną stylometria trzyma się lepiej niż DistilBERT trenowany na jednym źródle** (0,659 vs 0,516). 13 jawnych cech słabiej przeucza się do artefaktów jednego CSV niż 67 mln parametrów.
2. **Stylometria nie skaluje się na różnorodne dane.** Po dodaniu HC3 i Pile accuracy IID spada z 0,907 do 0,747, a OOD do 0,520 — 13 cech nie opisze naraz stylu esejów, Reddita, finansów i medycyny.
3. **Wszystkie modele na OOD przechylają się w stronę „AI”.** Model statystyczny z Kaggle rozpoznaje poprawnie tylko 36,6% ludzkich tekstów Ghostbuster.
4. **Problemem jest głównie próg, nie ranking.** ROC-AUC 0,84–0,87 przy accuracy 0,52–0,66 oznacza, że model sensownie sortuje teksty, ale próg 0,5 nie pasuje do nowej domeny. Następny krok: kalibracja progu na małej próbce z docelowej domeny.

Odtworzenie (wymaga `train/build_data.py` i `statistical/run_experiment.py`):

```powershell
.\.venv\Scripts\python.exe statistical\eval_ood.py
```

Wyniki: `statistical/runs/ood_results.json`.
