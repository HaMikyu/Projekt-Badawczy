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

