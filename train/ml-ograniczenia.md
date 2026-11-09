# Ograniczenia metody ML (detekcja tekstu AI)

Dokument uzupełnia ścieżkę ML (DistilBERT/BERT fine-tuning) i zbiera ograniczenia,
które trzeba zgłosić razem z wynikami. Liczby pochodzą z `train/runs/` oraz z
analizy `data/Training_Essay_Data.csv`. Tabele wyników i instrukcję odtworzenia
eksperymentu zawiera [`WYNIKI.md`](./WYNIKI.md).

## 1. Zakres eksperymentu

- Klasyfikacja binarna: `human = 0`, `ai = 1`.
- Model: `distilbert-base-uncased` (67 mln parametrów), fine-tuning głowicy
  klasyfikacyjnej na całym modelu.
- Jeden seed (`42`) i jeden podział danych; wyniki są powtarzalne tylko w tym
  samym środowisku (PyTorch + ROCm na Windows).
- Zbiór testowy nie uczestniczy w strojeniu hiperparametrów ani w early stopping.

## 1a. Strojenie hiperparametrów

Strojenie jest jawną siatką (`train/tune.py --execute`), a nie pojedynczym runem:

| próba | lr | batch | max_length | best val F1 | epoka | czas |
|---|---|---|---|---|---|---|
| 1 | 2e-5 | 32 | 256 | **0.99821** | 4 | 186 s |
| 2 | 3e-5 | 32 | 256 | 0.99821 | 3 | 195 s |
| 3 | 4e-5 | 64 | 256 | 0.99821 | 3 | 212 s |
| 4 | 2e-5 | 32 | 128 | 0.99599 | 4 | 157 s |

Wybór: **lr = 2e-5, batch = 32, max_length = 256** (najwyższy `val_f1`, remis
rozstrzygany na korzyść krótszego czasu). Liczba epok nie jest zgadywana —
próby mają limit 4 epok, a finalny trening ma 20 epok i early stopping
(`patience = 5`) na `val_f1`. Test nie był dotykany podczas prób
(`--skip-test`).

## 1b. Wynik finalny (test, po czyszczeniu)

| metryka | wartość |
|---|---|
| accuracy | **0.99707** |
| precision | **1.00000** |
| recall | **0.99284** |
| F1 | **0.99641** |

Macierz pomyłek: `[[1611, 0], [8, 1110]]` — 8 błędów na 2 729 próbek, wszystkie
w stronę AI → human (zero fałszywych alarmów). Best `val_f1 = 0.99911` w
epoce 7; trening zatrzymał się po epoce 12 (5 epok bez poprawy). Cały finalny
trening: **8,4 min**.

## 2. Przeciek danych i czyszczenie

Przed poprawką split był losowy po wierszach, a w danych było dużo dokładnych
duplikatów:

- 29 145 wierszy wejściowych,
- 1 842 grupy zduplikowanych tekstów (1 846 wierszy do usunięcia),
- 3 wiersze krótsze niż 5 słów,
- po czyszczeniu: **27 296 wierszy**,
- przed poprawką **10,4% testu miało dokładny duplikat w train** (etykiety zgodne
  w 100%),
- po poprawce: **0 wspólnych tekstów** train/validation/test.

To oznacza, że wcześniejsze wyniki (~99,8% F1) były częściowo memorowaniem, a nie
generalizacją. Po czyszczeniu liczby są uczciwsze, choć nadal dotyczą tego samego,
stosunkowo łatwego zbioru.

**Czego nadal nie usunięto:** near-duplikatów (parafraz, tego samego tekstu z
drobinnymi zmianami). Wykrywanie ich wymagałoby MinHash/embeddingów; to
potencjalne źródło zawyżenia wyników i pozycja do dalszej pracy.

## 3. Zależność od długości tekstu

Dane mają wyraźny confound długości:

| klasa | średnio znaków | średnio słów |
|---|---|---|
| human | 2 398 | 430 |
| ai | 1 983 | 312 |

Sam klasyfikator oparty **wyłącznie na liczbie słów** osiąga
**AUC ≈ 0,695** (krótszy tekst → AI). Model transformerowy może więc częściowo
uczyć się długości/formatu, a nie stylu. Dlatego raportujemy metryki w koszykach
długości (kwartyle), a nie tylko zbiorczo:

| kwartyl długości (słowa) | liczba próbek | accuracy | precision | recall | F1 |
|---|---|---|---|---|---|
| Q1 (61–251) | 692 | 0.99855 | 1.00000 | 0.99746 | 0.99873 |
| Q2 (251–345) | 678 | 0.99410 | 1.00000 | 0.98892 | 0.99443 |
| Q3 (345–464) | 677 | 0.99557 | 1.00000 | 0.98790 | 0.99391 |
| Q4 (464–1436) | 682 | 1.00000 | 1.00000 | 1.00000 | 1.00000 |

Finalny model nie jest więc „głupszy na krótkich tekstach” — najsłabiej wypada w
środkowych kwartylach, najlepiej na najdłuższych. Sam `length_only_auc = 0.691`
pokazuje jednak, że długość niesie realną informację i nie można twierdzić, że
model używa wyłącznie stylu. Wniosek do raportu: metryki zbiorcze trzeba
uzupełnić tym rozbiciem, a porównanie ze ścieżką statystyczną zrobić na tych
samych koszykach.

## 4. Overfitting i koszt

- Model zbiega bardzo szybko: `val_loss` spada do ~0,0066 już w 4. epoce,
  a minimum (0,00468) osiąga w epoce 7.
- `train_loss` schodzi do ~1,3e-6, podczas gdy `val_loss` pozostaje ~0,006 —
  typowe przeuczenie; early stopping i tak wybiera wczesny checkpoint.
- 20 epok to bezpieczny limit, nie konieczność: finalny trening zatrzymał się
  na 12. epoce (5 epok bez poprawy `val_f1`).
- Po dodaniu gradient clippingu i niższego LR krzywa jest stabilna: najgorszy
  epizod to epoka 3 (`val_f1` 0,9933, `val_loss` 0,0172), a nie dawny skok do
  `val_f1` 0,957 i `val_loss` 0,124.
- Koszt obliczeniowy: na RX 9070 XT (ROCm) DistilBERT + bf16 + `batch_size=32`
  + `max_length=256` daje ~40 s na epokę; finalny trening (12 epok) to
  **~8,4 min**, a pojedyncza próba strojenia 1,7–3,5 min. Stary wariant
  (BERT-base, fp32, `batch_size=16`, `max_length=128`) potrzebował ~9 min
  **na epokę**.
- Koszt inferencji jest mały (jeden forward na tekst), ale fine-tuning wymaga
  GPU; na CPU trening byłby niepraktyczny.
- Większa `max_length` (256/512) zwiększa koszt attention i liczbę kroków;
  128 obcina większość esejów (średnio ~430 słów human), więc wybór długości
  jest kompromisem między jakością a kosztem. W strojeniu `max_length=128`
  wypadło wyraźnie gorzej (`val_f1` 0,9960 vs 0,9982).

## 5. Domena i generalizacja

- Dane to **angielskie eseje** z konkretnego źródła/rozkładu. Model może uczyć się
  specyfiki tego zbioru (tematy, sposób generowania, długość), a nie uniwersalnego
  „stylu AI”.
- Nie mamy zewnętrznego zbioru testowego, więc **nie znamy spadku jakości na innej
  domenie** (inne języki, inne modele LLM, inne gatunki tekstu). To trzeba
  wyraźnie zaznaczyć jako nierozwiązane ograniczenie.
- Wynik na tym samym zbiorze nie jest dowodem na detekcję „w ogóle”, tylko na
  separowalność tej konkretnej kolekcji.

## 6. Inne ograniczenia

- **Nierównowaga klas**: ~60% human / 40% AI. Accuracy jest myląca; raportujemy
  precision, recall i F1, a macierz pomyłek pokazuje, że model częściej pomija AI
  (AI → human) niż odwrotnie.
- **Szum etykiet**: w danych były pojedyncze skrajnie krótkie teksty (np. 1 słowo
  oznaczone jako AI); część usunięto, ale etykiety nie były weryfikowane ręcznie.
- **Brak kalibracji**: raportujemy argmax, nie prawdopodobieństwa; próg decyzyjny
  nie był dobierany pod koszt błędu.
- **Reproducibility**: bf16, ROCm/Windows i niedeterministyczne kernels mogą dawać
  minimalne różnice między uruchomieniami mimo tego samego seeda.
- **Brak porównania ze ścieżką statystyczną** w tej samej tabeli — zgodnie z
  zakresem projektu głównym modelem jest transformer, ale do rzetelnego wniosku
  „ML bije metodę statystyczną” potrzebne jest zestawienie na tym samym
  oczyszczonym zbiorze i tym samym podziale danych.

## 7. Co raportować

1. Metryki testowe (accuracy, precision, recall, F1) **po czyszczeniu**.
2. Informację, ile duplikatów usunięto i że po poprawce przeciek wynosi 0.
3. Metryki w kwartylach długości oraz AUC samej długości.
4. Czas treningu i inferencji oraz wymagany sprzęt.
5. Krzywą uczenia (`train_loss` vs `val_loss`) jako dowód przeuczenia i
   uzasadnienie early stopping.
6. Wyraźne zastrzeżenie: wynik dotyczy angielskich esejów z tego zbioru i nie
   przenosi się automatycznie na inne domeny.
