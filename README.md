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

- model ML (MLP w Lightning albo inny wariant, np. TF-IDF + SVM)
- trening, walidacja i strojenie hiperparametrów
- metryki: accuracy, precision, recall, F1
- porównanie wyniku ML z wynikiem ścieżki statystycznej
- krótki opis ograniczeń metody ML

## Wspólne założenia

- ten sam zbiór Kaggle (`Training_Essay_Data.csv`)
- ten sam seed i podział train/val/test
- język angielski (bez SpeakLeash / PL na ten etap)
- kod i wyniki wrzucić do repozytorium do terminu
