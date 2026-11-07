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
