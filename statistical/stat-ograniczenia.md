# Ograniczenia metody statystycznej (detekcja tekstu AI)

Dokument stanowi podsumowanie ograniczen metody statystyczno-stylometrycznej przygotowane przez Michala Niechcialkowskiego, bedace bezposrednim odpowiednikiem i uzupelnieniem analizy ograniczen metody ML Dominika Zabielskiego (`train/ml-ograniczenia.md`).

Eksperymenty przeprowadzono na zbiorze esejow z Kaggle (`data/Training_Essay_Data.csv`) przy uzyciu tego samego podzialu danych (`seed=42`, 80/10/10) i identycznych metryk ewaluacyjnych.

---

## 1. Wrazliwosc na dlugosc tekstu (Length Confounding i niestabilnosc probkowa)

* **Degradacja na krotkich tekstach:** Cechy statystyczne (takie jak Type-Token Ratio - TTR, wspolczynnik zmiennosci dlugosci zdan - burstiness czy udzial przecinkow na zdanie) wymagaja odpowiedniej liczby probek tekstowych (zdan i tokenow), aby estymatory byly stabilne. Dla tekstow ponizej 100-150 slow (np. krotkie maile, posty w mediach spolecznosciowych, pojedyncze akapity) rozklad dlugosci zdan staje sie zaszumiony, a prawo wielkich liczb przestaje dzialac.
* **Artefakt TTR:** Surowy wskaznik Type-Token Ratio ($V/N$) jest matematycznie zalezny od dlugosci tekstu $N$ — w miare wydluzania tekstu $V$ rosnie wolniej niz $N$ (prawo Herbana), przez co dluzsze eseje naturalnie maja nizszy TTR niezaleznie od tego, czy autorem jest czlowiek, czy AI. Zastosowanie korekt (np. Root-TTR / indeks Guirauda) czesciowo lagodzi ten problem, ale go nie eliminuje.
* **Korelacja dlugosci ze zrodlem tekstu:** Podobnie jak w sciezce ML, w zbiorze uczacym teksty generowane przez ludzi byly srednio dluzsze niz teksty AI. Istnieje ryzyko, ze model statystyczny czesciowo uczy sie korelacji pozornej miedzy dlugoscia eseju a autorem.

---

## 2. Podatnosc na proste ataki i modyfikacje (Adversarial Robustness)

* **Inzynieria promptu (Prompt Engineering):** W metodach glebokich transformer uczy sie zlozonych zaleznosci atencyjnych w osadzeniach tokenow. W metodzie statystycznej zaleznosc opiera sie na jawnych formulach. Wystarczy podac modelowi LLM w prompcie instrukcje:
  > *"Napisz esej uzywajac naprzemiennie bardzo krotkich i bardzo dlugich zdan, unikaj typowych spójnikow i stosuj bogate, zróznicowane synonimy."*
  Taki zabieg natychmiast zwieksza burstiness i TTR wygenerowanego tekstu, powodujac ze klasyfikator statystyczny blednie kwalifikuje tekst jako ludzki (False Negative).
* **Parafrazowanie i edycja hybrydowa:** Przepuszczenie tekstu AI przez prosty parafrazator (np. QuillBot) lub reczne wstawienie kilku zdan o nieregularnej strukturze calkowicie zamazuje statystyczny profil tekstu.

---

## 3. Brak modelowania semantyki, kontekstu i logiki

* **Analiza wylacznie powierzchniowa (Surface Features):** Stylometria bada jedynie cechy leksykalno-syntaktyczne (dlugosci, proporcje, znaki interpunkcyjne). Nie posiada zadnej wiedzy o:
  - logicznej spojnosci wywodu,
  - faktografii i halucynacjach (typowych dla LLM),
  - glebszych strukturach dyskursu.
  Tekst moze byc zupelnie bezsensownym zlepkiem zdan, lecz jesli jego rozklad dlugosci i udzial stop-words przypomina profil czlowieka, model uzna go za tekst ludzki.

---

## 4. Zaleznosc od domeny i rejestru jezykowego (Domain Bias)

* **Falszywe alarmy na tekstach formalnych (False Positives):** Czlowiek piszacy prace scisla, notatke prawnicza, dokumentacje techniczna lub artykul encyklopedyczny stosuje styl bardzo uporzadkowany, o malo zroznicowanej dlugosci zdan i powtarzalnym, specjalistycznym slownictwie. Stylometrycznie taki tekst wykazuje niski burstiness i umiarkowany TTR, przez co klasyfikator statystyczny ma tendencje do falszywego oskarzenia czlowieka o uzycie AI.
* **Potwierdzenie empiryczne (Ghostbuster OOD):** model wytrenowany na esejach Kaggle osiaga poza domena accuracy 0,659 i poprawnie rozpoznaje tylko 36,6% ludzkich tekstow (reszta = falszywe alarmy). Trening na mieszance zrodel (HC3, Pile) nie pomaga: accuracy IID spada do 0,747, a OOD do 0,520. ROC-AUC na OOD pozostaje jednak 0,84-0,87, wiec glownym problemem jest prog decyzyjny, ktory trzeba kalibrowac pod domene. Szczegoly: `WYNIKI.md`, sekcja 7.
* **Eseje uczniowskie vs literatura:** Badany model zostal wytrenowany na esejach. Zastosowanie go do poezji, dialogow, transkrypcji mowy czy kodu zrodlowego bedzie calkowicie niewiarygodne.

---

## 5. Bariera jezykowa i brak uniwersalnosci

* **Zaleznosc od jezyka angielskiego:** Cechy takie jak lista `STOPWORDS`, reguly segmentacji zdan czy typowa dlugosc slow sa silnie zalezne od jezyka.
* Przeniesienie metody na jezyk polski wymagaloby:
  - uzycia lematyzatora z uwagi na bogata fleksje (inaczej TTR traci porownywalnosc),
  - uwzglednienia zupelnie innej interpunkcji i swobodniejszego szyku zdan,
  - opracowania nowej bazy stop-words.

---

## 6. Podsumowanie porownawcze: Statystyka vs Machine Learning

| Kryterium | Sciezka Statystyczna (Regresja / Progi) | Sciezka ML (DistilBERT Fine-tuning) |
|---|---|---|
| **Zapotrzebowanie obliczeniowe** | Minimalne (CPU, kilka sekund na 27 tys. tekstow) | Wysokie (wymaga GPU, ~8-10 minut treningu) |
| **Interpretowalnosc** | Pelna (znane wagi kazdej cechy, jawne zasady decyzyjne) | Czarna skrzynka (trudna interpretacja atencji 67M parametrow) |
| **Pojemnosc modelu (Accuracy/F1)** | Nizsza/umiarkowana (ograniczona recznie zdefiniowanymi cechami) | Bardzo wysoka (model uczy sie zlozonych zaleznosci wielotokenowych) |
| **Odpornosc na modyfikacje promptu** | Bardzo niska (latwe zmanipulowanie statystyk dlugosci) | Umiarkowana (model bada rowniez powiazania semantyczne) |
| **Wymagania danych** | Dziala nawet przy malych zbiorach uczacych | Wymaga duzych, reprezentatywnych korpusow |
| **Generalizacja poza domene (Ghostbuster)** | 0,659 acc. przy treningu na Kaggle (lepiej niz ML z Kaggle: 0,516) | 0,838 acc. przy treningu na mieszance (DistilBERT v3) |

