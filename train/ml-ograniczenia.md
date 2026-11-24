# Ograniczenia metody ML

## Najważniejsze

1. **Wynik na jednym Kaggle CSV nie mierzy detekcji AI w ogóle.**
   Baseline miał ~99,7% na Kaggle i ~51,6% na Ghostbuster OOD.
2. **Domena i generator mają znaczenie.** Inny styl (wiki, eseje Claude)
   psuje modele wytrenowane na jednym rozkładzie.
3. **Długość tekstu jest confounded.** Dlatego stosujemy length-matching.
4. **Bias „wygląda gładko = AI”** nadal występuje (więcej FP niż FN na OOD).
5. **Koszt:** fine-tuning DistilBERT wymaga GPU (tu RTX 3070 Ti, ~18 min).
6. **Brak gwarancji na nowe LLM** i teksty parafrazowane / adversarial.
7. **Okno 384 tokenów.** Model widzi tylko początek tekstu, a wstęp eseju jest
   najbardziej szablonowy. Aplikacja ocenia więc tekst fragmentami i uśrednia
   wynik. Pojedyncze fragmenty ludzkich esejów (wstęp, zakończenie) nadal bywają
   oceniane jako AI.
8. **Ludzkie eseje akademickie spoza treningu:** 7% fałszywych alarmów i 12,7%
   wyników niepewnych na `essays_2021`. Na zbiorze dev było to 3,4% fałszywych
   alarmów. Swobodny, „ludzki” styl tekstu AI potrafi oszukać model.

Szczegóły liczb: [`WYNIKI.md`](./WYNIKI.md).
