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

Szczegóły liczb: [`WYNIKI.md`](./WYNIKI.md).
