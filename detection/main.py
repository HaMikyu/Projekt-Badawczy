import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from helpers import FEATURE_NAMES, extract_features, load_kaggle_data, ensure_kaggle_csv
from datamodule import TextDataModule
from model import AiTextDetector


def print_dataset_stats(texts, labels):
    print("=== KAGGLE DATASET ===")
    print(f"csv: {ensure_kaggle_csv()}")
    print(f"tekstow: {len(texts)}")
    print(f"human: {labels.count(0)}")
    print(f"ai: {labels.count(1)}")
    lengths = [len(t.split()) for t in texts]
    print(f"slowa: min={min(lengths)}, max={max(lengths)}, avg={np.mean(lengths):.1f}")
    print()

    n_stats = min(500, len(texts))
    feats = np.stack([extract_features(t) for t in texts[:n_stats]])
    y = np.array(labels[:n_stats])
    print(f"=== SREDNIE CECHY na {n_stats} tekstach ===")
    print(f"{'cecha':22} {'human':>10} {'ai':>10}")
    for i, name in enumerate(FEATURE_NAMES):
        h = feats[y == 0, i].mean() if (y == 0).any() else float("nan")
        a = feats[y == 1, i].mean() if (y == 1).any() else float("nan")
        print(f"{name:22} {h:10.4f} {a:10.4f}")
    print()


def print_examples(texts, labels, n=2):
    print("=== PRZYKLADY Z KAGGLE ===")
    for lab, title in [(0, "HUMAN"), (1, "AI")]:
        print(f"-- {title} --")
        shown = 0
        for t, y in zip(texts, labels):
            if y != lab:
                continue
            print(f"  {t[:120].replace(chr(10), ' ')}...")
            shown += 1
            if shown >= n:
                break
    print()


def evaluate(model, texts, labels):
    preds = []
    for t in texts:
        out = model.predict_text(t)
        preds.append(1 if out["label"] == "ai" else 0)
    acc = accuracy_score(labels, preds)
    p, r, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )
    cm = confusion_matrix(labels, preds)
    print("=== WYNIKI ===")
    print(f"accuracy:  {acc:.3f}")
    print(f"precision: {p:.3f}")
    print(f"recall:    {r:.3f}")
    print(f"f1:        {f1:.3f}")
    print("confusion matrix [[tn, fp], [fn, tp]]:")
    print(cm)
    print()


def demo_predictions(model, texts, labels):
    print("=== DEMO PREDYKCJI ===")
    idxs = [0, 1, len(texts) // 2, len(texts) - 2, len(texts) - 1]
    for i in idxs:
        out = model.predict_text(texts[i])
        true = "ai" if labels[i] else "human"
        print(f"true={true:5} pred={out['label']:5} p_ai={out['prob_ai']:.3f}")
        print(f"  text: {texts[i][:90].replace(chr(10), ' ')}...")
    print()


def main():
    L.seed_everything(42)

    texts, labels = load_kaggle_data(seed=42)
    print_dataset_stats(texts, labels)
    print_examples(texts, labels)

    dm = TextDataModule(
        texts,
        labels,
        batch_size=32,
        val_ratio=0.2,
        seed=42,
    )
    model = AiTextDetector(hidden_size=64, lr=1e-3)

    trainer = L.Trainer(
        max_epochs=15,
        accelerator="cpu",
        enable_checkpointing=False,
        logger=False,
        callbacks=[EarlyStopping(monitor="val_loss", patience=4, mode="min")],
    )
    print("=== TRENING ===")
    trainer.fit(model, datamodule=dm)
    print()

    evaluate(model, texts, labels)
    demo_predictions(model, texts, labels)


if __name__ == "__main__":
    main()
