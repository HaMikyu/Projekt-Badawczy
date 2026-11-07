import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from helpers import extract_features, FEATURE_NAMES, ensure_kaggle_csv, load_kaggle_data


class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = list(texts)
        self.labels = list(labels)
        self.features = [
            extract_features(t)
            for t in tqdm(self.texts, desc="features")
        ]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        x = torch.tensor(self.features[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


if __name__ == "__main__":
    csv_path = ensure_kaggle_csv()
    full = pd.read_csv(csv_path)
    print("dataset smoke test")
    print(f"csv: {csv_path}")
    print(f"rows: {len(full)}")
    print(f"cols: {list(full.columns)}")
    print(full["generated"].value_counts().to_string())
    texts, labels = load_kaggle_data(seed=42)
    ds = TextDataset(texts, labels)
    print(f"len={len(ds)} human={labels.count(0)} ai={labels.count(1)}")
    x, y = ds[0]
    print(f"sample0 label={int(y)} shape={tuple(x.shape)}")
    for name, value in zip(FEATURE_NAMES, x.tolist()):
        print(f"  {name}: {value:.4f}")
