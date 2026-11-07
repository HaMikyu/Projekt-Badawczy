import torch
from torch.utils.data import DataLoader, random_split
import lightning as L
from dataset import TextDataset
from helpers import load_kaggle_data


class TextDataModule(L.LightningDataModule):
    def __init__(
        self,
        texts=None,
        labels=None,
        batch_size=32,
        val_ratio=0.2,
        seed=42,
    ):
        super().__init__()
        self.texts = texts
        self.labels = labels
        self.batch_size = batch_size
        self.val_ratio = val_ratio
        self.seed = seed
        self.train_set = None
        self.val_set = None
        self.full_set = None

    def setup(self, stage=None):
        if self.texts is None or self.labels is None:
            self.texts, self.labels = load_kaggle_data(seed=self.seed)
        self.full_set = TextDataset(self.texts, self.labels)
        n_val = max(1, int(len(self.full_set) * self.val_ratio))
        n_train = len(self.full_set) - n_val
        gen = torch.Generator().manual_seed(self.seed)
        self.train_set, self.val_set = random_split(
            self.full_set, [n_train, n_val], generator=gen
        )

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.full_set, batch_size=self.batch_size)


if __name__ == "__main__":
    dm = TextDataModule(batch_size=8, seed=42)
    dm.setup()
    print("datamodule smoke test (kaggle)")
    print(f"train={len(dm.train_set)} val={len(dm.val_set)} full={len(dm.full_set)}")
    xb, yb = next(iter(dm.train_dataloader()))
    print(f"batch x={tuple(xb.shape)} y={yb.tolist()}")
