import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from helpers import FEATURE_NAMES, extract_features


class AiTextDetector(L.LightningModule):
    def __init__(self, input_size=None, hidden_size=32, lr=1e-3):
        super().__init__()
        if input_size is None:
            input_size = len(FEATURE_NAMES)
        self.save_hyperparameters()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2),
        )
        self.lr = lr

    def forward(self, x):
        return self.net(x)

    def _step(self, batch, name):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = logits.argmax(dim=1)
        acc = (preds == y).float().mean()
        self.log(f"{name}_loss", loss, prog_bar=True)
        self.log(f"{name}_acc", acc, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def predict_text(self, text):
        self.eval()
        feats = extract_features(text)
        x = torch.tensor(feats, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits = self(x)
            probs = F.softmax(logits, dim=1)[0]
        label = int(probs.argmax().item())
        return {
            "label": "ai" if label == 1 else "human",
            "prob_human": float(probs[0].item()),
            "prob_ai": float(probs[1].item()),
            "features": {n: float(v) for n, v in zip(FEATURE_NAMES, feats)},
        }


if __name__ == "__main__":
    model = AiTextDetector(hidden_size=16, lr=1e-2)
    x = torch.randn(4, len(FEATURE_NAMES))
    y = torch.tensor([0, 1, 0, 1])
    logits = model(x)
    loss = F.cross_entropy(logits, y)
    print("model smoke test")
    print(f"params={sum(p.numel() for p in model.parameters())}")
    print(f"logits_shape={tuple(logits.shape)}")
    print(f"loss={float(loss.detach()):.4f}")
    out = model.predict_text("Overall, the proposed approach demonstrates promising experimental results.")
    print(f"pred={out['label']} p_ai={out['prob_ai']:.3f} (untrained, random-ish)")
