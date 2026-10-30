"""Supervised Random Forest and unsupervised Transformer next-event model."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class TransformerPredictor(nn.Module):
    """Predict the next DNS feature vector from a client's previous query sequence."""
    def __init__(self, features: int, d_model: int = 32, heads: int = 4, layers: int = 2, seq_len: int = 8):
        super().__init__()
        self.project = nn.Linear(features, d_model)
        self.position = nn.Parameter(torch.zeros(1, seq_len, d_model))
        block = nn.TransformerEncoderLayer(d_model=d_model, nhead=heads, dim_feedforward=64,
                                           batch_first=True, dropout=0.1, activation="gelu")
        self.encoder = nn.TransformerEncoder(block, num_layers=layers)
        self.output = nn.Linear(d_model, features)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        z = self.project(context) + self.position[:, :context.size(1)]
        encoded = self.encoder(z)
        return self.output(encoded[:, -1, :])


def fit_transformer(context: np.ndarray, target: np.ndarray, epochs: int = 25, batch_size: int = 64):
    """Fit only on benign (context, next-event) examples; no attack labels are used."""
    torch.manual_seed(42)
    model = TransformerPredictor(target.shape[-1], seq_len=context.shape[1])
    x = torch.tensor(context, dtype=torch.float32)
    y = torch.tensor(target, dtype=torch.float32)
    loader = DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    model.train()
    for _ in range(epochs):
        for history, expected in loader:
            opt.zero_grad()
            noisy_history = history + 0.05 * torch.randn_like(history)
            loss = loss_fn(model(noisy_history), expected)
            loss.backward()
            opt.step()
    model.eval()
    return model, transformer_scores(model, context, target)


def transformer_scores(model: TransformerPredictor, context: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Mean squared prediction error for the next event; this is an anomaly score, not a probability."""
    model.eval()
    with torch.no_grad():
        x = torch.tensor(context, dtype=torch.float32)
        y = torch.tensor(target, dtype=torch.float32)
        return ((model(x) - y) ** 2).mean(dim=1).numpy()

