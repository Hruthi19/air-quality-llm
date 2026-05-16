"""PyTorch LSTM forecaster for PM2.5 prediction.

We use PyTorch (not Keras) because the production environment has CUDA-12
PyTorch and the Qwen 7B explainer also runs on PyTorch — keeping a single
framework simplifies the submit script and avoids a TF/cuDNN driver clash
on the B200 node.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class LSTMConfig:
    input_size:    int   = 5
    hidden_size:   int   = 64
    num_layers:    int   = 2
    dropout:       float = 0.2
    output_size:   int   = 1
    learning_rate: float = 1e-3
    batch_size:    int   = 64
    epochs:        int   = 5
    weight_decay:  float = 1e-6
    grad_clip:     float = 1.0
    seed:          int   = 42

    def to_dict(self) -> dict:
        return asdict(self)


class LSTMForecaster(nn.Module):
    """Two-layer stacked LSTM with dropout and a linear regression head."""

    def __init__(self, cfg: LSTMConfig):
        super().__init__()
        self.cfg = cfg
        self.lstm = nn.LSTM(
            input_size=cfg.input_size,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(cfg.dropout)
        self.head = nn.Linear(cfg.hidden_size, cfg.output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(self.dropout(last)).squeeze(-1)


def make_loaders(X_train, y_train, X_val, y_val, X_test, y_test, batch_size: int):
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    test_ds  = TensorDataset(torch.from_numpy(X_test),  torch.from_numpy(y_test))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, drop_last=False)
    val_loader = None
    if len(X_val) > 0:
        val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    return train_loader, val_loader, test_loader


def train_lstm(cfg: LSTMConfig, X_train, y_train, X_val, y_val, X_test, y_test,
               device: str = "cuda", out_dir: Path | None = None,
               tag: str = "lstm") -> dict:
    """Train one LSTM forecaster and return metrics + predictions."""
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    train_loader, val_loader, test_loader = make_loaders(
        X_train, y_train, X_val, y_val, X_test, y_test, cfg.batch_size
    )

    model = LSTMForecaster(cfg).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate,
                             weight_decay=cfg.weight_decay)
    loss_fn = nn.MSELoss()

    history = []
    t0 = time.time()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        ep_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            ep_losses.append(loss.item())
        train_mse = float(np.mean(ep_losses))

        val_mse = float("nan")
        if val_loader is not None:
            model.eval()
            with torch.no_grad():
                vs = []
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    vs.append(loss_fn(model(xb), yb).item())
                val_mse = float(np.mean(vs))

        history.append({
            "tag": tag,
            "epoch": epoch,
            "train_mse_scaled": train_mse,
            "val_mse_scaled": val_mse,
            "wall_time_s": float(time.time() - t0),
        })
        print(f"[{tag}] epoch {epoch}/{cfg.epochs} "
              f"train_mse={train_mse:.6f} val_mse={val_mse:.6f}")

    # ---------- predict on the held-out test split ----------
    model.eval()
    preds_scaled = []
    with torch.no_grad():
        for xb, _ in test_loader:
            preds_scaled.append(model(xb.to(device)).cpu().numpy())
    preds_scaled = np.concatenate(preds_scaled, axis=0)

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), out_dir / f"{tag}_weights.pt")
        pd.DataFrame(history).to_csv(out_dir / f"{tag}_train_history.csv", index=False)

    return {
        "model": model,
        "history": history,
        "preds_scaled": preds_scaled,
        "config": cfg.to_dict(),
        "wall_time_s": float(time.time() - t0),
    }
