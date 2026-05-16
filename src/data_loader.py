"""Beijing PRSA PM2.5 dataset loader and preprocessor."""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


FEATURE_COLUMNS = ["pm25", "dew", "temp", "pressure", "wind"]
RAW_TO_CLEAN = {
    "pm2.5": "pm25",
    "DEWP":  "dew",
    "TEMP":  "temp",
    "PRES":  "pressure",
    "Iws":   "wind",
    "Is":    "snow_hours",
    "Ir":    "rain_hours",
    "cbwd":  "wind_dir",
}


@dataclass
class DatasetStats:
    n_rows_raw:        int
    n_rows_clean:      int
    n_missing_pm25:    int
    pct_missing_pm25:  float
    period_start:      str
    period_end:        str
    feature_cols:      list
    feature_means:     dict
    feature_stds:      dict
    feature_mins:      dict
    feature_maxs:      dict
    train_rows:        int
    val_rows:          int
    test_rows:         int
    lookback:          int

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, default=str))


def load_prsa(csv_path: str | Path) -> pd.DataFrame:
    """Load the Beijing PRSA dataset and return a clean datetime-indexed DataFrame."""
    raw = pd.read_csv(csv_path)
    raw["datetime"] = pd.to_datetime(raw[["year", "month", "day", "hour"]])
    raw = raw.set_index("datetime").rename(columns=RAW_TO_CLEAN)
    return raw


def preprocess(
    df: pd.DataFrame,
    lookback: int = 60,
    train_frac: float = 0.8,
    val_frac: float = 0.0,
    feature_cols: list | None = None,
):
    """Forward-fill PM2.5 NaNs, clip, scale, build (X, y) sequences and split.

    Returns a dict with: scaler, X_train, y_train, X_test, y_test, df_clean,
    feature_cols, lookback, stats (DatasetStats).
    """
    feature_cols = feature_cols or FEATURE_COLUMNS
    df_features = df[feature_cols].copy()

    n_missing_pm25 = int(df_features["pm25"].isna().sum())
    pct_missing = float(n_missing_pm25 / max(1, len(df_features)))

    # Forward / backward fill missing values (consistent with the original notebook
    # but documented here so the report can cite the exact strategy).
    df_features = df_features.ffill().bfill()
    df_features = df_features.dropna()

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df_features.values)

    X, y = [], []
    for i in range(lookback, len(scaled)):
        X.append(scaled[i - lookback:i])
        y.append(scaled[i, 0])  # target = scaled pm25 at time t
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)

    # Chronological split (no shuffling — this is a time series).
    n = len(X)
    n_train = int(train_frac * n)
    n_val   = int(val_frac   * n)
    X_train, y_train = X[:n_train], y[:n_train]
    X_val,   y_val   = X[n_train:n_train + n_val], y[n_train:n_train + n_val]
    X_test,  y_test  = X[n_train + n_val:],        y[n_train + n_val:]

    means = df_features.mean().to_dict()
    stds  = df_features.std().to_dict()
    mins  = df_features.min().to_dict()
    maxs  = df_features.max().to_dict()

    stats = DatasetStats(
        n_rows_raw=len(df),
        n_rows_clean=len(df_features),
        n_missing_pm25=n_missing_pm25,
        pct_missing_pm25=pct_missing,
        period_start=str(df_features.index.min()),
        period_end=str(df_features.index.max()),
        feature_cols=list(feature_cols),
        feature_means={k: float(v) for k, v in means.items()},
        feature_stds={k: float(v) for k, v in stds.items()},
        feature_mins={k: float(v) for k, v in mins.items()},
        feature_maxs={k: float(v) for k, v in maxs.items()},
        train_rows=int(len(X_train)),
        val_rows=int(len(X_val)),
        test_rows=int(len(X_test)),
        lookback=int(lookback),
    )

    return {
        "scaler":       scaler,
        "X_train":      X_train, "y_train": y_train,
        "X_val":        X_val,   "y_val":   y_val,
        "X_test":       X_test,  "y_test":  y_test,
        "df_clean":     df_features,
        "feature_cols": list(feature_cols),
        "lookback":     int(lookback),
        "stats":        stats,
    }


def inverse_transform_pm25(scaler: MinMaxScaler, scaled_pm25: np.ndarray,
                           feature_cols: list, last_window: np.ndarray) -> np.ndarray:
    """Invert MinMax scaling for a 1-d array of scaled PM2.5 predictions."""
    scaled_pm25 = np.asarray(scaled_pm25).reshape(-1, 1)
    n = scaled_pm25.shape[0]
    if last_window.ndim == 3:
        # last_window has shape (n, lookback, F); we use timestep -1 for the
        # other features as a plausible inverse vector.
        other = last_window[:, -1, 1:]
    else:
        other = last_window[:, 1:]
    if other.shape[0] != n:
        # Fall back to broadcasting from the last available row.
        other = np.tile(other[-1:], (n, 1))
    pad = np.concatenate((scaled_pm25, other), axis=1)
    return scaler.inverse_transform(pad)[:, 0]
