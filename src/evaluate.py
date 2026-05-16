"""Numerical evaluation utilities for the AQI forecasting pipeline."""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from .aqi_utils import pm25_to_aqi


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    err = y_pred - y_true
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae  = float(np.mean(np.abs(err)))
    bias = float(np.mean(err))
    eps = 1e-6
    mape = float(np.mean(np.abs(err) / np.maximum(np.abs(y_true), eps)) * 100.0)
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    pcc = float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) > 1 else float("nan")
    return {
        "rmse": rmse, "mae": mae, "bias": bias, "mape_pct": mape,
        "r2": r2, "pearson_r": pcc,
        "n": int(len(y_true)),
        "y_true_mean": float(y_true.mean()), "y_pred_mean": float(y_pred.mean()),
    }


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Treat AQI categorisation as a classification problem."""
    cats_true = [pm25_to_aqi(v).category for v in y_true]
    cats_pred = [pm25_to_aqi(v).category for v in y_pred]
    n = len(cats_true)
    if n == 0:
        return {"n": 0, "accuracy_pct": float("nan")}
    correct = sum(int(a == b) for a, b in zip(cats_true, cats_pred))
    cats = sorted(set(cats_true) | set(cats_pred))
    cm = pd.DataFrame(0, index=cats, columns=cats, dtype=int)
    for a, b in zip(cats_true, cats_pred):
        cm.loc[a, b] += 1
    return {
        "n": int(n),
        "accuracy_pct": float(correct / n * 100.0),
        "confusion_matrix": cm,
    }


def per_category_breakdown(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    rows = []
    cats_true = [pm25_to_aqi(v).category for v in y_true]
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "cat": cats_true})
    for cat, grp in df.groupby("cat"):
        rows.append({
            "category": cat, "n": int(len(grp)),
            **{("category_" + k if k in {"n"} else k): v
               for k, v in regression_metrics(grp["y_true"].values,
                                              grp["y_pred"].values).items()},
        })
    return pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)
