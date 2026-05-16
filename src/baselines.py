"""Naive baselines for PM2.5 forecasting.

We want simple yardsticks against which the LSTM and ISSA-LSTM can be
compared. Two are sufficient and avoid extra dependencies (no statsmodels):

    1. PERSISTENCE — y_hat[t+1] = pm25[t]   ("yesterday's weather is today's").
    2. AR(p)       — fit a one-step autoregressive model on the scaled
                     PM2.5 column with ordinary least squares.
"""
from __future__ import annotations

import numpy as np


def persistence_predict(X_test: np.ndarray) -> np.ndarray:
    """Predict the next scaled PM2.5 as the most recent observation."""
    return X_test[:, -1, 0].astype(np.float32)


def ar_fit(series: np.ndarray, p: int = 24) -> np.ndarray:
    """Fit an AR(p) model on a 1-d series via ordinary least squares.

    Returns the coefficient vector of length p+1 (last entry is the bias).
    """
    series = np.asarray(series, dtype=np.float64).reshape(-1)
    n = len(series)
    if n <= p + 1:
        raise ValueError("series shorter than lag")
    X = np.zeros((n - p, p + 1), dtype=np.float64)
    y = series[p:]
    for i in range(p):
        X[:, i] = series[p - i - 1: n - i - 1]
    X[:, -1] = 1.0
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return coef


def ar_predict(X_test: np.ndarray, coef: np.ndarray) -> np.ndarray:
    """Apply AR coefficients to each test window's last p observations of PM2.5."""
    p = len(coef) - 1
    pm25 = X_test[:, -p:, 0].astype(np.float64)        # shape (n, p)
    coef_lags = coef[:p]
    bias      = coef[-1]
    # Most recent observation maps to coef_lags[0].
    pm25_rev = pm25[:, ::-1]
    return (pm25_rev @ coef_lags + bias).astype(np.float32)
