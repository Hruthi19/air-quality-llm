"""Plot generation. All figures are saved as PNG into results/plots/."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .aqi_utils import EPA_PM25_BREAKPOINTS


def _ensure(d: Path) -> Path:
    d = Path(d); d.mkdir(parents=True, exist_ok=True); return d


def feature_correlation(df: pd.DataFrame, out_dir: Path) -> Path:
    out = _ensure(out_dir) / "feature_correlation.png"
    plt.figure(figsize=(7, 5))
    corr = df.corr()
    plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(label="Pearson r")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
    plt.yticks(range(len(corr.columns)), corr.columns)
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            plt.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center",
                     fontsize=8)
    plt.title("Feature correlation matrix")
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()
    return out


def rf_importance_plot(rf_df: pd.DataFrame, out_dir: Path) -> Path:
    out = _ensure(out_dir) / "rf_feature_importance.png"
    plt.figure(figsize=(7, 4))
    rf_df = rf_df.sort_values("importance")
    plt.barh(rf_df["feature"], rf_df["importance"])
    for i, (f, v) in enumerate(zip(rf_df["feature"], rf_df["importance"])):
        plt.text(v, i, f"  {v*100:.1f}%", va="center")
    plt.xlabel("Random-Forest impurity importance")
    plt.title("Random-Forest feature importance")
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()
    return out


def loss_curves(history_df: pd.DataFrame, out_dir: Path, tag: str) -> Path:
    out = _ensure(out_dir) / f"{tag}_loss_curve.png"
    plt.figure(figsize=(7, 4))
    plt.plot(history_df["epoch"], history_df["train_mse_scaled"], "o-",
             label="train")
    if "val_mse_scaled" in history_df.columns and \
       history_df["val_mse_scaled"].notna().any():
        plt.plot(history_df["epoch"], history_df["val_mse_scaled"], "s-",
                 label="val")
    plt.xlabel("epoch"); plt.ylabel("MSE (scaled)")
    plt.title(f"{tag} training loss")
    plt.legend(); plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()
    return out


def predictions_overlay(y_true: np.ndarray, y_pred: np.ndarray,
                        out_dir: Path, tag: str, n: int = 1000) -> Path:
    out = _ensure(out_dir) / f"{tag}_predictions_overlay.png"
    n = min(n, len(y_true))
    plt.figure(figsize=(11, 4))
    plt.plot(y_true[:n], label="actual", lw=1)
    plt.plot(y_pred[:n], label="predicted", lw=1, alpha=0.85)
    plt.xlabel("test step (hours)"); plt.ylabel("PM2.5 (ug/m^3)")
    plt.title(f"{tag} — actual vs predicted PM2.5 (first {n} test hours)")
    plt.legend(); plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()
    return out


def scatter_pred_vs_true(y_true: np.ndarray, y_pred: np.ndarray,
                         out_dir: Path, tag: str) -> Path:
    out = _ensure(out_dir) / f"{tag}_scatter.png"
    plt.figure(figsize=(5, 5))
    plt.scatter(y_true, y_pred, s=4, alpha=0.4)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.plot(lims, lims, "k--", lw=1)
    plt.xlabel("actual PM2.5 (ug/m^3)")
    plt.ylabel("predicted PM2.5 (ug/m^3)")
    plt.title(f"{tag} — predictions vs ground truth")
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()
    return out


def baseline_bar(metrics_df: pd.DataFrame, out_dir: Path) -> Path:
    out = _ensure(out_dir) / "baseline_comparison.png"
    plt.figure(figsize=(8, 4))
    x = np.arange(len(metrics_df))
    plt.bar(x - 0.2, metrics_df["rmse"], 0.4, label="RMSE")
    plt.bar(x + 0.2, metrics_df["mae"],  0.4, label="MAE")
    plt.xticks(x, metrics_df["model"], rotation=15)
    plt.ylabel("error (ug/m^3)")
    plt.title("Per-model error (lower is better)")
    plt.legend(); plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()
    return out


def hallucination_summary_plot(summary: dict, out_dir: Path) -> Path:
    out = _ensure(out_dir) / "hallucination_summary.png"
    keys = ["h1_number_match_pct", "h2_aqi_match_pct", "h3_category_ok_pct",
            "h4_feature_ok_pct",   "h5_range_ok_pct"]
    labels = ["H1 numeric", "H2 AQI", "H3 category", "H4 feature",
              "H5 range"]
    vals = [summary.get(k, float("nan")) for k in keys]
    plt.figure(figsize=(7, 4))
    bars = plt.bar(labels, vals)
    for b, v in zip(bars, vals):
        plt.text(b.get_x() + b.get_width()/2, v + 1, f"{v:.1f}%",
                 ha="center", va="bottom", fontsize=9)
    plt.ylim(0, 110)
    plt.ylabel("pass rate (%)")
    plt.title("LLM grounding pass rates (higher is better)")
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()
    return out


def aqi_category_distribution(y_true: np.ndarray, y_pred: np.ndarray,
                              out_dir: Path) -> Path:
    from .aqi_utils import pm25_to_aqi
    out = _ensure(out_dir) / "aqi_category_distribution.png"
    cats_t = pd.Series([pm25_to_aqi(v).category for v in y_true]).value_counts()
    cats_p = pd.Series([pm25_to_aqi(v).category for v in y_pred]).value_counts()
    cats = sorted(set(cats_t.index) | set(cats_p.index))
    t = [cats_t.get(c, 0) for c in cats]
    p = [cats_p.get(c, 0) for c in cats]
    x = np.arange(len(cats))
    plt.figure(figsize=(8, 4))
    plt.bar(x - 0.2, t, 0.4, label="actual")
    plt.bar(x + 0.2, p, 0.4, label="predicted")
    plt.xticks(x, cats, rotation=20, ha="right")
    plt.ylabel("count")
    plt.title("AQI category distribution (test split)")
    plt.legend(); plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()
    return out
