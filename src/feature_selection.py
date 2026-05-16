"""mRMR feature selection + Random Forest importance ranking.

mRMR is implemented from scratch using sklearn's mutual_info_regression, so
the pipeline does not depend on the optional `mrmr-selection` package.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression


def mrmr_select(X: pd.DataFrame, y: pd.Series, k: int = 3,
                random_state: int = 42) -> tuple[list, pd.DataFrame]:
    """Maximum-Relevance-Minimum-Redundancy feature selection (FCQ variant).

    relevance(f)        = MI(f, y)
    redundancy(f, S)    = mean_{s in S} |corr(f, s)|
    score(f | S)        = relevance(f) / (redundancy(f, S) + eps)

    The first feature is the most relevant; subsequent features maximise the
    above score until ``k`` features have been chosen.
    """
    rng = np.random.RandomState(random_state)
    feats = list(X.columns)
    Xv = X.values.astype(np.float64)
    yv = y.values.astype(np.float64)

    relevance = mutual_info_regression(Xv, yv, random_state=random_state)
    rel_series = pd.Series(relevance, index=feats, name="relevance")

    selected: list = []
    scores: list[dict] = []
    remaining = feats.copy()

    for step in range(min(k, len(feats))):
        best_feat, best_score = None, -np.inf
        for f in remaining:
            if not selected:
                score = rel_series[f]
                redundancy = 0.0
            else:
                corrs = []
                for s in selected:
                    c = np.corrcoef(X[f].values, X[s].values)[0, 1]
                    corrs.append(0.0 if np.isnan(c) else abs(c))
                redundancy = float(np.mean(corrs))
                score = float(rel_series[f] / (redundancy + 1e-9))
            if score > best_score:
                best_score, best_feat = score, f
            scores.append({
                "step": step + 1,
                "candidate": f,
                "relevance": float(rel_series[f]),
                "redundancy": float(redundancy),
                "score": float(score),
                "selected_so_far": ",".join(selected),
            })
        selected.append(best_feat)
        remaining.remove(best_feat)
        # Add a marker row noting which feature was actually picked at this step.
        scores.append({
            "step": step + 1,
            "candidate": f"[CHOSEN] {best_feat}",
            "relevance": float(rel_series[best_feat]),
            "redundancy": float(np.nan),
            "score": float(best_score),
            "selected_so_far": ",".join(selected),
        })

    log_df = pd.DataFrame(scores)
    return selected, log_df


def rf_importance(X: pd.DataFrame, y: pd.Series,
                  n_estimators: int = 200, random_state: int = 42) -> pd.DataFrame:
    """Train a Random Forest and return per-feature importance + permutation OOB."""
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
        oob_score=True,
    )
    rf.fit(X.values, y.values)
    imp = rf.feature_importances_
    df = pd.DataFrame({
        "feature": list(X.columns),
        "importance": imp,
        "importance_pct": imp / imp.sum() * 100.0,
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    return df, float(rf.oob_score_)


def save_results(out_dir: Path,
                 mrmr_selected: list,
                 mrmr_log: pd.DataFrame,
                 rf_df: pd.DataFrame,
                 rf_oob: float) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mrmr_log.to_csv(out_dir / "mrmr_log.csv", index=False)
    pd.DataFrame({"selected_feature": mrmr_selected,
                  "rank": list(range(1, len(mrmr_selected) + 1))})\
      .to_csv(out_dir / "mrmr_selected.csv", index=False)
    rf_df.to_csv(out_dir / "rf_importance.csv", index=False)
    (out_dir / "rf_oob_score.txt").write_text(f"{rf_oob:.6f}\n")
    return {
        "mrmr_selected_csv": str(out_dir / "mrmr_selected.csv"),
        "mrmr_log_csv": str(out_dir / "mrmr_log.csv"),
        "rf_importance_csv": str(out_dir / "rf_importance.csv"),
        "rf_oob_score": rf_oob,
    }
