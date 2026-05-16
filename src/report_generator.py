"""Generate the final markdown report from artefacts produced by the pipeline."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


# ----------------------------------------------------------------------------- helpers
def _md_table(df: pd.DataFrame, float_fmt: str = "{:.4f}") -> str:
    if df is None or len(df) == 0:
        return "_no rows_"
    cols = list(df.columns)
    out = ["| " + " | ".join(str(c) for c in cols) + " |",
           "| " + " | ".join("---" for _ in cols) + " |"]
    for _, row in df.iterrows():
        cells = []
        for v in row.values:
            if isinstance(v, float):
                cells.append(float_fmt.format(v))
            elif isinstance(v, (list, dict)):
                cells.append(str(v))
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def _read_json(path: Path) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def _read_csv(path: Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()


# ----------------------------------------------------------------------------- main
def generate_report(results_dir: str | Path) -> Path:
    """Stitch every artefact together into final_report.md.

    Expected layout::

        results/
          dataset_stats.json
          features/
            mrmr_log.csv
            mrmr_selected.csv
            rf_importance.csv
            rf_oob_score.txt
          models/
            lstm_baseline_train_history.csv
            issa_log.csv
            issa_best_params.json
            lstm_issa_train_history.csv
            metrics_summary.csv
            per_category_breakdown.csv
            confusion_matrix.csv
            test_predictions.csv
          llm/
            explanations.csv
            hallucination_flags.csv
            hallucination_summary.json
            qa_examples.csv
          plots/*.png
          run_metadata.json
    """
    root = Path(results_dir)
    plots = root / "plots"

    meta = _read_json(root / "run_metadata.json")
    stats = _read_json(root / "dataset_stats.json")
    rf_oob_path = root / "features" / "rf_oob_score.txt"
    rf_oob = rf_oob_path.read_text().strip() if rf_oob_path.exists() else "n/a"

    mrmr_sel = _read_csv(root / "features" / "mrmr_selected.csv")
    rf_df    = _read_csv(root / "features" / "rf_importance.csv")
    issa_best = _read_json(root / "models" / "issa_best_params.json")
    metrics_df = _read_csv(root / "models" / "metrics_summary.csv")
    per_cat   = _read_csv(root / "models" / "per_category_breakdown.csv")
    cm        = _read_csv(root / "models" / "confusion_matrix.csv")
    hist_base = _read_csv(root / "models" / "lstm_baseline_train_history.csv")
    hist_issa = _read_csv(root / "models" / "lstm_issa_train_history.csv")

    expl_df   = _read_csv(root / "llm" / "explanations.csv")
    halluc_df = _read_csv(root / "llm" / "hallucination_flags.csv")
    halluc_sum = _read_json(root / "llm" / "hallucination_summary.json")
    qa_df     = _read_csv(root / "llm" / "qa_examples.csv")

    plot_imgs = {
        "corr":      "plots/feature_correlation.png",
        "rf":        "plots/rf_feature_importance.png",
        "loss_base": "plots/lstm_baseline_loss_curve.png",
        "loss_issa": "plots/lstm_issa_loss_curve.png",
        "pred_base": "plots/lstm_baseline_predictions_overlay.png",
        "pred_issa": "plots/lstm_issa_predictions_overlay.png",
        "scatter_b": "plots/lstm_baseline_scatter.png",
        "scatter_i": "plots/lstm_issa_scatter.png",
        "bar":       "plots/baseline_comparison.png",
        "halluc":    "plots/hallucination_summary.png",
        "aqi_dist":  "plots/aqi_category_distribution.png",
    }

    md = []
    md += [
        "# Enhancing Air Quality Prediction with LLM-Based Explanation and Decision Support",
        "",
        "**Authors:** Hruthi Muggalla · Akanksha Karra · Atharva Nilkanth · Sai Krishna Ghanta",
        "",
        f"**Run ID:** `{meta.get('run_id', 'n/a')}`  ·  "
        f"**Timestamp:** `{meta.get('timestamp', 'n/a')}`  ·  "
        f"**Host:** `{meta.get('hostname', 'n/a')}`  ·  "
        f"**Device:** `{meta.get('device', 'n/a')}`",
        "",
        "---",
        "",
        "## 1. Executive Summary",
        "",
        "This report documents an end-to-end pipeline that (i) forecasts hourly "
        "PM2.5 concentration in Beijing using an LSTM whose hyper-parameters are "
        "tuned by an Improved Sparrow Search Algorithm (ISSA), (ii) translates the "
        "numerical forecast into a natural-language explanation through "
        "Qwen2.5-7B-Instruct, and (iii) audits every explanation against five "
        "grounding constraints to detect hallucinations. The base prediction model "
        "follows Wu et al. (2023). The LLM layer is novel: it consumes the LSTM "
        "output plus the top-k features selected by mRMR + Random Forest, and is "
        "constrained by a JSON contract that the hallucination detector verifies.",
        "",
        "Headline numbers are summarised in Section 5; full per-model metrics are "
        "in `results/models/metrics_summary.csv` and full per-explanation flags "
        "are in `results/llm/hallucination_flags.csv`.",
        "",
        "---",
        "",
        "## 2. Dataset",
        "",
        f"- **Source.** Beijing PM2.5 hourly data, UCI Machine Learning Repository "
        f"(file `PRSA_data_2010.1.1-2014.12.31.csv`).",
        f"- **Period covered.** {stats.get('period_start', 'n/a')} → "
        f"{stats.get('period_end', 'n/a')} ({stats.get('n_rows_clean', 'n/a')} hourly rows after cleaning).",
        f"- **Missing PM2.5 originally.** {stats.get('n_missing_pm25', 'n/a')} rows "
        f"({stats.get('pct_missing_pm25', 0)*100:.2f}% of the file).",
        f"- **Imputation.** Forward-fill followed by backward-fill on the PM2.5 "
        f"column; remaining columns are complete by construction.",
        f"- **Features kept.** {', '.join(stats.get('feature_cols', []))}.",
        f"- **Lookback window.** {stats.get('lookback', 'n/a')} hours.",
        f"- **Split.** {stats.get('train_rows', 'n/a')} train / "
        f"{stats.get('val_rows', 'n/a')} val / {stats.get('test_rows', 'n/a')} test "
        f"(chronological — no shuffling).",
        "",
        "Per-feature descriptive statistics:",
        "",
    ]
    if stats:
        feats = stats.get("feature_cols", [])
        rows = []
        for f in feats:
            rows.append({
                "feature": f,
                "mean":    stats.get("feature_means", {}).get(f, float("nan")),
                "std":     stats.get("feature_stds", {}).get(f, float("nan")),
                "min":     stats.get("feature_mins", {}).get(f, float("nan")),
                "max":     stats.get("feature_maxs", {}).get(f, float("nan")),
            })
        md.append(_md_table(pd.DataFrame(rows), "{:.3f}"))
    md += [
        "",
        f"![Feature correlation heatmap]({plot_imgs['corr']})",
        "",
        "---",
        "",
        "## 3. Feature Selection",
        "",
        "### 3.1 mRMR (Maximum Relevance, Minimum Redundancy)",
        "",
        "Mutual information between each feature and the next-hour PM2.5 is the "
        "**relevance** signal; the mean absolute Pearson correlation against "
        "already-selected features is the **redundancy** penalty. The top-k "
        "features are picked greedily (FCQ variant).",
        "",
    ]
    if len(mrmr_sel):
        md.append(_md_table(mrmr_sel, "{:.4f}"))
    md += [
        "",
        "Full search log: `results/features/mrmr_log.csv`.",
        "",
        "### 3.2 Random Forest impurity importance",
        "",
        f"Out-of-bag R² of the supporting Random Forest: **{rf_oob}**.",
        "",
    ]
    if len(rf_df):
        md.append(_md_table(rf_df, "{:.4f}"))
    md += [
        "",
        f"![Random Forest importance]({plot_imgs['rf']})",
        "",
        "_Confirms the well-known fact that the previous-hour PM2.5 value "
        "dominates the next-hour signal — see the bar chart above. The other "
        "weather features still contribute modestly._",
        "",
        "---",
        "",
        "## 4. Forecasting Models",
        "",
        "### 4.1 LSTM baseline",
        "",
        "Two stacked LSTM layers with 64 hidden units, dropout 0.2, MSE loss, "
        "Adam at 1e-3, batch 64. Inputs: 60-hour windows of all five features "
        "after MinMax scaling. Output: scaled PM2.5 at the next hour.",
        "",
    ]
    if len(hist_base):
        md.append(_md_table(hist_base, "{:.6f}"))
    md += [
        "",
        f"![Baseline LSTM loss curve]({plot_imgs['loss_base']})",
        "",
        "### 4.2 ISSA-LSTM",
        "",
        "The Improved Sparrow Search Algorithm tunes "
        "`hidden_size`, `num_layers`, `dropout`, `learning_rate`, `batch_size` "
        "by minimising one-epoch validation MSE on the same training data. "
        "Producers (top sparrows) move with a Levy-flight perturbation that "
        "decays as `exp(-(t/T)^2)`; scroungers chase the best producer; scouts "
        "perform anti-predator jumps when they end up at the worst rank.",
        "",
        "**Best hyper-parameters found:**",
        "",
        "```json",
        json.dumps(issa_best, indent=2),
        "```",
        "",
    ]
    if len(hist_issa):
        md.append(_md_table(hist_issa, "{:.6f}"))
    md += [
        "",
        f"![ISSA-LSTM loss curve]({plot_imgs['loss_issa']})",
        "",
        "Full ISSA search trajectory: `results/models/issa_log.csv`.",
        "",
        "### 4.3 Naive baselines",
        "",
        "- **Persistence.** Predicts next-hour PM2.5 = current PM2.5.",
        "- **AR(p).** OLS-fitted autoregression on the PM2.5 series, p=24.",
        "",
        "---",
        "",
        "## 5. Headline Forecasting Metrics",
        "",
        "All errors are reported in physical units (ug/m^3) on the held-out "
        "chronological test split.",
        "",
    ]
    if len(metrics_df):
        md.append(_md_table(metrics_df, "{:.4f}"))
    md += [
        "",
        f"![Baseline comparison bar chart]({plot_imgs['bar']})",
        "",
        f"![Predictions overlay — baseline]({plot_imgs['pred_base']})",
        "",
        f"![Predictions overlay — ISSA-LSTM]({plot_imgs['pred_issa']})",
        "",
        f"![Pred vs True — baseline]({plot_imgs['scatter_b']})",
        "",
        f"![Pred vs True — ISSA-LSTM]({plot_imgs['scatter_i']})",
        "",
        "### 5.1 Per-AQI-category breakdown (best model)",
        "",
    ]
    if len(per_cat):
        md.append(_md_table(per_cat, "{:.3f}"))
    md += [
        "",
        f"![AQI category distribution]({plot_imgs['aqi_dist']})",
        "",
        "### 5.2 AQI category confusion matrix (best model)",
        "",
    ]
    if len(cm):
        md.append(_md_table(cm))
    md += [
        "",
        "---",
        "",
        "## 6. LLM-Based Explanation Layer",
        "",
        "**Model.** Qwen2.5-7B-Instruct, loaded from "
        "`/scratch/sg41479/hf-cache/models--Qwen--Qwen2.5-7B-Instruct` "
        "in bfloat16 with `device_map='auto'`. The wrapper emits a "
        "single-shot JSON response under a fixed schema "
        "(see `src/llm_explainer.py`).",
        "",
        "**Inputs to the prompt.**",
        "",
        "```",
        "{ predicted_pm25_ugm3, predicted_aqi, aqi_category,",
        "  top_features: { pm25, dew, temp, pressure, wind } }",
        "```",
        "",
        "**Outputs (JSON schema).**",
        "",
        "```",
        "{ predicted_pm25_ugm3, predicted_aqi, aqi_category, primary_driver,",
        "  explanation, health_recommendation, confidence_note }",
        "```",
        "",
        f"Total explanations generated: **{len(expl_df)}**. "
        f"Mean wall-time per explanation: "
        f"**{expl_df['wall_time_s'].mean() if len(expl_df) else float('nan'):.2f} s**.",
        "",
    ]
    if len(expl_df):
        cols = ["sample_index", "pm25_pred", "pm25_actual", "aqi", "aqi_category",
                "parsed_ok", "avg_logprob", "n_tokens", "wall_time_s",
                "fallback_used"]
        cols = [c for c in cols if c in expl_df.columns]
        md.append(_md_table(expl_df[cols].head(20), "{:.3f}"))
    md += [
        "",
        "Five sample explanations (raw output, lightly truncated):",
        "",
    ]
    if len(expl_df):
        for _, row in expl_df.head(5).iterrows():
            txt = str(row.get("raw_output", ""))[:600].replace("\n", " ").strip()
            md += [
                f"**Sample {int(row['sample_index'])}** "
                f"(LSTM PM2.5 = {row['pm25_pred']:.1f} → AQI category "
                f"{row['aqi_category']}):",
                "",
                "> " + (txt or "_empty_"),
                "",
            ]
    md += [
        "Full per-explanation table: `results/llm/explanations.csv`.",
        "",
        "---",
        "",
        "## 7. Hallucination Detection",
        "",
        "Five hard checks (H1-H5) plus one soft signal (H6):",
        "",
        "| Code | Check | Test |",
        "| ---- | ----- | ---- |",
        "| H1 | Numeric match    | `|claimed_pm25 - lstm_pm25| / lstm_pm25 ≤ 1%` |",
        "| H2 | AQI integer      | `|claimed_aqi - epa_aqi| ≤ 1` |",
        "| H3 | Category bracket | `claimed_category ∈ EPA_BRACKET(claimed_aqi)` |",
        "| H4 | Feature grounding | every pollutant name in the explanation must be in the supplied feature set (no PM10/NO2/etc invented) |",
        "| H5 | Range envelope   | every (number, unit) pair must lie inside the physical range table |",
        "| H6 | Confidence       | average token log-prob `< -2.0` (soft signal only) |",
        "",
    ]
    if halluc_sum:
        sm = pd.DataFrame([halluc_sum]).T.reset_index()
        sm.columns = ["metric", "value"]
        md.append(_md_table(sm, "{:.3f}"))
    md += [
        "",
        f"![Hallucination summary]({plot_imgs['halluc']})",
        "",
    ]
    if len(halluc_df):
        cols = ["sample_index", "pm25_pred", "pm25_claimed", "aqi_pred", "aqi_claimed",
                "h1_number_match", "h2_aqi_match", "h3_category_ok",
                "h4_feature_ok", "h5_range_ok", "h6_confidence_low",
                "is_hallucination", "notes"]
        cols = [c for c in cols if c in halluc_df.columns]
        md += [
            "Per-sample flags (first 25 rows):",
            "",
            _md_table(halluc_df[cols].head(25)),
            "",
        ]
    md += [
        "Full per-sample flags: `results/llm/hallucination_flags.csv`.",
        "",
        "---",
        "",
        "## 8. Interactive QA Module",
        "",
        f"Each test sample is paired with five canonical user questions "
        f"(\"is it safe to run today?\", etc.) and re-passed through the LLM. "
        f"Total Q-A pairs generated: **{len(qa_df)}**.",
        "",
    ]
    if len(qa_df):
        cols = [c for c in ["sample_index", "question", "pm25_pred", "raw_output"]
                if c in qa_df.columns]
        sample = qa_df[cols].head(10).copy()
        if "raw_output" in sample.columns:
            sample["raw_output"] = sample["raw_output"].str[:280]
        md.append(_md_table(sample))
    md += [
        "",
        "Full Q-A log: `results/llm/qa_examples.csv`.",
        "",
        "---",
        "",
        "## 9. Methodology Recap",
        "",
        "1. **Load** PRSA CSV → datetime-indexed dataframe (43 824 hourly rows).",
        "2. **Clean** PM2.5 with forward+backward fill; drop the categorical wind direction.",
        "3. **Scale** with MinMax to [0, 1].",
        "4. **Feature select** with mRMR (top-3 by FCQ score) and rank with Random Forest.",
        "5. **Sequence** into 60-hour lookback windows.",
        "6. **Train** baseline LSTM (5 epochs).",
        "7. **Tune** with ISSA — 8 sparrows × 6 iterations × 1-epoch fitness probe; retrain best.",
        "8. **Compare** against PERSISTENCE and AR(24).",
        "9. **Explain** the first N test predictions with Qwen2.5-7B-Instruct under a JSON contract.",
        "10. **Audit** every explanation against H1-H6.",
        "11. **Generate** this report from CSV + JSON artefacts.",
        "",
        "---",
        "",
        "## 10. Reproducibility",
        "",
        "All artefacts referenced in this document are saved verbatim under "
        "`results/` (CSV + JSON + PNG). The pipeline is invoked by "
        "`scripts/run_pipeline.py` and the SLURM submission file is `submit.sh`. "
        "Random seeds (numpy, torch, ISSA) are pinned to `42` and recorded in "
        "`run_metadata.json`.",
        "",
        "---",
        "",
        "## 11. References",
        "",
        "1. Wu, H., Yang, T., Li, H., & Zhou, Z. (2023). Air quality prediction model based on mRMR-RF feature selection and ISSA-LSTM. *Scientific Reports*, 13.",
        "2. Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. *Neural Computation*, 9(8), 1735-1780.",
        "3. Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5-32.",
        "4. Brown, T., et al. (2020). Language Models are Few-Shot Learners. *NeurIPS* 33.",
        "5. Lewis, P., et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. *NeurIPS* 33.",
        "6. Lundberg, S. M., & Lee, S. I. (2017). A Unified Approach to Interpreting Model Predictions. *NeurIPS* 30.",
        "7. De Vito, S., et al. (2008). UCI Air Quality Dataset.",
        "8. Xue, J., & Shen, B. (2020). A novel swarm intelligence optimization approach: sparrow search algorithm. *Systems Science & Control Engineering*, 8(1), 22-34.",
        "9. US EPA. *Technical Assistance Document for the Reporting of Daily Air Quality.* EPA-454/B-18-007 (2018).",
        "",
    ]

    out = root / "final_report.md"
    out.write_text("\n".join(md))
    return out
