#!/usr/bin/env python3
"""End-to-end Air-Quality + LLM-Explanation pipeline.

Saves every artefact (CSV / JSON / PNG / weights) under ``--results_dir`` so
the report generator can stitch a complete markdown without re-running.

Usage::

    python scripts/run_pipeline.py \
        --csv /scratch/sg41479/AdvSplIoT/PRSA_data.csv \
        --results_dir /scratch/sg41479/AdvSplIoT/results \
        --epochs_baseline 5 --epochs_issa 5 \
        --issa_iters 6 --issa_pop 8 \
        --n_explanations 60 \
        --device cuda
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import sys
import time
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

# allow `python scripts/run_pipeline.py` from anywhere
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.aqi_utils import pm25_to_aqi
from src.baselines import ar_fit, ar_predict, persistence_predict
from src.data_loader import (
    FEATURE_COLUMNS, inverse_transform_pm25, load_prsa, preprocess
)
from src.evaluate import (
    classification_metrics, per_category_breakdown, regression_metrics
)
from src.feature_selection import mrmr_select, rf_importance, save_results as save_fs
from src.hallucination_detect import detect_many, summarise as halluc_summary
from src.issa import ISSAConfig, issa_search
from src.llm_explainer import LLMConfig, QwenExplainer
from src.lstm_model import LSTMConfig, train_lstm
from src.plotting import (
    aqi_category_distribution, baseline_bar, feature_correlation,
    hallucination_summary_plot, loss_curves, predictions_overlay,
    rf_importance_plot, scatter_pred_vs_true,
)
from src.qa_module import DEFAULT_QUESTIONS, run_qa
from src.report_generator import generate_report


# ----------------------------------------------------------------------------- args
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str,
                   default=str(ROOT / "PRSA_data.csv"))
    p.add_argument("--results_dir", type=str,
                   default=str(ROOT / "results"))
    p.add_argument("--lookback", type=int, default=60)
    p.add_argument("--train_frac", type=float, default=0.72)
    p.add_argument("--val_frac",   type=float, default=0.08)
    p.add_argument("--epochs_baseline", type=int, default=5)
    p.add_argument("--epochs_issa",     type=int, default=5)
    p.add_argument("--issa_iters",      type=int, default=6)
    p.add_argument("--issa_pop",        type=int, default=8)
    p.add_argument("--issa_fitness_epochs", type=int, default=1)
    p.add_argument("--n_explanations",  type=int, default=60,
                   help="how many test rows to explain with the LLM")
    p.add_argument("--n_qa_samples", type=int, default=4,
                   help="how many test rows to feed into the QA module")
    p.add_argument("--qa_questions", nargs="*", default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_llm", action="store_true",
                   help="skip Qwen and use the deterministic template fallback")
    p.add_argument("--llm_model_path", type=str, default=None)
    p.add_argument("--llm_cache_dir",  type=str, default="/scratch/sg41479/hf-cache")
    p.add_argument("--llm_dtype", type=str, default="bfloat16")
    p.add_argument("--llm_max_new_tokens", type=int, default=320)
    p.add_argument("--llm_temperature", type=float, default=0.3)
    p.add_argument("--num_tol_pct", type=float, default=1.0,
                   help="numeric tolerance (%) for hallucination check H1")
    p.add_argument("--logprob_threshold", type=float, default=-2.0)
    p.add_argument("--rf_estimators", type=int, default=200)
    return p.parse_args()


# ----------------------------------------------------------------------------- main
def main() -> None:
    args = parse_args()

    # ---------- pick a device ----------
    import torch
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    results = Path(args.results_dir)
    (results / "features").mkdir(parents=True, exist_ok=True)
    (results / "models").mkdir(parents=True, exist_ok=True)
    (results / "plots").mkdir(parents=True, exist_ok=True)
    (results / "llm").mkdir(parents=True, exist_ok=True)

    run_meta = {
        "run_id":   os.environ.get("SLURM_JOB_ID", time.strftime("%Y%m%d_%H%M%S")),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hostname":  socket.gethostname(),
        "python":    sys.version.split()[0],
        "platform":  platform.platform(),
        "device":    str(args.device),
        "torch":     torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "args":      vars(args),
        "seed":      args.seed,
    }
    if torch.cuda.is_available():
        run_meta["gpu_name"] = torch.cuda.get_device_name(0)
        run_meta["gpu_capability"] = list(torch.cuda.get_device_capability(0))
    (results / "run_metadata.json").write_text(json.dumps(run_meta, indent=2))
    print("[meta]", json.dumps({k: run_meta[k] for k in
                                ("run_id", "timestamp", "hostname", "device", "torch")}))

    # ========================================================== 1. data
    print("\n[1/9] loading + preprocessing dataset")
    raw = load_prsa(args.csv)
    pre = preprocess(
        raw, lookback=args.lookback,
        train_frac=args.train_frac, val_frac=args.val_frac,
    )
    pre["stats"].save(results / "dataset_stats.json")
    pre["df_clean"].head(10000).to_csv(results / "dataset_head.csv")
    feature_correlation(pre["df_clean"], results / "plots")
    aqi_category_distribution(
        pre["df_clean"]["pm25"].values,
        pre["df_clean"]["pm25"].values,  # reuse for the actual distribution
        results / "plots",
    )

    # ========================================================== 2. feature selection
    print("\n[2/9] mRMR + Random Forest feature selection")
    df_fs = pre["df_clean"].copy()
    df_fs["target"] = df_fs["pm25"].shift(-1)
    df_fs = df_fs.dropna()
    X_fs = df_fs[pre["feature_cols"]]
    y_fs = df_fs["target"]
    selected, mrmr_log = mrmr_select(X_fs, y_fs, k=3, random_state=args.seed)
    rf_df, rf_oob = rf_importance(X_fs, y_fs, n_estimators=args.rf_estimators,
                                  random_state=args.seed)
    save_fs(results / "features", selected, mrmr_log, rf_df, rf_oob)
    rf_importance_plot(rf_df, results / "plots")
    print(f"[features] mRMR top-3 = {selected}")
    print(f"[features] RF importances:\n{rf_df}")

    # ========================================================== 3. baselines (no train)
    print("\n[3/9] naive baselines")
    persistence_pred_scaled = persistence_predict(pre["X_test"])
    # Build a *contiguous* training-period scaled-PM2.5 series. The first
    # window's full lookback covers timesteps [0, lookback); each subsequent
    # y_train value covers timestep `lookback + i`. Concatenating these
    # reconstructs scaled[:lookback + n_train, 0] in chronological order
    # without duplication.
    train_pm25_series = np.concatenate(
        [pre["X_train"][0, :, 0], pre["y_train"]]
    )
    coef = ar_fit(train_pm25_series, p=24)
    ar_pred_scaled = ar_predict(pre["X_test"], coef)
    np.savetxt(results / "models" / "ar_coefficients.csv", coef,
               delimiter=",", header="ar_coefficient_lag1..lagP_then_bias",
               comments="")

    # ========================================================== 4. baseline LSTM
    print("\n[4/9] training baseline LSTM")
    base_cfg = LSTMConfig(
        input_size=pre["X_train"].shape[-1], hidden_size=64, num_layers=2,
        dropout=0.2, learning_rate=1e-3, batch_size=64,
        epochs=args.epochs_baseline, seed=args.seed,
    )
    base_out = train_lstm(
        base_cfg,
        pre["X_train"], pre["y_train"],
        pre["X_val"],   pre["y_val"],
        pre["X_test"],  pre["y_test"],
        device=args.device, out_dir=results / "models", tag="lstm_baseline",
    )

    # ========================================================== 5. ISSA-LSTM
    print("\n[5/9] ISSA hyper-parameter search")

    def fitness(params: dict) -> float:
        cfg = LSTMConfig(
            input_size=pre["X_train"].shape[-1],
            hidden_size=int(params["hidden_size"]),
            num_layers=int(params["num_layers"]),
            dropout=float(params["dropout"]),
            learning_rate=float(params["learning_rate"]),
            batch_size=int(params["batch_size"]),
            epochs=args.issa_fitness_epochs, seed=args.seed,
        )
        # Use the first 30% of training as the fitness probe to keep the
        # search affordable. Validation set acts as the score.
        n = int(0.3 * len(pre["X_train"]))
        out = train_lstm(
            cfg,
            pre["X_train"][:n], pre["y_train"][:n],
            pre["X_val"], pre["y_val"],
            pre["X_test"][:1024], pre["y_test"][:1024],
            device=args.device, out_dir=None, tag="issa_probe",
        )
        # Lower MSE on the validation set is better; if no val split, use the
        # last training MSE.
        last = out["history"][-1]
        score = last["val_mse_scaled"]
        if not np.isfinite(score):
            score = last["train_mse_scaled"]
        return float(score)

    issa = issa_search(
        fitness,
        ISSAConfig(n_sparrows=args.issa_pop, n_iter=args.issa_iters,
                   seed=args.seed),
        log_path=results / "models" / "issa_log.csv",
    )
    (results / "models" / "issa_best_params.json").write_text(
        json.dumps(issa["best_params"], indent=2, default=str)
    )
    (results / "models" / "issa_summary.json").write_text(json.dumps({
        "best_fitness": issa["best_fitness"],
        "best_params":  issa["best_params"],
        "config":       issa["config"],
    }, indent=2, default=str))

    print("\n[6/9] re-training LSTM with ISSA-best hyper-parameters")
    bp = issa["best_params"]
    issa_cfg = LSTMConfig(
        input_size=pre["X_train"].shape[-1],
        hidden_size=int(bp["hidden_size"]),
        num_layers=int(bp["num_layers"]),
        dropout=float(bp["dropout"]),
        learning_rate=float(bp["learning_rate"]),
        batch_size=int(bp["batch_size"]),
        epochs=args.epochs_issa, seed=args.seed,
    )
    issa_out = train_lstm(
        issa_cfg,
        pre["X_train"], pre["y_train"],
        pre["X_val"], pre["y_val"],
        pre["X_test"], pre["y_test"],
        device=args.device, out_dir=results / "models", tag="lstm_issa",
    )

    # ---------- inverse-scale every prediction ----------
    scaler = pre["scaler"]
    feature_cols = pre["feature_cols"]
    last_window = pre["X_test"]

    def inv(x):
        return inverse_transform_pm25(scaler, x, feature_cols, last_window)

    y_test_pm = inv(pre["y_test"])
    base_pm   = inv(base_out["preds_scaled"])
    issa_pm   = inv(issa_out["preds_scaled"])
    pers_pm   = inv(persistence_pred_scaled)
    ar_pm     = inv(ar_pred_scaled)

    # ---------- evaluate everyone ----------
    print("\n[7/9] computing metrics for every model")
    rows = []
    for tag, yp in [
        ("persistence", pers_pm),
        ("ar24",        ar_pm),
        ("lstm_baseline", base_pm),
        ("lstm_issa",     issa_pm),
    ]:
        m = regression_metrics(y_test_pm, yp)
        c = classification_metrics(y_test_pm, yp)
        rows.append({"model": tag, **m, "category_accuracy_pct": c["accuracy_pct"]})
    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(results / "models" / "metrics_summary.csv", index=False)
    baseline_bar(metrics_df, results / "plots")

    # save raw predictions for traceability
    pred_df = pd.DataFrame({
        "actual_pm25":      y_test_pm,
        "lstm_baseline_pm25": base_pm,
        "lstm_issa_pm25":   issa_pm,
        "persistence_pm25": pers_pm,
        "ar24_pm25":        ar_pm,
    })
    pred_df.to_csv(results / "models" / "test_predictions.csv", index=False)

    # ---------- pick the best for the LLM input ----------
    best = metrics_df.sort_values("rmse").iloc[0]["model"]
    best_pred = {
        "lstm_baseline":  base_pm,
        "lstm_issa":      issa_pm,
        "persistence":    pers_pm,
        "ar24":           ar_pm,
    }[best]
    print(f"[metrics] best model = {best}")

    # plots for the two LSTM variants
    loss_curves(pd.DataFrame(base_out["history"]), results / "plots", "lstm_baseline")
    loss_curves(pd.DataFrame(issa_out["history"]), results / "plots", "lstm_issa")
    predictions_overlay(y_test_pm, base_pm, results / "plots", "lstm_baseline", n=1000)
    predictions_overlay(y_test_pm, issa_pm, results / "plots", "lstm_issa",     n=1000)
    scatter_pred_vs_true(y_test_pm, base_pm, results / "plots", "lstm_baseline")
    scatter_pred_vs_true(y_test_pm, issa_pm, results / "plots", "lstm_issa")
    aqi_category_distribution(y_test_pm, best_pred, results / "plots")

    pcb = per_category_breakdown(y_test_pm, best_pred)
    pcb.to_csv(results / "models" / "per_category_breakdown.csv", index=False)
    cls = classification_metrics(y_test_pm, best_pred)
    cm = cls.get("confusion_matrix")
    if isinstance(cm, pd.DataFrame):
        cm.to_csv(results / "models" / "confusion_matrix.csv", index=True)

    # ========================================================== 8. LLM explanations
    print(f"\n[8/9] LLM explanations on first {args.n_explanations} test rows")
    explainer_cfg = LLMConfig(
        cache_dir=args.llm_cache_dir,
        max_new_tokens=args.llm_max_new_tokens,
        temperature=args.llm_temperature,
        dtype=args.llm_dtype,
        seed=args.seed,
    )
    if args.llm_model_path:
        explainer_cfg.model_path = args.llm_model_path

    n_expl = max(0, min(args.n_explanations, len(y_test_pm)))
    feat_cols = pre["feature_cols"]

    # Build a "top features" dict for each chosen sample. We use the values of
    # the actual features at the *last* step of the lookback window, so the
    # LLM sees the same sensor values the model saw most recently.
    raw_last = scaler.inverse_transform(pre["X_test"][:, -1, :])  # shape (n, F)

    samples = []
    for i in range(n_expl):
        feats = {f: float(round(raw_last[i, j], 4))
                 for j, f in enumerate(feat_cols)}
        samples.append({
            "sample_index": int(i),
            "pm25_pred":    float(best_pred[i]),
            "pm25_actual":  float(y_test_pm[i]),
            "top_features": feats,
            "user_question": "",
        })

    explainer = QwenExplainer(explainer_cfg, enabled=not args.no_llm)
    expl_records = []
    t_expl_start = time.time()
    for k, s in enumerate(samples):
        rec = explainer.explain(
            aqi=pm25_to_aqi(s["pm25_pred"]),
            top_features=s["top_features"],
            user_question=s["user_question"],
            sample_index=s["sample_index"],
            pm25_actual=s["pm25_actual"],
        )
        expl_records.append(rec)
        if (k + 1) % 10 == 0 or k == n_expl - 1:
            print(f"  [llm] {k + 1}/{n_expl} done  "
                  f"(parsed_ok={rec.parsed_ok} fb={rec.fallback_used} "
                  f"avg_lp={rec.avg_logprob:.3f})")
    print(f"[llm] total wall time = {time.time() - t_expl_start:.1f}s")

    expl_rows = []
    for r in expl_records:
        d = r.to_dict()
        d["top_features"] = json.dumps(d["top_features"])
        d["parsed_json"] = json.dumps(d["parsed_json"])
        expl_rows.append(d)
    expl_df = pd.DataFrame(expl_rows)
    expl_df.to_csv(results / "llm" / "explanations.csv", index=False)
    (results / "llm" / "explanations.jsonl").write_text(
        "\n".join(json.dumps(r.to_dict(), default=str) for r in expl_records)
    )

    # ========================================================== 9. hallucination + QA + report
    print("\n[9/9] hallucination detection + QA + report")
    halluc_df = detect_many(expl_records, num_tol_pct=args.num_tol_pct,
                            logprob_threshold=args.logprob_threshold)
    halluc_df.to_csv(results / "llm" / "hallucination_flags.csv", index=False)
    summary = halluc_summary(halluc_df)
    (results / "llm" / "hallucination_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    if summary.get("n_explanations", 0) > 0:
        hallucination_summary_plot(summary, results / "plots")
    print(f"[halluc] summary = {json.dumps(summary, indent=2)}")

    # ----- QA module -----
    n_qa = max(0, min(args.n_qa_samples, n_expl))
    qa_samples = samples[:n_qa]
    qa_records = run_qa(
        qa_samples, explainer=explainer,
        questions=args.qa_questions or DEFAULT_QUESTIONS,
    )
    qa_rows = []
    for q in qa_records:
        r = q.record.to_dict()
        r["top_features"] = json.dumps(r["top_features"])
        r["parsed_json"]  = json.dumps(r["parsed_json"])
        r["question"] = q.question
        qa_rows.append(r)
    pd.DataFrame(qa_rows).to_csv(results / "llm" / "qa_examples.csv", index=False)

    # ----- final report -----
    rep_path = generate_report(results)
    print(f"[report] written → {rep_path}")
    print("\n[done] all artefacts under:", results)


if __name__ == "__main__":
    main()
