"""Hallucination detection for LLM-generated AQI explanations.

Six checks, each producing a boolean flag and an evidence string:

    H1  number_match       — the LLM-claimed PM2.5 must be within tol of
                             the LSTM ground truth (default 1%).
    H2  aqi_match          — the LLM-claimed AQI integer must match the
                             EPA-derived AQI from PM2.5 within +/-1.
    H3  category_consistency — claimed category must lie inside the AQI
                             integer's bracket.
    H4  feature_grounding  — every feature name mentioned in the
                             explanation must be a member of the supplied
                             top_features set (no fabricated pollutants).
    H5  range_check        — every number in the explanation that is
                             clearly tagged with a unit must be inside the
                             physical envelope (e.g. PM2.5 in [0, 1000],
                             wind in [0, 50] m/s, temp in [-50, 60] C).
    H6  confidence_low     — average token log-prob below threshold
                             (proxy for model uncertainty).

The aggregate ``is_hallucination`` flag is the disjunction of H1..H4 plus
any H5 violation that is in a known unit. H6 is reported as a soft signal.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .aqi_utils import aqi_to_category_range, pm25_to_aqi


KNOWN_POLLUTANT_NAMES = {
    "pm2.5", "pm25", "pm 2.5", "pm-2.5",
    "pm10", "pm 10",
    "no2", "no", "nox",
    "so2", "ozone", "o3",
    "co", "carbon monoxide",
    "benzene", "voc",
}

# Numeric ranges for common units we expect the model to keep within.
RANGE_BY_UNIT = {
    "ug/m3":      (0.0,    1000.0),
    "ug/m^3":     (0.0,    1000.0),
    "µg/m3":      (0.0,    1000.0),
    "ug m-3":     (0.0,    1000.0),
    "%":          (0.0,    100.0),
    "m/s":        (-5.0,   60.0),
    "kph":        (0.0,    250.0),
    "mph":        (0.0,    180.0),
    "c":          (-60.0,  60.0),
    "celsius":    (-60.0,  60.0),
    "f":          (-90.0,  150.0),
    "hpa":        (800.0,  1100.0),
    "mbar":       (800.0,  1100.0),
}

NUMBER_UNIT_RE = re.compile(
    r"(-?\d+(?:\.\d+)?)\s*(ug/m3|ug/m\^3|µg/m3|ug m-3|m/s|hpa|mbar|kph|mph|%|°c|°f|c|f|celsius)",
    flags=re.IGNORECASE,
)
NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


@dataclass
class HallucinationFlags:
    sample_index:        int
    pm25_pred:           float
    pm25_claimed:        float
    aqi_pred:            int
    aqi_claimed:         int
    category_pred:       str
    category_claimed:    str
    h1_number_match:     bool
    h2_aqi_match:        bool
    h3_category_ok:      bool
    h4_feature_ok:       bool
    h5_range_ok:         bool
    h6_confidence_low:   bool
    fabricated_features: list = field(default_factory=list)
    out_of_range_values: list = field(default_factory=list)
    avg_logprob:         float = float("nan")
    is_hallucination:    bool = False
    notes:               str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _norm(s: str) -> str:
    return s.lower().strip() if isinstance(s, str) else ""


def _safe_float(v, default: float = float("nan")) -> float:
    """Coerce JSON-loaded values (None, str, int, float) to float without raising."""
    if v is None:
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _safe_int(v, default: int = -10**9) -> int:
    """Coerce JSON-loaded values to int without raising. Strings like '178' work."""
    if v is None:
        return default
    try:
        return int(v)
    except (TypeError, ValueError):
        try:
            return int(round(float(v)))
        except (TypeError, ValueError):
            return default


def detect_one(parsed: dict, raw_output: str, top_features: dict,
               pm25_pred: float, sample_index: int = -1,
               avg_logprob: float = float("nan"),
               num_tol_pct: float = 1.0,
               logprob_threshold: float = -2.0) -> HallucinationFlags:
    """Run all six checks against a single LLM output."""
    if not isinstance(parsed, dict):
        parsed = {}
    aqi_pred = pm25_to_aqi(pm25_pred)

    # ---------- H1 number_match ----------
    pm25_claimed = _safe_float(parsed.get("predicted_pm25_ugm3"))
    if np.isnan(pm25_claimed) or pm25_pred == 0:
        h1 = (not np.isnan(pm25_claimed)) and (pm25_claimed == pm25_pred)
    else:
        rel = abs(pm25_claimed - pm25_pred) / max(1e-6, abs(pm25_pred)) * 100.0
        h1 = rel <= num_tol_pct
    h1_pass = bool(h1)

    # ---------- H2 aqi_match ----------
    aqi_claimed = _safe_int(parsed.get("predicted_aqi"))
    h2_pass = bool(abs(aqi_claimed - aqi_pred.aqi) <= 1)

    # ---------- H3 category_consistency ----------
    cat_raw = parsed.get("aqi_category")
    cat_claimed = cat_raw if isinstance(cat_raw, str) else ""
    h3_pass = bool(cat_claimed) and (_norm(cat_claimed) == _norm(aqi_pred.category))

    # ---------- H4 feature_grounding ----------
    expl_raw = parsed.get("explanation", "")
    drv_raw  = parsed.get("primary_driver", "")
    expl = expl_raw if isinstance(expl_raw, str) else ""
    drv  = drv_raw  if isinstance(drv_raw,  str) else ""
    text = f"{expl} {drv}".lower()
    allowed = {_norm(k) for k in top_features.keys()}
    allowed_clean = {a.replace(" ", "").replace("-", "").replace(".", "") for a in allowed}
    pm25_aliases = {"pm25", "pm2.5", "pm 2.5", "pm-2.5"}
    fabricated = []
    for tok in KNOWN_POLLUTANT_NAMES:
        # Word-boundary match — avoids "no" matching "not"/"now"/"none".
        pattern = r"(?<![a-z0-9])" + re.escape(tok) + r"(?![a-z0-9])"
        if not re.search(pattern, text):
            continue
        clean = tok.replace(" ", "").replace("-", "").replace(".", "")
        if tok in pm25_aliases or clean in pm25_aliases:
            continue
        if tok in allowed or clean in allowed_clean:
            continue
        fabricated.append(tok)
    h4_pass = (len(fabricated) == 0)

    # ---------- H5 range_check ----------
    out_of_range = []
    raw = raw_output if isinstance(raw_output, str) else ""
    for num, unit in NUMBER_UNIT_RE.findall(expl + " " + raw):
        try:
            val = float(num)
        except ValueError:
            continue
        u = unit.lower().replace("°", "").strip()
        bounds = RANGE_BY_UNIT.get(u)
        if bounds is None:
            continue
        lo, hi = bounds
        if val < lo or val > hi:
            out_of_range.append({"value": val, "unit": u, "bounds": [lo, hi]})
    h5_pass = (len(out_of_range) == 0)

    # ---------- H6 confidence ----------
    h6_low = (not np.isnan(avg_logprob)) and (avg_logprob < logprob_threshold)

    is_hall = (not h1_pass) or (not h2_pass) or (not h3_pass) \
              or (not h4_pass) or (not h5_pass)

    notes = []
    if not h1_pass:
        notes.append(f"H1 numeric mismatch (claim={pm25_claimed}, lstm={pm25_pred})")
    if not h2_pass:
        notes.append(f"H2 AQI mismatch (claim={aqi_claimed}, lstm={aqi_pred.aqi})")
    if not h3_pass:
        notes.append(f"H3 category mismatch (claim='{cat_claimed}', expected='{aqi_pred.category}')")
    if not h4_pass:
        notes.append(f"H4 fabricated features: {fabricated}")
    if not h5_pass:
        notes.append(f"H5 out-of-range values: {out_of_range}")
    if h6_low:
        notes.append(f"H6 low confidence (avg_logprob={avg_logprob:.3f})")

    return HallucinationFlags(
        sample_index=sample_index,
        pm25_pred=float(pm25_pred), pm25_claimed=float(pm25_claimed),
        aqi_pred=int(aqi_pred.aqi),  aqi_claimed=int(aqi_claimed),
        category_pred=str(aqi_pred.category),
        category_claimed=str(cat_claimed),
        h1_number_match=h1_pass, h2_aqi_match=h2_pass,
        h3_category_ok=h3_pass,  h4_feature_ok=h4_pass,
        h5_range_ok=h5_pass,     h6_confidence_low=h6_low,
        fabricated_features=fabricated, out_of_range_values=out_of_range,
        avg_logprob=float(avg_logprob), is_hallucination=bool(is_hall),
        notes="; ".join(notes),
    )


def detect_many(records: Iterable, num_tol_pct: float = 1.0,
                logprob_threshold: float = -2.0) -> pd.DataFrame:
    """Run detect_one over an iterable of ExplanationRecord objects."""
    rows = []
    for rec in records:
        flags = detect_one(
            parsed=rec.parsed_json,
            raw_output=rec.raw_output,
            top_features=rec.top_features,
            pm25_pred=rec.pm25_pred,
            sample_index=rec.sample_index,
            avg_logprob=rec.avg_logprob,
            num_tol_pct=num_tol_pct,
            logprob_threshold=logprob_threshold,
        )
        rows.append(flags.to_dict())
    return pd.DataFrame(rows)


def summarise(flags_df: pd.DataFrame) -> dict:
    n = len(flags_df)
    if n == 0:
        return {"n": 0}
    return {
        "n_explanations": int(n),
        "n_hallucinations": int(flags_df["is_hallucination"].sum()),
        "hallucination_rate_pct": float(flags_df["is_hallucination"].mean() * 100.0),
        "h1_number_match_pct":   float(flags_df["h1_number_match"].mean() * 100.0),
        "h2_aqi_match_pct":      float(flags_df["h2_aqi_match"].mean() * 100.0),
        "h3_category_ok_pct":    float(flags_df["h3_category_ok"].mean() * 100.0),
        "h4_feature_ok_pct":     float(flags_df["h4_feature_ok"].mean() * 100.0),
        "h5_range_ok_pct":       float(flags_df["h5_range_ok"].mean() * 100.0),
        "h6_confidence_low_pct": float(flags_df["h6_confidence_low"].mean() * 100.0),
        "mean_avg_logprob":      float(flags_df["avg_logprob"].mean(skipna=True))
                                 if "avg_logprob" in flags_df else float("nan"),
    }
