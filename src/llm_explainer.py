"""Qwen2.5-7B-Instruct wrapper for grounded AQI explanations.

The wrapper is intentionally narrow:

    * the prompt is ALWAYS prefixed with the structured AQI dict so the LLM
      cannot fabricate inputs,
    * the model is asked to emit JSON with a fixed schema so we can extract
      numbers programmatically and check them against the LSTM output,
    * average token log-probability is recorded as a uncertainty proxy,
    * a deterministic template fallback is provided for when the LLM is
      unavailable (so the rest of the pipeline still produces a report).

Model files live at:
    /scratch/sg41479/hf-cache/models--Qwen--Qwen2.5-7B-Instruct
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import List, Optional

import numpy as np

from .aqi_utils import AQIResult, pm25_to_aqi


SYSTEM_PROMPT = (
    "You are an air-quality assistant. Your only job is to explain a PM2.5 "
    "prediction made by a downstream LSTM model and give safe, generic "
    "health advice. You MUST obey three hard rules:\n"
    "1. Use ONLY the numbers and feature names provided in the INPUT block. "
    "Do not invent pollutants (no PM10, NO2, O3, SO2, CO unless explicitly "
    "given). If a quantity is not in the INPUT, do not mention it.\n"
    "2. Output a single JSON object matching the schema, with no prose "
    "outside the JSON. Use the EXACT predicted PM2.5 and AQI values; do "
    "not round or modify them.\n"
    "3. Recommendations must follow EPA category guidance. Never claim a "
    "category that disagrees with the AQI integer."
)

USER_TEMPLATE = """\
INPUT (ground truth from the LSTM forecaster):
{ground_truth_json}

JSON SCHEMA you must produce:
{{
  "predicted_pm25_ugm3":      <number, must equal {pm25_value}>,
  "predicted_aqi":            <integer, must equal {aqi_value}>,
  "aqi_category":             "{aqi_category}",
  "primary_driver":           "<one of the feature names listed under top_features>",
  "explanation":              "<2-3 sentences using ONLY the supplied numbers>",
  "health_recommendation":    "<one short, generic sentence>",
  "confidence_note":          "<one short sentence>"
}}

Question (optional, may be empty): {user_question}

Reply with the JSON object only.
"""


@dataclass
class LLMConfig:
    model_path:       str   = "/scratch/sg41479/hf-cache/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
    cache_dir:        str   = "/scratch/sg41479/hf-cache"
    max_new_tokens:   int   = 320
    temperature:      float = 0.3
    top_p:            float = 0.9
    do_sample:        bool  = True
    dtype:            str   = "bfloat16"
    device_map:       str   = "auto"
    seed:             int   = 42

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ExplanationRecord:
    sample_index:    int
    pm25_pred:       float
    pm25_actual:     float
    aqi:             int
    aqi_category:    str
    top_features:    dict
    user_question:   str
    raw_output:      str
    parsed_json:     dict
    parsed_ok:       bool
    avg_logprob:     float
    n_tokens:        int
    wall_time_s:     float
    fallback_used:   bool

    def to_dict(self) -> dict:
        return asdict(self)


def _build_prompt(aqi: AQIResult, top_features: dict, user_question: str = "") -> str:
    gt = {
        "predicted_pm25_ugm3": float(round(aqi.pm25, 4)),
        "predicted_aqi":       int(aqi.aqi),
        "aqi_category":        aqi.category,
        "top_features":        {k: float(round(v, 4)) for k, v in top_features.items()},
        "epa_health_message":  aqi.health_message,
    }
    return USER_TEMPLATE.format(
        ground_truth_json=json.dumps(gt, indent=2),
        pm25_value=gt["predicted_pm25_ugm3"],
        aqi_value=gt["predicted_aqi"],
        aqi_category=gt["aqi_category"],
        user_question=user_question or "(none)"
    )


def _template_fallback(aqi: AQIResult, top_features: dict, user_question: str) -> dict:
    """Deterministic JSON used when the LLM is unavailable."""
    if top_features:
        primary = max(top_features, key=lambda k: abs(top_features[k]))
    else:
        primary = "pm25"
    explanation = (
        f"PM2.5 is forecast at {aqi.pm25:.1f} ug/m^3, mapping to AQI {aqi.aqi} "
        f"({aqi.category}). The dominant driver among the supplied features is "
        f"{primary} (value {top_features.get(primary, float('nan'))}). "
        f"{aqi.health_message}"
    )
    return {
        "predicted_pm25_ugm3":   round(float(aqi.pm25), 4),
        "predicted_aqi":         int(aqi.aqi),
        "aqi_category":          aqi.category,
        "primary_driver":        primary,
        "explanation":           explanation,
        "health_recommendation": aqi.recommended_action,
        "confidence_note":       "Template fallback used (LLM disabled).",
    }


def _try_parse_json(text: str) -> Optional[dict]:
    """Extract the first JSON object from a string."""
    if not text:
        return None
    # Strip fenced code blocks the model may add in spite of instructions.
    text = re.sub(r"```(json)?", "", text).replace("```", "")
    # Find the first balanced curly-brace block.
    depth = 0
    start = -1
    for i, c in enumerate(text):
        if c == "{":
            if depth == 0:
                start = i
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                blob = text[start:i + 1]
                try:
                    return json.loads(blob)
                except Exception:
                    return None
    return None


class QwenExplainer:
    """Lazy-loaded Qwen2.5-7B-Instruct wrapper."""

    def __init__(self, cfg: LLMConfig | None = None, enabled: bool = True,
                 verbose: bool = False):
        self.cfg = cfg or LLMConfig()
        self.enabled = enabled
        self.verbose = verbose
        self._model = None
        self._tokenizer = None
        self._device = None
        self._load_error: Optional[str] = None

    # --------------------------------------------------------------------- load
    def _load(self):
        if self._model is not None or not self.enabled:
            return
        try:
            os.environ.setdefault("HF_HOME", self.cfg.cache_dir)
            os.environ.setdefault("TRANSFORMERS_CACHE", self.cfg.cache_dir)
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            torch_dtype = {
                "bfloat16": torch.bfloat16,
                "float16":  torch.float16,
                "float32":  torch.float32,
            }.get(self.cfg.dtype, torch.bfloat16)

            t0 = time.time()
            tok = AutoTokenizer.from_pretrained(
                self.cfg.model_path, cache_dir=self.cfg.cache_dir,
                trust_remote_code=True,
            )
            mdl = AutoModelForCausalLM.from_pretrained(
                self.cfg.model_path, cache_dir=self.cfg.cache_dir,
                torch_dtype=torch_dtype, device_map=self.cfg.device_map,
                trust_remote_code=True,
            )
            mdl.eval()
            self._tokenizer = tok
            self._model = mdl
            try:
                self._device = next(mdl.parameters()).device
            except StopIteration:
                self._device = torch.device("cpu")
            print(f"[llm] Qwen2.5-7B-Instruct loaded in {time.time() - t0:.1f}s "
                  f"on {self._device} dtype={self.cfg.dtype}")
        except Exception as e:
            self._load_error = repr(e)
            self.enabled = False
            print(f"[llm] disabled — load failed: {e!r}")

    # ---------------------------------------------------------------- generate
    @staticmethod
    def _avg_logprob(scores, sequences, prompt_len: int):
        """Compute the average per-token log-probability of the generated tail."""
        import torch
        if scores is None or len(scores) == 0:
            return float("nan"), 0
        gen = sequences[0, prompt_len:]
        logprobs = []
        n = min(len(scores), len(gen))
        for t in range(n):
            log_softmax = torch.log_softmax(scores[t][0].float(), dim=-1)
            logprobs.append(float(log_softmax[gen[t]].cpu().item()))
        if not logprobs:
            return float("nan"), 0
        return float(np.mean(logprobs)), int(len(logprobs))

    def explain(self, aqi: AQIResult, top_features: dict,
                user_question: str = "", sample_index: int = -1,
                pm25_actual: float = float("nan")) -> ExplanationRecord:
        t0 = time.time()
        prompt_user = _build_prompt(aqi, top_features, user_question)

        # Lazy-load on the first call. If the load fails the loader sets
        # self.enabled = False, so the next branch falls through to the
        # template fallback.
        if self.enabled and self._model is None:
            self._load()
        if not self.enabled or self._model is None or self._tokenizer is None:
            parsed = _template_fallback(aqi, top_features, user_question)
            return ExplanationRecord(
                sample_index=sample_index, pm25_pred=float(aqi.pm25),
                pm25_actual=float(pm25_actual), aqi=int(aqi.aqi),
                aqi_category=aqi.category, top_features=dict(top_features),
                user_question=user_question, raw_output=json.dumps(parsed),
                parsed_json=parsed, parsed_ok=True, avg_logprob=float("nan"),
                n_tokens=0, wall_time_s=time.time() - t0, fallback_used=True,
            )

        import torch

        torch.manual_seed(self.cfg.seed)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt_user},
        ]
        prompt_text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(prompt_text, return_tensors="pt").to(self._device)
        prompt_len = inputs["input_ids"].shape[1]

        gen_kwargs = dict(
            max_new_tokens=self.cfg.max_new_tokens,
            do_sample=self.cfg.do_sample,
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
            pad_token_id=self._tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True,
        )
        with torch.no_grad():
            out = self._model.generate(**inputs, **gen_kwargs)

        avg_lp, n_tok = self._avg_logprob(out.scores, out.sequences, prompt_len)
        gen_ids = out.sequences[0, prompt_len:]
        text = self._tokenizer.decode(gen_ids, skip_special_tokens=True)

        parsed = _try_parse_json(text)
        ok = isinstance(parsed, dict) and "predicted_pm25_ugm3" in parsed
        if not ok:
            # graceful degradation — keep the raw text but fill the JSON
            # contract from the template so downstream code never crashes.
            parsed = _template_fallback(aqi, top_features, user_question)
            ok = False

        return ExplanationRecord(
            sample_index=sample_index, pm25_pred=float(aqi.pm25),
            pm25_actual=float(pm25_actual), aqi=int(aqi.aqi),
            aqi_category=aqi.category, top_features=dict(top_features),
            user_question=user_question, raw_output=text, parsed_json=parsed,
            parsed_ok=bool(ok), avg_logprob=float(avg_lp), n_tokens=int(n_tok),
            wall_time_s=time.time() - t0, fallback_used=False,
        )


def explain_many(samples: List[dict], cfg: LLMConfig | None = None,
                 enabled: bool = True) -> List[ExplanationRecord]:
    """Convenience: feed a list of dicts {sample_index, pm25_pred, pm25_actual,
    top_features, user_question?} into the explainer."""
    expl = QwenExplainer(cfg, enabled=enabled)
    out = []
    for s in samples:
        aqi = pm25_to_aqi(s["pm25_pred"])
        rec = expl.explain(
            aqi=aqi,
            top_features=s.get("top_features", {}),
            user_question=s.get("user_question", ""),
            sample_index=s.get("sample_index", -1),
            pm25_actual=s.get("pm25_actual", float("nan")),
        )
        out.append(rec)
    return out
