"""Interactive QA module that re-uses the LLM explainer with a user query."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .aqi_utils import pm25_to_aqi
from .llm_explainer import QwenExplainer, ExplanationRecord


DEFAULT_QUESTIONS = [
    "Is it safe to go for a run today?",
    "Should children play outside this afternoon?",
    "Do I need to wear an N95 mask if I commute by bicycle?",
    "Is it OK to open the windows for ventilation?",
    "Should outdoor school sports be cancelled today?",
]


@dataclass
class QAExample:
    sample_index:  int
    question:      str
    pm25_pred:     float
    pm25_actual:   float
    record:        ExplanationRecord


def run_qa(samples: List[dict], explainer: QwenExplainer | None = None,
           questions: List[str] | None = None) -> List[QAExample]:
    """Pair every sample with every question and call the explainer."""
    questions = questions or DEFAULT_QUESTIONS
    explainer = explainer or QwenExplainer()
    out: List[QAExample] = []
    for s in samples:
        aqi = pm25_to_aqi(s["pm25_pred"])
        for q in questions:
            rec = explainer.explain(
                aqi=aqi,
                top_features=s.get("top_features", {}),
                user_question=q,
                sample_index=s.get("sample_index", -1),
                pm25_actual=s.get("pm25_actual", float("nan")),
            )
            out.append(QAExample(
                sample_index=s.get("sample_index", -1),
                question=q,
                pm25_pred=float(s["pm25_pred"]),
                pm25_actual=float(s.get("pm25_actual", float("nan"))),
                record=rec,
            ))
    return out
