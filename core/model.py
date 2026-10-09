"""Singleton Laya model loader and raw-output normalizer.

Laya's raw output is not API-shaped: score is an expected ordinal level
(0..N-1) and noul is P(true). `normalize()` turns it into the decision dict
the router and API consume.
"""

import logging
import os
import threading
import time
from typing import Any

os.environ.setdefault("USE_TF", "0")  # laya.load() can hang if TensorFlow is probed

from core.questions import TRIAGE_QUESTIONS, URGENCY_MAX_LEVEL  # noqa: E402

logger = logging.getLogger("triage.model")

DEFAULT_MODEL = "convaiinnovations/laya-typed-decisions"


def noul_thresholds() -> tuple[float, float]:
    """(false_below, true_at_or_above). Between them the answer is None (unsure)."""
    low = float(os.environ.get("NOUL_FALSE_BELOW", "0.4"))
    high = float(os.environ.get("NOUL_TRUE_AT_OR_ABOVE", "0.6"))
    return low, high


def noul_answer(p: float) -> bool | None:
    low, high = noul_thresholds()
    if p >= high:
        return True
    if p < low:
        return False
    return None


def normalize(raw: dict[str, Any], top_n: int = 3) -> dict[str, Any]:
    """Convert one raw Laya result into normalized decisions."""
    answers = raw["answers"]
    out: dict[str, Any] = {}
    for name in ("issue_type", "assign_to"):
        a = answers[name]
        top = sorted(a["probabilities"].items(), key=lambda kv: kv[1], reverse=True)[:top_n]
        out[name] = {
            "answer": a["choice"],
            "confidence": round(float(a["answer_confidence"]), 4),
            "probabilities": {k: round(float(v), 4) for k, v in top},
        }
    u = answers["urgency"]
    out["urgency"] = {
        "score": round(min(max(float(u["score"]) / URGENCY_MAX_LEVEL, 0.0), 1.0), 4),
        "confidence": round(float(u["answer_confidence"]), 4),
    }
    for name in ("mentions_refund", "customer_angry"):
        p = float(answers[name]["noul"])
        out[name] = {
            "answer": noul_answer(p),
            "probability": round(p, 4),
            "confidence": round(max(p, 1.0 - p), 4),
        }
    return out


class TriageModel:
    """Loads Laya once; serializes inference (torch module is not request-safe)."""

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or os.environ.get("MODEL_NAME", DEFAULT_MODEL)
        self._agent: Any = None
        self._load_lock = threading.Lock()
        self._infer_lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._agent is not None

    def load(self) -> None:
        with self._load_lock:
            if self._agent is not None:
                return
            import laya  # deferred: heavy import, and keeps tests/stubs light

            kwargs: dict[str, Any] = {}
            if token := os.environ.get("HF_TOKEN"):
                kwargs["token"] = token
            t0 = time.perf_counter()
            self._agent = laya.load(self.model_name, **kwargs)
            logger.info(
                "model_loaded model=%s load_s=%.1f", self.model_name, time.perf_counter() - t0
            )

    def predict(self, text: str) -> dict[str, Any]:
        return self.predict_batch([text])[0]

    def predict_batch(self, texts: list[str]) -> list[dict[str, Any]]:
        if self._agent is None:
            raise RuntimeError("model not loaded")
        with self._infer_lock:
            if len(texts) == 1:
                raws = [self._agent.predict(texts[0], TRIAGE_QUESTIONS)]
            else:
                raws = self._agent.predict_batch(texts, TRIAGE_QUESTIONS)
        return [normalize(r) for r in raws]
