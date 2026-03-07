from __future__ import annotations

import os
import threading
from typing import Any


def map_label_confidence_to_ten(label: str, confidence: float) -> float:
    """Map DistilBERT sentiment output to a 1-10 score."""
    bounded_confidence = max(0.0, min(1.0, confidence))
    upper_label = label.upper()

    if upper_label == "POSITIVE":
        score = 5.5 + (4.5 * bounded_confidence)
    elif upper_label == "NEGATIVE":
        score = 5.5 - (4.5 * bounded_confidence)
    else:
        score = 5.5

    return round(max(1.0, min(10.0, score)), 2)


def classify_nps_bucket(score: float) -> str:
    if score >= 9.0:
        return "Promoter"
    if score >= 7.0:
        return "Passive"
    return "Detractor"


def calculate_nps(scores: list[float]) -> dict[str, Any]:
    total = len(scores)
    if total == 0:
        return {
            "total_reviews": 0,
            "promoters": 0,
            "passives": 0,
            "detractors": 0,
            "nps_score": 0.0,
        }

    promoters = sum(1 for score in scores if score >= 9.0)
    passives = sum(1 for score in scores if 7.0 <= score < 9.0)
    detractors = sum(1 for score in scores if score < 7.0)

    nps_score = ((promoters / total) * 100.0) - ((detractors / total) * 100.0)
    return {
        "total_reviews": total,
        "promoters": promoters,
        "passives": passives,
        "detractors": detractors,
        "nps_score": round(nps_score, 2),
    }


class SentimentScorer:
    MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
    # Pin revision for deterministic downloads and reduced supply-chain risk.
    MODEL_REVISION = os.getenv(
        "SENTIMENT_MODEL_REVISION",
        "714eb0fa89d2f80546fda750413ed43d93601a13",
    ).strip()
    BATCH_SIZE = int(os.getenv("SENTIMENT_BATCH_SIZE", "12"))
    QUANTIZE_DYNAMIC = os.getenv("SENTIMENT_DYNAMIC_QUANTIZE", "1").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    _pipeline_instance: Any | None = None
    _init_lock = threading.Lock()

    def __init__(self) -> None:
        if SentimentScorer._pipeline_instance is None:
            with SentimentScorer._init_lock:
                if SentimentScorer._pipeline_instance is None:
                    SentimentScorer._pipeline_instance = self._build_pipeline()

        self._pipeline = SentimentScorer._pipeline_instance

    @classmethod
    def _build_pipeline(cls) -> Any:
        try:
            import torch
            import torch.nn as nn
            from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
        except ImportError as exc:
            raise RuntimeError(
                "transformers/torch are not installed. Run: pip install -r requirements.txt"
            ) from exc

        torch.set_num_threads(max(1, int(os.getenv("TORCH_NUM_THREADS", "1"))))

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                cls.MODEL_NAME,
                revision=cls.MODEL_REVISION or None,
            )
            try:
                model = AutoModelForSequenceClassification.from_pretrained(
                    cls.MODEL_NAME,
                    revision=cls.MODEL_REVISION or None,
                    low_cpu_mem_usage=True,
                )
            except ImportError:
                # `low_cpu_mem_usage` needs `accelerate`; fallback keeps compatibility.
                model = AutoModelForSequenceClassification.from_pretrained(
                    cls.MODEL_NAME,
                    revision=cls.MODEL_REVISION or None,
                )

            if cls.QUANTIZE_DYNAMIC:
                try:
                    model = torch.quantization.quantize_dynamic(
                        model,
                        {nn.Linear},
                        dtype=torch.qint8,
                    )
                except Exception as quantization_exc:
                    # Continue with full precision model if quantization is unsupported.
                    if os.getenv("SENTIMENT_LOG_QUANTIZE_ERRORS", "").strip().lower() in {
                        "1",
                        "true",
                        "yes",
                        "on",
                    }:
                        print(f"Dynamic quantization skipped: {quantization_exc}")

            return pipeline(
                "sentiment-analysis",
                model=model,
                tokenizer=tokenizer,
                device=-1,
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to load DistilBERT sentiment model. "
                "Check network/model download access and retry."
            ) from exc

    @staticmethod
    def _extract_result_fields(result: Any) -> tuple[str, float]:
        if not isinstance(result, dict):
            return "", 0.0

        label = str(result.get("label", ""))
        raw_confidence = result.get("score", 0.0)
        try:
            confidence = float(raw_confidence)
        except (TypeError, ValueError):
            confidence = 0.0
        return label, confidence

    def score_many_to_ten(self, texts: list[str], batch_size: int | None = None) -> list[float]:
        if not texts:
            return []

        scores = [5.5 for _ in texts]
        clean_inputs: list[str] = []
        clean_indexes: list[int] = []

        for index, raw_text in enumerate(texts):
            cleaned = str(raw_text).strip()
            if not cleaned:
                continue
            clean_inputs.append(cleaned)
            clean_indexes.append(index)

        if not clean_inputs:
            return scores

        raw_results = self._pipeline(
            clean_inputs,
            truncation=True,
            batch_size=batch_size or self.BATCH_SIZE,
        )
        if isinstance(raw_results, dict):
            raw_results = [raw_results]
        if not isinstance(raw_results, list):
            return scores

        for index, result in zip(clean_indexes, raw_results):
            label, confidence = self._extract_result_fields(result)
            scores[index] = map_label_confidence_to_ten(label, confidence)

        return scores

    def score_to_ten(self, text: str) -> float:
        return self.score_many_to_ten([text])[0]
