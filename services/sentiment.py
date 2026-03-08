from __future__ import annotations

import os
import re
import threading
from typing import Any

ASPECT_CHOICES: tuple[dict[str, str], ...] = (
    {"value": "cost", "label": "Cost"},
    {"value": "customer_service", "label": "Customer service"},
    {"value": "location", "label": "Location"},
)
DEFAULT_SELECTED_ASPECT_VALUES: tuple[str, ...] = ("cost", "customer_service", "location")
ASPECT_KEYWORDS: dict[str, tuple[str, ...]] = {
    "cost": (
        "cost",
        "price",
        "pricing",
        "expensive",
        "cheap",
        "affordable",
        "overpriced",
        "value",
        "fees",
        "costly",
    ),
    "customer_service": (
        "service",
        "staff",
        "customer service",
        "support",
        "manager",
        "employee",
        "friendly",
        "helpful",
        "attentive",
        "rude",
    ),
    "location": (
        "location",
        "area",
        "parking",
        "distance",
        "nearby",
        "access",
        "accessible",
        "branch",
        "store",
        "neighborhood",
    ),
}


def aspect_choices() -> list[dict[str, str]]:
    return [dict(choice) for choice in ASPECT_CHOICES]


def default_aspect_values() -> list[str]:
    return list(DEFAULT_SELECTED_ASPECT_VALUES)


def normalize_custom_aspect(raw_value: str | None) -> str:
    return str(raw_value or "").strip()


def normalize_custom_aspect_values(raw_values: list[str] | str | None) -> list[str]:
    if raw_values is None:
        return []

    source_values: list[str]
    if isinstance(raw_values, str):
        source_values = [raw_values]
    else:
        source_values = [str(value) for value in raw_values]

    normalized: list[str] = []
    seen: set[str] = set()
    for value in source_values:
        cleaned = normalize_custom_aspect(value)
        if not cleaned:
            continue
        lowered = cleaned.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        normalized.append(cleaned)
    return normalized


def _slugify_aspect(raw_value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", raw_value.lower()).strip("_")
    return slug[:24] or "custom"


def build_selected_aspects(
    raw_values: list[str] | None,
    custom_aspect_values: list[str] | str | None = None,
) -> tuple[list[dict[str, Any]], str | None]:
    selected_values = {
        str(value).strip().lower()
        for value in (raw_values or [])
        if str(value).strip()
    }

    aspects: list[dict[str, Any]] = []
    for value in DEFAULT_SELECTED_ASPECT_VALUES:
        if value not in selected_values:
            continue
        aspects.append(
            {
                "value": value,
                "key": value,
                "label": next(
                    choice["label"]
                    for choice in ASPECT_CHOICES
                    if choice["value"] == value
                ),
                "keywords": list(ASPECT_KEYWORDS.get(value, (value,))),
            }
        )

    custom_values = normalize_custom_aspect_values(custom_aspect_values)
    used_keys: set[str] = {str(aspect.get("key", "")) for aspect in aspects}
    for custom_text in custom_values:
        lower_custom = custom_text.lower()
        token_keywords = [
            token
            for token in re.split(r"[^a-z0-9]+", lower_custom)
            if len(token) > 1
        ]
        ordered_keywords: list[str] = []
        for keyword in [lower_custom, *token_keywords]:
            if keyword and keyword not in ordered_keywords:
                ordered_keywords.append(keyword)

        base_key = f"custom_{_slugify_aspect(lower_custom)}"
        key = base_key
        suffix = 2
        while key in used_keys:
            key = f"{base_key}_{suffix}"
            suffix += 1
        used_keys.add(key)

        aspects.append(
            {
                "value": "custom",
                "key": key,
                "label": custom_text,
                "keywords": ordered_keywords,
            }
        )

    if not aspects:
        return [], "Select at least one aspect before running analysis."
    return aspects, None


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


class AspectSentimentScorer:
    SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")

    def __init__(self, base_scorer: SentimentScorer | None = None) -> None:
        self._base_scorer = base_scorer or SentimentScorer()

    @staticmethod
    def _keyword_in_text(lowered_text: str, keyword: str) -> bool:
        normalized = keyword.strip().lower()
        if not normalized:
            return False
        if " " in normalized:
            return normalized in lowered_text
        return re.search(rf"\b{re.escape(normalized)}\b", lowered_text) is not None

    @classmethod
    def _extract_aspect_snippet(
        cls,
        text: str,
        aspect_keywords: list[str],
    ) -> str:
        cleaned = str(text).strip()
        if not cleaned:
            return ""

        sentences = [
            sentence.strip()
            for sentence in cls.SENTENCE_SPLIT_RE.split(cleaned)
            if sentence.strip()
        ]
        if not sentences:
            sentences = [cleaned]

        matching_sentences: list[str] = []
        for sentence in sentences:
            lowered_sentence = sentence.lower()
            if any(cls._keyword_in_text(lowered_sentence, keyword) for keyword in aspect_keywords):
                matching_sentences.append(sentence)

        if matching_sentences:
            return " ".join(matching_sentences[:2]).strip()

        lowered_review = cleaned.lower()
        if any(cls._keyword_in_text(lowered_review, keyword) for keyword in aspect_keywords):
            return cleaned
        return ""

    def score_aspects_for_many(
        self,
        texts: list[str],
        selected_aspects: list[dict[str, Any]],
        batch_size: int | None = None,
    ) -> list[dict[str, float | None]]:
        if not texts:
            return []

        results = [
            {
                str(aspect.get("key", "")): None
                for aspect in selected_aspects
            }
            for _ in texts
        ]

        model_inputs: list[str] = []
        score_index_map: list[tuple[int, str]] = []
        for review_index, text in enumerate(texts):
            for aspect in selected_aspects:
                aspect_key = str(aspect.get("key", "")).strip()
                keywords = [
                    str(keyword).strip().lower()
                    for keyword in aspect.get("keywords", [])
                    if str(keyword).strip()
                ]
                if not aspect_key or not keywords:
                    continue

                snippet = self._extract_aspect_snippet(text, keywords)
                if not snippet:
                    continue
                model_inputs.append(snippet)
                score_index_map.append((review_index, aspect_key))

        if not model_inputs:
            return results

        scores = self._base_scorer.score_many_to_ten(
            model_inputs,
            batch_size=batch_size,
        )
        for (review_index, aspect_key), score in zip(score_index_map, scores):
            results[review_index][aspect_key] = score

        return results
