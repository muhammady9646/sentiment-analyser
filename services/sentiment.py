from __future__ import annotations

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
    _pipeline_instance: Any | None = None

    def __init__(self) -> None:
        try:
            from transformers import pipeline
        except ImportError as exc:
            raise RuntimeError(
                "transformers/torch are not installed. Run: pip install -r requirements.txt"
            ) from exc

        if SentimentScorer._pipeline_instance is None:
            try:
                SentimentScorer._pipeline_instance = pipeline(
                    "sentiment-analysis",
                    model=self.MODEL_NAME,
                    tokenizer=self.MODEL_NAME,
                    device=-1,
                )
            except Exception as exc:
                raise RuntimeError(
                    "Failed to load DistilBERT sentiment model. "
                    "Check network/model download access and retry."
                ) from exc

        self._pipeline = SentimentScorer._pipeline_instance

    def score_to_ten(self, text: str) -> float:
        cleaned = text.strip()
        if not cleaned:
            return 5.5
        result = self._pipeline(cleaned, truncation=True)[0]
        label = str(result.get("label", ""))
        confidence = float(result.get("score", 0.0))
        return map_label_confidence_to_ten(label, confidence)
