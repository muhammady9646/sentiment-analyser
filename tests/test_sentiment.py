import unittest

from services.sentiment import (
    AspectSentimentScorer,
    SentimentScorer,
    build_selected_aspects,
    calculate_nps,
    classify_nps_bucket,
    map_label_confidence_to_ten,
)


class TestSentimentUtilities(unittest.TestCase):
    def test_label_confidence_mapping(self) -> None:
        self.assertEqual(map_label_confidence_to_ten("NEGATIVE", 1.0), 1.0)
        self.assertEqual(map_label_confidence_to_ten("POSITIVE", 1.0), 10.0)
        self.assertEqual(map_label_confidence_to_ten("POSITIVE", 0.0), 5.5)
        self.assertEqual(map_label_confidence_to_ten("NEUTRAL", 0.7), 5.5)

    def test_nps_bucket_classification(self) -> None:
        self.assertEqual(classify_nps_bucket(9.4), "Promoter")
        self.assertEqual(classify_nps_bucket(7.5), "Passive")
        self.assertEqual(classify_nps_bucket(6.9), "Detractor")

    def test_calculate_nps(self) -> None:
        scores = [9.2, 9.0, 8.0, 6.8, 5.5]
        result = calculate_nps(scores)
        self.assertEqual(result["total_reviews"], 5)
        self.assertEqual(result["promoters"], 2)
        self.assertEqual(result["passives"], 1)
        self.assertEqual(result["detractors"], 2)
        self.assertEqual(result["nps_score"], 0.0)

    def test_score_many_to_ten_batches_and_keeps_blank_defaults(self) -> None:
        class FakePipeline:
            def __init__(self) -> None:
                self.seen_texts: list[str] = []
                self.seen_batch_size: int | None = None

            def __call__(self, texts, truncation=True, batch_size=None):
                self.seen_texts = list(texts)
                self.seen_batch_size = batch_size
                return [
                    {"label": "POSITIVE", "score": 1.0},
                    {"label": "NEGATIVE", "score": 1.0},
                ]

        fake = FakePipeline()
        scorer = SentimentScorer.__new__(SentimentScorer)
        scorer._pipeline = fake

        scores = scorer.score_many_to_ten(["Amazing service", " ", "Terrible support"], batch_size=16)
        self.assertEqual(scores, [10.0, 5.5, 1.0])
        self.assertEqual(fake.seen_texts, ["Amazing service", "Terrible support"])
        self.assertEqual(fake.seen_batch_size, 16)

    def test_build_selected_aspects_requires_any_selection(self) -> None:
        aspects, error = build_selected_aspects([], "")
        self.assertEqual(aspects, [])
        self.assertEqual(error, "Select at least one aspect before running analysis.")

    def test_build_selected_aspects_supports_custom_only(self) -> None:
        aspects, error = build_selected_aspects([], ["Queue speed"])
        self.assertIsNone(error)
        self.assertEqual(len(aspects), 1)
        self.assertEqual(aspects[0]["value"], "custom")
        self.assertEqual(aspects[0]["label"], "Queue speed")

    def test_build_selected_aspects_accepts_multiple_custom_names(self) -> None:
        aspects, error = build_selected_aspects(["cost"], ["Menu variety", "Queue speed"])
        self.assertIsNone(error)
        self.assertEqual(aspects[0]["value"], "cost")
        custom_labels = [aspect["label"] for aspect in aspects if aspect["value"] == "custom"]
        self.assertEqual(custom_labels, ["Menu variety", "Queue speed"])

    def test_aspect_scorer_returns_none_for_missing_aspect(self) -> None:
        class FakeBaseScorer:
            def score_many_to_ten(self, texts, batch_size=None):  # type: ignore[no-untyped-def]
                return [9.0 for _ in texts]

        aspects, error = build_selected_aspects(["cost", "location"], "")
        self.assertIsNone(error)
        scorer = AspectSentimentScorer(base_scorer=FakeBaseScorer())

        rows = scorer.score_aspects_for_many(
            [
                "Great value and affordable prices.",
                "The location is easy to access.",
                "Staff were polite and quick.",
            ],
            selected_aspects=aspects,
        )

        self.assertEqual(rows[0]["cost"], 9.0)
        self.assertIsNone(rows[0]["location"])
        self.assertIsNone(rows[1]["cost"])
        self.assertEqual(rows[1]["location"], 9.0)
        self.assertIsNone(rows[2]["cost"])
        self.assertIsNone(rows[2]["location"])


if __name__ == "__main__":
    unittest.main()
