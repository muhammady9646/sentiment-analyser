import unittest

from services.sentiment import calculate_nps, classify_nps_bucket, map_label_confidence_to_ten


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


if __name__ == "__main__":
    unittest.main()
