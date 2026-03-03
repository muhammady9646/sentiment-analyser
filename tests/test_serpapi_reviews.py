import unittest

from services.serpapi_reviews import SerpApiError, SerpApiReviewClient


class StubSerpApiClient(SerpApiReviewClient):
    def __init__(self, responses):
        super().__init__(api_key="test")
        self._responses = responses
        self.calls = []

    def _request_json(self, params):  # type: ignore[override]
        self.calls.append(dict(params))
        if not self._responses:
            return {}
        return self._responses.pop(0)


class TestSerpApiReviews(unittest.TestCase):
    def test_find_place_candidates_returns_multiple_unique(self):
        responses = [
            {
                "place_results": {
                    "place_id": "p0",
                    "title": "Brand - Primary",
                    "address": "A",
                },
                "local_results": [
                    {"place_id": "p0", "title": "Brand - Primary Duplicate"},
                    {"place_id": "p1", "title": "Brand Downtown", "address": "B"},
                    {"data_id": "d2", "title": "Brand Midtown", "address": "C"},
                ],
            }
        ]
        client = StubSerpApiClient(responses)
        candidates = client.find_place_candidates("brand", limit=5)

        self.assertEqual(len(candidates), 3)
        self.assertEqual(candidates[0]["id_value"], "p0")
        self.assertEqual(candidates[1]["id_value"], "p1")
        self.assertEqual(candidates[2]["id_value"], "d2")

    def test_find_place_candidate_prefers_place_results(self):
        responses = [
            {
                "place_results": {
                    "place_id": "abc123",
                    "title": "Sample Company",
                    "address": "123 Main St",
                    "type": "Coffee shop",
                    "rating": 4.4,
                    "reviews": 120,
                }
            }
        ]
        client = StubSerpApiClient(responses)
        candidate = client.find_place_candidate("sample")
        self.assertEqual(candidate["id_key"], "place_id")
        self.assertEqual(candidate["id_value"], "abc123")
        self.assertEqual(candidate["name"], "Sample Company")
        self.assertEqual(candidate["address"], "123 Main St")
        self.assertEqual(candidate["category"], "Coffee shop")
        self.assertEqual(candidate["rating"], 4.4)
        self.assertEqual(candidate["review_count"], 120)

    def test_fetch_reviews_paginated(self):
        responses = [
            {
                "reviews": [
                    {
                        "user": {"name": "A"},
                        "snippet": "Great service",
                        "rating": 5,
                        "date": "1 week ago",
                    }
                ],
                "serpapi_pagination": {"next_page_token": "token-1"},
            },
            {
                "reviews": [
                    {
                        "user": {"name": "B"},
                        "snippet": "Not good",
                        "rating": 2,
                        "date": "2 weeks ago",
                    }
                ],
            },
        ]
        client = StubSerpApiClient(responses)
        rows = client._fetch_reviews_paginated(
            id_key="place_id", place_id_or_data_id="abc123", max_reviews=5
        )

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0].author_name, "A")
        self.assertEqual(rows[1].author_name, "B")
        self.assertEqual(rows[0].google_rating, 5.0)
        self.assertEqual(rows[1].google_rating, 2.0)
        self.assertNotIn("num", client.calls[0])
        self.assertEqual(client.calls[1]["num"], 4)

    def test_find_place_candidate_uses_data_id_when_place_id_missing(self):
        responses = [
            {
                "local_results": [
                    {"data_id": "data-1", "title": "Alt Company", "address": "45 Park Ave"}
                ]
            }
        ]
        client = StubSerpApiClient(responses)
        candidate = client.find_place_candidate("alt")
        self.assertEqual(candidate["id_key"], "data_id")
        self.assertEqual(candidate["id_value"], "data-1")
        self.assertEqual(candidate["name"], "Alt Company")
        self.assertEqual(candidate["address"], "45 Park Ave")

    def test_find_place_candidate_uses_geography_and_location_hint(self):
        responses = [
            {
                "local_results": [
                    {"place_id": "p1", "title": "Hinted Place", "address": "Austin, TX"}
                ]
            }
        ]
        client = StubSerpApiClient(responses)
        candidate = client.find_place_candidate(
            company_name="Hinted",
            geography="us",
            location_hint="Austin, TX",
        )

        self.assertEqual(candidate["geography_key"], "us")
        self.assertEqual(candidate["geography_label"], "United States")
        self.assertEqual(client.calls[0]["gl"], "us")
        self.assertEqual(client.calls[0]["q"], "Hinted, Austin, TX")

    def test_fetch_reviews_for_candidate_requires_identifier(self):
        client = StubSerpApiClient([])
        with self.assertRaises(SerpApiError):
            client.fetch_reviews_for_candidate(candidate={"name": "Missing"}, max_reviews=5)

    def test_fetch_reviews_for_candidates_tags_store_name(self):
        responses = [
            {
                "reviews": [
                    {
                        "user": {"name": "U1"},
                        "snippet": "Great",
                        "rating": 5,
                        "date": "1 day ago",
                    },
                    {
                        "user": {"name": "U2"},
                        "snippet": "Good",
                        "rating": 4,
                        "date": "2 days ago",
                    },
                ]
            },
            {
                "reviews": [
                    {
                        "user": {"name": "U3"},
                        "snippet": "Okay",
                        "rating": 3,
                        "date": "3 days ago",
                    }
                ]
            },
        ]
        client = StubSerpApiClient(responses)
        rows, stores = client.fetch_reviews_for_candidates(
            candidates=[
                {"id_key": "place_id", "id_value": "p1", "name": "Store One", "address": "A"},
                {"id_key": "place_id", "id_value": "p2", "name": "Store Two", "address": "B"},
            ],
            max_reviews=4,
        )

        self.assertEqual(len(rows), 3)
        self.assertEqual(stores, ["Store One", "Store Two"])
        self.assertEqual(rows[0]["store_name"], "Store One")
        self.assertEqual(rows[2]["store_name"], "Store Two")

    def test_fetch_business_description_from_knowledge_graph(self):
        responses = [
            {
                "knowledge_graph": {
                    "description": "A multinational coffeehouse chain headquartered in Seattle.",
                }
            }
        ]
        client = StubSerpApiClient(responses)
        description = client.fetch_business_description(
            company_query="Starbucks",
            candidate={
                "name": "Starbucks Coffee Company",
                "address": "1500 Broadway, New York, NY",
                "geography_key": "us",
            },
        )

        self.assertIn("multinational coffeehouse chain", description)
        self.assertEqual(client.calls[0]["engine"], "google")
        self.assertEqual(client.calls[0]["gl"], "us")
        self.assertIn("Starbucks Coffee Company", client.calls[0]["q"])

    def test_fetch_business_description_fallback(self):
        responses = [{"organic_results": []}]
        client = StubSerpApiClient(responses)
        description = client.fetch_business_description(
            company_query="Fallback Biz",
            candidate={
                "name": "Fallback Biz",
                "address": "Unknown",
                "category": "Restaurant",
                "rating": 4.2,
                "review_count": 85,
            },
        )

        self.assertIn("Fallback Biz is listed on Google as Restaurant.", description)
        self.assertIn("Google rating: 4.2 from about 85 reviews.", description)


if __name__ == "__main__":
    unittest.main()
