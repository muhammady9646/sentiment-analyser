from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Iterator

import requests


class SerpApiError(Exception):
    pass


@dataclass
class ReviewRecord:
    author_name: str
    text: str
    google_rating: float | None
    review_time: str


class SerpApiReviewClient:
    SEARCH_URL = "https://serpapi.com/search.json"
    REQUEST_TIMEOUT_SECONDS = float(os.getenv("SERPAPI_TIMEOUT_SECONDS", "12"))
    GEOGRAPHY_MAP: dict[str, dict[str, str | None]] = {
        "global": {"label": "Global (No Bias)", "gl": None},
        "us": {"label": "United States", "gl": "us"},
        "uk": {"label": "United Kingdom", "gl": "uk"},
        "ca": {"label": "Canada", "gl": "ca"},
        "au": {"label": "Australia", "gl": "au"},
        "in": {"label": "India", "gl": "in"},
    }
    DEFAULT_GEOGRAPHY = "global"

    def __init__(self, api_key: str | None) -> None:
        self.api_key = api_key
        if not self.api_key:
            raise SerpApiError("SERPAPI_KEY is missing. Add it to your .env file.")

    @classmethod
    def geography_choices(cls) -> list[dict[str, str]]:
        return [{"value": key, "label": value["label"] or key} for key, value in cls.GEOGRAPHY_MAP.items()]

    @classmethod
    def normalize_geography(cls, geography: str | None) -> str:
        key = (geography or "").strip().lower()
        if key in cls.GEOGRAPHY_MAP:
            return key
        return cls.DEFAULT_GEOGRAPHY

    @classmethod
    def geography_label(cls, geography: str | None) -> str:
        key = cls.normalize_geography(geography)
        return cls.GEOGRAPHY_MAP[key]["label"] or key

    def find_place_candidate(
        self,
        company_name: str,
        geography: str = DEFAULT_GEOGRAPHY,
        location_hint: str = "",
    ) -> dict[str, Any]:
        candidates = self.find_place_candidates(
            company_name=company_name,
            geography=geography,
            location_hint=location_hint,
            limit=1,
        )
        return candidates[0]

    def find_place_candidates(
        self,
        company_name: str,
        geography: str = DEFAULT_GEOGRAPHY,
        location_hint: str = "",
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        geography_key = self.normalize_geography(geography)
        search_query = company_name.strip()
        hint = location_hint.strip()
        if hint:
            search_query = f"{search_query}, {hint}"

        params = {
            "engine": "google_maps",
            "q": search_query,
            "type": "search",
            "api_key": self.api_key,
            "hl": "en",
        }
        gl = self.GEOGRAPHY_MAP[geography_key]["gl"]
        if gl:
            params["gl"] = gl

        payload = self._request_json(params)
        self._raise_serpapi_error(payload)

        return self._extract_candidates(
            payload=payload,
            fallback_name=company_name,
            geography_key=geography_key,
            query=search_query,
            limit=max(1, limit),
        )

    def fetch_business_description(
        self, company_query: str, candidate: dict[str, Any]
    ) -> str:
        geography_key = self.normalize_geography(str(candidate.get("geography_key") or ""))
        name = str(candidate.get("name") or company_query).strip()
        address = str(candidate.get("address") or "").strip()
        search_query = name
        if address and address.lower() != "unknown":
            search_query = f"{name} {address}".strip()

        params: dict[str, Any] = {
            "engine": "google",
            "q": search_query,
            "api_key": self.api_key,
            "hl": "en",
        }
        gl = self.GEOGRAPHY_MAP[geography_key]["gl"]
        if gl:
            params["gl"] = gl

        payload = self._request_json(params)
        self._raise_serpapi_error(payload)
        return self._extract_business_description(payload=payload, candidate=candidate)

    def fetch_reviews_for_candidate(
        self, candidate: dict[str, Any], max_reviews: int = 20
    ) -> tuple[str, str, list[dict[str, Any]]]:
        id_key = candidate.get("id_key")
        id_value = candidate.get("id_value")
        if not isinstance(id_key, str) or not isinstance(id_value, str):
            raise SerpApiError("Place candidate is missing a valid identifier.")

        reviews = self._fetch_reviews_paginated(
            id_key=id_key,
            place_id_or_data_id=id_value,
            max_reviews=max_reviews,
        )

        if not reviews:
            raise SerpApiError(
                "No text reviews found for that company. Try a more specific company name."
            )
        place_name = str(candidate.get("name") or "Unknown")
        address = str(candidate.get("address") or "Unknown")
        return place_name, address, [review.__dict__ for review in reviews]

    def fetch_reviews_for_candidates(
        self,
        candidates: list[dict[str, Any]],
        max_reviews: int = 20,
    ) -> tuple[list[dict[str, Any]], list[str]]:
        if not candidates:
            raise SerpApiError("No store candidates found to analyze.")

        total_target = max(1, max_reviews)

        aggregated: list[dict[str, Any]] = []
        stores_with_reviews: list[str] = []
        errors: list[str] = []

        for candidate in candidates:
            remaining = total_target - len(aggregated)
            if remaining <= 0:
                break

            try:
                store_name, _, reviews = self.fetch_reviews_for_candidate(
                    candidate=candidate,
                    max_reviews=remaining,
                )
            except SerpApiError as exc:
                # Allow partial success across a set of stores.
                errors.append(str(exc))
                continue

            if not reviews:
                continue

            stores_with_reviews.append(store_name)
            for review in reviews:
                row = dict(review)
                row["store_name"] = store_name
                row["store_id"] = str(candidate.get("id_value") or "")
                aggregated.append(row)
                if len(aggregated) >= total_target:
                    break

        if aggregated:
            return aggregated, stores_with_reviews

        if errors:
            raise SerpApiError(errors[0])
        raise SerpApiError("No text reviews found for the matched stores.")

    def iter_reviews_for_candidates(
        self,
        candidates: list[dict[str, Any]],
        max_reviews: int = 20,
    ) -> Iterator[dict[str, Any]]:
        if not candidates:
            raise SerpApiError("No store candidates found to analyze.")

        total_target = max(1, max_reviews)
        yielded = 0
        errors: list[str] = []

        for candidate in candidates:
            if yielded >= total_target:
                break

            id_key = candidate.get("id_key")
            id_value = candidate.get("id_value")
            if not isinstance(id_key, str) or not isinstance(id_value, str):
                errors.append("Place candidate is missing a valid identifier.")
                continue

            store_name = str(candidate.get("name") or "Unknown")
            try:
                for review in self._iter_reviews_paginated(
                    id_key=id_key,
                    place_id_or_data_id=id_value,
                    max_reviews=total_target - yielded,
                ):
                    row = dict(review.__dict__)
                    row["store_name"] = store_name
                    row["store_id"] = str(candidate.get("id_value") or "")
                    yield row
                    yielded += 1
                    if yielded >= total_target:
                        break
            except SerpApiError as exc:
                errors.append(str(exc))

        if yielded == 0:
            if errors:
                raise SerpApiError(errors[0])
            raise SerpApiError("No text reviews found for the matched stores.")

    def fetch_reviews(
        self,
        company_name: str,
        max_reviews: int = 20,
        geography: str = DEFAULT_GEOGRAPHY,
        location_hint: str = "",
    ) -> tuple[str, str, list[dict[str, Any]]]:
        candidate = self.find_place_candidate(
            company_name=company_name,
            geography=geography,
            location_hint=location_hint,
        )
        return self.fetch_reviews_for_candidate(candidate=candidate, max_reviews=max_reviews)

    def _extract_candidates(
        self,
        payload: dict[str, Any],
        fallback_name: str,
        geography_key: str,
        query: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        seen_ids: set[tuple[str, str]] = set()

        source = payload.get("place_results")
        if isinstance(source, dict):
            candidate = self._candidate_from_item(
                item=source,
                fallback_name=fallback_name,
                geography_key=geography_key,
                query=query,
            )
            if candidate:
                key = (candidate["id_key"], candidate["id_value"])
                seen_ids.add(key)
                candidates.append(candidate)
                if len(candidates) >= limit:
                    return candidates

        local_results = payload.get("local_results", [])
        if isinstance(local_results, list):
            for item in local_results:
                if not isinstance(item, dict):
                    continue
                candidate = self._candidate_from_item(
                    item=item,
                    fallback_name=fallback_name,
                    geography_key=geography_key,
                    query=query,
                )
                if candidate:
                    key = (candidate["id_key"], candidate["id_value"])
                    if key in seen_ids:
                        continue
                    seen_ids.add(key)
                    candidates.append(candidate)
                    if len(candidates) >= limit:
                        return candidates

        if candidates:
            return candidates
        raise SerpApiError(f'No place found for "{fallback_name}".')

    def _candidate_from_item(
        self,
        item: dict[str, Any],
        fallback_name: str,
        geography_key: str,
        query: str,
    ) -> dict[str, Any] | None:
        place_id = item.get("place_id")
        data_id = item.get("data_id")
        id_key: str | None = None
        id_value: str | None = None
        if isinstance(place_id, str) and place_id.strip():
            id_key = "place_id"
            id_value = place_id.strip()
        elif isinstance(data_id, str) and data_id.strip():
            id_key = "data_id"
            id_value = data_id.strip()

        if not id_key or not id_value:
            return None

        name = item.get("title") or fallback_name
        address = item.get("address") or "Unknown"
        return {
            "id_key": id_key,
            "id_value": id_value,
            "name": str(name),
            "address": str(address),
            "category": self._extract_category(item),
            "rating": self._extract_rating(item.get("rating")),
            "review_count": self._extract_int(item.get("reviews") or item.get("reviews_count")),
            "geography_key": geography_key,
            "geography_label": self.geography_label(geography_key),
            "query": query,
        }

    def _extract_business_description(
        self, payload: dict[str, Any], candidate: dict[str, Any]
    ) -> str:
        knowledge_graph = payload.get("knowledge_graph")
        if isinstance(knowledge_graph, dict):
            for key in ("description", "merchant_description"):
                value = knowledge_graph.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()

        organic_results = payload.get("organic_results")
        if isinstance(organic_results, list) and organic_results:
            first = organic_results[0]
            if isinstance(first, dict):
                snippet = first.get("snippet")
                if isinstance(snippet, str) and snippet.strip():
                    return snippet.strip()

        name = str(candidate.get("name") or "This business").strip()
        category = str(candidate.get("category") or "Unknown category").strip()
        address = str(candidate.get("address") or "").strip()
        rating = candidate.get("rating")
        review_count = candidate.get("review_count")

        parts = [f"{name} is listed on Google as {category}."]
        if address and address.lower() != "unknown":
            parts.append(f"Address: {address}.")
        if rating is not None and review_count is not None:
            parts.append(f"Google rating: {rating} from about {review_count} reviews.")
        elif rating is not None:
            parts.append(f"Google rating: {rating}.")
        return " ".join(parts)

    def _fetch_reviews_paginated(
        self, id_key: str, place_id_or_data_id: str, max_reviews: int
    ) -> list[ReviewRecord]:
        return list(
            self._iter_reviews_paginated(
                id_key=id_key,
                place_id_or_data_id=place_id_or_data_id,
                max_reviews=max_reviews,
            )
        )

    def _iter_reviews_paginated(
        self, id_key: str, place_id_or_data_id: str, max_reviews: int
    ) -> Iterator[ReviewRecord]:
        yielded = 0
        next_page_token: str | None = None
        seen_tokens: set[str] = set()
        seen_reviews: set[tuple[str, str, str]] = set()

        while yielded < max_reviews:
            params: dict[str, Any] = {
                "engine": "google_maps_reviews",
                "api_key": self.api_key,
                "hl": "en",
            }
            params[id_key] = place_id_or_data_id
            if next_page_token:
                if next_page_token in seen_tokens:
                    break
                params["next_page_token"] = next_page_token
                params["num"] = min(20, max_reviews - yielded)
                seen_tokens.add(next_page_token)

            payload = self._request_json(params)
            self._raise_serpapi_error(payload)

            raw_reviews = payload.get("reviews", [])
            if not raw_reviews:
                break

            for raw in raw_reviews:
                text = self._extract_review_text(raw)
                if not text:
                    continue

                author_name = self._extract_author(raw)
                review_time = (
                    str(raw.get("date") or raw.get("published_at") or raw.get("relative_date") or "Unknown")
                )
                signature = (author_name, text, review_time)
                if signature in seen_reviews:
                    continue

                seen_reviews.add(signature)
                yielded += 1
                yield ReviewRecord(
                    author_name=author_name,
                    text=text,
                    google_rating=self._extract_rating(raw.get("rating")),
                    review_time=review_time,
                )
                if yielded >= max_reviews:
                    break

            if yielded >= max_reviews:
                break

            next_page_token = payload.get("serpapi_pagination", {}).get("next_page_token")
            if not next_page_token:
                break

    @staticmethod
    def _extract_review_text(review: dict[str, Any]) -> str:
        for key in ("snippet", "text", "description", "summary"):
            value = review.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""

    @staticmethod
    def _extract_author(review: dict[str, Any]) -> str:
        user = review.get("user")
        if isinstance(user, dict):
            name = user.get("name")
            if isinstance(name, str) and name.strip():
                return name.strip()
        author = review.get("author_name")
        if isinstance(author, str) and author.strip():
            return author.strip()
        return "Unknown"

    @staticmethod
    def _extract_rating(rating: Any) -> float | None:
        if isinstance(rating, (int, float)):
            return float(rating)
        if isinstance(rating, str):
            try:
                return float(rating.strip())
            except ValueError:
                return None
        return None

    @staticmethod
    def _extract_int(value: Any) -> int | None:
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        if isinstance(value, str):
            raw = value.replace(",", "").strip()
            if raw.isdigit():
                return int(raw)
        return None

    @staticmethod
    def _extract_category(item: dict[str, Any]) -> str:
        for key in ("type", "category"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

        types = item.get("types")
        if isinstance(types, list):
            parts = [part.strip() for part in types if isinstance(part, str) and part.strip()]
            if parts:
                return ", ".join(parts[:3])
        return "Unknown"

    def _request_json(self, params: dict[str, Any]) -> dict[str, Any]:
        try:
            response = requests.get(
                self.SEARCH_URL,
                params=params,
                timeout=self.REQUEST_TIMEOUT_SECONDS,
            )
            if response.status_code >= 400:
                try:
                    payload = response.json()
                    if isinstance(payload, dict):
                        error_message = payload.get("error")
                        if isinstance(error_message, str) and error_message.strip():
                            raise SerpApiError(error_message.strip())
                except ValueError:
                    pass
                response.raise_for_status()

            payload = response.json()
            if not isinstance(payload, dict):
                raise SerpApiError("Unexpected response format from SerpAPI.")
            return payload
        except requests.Timeout as exc:
            raise SerpApiError("SerpAPI request timed out. Please try again.") from exc
        except requests.RequestException as exc:
            raise SerpApiError("Network error while reaching SerpAPI.") from exc
        except ValueError as exc:
            raise SerpApiError("SerpAPI returned invalid JSON.") from exc

    @staticmethod
    def _raise_serpapi_error(payload: dict[str, Any]) -> None:
        error_message = payload.get("error")
        if isinstance(error_message, str) and error_message.strip():
            raise SerpApiError(error_message.strip())
