import csv
import io
import os
import uuid
from typing import Any

from dotenv import load_dotenv
from flask import Flask, Response, abort, render_template, request

from services.serpapi_reviews import SerpApiError, SerpApiReviewClient
from services.sentiment import SentimentScorer, calculate_nps, classify_nps_bucket

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(dotenv_path=os.path.join(BASE_DIR, ".env"))

app = Flask(__name__)

# In-memory caches for previews and downloadable reports.
REPORT_CACHE: dict[str, dict[str, Any]] = {}
PREVIEW_CACHE: dict[str, dict[str, Any]] = {}
MAX_CACHE_ITEMS = 100
MAX_BRANDS = 8
MAX_REVIEWS_PER_BRAND = 200


def _cache_record(cache: dict[str, dict[str, Any]], record: dict[str, Any]) -> str:
    record_id = str(uuid.uuid4())
    cache[record_id] = record
    if len(cache) > MAX_CACHE_ITEMS:
        oldest_key = next(iter(cache))
        cache.pop(oldest_key, None)
    return record_id


def _parse_requested_reviews(raw_value: str) -> int:
    if not raw_value.strip():
        return 0
    try:
        return max(0, min(int(raw_value.strip()), MAX_REVIEWS_PER_BRAND))
    except ValueError:
        return 0


def _extract_raw_brand_inputs(form: Any) -> list[dict[str, str]]:
    names = form.getlist("brand_name[]")
    row_count = max(len(names), 1)

    rows: list[dict[str, str]] = []
    for i in range(row_count):
        name = names[i].strip() if i < len(names) else ""
        rows.append({"name": name})
    return rows[:MAX_BRANDS]


def _extract_valid_brand_inputs(raw_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in raw_rows if row["name"]]


def _build_preview_rows(preview_brands: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for brand in preview_brands:
        rows.append(
            {
                "brand_name": brand.get("brand_name", ""),
                "store_examples": brand.get("store_examples", []),
                "business_description": brand.get("business_description", ""),
                "geography_label": brand.get("geography_label", "Global (No Bias)"),
            }
        )
    return rows


def _build_analysis_rows(
    preview_brands: list[dict[str, Any]],
    requested_counts: dict[int, int] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, brand in enumerate(preview_brands):
        requested = (requested_counts or {}).get(index, 0)
        rows.append(
            {
                "index": index,
                "brand_name": brand.get("brand_name", ""),
                "brand_description": brand.get("business_description", ""),
                "estimated_review_count": brand.get("estimated_review_count"),
                "requested_reviews": requested,
            }
        )
    return rows


def _brand_inputs_from_preview(preview_brands: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [
        {
            "name": str(brand.get("brand_name", "")),
        }
        for brand in preview_brands
    ]


def _render_index(**kwargs: Any) -> str:
    selected_geography = SerpApiReviewClient.normalize_geography(kwargs.get("selected_geography"))
    brand_inputs = kwargs.get("brand_inputs") or [{"name": ""}]
    context = {
        "geography_choices": SerpApiReviewClient.geography_choices(),
        "selected_geography": selected_geography,
        "brand_inputs": brand_inputs,
        "preview_rows": kwargs.get("preview_rows", []),
        "analysis_rows": kwargs.get("analysis_rows", []),
        "summary_rows": kwargs.get("summary_rows", []),
        "detailed_rows": kwargs.get("detailed_rows", []),
        "analysis_errors": kwargs.get("analysis_errors", []),
        "show_analysis_panel": kwargs.get("show_analysis_panel", False),
        "auto_scroll_target": kwargs.get("auto_scroll_target", ""),
    }
    context.update(kwargs)
    return render_template("index.html", **context)


@app.route("/", methods=["GET"])
def home() -> str:
    return _render_index(
        selected_geography=SerpApiReviewClient.DEFAULT_GEOGRAPHY,
        brand_inputs=[{"name": ""}],
    )


@app.route("/preview", methods=["POST"])
def preview() -> str:
    geography = request.form.get("geography", SerpApiReviewClient.DEFAULT_GEOGRAPHY)
    raw_brand_inputs = _extract_raw_brand_inputs(request.form)
    valid_brands = _extract_valid_brand_inputs(raw_brand_inputs)

    if not valid_brands:
        return _render_index(
            error="Add at least one brand name.",
            selected_geography=geography,
            brand_inputs=raw_brand_inputs,
        )

    try:
        review_client = SerpApiReviewClient(api_key=os.getenv("SERPAPI_KEY"))
        preview_brands: list[dict[str, Any]] = []

        for brand in valid_brands:
            candidates = review_client.find_place_candidates(
                company_name=brand["name"],
                geography=geography,
                location_hint="",
                limit=5,
            )
            primary_candidate = candidates[0]
            business_description = review_client.fetch_business_description(
                company_query=brand["name"],
                candidate=primary_candidate,
            )

            counts = [
                c.get("review_count")
                for c in candidates
                if isinstance(c.get("review_count"), int)
            ]
            estimated_review_count = sum(counts) if counts else None

            preview_brands.append(
                {
                    "brand_name": brand["name"],
                    "geography_key": primary_candidate.get("geography_key", geography),
                    "geography_label": primary_candidate.get(
                        "geography_label",
                        SerpApiReviewClient.geography_label(geography),
                    ),
                    "store_examples": [c.get("name", "Unknown") for c in candidates],
                    "business_description": business_description,
                    "estimated_review_count": estimated_review_count,
                    "candidates": candidates,
                }
            )
    except SerpApiError as exc:
        return _render_index(
            error=str(exc),
            selected_geography=geography,
            brand_inputs=raw_brand_inputs,
        )

    preview_id = _cache_record(
        PREVIEW_CACHE,
        {
            "brands": preview_brands,
            "selected_geography": geography,
        },
    )

    return _render_index(
        info="Preview complete. Confirm brands to continue.",
        selected_geography=geography,
        brand_inputs=raw_brand_inputs,
        preview_id=preview_id,
        preview_rows=_build_preview_rows(preview_brands),
        analysis_rows=_build_analysis_rows(preview_brands),
        show_analysis_panel=False,
    )


@app.route("/analyze", methods=["POST"])
def analyze() -> str:
    preview_id = request.form.get("preview_id", "").strip()
    preview_record = PREVIEW_CACHE.get(preview_id)
    if not preview_record:
        return _render_index(
            error="Preview expired or missing. Please run Preview Places again before analyzing.",
        )

    preview_brands = preview_record.get("brands", [])
    if not preview_brands:
        return _render_index(
            error="No brand preview data found. Please run Preview Places again.",
        )

    brand_inputs = _brand_inputs_from_preview(preview_brands)
    selected_geography = preview_record.get(
        "selected_geography", SerpApiReviewClient.DEFAULT_GEOGRAPHY
    )

    requested_counts: dict[int, int] = {}
    for index in range(len(preview_brands)):
        raw = request.form.get(f"requested_reviews_{index}", "")
        requested_counts[index] = _parse_requested_reviews(raw)

    if not any(count > 0 for count in requested_counts.values()):
        return _render_index(
            error="Enter review counts for at least one brand before running analysis.",
            selected_geography=selected_geography,
            brand_inputs=brand_inputs,
            preview_id=preview_id,
            preview_rows=_build_preview_rows(preview_brands),
            analysis_rows=_build_analysis_rows(preview_brands, requested_counts),
            show_analysis_panel=True,
            auto_scroll_target="analysis-section",
        )

    try:
        review_client = SerpApiReviewClient(api_key=os.getenv("SERPAPI_KEY"))
        scorer = SentimentScorer()
    except (SerpApiError, RuntimeError) as exc:
        return _render_index(
            error=str(exc),
            selected_geography=selected_geography,
            brand_inputs=brand_inputs,
            preview_id=preview_id,
            preview_rows=_build_preview_rows(preview_brands),
            analysis_rows=_build_analysis_rows(preview_brands, requested_counts),
            show_analysis_panel=True,
            auto_scroll_target="analysis-section",
        )

    summary_rows: list[dict[str, Any]] = []
    detailed_rows: list[dict[str, Any]] = []
    analysis_errors: list[str] = []

    for index, brand in enumerate(preview_brands):
        desired_reviews = requested_counts.get(index, 0)
        brand_name = str(brand.get("brand_name", "Unknown"))
        brand_description = str(brand.get("business_description", ""))
        candidates = brand.get("candidates") or []

        brand_sentiment_scores: list[float] = []
        brand_google_ratings: list[float] = []

        if desired_reviews > 0:
            try:
                raw_reviews, _ = review_client.fetch_reviews_for_candidates(
                    candidates=candidates,
                    max_reviews=desired_reviews,
                )
            except SerpApiError as exc:
                analysis_errors.append(f"{brand_name}: {exc}")
                raw_reviews = []

            for review in raw_reviews:
                sentiment_score = scorer.score_to_ten(review["text"])
                brand_sentiment_scores.append(sentiment_score)

                rating = review.get("google_rating")
                if isinstance(rating, (int, float)):
                    brand_google_ratings.append(float(rating))

                detailed_rows.append(
                    {
                        "brand_name": brand_name,
                        "store_name": review.get("store_name", "Unknown"),
                        "store_id": review.get("store_id", ""),
                        "review_text": review["text"],
                        "review_date": review["review_time"],
                        "sentiment_score": sentiment_score,
                        "google_rating": review.get("google_rating"),
                        "nps_bucket": classify_nps_bucket(sentiment_score),
                    }
                )

        avg_google_score = (
            round(sum(brand_google_ratings) / len(brand_google_ratings), 2)
            if brand_google_ratings
            else None
        )
        nps_score = calculate_nps(brand_sentiment_scores)["nps_score"] if brand_sentiment_scores else 0.0

        summary_rows.append(
            {
                "brand_name": brand_name,
                "brand_description": brand_description,
                "review_count": len(brand_sentiment_scores),
                "avg_google_score": avg_google_score,
                "nps_score": nps_score,
            }
        )

    report_id = _cache_record(
        REPORT_CACHE,
        {
            "summary_rows": summary_rows,
            "detailed_rows": detailed_rows,
        },
    )

    info = "Analysis complete." if detailed_rows else "No review rows returned for selected brands."
    return _render_index(
        info=info,
        selected_geography=selected_geography,
        brand_inputs=brand_inputs,
        preview_id=preview_id,
        preview_rows=_build_preview_rows(preview_brands),
        analysis_rows=_build_analysis_rows(preview_brands, requested_counts),
        summary_rows=summary_rows,
        detailed_rows=detailed_rows,
        analysis_errors=analysis_errors,
        show_analysis_panel=True,
        auto_scroll_target="analysis-section",
        report_id=report_id,
    )


@app.route("/download/<report_id>", methods=["GET"])
def download_report(report_id: str) -> Response:
    report = REPORT_CACHE.get(report_id)
    if not report:
        abort(404, description="Report not found or expired.")

    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(
        [
            "brand_name",
            "store_name",
            "store_id",
            "review",
            "date_of_review",
            "sentiment_score_1_to_10",
            "google_rating",
            "nps_bucket",
        ]
    )

    for row in report["detailed_rows"]:
        writer.writerow(
            [
                row.get("brand_name", ""),
                row.get("store_name", ""),
                row.get("store_id", ""),
                row.get("review_text", ""),
                row.get("review_date", ""),
                row.get("sentiment_score", ""),
                row.get("google_rating", ""),
                row.get("nps_bucket", ""),
            ]
        )

    csv_data = buffer.getvalue()
    buffer.close()

    return Response(
        csv_data,
        mimetype="text/csv",
        headers={"Content-Disposition": 'attachment; filename="sentiment_detailed_dataset.csv"'},
    )


@app.route("/download-summary/<report_id>", methods=["GET"])
def download_summary(report_id: str) -> Response:
    report = REPORT_CACHE.get(report_id)
    if not report:
        abort(404, description="Report not found or expired.")

    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(
        [
            "brand_name",
            "brand_description",
            "number_of_reviews",
            "average_google_score",
            "calculated_nps",
        ]
    )

    for row in report["summary_rows"]:
        writer.writerow(
            [
                row.get("brand_name", ""),
                row.get("brand_description", ""),
                row.get("review_count", 0),
                row.get("avg_google_score", ""),
                row.get("nps_score", 0.0),
            ]
        )

    csv_data = buffer.getvalue()
    buffer.close()

    return Response(
        csv_data,
        mimetype="text/csv",
        headers={"Content-Disposition": 'attachment; filename="sentiment_summary_dataset.csv"'},
    )


if __name__ == "__main__":
    app.run(debug=True)
