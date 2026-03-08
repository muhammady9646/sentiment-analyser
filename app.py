import csv
import concurrent.futures
import io
import os
import threading
import time
import uuid
from typing import Any

from dotenv import load_dotenv
from flask import Flask, Response, abort, jsonify, render_template, request, send_file

from services.serpapi_reviews import SerpApiError, SerpApiReviewClient
from services.sentiment import (
    AspectSentimentScorer,
    SentimentScorer,
    aspect_choices,
    build_selected_aspects,
    classify_nps_bucket,
    default_aspect_values,
    normalize_custom_aspect_values,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(dotenv_path=os.path.join(BASE_DIR, ".env"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

app = Flask(__name__)

# In-memory caches for previews and downloadable reports.
REPORT_CACHE: dict[str, dict[str, Any]] = {}
PREVIEW_CACHE: dict[str, dict[str, Any]] = {}
ANALYSIS_JOB_CACHE: dict[str, dict[str, Any]] = {}
ANALYSIS_JOB_LOCK = threading.Lock()
REPORT_CACHE_LOCK = threading.Lock()
ANALYSIS_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=1,
    thread_name_prefix="analysis-job",
)
MAX_CACHE_ITEMS = 100
MAX_BRANDS = 8
MAX_REVIEWS_PER_BRAND = int(os.getenv("MAX_REVIEWS_PER_BRAND", "1000"))
MAX_TOTAL_REVIEWS_PER_ANALYSIS = int(os.getenv("MAX_TOTAL_REVIEWS_PER_ANALYSIS", "1000"))
MAX_STORE_CANDIDATES_PER_BRAND = 3
MAX_ANALYSIS_CANDIDATES_PER_BRAND = max(
    1,
    int(os.getenv("MAX_ANALYSIS_CANDIDATES_PER_BRAND", "3")),
)
ANALYSIS_TIME_BUDGET_SECONDS = int(os.getenv("ANALYSIS_TIME_BUDGET_SECONDS", "1800"))
ANALYSIS_SENTIMENT_BATCH_SIZE = int(os.getenv("ANALYSIS_SENTIMENT_BATCH_SIZE", "12"))
DETAILED_PREVIEW_LIMIT = int(os.getenv("DETAILED_PREVIEW_LIMIT", "200"))
REPORTS_DIR = os.path.join(BASE_DIR, "data", "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)
DEFAULT_SELECTED_ASPECT_VALUES = default_aspect_values()
DEFAULT_SELECTED_ASPECTS, _ = build_selected_aspects(DEFAULT_SELECTED_ASPECT_VALUES, "")
TERMINAL_ANALYSIS_STATUSES = {"completed", "failed", "cancelled"}
ACTIVE_ANALYSIS_STATUSES = {"queued", "running", "paused"}
PAUSED_ANALYSIS_PROGRESS = "Paused. Click Resume to continue."
CANCELLING_ANALYSIS_PROGRESS = "Cancelling analysis..."


class AnalysisCancelledError(Exception):
    pass


def _cache_record(cache: dict[str, dict[str, Any]], record: dict[str, Any]) -> str:
    record_id = str(uuid.uuid4())
    cache[record_id] = record
    if len(cache) > MAX_CACHE_ITEMS:
        oldest_key = next(iter(cache))
        cache.pop(oldest_key, None)
    return record_id


def _cache_record_by_id(
    cache: dict[str, dict[str, Any]],
    record_id: str,
    record: dict[str, Any],
    lock: Any = None,
) -> str:
    if lock:
        with lock:
            cache[record_id] = record
            if len(cache) > MAX_CACHE_ITEMS:
                oldest_key = next(iter(cache))
                cache.pop(oldest_key, None)
    else:
        cache[record_id] = record
        if len(cache) > MAX_CACHE_ITEMS:
            oldest_key = next(iter(cache))
            cache.pop(oldest_key, None)
    return record_id


def _report_paths(report_id: str) -> tuple[str, str]:
    summary_path = os.path.join(REPORTS_DIR, f"{report_id}_summary.csv")
    detailed_path = os.path.join(REPORTS_DIR, f"{report_id}_detailed.csv")
    return summary_path, detailed_path


def _write_summary_csv(
    path: str,
    summary_rows: list[dict[str, Any]],
    selected_aspects: list[dict[str, Any]],
) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        headers = [
            "brand_name",
            "brand_description",
            "number_of_reviews",
            "average_google_score",
            "calculated_nps",
        ]
        for aspect in selected_aspects:
            aspect_label = str(aspect.get("label", "")).strip() or "Aspect"
            headers.append(f"{aspect_label} sentiment (1-10)")
            headers.append(f"{aspect_label} mentions")

        writer.writerow(headers)
        for row in summary_rows:
            values = [
                row.get("brand_name", ""),
                row.get("brand_description", ""),
                row.get("review_count", 0),
                row.get("avg_google_score", ""),
                row.get("nps_score", 0.0),
            ]
            aspect_averages = row.get("aspect_averages", {})
            aspect_mentions = row.get("aspect_mentions", {})
            for aspect in selected_aspects:
                aspect_key = str(aspect.get("key", "")).strip()
                avg_score = (
                    aspect_averages.get(aspect_key)
                    if isinstance(aspect_averages, dict)
                    else None
                )
                mention_count = (
                    aspect_mentions.get(aspect_key)
                    if isinstance(aspect_mentions, dict)
                    else None
                )
                values.append(avg_score if avg_score is not None else "N/A")
                values.append(mention_count if isinstance(mention_count, int) else 0)
            writer.writerow(values)

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
    source_indexes: list[int] | None = None,
) -> list[dict[str, Any]]:
    source_indexes = source_indexes or list(range(len(preview_brands)))
    rows: list[dict[str, Any]] = []
    for position, brand in enumerate(preview_brands):
        source_index = source_indexes[position] if position < len(source_indexes) else position
        requested = (requested_counts or {}).get(source_index, 0)
        rows.append(
            {
                "index": source_index,
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


def _parse_selected_brand_indexes(raw_values: list[str], total_brands: int) -> list[int]:
    if total_brands <= 0:
        return []
    if not raw_values:
        return list(range(total_brands))

    selected: list[int] = []
    seen: set[int] = set()
    for raw in raw_values:
        try:
            index = int(str(raw).strip())
        except ValueError:
            continue
        if index < 0 or index >= total_brands:
            continue
        if index in seen:
            continue
        seen.add(index)
        selected.append(index)
    return selected


def _serialize_aspect_values(raw_values: list[str] | None) -> list[str]:
    normalized: list[str] = []
    for value in raw_values or []:
        cleaned = str(value).strip().lower()
        if not cleaned or cleaned in normalized:
            continue
        normalized.append(cleaned)
    return normalized


def _parse_selected_aspects(form: Any) -> tuple[list[dict[str, Any]], list[str], list[str], str | None]:
    selected_values = _serialize_aspect_values(form.getlist("aspects[]"))
    legacy_custom = str(form.get("custom_aspect", "")).strip()
    raw_custom_values = form.getlist("custom_aspects[]")
    if legacy_custom:
        raw_custom_values.append(legacy_custom)
    custom_aspect_values = normalize_custom_aspect_values(raw_custom_values)
    selected_aspects, aspect_error = build_selected_aspects(selected_values, custom_aspect_values)

    return selected_aspects, selected_values, custom_aspect_values, aspect_error


def _render_index(**kwargs: Any) -> str:
    selected_geography = SerpApiReviewClient.normalize_geography(kwargs.get("selected_geography"))
    brand_inputs = kwargs.get("brand_inputs") or [{"name": ""}]
    selected_aspect_values_raw = kwargs.get("selected_aspect_values")
    if selected_aspect_values_raw is None:
        selected_aspect_values = DEFAULT_SELECTED_ASPECT_VALUES.copy()
    elif isinstance(selected_aspect_values_raw, str):
        selected_aspect_values = _serialize_aspect_values([selected_aspect_values_raw])
    else:
        selected_aspect_values = _serialize_aspect_values(
            [str(value) for value in selected_aspect_values_raw]
        )
    custom_aspect_values = normalize_custom_aspect_values(kwargs.get("custom_aspect_values"))
    selected_aspects_raw = kwargs.get("selected_aspects")
    selected_aspects = (
        selected_aspects_raw if selected_aspects_raw is not None else DEFAULT_SELECTED_ASPECTS
    )

    context = {
        "geography_choices": SerpApiReviewClient.geography_choices(),
        "aspect_choices": aspect_choices(),
        "selected_geography": selected_geography,
        "selected_aspect_values": selected_aspect_values,
        "custom_aspect_values": custom_aspect_values,
        "selected_aspects": selected_aspects,
        "brand_inputs": brand_inputs,
        "preview_rows": kwargs.get("preview_rows", []),
        "analysis_rows": kwargs.get("analysis_rows", []),
        "summary_rows": kwargs.get("summary_rows", []),
        "detailed_rows": kwargs.get("detailed_rows", []),
        "analysis_errors": kwargs.get("analysis_errors", []),
        "show_analysis_panel": kwargs.get("show_analysis_panel", False),
        "auto_scroll_target": kwargs.get("auto_scroll_target", ""),
        "max_reviews_per_brand": MAX_REVIEWS_PER_BRAND,
        "max_total_reviews": MAX_TOTAL_REVIEWS_PER_ANALYSIS,
        "analysis_time_budget_seconds": ANALYSIS_TIME_BUDGET_SECONDS,
        "analysis_job_id": kwargs.get("analysis_job_id"),
        "analysis_job_status": kwargs.get("analysis_job_status", ""),
        "analysis_job_progress": kwargs.get("analysis_job_progress", ""),
        "analysis_job_error": kwargs.get("analysis_job_error", ""),
        "selected_brand_indexes": kwargs.get("selected_brand_indexes", []),
    }
    context.update(kwargs)
    return render_template("index.html", **context)


def _update_analysis_job(job_id: str, **updates: Any) -> None:
    with ANALYSIS_JOB_LOCK:
        job = ANALYSIS_JOB_CACHE.get(job_id)
        if not job:
            return
        job.update(updates)


def _get_analysis_job(job_id: str) -> dict[str, Any] | None:
    with ANALYSIS_JOB_LOCK:
        job = ANALYSIS_JOB_CACHE.get(job_id)
        if not job:
            return None
        return dict(job)


def _check_job_pause_or_cancel(job_id: str | None) -> None:
    if not job_id:
        return

    while True:
        job = _get_analysis_job(job_id)
        if not job:
            raise AnalysisCancelledError("Analysis job no longer exists.")

        if bool(job.get("cancel_requested")):
            raise AnalysisCancelledError("Analysis cancelled by user.")

        if bool(job.get("pause_requested")):
            _update_analysis_job(
                job_id,
                status="paused",
                progress=PAUSED_ANALYSIS_PROGRESS,
            )
            time.sleep(0.5)
            continue

        if str(job.get("status", "")) == "paused":
            _update_analysis_job(
                job_id,
                status="running",
                progress="Resuming analysis...",
            )
        return


def _run_analysis_pipeline(
    preview_brands: list[dict[str, Any]],
    requested_counts: dict[int, int],
    detailed_csv_path: str,
    selected_aspects: list[dict[str, Any]],
    job_id: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int, list[str]]:
    review_client = SerpApiReviewClient(api_key=os.getenv("SERPAPI_KEY"))
    scorer = SentimentScorer()
    aspect_scorer = AspectSentimentScorer(base_scorer=scorer)

    summary_rows: list[dict[str, Any]] = []
    detailed_preview_rows: list[dict[str, Any]] = []
    detailed_row_count = 0
    analysis_errors: list[str] = []
    started_at = time.monotonic()
    total_brands = len(preview_brands)
    os.makedirs(os.path.dirname(detailed_csv_path), exist_ok=True)

    with open(detailed_csv_path, "w", newline="", encoding="utf-8") as detailed_fp:
        detailed_writer = csv.writer(detailed_fp)
        detailed_writer.writerow(
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
            + [
                f"{str(aspect.get('label', 'Aspect')).strip() or 'Aspect'} sentiment (1-10)"
                for aspect in selected_aspects
            ]
        )

        for index, brand in enumerate(preview_brands):
            _check_job_pause_or_cancel(job_id)
            brand_name = str(brand.get("brand_name", "Unknown"))
            if job_id:
                _update_analysis_job(
                    job_id,
                    progress=f"Processing brand {index + 1}/{total_brands}: {brand_name}",
                )

            elapsed_seconds = time.monotonic() - started_at
            if elapsed_seconds >= ANALYSIS_TIME_BUDGET_SECONDS:
                analysis_errors.append(
                    "Analysis reached the server time budget. "
                    "Try smaller review counts or fewer brands per run."
                )
                break

            desired_reviews = requested_counts.get(index, 0)
            brand_description = str(brand.get("business_description", ""))
            candidates = (brand.get("candidates") or [])[:MAX_ANALYSIS_CANDIDATES_PER_BRAND]

            review_count = 0
            promoter_count = 0
            detractor_count = 0
            google_rating_total = 0.0
            google_rating_count = 0
            store_review_counts: dict[str, int] = {}
            aspect_score_totals = {
                str(aspect.get("key", "")): 0.0 for aspect in selected_aspects
            }
            aspect_score_counts = {
                str(aspect.get("key", "")): 0 for aspect in selected_aspects
            }
            batch_reviews: list[dict[str, Any]] = []
            batch_texts: list[str] = []

            def flush_batch() -> None:
                nonlocal review_count
                nonlocal promoter_count
                nonlocal detractor_count
                nonlocal google_rating_total
                nonlocal google_rating_count
                nonlocal detailed_row_count
                nonlocal batch_reviews
                nonlocal batch_texts
                if not batch_reviews:
                    return
                _check_job_pause_or_cancel(job_id)
                sentiment_scores = scorer.score_many_to_ten(
                    batch_texts,
                    batch_size=ANALYSIS_SENTIMENT_BATCH_SIZE,
                )
                _check_job_pause_or_cancel(job_id)
                aspect_scores_list = aspect_scorer.score_aspects_for_many(
                    batch_texts,
                    selected_aspects=selected_aspects,
                    batch_size=ANALYSIS_SENTIMENT_BATCH_SIZE,
                )

                for review, sentiment_score, aspect_scores in zip(
                    batch_reviews,
                    sentiment_scores,
                    aspect_scores_list,
                ):
                    review_count += 1
                    if sentiment_score >= 9.0:
                        promoter_count += 1
                    elif sentiment_score < 7.0:
                        detractor_count += 1

                    rating = review.get("google_rating")
                    if isinstance(rating, (int, float)):
                        google_rating_total += float(rating)
                        google_rating_count += 1

                    store_name = str(review.get("store_name", "Unknown"))
                    store_review_counts[store_name] = store_review_counts.get(store_name, 0) + 1

                    nps_bucket = classify_nps_bucket(sentiment_score)
                    row_values: list[Any] = [
                        brand_name,
                        store_name,
                        review.get("store_id", ""),
                        review.get("text", ""),
                        review.get("review_time", ""),
                        sentiment_score,
                        review.get("google_rating", ""),
                        nps_bucket,
                    ]

                    for aspect in selected_aspects:
                        aspect_key = str(aspect.get("key", ""))
                        aspect_score = (
                            aspect_scores.get(aspect_key)
                            if isinstance(aspect_scores, dict)
                            else None
                        )
                        if isinstance(aspect_score, (int, float)):
                            aspect_score_totals[aspect_key] = (
                                aspect_score_totals.get(aspect_key, 0.0) + float(aspect_score)
                            )
                            aspect_score_counts[aspect_key] = (
                                aspect_score_counts.get(aspect_key, 0) + 1
                            )
                            row_values.append(aspect_score)
                        else:
                            row_values.append("N/A")

                    detailed_writer.writerow(row_values)

                    if len(detailed_preview_rows) < DETAILED_PREVIEW_LIMIT:
                        detailed_preview_rows.append(
                            {
                                "brand_name": brand_name,
                                "store_name": store_name,
                                "store_id": review.get("store_id", ""),
                                "review_text": review.get("text", ""),
                                "review_date": review.get("review_time", ""),
                                "sentiment_score": sentiment_score,
                                "google_rating": review.get("google_rating"),
                                "nps_bucket": nps_bucket,
                                "aspect_scores": {
                                    str(aspect.get("key", "")): (
                                        aspect_scores.get(str(aspect.get("key", "")))
                                        if isinstance(aspect_scores, dict)
                                        else None
                                    )
                                    for aspect in selected_aspects
                                },
                            }
                        )
                    detailed_row_count += 1
                batch_reviews = []
                batch_texts = []

            if desired_reviews > 0:
                try:
                    for review in review_client.iter_reviews_for_candidates(
                        candidates=candidates,
                        max_reviews=desired_reviews,
                    ):
                        _check_job_pause_or_cancel(job_id)
                        batch_reviews.append(review)
                        batch_texts.append(str(review.get("text", "")))
                        if len(batch_reviews) >= ANALYSIS_SENTIMENT_BATCH_SIZE:
                            flush_batch()
                    flush_batch()
                except SerpApiError as exc:
                    analysis_errors.append(f"{brand_name}: {exc}")
                    flush_batch()

            avg_google_score = (
                round(google_rating_total / google_rating_count, 2)
                if google_rating_count
                else None
            )
            nps_score = 0.0
            if review_count:
                promoter_pct = (promoter_count / review_count) * 100.0
                detractor_pct = (detractor_count / review_count) * 100.0
                nps_score = round(promoter_pct - detractor_pct, 2)

            aspect_averages: dict[str, float | None] = {}
            for aspect in selected_aspects:
                aspect_key = str(aspect.get("key", ""))
                count = aspect_score_counts.get(aspect_key, 0)
                if count > 0:
                    aspect_averages[aspect_key] = round(
                        aspect_score_totals.get(aspect_key, 0.0) / count,
                        2,
                    )
                else:
                    aspect_averages[aspect_key] = None

            summary_rows.append(
                {
                    "brand_name": brand_name,
                    "brand_description": brand_description,
                    "review_count": review_count,
                    "avg_google_score": avg_google_score,
                    "nps_score": nps_score,
                    "stores_covered": len(store_review_counts),
                    "aspect_averages": aspect_averages,
                    "aspect_mentions": dict(aspect_score_counts),
                }
            )

    return summary_rows, detailed_preview_rows, detailed_row_count, analysis_errors


def _run_analysis_job(job_id: str) -> None:
    job = _get_analysis_job(job_id)
    if not job:
        return

    if bool(job.get("cancel_requested")) or str(job.get("status", "")) == "cancelled":
        _update_analysis_job(
            job_id,
            status="cancelled",
            finished_at=time.time(),
            progress="Cancelled.",
            info="Analysis was cancelled.",
            error="Analysis cancelled by user.",
        )
        return

    preview_brands = job.get("preview_brands") or []
    requested_counts = job.get("requested_counts") or {}
    selected_aspects = job.get("selected_aspects") or DEFAULT_SELECTED_ASPECTS
    if (
        not isinstance(preview_brands, list)
        or not isinstance(requested_counts, dict)
        or not isinstance(selected_aspects, list)
    ):
        _update_analysis_job(
            job_id,
            status="failed",
            finished_at=time.time(),
            progress="Failed.",
            error="Analysis payload is invalid. Please run Preview again.",
        )
        return

    _update_analysis_job(
        job_id,
        status="running",
        started_at=time.time(),
        progress="Initializing analysis resources...",
    )

    try:
        _check_job_pause_or_cancel(job_id)
        report_id = str(uuid.uuid4())
        summary_csv_path, detailed_csv_path = _report_paths(report_id)

        summary_rows, detailed_preview_rows, detailed_row_count, analysis_errors = _run_analysis_pipeline(
            preview_brands=preview_brands,
            requested_counts=requested_counts,
            detailed_csv_path=detailed_csv_path,
            selected_aspects=selected_aspects,
            job_id=job_id,
        )

        _write_summary_csv(summary_csv_path, summary_rows, selected_aspects)
        _cache_record_by_id(
            REPORT_CACHE,
            report_id,
            {
                "summary_rows": summary_rows,
                "detailed_rows": detailed_preview_rows,
                "detailed_row_count": detailed_row_count,
                "summary_csv_path": summary_csv_path,
                "detailed_csv_path": detailed_csv_path,
                "selected_aspects": selected_aspects,
            },
            lock=REPORT_CACHE_LOCK,
        )
        info = "Analysis complete." if detailed_row_count else "No review rows returned for selected brands."
        if detailed_row_count > DETAILED_PREVIEW_LIMIT:
            info = (
                f"{info} Showing first {DETAILED_PREVIEW_LIMIT} rows in page preview; "
                "download CSV for full dataset."
            )
        _update_analysis_job(
            job_id,
            status="completed",
            finished_at=time.time(),
            progress="Completed.",
            report_id=report_id,
            summary_rows=summary_rows,
            detailed_rows=detailed_preview_rows,
            detailed_row_count=detailed_row_count,
            analysis_errors=analysis_errors,
            info=info,
            error="",
        )
    except AnalysisCancelledError as exc:
        _update_analysis_job(
            job_id,
            status="cancelled",
            finished_at=time.time(),
            progress="Cancelled.",
            info="Analysis was cancelled.",
            error=str(exc),
        )
    except (SerpApiError, RuntimeError) as exc:
        _update_analysis_job(
            job_id,
            status="failed",
            finished_at=time.time(),
            progress="Failed.",
            error=str(exc),
        )
    except Exception as exc:
        _update_analysis_job(
            job_id,
            status="failed",
            finished_at=time.time(),
            progress="Failed.",
            error=f"Unexpected analysis failure: {exc}",
        )


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
                limit=MAX_STORE_CANDIDATES_PER_BRAND,
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
        selected_aspects=DEFAULT_SELECTED_ASPECTS,
        selected_aspect_values=DEFAULT_SELECTED_ASPECT_VALUES,
        custom_aspect_values=[],
        selected_brand_indexes=list(range(len(preview_brands))),
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

    all_preview_brands = preview_record.get("brands", [])
    if not all_preview_brands:
        return _render_index(
            error="No brand preview data found. Please run Preview Places again.",
        )

    selected_geography = preview_record.get(
        "selected_geography", SerpApiReviewClient.DEFAULT_GEOGRAPHY
    )
    selected_source_indexes = _parse_selected_brand_indexes(
        request.form.getlist("selected_brand_indexes[]"),
        len(all_preview_brands),
    )
    if not selected_source_indexes:
        return _render_index(
            error="Select at least one brand in Confirm Brands before running analysis.",
            selected_geography=selected_geography,
            brand_inputs=_brand_inputs_from_preview(all_preview_brands),
            preview_id=preview_id,
            preview_rows=_build_preview_rows(all_preview_brands),
            analysis_rows=_build_analysis_rows(all_preview_brands),
            selected_brand_indexes=list(range(len(all_preview_brands))),
            selected_aspects=DEFAULT_SELECTED_ASPECTS,
            selected_aspect_values=DEFAULT_SELECTED_ASPECT_VALUES,
            custom_aspect_values=[],
            show_analysis_panel=True,
            auto_scroll_target="analysis-section",
        )

    selected_preview_brands = [all_preview_brands[index] for index in selected_source_indexes]
    brand_inputs = _brand_inputs_from_preview(selected_preview_brands)
    preview_rows = _build_preview_rows(all_preview_brands)
    selected_aspects, selected_aspect_values, custom_aspect_values, aspect_error = _parse_selected_aspects(
        request.form
    )
    if aspect_error:
        return _render_index(
            error=aspect_error,
            selected_geography=selected_geography,
            brand_inputs=brand_inputs,
            preview_id=preview_id,
            preview_rows=preview_rows,
            analysis_rows=_build_analysis_rows(
                selected_preview_brands,
                source_indexes=selected_source_indexes,
            ),
            selected_brand_indexes=selected_source_indexes,
            selected_aspects=selected_aspects,
            selected_aspect_values=selected_aspect_values,
            custom_aspect_values=custom_aspect_values,
            show_analysis_panel=True,
            auto_scroll_target="analysis-section",
        )

    requested_counts_by_source: dict[int, int] = {}
    for source_index in selected_source_indexes:
        raw = request.form.get(f"requested_reviews_{source_index}", "")
        requested_counts_by_source[source_index] = _parse_requested_reviews(raw)

    total_requested = sum(requested_counts_by_source.values())
    analysis_rows = _build_analysis_rows(
        selected_preview_brands,
        requested_counts_by_source,
        source_indexes=selected_source_indexes,
    )

    if not any(count > 0 for count in requested_counts_by_source.values()):
        return _render_index(
            error="Enter review counts for at least one brand before running analysis.",
            selected_geography=selected_geography,
            brand_inputs=brand_inputs,
            preview_id=preview_id,
            preview_rows=preview_rows,
            analysis_rows=analysis_rows,
            selected_brand_indexes=selected_source_indexes,
            selected_aspects=selected_aspects,
            selected_aspect_values=selected_aspect_values,
            custom_aspect_values=custom_aspect_values,
            show_analysis_panel=True,
            auto_scroll_target="analysis-section",
        )

    if total_requested > MAX_TOTAL_REVIEWS_PER_ANALYSIS:
        return _render_index(
            error=(
                "Requested review volume is too high for one analysis run. "
                f"Keep total requested reviews at or below {MAX_TOTAL_REVIEWS_PER_ANALYSIS}."
            ),
            selected_geography=selected_geography,
            brand_inputs=brand_inputs,
            preview_id=preview_id,
            preview_rows=preview_rows,
            analysis_rows=analysis_rows,
            selected_brand_indexes=selected_source_indexes,
            selected_aspects=selected_aspects,
            selected_aspect_values=selected_aspect_values,
            custom_aspect_values=custom_aspect_values,
            show_analysis_panel=True,
            auto_scroll_target="analysis-section",
        )

    try:
        # Validate key early; model load and network-heavy work run in the background.
        SerpApiReviewClient(api_key=os.getenv("SERPAPI_KEY"))
    except SerpApiError as exc:
        return _render_index(
            error=str(exc),
            selected_geography=selected_geography,
            brand_inputs=brand_inputs,
            preview_id=preview_id,
            preview_rows=preview_rows,
            analysis_rows=analysis_rows,
            selected_brand_indexes=selected_source_indexes,
            selected_aspects=selected_aspects,
            selected_aspect_values=selected_aspect_values,
            custom_aspect_values=custom_aspect_values,
            show_analysis_panel=True,
            auto_scroll_target="analysis-section",
        )

    analysis_requested_counts = {
        target_index: requested_counts_by_source[source_index]
        for target_index, source_index in enumerate(selected_source_indexes)
    }

    job_record = {
        "status": "queued",
        "progress": "Queued for processing.",
        "error": "",
        "pause_requested": False,
        "cancel_requested": False,
        "created_at": time.time(),
        "started_at": None,
        "finished_at": None,
        "preview_id": preview_id,
        "selected_geography": selected_geography,
        "brand_inputs": brand_inputs,
        "preview_rows": preview_rows,
        "selected_brand_indexes": selected_source_indexes,
        "selected_aspect_values": selected_aspect_values,
        "custom_aspect_values": custom_aspect_values,
        "selected_aspects": selected_aspects,
        "analysis_rows": analysis_rows,
        "preview_brands": selected_preview_brands,
        "requested_counts": analysis_requested_counts,
        "report_id": None,
        "summary_rows": [],
        "detailed_rows": [],
        "detailed_row_count": 0,
        "analysis_errors": [],
        "info": "",
    }

    with ANALYSIS_JOB_LOCK:
        job_id = _cache_record(ANALYSIS_JOB_CACHE, job_record)

    try:
        ANALYSIS_EXECUTOR.submit(_run_analysis_job, job_id)
    except RuntimeError:
        _update_analysis_job(
            job_id,
            status="failed",
            finished_at=time.time(),
            progress="Failed.",
            error="Unable to queue analysis job. Please try again.",
        )
        return _render_index(
            error="Unable to start background analysis. Please retry.",
            selected_geography=selected_geography,
            brand_inputs=brand_inputs,
            preview_id=preview_id,
            preview_rows=preview_rows,
            analysis_rows=analysis_rows,
            selected_brand_indexes=selected_source_indexes,
            selected_aspects=selected_aspects,
            selected_aspect_values=selected_aspect_values,
            custom_aspect_values=custom_aspect_values,
            show_analysis_panel=True,
            auto_scroll_target="analysis-section",
            analysis_job_id=job_id,
            analysis_job_status="failed",
            analysis_job_error="Unable to queue analysis job. Please retry.",
        )

    return _render_index(
        info="Analysis started in the background. This page will refresh automatically when it is done.",
        selected_geography=selected_geography,
        brand_inputs=brand_inputs,
        preview_id=preview_id,
        preview_rows=preview_rows,
        analysis_rows=analysis_rows,
        selected_brand_indexes=selected_source_indexes,
        selected_aspects=selected_aspects,
        selected_aspect_values=selected_aspect_values,
        custom_aspect_values=custom_aspect_values,
        show_analysis_panel=True,
        auto_scroll_target="analysis-section",
        analysis_job_id=job_id,
        analysis_job_status="queued",
        analysis_job_progress="Queued for processing.",
    )


@app.route("/analysis-status/<job_id>", methods=["GET"])
def analysis_status(job_id: str) -> Response:
    job = _get_analysis_job(job_id)
    if not job:
        return jsonify({"status": "missing", "error": "Analysis job not found."}), 404

    status = str(job.get("status", "missing"))
    payload: dict[str, Any] = {
        "job_id": job_id,
        "status": status,
        "progress": str(job.get("progress", "")),
        "error": str(job.get("error", "")),
    }
    if status in TERMINAL_ANALYSIS_STATUSES:
        payload["result_url"] = f"/analysis-result/{job_id}"
    return jsonify(payload)


@app.route("/analysis-control/<job_id>", methods=["POST"])
def analysis_control(job_id: str) -> Response:
    payload = request.get_json(silent=True) or {}
    action = str(payload.get("action") or request.form.get("action", "")).strip().lower()
    if action not in {"pause", "resume", "cancel"}:
        return jsonify({"status": "invalid", "error": "Invalid analysis control action."}), 400

    with ANALYSIS_JOB_LOCK:
        job = ANALYSIS_JOB_CACHE.get(job_id)
        if not job:
            return jsonify({"status": "missing", "error": "Analysis job not found."}), 404

        status = str(job.get("status", "missing"))
        if status in TERMINAL_ANALYSIS_STATUSES:
            response_payload = {
                "job_id": job_id,
                "status": status,
                "progress": str(job.get("progress", "")),
                "error": str(job.get("error", "")),
                "result_url": f"/analysis-result/{job_id}",
            }
            return jsonify(response_payload)

        if action == "pause":
            job["pause_requested"] = True
            job["status"] = "paused"
            job["progress"] = PAUSED_ANALYSIS_PROGRESS
        elif action == "resume":
            job["pause_requested"] = False
            if status == "paused":
                if job.get("started_at"):
                    job["status"] = "running"
                    job["progress"] = "Resuming analysis..."
                else:
                    job["status"] = "queued"
                    job["progress"] = "Queued for processing."
        else:  # action == "cancel"
            job["cancel_requested"] = True
            job["pause_requested"] = False
            if status in {"queued", "paused"}:
                job["status"] = "cancelled"
                job["finished_at"] = time.time()
                job["progress"] = "Cancelled."
                job["info"] = "Analysis was cancelled."
                job["error"] = "Analysis cancelled by user."
            else:
                job["progress"] = CANCELLING_ANALYSIS_PROGRESS

        updated_status = str(job.get("status", "missing"))
        response_payload = {
            "job_id": job_id,
            "status": updated_status,
            "progress": str(job.get("progress", "")),
            "error": str(job.get("error", "")),
        }
        if updated_status in TERMINAL_ANALYSIS_STATUSES:
            response_payload["result_url"] = f"/analysis-result/{job_id}"
        return jsonify(response_payload)


@app.route("/analysis-result/<job_id>", methods=["GET"])
def analysis_result(job_id: str) -> str:
    job = _get_analysis_job(job_id)
    if not job:
        return _render_index(
            error="Analysis job not found or expired. Please run analysis again.",
            show_analysis_panel=True,
            auto_scroll_target="analysis-section",
        )

    selected_geography = str(
        job.get("selected_geography", SerpApiReviewClient.DEFAULT_GEOGRAPHY)
    )
    brand_inputs = job.get("brand_inputs") or [{"name": ""}]
    preview_id = str(job.get("preview_id", ""))
    preview_rows = job.get("preview_rows") or []
    analysis_rows = job.get("analysis_rows") or []
    status = str(job.get("status", "missing"))

    base_context = {
        "selected_geography": selected_geography,
        "brand_inputs": brand_inputs,
        "preview_id": preview_id,
        "preview_rows": preview_rows,
        "analysis_rows": analysis_rows,
        "selected_brand_indexes": job.get("selected_brand_indexes", []),
        "selected_aspect_values": job.get(
            "selected_aspect_values",
            DEFAULT_SELECTED_ASPECT_VALUES,
        ),
        "custom_aspect_values": job.get("custom_aspect_values", []),
        "selected_aspects": job.get("selected_aspects", DEFAULT_SELECTED_ASPECTS),
        "show_analysis_panel": True,
        "auto_scroll_target": "analysis-section",
        "analysis_job_id": job_id,
        "analysis_job_status": status,
        "analysis_job_progress": str(job.get("progress", "")),
        "analysis_job_error": str(job.get("error", "")),
    }

    if status in ACTIVE_ANALYSIS_STATUSES:
        info_message = (
            "Analysis is paused. Resume when you are ready."
            if status == "paused"
            else "Analysis is still running in the background."
        )
        return _render_index(
            info=info_message,
            **base_context,
        )

    if status == "cancelled":
        return _render_index(
            info=str(job.get("info") or "Analysis was cancelled."),
            **base_context,
        )

    if status == "failed":
        return _render_index(
            error=str(job.get("error") or "Analysis failed."),
            analysis_errors=job.get("analysis_errors", []),
            **base_context,
        )

    return _render_index(
        info=str(job.get("info") or "Analysis complete."),
        summary_rows=job.get("summary_rows", []),
        detailed_rows=job.get("detailed_rows", []),
        analysis_errors=job.get("analysis_errors", []),
        report_id=job.get("report_id"),
        **base_context,
    )


@app.route("/download/<report_id>", methods=["GET"])
def download_report(report_id: str) -> Response:
    with REPORT_CACHE_LOCK:
        report = REPORT_CACHE.get(report_id)
    if not report:
        abort(404, description="Report not found or expired.")

    detailed_csv_path = str(report.get("detailed_csv_path") or "")
    if detailed_csv_path and os.path.isfile(detailed_csv_path):
        return send_file(
            detailed_csv_path,
            mimetype="text/csv",
            as_attachment=True,
            download_name="sentiment_detailed_dataset.csv",
            max_age=0,
        )

    # Fallback for in-memory legacy reports.
    selected_aspects = report.get("selected_aspects", [])
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    headers = [
        "brand_name",
        "store_name",
        "store_id",
        "review",
        "date_of_review",
        "sentiment_score_1_to_10",
        "google_rating",
        "nps_bucket",
    ]
    for aspect in selected_aspects if isinstance(selected_aspects, list) else []:
        headers.append(f"{str(aspect.get('label', 'Aspect')).strip() or 'Aspect'} sentiment (1-10)")
    writer.writerow(headers)

    for row in report.get("detailed_rows", []):
        values = [
            row.get("brand_name", ""),
            row.get("store_name", ""),
            row.get("store_id", ""),
            row.get("review_text", ""),
            row.get("review_date", ""),
            row.get("sentiment_score", ""),
            row.get("google_rating", ""),
            row.get("nps_bucket", ""),
        ]
        aspect_scores = row.get("aspect_scores", {})
        for aspect in selected_aspects if isinstance(selected_aspects, list) else []:
            aspect_key = str(aspect.get("key", ""))
            score = aspect_scores.get(aspect_key) if isinstance(aspect_scores, dict) else None
            values.append(score if score is not None else "N/A")
        writer.writerow(values)
    return Response(
        buffer.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": 'attachment; filename="sentiment_detailed_dataset.csv"'},
    )


@app.route("/download-summary/<report_id>", methods=["GET"])
def download_summary(report_id: str) -> Response:
    with REPORT_CACHE_LOCK:
        report = REPORT_CACHE.get(report_id)
    if not report:
        abort(404, description="Report not found or expired.")

    summary_csv_path = str(report.get("summary_csv_path") or "")
    if summary_csv_path and os.path.isfile(summary_csv_path):
        return send_file(
            summary_csv_path,
            mimetype="text/csv",
            as_attachment=True,
            download_name="sentiment_summary_dataset.csv",
            max_age=0,
        )

    # Fallback for in-memory legacy reports.
    selected_aspects = report.get("selected_aspects", [])
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    headers = [
        "brand_name",
        "brand_description",
        "number_of_reviews",
        "average_google_score",
        "calculated_nps",
    ]
    for aspect in selected_aspects if isinstance(selected_aspects, list) else []:
        aspect_label = str(aspect.get("label", "")).strip() or "Aspect"
        headers.append(f"{aspect_label} sentiment (1-10)")
        headers.append(f"{aspect_label} mentions")
    writer.writerow(headers)

    for row in report.get("summary_rows", []):
        values = [
            row.get("brand_name", ""),
            row.get("brand_description", ""),
            row.get("review_count", 0),
            row.get("avg_google_score", ""),
            row.get("nps_score", 0.0),
        ]
        aspect_averages = row.get("aspect_averages", {})
        aspect_mentions = row.get("aspect_mentions", {})
        for aspect in selected_aspects if isinstance(selected_aspects, list) else []:
            aspect_key = str(aspect.get("key", ""))
            avg_score = aspect_averages.get(aspect_key) if isinstance(aspect_averages, dict) else None
            mention_count = aspect_mentions.get(aspect_key) if isinstance(aspect_mentions, dict) else None
            values.append(avg_score if avg_score is not None else "N/A")
            values.append(mention_count if isinstance(mention_count, int) else 0)
        writer.writerow(values)
    return Response(
        buffer.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": 'attachment; filename="sentiment_summary_dataset.csv"'},
    )


if __name__ == "__main__":
    debug_enabled = os.getenv("FLASK_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}
    app.run(debug=debug_enabled)
