# API And Data

## Web Routes

## `GET /`

- Renders search form.

## `POST /preview`

Form fields:

- `brand_name[]` (required, supports multiple rows)
- `geography` (optional, defaults to global)

Behavior:

- Finds store candidates per brand using SerpAPI `google_maps` engine.
- Builds confirmation rows:
  - brand
  - store name examples
  - Google business description
  - geography
- Creates preview cache record and `preview_id`.

## `POST /analyze`

Form fields:

- `preview_id` (required)
- `requested_reviews_<index>` (optional per brand, clamped to `0-200`)

Behavior:

- Queues analysis as a background job and returns immediately.
- Stores job metadata in memory and starts async processing.

## `GET /analysis-status/<job_id>`

- Returns JSON job status:
  - `queued`
  - `running`
  - `completed`
  - `failed`
- Includes progress text and a result URL when finished.

## `GET /analysis-result/<job_id>`

- Renders the analysis workspace for that job.
- When completed, shows summary + detailed tables.
- When still running, keeps polling status from the frontend.

Failure cases returned on page:

- Missing company name
- Missing `SERPAPI_KEY`
- No matching place
- No reviews returned
- SerpAPI/network failures
- Missing `transformers`/`torch` package

## `GET /download/<report_id>`

- Returns detailed CSV (row-level reviews with store source tags).
- Returns HTTP 404 if report is missing/expired.

## `GET /download-summary/<report_id>`

- Returns summary CSV (brand-level aggregate rows).
- Returns HTTP 404 if report is missing/expired.

## Sentiment Score Mapping

DistilBERT returns:

- `label`: `POSITIVE` or `NEGATIVE`
- `score`: confidence in `0.0-1.0`

Mapped to `1-10` by:

```text
POSITIVE -> 5.5 + (4.5 * confidence)
NEGATIVE -> 5.5 - (4.5 * confidence)
```

Then rounded to 2 decimals.

## NPS Bucket Mapping

- Promoter: sentiment score `>= 9.0`
- Passive: sentiment score `>= 7.0 and < 9.0`
- Detractor: sentiment score `< 7.0`

## NPS Formula

```text
NPS = (% Promoters) - (% Detractors)
```

## CSV Schemas

### Summary CSV (`/download-summary/<report_id>`)

- `brand_name`
- `brand_description`
- `number_of_reviews`
- `average_google_score`
- `calculated_nps`

### Detailed CSV (`/download/<report_id>`)

- `brand_name`
- `store_name`
- `store_id`
- `review`
- `date_of_review`
- `sentiment_score_1_to_10`
- `google_rating`
- `nps_bucket`
