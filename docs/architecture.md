# Architecture

## Goal

Given a company name, first preview geography-biased full store-name examples and a Google-sourced business description, then fetch Google reviews across multiple matched stores through SerpAPI, score each review from 1-10, compute NPS, and allow CSV export.

## Components

- `app.py`
  - Flask web server
  - HTTP routes for preview, async analysis, polling, and download
  - In-memory preview/report/job metadata caches
- `services/serpapi_reviews.py`
  - SerpAPI integration
  - Place discovery via `google_maps` engine
  - Business description retrieval via Google Search result metadata
  - Review retrieval via `google_maps_reviews` engine (supports pagination)
- `services/sentiment.py`
  - DistilBERT sentiment scoring
  - Conversion from DistilBERT label/confidence to score `[1, 10]`
  - NPS bucket classification and aggregate NPS computation
- `templates/index.html`
  - Header + tab navigation
  - Page 1 brand input and confirmation
  - Page 2 split workspace (inputs on left, data preview on right)
- `static/styles.css`
  - Styling for responsive UI

## Runtime Flow

1. User submits one or more brand rows and geography.
2. `/preview` route:
   - Resolves multiple store candidates via SerpAPI
   - Pulls Google-sourced business description text for verification
3. User confirms the brand scope and transitions to page 2.
4. User enters desired review counts per brand.
5. `/analyze` route queues a background analysis job and returns immediately.
6. Frontend polls `/analysis-status/<job_id>` until complete.
7. Worker fetches review pages across matched stores and computes sentiment in batches.
8. Each scored row is streamed to on-disk CSV files for large-pull safety.
9. Each review text is scored with DistilBERT and mapped to `1.0-10.0`.
10. Scores are bucketed as:
   - Promoter: `>= 9.0`
   - Passive: `>= 7.0 and < 9.0`
   - Detractor: `< 7.0`
11. NPS is computed per brand.
12. UI redirects to `/analysis-result/<job_id>` and renders summary plus detailed preview rows.
13. CSV downloads are generated from report files:
   - `/download-summary/<report_id>` for summary rows
   - `/download/<report_id>` for detailed rows with store tags

## State and Storage

- Current implementation uses process memory for `PREVIEW_CACHE`, `REPORT_CACHE`, and `ANALYSIS_JOB_CACHE`.
- Full CSV outputs are persisted to `data/reports/` on local disk.
- In-memory caches are still ephemeral and reset on server restart.
- Max in-memory items retained per cache: `100`.
