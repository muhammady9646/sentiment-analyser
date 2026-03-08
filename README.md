# Sentiment Analysis Website (SerpAPI Google Maps Reviews + NPS)

This project is a Flask website where a user can:

1. Enter a company name.
2. Select a geography to improve place matching.
3. Preview a few matched full store names plus a Google-sourced business description.
4. Confirm and fetch reviews across those brand stores (not just one store).
5. In Define Paramaters, set review counts per brand in a left-side input table.
6. View generated summary and detailed datasets on the right side.
7. Download summary CSV and detailed CSV (with store source tags).

## Tech Stack

- Python + Flask
- SerpAPI (Google Maps search + reviews scraping endpoint)
- DistilBERT sentiment model (`distilbert-base-uncased-finetuned-sst-2-english`)

## Setup

1. Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create `.env` from `.env.example` and add your API key:

```env
SERPAPI_KEY=your_serpapi_key_here
FLASK_DEBUG=0
SENTIMENT_MODEL_REVISION=714eb0fa89d2f80546fda750413ed43d93601a13
```

4. Ensure your SerpAPI account has quota enabled.

## Run

```bash
flask --app app run --debug
```

Then open `http://127.0.0.1:5000`.

## Long-Running Analysis Notes

- Analysis runs asynchronously in the background and UI polls job status.
- Full detailed datasets are written to on-disk CSV files before download.
- Large pull limits are configurable via env vars:
  - `MAX_REVIEWS_PER_BRAND` (default `1000`)
  - `MAX_TOTAL_REVIEWS_PER_ANALYSIS` (default `1000`)
  - `MAX_ANALYSIS_CANDIDATES_PER_BRAND` (default `3`)
  - `ANALYSIS_TIME_BUDGET_SECONDS` (default `1800`)
  - `ANALYSIS_SENTIMENT_BATCH_SIZE` (default `12`)
  - `SENTIMENT_BATCH_SIZE` (default `12`)
  - `SENTIMENT_DYNAMIC_QUANTIZE` (default `1`)
  - `SENTIMENT_MODEL_REVISION` (default pinned commit hash)
  - `TORCH_NUM_THREADS` (default `1`)

## Test

```bash
python -m unittest discover -s tests
```

## Deploy (Render Recommended)

This repo now includes:

- `render.yaml`
- `Procfile`
- `runtime.txt`
- `Dockerfile`

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/muhammady9646/sentiment-analyser)

Deploy steps:

1. Push this project to GitHub.
2. In Render, create a new Blueprint/Web Service from that repo.
3. Set environment variable `SERPAPI_KEY` in Render.
4. Deploy and wait for the first build to finish.
5. Run one analysis request after deploy to warm the Hugging Face model cache (first run is slower).
6. Add your custom domain in Render settings and point DNS records from your domain registrar.

## Downloads

- Summary CSV: `/download-summary/<report_id>`
- Detailed CSV: `/download/<report_id>`

## How NPS Is Computed Here

Each review is assigned a sentiment score from 1.0 to 10.0:

- DistilBERT predicts `POSITIVE` or `NEGATIVE` with confidence `0.0-1.0`.
- Mapping used:
  - `POSITIVE`: `5.5 + 4.5 * confidence`
  - `NEGATIVE`: `5.5 - 4.5 * confidence`

- `9.0 - 10.0`: Promoter
- `7.0 - 8.99`: Passive
- `< 7.0`: Detractor

NPS formula:

```text
NPS = (% Promoters) - (% Detractors)
```

## Project Structure

```text
.
|-- app.py
|-- docs/
|   |-- README.md
|   |-- architecture.md
|   |-- setup-and-operations.md
|   |-- api-and-data.md
|   |-- limitations-and-roadmap.md
|-- requirements.txt
|-- templates/
|   |-- index.html
|-- static/
|   |-- styles.css
|-- services/
|   |-- serpapi_reviews.py
|   |-- sentiment.py
|-- tests/
    |-- test_serpapi_reviews.py
    |-- test_sentiment.py
```

## Docs Folder

See `docs/README.md` for a map of detailed documentation.
