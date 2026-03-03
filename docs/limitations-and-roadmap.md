# Limitations And Roadmap

## Current Limitations

1. Google review volume:
   - Coverage depends on SerpAPI pagination and available upstream review data.
2. Ephemeral report storage:
   - Reports are cached in process memory and lost on restart.
3. Single-model sentiment:
   - DistilBERT is stronger than lexicon methods, but can still miss domain nuance/sarcasm.
4. API quota and billing:
   - Heavy usage depends on SerpAPI plan limits and paid billing.
5. Description coverage variance:
   - Some places have rich Google descriptions, others only sparse snippets.
6. No auth/rate limiting:
   - Current app is suitable for local/internal use, not open public traffic.

## Recommended Next Steps

1. Increase review coverage:
   - Add background jobs for deeper pagination and large review pulls.
2. Persist datasets:
   - Save report rows to SQLite/PostgreSQL and serve downloads from DB.
3. Improve scoring quality:
   - Add model benchmarking and domain-specific fine-tuning for better calibration.
4. Add charts:
   - Show sentiment distribution histogram and trend by review time.
5. Production hardening:
   - Add structured logging, error monitoring, auth, and rate limiting.
6. Export options:
   - Support JSON/XLSX and background jobs for larger reports.
