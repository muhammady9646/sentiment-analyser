# Setup And Operations

## Installed Runtime

- Python installed via winget:
  - `Python.Python.3.12`
  - Verified version: `3.12.10`
- Project virtual environment:
  - `C:\Users\PC\AI\Projects\Sentiment Analyser\.venv`

## First-Time Setup

From `C:\Users\PC\AI\Projects\Sentiment Analyser`:

```powershell
& "$env:LocalAppData\Programs\Python\Python312\python.exe" -m venv .venv
& ".\.venv\Scripts\python.exe" -m pip install --upgrade pip
& ".\.venv\Scripts\python.exe" -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Then edit `.env` and set:

```env
SERPAPI_KEY=your_serpapi_key_here
```

## Run

```powershell
& ".\.venv\Scripts\python.exe" -m flask --app app run --debug
```

App URL:

- `http://127.0.0.1:5000`

## Test

```powershell
& ".\.venv\Scripts\python.exe" -m unittest discover -s tests
```

Expected result:

- `Ran 12 tests ... OK`

## Production Deploy

Deployment files included in repo:

- `render.yaml`
- `Procfile`
- `runtime.txt`
- `Dockerfile`

Render flow:

1. Push repo to GitHub.
2. Create Render web service from repo (Blueprint or standard web service).
3. Add env var:
   - `SERPAPI_KEY`
   - `HF_HOME=/tmp/huggingface`
4. Start command:
   - `gunicorn --bind 0.0.0.0:$PORT --workers 1 --timeout 600 app:app`
5. After first deploy, run one analysis request to warm model cache.
6. Connect custom domain in Render and set DNS records at your domain registrar.

## API Key Requirements

Use a valid SerpAPI key with available quota.

If review calls fail, verify:

- API key is valid
- SerpAPI account has remaining credits/quota
- request limits are not exceeded

## Operational Notes

- Windows may keep the Microsoft Store `python.exe` alias active in old terminals.
- If `python` is not found, use explicit executable:
  - `C:\Users\PC\AppData\Local\Programs\Python\Python312\python.exe`
- First DistilBERT use downloads model files from Hugging Face into cache:
  - `C:\Users\PC\.cache\huggingface\hub`
