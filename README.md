# Bank Complaint Analyzer

I built this to speed up the first pass on banking complaints.

You paste in what the customer wrote. The app:

1. Classifies the complaint into a product category  
2. Pulls similar cases that were handled before  
3. Drafts a careful first reply for an agent to review  

Nothing gets sent automatically. The draft is only a starting point.

---

## Stack

- **Backend:** FastAPI  
- **Model:** TF-IDF + Logistic Regression  
- **Search:** PostgreSQL + pgvector when available, local case index otherwise  
- **Drafts:** LLM when an API key is configured, solid template fallback if not  
- **Frontend:** React + TypeScript (Vite)

---

## What’s in the repo

```text
backend/     API, training, model artifacts
frontend/    React UI
data/        Sample complaints used for training/demo
scripts/     Helpers to generate sample data
```

---

## Run it locally

### Backend

```powershell
cd "D:\002sep\projects\Bank Complaint Analyzer"

python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

python scripts/generate_sample_data.py
python -m backend.train

$env:PYTHONPATH = (Get-Location).Path
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

API docs: http://127.0.0.1:8000/docs  
Health: http://127.0.0.1:8000/health

### Frontend

```powershell
cd frontend
npm install
npm run dev
```

UI: http://127.0.0.1:5173

---

## Optional setup

Copy `.env.example` if you want local overrides.

| Variable | What it’s for |
| --- | --- |
| `DATABASE_URL` | Postgres with pgvector (optional) |
| `GROQ_API_KEY` / `OPENAI_API_KEY` | Richer draft wording (optional) |
| `CORS_ORIGINS` | Allowed frontend origins |
| `VITE_API_URL` | Frontend → API base URL in production |

If you skip the database URL, search still works through the local case index.  
If you skip LLM keys, drafts still work with the built-in template.

Docker option:

```powershell
docker compose up --build
```

---

## Main API

`POST /analyze`

```json
{
  "narrative": "Duplicate $250 charge on my credit card ending 4412...",
  "customer_name": "Alex Rivera",
  "account_last4": "4412",
  "channel": "web",
  "top_k": 5
}
```

You get back the predicted product, similar cases, and a draft reply.

Also available:

- `GET /health`
- `GET /metrics`
- `GET /products`

---

## Deploy notes

I usually split it like this:

- **API** → Render / Railway / similar (Docker)  
- **UI** → Vercel (`frontend` folder)

On the API host, set `CORS_ORIGINS` to your Vercel URL.  
On Vercel, set `VITE_API_URL` to the API URL (no trailing slash), then redeploy the UI.

---

## Tests

```powershell
$env:PYTHONPATH = (Get-Location).Path
pytest -q
```

---

## Note

This is a decision-support tool for portfolio/demo use. Always have a person review the draft before any customer communication.

---

## Author

**Sony Prashanthi**  
GitHub: [sonyprashanthi44](https://github.com/sonyprashanthi44)

## License

MIT
