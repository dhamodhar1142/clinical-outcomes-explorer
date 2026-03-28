# Backend

FastAPI backend that owns structured parsing, document generation, artifact persistence, planning, and orchestration.

## Run

```powershell
uvicorn backend.app.main:app --reload --app-dir job-agent
```

## Main Endpoints

- `POST /parse-job`
- `POST /tailor-resume`
- `POST /generate-cover-letter`
- `POST /build-apply-plan`
- `POST /run-application`
