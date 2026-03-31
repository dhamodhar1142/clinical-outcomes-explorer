# AI Job Application Agent

This monorepo implements a review-first job application assistant based on the guide in `C:\Users\dhamo\Downloads\ai_job_application_agent_guide.pdf`.

## Parts

- `extension/`: Chrome Extension (Manifest V3) for job-page extraction and workflow control.
- `backend/`: FastAPI API for parsing jobs, tailoring resumes, generating cover letters, rendering files, and orchestrating runs.
- `worker/`: Playwright automation worker for generic and portal-specific application flows.
- `data/`: Local artifacts, profiles, accounts, logs, resumes, cover letters, and screenshots.
- `tests/`: Focused regression tests for the new job-agent components.

## Default Mode

The system defaults to `review_first`. It prepares application materials and automation steps, but stops before final submit unless explicitly configured otherwise.

## Quick Start

1. Create a virtual environment and install the backend requirements.
2. Copy `.env.example` to `.env` and fill in the profile/artifact settings.
3. Start the API:

```powershell
uvicorn backend.app.main:app --reload --app-dir job-agent
```

4. Load `job-agent/extension/` as an unpacked Chrome extension.
5. Optionally install Playwright and browsers for worker execution.
6. Run tests:

```powershell
python -m pytest job-agent/tests
```

## Environment

The backend supports a deterministic local mode by default. You can later replace the tailoring and cover-letter generation layers with a server-side LLM provider without changing the API surface.
