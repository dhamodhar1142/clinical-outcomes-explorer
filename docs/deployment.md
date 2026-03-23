# Deployment Notes

## Recommended runtime

- Python 3.11 or 3.12
- Streamlit app entrypoint: `app.py`

## Local setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Optional extras:

```bash
pip install -r requirements-optional.txt
```

Useful environment settings:

```bash
# Select the deployment profile
SMART_DATASET_ANALYZER_ENV=local

# Document how secrets are supplied in the current environment
SMART_DATASET_ANALYZER_SECRETS_SOURCE=environment

# Local/demo SQLite persistence
SMART_DATASET_ANALYZER_SQLITE_PATH=./data/smart_dataset_analyzer.sqlite3

# Production-style PostgreSQL persistence
SMART_DATASET_ANALYZER_DB_URL=postgresql://user:password@host:5432/smart_dataset

# Local filesystem artifact storage
SMART_DATASET_ANALYZER_STORAGE_BACKEND=local
SMART_DATASET_ANALYZER_STORAGE_ROOT=./data/storage

# Or S3-compatible artifact storage
SMART_DATASET_ANALYZER_STORAGE_BACKEND=s3
SMART_DATASET_ANALYZER_STORAGE_BUCKET=smart-dataset-analyzer
SMART_DATASET_ANALYZER_STORAGE_PREFIX=pilot-artifacts
SMART_DATASET_ANALYZER_STORAGE_ENDPOINT_URL=https://minio.local

# Enable threaded background jobs
SMART_DATASET_ANALYZER_JOB_BACKEND=worker
SMART_DATASET_ANALYZER_JOB_MAX_WORKERS=4

# Or enable an external Redis + RQ worker backend
SMART_DATASET_ANALYZER_JOB_BACKEND=rq
SMART_DATASET_ANALYZER_JOB_QUEUE_URL=redis://localhost:6379/0
SMART_DATASET_ANALYZER_JOB_QUEUE_NAME=smart-dataset-analyzer
SMART_DATASET_ANALYZER_JOB_HEALTHCHECK_TIMEOUT=2.0
```

Recommended profile guidance:

- `local`
  - demo-friendly fallbacks
  - SQLite or session-only persistence is acceptable
  - direct environment variables are acceptable for local secrets
- `staging`
  - use shared persistence whenever possible
  - validate artifact storage reachability before relying on shared exports
  - validate optional integrations intentionally
  - use a documented secrets source
- `production`
  - prefer PostgreSQL-backed persistence
  - prefer object storage for shared artifacts and export bundles
  - set an explicit secrets source
  - avoid relying on session-only state or accidental local-only defaults
  - use external workers only when queue connectivity and worker health are validated

## Launch

```bash
streamlit run app.py
```

The app is designed to start directly with `streamlit run app.py` and degrade safely when optional integrations are unavailable.

## Streamlit Community Cloud

Recommended path:

1. Push the repository to GitHub.
2. Open [https://share.streamlit.io](https://share.streamlit.io).
3. Select the repository and target branch.
4. Set the main file path to `app.py`.
5. Deploy.

Recommended Streamlit Cloud notes:

- keep `requirements.txt` committed at the repo root
- keep the `data/` directory committed so built-in demo datasets remain available
- use the base runtime first; optional integrations are not required for startup
- if you use secrets, add them through the Streamlit Cloud secrets manager rather than hardcoding them

### Updating the deployed app

- push a new commit to the configured branch
- Streamlit Cloud will rebuild and redeploy automatically
- if the app does not pick up the change immediately, use the app management page to reboot the app

### Logs and debugging

- use the Streamlit Cloud app management page to inspect build logs and runtime logs
- if a deployment issue appears only in the cloud environment, compare it against:
  - `python -m compileall app.py src ui tests scripts`
  - `python -c "import app; print('app import ok')"`
  - `.\scripts\run_quick_validation.ps1`

### Demo-ready behavior

- first-time users can use the built-in `Try Demo Dataset` path from the empty state
- uploaded dataset mode remains authoritative until the user explicitly switches source
- large uploaded CSV files still use the same validated hybrid streaming path in cloud environments

## Streamlit deployment checklist

- ensure `app.py` is present at the repository root
- ensure `requirements.txt` is committed
- ensure the `data/` folder is included for built-in demo datasets
- optional packages are not required for app startup
- the app should degrade safely if:
  - `xgboost` is missing
  - `openai` is missing
  - PostgreSQL is not configured
  - the PostgreSQL driver is not installed
  - the background job backend is not configured
- choose one persistence path per environment:
  - SQLite for local/dev/demo
  - PostgreSQL for backend-backed environments
- choose one artifact storage path per environment:
  - local filesystem for local/dev or controlled staging
  - S3-compatible object storage for shared or production-style artifacts
- only enable `SMART_DATASET_ANALYZER_JOB_BACKEND` in environments that should execute background work outside the main request path
- if `SMART_DATASET_ANALYZER_JOB_BACKEND=rq`, also set `SMART_DATASET_ANALYZER_JOB_QUEUE_URL` and validate queue reachability through the deployment readiness checks
- if `SMART_DATASET_ANALYZER_STORAGE_BACKEND=s3`, also set `SMART_DATASET_ANALYZER_STORAGE_BUCKET` and validate storage reachability through the deployment readiness checks
- set `SMART_DATASET_ANALYZER_ENV` explicitly so readiness checks match the intended environment
- set `SMART_DATASET_ANALYZER_SECRETS_SOURCE` to document how credentials are injected for that deployment

## Validation commands

```bash
python -m compileall app.py src tests
python -m unittest tests.test_healthcare_regressions tests.test_modeling_serialization tests.test_temporal_detection tests.test_copilot_readmission_workflows tests.test_remediation_engine tests.test_decision_support tests.test_presentation_support tests.test_portfolio_support -v
```

## CI / Release Safety

Recommended release gate:

- compile validation
- unit test suite
- import sanity check
- browser smoke tests for launch and startup-critical flows

The repository now includes a GitHub Actions workflow that runs:

- `python -m compileall app.py src tests`
- `python -m unittest`
- `.\.venv\Scripts\python.exe -c "import app; print('app import ok')"` equivalent on CI Python
- `python -m unittest tests.browser_smoke -v`

Browser smoke remains lightweight and skip-safe for environment-specific limitations, but it is still treated as part of release screening for pilot-ready builds.

## Deployment behavior notes

- large datasets are sampled safely for profiling and quality review
- synthetic helper fields are disclosed explicitly in the UI and reports
- governance, privacy, and standards outputs are readiness aids, not certification outputs
- PostgreSQL persistence and background workers are optional operational upgrades, not requirements for demo mode
- object storage is optional for demo mode, but recommended for shared artifacts in staging and production-style deployments
