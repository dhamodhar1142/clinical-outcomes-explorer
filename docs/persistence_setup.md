# Persistence Setup

The app now supports a lightweight persistence foundation for workspace-scoped state.

## SQLite setup

Set one of these environment variables before starting Streamlit:

- `SMART_DATASET_ANALYZER_SQLITE_PATH`
- `SMART_DATASET_ANALYZER_DB_URL` with a `sqlite:///...` URL

Example:

```powershell
$env:SMART_DATASET_ANALYZER_SQLITE_PATH = ".\\data\\smart_dataset_analyzer.sqlite3"
streamlit run app.py
```

## PostgreSQL setup

Set `SMART_DATASET_ANALYZER_DB_URL` to a PostgreSQL connection string:

```powershell
$env:SMART_DATASET_ANALYZER_DB_URL = "postgresql://demo:secret@localhost:5432/smart_dataset"
streamlit run app.py
```

PostgreSQL support uses the optional `psycopg` or `psycopg2` driver. If the driver is not installed or the connection cannot be opened safely, the app falls back to session-only mode with a clear status note instead of crashing.

## What is persisted

- user identity records
- workspace identity and ownership records
- dataset metadata
- saved snapshots
- workflow packs
- collaboration notes
- beta-interest submissions
- usage analytics / audit-style events
- run history
- generated report metadata

## Fallback behavior

If no supported database setting is provided, the app stays in session-only mode and continues to work exactly as before.

If a PostgreSQL URL is configured but the driver is missing or the connection fails, the app also stays in session-only mode so demo and local workflows still work.

## Storage model

The current SQLite-first persistence layer now separates:

- explicit entity models for users, workspaces, datasets, usage events, and reports
- workspace document state
- dataset metadata
- usage events
- generated report metadata
- user/workspace identity records

This keeps the app compatible with local development while also supporting a production-oriented PostgreSQL-backed implementation.

## Migration / schema foundation

SQLite setup now includes a lightweight schema migration table:

- `schema_migrations`

This foundation currently tracks the active schema version and keeps setup idempotent for local/dev environments. It is intentionally simple, but it creates a clean path for future explicit migrations across both SQLite and PostgreSQL.

## Extension path

The persistence layer is organized around backend and repository helpers so:

- SQLite remains the default local/dev/demo backend
- PostgreSQL can be used for backend-backed persistence
- storage logic stays out of `app.py` and the UI layer
