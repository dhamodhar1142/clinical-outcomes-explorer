# Storage Setup

Smart Dataset Analyzer now includes a lightweight artifact storage abstraction for:

- dataset upload storage
- generated report artifact storage
- session bundle storage

## Local filesystem mode

The default backend is local filesystem storage, which is useful for local development, demos, and pilot environments.

Environment variables:

- `SMART_DATASET_ANALYZER_STORAGE_BACKEND`
- `SMART_DATASET_ANALYZER_STORAGE_ROOT`

Examples:

```powershell
$env:SMART_DATASET_ANALYZER_STORAGE_BACKEND = "local"
$env:SMART_DATASET_ANALYZER_STORAGE_ROOT = "C:\smart-dataset-analyzer-artifacts"
```

Recommended use:

- `local` profile
  - local filesystem storage is preferred
  - session fallback is acceptable for quick demos
- `staging` profile
  - local or shared object storage is acceptable
  - use the readiness checks to verify the storage target is reachable
- `production` profile
  - prefer object storage for shared artifacts
  - avoid relying on session-only artifact handling

If no storage root is configured, the app uses:

```text
data/storage
```

## S3-compatible object storage mode

The storage abstraction now supports an S3-compatible backend for:

- uploaded datasets
- report and export artifacts
- session bundles

Environment variables:

- `SMART_DATASET_ANALYZER_STORAGE_BACKEND`
- `SMART_DATASET_ANALYZER_STORAGE_BUCKET`
- `SMART_DATASET_ANALYZER_STORAGE_PREFIX`
- `SMART_DATASET_ANALYZER_STORAGE_REGION`
- `SMART_DATASET_ANALYZER_STORAGE_ENDPOINT_URL`
- `SMART_DATASET_ANALYZER_STORAGE_ACCESS_KEY`
- `SMART_DATASET_ANALYZER_STORAGE_SECRET_KEY`

Examples:

```powershell
$env:SMART_DATASET_ANALYZER_STORAGE_BACKEND = "s3"
$env:SMART_DATASET_ANALYZER_STORAGE_BUCKET = "smart-dataset-analyzer"
$env:SMART_DATASET_ANALYZER_STORAGE_PREFIX = "pilot-artifacts"
$env:SMART_DATASET_ANALYZER_STORAGE_ENDPOINT_URL = "https://minio.local"
```

Notes:

- `SMART_DATASET_ANALYZER_STORAGE_ENDPOINT_URL` is helpful for MinIO and other S3-compatible systems.
- If access credentials are already available through the environment or host runtime, the explicit access key variables can be left unset.
- If the S3 backend is selected but bucket/config initialization fails, the app falls back safely to session-only artifact mode.
- The deployment readiness checks now report both the selected storage backend and a storage health result so shared environments can validate artifact persistence before pilot use.

## Fallback behavior

If the configured storage backend is missing or not implemented yet, the app falls back safely to session-only artifact mode.

That means:

- uploads still work
- reports still generate
- session bundle export still works

The only difference is that artifacts are not persisted outside the active app session.

## Current backend path

The storage service now supports:

- local filesystem storage
- S3-compatible object storage

Additional backend types can still be added later without changing the app UI or the current storage service calls.
