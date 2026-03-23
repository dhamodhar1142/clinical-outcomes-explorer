# Auth and Workspace Ownership Foundation

The app now includes a lightweight authentication abstraction with guest-mode fallback.

## Current modes

- Guest mode
  - always available
  - no external auth provider required
  - keeps the app usable for demos
  - defaults to workspace role `viewer`
- Local sign-in mode
  - lightweight demo-safe sign-in
  - uses display name, email, and workspace name
  - supports workspace roles:
    - `owner`
    - `admin`
    - `analyst`
    - `viewer`
  - can persist user and workspace ownership in SQLite when configured

## Optional SQLite setup

Set one of these before starting the app:

```powershell
$env:SMART_DATASET_ANALYZER_SQLITE_PATH = ".\\data\\smart_dataset_analyzer.sqlite3"
```

or

```powershell
$env:SMART_DATASET_ANALYZER_DB_URL = "sqlite:///./data/smart_dataset_analyzer.sqlite3"
```

## What this foundation scopes

- snapshots
- workflow packs
- collaboration notes
- beta-interest submissions
- usage analytics event log
- run history

## Workspace role model

The app now separates:

- audience/report role
- workspace access role

Workspace access roles are lightweight SaaS foundations for ownership and collaboration boundaries:

- `owner`
  - full workspace management and write access
- `admin`
  - workspace administration and write access
- `analyst`
  - workspace write access
- `viewer`
  - read-only foundation

This role model is still local/demo-safe, but it creates a cleaner path for future tenant-aware access control.

## Extension path

The auth logic lives separately from the UI so a hosted provider or multi-tenant backend can be added later without moving authentication code into `app.py`.
