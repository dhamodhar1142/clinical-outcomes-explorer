from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path
import os
from typing import Any
from urllib.parse import urlparse

import pandas as pd

from src.jobs import (
    JOB_BACKEND_ENV,
    JOB_HEALTHCHECK_TIMEOUT_ENV,
    JOB_MAX_WORKERS_ENV,
    JOB_QUEUE_NAME_ENV,
    JOB_QUEUE_URL_ENV,
    build_job_backend_health,
    build_job_runtime,
)
from src.persistence import DATABASE_URL_ENV, SQLITE_PATH_ENV
from src.storage import (
    STORAGE_ACCESS_KEY_ENV,
    STORAGE_BACKEND_ENV,
    STORAGE_BUCKET_ENV,
    STORAGE_ENDPOINT_ENV,
    STORAGE_PREFIX_ENV,
    STORAGE_ROOT_ENV,
    STORAGE_SECRET_KEY_ENV,
    build_storage_backend_health,
    build_storage_service,
)

APP_ENV_ENV = 'SMART_DATASET_ANALYZER_ENV'
SECRETS_SOURCE_ENV = 'SMART_DATASET_ANALYZER_SECRETS_SOURCE'


def _exists(path: str) -> bool:
    return Path(path).exists()


def _optional_package_status(package_name: str) -> str:
    return 'Ready' if find_spec(package_name) else 'Optional'


def _configured_app_environment() -> tuple[str, str, str]:
    app_env = str(os.getenv(APP_ENV_ENV, 'local')).strip().lower() or 'local'
    if app_env not in {'local', 'staging', 'production'}:
        return 'Custom', f'{APP_ENV_ENV} is set to {app_env}. The readiness layer will apply generic config checks.', app_env
    detail = (
        f'{APP_ENV_ENV} is set to {app_env}. '
        + (
            'This profile keeps demo-safe defaults and tolerant fallbacks.'
            if app_env == 'local'
            else 'This profile expects shared config, persistent state, and safer operational defaults.'
        )
    )
    return app_env.title(), detail, app_env


def _openai_configuration_status() -> tuple[str, str]:
    package_installed = bool(find_spec('openai'))
    api_key_configured = bool(os.getenv('OPENAI_API_KEY'))
    if package_installed and api_key_configured:
        return 'Configured', 'OpenAI package and OPENAI_API_KEY are both available for enhanced Copilot responses.'
    if package_installed and not api_key_configured:
        return 'Package only', 'The OpenAI package is installed, but OPENAI_API_KEY is not configured. The rule-based Copilot remains available.'
    if not package_installed and api_key_configured:
        return 'Config only', 'OPENAI_API_KEY is configured, but the OpenAI package is not installed. Install requirements-optional.txt to enable enhanced Copilot responses.'
    return 'Not configured', 'OPENAI_API_KEY is only needed for enhanced Copilot responses; the rule-based Copilot remains available without it.'


def _configured_database_mode() -> tuple[str, str]:
    database_url = str(os.getenv(DATABASE_URL_ENV, '')).strip()
    sqlite_path = str(os.getenv(SQLITE_PATH_ENV, '')).strip()
    if database_url.startswith('sqlite:///'):
        return 'SQLite URL', f'{DATABASE_URL_ENV} is configured with a SQLite URL.'
    if database_url:
        parsed = urlparse(database_url)
        if parsed.scheme in {'postgres', 'postgresql'}:
            return 'PostgreSQL URL', f'{DATABASE_URL_ENV} is configured for PostgreSQL.'
        return 'Unsupported URL', f'{DATABASE_URL_ENV} is set, but the scheme is not supported by the current persistence layer.'
    if sqlite_path:
        return 'SQLite path', f'{SQLITE_PATH_ENV} is configured for SQLite-backed persistence.'
    return 'Session only', 'No persistence database is configured; workspace state stays session-scoped.'


def _postgres_configuration_status() -> tuple[str, str]:
    database_url = str(os.getenv(DATABASE_URL_ENV, '')).strip()
    if not database_url:
        return 'Optional', 'PostgreSQL is optional. Configure SMART_DATASET_ANALYZER_DB_URL only when you want backend-backed persistence.'
    parsed = urlparse(database_url)
    if parsed.scheme not in {'postgres', 'postgresql'}:
        return 'Not selected', 'The active persistence configuration is not using a PostgreSQL URL.'
    if find_spec('psycopg') or find_spec('psycopg2'):
        return 'Ready', 'A PostgreSQL URL is configured and a supported PostgreSQL driver is available.'
    return 'Driver missing', 'A PostgreSQL URL is configured, but neither psycopg nor psycopg2 is installed. Install requirements-optional.txt to enable PostgreSQL persistence.'


def _storage_backend_status() -> tuple[str, str]:
    storage_service = build_storage_service()
    mode = str(getattr(storage_service.status, 'mode', 'session'))
    storage_target = str(getattr(storage_service.status, 'storage_target', 'session-only'))
    if mode == 's3':
        return 'Object storage enabled', f'{STORAGE_BACKEND_ENV} is configured for object storage at {storage_target}.'
    if mode == 'local':
        return 'Local storage enabled', f'Artifacts are configured to persist to the local storage root at {storage_target}.'
    configured_backend = str(os.getenv(STORAGE_BACKEND_ENV, '')).strip().lower()
    if configured_backend and configured_backend not in {'local', 'filesystem'}:
        return 'Fallback active', f'{STORAGE_BACKEND_ENV} requested {configured_backend}, but the storage layer stayed in session fallback mode.'
    return 'Session fallback', 'No persistent artifact storage backend is configured; artifacts remain session-scoped unless storage is enabled.'


def _storage_backend_health_status() -> tuple[str, str]:
    health = build_storage_backend_health(build_storage_service())
    return str(health.get('status', 'Unknown')), str(health.get('detail', 'No storage health details are available.'))


def _job_backend_status() -> tuple[str, str]:
    runtime = build_job_runtime()
    mode = str(runtime.get('mode', 'sync')).lower()
    if mode == 'external':
        return 'External worker enabled', (
            f'{JOB_BACKEND_ENV} is set for external worker execution using queue '
            f'"{runtime.get("queue_name", "smart-dataset-analyzer")}".'
        )
    if mode in {'thread', 'worker', 'queue'}:
        max_workers = max(1, int(str(runtime.get('max_workers', 2) or 2)))
        return 'Worker enabled', f'{JOB_BACKEND_ENV} is set to {mode} with {max_workers} configured worker slot(s).'
    configured_backend = str(os.getenv(JOB_BACKEND_ENV, '')).strip().lower()
    if configured_backend:
        return 'Unsupported config', f'{JOB_BACKEND_ENV} is set to an unsupported value. The app will stay in synchronous fallback mode.'
    return 'Sync fallback', 'No worker backend is configured; jobs execute in the current process with managed status updates.'


def _job_backend_health_status() -> tuple[str, str]:
    health = build_job_backend_health(build_job_runtime())
    return str(health.get('status', 'Unknown')), str(health.get('detail', 'No worker health details are available.'))


def _deployment_profile_status(app_env: str) -> tuple[str, str]:
    database_url = str(os.getenv(DATABASE_URL_ENV, '')).strip()
    sqlite_path = str(os.getenv(SQLITE_PATH_ENV, '')).strip()
    if app_env == 'production':
        if database_url.startswith('postgres://') or database_url.startswith('postgresql://'):
            return 'Ready', 'Production profile is paired with PostgreSQL-backed persistence.'
        if sqlite_path or database_url.startswith('sqlite:///'):
            return 'Warning', 'Production profile is active, but persistence is still SQLite-backed. This is acceptable for controlled pilots, not ideal for multi-user SaaS.'
        return 'Warning', 'Production profile is active without a configured persistence backend. The app will fall back to session-only state.'
    if app_env == 'staging':
        if database_url or sqlite_path:
            return 'Ready', 'Staging profile has a configured persistence backend for shared validation.'
        return 'Info', 'Staging profile is active without persisted state. This is usable for UI checks but limits realistic pilot validation.'
    return 'Ready', 'Local profile keeps fallback-friendly defaults for demos and development.'


def _secrets_configuration_status(app_env: str) -> tuple[str, str]:
    secrets_source = str(os.getenv(SECRETS_SOURCE_ENV, '')).strip() or 'environment'
    database_url = str(os.getenv(DATABASE_URL_ENV, '')).strip()
    openai_api_key = bool(os.getenv('OPENAI_API_KEY'))
    issues: list[str] = []
    if app_env in {'staging', 'production'} and not database_url and not str(os.getenv(SQLITE_PATH_ENV, '')).strip():
        issues.append('no persistence credentials are configured')
    if app_env == 'production' and secrets_source == 'environment' and database_url:
        issues.append('database credentials are being provided directly through environment variables')
    if app_env == 'production' and openai_api_key and secrets_source == 'environment':
        issues.append('OPENAI_API_KEY is provided directly through environment variables')
    if issues:
        return 'Review', f"Secrets/config review recommended: {', '.join(issues)}."
    if app_env == 'local':
        return 'Ready', 'Local profile can rely on direct environment variables and optional unset secrets while preserving demo-safe fallbacks.'
    return 'Ready', f'{SECRETS_SOURCE_ENV} is set to {secrets_source}, and no immediate secret-handling risks were detected for the active profile.'


def build_environment_checks() -> dict[str, Any]:
    environment_status, environment_detail, app_env = _configured_app_environment()
    openai_config_status, openai_config_detail = _openai_configuration_status()
    persistence_status, persistence_detail = _configured_database_mode()
    postgres_status, postgres_detail = _postgres_configuration_status()
    storage_backend_status, storage_backend_detail = _storage_backend_status()
    storage_health_status, storage_health_detail = _storage_backend_health_status()
    job_backend_status, job_backend_detail = _job_backend_status()
    job_health_status, job_health_detail = _job_backend_health_status()
    deployment_profile_status, deployment_profile_detail = _deployment_profile_status(app_env)
    secrets_status, secrets_detail = _secrets_configuration_status(app_env)
    checks = [
        {
            'check_area': 'Deployment environment',
            'status': environment_status,
            'detail': environment_detail,
        },
        {
            'check_area': 'Streamlit entrypoint',
            'status': 'Ready' if _exists('app.py') else 'Missing',
            'detail': 'app.py is the expected Streamlit launch target.',
        },
        {
            'check_area': 'Core dependencies manifest',
            'status': 'Ready' if _exists('requirements.txt') else 'Missing',
            'detail': 'requirements.txt is available for deployment packaging.',
        },
        {
            'check_area': 'Optional dependencies manifest',
            'status': 'Ready' if _exists('requirements-optional.txt') else 'Info',
            'detail': 'requirements-optional.txt documents integrations that are not required for the base product runtime.',
        },
        {
            'check_area': 'Development and test manifest',
            'status': 'Ready' if _exists('requirements-dev.txt') else 'Info',
            'detail': 'requirements-dev.txt supports fuller local validation without expanding the production runtime by default.',
        },
        {
            'check_area': 'Demo datasets',
            'status': 'Ready' if _exists('data') else 'Warning',
            'detail': 'Built-in demo datasets support walkthroughs and smoke tests.',
        },
        {
            'check_area': 'Docker support',
            'status': 'Ready' if _exists('Dockerfile') else 'Info',
            'detail': 'Dockerfile is available for containerized demos or internal deployment previews.',
        },
        {
            'check_area': 'Persistence configuration',
            'status': persistence_status,
            'detail': persistence_detail,
        },
        {
            'check_area': 'Deployment profile fit',
            'status': deployment_profile_status,
            'detail': deployment_profile_detail,
        },
        {
            'check_area': 'PostgreSQL persistence support',
            'status': postgres_status,
            'detail': postgres_detail,
        },
        {
            'check_area': 'Artifact storage backend',
            'status': storage_backend_status,
            'detail': storage_backend_detail,
        },
        {
            'check_area': 'Artifact storage health',
            'status': storage_health_status,
            'detail': storage_health_detail,
        },
        {
            'check_area': 'Background job backend',
            'status': job_backend_status,
            'detail': job_backend_detail,
        },
        {
            'check_area': 'Background job health',
            'status': job_health_status,
            'detail': job_health_detail,
        },
        {
            'check_area': 'Secrets and config handling',
            'status': secrets_status,
            'detail': secrets_detail,
        },
        {
            'check_area': 'Optional XGBoost support',
            'status': _optional_package_status('xgboost'),
            'detail': 'XGBoost is optional; the platform will fall back to the supported built-in models when it is unavailable.',
        },
        {
            'check_area': 'Optional OpenAI support',
            'status': _optional_package_status('openai'),
            'detail': 'OpenAI-backed polish is optional; the platform stays functional without it.',
        },
        {
            'check_area': 'Optional OpenAI configuration',
            'status': openai_config_status,
            'detail': openai_config_detail,
        },
        {
            'check_area': 'Optional browser smoke tooling',
            'status': _optional_package_status('playwright'),
            'detail': 'Playwright is optional; it is only needed for browser smoke tests and demo-asset workflows.',
        },
    ]

    notes = [
        'This readiness layer focuses on packaging and launch safety, not infrastructure certification.',
        'Optional dependencies are handled with graceful fallbacks so demos and core analytics can still run.',
        'Base production installs should use requirements.txt; add requirements-optional.txt only when those integrations are intentionally enabled.',
        f'Use {APP_ENV_ENV} to distinguish local, staging, and production startup expectations without changing app code.',
        f'Use {SECRETS_SOURCE_ENV} to document whether secrets come from direct environment variables, a secret manager, or another runtime injection path.',
        f'Use {DATABASE_URL_ENV} or {SQLITE_PATH_ENV} to choose the persistence backend without changing app code.',
        f'Use {STORAGE_BACKEND_ENV}, {STORAGE_ROOT_ENV}, {STORAGE_BUCKET_ENV}, and {STORAGE_PREFIX_ENV} to control persistent artifact storage without changing app code.',
        f'Use {STORAGE_ENDPOINT_ENV} for S3-compatible services such as MinIO, and keep {STORAGE_ACCESS_KEY_ENV} / {STORAGE_SECRET_KEY_ENV} optional when the runtime already injects credentials.',
        f'Use {JOB_BACKEND_ENV} to enable threaded or external worker execution; leave it unset to preserve synchronous demo mode.',
        f'Use {JOB_QUEUE_URL_ENV} and {JOB_QUEUE_NAME_ENV} when deploying the external RQ worker backend, and {JOB_HEALTHCHECK_TIMEOUT_ENV} to tune worker health checks.',
    ]
    return {
        'checks_table': pd.DataFrame(checks),
        'notes': notes,
    }


def build_startup_readiness_summary(
    preflight: dict[str, Any],
    column_validation: dict[str, Any],
    source_meta: dict[str, Any],
    sample_info: dict[str, Any],
) -> dict[str, Any]:
    blocked = bool(preflight.get('blocked'))
    warning_count = len(preflight.get('warnings', [])) + len(column_validation.get('warnings', []))
    cards = [
        {'label': 'Startup Status', 'value': 'Blocked' if blocked else 'Ready'},
        {'label': 'Warnings', 'value': f'{warning_count:,}'},
        {'label': 'Source Mode', 'value': str(source_meta.get('source_mode', 'Unknown'))},
        {'label': 'Sampling Active', 'value': 'Yes' if sample_info.get('sampling_applied') else 'No'},
    ]

    notes = []
    if blocked:
        notes.append(str(preflight.get('block_reason', 'A startup safety guard is currently blocking analysis.')))
    else:
        notes.append('The current startup context is safe for interactive analysis and demo walkthroughs.')
    if source_meta.get('file_size_mb'):
        notes.append(f"Input file size: {float(source_meta.get('file_size_mb', 0.0)):.1f} MB.")
    if sample_info.get('sampling_applied'):
        notes.append(
            f"Sampling is active for responsiveness: {int(sample_info.get('profile_sample_rows', 0)):,} profiling rows and "
            f"{int(sample_info.get('quality_sample_rows', 0)):,} quality-review rows."
        )
    return {
        'summary_cards': cards,
        'notes': notes,
    }


def build_launch_checklist() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                'launch_step': 'Select deployment environment profile',
                'status': 'Required',
                'guidance': 'Set SMART_DATASET_ANALYZER_ENV to local, staging, or production so readiness checks and operational expectations match the target environment.',
            },
            {
                'launch_step': 'Verify startup command',
                'status': 'Required',
                'guidance': 'Use `streamlit run app.py` locally and keep app.py as the root entrypoint for Streamlit-style hosting.',
            },
            {
                'launch_step': 'Confirm base dependencies',
                'status': 'Required',
                'guidance': 'Keep requirements.txt committed for production installs, and use requirements-optional.txt only when those features are needed.',
            },
            {
                'launch_step': 'Confirm optional package intent',
                'status': 'Recommended',
                'guidance': 'Install optional packages only when you want OpenAI, XGBoost, or Playwright-backed workflows in that environment.',
            },
            {
                'launch_step': 'Prepare local validation environment',
                'status': 'Recommended',
                'guidance': 'Use requirements-dev.txt for fuller local validation, demos, and optional feature checks without changing the base deployment path.',
            },
            {
                'launch_step': 'Review demo assets',
                'status': 'Recommended',
                'guidance': 'Confirm data/ and docs/screenshots/ are present if you plan to use built-in demos or recruiter-facing previews.',
            },
            {
                'launch_step': 'Check secrets/config assumptions',
                'status': 'Recommended',
                'guidance': 'Document whether secrets come from direct environment variables or a secret manager by setting SMART_DATASET_ANALYZER_SECRETS_SOURCE, and keep optional integrations non-blocking for startup.',
            },
            {
                'launch_step': 'Select persistence backend',
                'status': 'Recommended',
                'guidance': 'Use SMART_DATASET_ANALYZER_SQLITE_PATH for local/demo persistence or SMART_DATASET_ANALYZER_DB_URL with a PostgreSQL URL for backend-backed environments.',
            },
            {
                'launch_step': 'Select artifact storage backend',
                'status': 'Recommended',
                'guidance': 'Use SMART_DATASET_ANALYZER_STORAGE_BACKEND with local filesystem storage for local/staging or S3-compatible storage for shared production-style artifacts.',
            },
            {
                'launch_step': 'Confirm worker backend intent',
                'status': 'Recommended',
                'guidance': 'Set SMART_DATASET_ANALYZER_JOB_BACKEND only in environments that should run background jobs outside the main Streamlit request path, and add queue settings when the external RQ backend is selected.',
            },
            {
                'launch_step': 'Validate artifact storage health',
                'status': 'Recommended',
                'guidance': 'If using object storage, configure SMART_DATASET_ANALYZER_STORAGE_BUCKET and confirm the readiness check can reach the storage backend before sharing the environment.',
            },
            {
                'launch_step': 'Validate worker health path',
                'status': 'Recommended',
                'guidance': 'If using the external worker backend, configure SMART_DATASET_ANALYZER_JOB_QUEUE_URL and confirm the readiness check can reach the queue before shipping a build.',
            },
            {
                'launch_step': 'Run validation commands',
                'status': 'Required',
                'guidance': 'Run compileall, unittest, and a Streamlit smoke render before sharing the build.',
            },
            {
                'launch_step': 'Require CI validation',
                'status': 'Recommended',
                'guidance': 'Use a CI pipeline that runs compileall, unittest, import sanity, and browser smoke coverage before a release or shared demo build is promoted.',
            },
            {
                'launch_step': 'Gate releases on browser smoke',
                'status': 'Recommended',
                'guidance': 'Keep browser smoke tests as a release gate for startup-critical flows such as launch, demo dataset selection, uploads, Copilot, and export rendering.',
            },
        ]
    )


def build_config_guidance() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                'config_area': 'Deployment environment',
                'guidance': 'Set SMART_DATASET_ANALYZER_ENV to local, staging, or production so startup checks and config expectations match the target environment.',
            },
            {
                'config_area': 'Python runtime',
                'guidance': 'Python 3.11 or 3.12 is the recommended local runtime for this project.',
            },
            {
                'config_area': 'Dependency tiers',
                'guidance': 'Install requirements.txt for the core app, requirements-optional.txt for extra integrations, and requirements-dev.txt for local development or broader validation.',
            },
            {
                'config_area': 'Optional package intent',
                'guidance': 'Only install OpenAI, XGBoost, or Playwright when that environment needs enhanced Copilot responses, model comparison add-ons, or browser smoke tooling.',
            },
            {
                'config_area': 'Optional AI integration',
                'guidance': 'If OpenAI-backed features are enabled later, provide OPENAI_API_KEY through environment variables rather than hardcoding secrets.',
            },
            {
                'config_area': 'Secrets handling',
                'guidance': 'Use SMART_DATASET_ANALYZER_SECRETS_SOURCE to document whether runtime secrets come from plain environment variables, a secret manager, or deployment-time injection.',
            },
            {
                'config_area': 'Persistence backend',
                'guidance': 'Use SMART_DATASET_ANALYZER_SQLITE_PATH for local SQLite persistence or SMART_DATASET_ANALYZER_DB_URL with a PostgreSQL URL for backend-backed deployments.',
            },
            {
                'config_area': 'Artifact storage',
                'guidance': 'Use SMART_DATASET_ANALYZER_STORAGE_BACKEND=local for filesystem storage or SMART_DATASET_ANALYZER_STORAGE_BACKEND=s3 with SMART_DATASET_ANALYZER_STORAGE_BUCKET and SMART_DATASET_ANALYZER_STORAGE_PREFIX for object storage.',
            },
            {
                'config_area': 'Object storage credentials',
                'guidance': 'Use SMART_DATASET_ANALYZER_STORAGE_ENDPOINT_URL for S3-compatible services and provide SMART_DATASET_ANALYZER_STORAGE_ACCESS_KEY / SMART_DATASET_ANALYZER_STORAGE_SECRET_KEY only when the runtime is not already injecting credentials.',
            },
            {
                'config_area': 'Background jobs',
                'guidance': 'Use SMART_DATASET_ANALYZER_JOB_BACKEND for sync, threaded, or external worker modes; use SMART_DATASET_ANALYZER_JOB_MAX_WORKERS for threaded sizing and SMART_DATASET_ANALYZER_JOB_QUEUE_URL / SMART_DATASET_ANALYZER_JOB_QUEUE_NAME for external RQ deployments.',
            },
            {
                'config_area': 'Worker health checks',
                'guidance': 'Use SMART_DATASET_ANALYZER_JOB_HEALTHCHECK_TIMEOUT to tune external worker health checks when queue connections are slow or remote.',
            },
            {
                'config_area': 'CI / release safety',
                'guidance': 'Run compileall, unittest, import sanity, and browser smoke tests in CI so release candidates are screened before deployment or pilot handoff.',
            },
            {
                'config_area': 'Large datasets',
                'guidance': 'Large extracts are handled with sampling safeguards; keep demo uploads narrower when responsiveness matters.',
            },
            {
                'config_area': 'Container support',
                'guidance': 'Docker support is available for packaging, but the default local path remains the Streamlit entrypoint.',
            },
        ]
    )
