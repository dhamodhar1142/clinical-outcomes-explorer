from __future__ import annotations

import hashlib
import json
import math
import os
import threading
import time
import urllib.request
from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.data_loader import DEMO_DATASETS, DataLoadError, load_uploaded_file_bundle
from src.jobs import get_job_result, get_job_status
from src.services.application_service import AnalysisExecutionResult
from src.services.dataset_service import build_demo_dataset_bundle, build_uploaded_dataset_bundle
from src.services.job_service import cancel_background_task, read_background_task, submit_background_task
from src.services.runtime_service import ensure_runtime_services

try:
    from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, Response, UploadFile, status
    from fastapi.responses import JSONResponse
except ImportError as exc:  # pragma: no cover - exercised only when optional dependency is missing.
    FastAPI = None
    _FASTAPI_IMPORT_ERROR = exc
else:
    _FASTAPI_IMPORT_ERROR = None


API_KEYS_ENV = 'SMART_DATASET_ANALYZER_API_KEYS'
API_RATE_LIMIT_ENV = 'SMART_DATASET_ANALYZER_API_RATE_LIMIT_PER_MINUTE'
API_RATE_WINDOW_ENV = 'SMART_DATASET_ANALYZER_API_RATE_LIMIT_WINDOW_SECONDS'
API_DEFAULT_WORKSPACE_ENV = 'SMART_DATASET_ANALYZER_API_DEFAULT_WORKSPACE'


@dataclass(frozen=True)
class APIKeyRecord:
    api_key: str
    label: str
    role: str
    workspace_id: str
    workspace_name: str


def _require_fastapi() -> None:
    if FastAPI is None:
        raise RuntimeError(
            'FastAPI support is optional and is not currently installed. '
            'Install the API dependencies to enable the REST API.'
        ) from _FASTAPI_IMPORT_ERROR


def _slug(value: str) -> str:
    cleaned = ''.join(char.lower() if char.isalnum() else '-' for char in str(value or '').strip())
    while '--' in cleaned:
        cleaned = cleaned.replace('--', '-')
    return cleaned.strip('-') or 'workspace'


def _dataset_version_hash(dataset_name: str, source_meta: dict[str, Any], data: pd.DataFrame) -> str:
    return hashlib.sha1(
        (
            f"{dataset_name}::{source_meta.get('source_mode', 'unknown')}::"
            f"{len(data)}::{len(data.columns)}::{source_meta.get('dataset_cache_key', '')}"
        ).encode('utf-8')
    ).hexdigest()[:16]


def _rate_limit_per_minute() -> int:
    raw = str(os.getenv(API_RATE_LIMIT_ENV, '60')).strip()
    try:
        return max(1, int(raw))
    except ValueError:
        return 60


def _rate_limit_window_seconds() -> int:
    raw = str(os.getenv(API_RATE_WINDOW_ENV, '60')).strip()
    try:
        return max(1, int(raw))
    except ValueError:
        return 60


def _default_workspace_name() -> str:
    return str(os.getenv(API_DEFAULT_WORKSPACE_ENV, 'API Workspace')).strip() or 'API Workspace'


def _parse_api_key_records() -> dict[str, APIKeyRecord]:
    configured = str(os.getenv(API_KEYS_ENV, '')).strip()
    if not configured:
        return {}
    records: dict[str, APIKeyRecord] = {}
    for raw_entry in configured.replace('\r', '\n').splitlines():
        for piece in raw_entry.split(','):
            entry = piece.strip()
            if not entry:
                continue
            parts = [part.strip() for part in entry.split('|')]
            api_key = parts[0]
            label = parts[1] if len(parts) > 1 and parts[1] else 'API Client'
            role = parts[2] if len(parts) > 2 and parts[2] else 'analyst'
            workspace_id = parts[3] if len(parts) > 3 and parts[3] else f'api-{_slug(label)}-workspace'
            workspace_name = parts[4] if len(parts) > 4 and parts[4] else _default_workspace_name()
            records[api_key] = APIKeyRecord(
                api_key=api_key,
                label=label,
                role=role.lower(),
                workspace_id=workspace_id,
                workspace_name=workspace_name,
            )
    return records


def _build_workspace_identity(record: APIKeyRecord) -> dict[str, Any]:
    role = str(record.role or 'analyst').lower()
    user_slug = _slug(record.label)
    return {
        'auth_mode': 'api_key',
        'signed_in': True,
        'display_name': record.label,
        'user_id': f'api::{user_slug}',
        'email': '',
        'provider': 'api-key',
        'provider_subject': f'api-key::{user_slug}',
        'workspace_id': record.workspace_id,
        'workspace_name': record.workspace_name,
        'role': role,
        'role_label': role.title(),
        'membership_validated': True,
        'ownership_validated': role == 'owner',
        'tenant_id': '',
        'owner_user_id': f'api::{user_slug}',
    }


def _new_request_session(app_state: dict[str, Any], workspace_identity: dict[str, Any]) -> dict[str, Any]:
    request_state = {
        'persistence_service': app_state['persistence_service'],
        'auth_service': app_state['auth_service'],
        'storage_service': app_state['storage_service'],
        'job_runtime': app_state['job_runtime'],
        'application_service': app_state['application_service'],
        'admin_ops_service': app_state['admin_ops_service'],
        'analysis_log': [],
        'saved_snapshots': {},
        'workflow_packs': {},
        'workspace_saved_snapshots': {},
        'workspace_workflow_packs': {},
        'collaboration_notes': [],
        'workspace_collaboration_notes': {},
        'beta_interest_submissions': [],
        'workspace_beta_interest_submissions': {},
        'workspace_analysis_logs': {},
        'workspace_run_history': {},
        'analysis_template': 'General Review',
        'report_mode': 'Executive Summary',
        'export_policy_name': 'Internal Review',
        'active_role': 'Analyst',
        'accuracy_benchmark_profile': 'Auto',
        'accuracy_reporting_threshold_profile': 'Role default',
        'accuracy_reporting_min_trust_score': 0.76,
        'accuracy_allow_directional_external_reporting': False,
        'active_plan': 'Pro',
        'plan_enforcement_mode': 'Demo-safe',
        'run_history': [],
        'demo_synthetic_helper_mode': 'Auto',
        'demo_bmi_remediation_mode': 'median',
        'demo_synthetic_cost_mode': 'Auto',
        'demo_synthetic_readmission_mode': 'Auto',
        'demo_executive_summary_verbosity': 'Concise',
        'demo_scenario_simulation_mode': 'Basic',
        'workspace_user_name': workspace_identity.get('display_name', 'API Client'),
        'workspace_user_email': '',
        'workspace_name': workspace_identity.get('workspace_name', _default_workspace_name()),
        'workspace_role': workspace_identity.get('role_label', 'Analyst'),
        'workspace_governance_redaction_level': 'Medium',
        'workspace_governance_export_access': 'Editors and owners',
        'workspace_governance_watermark_sensitive_exports': True,
        'auth_session': {
            'signed_in': True,
            'auth_mode': 'api_key',
            'provider': 'api-key',
            'display_name': workspace_identity.get('display_name', 'API Client'),
            'user_id': workspace_identity.get('user_id', 'api::client'),
            'role': workspace_identity.get('role', 'analyst'),
            'active_workspace_id': workspace_identity.get('workspace_id', ''),
        },
        'workspace_identity': workspace_identity,
        'visited_sections': [],
        'demo_usage_seeded_keys': [],
        'generated_report_outputs': {},
        'profile_cache_metrics': {'hits': 0, 'misses': 0, 'requests': 0, 'saved_seconds': 0.0, 'last_dataset_version_hash': ''},
    }
    app_state['application_service'].hydrate_workspace_state(request_state)
    return request_state


def _to_records(table: Any, limit: int = 25) -> list[dict[str, Any]]:
    if not isinstance(table, pd.DataFrame) or table.empty:
        return []
    safe_table = table.head(limit).copy()
    safe_table = safe_table.where(pd.notnull(safe_table), None)
    return [
        {str(column): value for column, value in row.items()}
        for row in safe_table.to_dict(orient='records')
    ]


def _json_safe(value: Any) -> Any:
    if isinstance(value, pd.DataFrame):
        return _to_records(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items() if not str(key).startswith('_')}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _sanitize_job_status(job_status: dict[str, Any]) -> dict[str, Any]:
    sanitized = {
        key: value
        for key, value in dict(job_status or {}).items()
        if key not in {'result'}
    }
    return _json_safe(sanitized)


def _serialize_analysis_result(result: AnalysisExecutionResult) -> dict[str, Any]:
    payload = {
        'dataset_name': result.dataset_name,
        'source_meta': dict(result.source_meta),
        'preflight': dict(result.preflight),
        'column_validation': dict(result.column_validation),
        'job_runtime': dict(result.job_runtime),
        'large_dataset_profile': dict(result.large_dataset_profile),
        'long_task_notice': result.long_task_notice,
        'empty_column_warning': result.empty_column_warning,
        'other_column_warnings': list(result.other_column_warnings),
        'blocked': bool(result.blocked),
        'demo_config': dict(result.demo_config),
    }
    if result.pipeline is None:
        payload['pipeline'] = None
        return payload
    pipeline = result.pipeline
    payload['pipeline'] = {
        'overview': dict(pipeline.get('overview', {})),
        'quality': {
            **{key: value for key, value in dict(pipeline.get('quality', {})).items() if not isinstance(value, pd.DataFrame)},
            'high_missing': _to_records(dict(pipeline.get('quality', {})).get('high_missing')),
        },
        'readiness': {
            **{key: value for key, value in dict(pipeline.get('readiness', {})).items() if not isinstance(value, pd.DataFrame)},
            'readiness_table': _to_records(dict(pipeline.get('readiness', {})).get('readiness_table')),
        },
        'healthcare': {
            key: value
            for key, value in dict(pipeline.get('healthcare', {})).items()
            if not isinstance(value, pd.DataFrame)
        },
        'insights': {
            'summary_lines': list(dict(pipeline.get('insights', {})).get('summary_lines', [])),
            'recommendations': list(dict(pipeline.get('insights', {})).get('recommendations', [])),
        },
        'field_profile': _to_records(pipeline.get('field_profile')),
        'action_recommendations': _to_records(pipeline.get('action_recommendations')),
        'sample_info': dict(pipeline.get('sample_info', {})),
    }
    if result.demo_usage_seed is not None:
        payload['demo_usage_seed'] = dict(result.demo_usage_seed)
    return _json_safe(payload)


def _build_uploaded_dataset_from_api(
    *,
    app_state: dict[str, Any],
    workspace_identity: dict[str, Any],
    file_name: str,
    file_bytes: bytes,
    sheet_name: str | None = None,
) -> tuple[pd.DataFrame, dict[str, str], dict[str, Any], dict[str, Any] | None]:
    return build_uploaded_dataset_bundle(
        file_name,
        file_bytes,
        sheet_name=sheet_name,
        source_mode='API upload',
        storage_service=app_state['storage_service'],
        persistence_service=app_state['persistence_service'],
        application_service=app_state['application_service'],
        workspace_identity=workspace_identity,
    )


def _load_uploaded_dataset_artifact(
    *,
    app_state: dict[str, Any],
    workspace_identity: dict[str, Any],
    artifact_path: str = '',
    relative_path: str = '',
    dataset_name: str = 'Uploaded dataset',
) -> tuple[pd.DataFrame, dict[str, str], dict[str, Any], dict[str, Any] | None]:
    payload = app_state['storage_service'].load_artifact_bytes(
        artifact_path=artifact_path,
        relative_path=relative_path,
    )
    data, original_lookup, load_stats = app_state['data_loader'](dataset_name, payload)
    source_meta = {
        'source_mode': 'API artifact',
        'description': 'Dataset loaded from a previously uploaded API artifact.',
        'best_for': 'API-driven re-analysis workflows and integration refreshes.',
        'file_size_mb': len(payload) / (1024 ** 2),
        'ingestion_strategy': str(load_stats.get('ingestion_strategy', 'standard')),
        'sampling_mode': str(load_stats.get('sampling_mode', 'full')),
        'source_row_count': int(load_stats.get('source_row_count', len(data))),
        'analyzed_row_count': int(load_stats.get('analyzed_row_count', len(data))),
        'chunk_count': int(load_stats.get('chunk_count', 1)),
        'dataset_cache_key': str(load_stats.get('dataset_cache_key', '')),
    }
    app_state['application_service'].record_dataset_metadata(
        workspace_identity,
        {
            'dataset_name': dataset_name,
            'source_mode': source_meta['source_mode'],
            'row_count': len(data),
            'column_count': len(data.columns),
            'source_columns': [str(column) for column in data.columns],
            'source_dtypes': {
                str(column): str(dtype)
                for column, dtype in data.dtypes.items()
            },
            'file_size_mb': source_meta['file_size_mb'],
            'description': source_meta['description'],
            'best_for': source_meta['best_for'],
            'dataset_version_hash': _dataset_version_hash(dataset_name, source_meta, data),
            'version_label': f'{dataset_name} | api-artifact',
        },
    )
    return data, original_lookup, source_meta, None


def _load_demo_dataset_from_api(
    *,
    app_state: dict[str, Any],
    workspace_identity: dict[str, Any],
    demo_dataset_name: str,
) -> tuple[pd.DataFrame, dict[str, str], dict[str, Any], None]:
    data, original_lookup, source_meta = build_demo_dataset_bundle(
        demo_dataset_name,
        persistence_service=app_state['persistence_service'],
        application_service=app_state['application_service'],
        workspace_identity=workspace_identity,
    )
    return data, original_lookup, source_meta, None


def _dispatch_webhook(url: str, payload: dict[str, Any]) -> None:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode('utf-8'),
        headers={'Content-Type': 'application/json'},
        method='POST',
    )
    with urllib.request.urlopen(request, timeout=10):
        pass


def _watch_job_for_webhook(app_state: dict[str, Any], job_id: str, webhook_url: str) -> None:
    registered = app_state.setdefault('webhook_registrations', {})
    registered[job_id] = {'webhook_url': webhook_url, 'registered_at': time.time()}
    state = read_background_task(app_state, job_id)
    status_name = str(state.get('status', {}).get('status', '')).lower()
    if status_name in {'completed', 'failed', 'cancelled'}:
        payload = {
            'job_id': job_id,
            'status': _sanitize_job_status(state.get('status', {})),
            'result': _json_safe(state.get('result')),
        }
        _dispatch_webhook(webhook_url, payload)
        return

    def _runner() -> None:
        deadline = time.time() + 1800
        while time.time() < deadline:
            state = read_background_task(app_state, job_id)
            status_name = str(state.get('status', {}).get('status', '')).lower()
            if status_name in {'completed', 'failed', 'cancelled'}:
                payload = {
                    'job_id': job_id,
                    'status': _sanitize_job_status(state.get('status', {})),
                    'result': _json_safe(state.get('result')),
                }
                try:
                    _dispatch_webhook(webhook_url, payload)
                finally:
                    registered.pop(job_id, None)
                return
            time.sleep(0.5)
        registered.pop(job_id, None)

    watcher = threading.Thread(target=_runner, daemon=True, name=f'api-webhook-{job_id}')
    watcher.start()


def _build_analysis_runner(
    *,
    app_state: dict[str, Any],
    workspace_identity: dict[str, Any],
    dataset_name: str,
    source_meta: dict[str, Any],
    data: pd.DataFrame,
) -> tuple[dict[str, Any], Any]:
    request_state = _new_request_session(app_state, workspace_identity)

    def _runner() -> dict[str, Any]:
        result = app_state['application_service'].execute_analysis_run(
            request_state,
            data=data,
            dataset_name=dataset_name,
            source_meta=source_meta,
        )
        return _serialize_analysis_result(result)

    return {}, _runner


def create_api_app() -> FastAPI:
    _require_fastapi()
    app = FastAPI(
        title='Smart Dataset Analyzer API',
        version='1.0.0',
        docs_url='/api/docs',
        redoc_url='/api/redoc',
        openapi_url='/api/openapi.json',
        description='Programmatic access to dataset ingestion, analytics runs, and export-ready healthcare intelligence.',
    )

    runtime_state: dict[str, Any] = {}
    services = ensure_runtime_services(runtime_state)
    app_state: dict[str, Any] = {
        **runtime_state,
        'persistence_service': services.persistence_service,
        'auth_service': services.auth_service,
        'storage_service': services.storage_service,
        'job_runtime': services.job_runtime,
        'application_service': services.application_service,
        'admin_ops_service': services.admin_ops_service,
        'api_keys': _parse_api_key_records(),
        'rate_limit_log': {},
        'webhook_registrations': {},
    }

    app_state['data_loader'] = load_uploaded_file_bundle
    app.state.smart_dataset_api = app_state

    async def require_api_client(x_api_key: str | None = Header(default=None)) -> dict[str, Any]:
        api_keys: dict[str, APIKeyRecord] = app.state.smart_dataset_api.get('api_keys', {})
        if not api_keys:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f'API access is not configured. Set {API_KEYS_ENV} to enable API clients.',
            )
        if not x_api_key or x_api_key not in api_keys:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='A valid X-API-Key header is required.')
        record = api_keys[x_api_key]
        return {
            'record': record,
            'workspace_identity': _build_workspace_identity(record),
            'api_key': x_api_key,
        }

    @app.middleware('http')
    async def api_rate_limit(request: Request, call_next):
        if not request.url.path.startswith('/api/v1/'):
            return await call_next(request)
        api_key = request.headers.get('X-API-Key', '')
        api_keys: dict[str, APIKeyRecord] = app.state.smart_dataset_api.get('api_keys', {})
        if not api_key or api_key not in api_keys:
            return JSONResponse(status_code=status.HTTP_401_UNAUTHORIZED, content={'detail': 'A valid X-API-Key header is required.'})
        log_store: dict[str, list[float]] = app.state.smart_dataset_api.setdefault('rate_limit_log', {})
        window_seconds = _rate_limit_window_seconds()
        max_requests = _rate_limit_per_minute()
        now = time.time()
        current = [stamp for stamp in log_store.get(api_key, []) if now - stamp < window_seconds]
        if len(current) >= max_requests:
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    'detail': 'API rate limit exceeded.',
                    'limit': max_requests,
                    'window_seconds': window_seconds,
                },
            )
        current.append(now)
        log_store[api_key] = current
        return await call_next(request)

    @app.get('/api/health')
    def health() -> dict[str, Any]:
        storage_status = getattr(app_state['storage_service'], 'status', None)
        persistence_status = getattr(app_state['persistence_service'], 'status', None)
        return {
            'status': 'ok',
            'api_keys_configured': len(app_state.get('api_keys', {})),
            'job_backend': dict(app_state['job_runtime']),
            'storage_mode': getattr(storage_status, 'mode', 'session'),
            'persistence_mode': getattr(persistence_status, 'mode', 'session'),
        }

    @app.get('/api/v1/datasets')
    def list_datasets(client: dict[str, Any] = Depends(require_api_client)) -> dict[str, Any]:
        identity = client['workspace_identity']
        return {
            'items': app_state['application_service'].list_dataset_metadata(identity),
        }

    @app.post('/api/v1/datasets/upload')
    async def upload_dataset(
        file: UploadFile = File(...),
        sheet_name: str | None = Form(default=None),
        client: dict[str, Any] = Depends(require_api_client),
    ) -> dict[str, Any]:
        identity = client['workspace_identity']
        file_bytes = await file.read()
        try:
            data, _, source_meta, artifact = _build_uploaded_dataset_from_api(
                app_state=app_state,
                workspace_identity=identity,
                file_name=file.filename or 'uploaded.csv',
                file_bytes=file_bytes,
                sheet_name=sheet_name,
            )
        except (PermissionError, DataLoadError) as error:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(error)) from error
        dataset_name = file.filename or 'uploaded.csv'
        version_hash = _dataset_version_hash(dataset_name, source_meta, data)
        return {
            'dataset_name': dataset_name,
            'row_count': int(len(data)),
            'column_count': int(len(data.columns)),
            'source_meta': source_meta,
            'dataset_version_hash': version_hash,
            'artifact': artifact,
        }

    @app.post('/api/v1/analysis/runs')
    def run_analysis(
        payload: dict[str, Any],
        client: dict[str, Any] = Depends(require_api_client),
    ) -> Response:
        identity = client['workspace_identity']
        demo_dataset_name = str(payload.get('demo_dataset_name', '')).strip()
        artifact_path = str(payload.get('artifact_path', '')).strip()
        relative_path = str(payload.get('relative_path', '')).strip()
        dataset_name = str(payload.get('dataset_name', '')).strip() or 'API Dataset'
        webhook_url = str(payload.get('webhook_url', '')).strip()
        if demo_dataset_name:
            if demo_dataset_name not in DEMO_DATASETS:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f'Unknown demo dataset: {demo_dataset_name}')
            data, _, source_meta, _ = _load_demo_dataset_from_api(
                app_state=app_state,
                workspace_identity=identity,
                demo_dataset_name=demo_dataset_name,
            )
            dataset_name = demo_dataset_name
        elif artifact_path or relative_path:
            try:
                data, _, source_meta, _ = _load_uploaded_dataset_artifact(
                    app_state=app_state,
                    workspace_identity=identity,
                    artifact_path=artifact_path,
                    relative_path=relative_path,
                    dataset_name=dataset_name,
                )
            except Exception as error:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f'Could not load uploaded artifact: {error}') from error
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail='Provide either demo_dataset_name or artifact_path/relative_path to start an analysis run.',
            )
        context_fields, runner = _build_analysis_runner(
            app_state=app_state,
            workspace_identity=identity,
            dataset_name=dataset_name,
            source_meta=source_meta,
            data=data,
        )
        submission = submit_background_task(
            app_state,
            job_runtime=app_state['job_runtime'],
            task_key='analysis_pipeline',
            task_label=f'Analysis: {dataset_name}',
            runner=runner,
            detail=f'API analysis requested for {dataset_name}.',
            stage_messages=[
                'Loading dataset for API analysis...',
                'Running analytics pipeline...',
                'API analytics are ready.',
            ],
        )
        if webhook_url:
            _watch_job_for_webhook(app_state, submission['job_id'], webhook_url)
        job_state = read_background_task(app_state, submission['job_id'])
        response_payload = {
            'job_id': submission['job_id'],
            'status': _sanitize_job_status(job_state['status']),
            'result': _json_safe(job_state['result']),
            'webhook_registered': bool(webhook_url),
        }
        if str(job_state['status'].get('status', '')).lower() == 'completed':
            return JSONResponse(status_code=status.HTTP_200_OK, content=response_payload)
        return JSONResponse(status_code=status.HTTP_202_ACCEPTED, content=response_payload)

    @app.post('/api/v1/analysis/runs/upload')
    async def upload_and_analyze(
        file: UploadFile = File(...),
        sheet_name: str | None = Form(default=None),
        webhook_url: str | None = Form(default=None),
        client: dict[str, Any] = Depends(require_api_client),
    ) -> Response:
        identity = client['workspace_identity']
        file_bytes = await file.read()
        try:
            data, _, source_meta, _ = _build_uploaded_dataset_from_api(
                app_state=app_state,
                workspace_identity=identity,
                file_name=file.filename or 'uploaded.csv',
                file_bytes=file_bytes,
                sheet_name=sheet_name,
            )
        except (PermissionError, DataLoadError) as error:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(error)) from error
        dataset_name = file.filename or 'uploaded.csv'
        context_fields, runner = _build_analysis_runner(
            app_state=app_state,
            workspace_identity=identity,
            dataset_name=dataset_name,
            source_meta=source_meta,
            data=data,
        )
        submission = submit_background_task(
            app_state,
            job_runtime=app_state['job_runtime'],
            task_key='analysis_pipeline',
            task_label=f'Analysis: {dataset_name}',
            runner=runner,
            detail=f'API analysis requested for {dataset_name}.',
            stage_messages=[
                'Loading dataset for API analysis...',
                'Running analytics pipeline...',
                'API analytics are ready.',
            ],
        )
        if webhook_url:
            _watch_job_for_webhook(app_state, submission['job_id'], webhook_url)
        job_state = read_background_task(app_state, submission['job_id'])
        response_payload = {
            'job_id': submission['job_id'],
            'status': _sanitize_job_status(job_state['status']),
            'result': _json_safe(job_state['result']),
            'webhook_registered': bool(webhook_url),
        }
        if str(job_state['status'].get('status', '')).lower() == 'completed':
            return JSONResponse(status_code=status.HTTP_200_OK, content=response_payload)
        return JSONResponse(status_code=status.HTTP_202_ACCEPTED, content=response_payload)

    @app.get('/api/v1/analysis/runs/{job_id}')
    def get_analysis_run(job_id: str, client: dict[str, Any] = Depends(require_api_client)) -> dict[str, Any]:
        state = get_job_status(app_state, job_id)
        return {'job_id': job_id, 'status': _sanitize_job_status(state)}

    @app.get('/api/v1/analysis/runs/{job_id}/results')
    def get_analysis_results(job_id: str, client: dict[str, Any] = Depends(require_api_client)) -> dict[str, Any]:
        return {
            'job_id': job_id,
            'status': _sanitize_job_status(get_job_status(app_state, job_id)),
            'result': _json_safe(get_job_result(app_state, job_id)),
        }

    @app.delete('/api/v1/analysis/runs/{job_id}')
    def cancel_analysis(job_id: str, client: dict[str, Any] = Depends(require_api_client)) -> dict[str, Any]:
        return cancel_background_task(app_state, job_id)

    return app


__all__ = ['APIKeyRecord', 'create_api_app']
