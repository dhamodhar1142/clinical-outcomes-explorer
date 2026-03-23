from __future__ import annotations

import json
import logging
import os
import traceback
import uuid
from collections import Counter, deque
from datetime import UTC, datetime, timedelta
from importlib import import_module
from pathlib import Path
from typing import Any
from urllib import request as urllib_request

from src.runtime_paths import data_path


LOGGER_NAME = 'smart_dataset_analyzer'
ERROR_HOOK_ENV = 'SMART_DATASET_ANALYZER_ERROR_HOOK'
SENTRY_DSN_ENV = 'SMART_DATASET_ANALYZER_SENTRY_DSN'
DATADOG_API_KEY_ENV = 'SMART_DATASET_ANALYZER_DATADOG_API_KEY'
DATADOG_SITE_ENV = 'SMART_DATASET_ANALYZER_DATADOG_SITE'
ERROR_LOG_DIR_ENV = 'SMART_DATASET_ANALYZER_ERROR_LOG_DIR'
ERROR_LOG_RETENTION_DAYS_ENV = 'SMART_DATASET_ANALYZER_ERROR_RETENTION_DAYS'
RECURRING_ERROR_ALERT_THRESHOLD_ENV = 'SMART_DATASET_ANALYZER_RECURRING_ERROR_THRESHOLD'
DEFAULT_ERROR_LOG_DIR = data_path('error_logs')
DEFAULT_ERROR_RETENTION_DAYS = 30
DEFAULT_RECURRING_ERROR_THRESHOLD = 3
LOG_BUFFER_LIMIT = 100
_DIAGNOSTIC_BUFFER: deque[dict[str, Any]] = deque(maxlen=LOG_BUFFER_LIMIT)


def _normalize_level(level_name: str | None) -> int:
    level = str(level_name or 'INFO').strip().upper()
    return getattr(logging, level, logging.INFO)


def _safe_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_safe_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _safe_value(item) for key, item in value.items()}
    return str(value)


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _error_log_directory() -> Path:
    configured = str(os.getenv(ERROR_LOG_DIR_ENV, '')).strip()
    return Path(configured) if configured else DEFAULT_ERROR_LOG_DIR


def _error_retention_days() -> int:
    configured = str(os.getenv(ERROR_LOG_RETENTION_DAYS_ENV, str(DEFAULT_ERROR_RETENTION_DAYS))).strip()
    try:
        return max(1, int(configured))
    except ValueError:
        return DEFAULT_ERROR_RETENTION_DAYS


def _recurring_error_threshold() -> int:
    configured = str(os.getenv(RECURRING_ERROR_ALERT_THRESHOLD_ENV, str(DEFAULT_RECURRING_ERROR_THRESHOLD))).strip()
    try:
        return max(2, int(configured))
    except ValueError:
        return DEFAULT_RECURRING_ERROR_THRESHOLD


def _current_error_log_path() -> Path:
    return _error_log_directory() / f"errors-{_utcnow().date().isoformat()}.jsonl"


def _iter_error_log_files() -> list[Path]:
    directory = _error_log_directory()
    if not directory.exists():
        return []
    return sorted(directory.glob('errors-*.jsonl'))


def _prune_expired_error_logs() -> None:
    directory = _error_log_directory()
    cutoff = _utcnow() - timedelta(days=_error_retention_days())
    if not directory.exists():
        return
    for path in _iter_error_log_files():
        try:
            modified = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
        except OSError:
            continue
        if modified < cutoff:
            try:
                path.unlink()
            except OSError:
                continue


def _persist_error_entry(payload: dict[str, Any]) -> dict[str, Any]:
    directory = _error_log_directory()
    directory.mkdir(parents=True, exist_ok=True)
    entry = {
        'timestamp': _utcnow().isoformat(),
        **{str(key): _safe_value(value) for key, value in payload.items()},
    }
    with _current_error_log_path().open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(entry, sort_keys=True, ensure_ascii=True) + '\n')
    _prune_expired_error_logs()
    return entry


def _load_persisted_error_entries(limit: int = 200) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    cutoff = _utcnow() - timedelta(days=_error_retention_days())
    for path in reversed(_iter_error_log_files()):
        try:
            with path.open('r', encoding='utf-8') as handle:
                lines = handle.readlines()
        except OSError:
            continue
        for line in reversed(lines):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            timestamp_value = str(entry.get('timestamp', '')).strip()
            try:
                timestamp = datetime.fromisoformat(timestamp_value.replace('Z', '+00:00'))
                if timestamp.tzinfo is None:
                    timestamp = timestamp.replace(tzinfo=UTC)
            except ValueError:
                timestamp = None
            if timestamp is not None and timestamp < cutoff:
                continue
            entries.append(entry)
            if len(entries) >= limit:
                return list(reversed(entries))
    return list(reversed(entries))


def configure_logging(level_name: str | None = None) -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    if logger.handlers:
        if level_name:
            logger.setLevel(_normalize_level(level_name))
        return logger

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(name)s %(message)s'))
    logger.addHandler(handler)
    logger.setLevel(_normalize_level(level_name or os.getenv('SMART_DATASET_ANALYZER_LOG_LEVEL', 'INFO')))
    logger.propagate = False
    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    configure_logging()
    return logging.getLogger(f'{LOGGER_NAME}.{name}') if name else logging.getLogger(LOGGER_NAME)


def ensure_platform_log_context(session_state: dict[str, Any]) -> dict[str, str]:
    session_id = str(session_state.get('platform_session_id', '')).strip()
    if not session_id:
        session_id = f"session-{uuid.uuid4().hex[:12]}"
        session_state['platform_session_id'] = session_id

    request_counter = int(session_state.get('_platform_request_counter', 0)) + 1
    session_state['_platform_request_counter'] = request_counter
    request_id = f'{session_id}-req-{request_counter}'
    session_state['platform_request_id'] = request_id
    return {
        'session_id': session_id,
        'request_id': request_id,
    }


def build_log_context(
    session_state: dict[str, Any] | None = None,
    **extra_fields: Any,
) -> dict[str, Any]:
    session_state = session_state or {}
    identity = session_state.get('workspace_identity') or {}
    context: dict[str, Any] = {
        'session_id': session_state.get('platform_session_id', ''),
        'request_id': session_state.get('platform_request_id', ''),
        'workspace_id': identity.get('workspace_id', ''),
        'workspace_name': identity.get('workspace_name', ''),
        'user_id': identity.get('user_id', ''),
        'auth_mode': identity.get('auth_mode', ''),
        'workspace_role': identity.get('role', ''),
    }
    context.update(extra_fields)
    return context


def _append_diagnostic_entry(entry: dict[str, Any]) -> None:
    _DIAGNOSTIC_BUFFER.append({str(key): _safe_value(value) for key, value in entry.items()})


def _dispatch_error_hook(payload: dict[str, Any]) -> None:
    hook_target = str(os.getenv(ERROR_HOOK_ENV, '')).strip()
    if not hook_target:
        return
    if ':' not in hook_target:
        raise ValueError(f'{ERROR_HOOK_ENV} must use module:function format.')
    module_name, function_name = hook_target.split(':', 1)
    callback = getattr(import_module(module_name), function_name)
    callback(dict(payload))


def _dispatch_sentry(payload: dict[str, Any], error: BaseException) -> None:
    dsn = str(os.getenv(SENTRY_DSN_ENV, '')).strip()
    if not dsn:
        return
    sentry_sdk = import_module('sentry_sdk')
    if getattr(sentry_sdk, 'Hub', None) is not None:
        current_client = getattr(getattr(sentry_sdk.Hub, 'current', None), 'client', None)
    else:
        current_client = None
    if current_client is None and hasattr(sentry_sdk, 'init'):
        sentry_sdk.init(dsn=dsn)
    push_scope = getattr(sentry_sdk, 'push_scope', None)
    if push_scope is None:
        sentry_sdk.capture_exception(error)
        return
    with push_scope() as scope:
        for key, value in payload.items():
            scope.set_extra(str(key), _safe_value(value))
        sentry_sdk.capture_exception(error)


def _dispatch_datadog(payload: dict[str, Any]) -> None:
    api_key = str(os.getenv(DATADOG_API_KEY_ENV, '')).strip()
    if not api_key:
        return
    site = str(os.getenv(DATADOG_SITE_ENV, 'datadoghq.com')).strip() or 'datadoghq.com'
    body = {
        'title': f"Clinverity error: {payload.get('event', 'platform_error')}",
        'text': json.dumps({str(key): _safe_value(value) for key, value in payload.items()}, sort_keys=True),
        'alert_type': 'error',
        'source_type_name': 'smart-dataset-analyzer',
        'tags': [
            f"event:{payload.get('event', 'platform_error')}",
            f"operation:{payload.get('operation_type', payload.get('event', 'unknown'))}",
            f"workspace:{payload.get('workspace_id', 'unknown')}",
        ],
    }
    request = urllib_request.Request(
        url=f'https://api.{site}/api/v1/events',
        data=json.dumps(body).encode('utf-8'),
        headers={
            'Content-Type': 'application/json',
            'DD-API-KEY': api_key,
        },
        method='POST',
    )
    with urllib_request.urlopen(request, timeout=5):
        return


def _dispatch_monitoring(payload: dict[str, Any], error: BaseException) -> list[str]:
    dispatched: list[str] = []
    if str(os.getenv(SENTRY_DSN_ENV, '')).strip():
        _dispatch_sentry(payload, error)
        dispatched.append('Sentry')
    if str(os.getenv(DATADOG_API_KEY_ENV, '')).strip():
        _dispatch_datadog(payload)
        dispatched.append('DataDog')
    return dispatched


def log_platform_event(
    event_name: str,
    *,
    logger_name: str | None = None,
    level: int = logging.INFO,
    **fields: Any,
) -> None:
    logger = get_logger(logger_name)
    payload = {'event': event_name}
    payload.update({str(key): _safe_value(value) for key, value in fields.items()})
    _append_diagnostic_entry(
        {
            'event': event_name,
            'logger_name': logger_name or LOGGER_NAME,
            'level': logging.getLevelName(level),
            **payload,
        }
    )
    logger.log(level, json.dumps(payload, sort_keys=True, ensure_ascii=True))


def log_platform_exception(
    event_name: str,
    error: BaseException,
    *,
    logger_name: str | None = None,
    **fields: Any,
) -> None:
    payload = {
        'error_type': type(error).__name__,
        'error_message': str(error),
        'traceback': ''.join(traceback.format_exception(type(error), error, error.__traceback__)),
        'operation_type': fields.get('operation_type', event_name),
        **fields,
    }
    log_platform_event(
        event_name,
        logger_name=logger_name,
        level=logging.ERROR,
        **payload,
    )
    persisted_entry = _persist_error_entry({'event': event_name, 'logger_name': logger_name or LOGGER_NAME, **payload})
    try:
        _dispatch_error_hook({'event': event_name, 'logger_name': logger_name or LOGGER_NAME, **payload})
    except Exception as hook_error:
        fallback_logger = get_logger('ops')
        fallback_payload = {
            'event': 'error_hook_failed',
            'hook_target': str(os.getenv(ERROR_HOOK_ENV, '')).strip(),
            'error_type': type(hook_error).__name__,
            'error_message': str(hook_error),
        }
        _append_diagnostic_entry({'logger_name': 'ops', 'level': 'ERROR', **fallback_payload})
        fallback_logger.error(json.dumps(fallback_payload, sort_keys=True, ensure_ascii=True))
    try:
        monitoring_targets = _dispatch_monitoring({**persisted_entry, 'logger_name': logger_name or LOGGER_NAME}, error)
        if monitoring_targets:
            log_platform_event(
                'error_report_dispatched',
                logger_name='ops',
                original_event=event_name,
                monitoring_targets=monitoring_targets,
                operation_type=payload.get('operation_type', event_name),
            )
    except Exception as monitoring_error:
        fallback_logger = get_logger('ops')
        fallback_payload = {
            'event': 'monitoring_dispatch_failed',
            'error_type': type(monitoring_error).__name__,
            'error_message': str(monitoring_error),
            'original_event': event_name,
        }
        _append_diagnostic_entry({'logger_name': 'ops', 'level': 'ERROR', **fallback_payload})
        fallback_logger.error(json.dumps(fallback_payload, sort_keys=True, ensure_ascii=True))


def build_error_capture_status() -> dict[str, Any]:
    hook_target = str(os.getenv(ERROR_HOOK_ENV, '')).strip()
    monitoring_targets: list[str] = []
    if hook_target:
        monitoring_targets.append('Generic hook')
    if str(os.getenv(SENTRY_DSN_ENV, '')).strip():
        monitoring_targets.append('Sentry')
    if str(os.getenv(DATADOG_API_KEY_ENV, '')).strip():
        monitoring_targets.append('DataDog')
    if not monitoring_targets:
        return {
            'status': 'Inline only',
            'detail': 'Structured errors are captured in logs, the in-process diagnostics buffer, and the retained local error log only. No external monitoring target is configured.',
        }
    return {
        'status': 'External monitoring configured',
        'detail': 'Errors will attempt to flow through ' + ', '.join(monitoring_targets) + ' in addition to structured logs and retained local error history.',
    }


def _build_error_frequency_table(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counter: Counter[tuple[str, str]] = Counter()
    for entry in entries:
        operation_type = str(entry.get('operation_type', entry.get('event', 'unknown'))).strip() or 'unknown'
        error_type = str(entry.get('error_type', 'UnknownError')).strip() or 'UnknownError'
        counter[(operation_type, error_type)] += 1
    rows = [
        {
            'operation_type': operation_type,
            'error_type': error_type,
            'error_count': count,
        }
        for (operation_type, error_type), count in counter.most_common()
    ]
    return rows


def _build_recurring_error_alerts(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    threshold = _recurring_error_threshold()
    cutoff = _utcnow() - timedelta(hours=24)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for entry in entries:
        timestamp_value = str(entry.get('timestamp', '')).strip()
        try:
            timestamp = datetime.fromisoformat(timestamp_value.replace('Z', '+00:00'))
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=UTC)
        except ValueError:
            continue
        if timestamp < cutoff:
            continue
        key = (
            str(entry.get('operation_type', entry.get('event', 'unknown'))).strip() or 'unknown',
            str(entry.get('error_type', 'UnknownError')).strip() or 'UnknownError',
        )
        grouped.setdefault(key, []).append(entry)
    rows: list[dict[str, Any]] = []
    for (operation_type, error_type), group in grouped.items():
        if len(group) < threshold:
            continue
        latest = group[-1]
        rows.append(
            {
                'operation_type': operation_type,
                'error_type': error_type,
                'error_count_24h': len(group),
                'latest_error_message': str(latest.get('error_message', '')),
                'latest_seen_at': str(latest.get('timestamp', '')),
                'alert_level': 'Recurring',
            }
        )
    rows.sort(key=lambda row: (-int(row.get('error_count_24h', 0)), str(row.get('operation_type', ''))))
    return rows


def build_support_diagnostics(limit: int = 25) -> dict[str, Any]:
    import pandas as pd

    entries = list(_DIAGNOSTIC_BUFFER)[-max(1, int(limit)) :]
    diagnostics_df = pd.DataFrame(entries)
    if diagnostics_df.empty:
        diagnostics_df = pd.DataFrame(columns=['event', 'level', 'logger_name', 'workspace_id', 'request_id'])
    error_count = int((diagnostics_df['level'].astype(str) == 'ERROR').sum()) if 'level' in diagnostics_df.columns else 0
    recent_events = int(len(diagnostics_df))
    persisted_errors = _load_persisted_error_entries(limit=200)
    frequency_rows = _build_error_frequency_table(persisted_errors)
    recurring_rows = _build_recurring_error_alerts(persisted_errors)
    error_capture = build_error_capture_status()
    summary_cards = [
        {'label': 'Recent diagnostic events', 'value': f'{recent_events:,}'},
        {'label': 'Recent errors', 'value': f'{error_count:,}'},
        {'label': 'Retained errors (30d)', 'value': f'{len(persisted_errors):,}'},
        {'label': 'Recurring error alerts', 'value': f'{len(recurring_rows):,}'},
        {'label': 'Error capture', 'value': str(error_capture.get('status', 'Inline only'))},
    ]
    notes = [
        'Support diagnostics are kept in a lightweight in-process buffer for recent operational review and in a retained error log for 30 days of error-history analysis.',
        str(error_capture.get('detail', '')),
        f"Retained error logs are written to '{_error_log_directory()}' and pruned after {_error_retention_days()} day(s).",
    ]
    return {
        'summary_cards': summary_cards,
        'diagnostics_table': diagnostics_df,
        'error_frequency_table': pd.DataFrame(frequency_rows, columns=['operation_type', 'error_type', 'error_count']),
        'recurring_error_table': pd.DataFrame(
            recurring_rows,
            columns=['operation_type', 'error_type', 'error_count_24h', 'latest_error_message', 'latest_seen_at', 'alert_level'],
        ),
        'recent_error_table': pd.DataFrame(persisted_errors),
        'notes': notes,
    }
