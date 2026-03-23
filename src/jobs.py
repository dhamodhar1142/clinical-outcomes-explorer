from __future__ import annotations

import concurrent.futures
import os
import threading
import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from importlib import import_module
from typing import Any

import pandas as pd

import src.logger as logger_module


JOB_BACKEND_ENV = 'SMART_DATASET_ANALYZER_JOB_BACKEND'
JOB_MAX_WORKERS_ENV = 'SMART_DATASET_ANALYZER_JOB_MAX_WORKERS'
JOB_QUEUE_URL_ENV = 'SMART_DATASET_ANALYZER_JOB_QUEUE_URL'
JOB_QUEUE_NAME_ENV = 'SMART_DATASET_ANALYZER_JOB_QUEUE_NAME'
JOB_HEALTHCHECK_TIMEOUT_ENV = 'SMART_DATASET_ANALYZER_JOB_HEALTHCHECK_TIMEOUT'

HEAVY_TASK_DEFINITIONS = [
    {
        'task_key': 'analysis_pipeline',
        'task_label': 'Large dataset profiling and analysis pipeline',
        'area': 'Pipeline',
        'why_heavy': 'Runs structure detection, profiling, readiness scoring, healthcare analytics, standards checks, and insight generation.',
        'current_mode': 'Synchronous with staged progress',
    },
    {
        'task_key': 'predictive_modeling',
        'task_label': 'Predictive modeling and model comparison',
        'area': 'Healthcare Analytics',
        'why_heavy': 'Builds train/test splits, model pipelines, feature transforms, comparison summaries, and explainability outputs.',
        'current_mode': 'Synchronous with managed job status',
    },
    {
        'task_key': 'report_generation',
        'task_label': 'Structured report generation',
        'area': 'Export Center',
        'why_heavy': 'Builds stakeholder-ready report text from multiple analytics layers and export-ready summaries.',
        'current_mode': 'Synchronous with managed job status',
    },
    {
        'task_key': 'export_bundle',
        'task_label': 'Large export bundle preparation',
        'area': 'Export Center',
        'why_heavy': 'Packages report text, compliance payloads, bundle manifests, and support tables for download.',
        'current_mode': 'Synchronous fallback only',
    },
]

REGISTERED_JOB_TASKS: dict[str, Callable[[dict[str, Any]], Any]] = {}
_GLOBAL_JOB_EXECUTORS: dict[str, concurrent.futures.ThreadPoolExecutor] = {}
_GLOBAL_JOB_FUTURES: dict[str, dict[str, concurrent.futures.Future[Any]]] = {}
_GLOBAL_CANCEL_FLAGS: dict[str, dict[str, bool]] = {}


def register_job_task(task_name: str, runner: Callable[[dict[str, Any]], Any]) -> None:
    REGISTERED_JOB_TASKS[str(task_name)] = runner


def execute_registered_job_task(task_name: str, task_kwargs: dict[str, Any] | None = None) -> Any:
    normalized_name = str(task_name or '').strip()
    if not normalized_name or normalized_name not in REGISTERED_JOB_TASKS:
        raise ValueError(f"Unknown registered job task: {normalized_name or 'empty task name'}")
    return REGISTERED_JOB_TASKS[normalized_name](dict(task_kwargs or {}))


def _build_generated_report_text_task(task_kwargs: dict[str, Any]) -> Any:
    from src.export_utils import build_generated_report_text

    return build_generated_report_text(
        str(task_kwargs.get('report_label', 'Generated Report')),
        str(task_kwargs.get('dataset_name', 'Current dataset')),
        task_kwargs.get('overview', {}),
        task_kwargs.get('quality', {}),
        task_kwargs.get('readiness', {}),
        task_kwargs.get('healthcare', {}),
        task_kwargs.get('insights', {}),
        task_kwargs.get('action_recommendations'),
    )


register_job_task('report_generation_text', _build_generated_report_text_task)


class ExternalRQJobAdapter:
    def __init__(self, queue_url: str, queue_name: str) -> None:
        redis_module = import_module('redis')
        rq_module = import_module('rq')
        self.queue_url = queue_url
        self.queue_name = queue_name
        self._redis_connection = redis_module.Redis.from_url(queue_url)
        self._queue_cls = rq_module.Queue
        self._job_cls = rq_module.job.Job

    def enqueue_registered_task(self, task_name: str, task_kwargs: dict[str, Any]) -> dict[str, Any]:
        queue = self._queue_cls(self.queue_name, connection=self._redis_connection)
        queued_job = queue.enqueue(
            execute_registered_job_task,
            task_name,
            task_kwargs,
        )
        return {
            'external_job_id': str(getattr(queued_job, 'id', '')),
            'queue_name': self.queue_name,
            'queue_url': self.queue_url,
        }

    def fetch(self, external_job_id: str) -> Any:
        return self._job_cls.fetch(external_job_id, connection=self._redis_connection)

    def cancel(self, external_job_id: str) -> bool:
        job = self.fetch(external_job_id)
        if hasattr(job, 'cancel'):
            job.cancel()
            return True
        return False


def _get_external_rq_adapter(queue_url: str, queue_name: str) -> ExternalRQJobAdapter:
    return ExternalRQJobAdapter(queue_url, queue_name)


def _healthcheck_timeout_seconds() -> float:
    configured = str(os.getenv(JOB_HEALTHCHECK_TIMEOUT_ENV, '2.0')).strip() or '2.0'
    try:
        return max(0.5, float(configured))
    except ValueError:
        return 2.0


def build_job_backend_health(job_runtime: dict[str, Any]) -> dict[str, Any]:
    mode = str(job_runtime.get('mode', 'sync')).lower()
    notes = list(job_runtime.get('notes', []))
    if mode == 'external':
        queue_url = str(job_runtime.get('queue_url', '')).strip()
        queue_name = str(job_runtime.get('queue_name', '')).strip() or 'smart-dataset-analyzer'
        if not queue_url:
            return {
                'status': 'Not configured',
                'mode': mode,
                'backend': str(job_runtime.get('external_backend', 'rq')),
                'detail': f'{JOB_QUEUE_URL_ENV} is required for the external worker backend.',
                'notes': notes,
            }
        try:
            adapter = _get_external_rq_adapter(queue_url, queue_name)
            queue = adapter._queue_cls(queue_name, connection=adapter._redis_connection)
            # A simple ping + queue metadata read is enough to confirm the worker transport is reachable.
            adapter._redis_connection.ping()
            pending_count = int(getattr(queue, 'count', 0))
            return {
                'status': 'Healthy',
                'mode': mode,
                'backend': str(job_runtime.get('external_backend', 'rq')),
                'detail': f'External queue "{queue_name}" is reachable and currently has {pending_count} queued job(s).',
                'notes': notes,
                'queue_name': queue_name,
                'queue_url': queue_url,
                'pending_jobs': pending_count,
                'healthcheck_timeout_seconds': _healthcheck_timeout_seconds(),
            }
        except Exception as error:
            return {
                'status': 'Unavailable',
                'mode': mode,
                'backend': str(job_runtime.get('external_backend', 'rq')),
                'detail': f'External queue health check failed: {type(error).__name__}: {error}',
                'notes': notes,
                'queue_name': queue_name,
                'queue_url': queue_url,
                'healthcheck_timeout_seconds': _healthcheck_timeout_seconds(),
            }
    if mode in {'thread', 'worker', 'queue'}:
        max_workers = max(1, int(job_runtime.get('max_workers', 1) or 1))
        return {
            'status': 'Healthy',
            'mode': mode,
            'backend': 'threadpool',
            'detail': f'Threaded worker backend is configured with {max_workers} worker slot(s).',
            'notes': notes,
            'max_workers': max_workers,
        }
    return {
        'status': 'Sync fallback',
        'mode': mode,
        'backend': 'inline',
        'detail': 'Jobs are running in-process. This is expected for local demos and fallback mode.',
        'notes': notes,
    }


def build_job_runtime() -> dict[str, Any]:
    configured_backend = str(os.getenv(JOB_BACKEND_ENV, '')).strip().lower()
    queue_url = str(os.getenv(JOB_QUEUE_URL_ENV, '')).strip()
    queue_name = str(os.getenv(JOB_QUEUE_NAME_ENV, 'smart-dataset-analyzer')).strip() or 'smart-dataset-analyzer'
    healthcheck_timeout = _healthcheck_timeout_seconds()
    if configured_backend in {'rq', 'external', 'redis-rq'}:
        if not queue_url:
            return {
                'backend_configured': False,
                'mode': 'sync',
                'max_workers': 0,
                'status_label': 'Synchronous fallback',
                'healthcheck_timeout_seconds': healthcheck_timeout,
                'notes': [
                    f'{JOB_BACKEND_ENV} requested an external worker backend, but no queue URL was configured.',
                    f'Set {JOB_QUEUE_URL_ENV} to enable Redis + RQ job execution. The platform stayed in synchronous fallback mode.',
                ],
            }
        return {
            'backend_configured': True,
            'mode': 'external',
            'external_backend': 'rq',
            'queue_url': queue_url,
            'queue_name': queue_name,
            'max_workers': 0,
            'status_label': 'External RQ worker backend',
            'healthcheck_timeout_seconds': healthcheck_timeout,
            'notes': [
                f'A managed Redis + RQ worker backend is enabled using queue "{queue_name}".',
                'Registered tasks can run outside the Streamlit process while the UI polls status and results through the existing job contract.',
            ],
        }
    if configured_backend in {'thread', 'worker', 'queue'}:
        max_workers = max(1, int(str(os.getenv(JOB_MAX_WORKERS_ENV, '2')).strip() or '2'))
        return {
            'backend_configured': True,
            'mode': configured_backend,
            'status_label': 'Threaded worker backend',
            'max_workers': max_workers,
            'healthcheck_timeout_seconds': healthcheck_timeout,
            'notes': [
                f'A managed threaded worker backend is enabled with up to {max_workers} concurrent jobs.',
                'Queued jobs now execute in the background while the UI polls status and results through the existing job contract.',
            ],
        }
    return {
        'backend_configured': False,
        'mode': 'sync',
        'max_workers': 0,
        'status_label': 'Synchronous fallback',
        'healthcheck_timeout_seconds': healthcheck_timeout,
        'notes': [
            'Long-running tasks are managed in-process with job-style status updates.',
            'No external worker is configured, so the platform keeps the current Streamlit flow stable.',
        ],
    }


def build_heavy_task_catalog() -> pd.DataFrame:
    return pd.DataFrame(HEAVY_TASK_DEFINITIONS)


def _utcnow() -> str:
    return datetime.now(UTC).isoformat()


def _empty_job_runs() -> list[dict[str, Any]]:
    return []


def get_job_runs(session_state: dict[str, Any]) -> list[dict[str, Any]]:
    runs = session_state.get('job_runs')
    if not isinstance(runs, list):
        runs = _empty_job_runs()
        session_state['job_runs'] = runs
    return runs


def _append_job_run(session_state: dict[str, Any], run: dict[str, Any]) -> None:
    with _job_lock(session_state):
        runs = list(get_job_runs(session_state))
        runs.insert(0, run)
        session_state['job_runs'] = runs[:50]


def _initialize_job_store(session_state: dict[str, Any]) -> dict[str, dict[str, Any]]:
    with _job_lock(session_state):
        store = session_state.get('job_store')
        if not isinstance(store, dict):
            store = {}
            session_state['job_store'] = store
        return store


def _job_runtime_key(session_state: dict[str, Any]) -> str:
    key = str(session_state.get('_job_runtime_key', '')).strip()
    if key:
        return key
    key = f'job-runtime-{uuid.uuid4().hex}'
    session_state['_job_runtime_key'] = key
    return key


def get_job_runtime_key(session_state: dict[str, Any]) -> str:
    return _job_runtime_key(session_state)


def _store_job_entry(session_state: dict[str, Any], job_id: str, entry: dict[str, Any]) -> None:
    with _job_lock(session_state):
        store = _initialize_job_store(session_state)
        store[job_id] = entry


def _job_lock(session_state: dict[str, Any]) -> threading.RLock:
    lock = session_state.get('_job_lock')
    if lock is not None and hasattr(lock, 'acquire') and hasattr(lock, 'release'):
        return lock
    lock = threading.RLock()
    session_state['_job_lock'] = lock
    return lock


def _initialize_job_futures(session_state: dict[str, Any]) -> dict[str, concurrent.futures.Future[Any]]:
    runtime_key = _job_runtime_key(session_state)
    with _job_lock(session_state):
        futures = _GLOBAL_JOB_FUTURES.get(runtime_key)
        if futures is None:
            futures = {}
            _GLOBAL_JOB_FUTURES[runtime_key] = futures
        session_state['job_futures_count'] = len(futures)
        return futures


def _initialize_cancel_flags(session_state: dict[str, Any]) -> dict[str, bool]:
    runtime_key = _job_runtime_key(session_state)
    with _job_lock(session_state):
        flags = _GLOBAL_CANCEL_FLAGS.get(runtime_key)
        if flags is None:
            flags = {}
            _GLOBAL_CANCEL_FLAGS[runtime_key] = flags
        return flags


def _set_cancel_requested(session_state: dict[str, Any], job_id: str, requested: bool) -> None:
    flags = _initialize_cancel_flags(session_state)
    with _job_lock(session_state):
        flags[str(job_id)] = bool(requested)


def job_cancel_requested(runtime_key: str, job_id: str) -> bool:
    normalized_runtime_key = str(runtime_key or '').strip()
    normalized_job_id = str(job_id or '').strip()
    if not normalized_runtime_key or not normalized_job_id:
        return False
    return bool(_GLOBAL_CANCEL_FLAGS.get(normalized_runtime_key, {}).get(normalized_job_id, False))


def _get_worker_executor(
    session_state: dict[str, Any],
    job_runtime: dict[str, Any],
) -> concurrent.futures.ThreadPoolExecutor | None:
    if not bool(job_runtime.get('backend_configured')):
        return None
    runtime_key = _job_runtime_key(session_state)
    with _job_lock(session_state):
        executor = _GLOBAL_JOB_EXECUTORS.get(runtime_key)
        if isinstance(executor, concurrent.futures.ThreadPoolExecutor):
            session_state['_job_executor_token'] = runtime_key
            return executor
        max_workers = max(1, int(job_runtime.get('max_workers', 2) or 2))
        executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix='smart-dataset-job',
        )
        _GLOBAL_JOB_EXECUTORS[runtime_key] = executor
        session_state['_job_executor_token'] = runtime_key
        return executor


def _rotate_worker_executor(session_state: dict[str, Any]) -> None:
    runtime_key = _job_runtime_key(session_state)
    with _job_lock(session_state):
        executor = _GLOBAL_JOB_EXECUTORS.pop(runtime_key, None)
        _GLOBAL_JOB_FUTURES.pop(runtime_key, None)
        _GLOBAL_CANCEL_FLAGS.pop(runtime_key, None)
        session_state.pop('_job_executor_token', None)
        session_state['job_futures_count'] = 0
    if isinstance(executor, concurrent.futures.ThreadPoolExecutor):
        try:
            executor.shutdown(wait=False, cancel_futures=False)
        except Exception:
            pass


def _update_job_entry(
    session_state: dict[str, Any],
    job_id: str,
    **changes: Any,
) -> dict[str, Any]:
    with _job_lock(session_state):
        store = _initialize_job_store(session_state)
        entry = dict(store.get(job_id, {'job_id': job_id}))
        entry.update(changes)
        store[job_id] = entry
        return entry


def _get_job_entry(session_state: dict[str, Any], job_id: str) -> dict[str, Any]:
    store = _initialize_job_store(session_state)
    return dict(
        store.get(
            job_id,
            {
                'job_id': job_id,
                'status': 'unknown',
                'mode': 'sync',
                'cancel_requested': False,
                'backend_ready': False,
            },
        )
    )


def _new_job_entry(
    *,
    job_id: str,
    task_key: str,
    task_label: str,
    mode: str,
    detail: str,
    status: str,
) -> dict[str, Any]:
    return {
        'job_id': job_id,
        'task_key': task_key,
        'task_label': task_label,
        'status': status,
        'mode': mode,
        'detail': detail,
        'duration_seconds': 0.0,
        'result': None,
        'created_at': _utcnow(),
        'started_at': None,
        'completed_at': None,
        'cancel_requested': False,
        'backend_ready': mode != 'sync',
        'context_fields': {},
        '_perf_started': None,
    }


def _build_external_adapter_from_entry(entry: dict[str, Any]) -> ExternalRQJobAdapter | None:
    if str(entry.get('external_backend', '')).lower() != 'rq':
        return None
    queue_url = str(entry.get('queue_url', '')).strip()
    queue_name = str(entry.get('queue_name', '')).strip() or 'smart-dataset-analyzer'
    if not queue_url:
        return None
    return _get_external_rq_adapter(queue_url, queue_name)


def _map_external_job_status(raw_status: str) -> str:
    normalized = str(raw_status or '').strip().lower()
    if normalized in {'queued', 'scheduled', 'deferred'}:
        return 'queued'
    if normalized in {'started', 'busy', 'running'}:
        return 'running'
    if normalized in {'finished', 'complete', 'completed'}:
        return 'completed'
    if normalized in {'failed'}:
        return 'failed'
    if normalized in {'stopped', 'canceled', 'cancelled'}:
        return 'cancelled'
    return 'queued'


def build_job_status_view(job_runs: list[dict[str, Any]]) -> pd.DataFrame:
    if not job_runs:
        return pd.DataFrame(columns=['task', 'status', 'mode', 'duration_seconds', 'detail', 'cancel_requested'])
    return pd.DataFrame(
        [
            {
                'task': run.get('task_label', run.get('task_key', 'Managed task')),
                'status': run.get('status', 'completed'),
                'mode': run.get('mode', 'sync'),
                'duration_seconds': round(float(run.get('duration_seconds', 0.0)), 2),
                'detail': run.get('detail', ''),
                'cancel_requested': 'Yes' if bool(run.get('cancel_requested')) else 'No',
            }
            for run in job_runs
        ]
    )


def get_job_status(session_state: dict[str, Any], job_id: str) -> dict[str, Any]:
    _synchronize_job_futures(session_state)
    _synchronize_external_jobs(session_state)
    return _get_job_entry(session_state, job_id)


def get_job_result(session_state: dict[str, Any], job_id: str) -> Any:
    _synchronize_job_futures(session_state)
    _synchronize_external_jobs(session_state)
    store = _initialize_job_store(session_state)
    entry = store.get(job_id, {})
    if str(entry.get('status', '')).lower() in {'cancelled', 'failed', 'queued'}:
        return None
    return entry.get('result')


def cancel_job(session_state: dict[str, Any], job_id: str) -> dict[str, Any]:
    _synchronize_job_futures(session_state)
    _synchronize_external_jobs(session_state)
    store = _initialize_job_store(session_state)
    entry = dict(store.get(job_id, {'job_id': job_id, 'status': 'unknown', 'mode': 'sync'}))
    context_fields = dict(entry.get('context_fields', {}))
    current_status = str(entry.get('status', 'unknown')).lower()
    _set_cancel_requested(session_state, job_id, True)
    if current_status in {'completed', 'failed', 'cancelled', 'unknown'}:
        entry['cancel_requested'] = current_status not in {'completed', 'failed'}
        return entry
    entry['cancel_requested'] = True
    future = _initialize_job_futures(session_state).get(job_id)
    if future is not None and future.cancel():
        entry['status'] = 'cancelled'
        entry['completed_at'] = _utcnow()
        entry['detail'] = f"{entry.get('detail', 'Managed task')} Cancellation was requested before execution started."
        entry['result'] = None
        _store_job_entry(session_state, job_id, entry)
        _append_job_run(session_state, entry)
        logger_module.log_platform_event(
            'job_cancelled',
            logger_name='jobs',
            job_id=job_id,
            task_key=entry.get('task_key', ''),
            mode=entry.get('mode', ''),
            detail=entry.get('detail', ''),
            operation_type=entry.get('task_key', ''),
            **context_fields,
        )
        return entry
    external_job_id = str(entry.get('external_job_id', '')).strip()
    if external_job_id:
        try:
            adapter = _build_external_adapter_from_entry(entry)
            if adapter is not None and adapter.cancel(external_job_id):
                entry['status'] = 'cancelled'
                entry['completed_at'] = _utcnow()
                entry['detail'] = f"{entry.get('detail', 'Managed task')} Cancellation was requested through the external worker backend."
                entry['result'] = None
                _store_job_entry(session_state, job_id, entry)
                _append_job_run(session_state, entry)
                logger_module.log_platform_event(
                    'job_cancelled',
                    logger_name='jobs',
                    job_id=job_id,
                    task_key=entry.get('task_key', ''),
                    mode=entry.get('mode', ''),
                    detail=entry.get('detail', ''),
                    operation_type=entry.get('task_key', ''),
                    **context_fields,
                )
                return entry
        except Exception:
            pass
    if current_status == 'queued':
        entry['status'] = 'cancelled'
        entry['completed_at'] = _utcnow()
        entry['detail'] = f"{entry.get('detail', 'Managed task')} Cancellation was requested before execution started."
        _append_job_run(session_state, entry)
    _store_job_entry(session_state, job_id, entry)
    if entry.get('status') == 'cancelled':
        logger_module.log_platform_event(
            'job_cancelled',
            logger_name='jobs',
            job_id=job_id,
            task_key=entry.get('task_key', ''),
            mode=entry.get('mode', ''),
            detail=entry.get('detail', ''),
            operation_type=entry.get('task_key', ''),
            **context_fields,
        )
    return entry


def force_cancel_job(
    session_state: dict[str, Any],
    job_id: str,
    *,
    reason: str = 'Cancellation timed out and the managed job was force-killed.',
) -> dict[str, Any]:
    _synchronize_job_futures(session_state)
    _synchronize_external_jobs(session_state)
    current = _get_job_entry(session_state, job_id)
    _set_cancel_requested(session_state, job_id, True)
    terminal_status = str(current.get('status', '')).lower()
    if terminal_status in {'completed', 'failed', 'cancelled'}:
        current['force_cancelled'] = bool(current.get('force_cancelled', False))
        current['cancellation_completed_at'] = current.get('cancellation_completed_at') or _utcnow()
        return current
    future = _initialize_job_futures(session_state).pop(job_id, None)
    should_rotate_executor = str(current.get('mode', '')).lower() in {'worker', 'thread', 'queue'}
    if future is not None:
        try:
            future.cancel()
        except Exception:
            pass
    external_job_id = str(current.get('external_job_id', '')).strip()
    if external_job_id:
        try:
            adapter = _build_external_adapter_from_entry(current)
            if adapter is not None:
                adapter.cancel(external_job_id)
        except Exception:
            pass
    cancelled_entry = dict(current)
    cancelled_entry.update(
        {
            'status': 'cancelled',
            'cancel_requested': True,
            'force_cancelled': True,
            'force_cancel_reason': reason,
            'completed_at': _utcnow(),
            'cancellation_completed_at': _utcnow(),
            'detail': reason,
            'result': None,
        }
    )
    _store_job_entry(session_state, job_id, cancelled_entry)
    if should_rotate_executor:
        _rotate_worker_executor(session_state)
    _append_job_run(session_state, cancelled_entry)
    logger_module.log_platform_event(
        'job_force_cancelled',
        logger_name='jobs',
        job_id=job_id,
        task_key=cancelled_entry.get('task_key', ''),
        mode=cancelled_entry.get('mode', ''),
        detail=reason,
        operation_type=cancelled_entry.get('task_key', ''),
        completed_at=cancelled_entry.get('completed_at', ''),
        **dict(cancelled_entry.get('context_fields', {})),
    )
    return cancelled_entry


def build_job_user_message(job_runtime: dict[str, Any], task_key: str) -> str:
    row = next((task for task in HEAVY_TASK_DEFINITIONS if task['task_key'] == task_key), None)
    task_label = row['task_label'] if row else 'This task'
    if str(job_runtime.get('mode', '')).lower() == 'external':
        return f'{task_label} is using the external worker backend. The UI can keep polling queued, running, and completion states while the task finishes outside the Streamlit process.'
    if bool(job_runtime.get('backend_configured')):
        return f'{task_label} is using the managed background worker backend. The UI can keep polling queued, running, and completion states while the task finishes outside the main Streamlit request.'
    return f'{task_label} may take a little longer. The platform will keep showing managed job status while this task runs in the current session.'


def _execute_job_inline(
    session_state: dict[str, Any],
    job_runtime: dict[str, Any],
    *,
    job_id: str,
    task_key: str,
    task_label: str,
    detail: str,
    runner: Callable[[], Any],
    started: float,
    stage_messages: list[str],
    progress_callback: Callable[[float, str], None] | None,
) -> dict[str, Any]:
    existing_entry = _get_job_entry(session_state, job_id)
    running_entry = _new_job_entry(
        job_id=job_id,
        task_key=task_key,
        task_label=task_label,
        mode=str(job_runtime.get('mode', 'sync')),
        detail=detail,
        status='running',
    )
    running_entry['started_at'] = _utcnow()
    running_entry['context_fields'] = dict(existing_entry.get('context_fields', {}))
    context_fields = dict(running_entry.get('context_fields', {}))
    _store_job_entry(session_state, job_id, running_entry)
    if progress_callback and stage_messages:
        for index, message in enumerate(stage_messages[:-1], start=1):
            progress_callback(index / max(len(stage_messages), 1), message)
    try:
        result = runner()
    except Exception as error:
        duration = time.perf_counter() - started
        latest_entry = _get_job_entry(session_state, job_id)
        if bool(latest_entry.get('force_cancelled')):
            logger_module.log_platform_event(
                'job_failure_ignored_after_force_cancel',
                logger_name='jobs',
                job_id=job_id,
                task_key=task_key,
                mode=job_runtime.get('mode', 'sync'),
                operation_type=task_key,
                **context_fields,
            )
            return latest_entry
        if type(error).__name__ == 'AnalysisCancelledError':
            cancelled_run = dict(running_entry)
            cancelled_run.update(
                {
                    'status': 'cancelled',
                    'duration_seconds': duration,
                    'completed_at': _utcnow(),
                    'cancel_requested': True,
                    'detail': f'{task_label} was cancelled before the next analysis stage started.',
                    'result': None,
                }
            )
            _store_job_entry(session_state, job_id, cancelled_run)
            _append_job_run(session_state, cancelled_run)
            logger_module.log_platform_event(
                'job_cancelled',
                logger_name='jobs',
                job_id=job_id,
                task_key=task_key,
                mode=job_runtime.get('mode', 'sync'),
                detail=cancelled_run.get('detail', ''),
                operation_type=task_key,
                **context_fields,
            )
            if progress_callback:
                progress_callback(1.0, f'{task_label} was cancelled.')
            raise
        failed_run = dict(running_entry)
        failed_run.update(
            {
                'status': 'failed',
                'duration_seconds': duration,
                'error_type': type(error).__name__,
                'error_message': str(error),
                'completed_at': _utcnow(),
                'result': None,
            }
        )
        _store_job_entry(session_state, job_id, failed_run)
        _append_job_run(session_state, failed_run)
        logger_module.log_platform_exception(
            'job_failed',
            error,
            logger_name='jobs',
            job_id=job_id,
            task_key=task_key,
            mode=job_runtime.get('mode', 'sync'),
            operation_type=task_key,
            duration_seconds=duration,
            **context_fields,
        )
        if progress_callback:
            progress_callback(1.0, f'{task_label} could not finish.')
        raise

    duration = time.perf_counter() - started
    latest_entry = _get_job_entry(session_state, job_id)
    if bool(latest_entry.get('force_cancelled')):
        logger_module.log_platform_event(
            'job_completion_ignored_after_force_cancel',
            logger_name='jobs',
            job_id=job_id,
            task_key=task_key,
            mode=job_runtime.get('mode', 'sync'),
            operation_type=task_key,
            **context_fields,
        )
        return latest_entry
    completed_run = dict(running_entry)
    completed_run.update(
        {
            'status': 'completed',
            'duration_seconds': duration,
            'completed_at': _utcnow(),
            'result': result,
        }
    )
    _store_job_entry(session_state, job_id, completed_run)
    _append_job_run(session_state, completed_run)
    logger_module.log_platform_event(
        'job_completed',
        logger_name='jobs',
        job_id=job_id,
        task_key=task_key,
        mode=job_runtime.get('mode', 'sync'),
        duration_seconds=duration,
        operation_type=task_key,
        **context_fields,
    )
    if progress_callback:
        final_message = stage_messages[-1] if stage_messages else f'{task_label} completed.'
        progress_callback(1.0, final_message)
    return completed_run


def _execute_job_background(runner: Callable[[], Any]) -> Any:
    return runner()


def _synchronize_job_futures(session_state: dict[str, Any]) -> None:
    futures = _initialize_job_futures(session_state)
    finished: list[str] = []
    for job_id, future in list(futures.items()):
        if future.running():
            current = _get_job_entry(session_state, job_id)
            if current.get('status') == 'queued':
                _update_job_entry(
                    session_state,
                    job_id,
                    status='running',
                    started_at=current.get('started_at') or _utcnow(),
                )
            continue
        if future.cancelled():
            finished.append(job_id)
            current = _get_job_entry(session_state, job_id)
            if current.get('status') != 'cancelled':
                cancelled = dict(current)
                cancelled.update(
                    {
                        'status': 'cancelled',
                        'cancel_requested': True,
                        'completed_at': _utcnow(),
                        'detail': f"{current.get('detail', 'Managed task')} Cancellation was requested before execution started.",
                    }
                )
                _store_job_entry(session_state, job_id, cancelled)
                _append_job_run(session_state, cancelled)
            continue
        if future.done():
            finished.append(job_id)
            current = _get_job_entry(session_state, job_id)
            perf_started = float(current.get('_perf_started', 0.0) or 0.0)
            duration = max(time.perf_counter() - perf_started, 0.0) if perf_started else 0.0
            try:
                result = future.result()
            except Exception as error:
                latest_entry = _get_job_entry(session_state, job_id)
                context_fields = dict(latest_entry.get('context_fields', {}))
                if bool(latest_entry.get('force_cancelled')):
                    logger_module.log_platform_event(
                        'job_failure_ignored_after_force_cancel',
                        logger_name='jobs',
                        job_id=job_id,
                        task_key=latest_entry.get('task_key', ''),
                        mode=latest_entry.get('mode', 'worker'),
                        operation_type=latest_entry.get('task_key', ''),
                        **context_fields,
                    )
                    continue
                if type(error).__name__ == 'AnalysisCancelledError':
                    cancelled = dict(latest_entry)
                    cancelled.update(
                        {
                            'status': 'cancelled',
                            'duration_seconds': duration,
                            'completed_at': _utcnow(),
                            'cancel_requested': True,
                            'detail': f"{latest_entry.get('task_label', 'Managed task')} was cancelled before the next analysis stage started.",
                            'result': None,
                        }
                    )
                    _store_job_entry(session_state, job_id, cancelled)
                    _append_job_run(session_state, cancelled)
                    logger_module.log_platform_event(
                        'job_cancelled',
                        logger_name='jobs',
                        job_id=job_id,
                        task_key=cancelled.get('task_key', ''),
                        mode=cancelled.get('mode', 'worker'),
                        detail=cancelled.get('detail', ''),
                        operation_type=cancelled.get('task_key', ''),
                        **context_fields,
                    )
                    continue
                failed = dict(latest_entry)
                failed.update(
                    {
                        'status': 'failed',
                        'duration_seconds': duration,
                        'error_type': type(error).__name__,
                        'error_message': str(error),
                        'completed_at': _utcnow(),
                        'result': None,
                    }
                )
                _store_job_entry(session_state, job_id, failed)
                _append_job_run(session_state, failed)
                logger_module.log_platform_exception(
                    'job_failed',
                    error,
                    logger_name='jobs',
                    job_id=job_id,
                    task_key=failed.get('task_key', ''),
                    mode=failed.get('mode', 'worker'),
                    operation_type=failed.get('task_key', ''),
                    duration_seconds=duration,
                    **context_fields,
                )
                continue
            latest_entry = _get_job_entry(session_state, job_id)
            context_fields = dict(latest_entry.get('context_fields', {}))
            if bool(latest_entry.get('force_cancelled')):
                logger_module.log_platform_event(
                    'job_completion_ignored_after_force_cancel',
                    logger_name='jobs',
                    job_id=job_id,
                    task_key=latest_entry.get('task_key', ''),
                    mode=latest_entry.get('mode', 'worker'),
                    operation_type=latest_entry.get('task_key', ''),
                    **context_fields,
                )
                continue
            completed = dict(latest_entry)
            completed.update(
                {
                    'status': 'completed',
                    'duration_seconds': duration,
                    'completed_at': _utcnow(),
                    'result': result,
                }
            )
            _store_job_entry(session_state, job_id, completed)
            _append_job_run(session_state, completed)
            logger_module.log_platform_event(
                'job_completed',
                logger_name='jobs',
                job_id=job_id,
                task_key=completed.get('task_key', ''),
                mode=completed.get('mode', 'worker'),
                duration_seconds=duration,
                operation_type=completed.get('task_key', ''),
                **context_fields,
            )
    with _job_lock(session_state):
        for job_id in finished:
            futures.pop(job_id, None)
        session_state['job_futures_count'] = len(futures)


def _synchronize_external_jobs(session_state: dict[str, Any]) -> None:
    store = _initialize_job_store(session_state)
    for job_id, entry in list(store.items()):
        external_job_id = str(entry.get('external_job_id', '')).strip()
        if not external_job_id:
            continue
        current_status = str(entry.get('status', '')).lower()
        if current_status in {'completed', 'failed', 'cancelled'}:
            continue
        try:
            adapter = _build_external_adapter_from_entry(entry)
            if adapter is None:
                continue
            external_job = adapter.fetch(external_job_id)
            raw_status = (
                external_job.get_status(refresh=True)
                if hasattr(external_job, 'get_status')
                else getattr(external_job, 'status', 'queued')
            )
            mapped_status = _map_external_job_status(raw_status)
            updates: dict[str, Any] = {'status': mapped_status}
            if mapped_status == 'running' and not entry.get('started_at'):
                updates['started_at'] = _utcnow()
            if mapped_status == 'completed':
                updates['completed_at'] = _utcnow()
                updates['result'] = getattr(external_job, 'result', None)
            elif mapped_status == 'failed':
                updates['completed_at'] = _utcnow()
                failure = getattr(external_job, 'exc_info', '') or getattr(external_job, 'exc_string', '')
                updates['error_message'] = str(failure)
                updates['result'] = None
            elif mapped_status == 'cancelled':
                updates['completed_at'] = _utcnow()
                updates['cancel_requested'] = True
                updates['result'] = None
            updated = _update_job_entry(session_state, job_id, **updates)
            if mapped_status in {'completed', 'failed', 'cancelled'} and not bool(updated.get('_terminal_logged')):
                terminal_entry = dict(updated)
                terminal_entry['_terminal_logged'] = True
                _store_job_entry(session_state, job_id, terminal_entry)
                _append_job_run(session_state, terminal_entry)
                logger_module.log_platform_event(
                    f"job_{mapped_status}",
                    logger_name='jobs',
                    job_id=job_id,
                    task_key=terminal_entry.get('task_key', ''),
                    mode=terminal_entry.get('mode', ''),
                    external_job_id=terminal_entry.get('external_job_id', ''),
                    operation_type=terminal_entry.get('task_key', ''),
                    **dict(terminal_entry.get('context_fields', {})),
                )
        except Exception as error:
            failed = _update_job_entry(
                session_state,
                job_id,
                status='failed',
                completed_at=_utcnow(),
                error_type=type(error).__name__,
                error_message=str(error),
                result=None,
            )
            if not bool(failed.get('_terminal_logged')):
                terminal_entry = dict(failed)
                terminal_entry['_terminal_logged'] = True
                _store_job_entry(session_state, job_id, terminal_entry)
                _append_job_run(session_state, terminal_entry)
            logger_module.log_platform_exception(
                'job_sync_failed',
                error,
                logger_name='jobs',
                job_id=job_id,
                task_key=failed.get('task_key', ''),
                mode=failed.get('mode', ''),
                operation_type=failed.get('task_key', 'job_sync'),
                **dict(failed.get('context_fields', {})),
            )


def submit_job(
    session_state: dict[str, Any],
    job_runtime: dict[str, Any],
    *,
    job_id: str | None = None,
    task_key: str,
    task_label: str,
    runner: Callable[[], Any] | None = None,
    detail: str,
    stage_messages: list[str] | None = None,
    progress_callback: Callable[[float, str], None] | None = None,
    task_name: str | None = None,
    task_kwargs: dict[str, Any] | None = None,
    context_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    stage_messages = [str(item) for item in (stage_messages or []) if str(item).strip()]
    task_kwargs = dict(task_kwargs or {})
    context_fields = dict(context_fields or {})
    job_id = str(job_id or f'{task_key}-{uuid.uuid4().hex[:10]}')
    started = time.perf_counter()
    queued_entry = _new_job_entry(
        job_id=job_id,
        task_key=task_key,
        task_label=task_label,
        mode=str(job_runtime.get('mode', 'sync')),
        detail=detail,
        status='queued',
    )
    _store_job_entry(session_state, job_id, queued_entry)
    queued_entry['context_fields'] = context_fields
    queued_entry['_perf_started'] = started
    _store_job_entry(session_state, job_id, queued_entry)
    logger_module.log_platform_event(
        'job_submitted',
        logger_name='jobs',
        job_id=job_id,
        task_key=task_key,
        task_label=task_label,
        mode=job_runtime.get('mode', 'sync'),
        backend_configured=job_runtime.get('backend_configured', False),
        operation_type=task_key,
        **context_fields,
    )
    if runner is None and not task_name:
        raise ValueError('submit_job requires either a runner or a registered task_name.')
    if bool(job_runtime.get('backend_configured')):
        if progress_callback:
            progress_callback(0.05, f'{task_label} has been queued in the managed worker backend.')
        if str(job_runtime.get('mode', '')).lower() == 'external':
            if task_name:
                try:
                    adapter = _get_external_rq_adapter(
                        str(job_runtime.get('queue_url', '')),
                        str(job_runtime.get('queue_name', 'smart-dataset-analyzer')),
                    )
                    external_ref = adapter.enqueue_registered_task(task_name, task_kwargs)
                    queued_entry.update(
                        {
                            'mode': 'external',
                            'backend_ready': True,
                            'external_backend': str(job_runtime.get('external_backend', 'rq')),
                            'external_job_id': external_ref.get('external_job_id'),
                            'queue_url': external_ref.get('queue_url'),
                            'queue_name': external_ref.get('queue_name'),
                            'task_name': task_name,
                        }
                    )
                    _store_job_entry(session_state, job_id, queued_entry)
                    return {'job_id': job_id, 'status': 'queued', 'job_run': dict(queued_entry)}
                except Exception as error:
                    queued_entry['detail'] = f'{detail} External worker submission fell back to in-process execution: {error}'
                    _store_job_entry(session_state, job_id, queued_entry)
            else:
                queued_entry['detail'] = (
                    f'{detail} External worker submission fell back to in-process execution because '
                    'this job was submitted as an in-memory callable rather than a registered task.'
                )
                _store_job_entry(session_state, job_id, queued_entry)
        executor = _get_worker_executor(session_state, job_runtime)
        if executor is not None:
            try:
                future = executor.submit(
                    _execute_job_background,
                    runner=runner or (lambda: execute_registered_job_task(str(task_name), task_kwargs)),
                )
                _initialize_job_futures(session_state)[job_id] = future
                return {'job_id': job_id, 'status': 'queued', 'job_run': dict(queued_entry)}
            except Exception as error:
                fallback_detail = f'{detail} Worker submission fell back to in-process execution: {error}'
                queued_entry['detail'] = fallback_detail
                _store_job_entry(session_state, job_id, queued_entry)
    completed_run = _execute_job_inline(
        session_state,
        job_runtime,
        job_id=job_id,
        task_key=task_key,
        task_label=task_label,
        detail=queued_entry.get('detail', detail),
        runner=runner or (lambda: execute_registered_job_task(str(task_name), task_kwargs)),
        started=started,
        stage_messages=stage_messages,
        progress_callback=progress_callback,
    )
    return {'job_id': job_id, 'status': completed_run['status'], 'job_run': completed_run}


def run_managed_job(
    session_state: dict[str, Any],
    job_runtime: dict[str, Any],
    *,
    task_key: str,
    task_label: str,
    runner: Callable[[], Any],
    detail: str,
    stage_messages: list[str] | None = None,
    progress_callback: Callable[[float, str], None] | None = None,
    context_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    submission = submit_job(
        session_state,
        job_runtime,
        task_key=task_key,
        task_label=task_label,
        runner=runner,
        detail=detail,
        stage_messages=stage_messages,
        progress_callback=progress_callback,
        context_fields=context_fields,
    )
    return {
        'result': get_job_result(session_state, submission['job_id']),
        'job_run': submission['job_run'],
        'job_id': submission['job_id'],
        'status': submission['status'],
    }
