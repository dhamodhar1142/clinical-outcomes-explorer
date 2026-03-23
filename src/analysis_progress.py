from __future__ import annotations

import time
from threading import RLock
from typing import Any

ETA_REFRESH_INTERVAL_SECONDS = 2.0
ETA_MIN_PROGRESS_FOR_OBSERVED_RATE = 0.03
ETA_SMOOTHING_ALPHA = 0.35


ANALYSIS_STAGE_LABELS = [
    'Loading data',
    'Profiling columns',
    'Analyzing quality',
    'Computing healthcare metrics',
    'Finalizing insights',
]


_GLOBAL_PROGRESS_LOCK = RLock()
_GLOBAL_PROGRESS_SNAPSHOTS: dict[str, dict[str, dict[str, Any]]] = {}


def estimate_analysis_seconds(
    *,
    file_size_mb: float,
    row_count: int,
    column_count: int,
    source_mode: str = '',
    dataset_name: str = '',
) -> float:
    normalized_source_mode = str(source_mode or '').strip().lower()
    normalized_dataset_name = str(dataset_name or '').strip().lower()
    if normalized_source_mode == 'demo dataset':
        if normalized_dataset_name == 'healthcare operations demo':
            return 45.0
        if normalized_dataset_name == 'hospital reporting demo':
            return 30.0
        if normalized_dataset_name == 'generic business demo':
            return 20.0
    size_component = max(float(file_size_mb), 0.0) * 0.7
    row_component = max(int(row_count), 0) / 120_000.0
    column_component = max(int(column_count), 0) / 40.0
    return max(4.0, size_component + row_component + column_component)


def _analysis_complexity_score(*, file_size_mb: float, row_count: int, column_count: int) -> float:
    normalized_size = max(float(file_size_mb), 0.0) / 50.0
    normalized_rows = max(int(row_count), 0) / 250_000.0
    normalized_columns = max(int(column_count), 0) / 30.0
    return max(1.0, 1.0 + normalized_size + normalized_rows + normalized_columns)


def _eta_confidence(*, progress: float, elapsed_seconds: float, sample_count: int) -> tuple[str, float]:
    if progress >= 0.65 or elapsed_seconds >= 25.0 or sample_count >= 6:
        return 'High', 0.12
    if progress >= 0.25 or elapsed_seconds >= 10.0 or sample_count >= 3:
        return 'Medium', 0.22
    return 'Low', 0.35


def _calculate_eta_seconds(
    *,
    progress: float,
    elapsed_seconds: float,
    baseline_seconds: float,
    complexity_score: float,
) -> float:
    bounded_progress = max(0.0, min(1.0, float(progress)))
    if bounded_progress >= 0.999:
        return 0.0
    baseline_total = baseline_seconds * min(max(complexity_score / 2.0, 0.85), 1.75)
    if bounded_progress < ETA_MIN_PROGRESS_FOR_OBSERVED_RATE or elapsed_seconds <= 0.0:
        return max(baseline_total - elapsed_seconds, 0.0)
    observed_total = elapsed_seconds / max(bounded_progress, ETA_MIN_PROGRESS_FOR_OBSERVED_RATE)
    blended_total = (observed_total * 0.7) + (baseline_total * 0.3)
    return max(blended_total - elapsed_seconds, 0.0)


def _smoothed_eta(
    *,
    raw_eta_seconds: float,
    now: float,
    previous_snapshot: dict[str, Any] | None,
) -> tuple[float, float, int]:
    if not isinstance(previous_snapshot, dict):
        return raw_eta_seconds, now, 1
    previous_eta = float(previous_snapshot.get('estimated_remaining_seconds', raw_eta_seconds) or raw_eta_seconds)
    previous_updated_at = float(previous_snapshot.get('eta_updated_at', 0.0) or 0.0)
    previous_sample_count = int(previous_snapshot.get('eta_sample_count', 1) or 1)
    if previous_updated_at > 0.0 and (now - previous_updated_at) < ETA_REFRESH_INTERVAL_SECONDS:
        return previous_eta, previous_updated_at, previous_sample_count
    smoothed_eta = (ETA_SMOOTHING_ALPHA * raw_eta_seconds) + ((1.0 - ETA_SMOOTHING_ALPHA) * previous_eta)
    return max(smoothed_eta, 0.0), now, previous_sample_count + 1


def build_analysis_progress_snapshot(
    *,
    progress: float,
    message: str,
    current_operation: str,
    step_index: int,
    total_steps: int,
    started_at: float,
    file_size_mb: float,
    row_count: int,
    column_count: int,
    status: str = 'running',
    cancel_requested: bool = False,
    previous_snapshot: dict[str, Any] | None = None,
    source_mode: str = '',
    dataset_name: str = '',
) -> dict[str, Any]:
    now = time.monotonic()
    bounded_progress = max(0.0, min(1.0, float(progress)))
    elapsed_seconds = max(now - float(started_at), 0.0)
    baseline_seconds = estimate_analysis_seconds(
        file_size_mb=file_size_mb,
        row_count=row_count,
        column_count=column_count,
        source_mode=source_mode,
        dataset_name=dataset_name,
    )
    complexity_score = _analysis_complexity_score(
        file_size_mb=file_size_mb,
        row_count=row_count,
        column_count=column_count,
    )
    if str(source_mode or '').strip().lower() == 'demo dataset':
        complexity_score = max(complexity_score, 2.0)
    raw_remaining_seconds = _calculate_eta_seconds(
        progress=bounded_progress,
        elapsed_seconds=elapsed_seconds,
        baseline_seconds=baseline_seconds,
        complexity_score=complexity_score,
    )
    remaining_seconds, eta_updated_at, eta_sample_count = _smoothed_eta(
        raw_eta_seconds=raw_remaining_seconds,
        now=now,
        previous_snapshot=previous_snapshot,
    )
    confidence_label, uncertainty_ratio = _eta_confidence(
        progress=bounded_progress,
        elapsed_seconds=elapsed_seconds,
        sample_count=eta_sample_count,
    )
    uncertainty_seconds = max(2.0, min(30.0, remaining_seconds * uncertainty_ratio))
    return {
        'status': str(status),
        'progress': bounded_progress,
        'percent_complete': int(round(bounded_progress * 100.0)),
        'message': str(message),
        'current_operation': str(current_operation or message),
        'step_index': int(step_index),
        'total_steps': int(total_steps),
        'step_label': ANALYSIS_STAGE_LABELS[max(min(int(step_index) - 1, len(ANALYSIS_STAGE_LABELS) - 1), 0)] if total_steps else str(message),
        'elapsed_seconds': elapsed_seconds,
        'estimated_remaining_seconds': remaining_seconds,
        'eta_confidence': confidence_label,
        'eta_uncertainty_seconds': uncertainty_seconds,
        'eta_updated_at': eta_updated_at,
        'eta_sample_count': eta_sample_count,
        'complexity_score': complexity_score,
        'cancel_requested': bool(cancel_requested),
    }


def store_analysis_progress_snapshot(runtime_key: str, job_id: str, snapshot: dict[str, Any]) -> None:
    normalized_runtime_key = str(runtime_key or '').strip()
    normalized_job_id = str(job_id or '').strip()
    if not normalized_runtime_key or not normalized_job_id:
        return
    with _GLOBAL_PROGRESS_LOCK:
        runtime_snapshots = _GLOBAL_PROGRESS_SNAPSHOTS.setdefault(normalized_runtime_key, {})
        runtime_snapshots[normalized_job_id] = dict(snapshot)


def get_analysis_progress_snapshot(runtime_key: str, job_id: str) -> dict[str, Any]:
    normalized_runtime_key = str(runtime_key or '').strip()
    normalized_job_id = str(job_id or '').strip()
    if not normalized_runtime_key or not normalized_job_id:
        return {}
    with _GLOBAL_PROGRESS_LOCK:
        return dict(_GLOBAL_PROGRESS_SNAPSHOTS.get(normalized_runtime_key, {}).get(normalized_job_id, {}))


def clear_analysis_progress_snapshot(runtime_key: str, job_id: str) -> None:
    normalized_runtime_key = str(runtime_key or '').strip()
    normalized_job_id = str(job_id or '').strip()
    if not normalized_runtime_key or not normalized_job_id:
        return
    with _GLOBAL_PROGRESS_LOCK:
        runtime_snapshots = _GLOBAL_PROGRESS_SNAPSHOTS.get(normalized_runtime_key)
        if not isinstance(runtime_snapshots, dict):
            return
        runtime_snapshots.pop(normalized_job_id, None)
        if not runtime_snapshots:
            _GLOBAL_PROGRESS_SNAPSHOTS.pop(normalized_runtime_key, None)
