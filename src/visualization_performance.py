from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.enterprise_features import cohort_monitoring_over_time
from src.healthcare_analysis import build_cohort_summary


def _safe_df(table: Any) -> pd.DataFrame:
    return table if isinstance(table, pd.DataFrame) else pd.DataFrame()


def _json_default(value: Any) -> Any:
    if isinstance(value, tuple):
        return list(value)
    return str(value)


def _dataset_cache_token(pipeline: dict[str, Any]) -> str:
    data = pipeline.get('data')
    if isinstance(data, pd.DataFrame):
        cache_key = str(data.attrs.get('dataset_cache_key', '')).strip()
        if cache_key:
            return cache_key
        return f"{len(data)}x{len(data.columns)}:{'|'.join(map(str, data.columns[:8]))}"
    return 'unknown-dataset'


def _serialize_filters(filters: dict[str, Any]) -> str:
    normalized = {key: value for key, value in sorted(filters.items())}
    return json.dumps(normalized, default=_json_default, sort_keys=True)


def build_visual_cache_key(pipeline: dict[str, Any], view_name: str, extra: str = '') -> str:
    base = f"{view_name}:{_dataset_cache_token(pipeline)}"
    return f"{base}:{extra}" if extra else base


def get_cached_visual_payload(session_state: dict[str, Any], cache_key: str) -> Any:
    entry = session_state.setdefault('visualization_payload_cache', {}).get(cache_key)
    if isinstance(entry, dict) and 'payload' in entry:
        return entry.get('payload')
    return entry


def store_cached_visual_payload(session_state: dict[str, Any], cache_key: str, payload: Any) -> Any:
    session_state.setdefault('visualization_payload_cache', {})[cache_key] = payload
    return payload


def store_visual_payload_for_pipeline(
    session_state: dict[str, Any],
    pipeline: dict[str, Any],
    cache_key: str,
    payload: Any,
) -> Any:
    data = pipeline.get('data')
    session_state.setdefault('visualization_payload_cache', {})[cache_key] = {
        'payload': payload,
        'dataset_token': _dataset_cache_token(pipeline),
        'dataset_identifier': str(getattr(data, 'attrs', {}).get('dataset_identifier', '')),
        'dataset_name': str(getattr(data, 'attrs', {}).get('dataset_name', '')),
        'row_count': int(len(data)) if isinstance(data, pd.DataFrame) else 0,
        'column_count': int(len(data.columns)) if isinstance(data, pd.DataFrame) else 0,
    }
    return payload


def build_visual_cache_diagnostics(session_state: dict[str, Any], pipeline: dict[str, Any], cache_key: str) -> dict[str, Any]:
    entry = session_state.setdefault('visualization_payload_cache', {}).get(cache_key)
    expected_token = _dataset_cache_token(pipeline)
    data = pipeline.get('data')
    expected_identifier = str(getattr(data, 'attrs', {}).get('dataset_identifier', ''))
    if isinstance(entry, dict) and 'payload' in entry:
        cached_token = str(entry.get('dataset_token', ''))
        cached_identifier = str(entry.get('dataset_identifier', ''))
        return {
            'cache_key': cache_key,
            'expected_dataset_token': expected_token,
            'cached_dataset_token': cached_token,
            'expected_dataset_identifier': expected_identifier,
            'cached_dataset_identifier': cached_identifier,
            'cache_belongs_to_current_dataset': cached_token == expected_token and (
                not expected_identifier or cached_identifier == expected_identifier
            ),
            'row_count': int(entry.get('row_count', 0) or 0),
            'column_count': int(entry.get('column_count', 0) or 0),
            'dataset_name': str(entry.get('dataset_name', '')),
        }
    return {
        'cache_key': cache_key,
        'expected_dataset_token': expected_token,
        'expected_dataset_identifier': expected_identifier,
        'cache_belongs_to_current_dataset': False,
        'row_count': 0,
        'column_count': 0,
        'dataset_name': '',
    }


def build_trend_analysis_payload(pipeline: dict[str, Any]) -> dict[str, Any]:
    structure = pipeline['structure']
    data = pipeline['data']
    date_col = getattr(structure, 'default_date_column', None)
    monthly = pd.DataFrame()
    if date_col and date_col in data.columns:
        trend_source = data[[date_col]].copy()
        trend_source[date_col] = pd.to_datetime(trend_source[date_col], errors='coerce')
        trend_source = trend_source.dropna(subset=[date_col])
        if not trend_source.empty:
            monthly = (
                trend_source.assign(month=trend_source[date_col].dt.to_period('M').dt.to_timestamp())
                .groupby('month')
                .size()
                .reset_index(name='record_count')
            )

    correlation = data[structure.numeric_columns].corr(numeric_only=True) if getattr(structure, 'numeric_columns', []) else pd.DataFrame()
    survival = pipeline['healthcare'].get('survival_outcomes', {})
    readmission = pipeline['healthcare'].get('readmission', {})
    return {
        'date_col': date_col,
        'monthly': monthly,
        'survival_available': bool(survival.get('available')),
        'survival_tables': {
            'stage_table': _safe_df(survival.get('stage_table')),
            'treatment_table': _safe_df(survival.get('treatment_table')),
            'duration_distribution': _safe_df(survival.get('duration_distribution')),
            'outcome_trend': _safe_df(survival.get('outcome_trend')),
            'treatment_duration_trend': _safe_df(survival.get('treatment_duration_trend')),
            'progression_timeline': _safe_df(survival.get('progression_timeline')),
        },
        'readmission_available': bool(readmission.get('available')),
        'readmission_trend': _safe_df(readmission.get('trend')),
        'correlation': correlation,
    }


def build_cohort_analysis_payload(pipeline: dict[str, Any], filters: dict[str, Any]) -> dict[str, Any]:
    data = pipeline['data']
    canonical_map = pipeline['semantic']['canonical_map']
    cohort = build_cohort_summary(
        data,
        canonical_map,
        age_range=filters.get('age_range'),
        genders=filters.get('gender'),
        diagnoses=filters.get('diagnosis'),
        treatments=filters.get('treatment'),
        smoking_statuses=filters.get('smoking_status'),
        cancer_stages=filters.get('cancer_stage'),
        risk_segments=filters.get('risk_segment'),
        comorbidity_filters=filters.get('comorbidity'),
    )
    monitoring = cohort_monitoring_over_time(data, canonical_map, cohort.get('cohort_frame'))
    return {
        'filters': dict(filters),
        'cohort': cohort,
        'monitoring': monitoring,
    }


def warm_healthcare_visualization_payloads(session_state: dict[str, Any], pipeline: dict[str, Any]) -> None:
    warmed = session_state.setdefault('visualization_warm_cache', set())
    dataset_token = _dataset_cache_token(pipeline)
    if dataset_token in warmed:
        return
    trend_key = build_visual_cache_key(pipeline, 'trend_analysis')
    cohort_key = build_visual_cache_key(pipeline, 'cohort_analysis', extra=_serialize_filters({}))
    if get_cached_visual_payload(session_state, trend_key) is None:
        store_visual_payload_for_pipeline(session_state, pipeline, trend_key, build_trend_analysis_payload(pipeline))
    if get_cached_visual_payload(session_state, cohort_key) is None:
        store_visual_payload_for_pipeline(session_state, pipeline, cohort_key, build_cohort_analysis_payload(pipeline, {}))
    warmed.add(dataset_token)


@dataclass
class DebouncedFilterState:
    applied_filters: dict[str, Any]
    pending: bool
    remaining_seconds: float


def resolve_debounced_filters(
    session_state: dict[str, Any],
    key: str,
    current_filters: dict[str, Any],
    debounce_seconds: float = 0.35,
) -> DebouncedFilterState:
    registry = session_state.setdefault('visualization_filter_debounce', {})
    entry = registry.setdefault(
        key,
        {
            'applied_serialized': None,
            'applied_filters': {},
            'current_serialized': None,
            'changed_at': 0.0,
        },
    )
    current_serialized = _serialize_filters(current_filters)
    now = time.monotonic()

    if entry['applied_serialized'] is None:
        entry['applied_serialized'] = current_serialized
        entry['applied_filters'] = dict(current_filters)
        entry['current_serialized'] = current_serialized
        entry['changed_at'] = now
        return DebouncedFilterState(dict(current_filters), False, 0.0)

    if current_serialized != entry['current_serialized']:
        entry['current_serialized'] = current_serialized
        entry['changed_at'] = now

    if current_serialized == entry['applied_serialized']:
        return DebouncedFilterState(dict(entry['applied_filters']), False, 0.0)

    elapsed = now - float(entry['changed_at'])
    if elapsed >= debounce_seconds:
        entry['applied_serialized'] = current_serialized
        entry['applied_filters'] = dict(current_filters)
        return DebouncedFilterState(dict(current_filters), False, 0.0)

    return DebouncedFilterState(
        dict(entry['applied_filters']),
        True,
        max(debounce_seconds - elapsed, 0.0),
    )


def record_tab_render_metric(session_state: dict[str, Any], section_name: str, duration_seconds: float) -> dict[str, Any]:
    metrics = session_state.setdefault('tab_render_metrics', {})
    metric = {
        'duration_seconds': float(duration_seconds),
        'duration_ms': int(round(duration_seconds * 1000.0)),
        'meets_target': bool(duration_seconds < 0.5),
    }
    metrics[section_name] = metric
    return metric
