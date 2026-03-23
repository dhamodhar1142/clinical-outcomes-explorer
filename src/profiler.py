from __future__ import annotations

import gc
import hashlib
from pathlib import Path
import pickle
import shutil
from time import perf_counter
from typing import Any

import pandas as pd

from src.runtime_paths import data_path
from src.schema_detection import StructureSummary, detect_structure


PROFILE_CACHE_ROOT = data_path('cache', 'profiles')
PROFILE_CACHE_VERSION = 'v3'


def _cardinality_label(unique_count: int, non_null_count: int, inferred_type: str) -> str:
    if non_null_count == 0:
        return 'Empty'
    ratio = unique_count / max(non_null_count, 1)
    if inferred_type == 'identifier' or ratio >= 0.85:
        return 'High'
    if ratio >= 0.20:
        return 'Medium'
    return 'Low'


def _variance_indicator(std_value: float | None, mean_value: float | None) -> str:
    if std_value is None or mean_value is None:
        return 'Not applicable'
    baseline = abs(mean_value) if abs(mean_value) > 1e-9 else 1.0
    ratio = abs(std_value) / baseline
    if ratio >= 1.0:
        return 'High variance'
    if ratio >= 0.35:
        return 'Moderate variance'
    return 'Low variance'


def _analysis_sample(data: pd.DataFrame, sample_size: int = 10000, large_sample_size: int = 20000) -> pd.DataFrame:
    if len(data) > 100000:
        return data.sample(min(len(data), large_sample_size), random_state=42)
    if len(data) > sample_size:
        return data.sample(min(len(data), sample_size), random_state=42)
    return data


def _sampling_plan(sampling_plan: dict[str, int] | None = None) -> dict[str, int]:
    plan = dict(sampling_plan or {})
    return {
        'profile_sample_rows': int(plan.get('profile_sample_rows', 10_000)),
        'profile_large_sample_rows': int(plan.get('profile_large_sample_rows', 20_000)),
        'quality_sample_rows': int(plan.get('quality_sample_rows', 15_000)),
        'quality_large_sample_rows': int(plan.get('quality_large_sample_rows', 25_000)),
        'very_large_dataset_rows': int(plan.get('very_large_dataset_rows', 100_000)),
        'profile_name': str(plan.get('profile_name', 'Standard')),
    }


def default_profile_cache_metrics() -> dict[str, Any]:
    return {
        'requests': 0,
        'hits': 0,
        'misses': 0,
        'structure_requests': 0,
        'structure_hits': 0,
        'field_profile_requests': 0,
        'field_profile_hits': 0,
        'quality_requests': 0,
        'quality_hits': 0,
        'saved_ms': 0.0,
        'last_cache_name': '',
        'last_hit': False,
        'last_latency_ms': 0.0,
        'last_saved_ms': 0.0,
        'last_dataset_version_hash': '',
        'last_config_hash': '',
        'last_cache_key': '',
        'last_generated_ms': 0.0,
        'clear_count': 0,
    }


def _dataset_version_hash(data: pd.DataFrame) -> str:
    return str(data.attrs.get('dataset_cache_key', '')).strip()


def _config_hash(plan: dict[str, int]) -> str:
    raw_signature = (
        f"{plan['profile_name']}:"
        f"{plan['profile_sample_rows']}:"
        f"{plan['profile_large_sample_rows']}:"
        f"{plan['quality_sample_rows']}:"
        f"{plan['quality_large_sample_rows']}:"
        f"{plan['very_large_dataset_rows']}"
    )
    return hashlib.sha256(raw_signature.encode('utf-8')).hexdigest()[:16]


def _cache_key(data: pd.DataFrame, plan: dict[str, int], suffix: str) -> str | None:
    dataset_cache_key = _dataset_version_hash(data)
    if not dataset_cache_key:
        return None
    return f"{PROFILE_CACHE_VERSION}-{dataset_cache_key}-{_config_hash(plan)}-{suffix}"


def _cache_path(cache_key: str | None) -> Path | None:
    if not cache_key:
        return None
    PROFILE_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    return PROFILE_CACHE_ROOT / f'{cache_key}.pkl'


def _read_cached_value(cache_key: str | None):
    path = _cache_path(cache_key)
    if path is None or not path.exists():
        return None
    try:
        with path.open('rb') as handle:
            cached = pickle.load(handle)
            if isinstance(cached, dict) and 'value' in cached:
                return cached
            return {
                'value': cached,
                'generation_latency_ms': 0.0,
                'cache_version': PROFILE_CACHE_VERSION,
                'dataset_version_hash': '',
                'config_hash': '',
                'cache_key': cache_key or '',
            }
    except Exception:
        return None


def _write_cached_value(
    cache_key: str | None,
    value,
    *,
    generation_latency_ms: float = 0.0,
    dataset_version_hash: str = '',
    config_hash: str = '',
) -> None:
    path = _cache_path(cache_key)
    if path is None:
        return
    try:
        with path.open('wb') as handle:
            pickle.dump(
                {
                    'value': value,
                    'generation_latency_ms': float(generation_latency_ms),
                    'cache_version': PROFILE_CACHE_VERSION,
                    'dataset_version_hash': dataset_version_hash,
                    'config_hash': config_hash,
                    'cache_key': cache_key or '',
                },
                handle,
            )
    except Exception:
        return


def _cached_payload(cached) -> tuple[Any, dict[str, Any]]:
    if isinstance(cached, dict) and 'value' in cached:
        return cached.get('value'), {
            'generation_latency_ms': float(cached.get('generation_latency_ms', 0.0) or 0.0),
            'dataset_version_hash': str(cached.get('dataset_version_hash', '') or ''),
            'config_hash': str(cached.get('config_hash', '') or ''),
            'cache_key': str(cached.get('cache_key', '') or ''),
        }
    return cached, {
        'generation_latency_ms': 0.0,
        'dataset_version_hash': '',
        'config_hash': '',
        'cache_key': '',
    }


def _record_cache_metric(
    cache_metrics: dict[str, Any] | None,
    *,
    cache_name: str,
    hit: bool,
    latency_ms: float,
    saved_ms: float,
    dataset_version_hash: str,
    config_hash: str,
    cache_key: str,
    generated_ms: float,
) -> None:
    if cache_metrics is None:
        return
    defaults = default_profile_cache_metrics()
    for key, value in defaults.items():
        cache_metrics.setdefault(key, value)
    cache_metrics['requests'] += 1
    cache_metrics['hits'] += int(hit)
    cache_metrics['misses'] += int(not hit)
    cache_metrics[f'{cache_name}_requests'] += 1
    cache_metrics[f'{cache_name}_hits'] += int(hit)
    cache_metrics['saved_ms'] = float(cache_metrics.get('saved_ms', 0.0)) + float(saved_ms)
    cache_metrics['last_cache_name'] = cache_name
    cache_metrics['last_hit'] = bool(hit)
    cache_metrics['last_latency_ms'] = float(latency_ms)
    cache_metrics['last_saved_ms'] = float(saved_ms)
    cache_metrics['last_dataset_version_hash'] = dataset_version_hash
    cache_metrics['last_config_hash'] = config_hash
    cache_metrics['last_cache_key'] = cache_key
    cache_metrics['last_generated_ms'] = float(generated_ms)


def clear_profile_cache(cache_metrics: dict[str, Any] | None = None) -> dict[str, Any]:
    if PROFILE_CACHE_ROOT.exists():
        shutil.rmtree(PROFILE_CACHE_ROOT, ignore_errors=True)
    if cache_metrics is None:
        return default_profile_cache_metrics()
    clear_count = int(cache_metrics.get('clear_count', 0)) + 1
    cache_metrics.clear()
    cache_metrics.update(default_profile_cache_metrics())
    cache_metrics['clear_count'] = clear_count
    return cache_metrics


def build_profile_cache_summary(cache_metrics: dict[str, Any] | None) -> dict[str, Any]:
    metrics = default_profile_cache_metrics()
    metrics.update(cache_metrics or {})
    requests = int(metrics.get('requests', 0))
    hits = int(metrics.get('hits', 0))
    hit_rate = hits / requests if requests else 0.0
    structure_requests = int(metrics.get('structure_requests', 0))
    structure_hits = int(metrics.get('structure_hits', 0))
    field_profile_requests = int(metrics.get('field_profile_requests', 0))
    field_profile_hits = int(metrics.get('field_profile_hits', 0))
    summary_cards = [
        {'label': 'Profile Cache Hit Rate', 'value': f'{hit_rate:.1%}' if requests else 'No requests yet'},
        {'label': 'Cache Requests', 'value': f'{requests:,}'},
        {'label': 'Saved Latency', 'value': f"{float(metrics.get('saved_ms', 0.0)):.0f} ms"},
        {'label': 'Dataset Version Hash', 'value': str(metrics.get('last_dataset_version_hash', '') or 'Not available')[:12]},
    ]
    settings_table = pd.DataFrame(
        [
            {
                'cache_area': 'Column Detection',
                'requests': structure_requests,
                'hit_rate': f"{(structure_hits / structure_requests):.1%}" if structure_requests else 'No requests yet',
                'last_latency_ms': float(metrics.get('last_latency_ms', 0.0)) if metrics.get('last_cache_name') == 'structure' else pd.NA,
                'cache_key': str(metrics.get('last_cache_key', '') or 'Not available') if metrics.get('last_cache_name') == 'structure' else 'See most recent matching request',
            },
            {
                'cache_area': 'Field Profiling',
                'requests': field_profile_requests,
                'hit_rate': f"{(field_profile_hits / field_profile_requests):.1%}" if field_profile_requests else 'No requests yet',
                'last_latency_ms': float(metrics.get('last_latency_ms', 0.0)) if metrics.get('last_cache_name') == 'field_profile' else pd.NA,
                'cache_key': str(metrics.get('last_cache_key', '') or 'Not available') if metrics.get('last_cache_name') == 'field_profile' else 'See most recent matching request',
            },
        ]
    )
    return {
        'summary_cards': summary_cards,
        'settings_table': settings_table,
        'notes': [
            'Cache invalidation occurs automatically when the dataset version hash or active large-dataset profile changes.',
            'Latency savings are session-based estimates comparing cache-hit retrieval time with the original cached generation time.',
        ],
        'hit_rate': hit_rate,
    }

def analysis_sample_info(data: pd.DataFrame, sample_size: int = 10000, large_sample_size: int = 20000, quality_sample_size: int = 15000, quality_large_sample_size: int = 25000, sampling_plan: dict[str, int] | None = None) -> dict[str, int | bool]:
    plan = _sampling_plan(sampling_plan)
    profile_sample_rows = len(_analysis_sample(data, sample_size=plan['profile_sample_rows'], large_sample_size=plan['profile_large_sample_rows']))
    quality_sample_rows = len(_analysis_sample(data, sample_size=plan['quality_sample_rows'], large_sample_size=plan['quality_large_sample_rows']))
    total_rows = int(data.attrs.get('source_row_count', len(data)))
    return {
        'total_rows': int(total_rows),
        'analyzed_rows': int(len(data)),
        'profile_sample_rows': int(profile_sample_rows),
        'quality_sample_rows': int(quality_sample_rows),
        'sampling_applied': profile_sample_rows < total_rows or quality_sample_rows < total_rows,
        'very_large_dataset': total_rows > plan['very_large_dataset_rows'],
        'sampling_plan_name': str(plan.get('profile_name', 'Standard')),
        'sampling_mode': str(data.attrs.get('sampling_mode', 'full')),
        'ingestion_strategy': str(data.attrs.get('ingestion_strategy', 'standard')),
    }


def build_structure_profile_bundle(
    data: pd.DataFrame,
    *,
    sampling_plan: dict[str, int] | None = None,
    cache_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    plan = _sampling_plan(sampling_plan)
    dataset_version_hash = _dataset_version_hash(data)
    config_hash = _config_hash(plan)
    structure_cache_key = _cache_key(data, plan, 'structure')
    structure_start = perf_counter()
    cached_structure = _read_cached_value(structure_cache_key)
    if cached_structure is not None:
        structure, structure_meta = _cached_payload(cached_structure)
        structure_latency_ms = (perf_counter() - structure_start) * 1000
        _record_cache_metric(
            cache_metrics,
            cache_name='structure',
            hit=True,
            latency_ms=structure_latency_ms,
            saved_ms=max(float(structure_meta.get('generation_latency_ms', 0.0)) - structure_latency_ms, 0.0),
            dataset_version_hash=dataset_version_hash or str(structure_meta.get('dataset_version_hash', '')),
            config_hash=config_hash or str(structure_meta.get('config_hash', '')),
            cache_key=structure_cache_key or '',
            generated_ms=float(structure_meta.get('generation_latency_ms', 0.0)),
        )
    else:
        structure = detect_structure(data)
        structure_latency_ms = (perf_counter() - structure_start) * 1000
        _write_cached_value(
            structure_cache_key,
            structure,
            generation_latency_ms=structure_latency_ms,
            dataset_version_hash=dataset_version_hash,
            config_hash=config_hash,
        )
        _record_cache_metric(
            cache_metrics,
            cache_name='structure',
            hit=False,
            latency_ms=structure_latency_ms,
            saved_ms=0.0,
            dataset_version_hash=dataset_version_hash,
            config_hash=config_hash,
            cache_key=structure_cache_key or '',
            generated_ms=structure_latency_ms,
        )
    field_profile = build_field_profile(
        data,
        structure,
        sampling_plan=plan,
        cache_metrics=cache_metrics,
    )
    return {
        'structure': structure,
        'field_profile': field_profile,
        'dataset_version_hash': dataset_version_hash,
        'config_hash': config_hash,
        'structure_cache_key': structure_cache_key or '',
        'field_profile_cache_key': _cache_key(data, plan, 'field_profile') or '',
    }


def build_dataset_overview(data: pd.DataFrame, memory_mb: float) -> dict[str, float | int]:
    return {
        'rows': int(data.attrs.get('source_row_count', len(data))),
        'analyzed_rows': int(len(data)),
        'columns': int(len(data.columns)),
        'duplicate_rows': int(data.duplicated().sum()),
        'missing_values': int(data.isna().sum().sum()),
        'memory_mb': float(memory_mb),
        'sampling_mode': str(data.attrs.get('sampling_mode', 'full')),
        'ingestion_strategy': str(data.attrs.get('ingestion_strategy', 'standard')),
    }


def _sample_values(series: pd.Series, limit: int = 3) -> str:
    sample = series.dropna().astype(str).head(limit).tolist()
    return ', '.join(sample) if sample else '-'


def _top_values(series: pd.Series, limit: int = 5) -> str:
    counts = series.fillna('Missing').astype(str).value_counts(dropna=False).head(limit)
    return '; '.join(f'{idx} ({count})' for idx, count in counts.items())


def build_field_profile(
    data: pd.DataFrame,
    structure: StructureSummary,
    sample_size: int = 10000,
    sampling_plan: dict[str, int] | None = None,
    cache_metrics: dict[str, Any] | None = None,
) -> pd.DataFrame:
    plan = _sampling_plan(sampling_plan)
    cache_key = _cache_key(data, plan, 'field_profile')
    dataset_version_hash = _dataset_version_hash(data)
    config_hash = _config_hash(plan)
    started = perf_counter()
    cached = _read_cached_value(cache_key)
    if cached is not None:
        cached_value, cached_meta = _cached_payload(cached)
        if isinstance(cached_value, pd.DataFrame):
            latency_ms = (perf_counter() - started) * 1000
            _record_cache_metric(
                cache_metrics,
                cache_name='field_profile',
                hit=True,
                latency_ms=latency_ms,
                saved_ms=max(float(cached_meta.get('generation_latency_ms', 0.0)) - latency_ms, 0.0),
                dataset_version_hash=dataset_version_hash or str(cached_meta.get('dataset_version_hash', '')),
                config_hash=config_hash or str(cached_meta.get('config_hash', '')),
                cache_key=cache_key or '',
                generated_ms=float(cached_meta.get('generation_latency_ms', 0.0)),
            )
            return cached_value.copy()
    analysis_df = _analysis_sample(data, sample_size=plan['profile_sample_rows'], large_sample_size=plan['profile_large_sample_rows'])
    rows: list[dict[str, object]] = []
    detection_lookup = structure.detection_table.set_index('column_name').to_dict('index') if not structure.detection_table.empty else {}

    for column in data.columns:
        series = data[column]
        analysis_series = analysis_df[column] if column in analysis_df.columns else series
        sample_series = analysis_df[column]
        detected = detection_lookup.get(column, {})
        inferred_type = detected.get('inferred_type', 'unknown')
        non_null_count = int(series.notna().sum())
        unique_count = int(series.nunique(dropna=True))
        uniqueness_percentage = float(unique_count / max(non_null_count, 1))
        row = {
            'column_name': column,
            'inferred_type': inferred_type,
            'non_null_count': non_null_count,
            'null_count': int(series.isna().sum()),
            'null_percentage': float(series.isna().mean()),
            'unique_count': unique_count,
            'uniqueness_percentage': uniqueness_percentage,
            'cardinality_category': _cardinality_label(unique_count, non_null_count, inferred_type),
            'sample_values': _sample_values(sample_series),
            'top_values': _top_values(sample_series),
            'average_string_length': float(sample_series.dropna().astype(str).str.len().mean()) if sample_series.dropna().shape[0] else 0.0,
            'looks_potentially_categorical': inferred_type in {'categorical', 'boolean'} or (inferred_type == 'text' and unique_count <= max(25, int(len(sample_series) * 0.1))),
            'is_high_cardinality_identifier': inferred_type == 'identifier' or uniqueness_percentage >= 0.85,
            'is_near_constant': unique_count <= 1 or uniqueness_percentage <= 0.02,
            'has_numeric_outlier_signal': False,
            'min_value': None,
            'max_value': None,
            'mean_value': None,
            'median_value': None,
            'std_value': None,
            'variance_indicator': 'Not applicable',
            'outlier_count': 0,
            'min_date': None,
            'max_date': None,
        }
        if inferred_type == 'numeric':
            numeric = pd.to_numeric(sample_series, errors='coerce').dropna()
            if not numeric.empty:
                std_value = float(numeric.std()) if len(numeric) > 1 else 0.0
                q1 = numeric.quantile(0.25)
                q3 = numeric.quantile(0.75)
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                outlier_count = int(((numeric < lower) | (numeric > upper)).sum()) if len(numeric) >= 8 and iqr >= 0 else 0
                mean_value = float(numeric.mean())
                row.update({
                    'min_value': float(numeric.min()),
                    'max_value': float(numeric.max()),
                    'mean_value': mean_value,
                    'median_value': float(numeric.median()),
                    'std_value': std_value,
                    'variance_indicator': _variance_indicator(std_value, mean_value),
                    'outlier_count': outlier_count,
                    'has_numeric_outlier_signal': outlier_count > 0,
                })
        if inferred_type == 'datetime':
            dt = pd.to_datetime(sample_series, errors='coerce').dropna()
            if not dt.empty:
                row.update({'min_date': str(dt.min()), 'max_date': str(dt.max())})
        rows.append(row)
    result = pd.DataFrame(rows)
    generation_latency_ms = (perf_counter() - started) * 1000
    _write_cached_value(
        cache_key,
        result,
        generation_latency_ms=generation_latency_ms,
        dataset_version_hash=dataset_version_hash,
        config_hash=config_hash,
    )
    _record_cache_metric(
        cache_metrics,
        cache_name='field_profile',
        hit=False,
        latency_ms=generation_latency_ms,
        saved_ms=0.0,
        dataset_version_hash=dataset_version_hash,
        config_hash=config_hash,
        cache_key=cache_key or '',
        generated_ms=generation_latency_ms,
    )
    del analysis_df
    gc.collect()
    return result


def build_quality_checks(data: pd.DataFrame, structure: StructureSummary, field_profile: pd.DataFrame, sampling_plan: dict[str, int] | None = None) -> dict[str, pd.DataFrame | int]:
    plan = _sampling_plan(sampling_plan)
    cache_key = _cache_key(data, plan, 'quality_checks')
    dataset_version_hash = _dataset_version_hash(data)
    config_hash = _config_hash(plan)
    started = perf_counter()
    cached = _read_cached_value(cache_key)
    if cached is not None:
        cached_value, _ = _cached_payload(cached)
        if isinstance(cached_value, dict):
            _record_cache_metric(
                data.attrs.get('profile_cache_metrics'),
                cache_name='quality',
                hit=True,
                latency_ms=(perf_counter() - started) * 1000,
                saved_ms=0.0,
                dataset_version_hash=dataset_version_hash,
                config_hash=config_hash,
                cache_key=cache_key or '',
                generated_ms=0.0,
            )
            return cached_value
    analysis_df = _analysis_sample(data, sample_size=plan['quality_sample_rows'], large_sample_size=plan['quality_large_sample_rows'])
    helper_field_names = {
        str(name).strip()
        for name in data.attrs.get('helper_field_names', [])
        if str(name).strip()
    }
    high_missing_candidates = field_profile[field_profile['null_percentage'] >= 0.4][['column_name', 'null_percentage']].sort_values('null_percentage', ascending=False)
    helper_missing = high_missing_candidates[high_missing_candidates['column_name'].astype(str).isin(helper_field_names)].reset_index(drop=True)
    high_missing = high_missing_candidates[~high_missing_candidates['column_name'].astype(str).isin(helper_field_names)].reset_index(drop=True)
    near_constant = field_profile[(field_profile['uniqueness_percentage'] <= 0.02) & (field_profile['non_null_count'] > 0)][['column_name', 'uniqueness_percentage']]
    empty_columns = field_profile[field_profile['non_null_count'] == 0][['column_name']]

    suspicious_numeric_text_rows: list[dict[str, object]] = []
    mixed_type_rows: list[dict[str, object]] = []
    duplicate_identifier_rows: list[dict[str, object]] = []
    outlier_rows: list[dict[str, object]] = []

    for column in data.columns:
        series = data[column]
        analysis_series = analysis_df[column] if column in analysis_df.columns else series
        inferred_type = field_profile.loc[field_profile['column_name'] == column, 'inferred_type'].iloc[0]
        numeric_conversion = pd.to_numeric(analysis_series.astype(str).str.replace(',', ''), errors='coerce')
        numeric_ratio = float(numeric_conversion.notna().mean())
        if inferred_type in {'categorical', 'text'} and numeric_ratio >= 0.85:
            suspicious_numeric_text_rows.append({'column_name': column, 'numeric_parse_ratio': numeric_ratio})
        if 0.2 <= numeric_ratio <= 0.8 and inferred_type in {'categorical', 'text'}:
            mixed_type_rows.append({'column_name': column, 'numeric_parse_ratio': numeric_ratio})
        if inferred_type == 'identifier':
            duplicate_ratio = float(series.duplicated().mean())
            if duplicate_ratio >= 0.05:
                duplicate_identifier_rows.append({'column_name': column, 'duplicate_ratio': duplicate_ratio})
        if inferred_type == 'numeric':
            numeric = numeric_conversion.dropna()
            if len(numeric) >= 8:
                q1 = numeric.quantile(0.25)
                q3 = numeric.quantile(0.75)
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                count = int(((numeric < lower) | (numeric > upper)).sum())
                if count > 0:
                    outlier_rows.append({'column_name': column, 'outlier_count': count, 'outlier_percentage': count / max(len(numeric), 1)})

    issue_count = sum([
        len(high_missing),
        len(near_constant),
        len(empty_columns),
        len(suspicious_numeric_text_rows),
        len(mixed_type_rows),
        len(duplicate_identifier_rows),
    ])
    quality_score = max(100 - min(issue_count * 6, 45) - min(int(data.duplicated().mean() * 100), 20), 0)

    result = {
        'quality_score': quality_score,
        'high_missing': high_missing,
        'suppressed_helper_missing': helper_missing,
        'near_constant': near_constant,
        'empty_columns': empty_columns,
        'suspicious_numeric_text': pd.DataFrame(suspicious_numeric_text_rows),
        'mixed_type_suspicions': pd.DataFrame(mixed_type_rows),
        'duplicate_identifiers': pd.DataFrame(duplicate_identifier_rows),
        'numeric_outliers': pd.DataFrame(outlier_rows),
    }
    _write_cached_value(
        cache_key,
        result,
        generation_latency_ms=(perf_counter() - started) * 1000,
        dataset_version_hash=dataset_version_hash,
        config_hash=config_hash,
    )
    del analysis_df
    gc.collect()
    return result


def build_numeric_summary(field_profile: pd.DataFrame) -> pd.DataFrame:
    expected_columns = [
        'column_name',
        'min_value',
        'max_value',
        'mean_value',
        'median_value',
        'std_value',
        'variance_indicator',
        'outlier_count',
        'null_percentage',
    ]
    numeric = field_profile[field_profile['inferred_type'] == 'numeric'].copy()
    if numeric.empty:
        return pd.DataFrame(columns=expected_columns)

    default_values = {
        'column_name': pd.NA,
        'min_value': pd.NA,
        'max_value': pd.NA,
        'mean_value': pd.NA,
        'median_value': pd.NA,
        'std_value': pd.NA,
        'variance_indicator': 'Not available in current profiling sample',
        'outlier_count': 0,
        'null_percentage': pd.NA,
    }
    for column, default in default_values.items():
        if column not in numeric.columns:
            numeric[column] = default

    numeric = numeric[expected_columns]
    return numeric.sort_values('mean_value', ascending=False, na_position='last')




