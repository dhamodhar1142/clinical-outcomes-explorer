from __future__ import annotations

import pandas as pd

from src.schema_detection import StructureSummary


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

def analysis_sample_info(data: pd.DataFrame, sample_size: int = 10000, large_sample_size: int = 20000, quality_sample_size: int = 15000, quality_large_sample_size: int = 25000) -> dict[str, int | bool]:
    profile_sample_rows = len(_analysis_sample(data, sample_size=sample_size, large_sample_size=large_sample_size))
    quality_sample_rows = len(_analysis_sample(data, sample_size=quality_sample_size, large_sample_size=quality_large_sample_size))
    total_rows = len(data)
    return {
        'total_rows': int(total_rows),
        'profile_sample_rows': int(profile_sample_rows),
        'quality_sample_rows': int(quality_sample_rows),
        'sampling_applied': profile_sample_rows < total_rows or quality_sample_rows < total_rows,
        'very_large_dataset': total_rows > 100000,
    }

def build_dataset_overview(data: pd.DataFrame, memory_mb: float) -> dict[str, float | int]:
    return {
        'rows': int(len(data)),
        'columns': int(len(data.columns)),
        'duplicate_rows': int(data.duplicated().sum()),
        'missing_values': int(data.isna().sum().sum()),
        'memory_mb': float(memory_mb),
    }


def _sample_values(series: pd.Series, limit: int = 3) -> str:
    sample = series.dropna().astype(str).head(limit).tolist()
    return ', '.join(sample) if sample else '-'


def _top_values(series: pd.Series, limit: int = 5) -> str:
    counts = series.fillna('Missing').astype(str).value_counts(dropna=False).head(limit)
    return '; '.join(f'{idx} ({count})' for idx, count in counts.items())


def build_field_profile(data: pd.DataFrame, structure: StructureSummary, sample_size: int = 10000) -> pd.DataFrame:
    analysis_df = _analysis_sample(data, sample_size=sample_size, large_sample_size=20000)
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
    return pd.DataFrame(rows)


def build_quality_checks(data: pd.DataFrame, structure: StructureSummary, field_profile: pd.DataFrame) -> dict[str, pd.DataFrame | int]:
    analysis_df = _analysis_sample(data, sample_size=15000, large_sample_size=25000)
    high_missing = field_profile[field_profile['null_percentage'] >= 0.4][['column_name', 'null_percentage']].sort_values('null_percentage', ascending=False)
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

    return {
        'quality_score': quality_score,
        'high_missing': high_missing,
        'near_constant': near_constant,
        'empty_columns': empty_columns,
        'suspicious_numeric_text': pd.DataFrame(suspicious_numeric_text_rows),
        'mixed_type_suspicions': pd.DataFrame(mixed_type_rows),
        'duplicate_identifiers': pd.DataFrame(duplicate_identifier_rows),
        'numeric_outliers': pd.DataFrame(outlier_rows),
    }


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




