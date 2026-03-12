from __future__ import annotations

from dataclasses import dataclass
import warnings

import pandas as pd
from pandas.api.types import is_bool_dtype, is_datetime64_any_dtype, is_numeric_dtype


ID_HINTS = {'id', 'member_id', 'patient_id', 'encounter_id', 'claim_id', 'record_id', 'provider_id'}
DATE_HINTS = {'date', 'time', 'month', 'year', 'admit', 'discharge', 'service'}
BOOLEAN_VALUES = {'0', '1', 'true', 'false', 'yes', 'no', 'y', 'n'}


@dataclass
class StructureSummary:
    detection_table: pd.DataFrame
    numeric_columns: list[str]
    date_columns: list[str]
    categorical_columns: list[str]
    text_columns: list[str]
    identifier_columns: list[str]
    boolean_columns: list[str]
    default_date_column: str | None
    confidence_score: float


def _sample(series: pd.Series, limit: int = 400) -> pd.Series:
    non_null = series.dropna()
    return non_null.head(limit)


def _date_ratio(series: pd.Series) -> float:
    sample = _sample(series)
    if sample.empty:
        return 0.0
    text = sample.astype(str).str.strip()
    # Skip obviously non-date content before attempting flexible parsing.
    likely_date = text.str.contains(r'\d{1,4}[-/]\d{1,2}[-/]\d{1,4}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|[A-Za-z]{3,9}\s+\d{1,2}', regex=True, na=False)
    if likely_date.mean() < 0.2:
        return 0.0
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        parsed = pd.to_datetime(text, errors='coerce')
    return float(parsed.notna().mean())


def _numeric_ratio(series: pd.Series) -> float:
    sample = _sample(series)
    if sample.empty:
        return 0.0
    parsed = pd.to_numeric(sample.astype(str).str.replace(',', ''), errors='coerce')
    return float(parsed.notna().mean())


def _boolean_ratio(series: pd.Series) -> float:
    sample = _sample(series).astype(str).str.strip().str.lower()
    if sample.empty:
        return 0.0
    return float(sample.isin(BOOLEAN_VALUES).mean())


def _avg_length(series: pd.Series) -> float:
    sample = _sample(series).astype(str)
    if sample.empty:
        return 0.0
    return float(sample.str.len().mean())


def detect_structure(data: pd.DataFrame) -> StructureSummary:
    rows: list[dict[str, object]] = []
    numeric_columns: list[str] = []
    date_columns: list[str] = []
    categorical_columns: list[str] = []
    text_columns: list[str] = []
    identifier_columns: list[str] = []
    boolean_columns: list[str] = []

    for column in data.columns:
        series = data[column]
        non_null_count = int(series.notna().sum())
        null_count = int(series.isna().sum())
        unique_count = int(series.nunique(dropna=True))
        uniqueness_ratio = unique_count / max(non_null_count, 1)
        date_ratio = _date_ratio(series)
        numeric_ratio = _numeric_ratio(series)
        boolean_ratio = _boolean_ratio(series)
        average_length = _avg_length(series)
        name = column.lower()
        looks_identifier = any(hint in name for hint in ID_HINTS) or (uniqueness_ratio >= 0.92 and unique_count > 20 and numeric_ratio < 0.98)
        evidence: list[str] = []

        if is_bool_dtype(series) or boolean_ratio >= 0.95:
            inferred_type = 'boolean'
            confidence = 0.95
            boolean_columns.append(column)
            evidence.append('binary-like values')
        elif is_datetime64_any_dtype(series) or date_ratio >= 0.8 or (any(token in name for token in DATE_HINTS) and date_ratio >= 0.45):
            inferred_type = 'datetime'
            confidence = max(date_ratio, 0.7)
            date_columns.append(column)
            evidence.append('parseable date values')
        elif looks_identifier and unique_count > 1:
            inferred_type = 'identifier'
            confidence = 0.88
            identifier_columns.append(column)
            evidence.append('high uniqueness consistent with an identifier')
        elif is_numeric_dtype(series) or numeric_ratio >= 0.9:
            inferred_type = 'numeric'
            confidence = max(numeric_ratio, 0.75)
            numeric_columns.append(column)
            evidence.append('reliable numeric conversion')
        elif unique_count <= max(20, int(len(data) * 0.15)) or uniqueness_ratio <= 0.25:
            inferred_type = 'categorical'
            confidence = 0.8
            categorical_columns.append(column)
            evidence.append('low-to-moderate cardinality')
        else:
            inferred_type = 'text'
            confidence = 0.72 if average_length >= 18 else 0.62
            text_columns.append(column)
            evidence.append('freeform or high-variability text')

        rows.append({
            'column_name': column,
            'inferred_type': inferred_type,
            'confidence_score': round(float(confidence), 3),
            'evidence': '; '.join(evidence),
            'non_null_count': non_null_count,
            'null_count': null_count,
            'null_percentage': null_count / max(len(data), 1),
            'unique_count': unique_count,
            'uniqueness_percentage': uniqueness_ratio,
            'average_string_length': average_length,
            'numeric_parse_ratio': numeric_ratio,
            'date_parse_ratio': date_ratio,
        })

    detection_table = pd.DataFrame(rows)
    default_date_column = None
    if not detection_table.empty and date_columns:
        best_dates = detection_table[detection_table['column_name'].isin(date_columns)].sort_values(['date_parse_ratio', 'confidence_score'], ascending=False)
        default_date_column = str(best_dates.iloc[0]['column_name'])

    confidence_score = float(detection_table['confidence_score'].mean()) if not detection_table.empty else 0.0
    return StructureSummary(
        detection_table=detection_table,
        numeric_columns=numeric_columns,
        date_columns=date_columns,
        categorical_columns=categorical_columns,
        text_columns=text_columns,
        identifier_columns=identifier_columns,
        boolean_columns=boolean_columns,
        default_date_column=default_date_column,
        confidence_score=confidence_score,
    )

