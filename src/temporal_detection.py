from __future__ import annotations

from dataclasses import dataclass
import re
import warnings

import pandas as pd


TEMPORAL_HINT_PATTERN = re.compile(
    r'(date|datetime|timestamp|event|admission|admit|visit|service|year|month|day)',
    re.IGNORECASE,
)


@dataclass
class TemporalContext:
    available: bool
    date_columns: list[str]
    default_date_column: str | None
    synthetic_date_created: bool
    source_columns: list[str]
    note: str


def _sample(series: pd.Series, limit: int = 400) -> pd.Series:
    return series.dropna().head(limit)


def _date_ratio(series: pd.Series) -> float:
    sample = _sample(series)
    if sample.empty:
        return 0.0
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        parsed = pd.to_datetime(sample.astype(str).str.strip(), errors='coerce')
    return float(parsed.notna().mean())


def _year_ratio(series: pd.Series) -> float:
    sample = _sample(series)
    if sample.empty:
        return 0.0
    numeric = pd.to_numeric(sample, errors='coerce')
    valid = numeric.between(1900, 2100, inclusive='both')
    return float(valid.mean())


def _bounded_ratio(series: pd.Series, low: int, high: int) -> float:
    sample = _sample(series)
    if sample.empty:
        return 0.0
    numeric = pd.to_numeric(sample, errors='coerce')
    return float(numeric.between(low, high, inclusive='both').mean())


def _looks_like_year_only(series: pd.Series) -> bool:
    sample = _sample(series).astype(str).str.strip()
    if sample.empty:
        return False
    numeric = pd.to_numeric(sample, errors='coerce')
    if numeric.isna().all():
        return False
    year_like = numeric.between(1900, 2100, inclusive='both').mean() >= 0.9
    four_digit = sample.str.fullmatch(r'\d{4}').mean() >= 0.9
    return bool(year_like and four_digit)


def _looks_like_month_only(series: pd.Series) -> bool:
    sample = _sample(series).astype(str).str.strip()
    if sample.empty:
        return False
    numeric = pd.to_numeric(sample, errors='coerce')
    if numeric.isna().all():
        return False
    return bool(numeric.between(1, 12, inclusive='both').mean() >= 0.9 and sample.str.fullmatch(r'\d{1,2}').mean() >= 0.9)


def _looks_like_day_only(series: pd.Series) -> bool:
    sample = _sample(series).astype(str).str.strip()
    if sample.empty:
        return False
    numeric = pd.to_numeric(sample, errors='coerce')
    if numeric.isna().all():
        return False
    return bool(numeric.between(1, 31, inclusive='both').mean() >= 0.9 and sample.str.fullmatch(r'\d{1,2}').mean() >= 0.9)


def detect_temporal_context(data: pd.DataFrame) -> TemporalContext:
    date_candidates: list[tuple[str, float]] = []
    for column in data.columns:
        series = data[column]
        lowered = column.lower()
        if ('year' in lowered and _looks_like_year_only(series)) or _looks_like_year_only(series):
            continue
        if 'month' in lowered and _looks_like_month_only(series):
            continue
        if 'day' in lowered and _looks_like_day_only(series):
            continue
        ratio = 1.0 if pd.api.types.is_datetime64_any_dtype(series) else _date_ratio(series)
        if ratio >= 0.75 or (TEMPORAL_HINT_PATTERN.search(column) and ratio >= 0.45):
            date_candidates.append((column, ratio))

    if date_candidates:
        date_candidates = sorted(date_candidates, key=lambda item: item[1], reverse=True)
        return TemporalContext(
            available=True,
            date_columns=[column for column, _ in date_candidates],
            default_date_column=date_candidates[0][0],
            synthetic_date_created=False,
            source_columns=[],
            note='Using a detected date-like field for temporal analysis.',
        )

    year_candidates = [
        column
        for column in data.columns
        if ('year' in column.lower() or TEMPORAL_HINT_PATTERN.search(column))
        and _year_ratio(data[column]) >= 0.75
    ]
    month_candidates = [column for column in data.columns if 'month' in column.lower() and _bounded_ratio(data[column], 1, 12) >= 0.75]
    day_candidates = [column for column in data.columns if 'day' in column.lower() and _bounded_ratio(data[column], 1, 31) >= 0.75]

    if year_candidates:
        source_columns = [year_candidates[0]]
        if month_candidates:
            source_columns.append(month_candidates[0])
        if day_candidates:
            source_columns.append(day_candidates[0])
        return TemporalContext(
            available=True,
            date_columns=['event_date'],
            default_date_column='event_date',
            synthetic_date_created=True,
            source_columns=source_columns,
            note='A synthetic event_date can be derived from partial temporal fields for trend-style analysis.',
        )

    return TemporalContext(
        available=False,
        date_columns=[],
        default_date_column=None,
        synthetic_date_created=False,
        source_columns=[],
        note='No reliable date field was detected.',
    )


def augment_temporal_fields(data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    context = detect_temporal_context(data)
    if not context.synthetic_date_created or 'event_date' in data.columns:
        return data, {
            'available': context.available,
            'date_columns': context.date_columns,
            'default_date_column': context.default_date_column,
            'synthetic_date_created': context.synthetic_date_created,
            'source_columns': context.source_columns,
            'note': context.note,
        }

    augmented = data.copy()
    year_col = context.source_columns[0]
    year_values = pd.to_numeric(augmented[year_col], errors='coerce')

    month_values = pd.Series(1, index=augmented.index, dtype='Int64')
    day_values = pd.Series(1, index=augmented.index, dtype='Int64')

    if len(context.source_columns) >= 2:
        month_values = pd.to_numeric(augmented[context.source_columns[1]], errors='coerce').round().astype('Int64')
        month_values = month_values.where(month_values.between(1, 12), 1)
    if len(context.source_columns) >= 3:
        day_values = pd.to_numeric(augmented[context.source_columns[2]], errors='coerce').round().astype('Int64')
        day_values = day_values.where(day_values.between(1, 28), 1)

    year_values = year_values.round().astype('Int64')
    valid_years = year_values.where(year_values.between(1900, 2100))
    augmented['event_date'] = pd.to_datetime(
        {
            'year': valid_years,
            'month': month_values,
            'day': day_values,
        },
        errors='coerce',
    )

    return augmented, {
        'available': True,
        'date_columns': ['event_date'],
        'default_date_column': 'event_date',
        'synthetic_date_created': True,
        'source_columns': context.source_columns,
        'note': f"Synthetic event_date was generated from {', '.join(context.source_columns)} for temporal analysis. Use a true encounter or event date when available.",
    }
