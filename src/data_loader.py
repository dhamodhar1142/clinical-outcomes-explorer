from __future__ import annotations

import io
import re
from pathlib import Path

import pandas as pd


DATA_DIR = Path('data')
DEMO_DATASETS = {
    'Healthcare Operations Demo': {
        'path': DATA_DIR / 'synthetic_hospital_data.csv',
        'description': 'Encounter-level hospital operations data with age, diagnosis, cost, length of stay, and readmission outcomes.',
        'best_for': 'Healthcare operations, cost review, utilization, and readmission-style analysis.',
    },
    'Hospital Reporting Demo': {
        'path': DATA_DIR / 'synthetic_hospital_quality_reporting.csv',
        'description': 'Hospital quality reporting data with provider IDs, measure names, scores, denominators, and benchmark comparisons.',
        'best_for': 'Reporting readiness, provider/facility summaries, and trend-style review.',
    },
    'Generic Business Demo': {
        'path': DATA_DIR / 'synthetic_generic_operations_data.csv',
        'description': 'General business operations data for generic profiling, data quality review, and trend analysis.',
        'best_for': 'General-purpose schema-flexible analytics and product demos.',
    },
}


class DataLoadError(Exception):
    pass


def normalize_column_name(name: str) -> str:
    normalized = re.sub(r'[^a-zA-Z0-9]+', '_', str(name).strip().lower())
    return re.sub(r'_+', '_', normalized).strip('_') or 'unnamed_column'


def _make_unique(names: list[str]) -> list[str]:
    counts: dict[str, int] = {}
    unique_names: list[str] = []
    for name in names:
        count = counts.get(name, 0)
        unique_names.append(name if count == 0 else f'{name}_{count + 1}')
        counts[name] = count + 1
    return unique_names


def standardize_dataframe(data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    cleaned = data.copy()
    original_names = [str(col) for col in cleaned.columns]
    normalized_names = _make_unique([normalize_column_name(col) for col in original_names])
    cleaned.columns = normalized_names
    original_lookup = dict(zip(normalized_names, original_names))
    return cleaned, original_lookup


def estimate_memory_mb(data: pd.DataFrame) -> float:
    return float(data.memory_usage(deep=True).sum() / (1024 ** 2))


def read_csv_bytes(file_bytes: bytes) -> pd.DataFrame:
    encodings = ['utf-8', 'utf-8-sig', 'latin-1']
    last_error: Exception | None = None
    for encoding in encodings:
        for read_options in ({'encoding': encoding}, {'encoding': encoding, 'sep': None, 'engine': 'python'}):
            try:
                return pd.read_csv(io.BytesIO(file_bytes), **read_options)
            except Exception as error:
                last_error = error
    raise DataLoadError(f'Unable to parse the uploaded CSV file. {last_error}')


def list_excel_sheets(file_bytes: bytes, suffix: str = '.xlsx') -> list[str]:
    if suffix == '.xls':
        raise DataLoadError('Legacy .xls workbooks are not supported in this demo environment. Please save the file as .xlsx and upload it again.')
    try:
        workbook = pd.ExcelFile(io.BytesIO(file_bytes), engine='openpyxl')
    except Exception as error:
        raise DataLoadError(f'Unable to open the Excel workbook. {error}') from error
    return workbook.sheet_names


def read_excel_bytes(file_bytes: bytes, sheet_name: str | None = None, suffix: str = '.xlsx') -> pd.DataFrame:
    if suffix == '.xls':
        raise DataLoadError('Legacy .xls workbooks are not supported in this demo environment. Please save the file as .xlsx and upload it again.')
    try:
        return pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet_name or 0, engine='openpyxl')
    except Exception as error:
        raise DataLoadError(f'Unable to parse the selected Excel sheet. {error}') from error


def load_uploaded_file(file_name: str, file_bytes: bytes, sheet_name: str | None = None) -> tuple[pd.DataFrame, dict[str, str]]:
    suffix = Path(file_name).suffix.lower()
    if suffix == '.csv':
        raw = read_csv_bytes(file_bytes)
    elif suffix in {'.xlsx', '.xlsm', '.xls'}:
        raw = read_excel_bytes(file_bytes, sheet_name=sheet_name, suffix=suffix)
    else:
        raise DataLoadError('Unsupported file type. Please upload a CSV or Excel file.')
    if raw.empty:
        raise DataLoadError('The uploaded file contains no rows to analyze.')
    return standardize_dataframe(raw)


def load_demo_dataset(dataset_name: str) -> tuple[pd.DataFrame, dict[str, str]]:
    if dataset_name not in DEMO_DATASETS:
        raise DataLoadError(f'Unknown demo dataset: {dataset_name}')
    path = Path(DEMO_DATASETS[dataset_name]['path'])
    if not path.exists():
        raise DataLoadError(f'Demo dataset not found at {path}.')
    raw = pd.read_csv(path)
    return standardize_dataframe(raw)
