from __future__ import annotations

import gc
import hashlib
import io
import math
import re
from csv import Error as CsvError
from pathlib import Path

import pandas as pd
from pandas.errors import ParserError
from src.runtime_paths import data_path
from src.schema_detection import detect_structure


DATA_DIR = data_path()
DEMO_DATASETS = {
    'Healthcare Operations Demo': {
        'path': DATA_DIR / 'synthetic_hospital_data.csv',
        'description': 'Hospital Operations Demo Dataset with encounter dates, treatment context, stage groupings, readmission outcomes, and financial signals for realistic healthcare walkthroughs.',
        'best_for': 'Healthcare operations, cohort analysis, trend review, risk segmentation, and readmission-style analysis.',
    },
    'Healthcare Claims Demo': {
        'path': DATA_DIR / 'synthetic_healthcare_claims_demo.csv',
        'description': 'Claims-focused payer and utilization demo with member, claim, provider, diagnosis, financial, and encounter fields plus a few intentional integrity issues for validation review.',
        'best_for': 'Claims validation, payer mix review, financial integrity checks, provider utilization, and recruiter-friendly healthcare claims walkthroughs.',
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


STREAMING_UPLOAD_MB = 50.0
SAMPLED_ANALYSIS_UPLOAD_MB = 100.0
MAX_SAMPLED_UPLOAD_MB = 500.0
STREAMING_CHUNK_ROWS = 50_000
STREAMING_CHUNK_BYTES = 10 * 1024 * 1024
STREAMING_SAMPLE_TARGET_ROWS = 120_000
STREAMING_CONFIDENCE_REFRESH_INTERVAL = 2


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
    data, _ = read_csv_bytes_with_strategy(file_bytes)
    return data


def _normalize_headers(columns: list[object]) -> tuple[list[str], dict[str, str]]:
    original_names = [str(col) for col in columns]
    normalized_names = _make_unique([normalize_column_name(col) for col in original_names])
    original_lookup = dict(zip(normalized_names, original_names))
    return normalized_names, original_lookup


def _build_dataset_cache_key(file_name: str, file_bytes: bytes, *, sheet_name: str | None = None) -> str:
    digest = hashlib.sha256()
    digest.update(str(file_name).encode('utf-8'))
    digest.update(str(sheet_name or '').encode('utf-8'))
    digest.update(str(len(file_bytes)).encode('utf-8'))
    digest.update(file_bytes[:65536])
    digest.update(file_bytes[-65536:] if len(file_bytes) > 65536 else file_bytes)
    return digest.hexdigest()


def _estimate_rows_per_chunk(file_bytes: bytes, *, target_chunk_bytes: int = STREAMING_CHUNK_BYTES) -> int:
    sample_bytes = file_bytes[: min(len(file_bytes), 1024 * 1024)]
    newline_count = max(sample_bytes.count(b'\n'), 1)
    average_row_bytes = max(len(sample_bytes) / newline_count, 64.0)
    estimated_rows = int(target_chunk_bytes / average_row_bytes)
    return max(5_000, min(estimated_rows, 250_000))


def _drop_malformed_csv_lines(file_bytes: bytes, encoding: str) -> bytes:
    text = file_bytes.decode(encoding, errors='ignore')
    cleaned_lines: list[str] = []
    for line in text.splitlines():
        if line.count('"') % 2 != 0:
            continue
        cleaned_lines.append(line)
    return '\n'.join(cleaned_lines).encode(encoding, errors='ignore')


def _resolve_sampling_strategy(file_size_mb: float, sampling_override: str) -> dict[str, object]:
    normalized_override = str(sampling_override or 'auto').strip().lower()
    if normalized_override not in {'auto', 'full', 'sampled'}:
        normalized_override = 'auto'
    if file_size_mb > MAX_SAMPLED_UPLOAD_MB and normalized_override == 'full':
        raise DataLoadError(
            f'The uploaded CSV is {file_size_mb:.1f} MB. Files above {MAX_SAMPLED_UPLOAD_MB:.0f} MB should use sampling for this interactive workflow.'
        )
    if normalized_override == 'full':
        return {
            'sampling_mode': 'full',
            'ingestion_strategy': 'streaming_csv',
            'step': 1,
            'first_chunk_full': True,
            'recommended_mode': 'Use sampling' if file_size_mb > MAX_SAMPLED_UPLOAD_MB else 'Analyze full dataset',
            'warning': (
                f'This file is {file_size_mb:.1f} MB. Sampling is recommended above {MAX_SAMPLED_UPLOAD_MB:.0f} MB.'
                if file_size_mb > MAX_SAMPLED_UPLOAD_MB
                else ''
            ),
        }
    if normalized_override == 'sampled':
        return {
            'sampling_mode': 'sampled',
            'ingestion_strategy': 'sampled_streaming_csv' if file_size_mb >= STREAMING_UPLOAD_MB else 'sampled_csv',
            'step': 5,
            'first_chunk_full': file_size_mb >= STREAMING_UPLOAD_MB and file_size_mb < SAMPLED_ANALYSIS_UPLOAD_MB,
            'recommended_mode': 'Use sampling',
            'warning': 'Processing a sampled view for faster initial analysis.',
        }
    if file_size_mb > MAX_SAMPLED_UPLOAD_MB:
        return {
            'sampling_mode': 'sampled',
            'ingestion_strategy': 'sampled_streaming_csv',
            'step': 5,
            'first_chunk_full': False,
            'recommended_mode': 'Use sampling',
            'warning': f'This file is {file_size_mb:.1f} MB. Sampling is recommended for interactive analysis.',
        }
    if file_size_mb >= SAMPLED_ANALYSIS_UPLOAD_MB:
        return {
            'sampling_mode': 'sampled',
            'ingestion_strategy': 'sampled_streaming_csv',
            'step': 5,
            'first_chunk_full': False,
            'recommended_mode': 'Use sampling',
            'warning': 'Processing a systematic 20% sample for faster initial analysis.',
        }
    if file_size_mb >= STREAMING_UPLOAD_MB:
        if normalized_override == 'sampled':
            return {
                'sampling_mode': 'sampled',
                'ingestion_strategy': 'sampled_streaming_csv',
                'step': 5,
                'first_chunk_full': False,
                'recommended_mode': 'Use sampling',
                'warning': 'Processing a sampled view for faster initial analysis.',
            }
        return {
            'sampling_mode': 'sampled',
            'ingestion_strategy': 'hybrid_streaming_csv',
            'step': 5,
            'first_chunk_full': True,
            'recommended_mode': 'Auto optimize',
            'warning': 'Processing the first chunk fully and sampling every 5th row from the remaining chunks.',
        }
    return {
        'sampling_mode': 'full',
        'ingestion_strategy': 'standard_csv',
        'step': 1,
        'first_chunk_full': True,
        'recommended_mode': 'Analyze full dataset',
        'warning': '',
    }


class StreamingDatasetAnalyzer:
    def __init__(
        self,
        *,
        file_name: str,
        file_bytes: bytes,
        encoding: str,
        sampling_override: str = 'auto',
        sample_target_rows: int = STREAMING_SAMPLE_TARGET_ROWS,
        progress_callback=None,
    ) -> None:
        self.file_name = file_name
        self.file_bytes = file_bytes
        self.encoding = encoding
        self.file_size_bytes = len(file_bytes)
        self.file_size_mb = self.file_size_bytes / (1024 ** 2)
        self.chunk_size_bytes = STREAMING_CHUNK_BYTES
        self.rows_per_chunk = _estimate_rows_per_chunk(file_bytes, target_chunk_bytes=self.chunk_size_bytes)
        self.total_chunks = max(1, int(math.ceil(self.file_size_bytes / self.chunk_size_bytes)))
        self.progress_callback = progress_callback
        self.sample_target_rows = int(sample_target_rows)
        self.strategy = _resolve_sampling_strategy(self.file_size_mb, sampling_override)
        self.chunks_processed = 0
        self.total_rows = 0
        self.analyzed_rows = 0
        self.normalized_names: list[str] | None = None
        self.original_lookup: dict[str, str] = {}
        self.analysis_chunks: list[pd.DataFrame] = []
        self.cumulative_preview: pd.DataFrame | None = None
        self.confidence_snapshots: list[dict[str, object]] = []
        self.retained_rows_peak = 0

    def _sample_chunk(self, chunk: pd.DataFrame, chunk_index: int) -> pd.DataFrame:
        if str(self.strategy.get('sampling_mode', 'full')) == 'full':
            return chunk
        if bool(self.strategy.get('first_chunk_full')) and chunk_index == 1:
            return chunk
        step = max(int(self.strategy.get('step', 5)), 1)
        if step <= 1:
            return chunk
        sample_fraction = min(1.0, 1.0 / float(step))
        sample_size = max(1, int(round(len(chunk) * sample_fraction)))
        sampled = chunk.sample(n=min(sample_size, len(chunk)), random_state=42 + chunk_index).copy()
        if sampled.empty and not chunk.empty:
            return chunk.head(1).copy()
        return sampled

    def _update_cumulative_preview(self, sampled_chunk: pd.DataFrame) -> None:
        if self.cumulative_preview is None:
            self.cumulative_preview = sampled_chunk.copy()
        else:
            self.cumulative_preview = pd.concat([self.cumulative_preview, sampled_chunk], ignore_index=True)
        if len(self.cumulative_preview) > self.sample_target_rows:
            self.cumulative_preview = (
                self.cumulative_preview.sample(n=self.sample_target_rows, random_state=42).reset_index(drop=True)
            )
        self.retained_rows_peak = max(self.retained_rows_peak, len(self.cumulative_preview))

    def _retain_analysis_chunk(self, sampled_chunk: pd.DataFrame) -> None:
        if str(self.strategy.get('sampling_mode', 'full')) == 'full':
            self.analysis_chunks.append(sampled_chunk)
            retained_rows = sum(len(chunk) for chunk in self.analysis_chunks)
            self.retained_rows_peak = max(self.retained_rows_peak, retained_rows)
            return
        self._update_cumulative_preview(sampled_chunk)

    def _update_confidence(self, chunk_index: int) -> None:
        if self.cumulative_preview is None or self.cumulative_preview.empty:
            return
        refresh_interval = max(int(STREAMING_CONFIDENCE_REFRESH_INTERVAL), 1)
        if chunk_index != self.total_chunks and chunk_index != 1 and chunk_index % refresh_interval != 0:
            return
        structure = detect_structure(self.cumulative_preview)
        detection_table = structure.detection_table[['column_name', 'inferred_type', 'confidence_score']].copy()
        self.confidence_snapshots.append(
            {
                'chunk_index': int(chunk_index),
                'rows_profiled': int(len(self.cumulative_preview)),
                'detection_table': detection_table,
                'overall_confidence': float(structure.confidence_score),
            }
        )

    def _emit_progress(self, chunk_index: int) -> None:
        if self.progress_callback is None:
            return
        processed_mb = min(chunk_index * (self.chunk_size_bytes / (1024 ** 2)), self.file_size_mb)
        progress_value = min(0.90, 0.12 + (chunk_index / max(self.total_chunks, 1)) * 0.78)
        self.progress_callback(
            progress_value,
            f'Processing chunk {chunk_index} of {self.total_chunks} ({processed_mb:.0f}MB/{self.file_size_mb:.0f}MB)',
        )

    def _reset_streaming_state(self) -> None:
        self.chunks_processed = 0
        self.total_rows = 0
        self.analyzed_rows = 0
        self.normalized_names = None
        self.original_lookup = {}
        self.analysis_chunks = []
        self.cumulative_preview = None
        self.confidence_snapshots = []
        self.retained_rows_peak = 0

    def analyze_streaming(self) -> tuple[pd.DataFrame, dict[str, object]]:
        def _process_payload(payload: bytes) -> tuple[pd.DataFrame, dict[str, object]]:
            self._reset_streaming_state()
            reader = pd.read_csv(
                io.BytesIO(payload),
                encoding=self.encoding,
                chunksize=self.rows_per_chunk,
                on_bad_lines='skip',
                low_memory=True,
                engine='python',
            )
            for chunk_index, chunk in enumerate(reader, start=1):
                self.chunks_processed = chunk_index
                if self.normalized_names is None:
                    self.normalized_names, self.original_lookup = _normalize_headers(list(chunk.columns))
                chunk = chunk.copy()
                chunk.columns = self.normalized_names
                self.total_rows += len(chunk)
                sampled_chunk = self._sample_chunk(chunk, chunk_index)
                self.analyzed_rows += len(sampled_chunk)
                self._retain_analysis_chunk(sampled_chunk)
                self._update_confidence(chunk_index)
                self._emit_progress(chunk_index)
                del chunk
                del sampled_chunk
                if chunk_index % 2 == 0:
                    gc.collect()
            if str(self.strategy.get('sampling_mode', 'full')) == 'full':
                if not self.analysis_chunks:
                    raise DataLoadError('The uploaded CSV file contains no rows to analyze.')
                raw = pd.concat(self.analysis_chunks, ignore_index=True)
            else:
                raw = self.cumulative_preview.copy() if isinstance(self.cumulative_preview, pd.DataFrame) else pd.DataFrame()
            if raw.empty:
                raise DataLoadError('The uploaded CSV file contains no rows to analyze.')
            latest_detection = self.confidence_snapshots[-1]['detection_table'] if self.confidence_snapshots else pd.DataFrame()
            return raw, {
                'ingestion_strategy': str(self.strategy.get('ingestion_strategy', 'streaming_csv')),
                'sampling_mode': str(self.strategy.get('sampling_mode', 'full')),
                'source_row_count': int(self.total_rows),
                'analyzed_row_count': int(len(raw)),
                'chunk_count': int(self.chunks_processed),
                'estimated_total_chunks': int(self.total_chunks),
                'chunk_size_mb': float(self.chunk_size_bytes / (1024 ** 2)),
                'rows_per_chunk': int(self.rows_per_chunk),
                'recommended_mode': str(self.strategy.get('recommended_mode', 'Auto optimize')),
                'sampling_warning': str(self.strategy.get('warning', '')),
                'sampling_override': str(self.strategy.get('sampling_mode', 'full')),
                'original_lookup': self.original_lookup,
                'confidence_snapshots': self.confidence_snapshots,
                'detection_confidence_table': latest_detection,
                'peak_retained_rows': int(self.retained_rows_peak),
            }

        try:
            return _process_payload(self.file_bytes)
        except (ParserError, CsvError):
            sanitized_payload = _drop_malformed_csv_lines(self.file_bytes, self.encoding)
            return _process_payload(sanitized_payload)


def _try_read_csv_chunks(
    file_bytes: bytes,
    *,
    file_name: str,
    encoding: str,
    sampling_override: str,
    sample_target_rows: int,
    progress_callback=None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    analyzer = StreamingDatasetAnalyzer(
        file_name=file_name,
        file_bytes=file_bytes,
        encoding=encoding,
        sampling_override=sampling_override,
        sample_target_rows=sample_target_rows,
        progress_callback=progress_callback,
    )
    return analyzer.analyze_streaming()


def read_csv_bytes_with_strategy(
    file_bytes: bytes,
    *,
    file_name: str = 'uploaded.csv',
    sampling_override: str = 'auto',
    progress_callback=None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    encodings = ['utf-8', 'utf-8-sig', 'latin-1']
    last_error: Exception | None = None
    file_size_mb = len(file_bytes) / (1024 ** 2)
    if file_size_mb >= STREAMING_UPLOAD_MB:
        for encoding in encodings:
            try:
                raw, stats = _try_read_csv_chunks(
                    file_bytes,
                    file_name=file_name,
                    encoding=encoding,
                    sampling_override=sampling_override,
                    sample_target_rows=STREAMING_SAMPLE_TARGET_ROWS,
                    progress_callback=progress_callback,
                )
                standardized, original_lookup = standardize_dataframe(raw)
                stats['original_lookup'] = original_lookup
                stats['file_size_mb'] = file_size_mb
                stats['dataset_cache_key'] = _build_dataset_cache_key(file_name, file_bytes)
                stats['memory_strategy'] = 'streaming'
                return standardized, stats
            except DataLoadError:
                raise
            except Exception as error:
                last_error = error
    for encoding in encodings:
        for read_options in ({'encoding': encoding}, {'encoding': encoding, 'sep': None, 'engine': 'python'}):
            try:
                raw = pd.read_csv(io.BytesIO(file_bytes), **read_options)
                standardized, original_lookup = standardize_dataframe(raw)
                return standardized, {
                    'ingestion_strategy': 'standard_csv',
                    'sampling_mode': 'full',
                    'source_row_count': int(len(standardized)),
                    'analyzed_row_count': int(len(standardized)),
                    'chunk_count': 1,
                    'original_lookup': original_lookup,
                    'file_size_mb': file_size_mb,
                    'dataset_cache_key': _build_dataset_cache_key(file_name, file_bytes),
                    'memory_strategy': 'standard',
                }
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
    data, original_lookup, _ = load_uploaded_file_bundle(file_name, file_bytes, sheet_name=sheet_name)
    return data, original_lookup


def load_uploaded_file_bundle(
    file_name: str,
    file_bytes: bytes,
    sheet_name: str | None = None,
    *,
    sampling_override: str = 'auto',
    progress_callback=None,
) -> tuple[pd.DataFrame, dict[str, str], dict[str, object]]:
    suffix = Path(file_name).suffix.lower()
    if suffix == '.csv':
        raw, stats = read_csv_bytes_with_strategy(
            file_bytes,
            file_name=file_name,
            sampling_override=sampling_override,
            progress_callback=progress_callback,
        )
        if raw.empty:
            raise DataLoadError('The uploaded file contains no rows to analyze.')
        return raw, dict(stats.get('original_lookup', {})), stats
    elif suffix in {'.xlsx', '.xlsm', '.xls'}:
        raw = read_excel_bytes(file_bytes, sheet_name=sheet_name, suffix=suffix)
        stats = {
            'ingestion_strategy': 'excel_standard',
            'sampling_mode': 'full',
            'source_row_count': int(len(raw)),
            'analyzed_row_count': int(len(raw)),
            'chunk_count': 1,
            'file_size_mb': len(file_bytes) / (1024 ** 2),
            'dataset_cache_key': _build_dataset_cache_key(file_name, file_bytes, sheet_name=sheet_name),
        }
    else:
        raise DataLoadError('Unsupported file type. Please upload a CSV or Excel file.')
    if raw.empty:
        raise DataLoadError('The uploaded file contains no rows to analyze.')
    data, original_lookup = standardize_dataframe(raw)
    stats['original_lookup'] = original_lookup
    return data, original_lookup, stats


def load_demo_dataset(dataset_name: str) -> tuple[pd.DataFrame, dict[str, str]]:
    if dataset_name not in DEMO_DATASETS:
        raise DataLoadError(f'Unknown demo dataset: {dataset_name}')
    path = Path(DEMO_DATASETS[dataset_name]['path'])
    if not path.exists():
        raise DataLoadError(f'Demo dataset not found at {path}.')
    raw = pd.read_csv(path)
    return standardize_dataframe(raw)
