from __future__ import annotations

from dataclasses import dataclass
import importlib
import sys
import hashlib
import io
from pathlib import Path
from typing import Any
import gc

import pandas as pd

import src.auth as auth_module
import src.logger as logger_module
from src.data_loader import DEMO_DATASETS, DataLoadError, list_excel_sheets, load_demo_dataset, load_uploaded_file_bundle

DATASET_SOURCE_OPTIONS = ['Built-in example dataset', 'Uploaded dataset']


@dataclass(frozen=True)
class PrimaryDatasetSelection:
    source_mode: str
    data: pd.DataFrame | None
    original_lookup: dict[str, str]
    dataset_name: str
    source_meta: dict[str, Any]


def _get_auth_module():
    global auth_module
    required_attrs = (
        'enforce_workspace_boundary',
        'enforce_workspace_permission',
        'enforce_workspace_minimum_role',
    )
    if all(hasattr(auth_module, name) for name in required_attrs):
        return auth_module
    sys.modules.pop('src.auth', None)
    auth_module = importlib.import_module('src.auth')
    return auth_module


def _persist_dataset_metadata(
    persistence_service: Any | None,
    application_service: Any | None,
    workspace_identity: dict[str, Any] | None,
    *,
    dataset_name: str,
    source_meta: dict[str, Any],
    data: pd.DataFrame,
) -> None:
    if application_service is not None:
        version_hash = hashlib.sha1(
            (
                f"{dataset_name}::{source_meta.get('source_mode', 'unknown')}::"
                f"{len(data)}::{len(data.columns)}::{source_meta.get('dataset_cache_key', '')}"
            ).encode('utf-8')
        ).hexdigest()[:16]
        application_service.record_dataset_metadata(
            workspace_identity or {},
            {
                'dataset_name': dataset_name,
                'source_mode': source_meta.get('source_mode', 'unknown'),
                'row_count': len(data),
                'column_count': len(data.columns),
                'source_columns': [str(column) for column in data.columns],
                'source_dtypes': {
                    str(column): str(dtype)
                    for column, dtype in data.dtypes.items()
                },
                'file_size_mb': source_meta.get('file_size_mb', 0.0),
                'description': source_meta.get('description', ''),
                'best_for': source_meta.get('best_for', ''),
                'dataset_version_hash': version_hash,
                'version_label': f"{dataset_name} | {source_meta.get('source_mode', 'unknown')}",
            },
        )
        return
    if persistence_service is None or not bool(getattr(persistence_service, 'enabled', False)):
        return
    persistence_service.save_dataset_metadata(
        workspace_identity or {},
        {
            'dataset_name': dataset_name,
            'source_mode': source_meta.get('source_mode', 'unknown'),
            'row_count': len(data),
            'column_count': len(data.columns),
            'source_columns': [str(column) for column in data.columns],
            'source_dtypes': {
                str(column): str(dtype)
                for column, dtype in data.dtypes.items()
            },
            'file_size_mb': source_meta.get('file_size_mb', 0.0),
            'description': source_meta.get('description', ''),
            'best_for': source_meta.get('best_for', ''),
        },
    )


def _log_dataset_loaded(dataset_name: str, source_mode: str, data: pd.DataFrame, *, file_size_mb: float, sheet_name: str = '') -> None:
    logger_module.log_platform_event(
        'dataset_loaded',
        logger_name='ingestion',
        dataset_name=dataset_name,
        source_mode=source_mode,
        file_size_mb=round(float(file_size_mb), 3),
        row_count=int(len(data)),
        column_count=int(len(data.columns)),
        sheet_name=sheet_name or '',
    )


def resolve_primary_dataset_selection(
    *,
    session_state: dict[str, Any],
    load_selection,
) -> PrimaryDatasetSelection:
    cached_bundle = session_state.get('active_dataset_bundle')
    cached_selection = build_selection_from_active_bundle(
        cached_bundle,
        storage_service=session_state.get('storage_service'),
    )
    ui_selection = load_selection()
    if ui_selection.data is not None:
        return ui_selection

    desired_source_mode = str(session_state.get('dataset_source_mode', ''))
    if cached_selection is not None:
        if cached_selection.source_mode == 'Uploaded dataset' and desired_source_mode != 'Built-in example dataset':
            return cached_selection
        if cached_selection.source_mode == 'Built-in example dataset' and desired_source_mode != 'Uploaded dataset':
            selected_demo_name = str(session_state.get('demo_dataset_name', '') or '')
            if not selected_demo_name or selected_demo_name == cached_selection.dataset_name:
                return cached_selection
    return ui_selection


def _build_uploaded_source_meta(
    file_name: str,
    file_bytes: bytes,
    load_stats: dict[str, Any],
    *,
    source_mode: str,
    sampling_override: str,
    sheet_name: str | None = None,
) -> dict[str, Any]:
    return {
        'source_mode': source_mode,
        'description': 'Uploaded source dataset provided by the current user session.',
        'best_for': 'Schema-flexible profiling, readiness review, and conditional healthcare analysis.',
        'file_size_mb': len(file_bytes) / (1024 ** 2),
        'ingestion_strategy': str(load_stats.get('ingestion_strategy', 'standard')),
        'sampling_mode': str(load_stats.get('sampling_mode', 'full')),
        'source_row_count': int(load_stats.get('source_row_count', len(load_stats.get('data', [])))),
        'analyzed_row_count': int(load_stats.get('analyzed_row_count', len(load_stats.get('data', [])))),
        'chunk_count': int(load_stats.get('chunk_count', 1)),
        'dataset_cache_key': str(load_stats.get('dataset_cache_key', '')),
        'estimated_total_chunks': int(load_stats.get('estimated_total_chunks', load_stats.get('chunk_count', 1))),
        'chunk_size_mb': float(load_stats.get('chunk_size_mb', 0.0) or 0.0),
        'rows_per_chunk': int(load_stats.get('rows_per_chunk', 0) or 0),
        'sampling_warning': str(load_stats.get('sampling_warning', '')),
        'recommended_large_file_mode': str(load_stats.get('recommended_mode', 'Auto optimize')),
        'sampling_override': str(sampling_override),
        'sheet_name': str(sheet_name or ''),
        'upload_status': 'ready',
    }


def _restore_uploaded_selection_from_bundle(
    bundle: dict[str, Any],
    *,
    storage_service: Any | None = None,
) -> PrimaryDatasetSelection | None:
    file_name = str(bundle.get('upload_file_name', '') or bundle.get('dataset_name', '') or '')
    source_meta = dict(bundle.get('source_meta') or {})
    file_bytes = bundle.get('upload_file_bytes')
    if not isinstance(file_bytes, (bytes, bytearray)):
        artifact = bundle.get('upload_artifact')
        if isinstance(artifact, dict) and storage_service is not None and bool(getattr(storage_service, 'enabled', False)):
            try:
                file_bytes = storage_service.load_artifact_bytes(
                    relative_path=str(artifact.get('relative_path', '') or ''),
                    artifact_path=str(artifact.get('artifact_path', '') or ''),
                )
            except Exception:
                file_bytes = b''
    if not isinstance(file_bytes, (bytes, bytearray)) or not file_name:
        return None
    binary = bytes(file_bytes)
    sheet_name = str(source_meta.get('sheet_name', '') or '') or None
    sampling_override = str(source_meta.get('sampling_override', 'auto') or 'auto')
    data, original_lookup, load_stats = load_uploaded_file_bundle(
        file_name,
        binary,
        sheet_name=sheet_name,
        sampling_override=sampling_override,
    )
    restored_source_meta = dict(source_meta)
    restored_source_meta.update(_build_uploaded_source_meta(
        file_name,
        binary,
        load_stats,
        source_mode='Uploaded dataset',
        sampling_override=sampling_override,
        sheet_name=sheet_name,
    ))
    if str(source_meta.get('dataset_cache_key', '')).strip():
        restored_source_meta['dataset_cache_key'] = str(source_meta.get('dataset_cache_key', '')).strip()
    data.attrs['dataset_cache_key'] = str(restored_source_meta.get('dataset_cache_key', ''))
    return PrimaryDatasetSelection(
        source_mode='Uploaded dataset',
        data=data,
        original_lookup=original_lookup,
        dataset_name=file_name,
        source_meta=restored_source_meta,
    )


def build_selection_from_active_bundle(
    bundle: Any,
    *,
    storage_service: Any | None = None,
) -> PrimaryDatasetSelection | None:
    if not isinstance(bundle, dict):
        return None
    required_keys = {'source_mode', 'original_lookup', 'dataset_name', 'source_meta'}
    if not required_keys.issubset(bundle):
        return None
    if isinstance(bundle.get('data'), pd.DataFrame):
        return PrimaryDatasetSelection(
            source_mode=str(bundle.get('source_mode', '')),
            data=bundle.get('data'),
            original_lookup=dict(bundle.get('original_lookup') or {}),
            dataset_name=str(bundle.get('dataset_name', '')),
            source_meta=dict(bundle.get('source_meta') or {}),
        )
    if str(bundle.get('source_mode', '')) == 'Uploaded dataset':
        return _restore_uploaded_selection_from_bundle(bundle, storage_service=storage_service)
    return None


def build_persistable_active_bundle(bundle: Any) -> dict[str, Any] | None:
    if not isinstance(bundle, dict):
        return None
    source_mode = str(bundle.get('source_mode', '') or '').strip()
    dataset_name = str(bundle.get('dataset_name', '') or '').strip()
    if not source_mode or not dataset_name:
        return None
    persistable = {
        'bundle_version': int(bundle.get('bundle_version', 2) or 2),
        'active_status': str(bundle.get('active_status', 'active') or 'active'),
        'source_mode': source_mode,
        'dataset_name': dataset_name,
        'original_lookup': dict(bundle.get('original_lookup') or {}),
        'source_meta': dict(bundle.get('source_meta') or {}),
    }
    if source_mode == 'Uploaded dataset':
        persistable.update(
            {
                'upload_file_name': str(bundle.get('upload_file_name', dataset_name) or dataset_name),
                'upload_size_bytes': int(bundle.get('upload_size_bytes', 0) or 0),
                'upload_status': str(bundle.get('upload_status', 'ready') or 'ready'),
                'dataset_cache_key': str(bundle.get('dataset_cache_key', '') or ''),
            }
        )
        artifact = bundle.get('upload_artifact')
        if isinstance(artifact, dict):
            persistable['upload_artifact'] = dict(artifact)
    return persistable


def inspect_excel_sheets(file_bytes: bytes, suffix: str) -> list[str]:
    return list_excel_sheets(file_bytes, suffix=suffix)


def load_primary_dataset_from_ui(
    *,
    sidebar: Any,
    ui: Any,
    session_state: dict[str, Any],
) -> PrimaryDatasetSelection:
    active_bundle = session_state.get('active_dataset_bundle')
    if 'dataset_source_mode' not in session_state and isinstance(active_bundle, dict):
        bundle_source_mode = str(active_bundle.get('source_mode', '') or '').strip()
        if bundle_source_mode in DATASET_SOURCE_OPTIONS:
            session_state['dataset_source_mode'] = bundle_source_mode
    pending_source_mode = str(session_state.pop('pending_dataset_source_mode', '') or '').strip()
    if pending_source_mode in DATASET_SOURCE_OPTIONS:
        session_state['dataset_source_mode'] = pending_source_mode
    pending_demo_dataset_name = str(session_state.pop('pending_demo_dataset_name', '') or '').strip()
    if pending_demo_dataset_name and pending_demo_dataset_name in DEMO_DATASETS:
        session_state['demo_dataset_name'] = pending_demo_dataset_name

    sidebar_caption = getattr(sidebar, 'caption', None)
    if callable(sidebar_caption):
        sidebar_caption('Dataset source')
    source_mode = sidebar.radio('Dataset source', DATASET_SOURCE_OPTIONS, key='dataset_source_mode')
    original_lookup: dict[str, str] = {}
    dataset_name = ''
    source_meta: dict[str, Any] = {
        'source_mode': source_mode,
        'description': '',
        'best_for': '',
        'file_size_mb': 0.0,
    }
    application_service = session_state.get('application_service')
    persistence_service = session_state.get('persistence_service')
    storage_service = session_state.get('storage_service')
    workspace_identity = session_state.get('workspace_identity', {})

    if source_mode == 'Built-in example dataset':
        if callable(sidebar_caption):
            sidebar_caption('Choose a built-in dataset for walkthroughs and controlled product review.')
        dataset_name = sidebar.selectbox('Example dataset', list(DEMO_DATASETS.keys()), key='demo_dataset_name')
        sidebar_info = getattr(sidebar, 'info', None)
        if callable(sidebar_info):
            sidebar_info('To upload your own file, switch Dataset source to Uploaded dataset in this sidebar.')
        data, original_lookup, source_meta = build_demo_dataset_bundle(
            dataset_name,
            persistence_service=persistence_service,
            application_service=application_service,
            workspace_identity=workspace_identity,
        )
        return PrimaryDatasetSelection(
            source_mode=source_mode,
            data=data,
            original_lookup=original_lookup,
            dataset_name=dataset_name,
            source_meta=source_meta,
        )

    if callable(sidebar_caption):
        sidebar_caption('Upload a CSV or Excel extract. The active uploaded dataset remains authoritative across reruns until you explicitly change source.')
    if callable(sidebar.markdown):
        sidebar.markdown('**Upload file**')
    uploaded = sidebar.file_uploader(
        'Upload a dataset',
        type=['csv', 'xlsx', 'xlsm', 'xls'],
        key='primary_upload',
    )
    if uploaded is None:
        active_bundle = session_state.get('active_dataset_bundle')
        if isinstance(active_bundle, dict) and str(active_bundle.get('source_mode', '')) == 'Uploaded dataset':
            if callable(sidebar_caption):
                sidebar_caption('The current uploaded dataset is still active. Upload another file here to replace it, or switch back to a built-in example dataset.')
            cached_selection = build_selection_from_active_bundle(
                active_bundle,
                storage_service=storage_service,
            )
            if cached_selection is not None and cached_selection.data is not None:
                return cached_selection
        else:
            ui.info('Upload a CSV or Excel file, or switch to a built-in example dataset to start exploring the platform.')
        return PrimaryDatasetSelection(
            source_mode=source_mode,
            data=None,
            original_lookup=original_lookup,
            dataset_name=dataset_name,
            source_meta=source_meta,
        )

    file_bytes = uploaded.getvalue()
    file_size_mb = len(file_bytes) / (1024 ** 2)
    load_status = ui.empty()
    load_progress = ui.progress(0, text='Preparing uploaded dataset...')

    def _load_progress(value: float, message: str) -> None:
        bounded = max(0.0, min(1.0, value))
        load_progress.progress(int(bounded * 100), text=message)
        load_status.caption(message)

    sheet_name = None
    suffix = uploaded.name.lower().rsplit('.', 1)[-1]
    sampling_override = 'auto'
    if suffix == 'csv' and file_size_mb >= 50.0:
        sampling_choice = sidebar.radio(
            'Large-file analysis mode',
            ['Auto optimize', 'Analyze full dataset', 'Use sampling'],
            key='large_file_analysis_mode',
        )
        sampling_override = {
            'Auto optimize': 'auto',
            'Analyze full dataset': 'full',
            'Use sampling': 'sampled',
        }[sampling_choice]
        if file_size_mb > 500.0:
            ui.warning('This file is larger than 500 MB. Sampling is recommended for interactive analysis.')
        elif file_size_mb >= 100.0:
            ui.caption('Files above 100 MB use systematic sampling unless you override to full analysis.')
        else:
            ui.caption('Files above 50 MB use hybrid chunked analysis by default to reduce memory usage.')
    if suffix in {'xlsx', 'xlsm', 'xls'}:
        try:
            sheets = inspect_excel_sheets(file_bytes, suffix=f'.{suffix}')
            if len(sheets) > 1:
                sheet_name = sidebar.selectbox('Excel sheet', sheets, key='primary_sheet_name')
        except DataLoadError as error:
            ui.error(str(error))
            ui.caption('Check that the workbook is a supported .xlsx file and that the selected sheet contains a tabular extract with a header row.')
            return PrimaryDatasetSelection(
                source_mode=source_mode,
                data=None,
                original_lookup=original_lookup,
                dataset_name=dataset_name,
                source_meta=source_meta,
            )

    try:
        _load_progress(0.05, 'Inspecting upload and selecting the safest ingestion path...')
        data, original_lookup, source_meta, artifact = build_uploaded_dataset_bundle(
            uploaded.name,
            file_bytes,
            sheet_name=sheet_name,
            source_mode=source_mode,
            storage_service=storage_service,
            persistence_service=persistence_service,
            application_service=application_service,
            workspace_identity=workspace_identity,
            progress_callback=_load_progress,
            sampling_override=sampling_override,
        )
    except PermissionError as error:
        load_progress.empty()
        load_status.empty()
        ui.warning(str(error))
        ui.caption('Switch to guest/demo mode or use a signed-in workspace role with upload permissions to add a new dataset.')
        return PrimaryDatasetSelection(
            source_mode=source_mode,
            data=None,
            original_lookup=original_lookup,
            dataset_name=uploaded.name,
            source_meta=source_meta,
        )
    except DataLoadError as error:
        load_progress.empty()
        load_status.empty()
        logger_module.log_platform_event(
            'dataset_load_failed',
            logger_name='ingestion',
            level=40,
            dataset_name=uploaded.name,
            source_mode='upload',
            error_type=type(error).__name__,
        )
        ui.error(str(error))
        ui.caption('Try a cleaner extract with one header row, clearer date/number columns, and fewer fully empty fields.')
        return PrimaryDatasetSelection(
            source_mode=source_mode,
            data=None,
            original_lookup=original_lookup,
            dataset_name=uploaded.name,
            source_meta=source_meta,
        )
    except Exception as error:
        load_progress.empty()
        load_status.empty()
        logger_module.log_platform_exception(
            'dataset_load_unexpected_failure',
            error,
            logger_name='ingestion',
            operation_type='dataset_upload',
            dataset_name=uploaded.name,
            source_mode='upload',
            file_size_mb=round(file_size_mb, 3),
        )
        ui.error('The upload could not be opened safely in this environment.')
        ui.caption(
            'Try a CSV or Excel export with one header row, fewer fully empty columns, and consistent date/number formatting. '
            'If the issue persists, use the demo dataset to confirm the app is healthy, then retry with a narrower extract.'
        )
        return PrimaryDatasetSelection(
            source_mode=source_mode,
            data=None,
            original_lookup=original_lookup,
            dataset_name=uploaded.name,
            source_meta=source_meta,
        )

    _load_progress(1.0, 'Uploaded dataset is ready for analysis.')
    load_progress.empty()
    load_status.empty()
    if artifact is not None:
        session_state['latest_dataset_artifact'] = artifact
    session_state['pending_uploaded_dataset_state'] = {
        'file_bytes': file_bytes,
        'file_size_bytes': len(file_bytes),
        'file_name': uploaded.name,
        'artifact': artifact,
        'upload_status': str(source_meta.get('upload_status', 'ready') or 'ready'),
        'dataset_cache_key': str(source_meta.get('dataset_cache_key', '') or ''),
    }
    return PrimaryDatasetSelection(
        source_mode=source_mode,
        data=data,
        original_lookup=original_lookup,
        dataset_name=uploaded.name,
        source_meta=source_meta,
    )


def build_demo_dataset_bundle(
    dataset_name: str,
    *,
    persistence_service: Any | None = None,
    application_service: Any | None = None,
    workspace_identity: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, str], dict[str, Any]]:
    data, original_lookup = load_demo_dataset(dataset_name)
    demo = DEMO_DATASETS[dataset_name]
    demo_path = demo['path']
    file_size_mb = (Path(demo_path).stat().st_size / (1024 ** 2)) if Path(demo_path).exists() else 0.0
    source_meta = {
        'source_mode': 'Demo dataset',
        'description': demo['description'],
        'best_for': demo['best_for'],
        'file_size_mb': file_size_mb,
    }
    _persist_dataset_metadata(
        persistence_service,
        application_service,
        workspace_identity,
        dataset_name=dataset_name,
        source_meta=source_meta,
        data=data,
    )
    _log_dataset_loaded(dataset_name, 'demo', data, file_size_mb=file_size_mb)
    return data, original_lookup, source_meta


def build_uploaded_dataset_bundle(
    file_name: str,
    file_bytes: bytes,
    *,
    sheet_name: str | None = None,
    source_mode: str = 'Uploaded dataset',
    storage_service: Any | None = None,
    persistence_service: Any | None = None,
    application_service: Any | None = None,
    workspace_identity: dict[str, Any] | None = None,
    progress_callback=None,
    sampling_override: str = 'auto',
) -> tuple[pd.DataFrame, dict[str, str], dict[str, Any], Any | None]:
    active_auth = _get_auth_module()
    active_auth.enforce_workspace_boundary(
        workspace_identity,
        workspace_id=str((workspace_identity or {}).get('workspace_id', '')),
        resource_label='dataset uploads',
    )
    active_auth.enforce_workspace_permission(
        workspace_identity,
        'dataset_upload',
        resource_label='dataset uploads',
    )
    active_auth.enforce_workspace_minimum_role(
        workspace_identity,
        'analyst',
        resource_label='dataset uploads',
    )
    data, original_lookup, load_stats = load_uploaded_file_bundle(
        file_name,
        file_bytes,
        sheet_name=sheet_name,
        sampling_override=sampling_override,
        progress_callback=progress_callback,
    )
    if progress_callback is not None:
        progress_callback(0.92, 'Finalizing dataset metadata and artifact storage...')
    source_meta = _build_uploaded_source_meta(
        file_name,
        file_bytes,
        {**load_stats, 'data': data},
        source_mode=source_mode,
        sampling_override=sampling_override,
        sheet_name=sheet_name,
    )
    data.attrs['source_row_count'] = source_meta['source_row_count']
    data.attrs['sampling_mode'] = source_meta['sampling_mode']
    data.attrs['ingestion_strategy'] = source_meta['ingestion_strategy']
    data.attrs['dataset_cache_key'] = source_meta['dataset_cache_key']
    data.attrs['source_file_size_mb'] = float(source_meta['file_size_mb'])
    data.attrs['estimated_total_chunks'] = int(source_meta['estimated_total_chunks'])
    data.attrs['rows_per_chunk'] = int(source_meta['rows_per_chunk'])
    data.attrs['sampling_warning'] = str(source_meta['sampling_warning'])
    data.attrs['recommended_large_file_mode'] = str(source_meta['recommended_large_file_mode'])
    detection_confidence_table = load_stats.get('detection_confidence_table')
    if isinstance(detection_confidence_table, pd.DataFrame):
        data.attrs['streaming_detection_confidence_table'] = detection_confidence_table.copy()
    artifact = None
    if storage_service is not None and bool(getattr(storage_service, 'enabled', False)):
        artifact = storage_service.save_dataset_upload(
            workspace_identity or {},
            dataset_name=file_name,
            file_name=file_name,
            payload=file_bytes,
        )
    _persist_dataset_metadata(
        persistence_service,
        application_service,
        workspace_identity,
        dataset_name=file_name,
        source_meta=source_meta,
        data=data,
    )
    _log_dataset_loaded(
        file_name,
        'upload',
        data,
        file_size_mb=float(source_meta['file_size_mb']),
        sheet_name=sheet_name or '',
    )
    gc.collect()
    return data, original_lookup, source_meta, artifact


__all__ = [
    'DATASET_SOURCE_OPTIONS',
    'DataLoadError',
    'build_demo_dataset_bundle',
    'build_uploaded_dataset_bundle',
    'inspect_excel_sheets',
    'PrimaryDatasetSelection',
    'build_selection_from_active_bundle',
    'load_primary_dataset_from_ui',
    'resolve_primary_dataset_selection',
]
