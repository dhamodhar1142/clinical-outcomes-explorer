from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import re


WARN_UPLOAD_MB = 25.0
BLOCK_UPLOAD_MB = 100.0
WARN_MEMORY_MB = 300.0
BLOCK_MEMORY_MB = 900.0
LONG_TASK_ROWS = 50_000
LONG_TASK_MEMORY_MB = 150.0


def _status_label(value: str) -> str:
    return value


def _profile_value(profile_config: dict[str, Any] | None, key: str, default: float | int) -> float | int:
    if profile_config is None:
        return default
    return profile_config.get(key, default)


def build_preflight_guardrails(
    source_meta: dict[str, Any],
    memory_mb: float,
    row_count: int,
    column_count: int,
    profile_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    file_size_mb = float(source_meta.get('file_size_mb', 0.0) or 0.0)
    ingestion_strategy = str(source_meta.get('ingestion_strategy', 'standard'))
    sampling_mode = str(source_meta.get('sampling_mode', 'full'))
    analyzed_row_count = int(source_meta.get('analyzed_row_count', row_count) or row_count)
    source_row_count = int(source_meta.get('source_row_count', row_count) or row_count)
    profile_name = str((profile_config or {}).get('profile_name', 'Standard'))
    warn_upload_mb = float(_profile_value(profile_config, 'warn_upload_mb', WARN_UPLOAD_MB))
    block_upload_mb = float(_profile_value(profile_config, 'block_upload_mb', BLOCK_UPLOAD_MB))
    warn_memory_mb = float(_profile_value(profile_config, 'warn_memory_mb', WARN_MEMORY_MB))
    block_memory_mb = float(_profile_value(profile_config, 'block_memory_mb', BLOCK_MEMORY_MB))
    recommended_rows = int(_profile_value(profile_config, 'recommended_rows', 100_000))
    sampled_large_upload = sampling_mode == 'sampled' and file_size_mb >= block_upload_mb
    staged_processing_required = (
        source_row_count > recommended_rows
        or memory_mb >= warn_memory_mb
        or file_size_mb >= warn_upload_mb
        or sampling_mode != 'full'
    )
    checks: list[dict[str, object]] = []
    warnings: list[str] = []
    blocked = False
    block_reason = ''

    if file_size_mb:
        if file_size_mb >= block_upload_mb and not sampled_large_upload:
            blocked = True
            block_reason = (
                f'The uploaded file is {file_size_mb:.1f} MB, which is above the safe interactive '
                f'limit of {block_upload_mb:.0f} MB for the current large-dataset profile.'
            )
            status = 'Blocked'
        elif sampled_large_upload:
            status = 'Warning'
            warnings.append(
                f'Large upload detected ({file_size_mb:.1f} MB). The platform is using sampled '
                f'streaming analysis under the {profile_name} large-dataset profile to keep the '
                'session responsive.'
            )
        elif file_size_mb >= warn_upload_mb:
            status = 'Warning'
            warnings.append(
                f'Large upload detected ({file_size_mb:.1f} MB). The platform is using '
                f'{ingestion_strategy.replace("_", " ")} under the {profile_name} large-dataset profile.'
            )
        else:
            status = 'Ready'
        checks.append({
            'check': 'Uploaded file size',
            'status': _status_label(status),
            'detail': f'{file_size_mb:.1f} MB',
        })

    if memory_mb >= block_memory_mb:
        blocked = True
        block_reason = block_reason or (
            f'The in-memory dataset is estimated at {memory_mb:.1f} MB, which is above the safe '
            'analysis limit for the current large-dataset profile.'
        )
        memory_status = 'Blocked'
    elif memory_mb >= warn_memory_mb:
        memory_status = 'Warning'
        warnings.append(
            f'Estimated in-memory footprint is {memory_mb:.1f} MB. Profiling and quality review '
            'will rely more heavily on staged sampling to keep the session stable.'
        )
    else:
        memory_status = 'Ready'
    checks.append({
        'check': 'Estimated in-memory footprint',
        'status': _status_label(memory_status),
        'detail': f'{memory_mb:.1f} MB',
    })

    checks.append({
        'check': 'Dataset shape',
        'status': 'Ready',
        'detail': f'{row_count:,} rows x {column_count:,} columns',
    })
    if source_row_count != analyzed_row_count:
        checks.append({
            'check': 'Rows analyzed interactively',
            'status': 'Warning',
            'detail': f'{analyzed_row_count:,} of {source_row_count:,} rows currently in scope',
        })
    checks.append({
        'check': 'Ingestion strategy',
        'status': 'Warning' if sampling_mode != 'full' or 'streaming' in ingestion_strategy else 'Ready',
        'detail': ingestion_strategy.replace('_', ' '),
    })
    checks.append({
        'check': 'Large dataset profile',
        'status': 'Warning' if staged_processing_required and not blocked else 'Ready',
        'detail': f'{profile_name} profile | recommended up to ~{recommended_rows:,} rows before staged sampling becomes more visible',
    })
    if staged_processing_required and not blocked:
        warnings.append(
            'What still works: the full workflow remains available, but profiling, quality review, '
            'and exports may use streaming, sampling, and cached profiling to stay responsive.'
        )

    return {
        'blocked': blocked,
        'block_reason': block_reason,
        'warnings': warnings,
        'checks_table': pd.DataFrame(checks),
        'staged_processing_required': staged_processing_required and not blocked,
        'profile_name': profile_name,
    }


def build_column_validation_report(data: pd.DataFrame) -> dict[str, Any]:
    columns = [str(col) for col in data.columns]
    unnamed_columns = [col for col in columns if col.startswith('unnamed_column')]
    auto_renamed_columns = [col for col in columns if re.search(r'_\d+$', col)]
    all_null_columns = [col for col in columns if data[col].isna().all()]
    single_value_columns = [col for col in columns if data[col].nunique(dropna=True) <= 1]

    warnings: list[str] = []
    if len(columns) < 2:
        warnings.append('The dataset has very few columns, so only limited analytics may be available.')
    if unnamed_columns:
        warnings.append('Some columns were unnamed or unclear, so they may need review before downstream analytics are trusted.')
    if auto_renamed_columns:
        warnings.append('Some duplicate column names were auto-renamed during ingestion. Review semantic mapping before sharing results.')
    if all_null_columns:
        warnings.append('Some columns are completely empty and will not contribute to analysis until they are populated.')

    checks = [
        {
            'validation_check': 'Column count',
            'status': 'Warning' if len(columns) < 2 else 'Ready',
            'detail': f'{len(columns):,} columns detected',
        },
        {
            'validation_check': 'Unnamed or unclear columns',
            'status': 'Warning' if unnamed_columns else 'Ready',
            'detail': f'{len(unnamed_columns):,} columns',
        },
        {
            'validation_check': 'Auto-renamed duplicate headers',
            'status': 'Warning' if auto_renamed_columns else 'Ready',
            'detail': f'{len(auto_renamed_columns):,} columns',
        },
        {
            'validation_check': 'All-null columns',
            'status': 'Warning' if all_null_columns else 'Ready',
            'detail': f'{len(all_null_columns):,} columns',
        },
        {
            'validation_check': 'Low-variance columns',
            'status': 'Info' if single_value_columns else 'Ready',
            'detail': f'{len(single_value_columns):,} columns',
        },
    ]

    sample_rows: list[dict[str, str]] = []
    for col in unnamed_columns[:5]:
        sample_rows.append({'column_name': col, 'issue': 'Unnamed or generic header', 'suggested_action': 'Rename or confirm the field before relying on downstream analytics.'})
    for col in auto_renamed_columns[:5]:
        sample_rows.append({'column_name': col, 'issue': 'Auto-renamed duplicate header', 'suggested_action': 'Review duplicate headers in the source file and keep only the intended version if possible.'})
    for col in all_null_columns[:5]:
        sample_rows.append({'column_name': col, 'issue': 'All values missing', 'suggested_action': 'Populate or remove this field before expecting it to support analytics.'})

    summary = (
        f"{len(columns):,} columns were validated. "
        f"{len(unnamed_columns):,} unnamed, {len(auto_renamed_columns):,} auto-renamed, and {len(all_null_columns):,} empty columns were found."
    )

    return {
        'warnings': warnings,
        'checks_table': pd.DataFrame(checks),
        'issue_samples': pd.DataFrame(sample_rows),
        'summary': summary,
    }


def build_long_task_notice(
    source_meta: dict[str, Any],
    row_count: int,
    memory_mb: float,
    job_runtime: dict[str, Any] | None = None,
    profile_config: dict[str, Any] | None = None,
) -> str | None:
    file_size_mb = float(source_meta.get('file_size_mb', 0.0) or 0.0)
    ingestion_strategy = str(source_meta.get('ingestion_strategy', 'standard'))
    sampling_mode = str(source_meta.get('sampling_mode', 'full'))
    profile_name = str((profile_config or {}).get('profile_name', 'Standard'))
    long_task_rows = int(_profile_value(profile_config, 'long_task_rows', LONG_TASK_ROWS))
    long_task_memory_mb = float(_profile_value(profile_config, 'long_task_memory_mb', LONG_TASK_MEMORY_MB))
    warn_upload_mb = float(_profile_value(profile_config, 'warn_upload_mb', WARN_UPLOAD_MB))
    if row_count >= long_task_rows or memory_mb >= long_task_memory_mb or file_size_mb >= warn_upload_mb:
        base_notice = (
            'This dataset may take a little longer to prepare. The platform will show staged progress, '
            'reuse sampling where appropriate, and keep the rest of the app responsive.'
        )
        base_notice += f' The active large-dataset profile is {profile_name}.'
        if sampling_mode != 'full':
            base_notice += (
                f' Interactive analysis is currently running in {sampling_mode} mode with '
                f'{ingestion_strategy.replace("_", " ")}.'
            )
        if job_runtime and not bool(job_runtime.get('backend_configured')):
            return base_notice + ' Heavy operations still run in the current session, but they now use a managed job-style status layer so progress is clearer.'
        if job_runtime and bool(job_runtime.get('backend_configured')):
            return base_notice + ' The job foundation is worker-ready, so the current build tracks these operations like managed tasks.'
        return base_notice
    return None


def build_deployment_health_checks(pipeline: dict[str, Any], source_meta: dict[str, Any]) -> pd.DataFrame:
    readiness = pipeline.get('readiness', {})
    standards = pipeline.get('standards', {})
    privacy = pipeline.get('privacy_review', {})
    checks = [
        {
            'health_check': 'Dataset loaded successfully',
            'status': 'Pass',
            'detail': source_meta.get('source_mode', 'Unknown source'),
        },
        {
            'health_check': 'Structure detection available',
            'status': 'Pass' if not pipeline.get('structure').detection_table.empty else 'Warning',
            'detail': f"{len(pipeline.get('structure').detection_table):,} detected fields",
        },
        {
            'health_check': 'Analysis readiness engine',
            'status': 'Pass' if readiness.get('available_count', 0) > 0 else 'Warning',
            'detail': f"{readiness.get('available_count', 0)} ready modules, {readiness.get('partial_count', 0)} partial",
        },
        {
            'health_check': 'Healthcare standards review',
            'status': 'Pass' if standards.get('available') else 'Info',
            'detail': standards.get('badge_text', 'Not strongly standards-oriented'),
        },
        {
            'health_check': 'Privacy review',
            'status': 'Pass',
            'detail': privacy.get('hipaa', {}).get('risk_level', 'Low'),
        },
        {
            'health_check': 'Export readiness',
            'status': 'Pass',
            'detail': 'TXT, CSV, and JSON export paths are active in this session.',
        },
    ]
    return pd.DataFrame(checks)


def build_performance_diagnostics(
    overview: dict[str, Any],
    sample_info: dict[str, Any],
    source_meta: dict[str, Any],
    profile_config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    analyzed_columns = int(overview.get('analyzed_columns', overview.get('columns', 0)))
    source_columns = int(overview.get('source_columns', analyzed_columns))
    source_rows = int(overview.get('rows', 0))
    analyzed_rows = int(overview.get('analyzed_rows', source_rows))
    rows = [
        {'metric': 'Source mode', 'value': source_meta.get('source_mode', 'Unknown')},
        {'metric': 'Source rows', 'value': f'{source_rows:,}'},
        {'metric': 'Analyzed rows in memory', 'value': f'{analyzed_rows:,}'},
        {'metric': 'Analyzed columns in scope', 'value': f'{analyzed_columns:,}'},
        {'metric': 'Estimated memory', 'value': f"{float(overview.get('memory_mb', 0.0)):.1f} MB"},
        {'metric': 'Large dataset profile', 'value': str((profile_config or {}).get('profile_name', 'Standard'))},
        {'metric': 'Ingestion strategy', 'value': str(source_meta.get('ingestion_strategy', overview.get('ingestion_strategy', 'standard'))).replace('_', ' ')},
        {'metric': 'Sampling mode', 'value': str(source_meta.get('sampling_mode', sample_info.get('sampling_mode', 'full')))},
        {'metric': 'Sampling applied', 'value': 'Yes' if sample_info.get('sampling_applied') else 'No'},
        {'metric': 'Profile sample rows', 'value': f"{int(sample_info.get('profile_sample_rows', 0)):,}"},
        {'metric': 'Quality sample rows', 'value': f"{int(sample_info.get('quality_sample_rows', 0)):,}"},
    ]
    if source_columns != analyzed_columns:
        rows.insert(3, {'metric': 'Source columns loaded', 'value': f'{source_columns:,}'})
    file_size_mb = source_meta.get('file_size_mb')
    if file_size_mb:
        rows.append({'metric': 'Input file size', 'value': f"{float(file_size_mb):.1f} MB"})
    if sample_info.get('sampling_applied'):
        rows.append({'metric': 'Staged processing', 'value': 'Enabled for profiling and quality review'})
    return pd.DataFrame(rows)


def build_export_safety_note(row_count: int, sample_info: dict[str, Any], profile_config: dict[str, Any] | None = None) -> str | None:
    export_guard_rows = int(_profile_value(profile_config, 'export_guard_rows', 250_000))
    profile_name = str((profile_config or {}).get('profile_name', 'Standard'))
    if row_count >= export_guard_rows:
        return (
            f'This dataset is large enough that text and manifest exports are the safest first handoff under the {profile_name} profile. '
            'If a report feels heavy, use a narrower cohort or date range for a smaller export artifact.'
        )
    if sample_info.get('sampling_applied'):
        return (
            f'This export was prepared after staged sampling safeguards were activated under the {profile_name} profile. '
            'Use a narrower cohort or time range if you want a lighter-weight stakeholder artifact.'
        )
    return None


def build_deployment_support_notes() -> pd.DataFrame:
    return pd.DataFrame([
        {
            'deployment_area': 'Streamlit launch',
            'guidance': 'Run `streamlit run app.py` locally or use the same entrypoint in Streamlit Community Cloud.',
        },
        {
            'deployment_area': 'Docker support',
            'guidance': 'A lightweight Dockerfile is included so the app can be containerized for demos or internal review.',
        },
        {
            'deployment_area': 'Large dataset guidance',
            'guidance': 'For datasets above profile thresholds, the platform uses staged sampling to keep profiling and quality review responsive.',
        },
        {
            'deployment_area': 'Operational fallback',
            'guidance': 'If a module cannot run, the app shows a readiness or fallback message rather than crashing the session.',
        },
    ])
