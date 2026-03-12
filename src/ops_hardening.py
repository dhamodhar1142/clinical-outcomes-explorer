from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


WARN_UPLOAD_MB = 25.0
BLOCK_UPLOAD_MB = 100.0
WARN_MEMORY_MB = 300.0
BLOCK_MEMORY_MB = 900.0


def _status_label(value: str) -> str:
    return value


def build_preflight_guardrails(source_meta: dict[str, Any], memory_mb: float, row_count: int, column_count: int) -> dict[str, Any]:
    file_size_mb = float(source_meta.get('file_size_mb', 0.0) or 0.0)
    checks: list[dict[str, object]] = []
    warnings: list[str] = []
    blocked = False
    block_reason = ''

    if file_size_mb:
        if file_size_mb >= BLOCK_UPLOAD_MB:
            blocked = True
            block_reason = f'The uploaded file is {file_size_mb:.1f} MB, which is above the safe interactive limit of {BLOCK_UPLOAD_MB:.0f} MB for this demo environment.'
            status = 'Blocked'
        elif file_size_mb >= WARN_UPLOAD_MB:
            status = 'Warning'
            warnings.append(f'Large upload detected ({file_size_mb:.1f} MB). Sampling and cache reuse will help, but a narrower extract may feel faster.')
        else:
            status = 'Ready'
        checks.append({
            'check': 'Uploaded file size',
            'status': _status_label(status),
            'detail': f'{file_size_mb:.1f} MB',
        })

    if memory_mb >= BLOCK_MEMORY_MB:
        blocked = True
        block_reason = block_reason or f'The in-memory dataset is estimated at {memory_mb:.1f} MB, which is above the safe analysis limit for a shared Streamlit session.'
        memory_status = 'Blocked'
    elif memory_mb >= WARN_MEMORY_MB:
        memory_status = 'Warning'
        warnings.append(f'Estimated in-memory footprint is {memory_mb:.1f} MB. The app will stay stable, but narrower extracts will improve responsiveness.')
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
        'detail': f'{row_count:,} rows × {column_count:,} columns',
    })

    return {
        'blocked': blocked,
        'block_reason': block_reason,
        'warnings': warnings,
        'checks_table': pd.DataFrame(checks),
    }


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


def build_performance_diagnostics(overview: dict[str, Any], sample_info: dict[str, Any], source_meta: dict[str, Any]) -> pd.DataFrame:
    rows = [
        {'metric': 'Source mode', 'value': source_meta.get('source_mode', 'Unknown')},
        {'metric': 'Rows in scope', 'value': f"{overview.get('rows', 0):,}"},
        {'metric': 'Columns in scope', 'value': f"{overview.get('columns', 0):,}"},
        {'metric': 'Estimated memory', 'value': f"{float(overview.get('memory_mb', 0.0)):.1f} MB"},
        {'metric': 'Sampling applied', 'value': 'Yes' if sample_info.get('sampling_applied') else 'No'},
        {'metric': 'Profile sample rows', 'value': f"{int(sample_info.get('profile_sample_rows', 0)):,}"},
        {'metric': 'Quality sample rows', 'value': f"{int(sample_info.get('quality_sample_rows', 0)):,}"},
    ]
    file_size_mb = source_meta.get('file_size_mb')
    if file_size_mb:
        rows.append({'metric': 'Input file size', 'value': f"{float(file_size_mb):.1f} MB"})
    return pd.DataFrame(rows)


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
            'guidance': 'For datasets above ~100k rows, the platform uses sampling to keep profiling and quality review responsive.',
        },
        {
            'deployment_area': 'Operational fallback',
            'guidance': 'If a module cannot run, the app shows a readiness or fallback message rather than crashing the session.',
        },
    ])

