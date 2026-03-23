from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace
from typing import Any

import pandas as pd

import src.auth as auth_module
from src.export_utils import (
    build_audience_report_text,
    build_generated_report_text,
    build_readmission_summary_text,
    build_text_report,
)
from src.modules.rbac import can_access
from src.services.job_service import read_background_task, submit_background_task


def _get_auth_module():
    global auth_module
    required_attrs = (
        'enforce_workspace_boundary',
        'enforce_workspace_permission',
        'enforce_workspace_minimum_role',
        'workspace_identity_can_access',
    )
    if all(hasattr(auth_module, name) for name in required_attrs):
        return auth_module
    sys.modules.pop('src.auth', None)
    auth_module = importlib.import_module('src.auth')
    return auth_module


def build_report_text_output(
    report_label: str,
    dataset_name: str,
    pipeline: dict[str, Any],
) -> bytes:
    normalized_label = str(report_label or '').strip() or 'Generated Report'
    overview = dict(pipeline.get('overview', {}))
    overview.setdefault('rows', 0)
    overview.setdefault('columns', 0)
    overview.setdefault('duplicate_rows', 0)
    overview.setdefault('missing_values', 0)
    quality = dict(pipeline.get('quality', {}))
    quality.setdefault('quality_score', 0.0)
    quality.setdefault('high_missing', pd.DataFrame())
    readiness = dict(pipeline.get('readiness', {}))
    readiness.setdefault('readiness_score', 0.0)
    readiness.setdefault('readiness_table', pd.DataFrame())
    healthcare = dict(pipeline.get('healthcare', {}))
    healthcare.setdefault('healthcare_readiness_score', 0.0)
    insights = dict(pipeline.get('insights', {}))
    insights.setdefault('summary_lines', [])
    insights.setdefault('recommendations', [])
    structure = pipeline.get('structure') or SimpleNamespace(
        numeric_columns=[],
        date_columns=[],
        categorical_columns=[],
    )
    semantic = dict(pipeline.get('semantic', {}))
    semantic.setdefault('canonical_map', {})
    action_recommendations = pipeline.get('action_recommendations', pd.DataFrame())
    trust_summary = dict(pipeline.get('analysis_trust_summary', {}))
    accuracy_summary = dict(pipeline.get('result_accuracy_summary', {}))
    if normalized_label == 'Analyst Report':
        return build_audience_report_text(
            'Analyst Report',
            dataset_name,
            overview,
            structure,
            quality,
            semantic,
            readiness,
            healthcare,
            insights,
            action_recommendations,
            trust_summary,
            accuracy_summary,
        )
    if normalized_label == 'Executive Report':
        return build_audience_report_text(
            'Executive Summary',
            dataset_name,
            overview,
            structure,
            quality,
            semantic,
            readiness,
            healthcare,
            insights,
            action_recommendations,
            trust_summary,
            accuracy_summary,
        )
    if normalized_label == 'Data Readiness Report':
        return build_audience_report_text(
            'Data Readiness Review',
            dataset_name,
            overview,
            structure,
            quality,
            semantic,
            readiness,
            healthcare,
            insights,
            action_recommendations,
            trust_summary,
            accuracy_summary,
        )
    if normalized_label == 'Clinical Summary':
        return build_audience_report_text(
            'Clinical Report',
            dataset_name,
            overview,
            structure,
            quality,
            semantic,
            readiness,
            healthcare,
            insights,
            action_recommendations,
            trust_summary,
            accuracy_summary,
        )
    if normalized_label == 'Readmission Report':
        return build_readmission_summary_text(
            dataset_name,
            healthcare.get('readmission', {}),
            action_recommendations,
        )
    if normalized_label == 'Dataset Summary Report':
        return build_text_report(
            dataset_name,
            overview,
            structure,
            pipeline.get('field_profile', pd.DataFrame()),
            quality,
            semantic,
            readiness,
            healthcare,
            insights,
        )
    return build_generated_report_text(
        normalized_label,
        dataset_name,
        overview,
        quality,
        readiness,
        healthcare,
        insights,
        action_recommendations,
        trust_summary,
        accuracy_summary,
    )


def generate_report_deliverable(
    session_state: dict[str, Any],
    *,
    job_runtime: dict[str, Any],
    report_label: str,
    dataset_name: str,
    pipeline: dict[str, Any],
    workspace_identity: dict[str, Any] | None = None,
    role: str = 'Analyst',
    progress_callback=None,
) -> dict[str, Any]:
    active_auth = _get_auth_module()
    if not can_access(role, 'exports'):
        raise PermissionError(f"The active audience role '{role}' does not currently allow report export actions.")
    active_auth.enforce_workspace_boundary(
        workspace_identity,
        workspace_id=str((workspace_identity or {}).get('workspace_id', '')),
        resource_label='report generation',
    )
    active_auth.enforce_workspace_permission(
        workspace_identity,
        'export_generate',
        resource_label='report generation',
    )
    active_auth.enforce_workspace_minimum_role(
        workspace_identity,
        'analyst',
        resource_label='report generation',
    )
    if not active_auth.workspace_identity_can_access(workspace_identity, 'export_download'):
        raise PermissionError('The active workspace does not allow export downloads for the current signed-in role.')
    task_kwargs = {
        'report_label': report_label,
        'dataset_name': dataset_name,
        'overview': pipeline['overview'],
        'quality': pipeline['quality'],
        'readiness': pipeline['readiness'],
        'healthcare': pipeline['healthcare'],
        'insights': pipeline['insights'],
        'action_recommendations': pipeline.get('action_recommendations', pd.DataFrame()),
    }
    report_job = submit_background_task(
        session_state,
        job_runtime=job_runtime,
        task_key='report_generation',
        task_label=report_label,
        detail=f'Generated {report_label} for {dataset_name}.',
        stage_messages=[
            'Collecting dataset and quality summaries...',
            'Composing stakeholder-ready report text...',
            f'{report_label} is ready.',
        ],
        progress_callback=progress_callback,
        task_name=None,
        task_kwargs=task_kwargs,
        runner=lambda report_label=report_label: build_report_text_output(
            report_label,
            dataset_name,
            pipeline,
        ),
    )
    job_state = read_background_task(session_state, report_job['job_id'])
    return {
        'job': report_job,
        'status': job_state['status'],
        'result': job_state['result'],
    }


__all__ = ['build_report_text_output', 'generate_report_deliverable']
