from __future__ import annotations

import importlib
import gc
import math
import time
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

import src.logger as logger_module
import src.services.dataset_service as dataset_service
from src.analysis_progress import (
    ANALYSIS_STAGE_LABELS,
    build_analysis_progress_snapshot,
    clear_analysis_progress_snapshot,
    get_analysis_progress_snapshot,
    store_analysis_progress_snapshot,
)
from src.actionable_fallbacks import (
    build_actionable_fallback_context,
    build_cancellation_timeout_message,
    build_large_dataset_actionable_message,
)
import src.visualization_performance as visualization_performance
from src.ai_copilot import append_copilot_message, build_copilot_panel_config, initialize_copilot_memory, plan_workflow_action, run_copilot_question
from src.beta_interest import append_beta_interest_submission, build_beta_conversion_panel, build_beta_interest_summary
from src.collaboration_notes import append_collaboration_note, build_collaboration_note_summary, build_collaboration_notes_view, build_note_target_options
from src.data_loader import DEMO_DATASETS
from src.enterprise_features import (
    build_audit_log_view,
    build_data_lineage_view,
    build_dataset_comparison_dashboard,
    build_join_recommendation,
    build_quality_rule_engine,
    build_workflow_pack_details,
    build_workflow_pack_summary,
    cohort_monitoring_over_time,
    detect_join_candidates,
    infer_linked_dataset_role,
    preview_linked_merge,
)
from src.evolution_engine import merge_evolution_memory, queue_execution_items
from src.evolution_memory_store import save_evolution_memory
from src.export_utils import (
    apply_export_policy,
    apply_role_based_redaction,
    build_audience_report_text,
    build_compliance_dashboard_csv,
    build_compliance_dashboard_payload,
    build_compliance_handoff_payload,
    build_compliance_summary_text,
    build_cross_setting_reporting_profile,
    build_compliance_support_csv,
    build_executive_summary_text,
    build_generated_report_text,
    build_readmission_summary_text,
    build_governance_review_csv,
    build_governance_review_payload,
    build_governance_review_text,
    build_policy_aware_bundle_profile,
    build_policy_note_text,
    build_report_support_csv,
    build_role_export_bundle_manifest,
    build_role_export_bundle_text,
    build_shared_report_bundles,
    build_shared_report_bundle_text,
    build_text_report,
    dataframe_to_csv_bytes,
    json_bytes,
    normalize_report_mode,
    recommended_report_mode_for_role,
)
from src.healthcare_analysis import build_cohort_summary, build_readmission_cohort_review, operational_alerts, plan_readmission_intervention
from src.modules.audit import log_audit_event
from src.modules.privacy_security import evaluate_export_policy, get_export_policy_presets, run_privacy_security_review
from src.modules.rbac import can_access
from src.ops_hardening import build_deployment_support_notes
from src.plan_awareness import build_plan_awareness, is_strict_plan_enforcement, plan_feature_enabled
from src.product_settings import LARGE_DATASET_PROFILES, build_product_settings_summary
from src.presentation_support import (
    build_demo_dataset_cards,
)
from src.services.runtime_service import initialize_app_session_state
from src.jobs import get_job_runtime_key, job_cancel_requested
from src.services.job_service import cancel_background_task, force_cancel_background_task, read_background_task, submit_background_task
from src.audience_modes import build_audience_mode_guidance
from src.deployment_readiness import build_config_guidance, build_environment_checks, build_launch_checklist, build_startup_readiness_summary
from src.profiler import build_numeric_summary, clear_profile_cache
from src.standards_validator import apply_standards_mapping_overrides, build_standards_override_catalog, build_terminology_override_catalog, validate_healthcare_standards
from src.session_portability import build_session_export_bundle, build_session_export_text, parse_session_import, restore_session_bundle
from src.ui_components import (
    BRAND_SUBTITLE,
    BRAND_TAGLINE,
    apply_theme,
    metric_row,
    plot_bar,
    plot_correlation,
    plot_missingness,
    plot_numeric_box,
    plot_numeric_distribution,
    plot_time_trend,
    plot_top_categories,
    render_app_header,
    render_badge_row,
    render_section_intro,
    render_surface_panel,
    render_sidebar_brand,
    render_sidebar_panel,
    render_sidebar_section,
    render_sidebar_status_panel,
    render_subsection_header,
    render_workflow_steps,
)
from src.usage_analytics import build_customer_success_summary, build_usage_analytics_view
from src.versioning import current_build_label
from ui.common import fmt, info_or_table, log_event, safe_df
from ui.data_intake import render_data_intake
from ui.data_quality import render_quality, render_readiness
from ui.dataset_profile import render_column_detection, render_overview, render_profiling
from ui.healthcare_analytics import render_cohort_analysis, render_healthcare, render_trend_analysis
from ui.insights_export import render_export_center, render_key_insights
from ui.admin_diagnostics import render_admin_diagnostics
from ui.policy_center import render_policy_center
from src.data_intake_support import apply_auto_mapping_suggestions, build_remap_board

APP_TITLE = 'Clinverity'
BUILD_LABEL = current_build_label()
TAB_LABELS = [
    'Data Intake',
    'Overview',
    'Column Detection',
    'Field Profiling',
    'Quality Review',
    'Readiness',
    'Healthcare Intelligence',
    'Trend Analysis',
    'Cohort Analysis',
    'Key Insights',
    'Export Center',
    'Policy Center',
    'Admin Diagnostics',
]

ROLE_OPTIONS = ['Admin', 'Analyst', 'Executive', 'Clinician', 'Researcher', 'Data Steward', 'Viewer']
REPORT_MODES = ['Analyst Report', 'Operational Report', 'Executive Summary', 'Clinical Report', 'Data Readiness Review', 'Population Health Summary']
PLAN_OPTIONS = ['Free', 'Pro', 'Team', 'Enterprise']
PLAN_ENFORCEMENT_OPTIONS = ['Demo-safe', 'Strict', 'Off']
ANALYSIS_CANCELLATION_TIMEOUT_SECONDS = 30

TAB_LABELS = [
    'Data Intake',
    'Overview',
    'Column Detection',
    'Field Profiling',
    'Quality Review',
    'Readiness',
    'Healthcare Intelligence',
    'Trend Analysis',
    'Cohort Analysis',
    'Key Insights',
    'Export Center',
    'Policy Center',
    'Admin Diagnostics',
]

DEFAULT_DEMO_DATASET_NAME = 'Healthcare Operations Demo'


def _get_logger_module():
    global logger_module
    required_attrs = (
        'configure_logging',
        'ensure_platform_log_context',
        'build_log_context',
        'log_platform_event',
        'log_platform_exception',
    )
    if all(hasattr(logger_module, name) for name in required_attrs):
        return logger_module
    logger_module = importlib.reload(importlib.import_module('src.logger'))
    return logger_module


def _get_dataset_service_module():
    global dataset_service
    required_attrs = (
        'load_primary_dataset_from_ui',
        'build_demo_dataset_bundle',
        'build_uploaded_dataset_bundle',
    )
    if all(hasattr(dataset_service, name) for name in required_attrs):
        return dataset_service
    dataset_service = importlib.reload(importlib.import_module('src.services.dataset_service'))
    return dataset_service


def run_pipeline(data: pd.DataFrame, dataset_name: str, source_meta: dict[str, Any]) -> dict[str, Any]:
    from src.pipeline import run_analysis_pipeline

    return run_analysis_pipeline(
        data,
        dataset_name,
        source_meta,
        demo_config={'synthetic_helper_mode': 'Auto'},
        active_control_values={'report_mode': 'Executive Summary'},
    )


def init_state():
    _get_logger_module().configure_logging()
    return initialize_app_session_state(st.session_state)


def _analysis_requires_background(source_meta: dict[str, Any], data: pd.DataFrame) -> bool:
    ingestion_strategy = str(source_meta.get('ingestion_strategy', '')).lower()
    return (
        float(source_meta.get('file_size_mb', 0.0) or 0.0) >= 25.0
        or len(data) >= 150_000
        or 'streaming' in ingestion_strategy
        or str(source_meta.get('sampling_mode', '')).lower() == 'sampled'
    )


def _store_active_dataset_bundle(dataset_selection, *, source_meta_override: dict[str, Any] | None = None) -> None:
    pending_upload_state = st.session_state.pop('pending_uploaded_dataset_state', None)
    effective_source_meta = dict(source_meta_override or dataset_selection.source_meta)
    active_bundle = {
        'bundle_version': 2,
        'active_status': 'active',
        'source_mode': dataset_selection.source_mode,
        'data': dataset_selection.data,
        'original_lookup': dataset_selection.original_lookup,
        'dataset_name': dataset_selection.dataset_name,
        'source_meta': effective_source_meta,
    }
    if dataset_selection.source_mode == 'Uploaded dataset':
        upload_artifact = st.session_state.get('latest_dataset_artifact')
        active_bundle.update(
            {
                'upload_file_name': dataset_selection.dataset_name,
                'upload_size_bytes': int(round(float(effective_source_meta.get('file_size_mb', 0.0) or 0.0) * 1024 ** 2)),
                'upload_artifact': upload_artifact if isinstance(upload_artifact, dict) else None,
                'upload_status': str(effective_source_meta.get('upload_status', 'ready') or 'ready'),
                'dataset_cache_key': str(effective_source_meta.get('dataset_cache_key', '') or ''),
            }
        )
        if isinstance(pending_upload_state, dict):
            active_bundle.update(
                {
                    'upload_file_bytes': pending_upload_state.get('file_bytes'),
                    'upload_size_bytes': int(pending_upload_state.get('file_size_bytes', active_bundle['upload_size_bytes']) or active_bundle['upload_size_bytes']),
                    'upload_artifact': pending_upload_state.get('artifact') or active_bundle.get('upload_artifact'),
                    'upload_status': str(pending_upload_state.get('upload_status', active_bundle['upload_status']) or active_bundle['upload_status']),
                }
            )
    st.session_state['active_dataset_bundle'] = {
        **active_bundle,
    }


def _load_cached_dataset_selection():
    return _get_dataset_service_module().build_selection_from_active_bundle(
        st.session_state.get('active_dataset_bundle'),
        storage_service=st.session_state.get('storage_service'),
    )


def _queue_dataset_source_change(source_mode: str, *, demo_dataset_name: str | None = None) -> None:
    st.session_state['pending_dataset_source_mode'] = source_mode
    if demo_dataset_name:
        st.session_state['pending_demo_dataset_name'] = demo_dataset_name
    st.rerun()


def _render_demo_ready_empty_state() -> None:
    render_app_header(
        title=APP_TITLE,
        subtitle=BRAND_SUBTITLE,
        tagline='Clinical Data Quality & Analytics Platform for Healthcare Datasets',
        build_label=BUILD_LABEL,
        context_items=[
            ('Mode', 'Awaiting dataset'),
            ('Demo dataset', DEFAULT_DEMO_DATASET_NAME),
            ('Workflow', 'Upload or demo start'),
            ('Readiness', 'Calculated after load'),
        ],
    )
    render_section_intro(
        'Get started',
        'Upload a healthcare dataset or launch a guided demo flow to explore readiness, intelligence, and export workflows without leaving the app.',
    )
    render_badge_row(
        [
            ('Demo-ready', 'accent'),
            ('Upload-first workflow', 'info'),
            ('Validation-backed', 'success'),
        ]
    )
    render_surface_panel(
        'Guided Walkthrough',
        '1. Upload dataset  2. View overview  3. Check readiness  4. Explore insights  5. Export reports',
        tone='info',
    )
    action_cols = st.columns([1, 1], gap='medium')
    with action_cols[0]:
        if st.button('Try Demo Dataset', use_container_width=True, type='primary'):
            _queue_dataset_source_change('Built-in example dataset', demo_dataset_name=DEFAULT_DEMO_DATASET_NAME)
    with action_cols[1]:
        st.button('Use Uploaded Dataset', use_container_width=True, disabled=True, help='Switch the sidebar source selector to Uploaded dataset and add a CSV or Excel file.')
    render_surface_panel(
        'No active uploaded dataset',
        'Use the sidebar to choose Uploaded dataset, then add a CSV or Excel extract. If you want an instant walkthrough, start with the built-in healthcare demo.',
    )
    render_workflow_steps(
        [
            ('Upload dataset', 'Use the sidebar uploader for a CSV or Excel healthcare extract.'),
            ('View overview', 'Confirm footprint, structure confidence, and mapped fields.'),
            ('Check readiness', 'Review blockers, trust, and what the dataset can support right now.'),
            ('Explore insights', 'Open healthcare intelligence, trends, and cohort views.'),
            ('Export reports', 'Generate stakeholder-ready outputs once the dataset is ready.'),
        ]
    )
    if DEFAULT_DEMO_DATASET_NAME in DEMO_DATASETS:
        demo = DEMO_DATASETS[DEFAULT_DEMO_DATASET_NAME]
        render_surface_panel(
            'Recommended demo dataset',
            f"{DEFAULT_DEMO_DATASET_NAME}: {demo.get('description', 'Built-in healthcare demo dataset.')}",
            tone='accent',
        )


def _render_active_dataset_status(source_meta: dict[str, Any], pipeline: dict[str, Any]) -> None:
    source_mode = str(source_meta.get('source_mode', 'Unknown'))
    sampling_mode = str(source_meta.get('sampling_mode', 'full') or 'full')
    readiness_score = float(pipeline.get('readiness', {}).get('readiness_score', 0.0) or 0.0)
    quality_score = float(pipeline.get('quality', {}).get('quality_score', 0.0) or 0.0)
    metric_row([
        ('Dataset status', 'Uploaded dataset active' if source_mode == 'Uploaded dataset' else 'Demo dataset active'),
        ('Analysis scope', 'Sampled dataset' if sampling_mode == 'sampled' else 'Full dataset'),
        ('Readiness score', fmt(readiness_score, 'score')),
        ('Quality score', fmt(quality_score)),
    ])
    render_surface_panel(
        'Readiness score explanation',
        (
            'Readiness combines schema support, detected healthcare semantics, quality signals, and module prerequisites. '
            'Sampled runs still preserve authoritative source identity and source row counts.'
        ),
        tone='info',
    )


def _resolve_primary_dataset_selection(active_dataset_service, *, sidebar: Any, ui: Any, session_state: dict[str, Any]):
    return active_dataset_service.resolve_primary_dataset_selection(
        session_state=session_state,
        load_selection=lambda: active_dataset_service.load_primary_dataset_from_ui(
            sidebar=sidebar,
            ui=ui,
            session_state=session_state,
        ),
    )


def _update_analysis_progress(
    job_id: str,
    *,
    progress: float,
    message: str,
    source_meta: dict[str, Any],
    dataset_name: str,
    row_count: int,
    column_count: int,
    started_at: float,
    step_index: int = 1,
    total_steps: int = 5,
    current_operation: str | None = None,
    status: str = 'running',
    cancel_requested: bool = False,
    runtime_key: str | None = None,
) -> dict[str, Any]:
    ignored_jobs = st.session_state.setdefault('ignored_analysis_jobs', set())
    resolved_runtime_key = str(runtime_key or get_job_runtime_key(st.session_state))
    if job_id in ignored_jobs:
        return get_analysis_progress_snapshot(resolved_runtime_key, job_id)
    previous_snapshot = get_analysis_progress_snapshot(resolved_runtime_key, job_id)
    snapshot = build_analysis_progress_snapshot(
        progress=progress,
        message=message,
        current_operation=current_operation or message,
        step_index=step_index,
        total_steps=total_steps,
        started_at=started_at,
        file_size_mb=float(source_meta.get('file_size_mb', 0.0) or 0.0),
        row_count=row_count,
        column_count=column_count,
        status=status,
        cancel_requested=cancel_requested,
        previous_snapshot=previous_snapshot,
        source_mode=str(source_meta.get('source_mode', '')),
        dataset_name=dataset_name,
    )
    store_analysis_progress_snapshot(resolved_runtime_key, job_id, snapshot)
    return snapshot


def _render_analysis_progress_ui(job_id: str, status: dict[str, Any]) -> None:
    snapshot = get_analysis_progress_snapshot(get_job_runtime_key(st.session_state), job_id)
    percent_complete = int(snapshot.get('percent_complete', 0))
    eta_seconds = int(round(float(snapshot.get('estimated_remaining_seconds', 0.0) or 0.0)))
    eta_uncertainty = int(round(float(snapshot.get('eta_uncertainty_seconds', 0.0) or 0.0)))
    eta_confidence = str(snapshot.get('eta_confidence', 'Low'))
    st.subheader('Large dataset analysis in progress')
    st.progress(percent_complete, text=snapshot.get('message', 'Preparing analysis...'))
    st.caption(f"Current operation: {snapshot.get('current_operation', 'Preparing analysis context')}")
    st.caption(
        f"Estimated time remaining: {eta_seconds} seconds (\u00b1{eta_uncertainty}s, {eta_confidence} confidence)"
    )
    st.caption(f"Completion: {percent_complete}%")
    for index, label in enumerate(ANALYSIS_STAGE_LABELS, start=1):
        if index < int(snapshot.get('step_index', 1)):
            prefix = '[Done]'
        elif index == int(snapshot.get('step_index', 1)):
            prefix = '[Active]'
        else:
            prefix = '[Pending]'
        st.write(f'{index}. {label} {prefix}')
    if status.get('status') in {'queued', 'running'}:
        if st.button('Stop analysis', key=f'stop_analysis_{job_id}'):
            cancel_background_task(st.session_state, job_id)
            requested_at = time.monotonic()
            st.session_state.setdefault('analysis_cancellation', {})[job_id] = {
                'requested_at': requested_at,
                'timeout_seconds': ANALYSIS_CANCELLATION_TIMEOUT_SECONDS,
                'status': 'pending',
            }
            _update_analysis_progress(
                job_id,
                progress=float(snapshot.get('progress', 0.0)),
                message=f'Cancellation in progress ({ANALYSIS_CANCELLATION_TIMEOUT_SECONDS} seconds remaining)',
                source_meta=st.session_state.get('active_dataset_bundle', {}).get('source_meta', {}),
                dataset_name=str(st.session_state.get('active_dataset_bundle', {}).get('dataset_name', 'Current dataset')),
                row_count=len(st.session_state.get('active_dataset_bundle', {}).get('data', pd.DataFrame())),
                column_count=len(st.session_state.get('active_dataset_bundle', {}).get('data', pd.DataFrame()).columns) if isinstance(st.session_state.get('active_dataset_bundle', {}).get('data'), pd.DataFrame) else 0,
                started_at=requested_at,
                step_index=int(snapshot.get('step_index', 1)),
                total_steps=int(snapshot.get('total_steps', 5)),
                current_operation='Cancelling the managed analysis job',
                status='cancelling',
                cancel_requested=True,
            )
            st.rerun()


def _clear_active_analysis_job() -> None:
    active_job = st.session_state.pop('active_analysis_job', None)
    if isinstance(active_job, dict):
        job_id = active_job.get('job_id')
        if job_id:
            clear_analysis_progress_snapshot(get_job_runtime_key(st.session_state), str(job_id))
            st.session_state.get('analysis_progress', {}).pop(job_id, None)
            st.session_state.get('analysis_cancellation', {}).pop(job_id, None)


def _set_analysis_status_notice(level: str, message: str, *, remediation_message: Any | None = None) -> None:
    st.session_state['analysis_status_notice'] = {'level': level, 'message': message, 'remediation_message': remediation_message}


def _render_analysis_status_notice() -> None:
    notice = st.session_state.pop('analysis_status_notice', None)
    if not isinstance(notice, dict):
        return
    level = str(notice.get('level', 'info')).lower()
    message = str(notice.get('message', '')).strip()
    if not message:
        return
    if level == 'success':
        st.success(message)
    elif level == 'warning':
        st.warning(message)
    else:
        st.info(message)
    remediation_message = notice.get('remediation_message')
    if remediation_message is not None:
        _render_actionable_message(remediation_message)


def _build_active_dataset_diagnostics(dataset_selection, pipeline: dict[str, Any] | None = None) -> dict[str, Any]:
    active_bundle = st.session_state.get('active_dataset_bundle', {})
    data = dataset_selection.data if dataset_selection is not None else None
    source_meta = dataset_selection.source_meta if dataset_selection is not None else {}
    diagnostics = {
        'active_dataset_source': str(source_meta.get('source_mode', 'Unknown')),
        'dataset_identifier': str(
            source_meta.get('dataset_cache_key', '')
            or getattr(data, 'attrs', {}).get('dataset_identifier', '')
            or getattr(data, 'attrs', {}).get('dataset_cache_key', '')
            or dataset_selection.dataset_name
        ),
        'dataset_name': str(dataset_selection.dataset_name if dataset_selection is not None else ''),
        'row_count': int(len(data)) if isinstance(data, pd.DataFrame) else 0,
        'column_count': int(len(data.columns)) if isinstance(data, pd.DataFrame) else 0,
        'dataset_cache_key': str(source_meta.get('dataset_cache_key', '') or getattr(data, 'attrs', {}).get('dataset_cache_key', '')),
        'active_bundle_exists': bool(isinstance(active_bundle, dict) and active_bundle),
        'bundle_active_status': str((active_bundle or {}).get('active_status', 'inactive') or 'inactive'),
        'upload_loading_status': str(source_meta.get('upload_status', (active_bundle or {}).get('upload_status', 'ready')) or 'ready'),
        'bundle_has_upload_bytes': bool((active_bundle or {}).get('upload_file_bytes')),
        'bundle_has_upload_artifact': bool((active_bundle or {}).get('upload_artifact')),
    }
    if isinstance(pipeline, dict):
        runtime_diag = pipeline.get('dataset_runtime_diagnostics', {})
        if isinstance(runtime_diag, dict):
            diagnostics.update(
                {
                    'pipeline_dataset_identifier': str(runtime_diag.get('dataset_identifier', '')),
                    'pipeline_row_count': int(runtime_diag.get('row_count', diagnostics['row_count']) or diagnostics['row_count']),
                    'pipeline_column_count': int(runtime_diag.get('column_count', diagnostics['column_count']) or diagnostics['column_count']),
                    'pipeline_cache_identity_preserved': bool(runtime_diag.get('cache_identity_preserved', False)),
                }
            )
    return diagnostics


def _analysis_cancellation_state(job_id: str) -> dict[str, Any]:
    return st.session_state.setdefault('analysis_cancellation', {}).setdefault(job_id, {})


def _begin_analysis_cancellation(job_id: str) -> dict[str, Any]:
    state = _analysis_cancellation_state(job_id)
    state.setdefault('requested_at', time.monotonic())
    state.setdefault('timeout_seconds', ANALYSIS_CANCELLATION_TIMEOUT_SECONDS)
    state['status'] = 'pending'
    return state


def _update_cancellation_countdown(job_id: str, snapshot: dict[str, Any], *, source_meta: dict[str, Any], row_count: int, column_count: int) -> dict[str, Any]:
    state = _analysis_cancellation_state(job_id)
    requested_at = float(state.get('requested_at', time.monotonic()))
    timeout_seconds = int(state.get('timeout_seconds', ANALYSIS_CANCELLATION_TIMEOUT_SECONDS))
    elapsed_seconds = max(time.monotonic() - requested_at, 0.0)
    remaining_seconds = max(timeout_seconds - int(math.floor(elapsed_seconds)), 0)
    current_operation = (
        f'Force-killing job (timeout in {remaining_seconds}s)'
        if remaining_seconds <= 15
        else 'Cancelling the managed analysis job'
    )
    return _update_analysis_progress(
        job_id,
        progress=float(snapshot.get('progress', 0.0)),
        message=f'Cancellation in progress ({remaining_seconds} seconds remaining)',
        source_meta=source_meta,
        dataset_name=str(st.session_state.get('active_dataset_bundle', {}).get('dataset_name', 'Current dataset')),
        row_count=row_count,
        column_count=column_count,
        started_at=requested_at,
        step_index=int(snapshot.get('step_index', 1) or 1),
        total_steps=int(snapshot.get('total_steps', 5) or 5),
        current_operation=current_operation,
        status='cancelling',
        cancel_requested=True,
    )


def _cleanup_cancelled_analysis_runtime(job_id: str, *, clear_profile_state: bool) -> None:
    st.session_state.setdefault('ignored_analysis_jobs', set()).add(job_id)
    clear_analysis_progress_snapshot(get_job_runtime_key(st.session_state), job_id)
    st.session_state.get('analysis_progress', {}).pop(job_id, None)
    st.session_state.get('analysis_cancellation', {}).pop(job_id, None)
    if clear_profile_state:
        clear_profile_cache(st.session_state.setdefault('profile_cache_metrics', {}))
    st.session_state['generated_report_outputs'] = {}
    st.session_state.pop('latest_dataset_artifact', None)
    active_bundle = st.session_state.get('active_dataset_bundle')
    if isinstance(active_bundle, dict):
        data = active_bundle.get('data')
        if isinstance(data, pd.DataFrame):
            data.attrs.pop('profile_cache_metrics', None)
    gc.collect()
    _clear_active_analysis_job()


def _force_cancel_managed_analysis(job_id: str, *, reason: str) -> dict[str, Any]:
    active_bundle = st.session_state.get('active_dataset_bundle', {})
    active_data = active_bundle.get('data') if isinstance(active_bundle, dict) else None
    source_meta = active_bundle.get('source_meta', {}) if isinstance(active_bundle, dict) else {}
    forced = force_cancel_background_task(st.session_state, job_id, reason=reason)
    _cleanup_cancelled_analysis_runtime(job_id, clear_profile_state=True)
    _set_analysis_status_notice(
        'success',
        'Cancellation completed. Analysis cancelled. Ready for new dataset.',
        remediation_message=build_cancellation_timeout_message(
            file_size_mb=float(source_meta.get('file_size_mb', 0.0) or 0.0),
            row_count=len(active_data) if isinstance(active_data, pd.DataFrame) else 0,
            column_count=len(active_data.columns) if isinstance(active_data, pd.DataFrame) else 0,
        ),
    )
    return forced


def _extract_pipeline_context(args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    candidate = kwargs.get('pipeline')
    if isinstance(candidate, dict) and 'semantic' in candidate and 'readiness' in candidate:
        return candidate
    for arg in args:
        if isinstance(arg, dict) and 'semantic' in arg and 'readiness' in arg:
            return arg
    return {}


def _offer_actionable_auto_fix(section_name: str, pipeline: dict[str, Any]) -> None:
    dataset_name = str(st.session_state.get('active_dataset_bundle', {}).get('dataset_name', '') or 'current-dataset')
    dataset_cache_key = str(
        pipeline.get('dataset_runtime_diagnostics', {}).get('dataset_cache_key', '')
        or st.session_state.get('active_dataset_bundle', {}).get('dataset_cache_key', '')
        or dataset_name
    )
    remap_key = f"remap_board::{dataset_name}"
    if st.button('Auto-fix likely field mappings', key=f'fallback_autofix::{section_name}'):
        updated_board = apply_auto_mapping_suggestions(build_remap_board(pipeline))
        st.session_state[remap_key] = updated_board
        st.session_state.setdefault('semantic_mapping_overrides_by_dataset', {})[dataset_cache_key] = {
            str(row.get('mapped_field', '')).strip(): str(row.get('source_column', '')).strip()
            for row in updated_board.to_dict(orient='records')
            if str(row.get('mapped_field', '')).strip() and str(row.get('mapped_field', '')).strip() != 'Not mapped'
        }
        st.success('Auto-fix suggestions were prepared in Data Intake > Field Remapping Studio. Review them there, then rerun the analysis for this section.')


def _render_actionable_message(message: Any) -> None:
    severity = str(getattr(message, 'severity', 'Error'))
    module = str(getattr(message, 'module', 'Module'))
    timestamp = str(getattr(message, 'timestamp', ''))
    icon = {
        'Warning': '⚠',
        'Error': '🔴',
        'Critical': '🔴',
    }.get(severity, '⚠')
    header = f"{icon} {severity.upper()}: {module}"
    if timestamp:
        header = f'{header} | {timestamp}'
    error_fn = getattr(st, 'error', None)
    warning_fn = getattr(st, 'warning', None)
    info_fn = getattr(st, 'info', None)
    write_fn = getattr(st, 'write', None)
    markdown_fn = getattr(st, 'markdown', None)
    if severity == 'Critical' and callable(error_fn):
        error_fn(header)
    elif severity == 'Warning' and callable(warning_fn):
        warning_fn(header)
    elif callable(error_fn):
        error_fn(header)
    elif callable(warning_fn):
        warning_fn(header)
    elif callable(info_fn):
        info_fn(header)
    elif callable(write_fn):
        write_fn(header)
    if callable(markdown_fn):
        markdown_fn(f"**ISSUE:** {getattr(message, 'issue', '')}")
        markdown_fn(f"**CAUSE:** {getattr(message, 'cause', '')}")
        markdown_fn('**REMEDIATION:**')
    elif callable(write_fn):
        write_fn(f"ISSUE: {getattr(message, 'issue', '')}")
        write_fn(f"CAUSE: {getattr(message, 'cause', '')}")
        write_fn('REMEDIATION:')
    for index, step in enumerate(getattr(message, 'remediations', []), start=1):
        if callable(markdown_fn):
            markdown_fn(f'{index}. {step}')
        elif callable(write_fn):
            write_fn(f'{index}. {step}')
    doc_links = list(getattr(message, 'doc_links', []) or [])
    if doc_links:
        if callable(markdown_fn):
            markdown_fn('**LEARN MORE:**')
        elif callable(write_fn):
            write_fn('LEARN MORE:')
        for link in doc_links:
            if callable(markdown_fn):
                markdown_fn(f"- {link.get('title', 'Guide')}: `{link.get('path', '')}`")
            elif callable(write_fn):
                write_fn(f"{link.get('title', 'Guide')}: {link.get('path', '')}")


def _render_actionable_fallback(section_name: str, pipeline: dict[str, Any]) -> None:
    context = build_actionable_fallback_context(section_name, pipeline)
    _render_actionable_message(context.get('message'))
    st.write('Required fields for this view:')
    info_or_table(
        safe_df(context.get('required_fields_table')),
        'Required-field guidance is not available for this section yet.',
    )
    if context.get('missing_fields'):
        st.warning('Missing fields: ' + ', '.join(context.get('missing_fields', [])))
    st.write('Likely field mapping suggestions:')
    info_or_table(
        safe_df(context.get('mapping_suggestions')),
        'No confident source-column suggestions are available yet for the missing fields.',
    )
    st.write('Example structure that unlocks this view:')
    info_or_table(
        safe_df(context.get('example_structure')),
        'Example structure guidance is not available yet.',
    )
    st.write('Documentation and sample datasets:')
    info_or_table(
        safe_df(context.get('docs_table')),
        'Documentation links are not available yet.',
    )
    if bool(context.get('auto_fix_available')):
        _offer_actionable_auto_fix(section_name, pipeline)











def render_safely(section_name: str, render_fn, *args, **kwargs) -> None:
    render_started = time.monotonic()
    try:
        visited = st.session_state.setdefault('visited_sections', [])
        if section_name not in visited:
            visited.append(section_name)
            log_event('Module Visited', f"Rendered '{section_name}' in the current session.", 'Module navigation', section_name)
        render_fn(*args, **kwargs)
        metric = visualization_performance.record_tab_render_metric(
            st.session_state,
            section_name,
            time.monotonic() - render_started,
        )
        if 'Trend Analysis' in section_name or 'Cohort Analysis' in section_name:
            target_status = 'Target met' if metric['meets_target'] else 'Above target'
            st.caption(f"Tab render time: {metric['duration_ms']} ms | target < 500 ms | {target_status}")
    except Exception as error:
        active_logger = _get_logger_module()
        pipeline_context = _extract_pipeline_context(args, kwargs)
        active_bundle = st.session_state.get('active_dataset_bundle', {})
        active_source_meta = active_bundle.get('source_meta', {}) if isinstance(active_bundle, dict) else {}
        active_data = active_bundle.get('data') if isinstance(active_bundle, dict) else None
        row_count = int(pipeline_context.get('overview', {}).get('rows', len(active_data) if isinstance(active_data, pd.DataFrame) else 0))
        column_count = int(
            pipeline_context.get('overview', {}).get(
                'columns',
                len(active_data.columns) if isinstance(active_data, pd.DataFrame) else 0,
            )
        )
        active_logger.log_platform_exception(
            'ui_section_render_failed',
            error,
            logger_name='ui',
            section_name=section_name,
            operation_type='ui_section_render',
            duration_seconds=round(time.monotonic() - render_started, 3),
            dataset_name=str(active_bundle.get('dataset_name', '')),
            source_mode=str(active_source_meta.get('source_mode', '')),
            file_size_mb=float(active_source_meta.get('file_size_mb', 0.0) or 0.0),
            row_count=row_count,
            column_count=column_count,
            **active_logger.build_log_context(st.session_state),
        )
        log_event('Module Fallback', f"{section_name} switched to a protected fallback view because {type(error).__name__} was raised.", 'Protected render', section_name)
        if pipeline_context:
            _render_actionable_fallback(section_name, pipeline_context)
        else:
            st.error(f"{section_name} could not render because {type(error).__name__} was raised before the dataset context was available.")
        st.info('What still works: the rest of the platform remains available, including overview, readiness, healthcare summaries, and exports that are already supported by the current dataset.')
        st.info('What to try next: review Dataset Intelligence and Analysis Readiness, confirm field mappings, or add the missing dates, identifiers, outcomes, or grouping fields needed by this section.')
        st.caption('A protected fallback was shown instead of a raw runtime error so the workflow can continue safely.')










































































def main() -> None:
    active_logger = _get_logger_module()
    active_dataset_service = _get_dataset_service_module()
    st.set_page_config(page_title=APP_TITLE, layout='wide')
    apply_theme()
    services = init_state()
    request_context = active_logger.ensure_platform_log_context(st.session_state)
    active_logger.log_platform_event(
        'app_request_started',
        logger_name='runtime',
        **active_logger.build_log_context(st.session_state, **request_context),
    )
    application_service = services.application_service
    _render_analysis_status_notice()
    render_sidebar_brand(
        st.sidebar,
        workspace_name=str(st.session_state.get('workspace_name', 'Guest Demo Workspace')),
        source_mode='Demo-safe workspace' if st.session_state.get('product_demo_mode_enabled', True) else 'Configured workspace',
        version_note='Clinical review shell',
        build_label=BUILD_LABEL,
    )
    render_sidebar_section(st.sidebar, 'Dataset session')
    render_sidebar_panel(
        st.sidebar,
        'Source & upload',
        'Choose a built-in example or keep working from the active uploaded dataset. Uploaded sessions remain active until you explicitly switch sources.',
    )
    render_sidebar_section(st.sidebar, 'Processing & remediation')
    with st.sidebar.expander('Processing configuration', expanded=False):
        st.selectbox('Synthetic helper fields', ['Auto', 'On', 'Off'], key='demo_synthetic_helper_mode')
        st.selectbox('BMI remediation mode', ['median', 'clip', 'null', 'none'], key='demo_bmi_remediation_mode')
        st.selectbox('Synthetic cost support', ['Auto', 'On', 'Off'], key='demo_synthetic_cost_mode')
        st.selectbox('Synthetic readmission support', ['Auto', 'On', 'Off'], key='demo_synthetic_readmission_mode')
        st.selectbox('Executive summary verbosity', ['Concise', 'Detailed'], key='demo_executive_summary_verbosity')
        st.selectbox('Scenario simulation mode', ['Basic', 'Expanded'], key='demo_scenario_simulation_mode')
        st.caption('These settings control demo-safe helper behavior and summary presentation without changing the source data.')
    active_analysis_job = st.session_state.get('active_analysis_job')
    if isinstance(active_analysis_job, dict):
        dataset_selection = _load_cached_dataset_selection()
        if dataset_selection is None:
            _clear_active_analysis_job()
            dataset_selection = _resolve_primary_dataset_selection(
                active_dataset_service,
                sidebar=st.sidebar,
                ui=st,
                session_state=st.session_state,
            )
    else:
        dataset_selection = _resolve_primary_dataset_selection(
            active_dataset_service,
            sidebar=st.sidebar,
            ui=st,
            session_state=st.session_state,
        )
    data = dataset_selection.data
    original_lookup = dataset_selection.original_lookup
    dataset_name = dataset_selection.dataset_name
    source_meta = dict(dataset_selection.source_meta)
    dataset_cache_key = str(
        source_meta.get('dataset_cache_key', '')
        or getattr(data, 'attrs', {}).get('dataset_cache_key', '')
        or dataset_name
    )
    semantic_overrides_by_dataset = st.session_state.setdefault('semantic_mapping_overrides_by_dataset', {})
    manual_semantic_overrides = semantic_overrides_by_dataset.get(dataset_cache_key, {})
    if manual_semantic_overrides:
        source_meta['manual_semantic_overrides'] = dict(manual_semantic_overrides)
        source_meta['manual_semantic_override_count'] = len(manual_semantic_overrides)
    if data is None:
        _render_demo_ready_empty_state()
        return
    render_sidebar_section(st.sidebar, 'Current session')
    render_sidebar_status_panel(
        st.sidebar,
        dataset_name=dataset_name,
        source_mode=str(source_meta.get('source_mode', 'Unknown')),
        row_count=fmt(len(data)),
        column_count=fmt(len(data.columns)),
    )
    _store_active_dataset_bundle(dataset_selection, source_meta_override=source_meta)
    st.session_state['active_dataset_diagnostics'] = _build_active_dataset_diagnostics(dataset_selection)
    if dataset_selection.source_mode == 'Uploaded dataset':
        upload_log_key = f"{dataset_name}:{source_meta.get('file_size_mb', 0.0)}"
        if st.session_state.get('last_logged_upload_key') != upload_log_key:
            log_event(
                'Dataset Uploaded',
                f"Uploaded dataset '{dataset_name}' into workspace '{st.session_state.get('workspace_identity', {}).get('workspace_name', 'Guest Demo Workspace')}'.",
                'Dataset upload',
                'Data intake',
                resource_type='dataset',
                resource_name=dataset_name,
            )
            st.session_state['last_logged_upload_key'] = upload_log_key
    progress_placeholder = st.empty()
    progress_bar = st.progress(0, text='Preparing analysis context...')
    analysis_started_at = time.monotonic()

    render_surface_panel(
        'Active workflow',
        'Move from intake to readiness, analytics, remediation, and export from a single clinically oriented review flow.',
    )
    render_workflow_steps(
        [
            ('Intake', 'Capture the active dataset, confirm lineage, and review the current session context.'),
            ('Assess', 'Profile fields, detect quality issues, and validate analysis readiness.'),
            ('Analyze', 'Use healthcare intelligence, trend views, cohorts, and downstream insights.'),
            ('Export', 'Generate audit-ready reports, bundles, and governed handoff artifacts.'),
        ]
    )

    def _progress_update(value: float, message: str, **metadata: Any) -> None:
        progress_bar.progress(int(max(0.0, min(1.0, value)) * 100), text=message)
        progress_placeholder.caption(message)
        _update_analysis_progress(
            st.session_state.get('active_analysis_job', {}).get('job_id', 'sync-analysis'),
            progress=value,
            message=message,
            source_meta=source_meta,
            dataset_name=dataset_name,
            row_count=len(data),
            column_count=len(data.columns),
            started_at=analysis_started_at,
            step_index=int(metadata.get('step_index', 1)),
            total_steps=int(metadata.get('total_steps', 5)),
            current_operation=str(metadata.get('current_operation', message)),
        )

    analysis_result = None
    use_background_analysis = _analysis_requires_background(source_meta, data)
    if use_background_analysis:
        current_job = st.session_state.get('active_analysis_job')
        if isinstance(current_job, dict) and current_job.get('dataset_name') != dataset_name:
            stale_job_id = str(current_job.get('job_id', '')).strip()
            if stale_job_id:
                active_logger.log_platform_event(
                    'analysis_job_force_cancelled_for_dataset_switch',
                    logger_name='runtime',
                    operation_type='analysis_pipeline',
                    previous_dataset_name=str(current_job.get('dataset_name', '')),
                    next_dataset_name=dataset_name,
                    job_id=stale_job_id,
                    **active_logger.build_log_context(st.session_state),
                )
                _force_cancel_managed_analysis(
                    stale_job_id,
                    reason='The previous large-dataset analysis was force-killed after the dataset selection changed.',
                )
            current_job = None
        if not isinstance(current_job, dict) or current_job.get('dataset_name') != dataset_name:
            managed_runtime = dict(st.session_state.get('job_runtime') or {})
            if not bool(managed_runtime.get('backend_configured')):
                managed_runtime.update(
                    {
                        'backend_configured': True,
                        'mode': 'worker',
                        'max_workers': 1,
                        'status_label': 'Threaded worker backend',
                    }
                )
            job_started_at = time.monotonic()
            current_job_id = f'analysis_pipeline-{int(job_started_at * 1000)}'
            runtime_key = get_job_runtime_key(st.session_state)

            def _job_progress(value: float, message: str, **metadata: Any) -> None:
                _update_analysis_progress(
                    current_job_id,
                    progress=value,
                    message=message,
                    source_meta=source_meta,
                    dataset_name=dataset_name,
                    row_count=len(data),
                    column_count=len(data.columns),
                    started_at=job_started_at,
                    step_index=int(metadata.get('step_index', 1)),
                    total_steps=int(metadata.get('total_steps', 5)),
                    current_operation=str(metadata.get('current_operation', message)),
                    runtime_key=runtime_key,
                )

            def _cancel_check() -> bool:
                return job_cancel_requested(runtime_key, current_job_id)

            execution_context = application_service.build_execution_context(st.session_state)

            submission = submit_background_task(
                st.session_state,
                job_runtime=managed_runtime,
                job_id=current_job_id,
                task_key='analysis_pipeline',
                task_label='Large dataset analysis',
                detail=f'Analyzing dataset {dataset_name}',
                runner=lambda: application_service.execute_analysis_run(
                    execution_context,
                    data=data,
                    dataset_name=dataset_name,
                    source_meta=source_meta,
                    progress_callback=_job_progress,
                    cancel_check=_cancel_check,
                    persist_runtime_state=False,
                ),
            )
            current_job_id = submission['job_id']
            st.session_state['active_analysis_job'] = {
                'job_id': current_job_id,
                'dataset_name': dataset_name,
            }
            _update_analysis_progress(
                current_job_id,
                progress=0.02,
                message='1. Loading data...',
                source_meta=source_meta,
                dataset_name=dataset_name,
                row_count=len(data),
                column_count=len(data.columns),
                started_at=job_started_at,
                step_index=1,
                total_steps=5,
                current_operation='Queuing the managed analysis job',
            )
            current_job = st.session_state['active_analysis_job']
        job_state = read_background_task(st.session_state, current_job['job_id'])
        status = job_state['status']
        cancellation_state = st.session_state.get('analysis_cancellation', {}).get(current_job['job_id'], {})
        if bool(status.get('cancel_requested')) or cancellation_state:
            snapshot = get_analysis_progress_snapshot(get_job_runtime_key(st.session_state), current_job['job_id'])
            snapshot = _update_cancellation_countdown(
                current_job['job_id'],
                snapshot,
                source_meta=source_meta,
                row_count=len(data),
                column_count=len(data.columns),
            )
            requested_at = float(cancellation_state.get('requested_at', time.monotonic()))
            timeout_seconds = int(cancellation_state.get('timeout_seconds', ANALYSIS_CANCELLATION_TIMEOUT_SECONDS))
            if (time.monotonic() - requested_at) >= timeout_seconds and status.get('status') not in {'completed', 'failed', 'cancelled'}:
                active_logger.log_platform_event(
                    'analysis_job_force_cancel_requested',
                    logger_name='runtime',
                    operation_type='analysis_pipeline',
                    dataset_name=dataset_name,
                    job_id=current_job['job_id'],
                    timeout_seconds=timeout_seconds,
                    requested_at=requested_at,
                    **active_logger.build_log_context(st.session_state),
                )
                _force_cancel_managed_analysis(
                    current_job['job_id'],
                    reason='Cancellation timed out after 30 seconds and the analysis job was force-killed.',
                )
                st.rerun()
        if status.get('status') == 'completed' and job_state.get('result') is not None:
            analysis_result = job_state['result']
            application_service.apply_completed_analysis_result(
                st.session_state,
                analysis_result=analysis_result,
            )
            _clear_active_analysis_job()
        elif status.get('status') == 'failed':
            progress_bar.empty()
            progress_placeholder.empty()
            _clear_active_analysis_job()
            st.error('The platform could not complete a full analysis for this dataset. Review the production hardening guidance below and try a smaller or cleaner extract.')
            st.info('If the file came from a spreadsheet export, review date columns, duplicate headers, and fully empty columns before trying again.')
            st.info('Support detail: internal diagnostic information was captured for this analysis failure. Review the dataset structure, then try a narrower or cleaner extract.')
            return
        elif status.get('status') == 'cancelled':
            progress_bar.empty()
            progress_placeholder.empty()
            _cleanup_cancelled_analysis_runtime(current_job['job_id'], clear_profile_state=False)
            _set_analysis_status_notice('success', 'Cancellation completed. Analysis cancelled. Ready for new dataset.')
            st.success('Cancellation completed')
            st.info('Analysis cancelled. Ready for new dataset.')
            return
        else:
            progress_bar.empty()
            progress_placeholder.empty()
            _render_analysis_progress_ui(current_job['job_id'], status)
            time.sleep(0.4)
            st.rerun()
    else:
        analysis_started = time.monotonic()
        try:
            analysis_result = application_service.execute_analysis_run(
                st.session_state,
                data=data,
                dataset_name=dataset_name,
                source_meta=source_meta,
                progress_callback=_progress_update,
            )
        except Exception as error:
            active_logger.log_platform_exception(
                'analysis_pipeline_failed',
                error,
                logger_name='runtime',
                operation_type='analysis_pipeline',
                dataset_name=dataset_name,
                source_mode=source_meta.get('source_mode', ''),
                file_size_mb=float(source_meta.get('file_size_mb', 0.0) or 0.0),
                row_count=len(data),
                column_count=len(data.columns),
                duration_seconds=round(time.monotonic() - analysis_started, 3),
                **active_logger.build_log_context(st.session_state),
            )
            progress_bar.empty()
            progress_placeholder.empty()
            st.error('The platform could not complete a full analysis for this dataset. Review the production hardening guidance below and try a smaller or cleaner extract.')
            st.info('If the file came from a spreadsheet export, review date columns, duplicate headers, and fully empty columns before trying again.')
            st.info('Support detail: internal diagnostic information was captured for this analysis failure. Review the dataset structure, then try a narrower or cleaner extract.')
            return
    preflight = analysis_result.preflight
    column_validation = analysis_result.column_validation
    if analysis_result.blocked:
        progress_bar.empty()
        progress_placeholder.empty()
        st.error(preflight.get('block_reason', 'This dataset exceeds the safe interactive limits for the current demo environment.'))
        info_or_table(safe_df(preflight.get('checks_table')), 'No preflight details are available.')
        st.info('Try a smaller extract, fewer columns, or a narrower date range before re-running the analysis.')
        return
    for warning in preflight.get('warnings', []):
        st.warning(warning)
    if analysis_result.empty_column_warning:
        st.warning(analysis_result.empty_column_warning)
        with st.expander('View column health details'):
            st.caption(str(column_validation.get('summary', '')))
            info_or_table(safe_df(column_validation.get('checks_table')), 'No column validation checks are available yet.')
            info_or_table(safe_df(column_validation.get('issue_samples')), 'No column validation issues were sampled for the current dataset.')
    for warning in analysis_result.other_column_warnings:
        st.warning(warning)
    if analysis_result.long_task_notice:
        st.info(analysis_result.long_task_notice)
    if bool((pipeline := (analysis_result.pipeline or {})).get('sample_info', {}).get('sampling_applied')):
        _render_actionable_message(
            build_large_dataset_actionable_message(
                source_meta,
                pipeline.get('sample_info', {}),
            )
        )
    log_event('Dataset Selected', f"Selected dataset '{dataset_name}'.", 'Dataset selection', 'Data intake')
    progress_bar.empty()
    progress_placeholder.caption('Analysis context is ready. Use the tabs below to explore the current dataset.')
    pipeline = analysis_result.pipeline or {}
    evolution_summary = dict(pipeline.get('evolution_summary', {}))
    if evolution_summary.get('available'):
        auto_execution_queue = queue_execution_items(
            st.session_state.get('evolution_execution_queue', []),
            evolution_summary.get('proposal_queue_table'),
            dataset_name=dataset_name,
            dataset_family_key=str(evolution_summary.get('dataset_family_key', 'generic-healthcare')),
        )
        st.session_state['evolution_execution_queue'] = auto_execution_queue
        updated_memory = merge_evolution_memory(
            st.session_state.get('evolution_memory', {}),
            evolution_summary,
            dataset_name=dataset_name,
        )
        updated_memory['execution_queue'] = list(auto_execution_queue)
        st.session_state['evolution_memory'] = updated_memory
        save_evolution_memory(
            updated_memory,
            storage_service=st.session_state.get('storage_service'),
            workspace_identity=st.session_state.get('workspace_identity', {}),
        )
        if application_service is not None:
            application_service.persist_user_settings(st.session_state)
        evolution_summary['memory_snapshot'] = updated_memory
        pipeline['evolution_summary'] = evolution_summary
    st.session_state['active_dataset_diagnostics'] = _build_active_dataset_diagnostics(dataset_selection, pipeline)
    visualization_performance.warm_healthcare_visualization_payloads(st.session_state, pipeline)
    demo_usage_seed = analysis_result.demo_usage_seed or {}
    if demo_usage_seed.get('seeded'):
        for event in demo_usage_seed.get('events', []):
            log_event(
                str(event.get('event_type', 'Demo mode seed')),
                str(event.get('details', 'Seeded demo usage event.')),
                str(event.get('user_interaction', 'Demo mode seed')),
                str(event.get('analysis_step', 'Demo onboarding')),
            )
        st.session_state['demo_usage_seeded_keys'] = list(demo_usage_seed.get('seeded_keys', []))
    active_logger.log_platform_event(
        'analysis_pipeline_ready',
        logger_name='runtime',
        dataset_name=dataset_name,
        dataset_identifier=str(st.session_state.get('active_dataset_diagnostics', {}).get('dataset_identifier', '')),
        source_mode=source_meta.get('source_mode', ''),
        row_count=int(st.session_state.get('active_dataset_diagnostics', {}).get('pipeline_row_count', len(data))),
        column_count=int(st.session_state.get('active_dataset_diagnostics', {}).get('pipeline_column_count', len(data.columns))),
        **active_logger.build_log_context(st.session_state),
    )
    with st.sidebar.expander('Developer diagnostics', expanded=False):
        diagnostics = dict(st.session_state.get('active_dataset_diagnostics', {}))
        diagnostics['current_visual_cache_size'] = len(st.session_state.get('visualization_payload_cache', {}))
        diagnostic_lines = [
            f"active_dataset_source: {diagnostics.get('source_mode', source_meta.get('source_mode', ''))}",
            f"dataset_name: {diagnostics.get('dataset_name', dataset_name)}",
            f"dataset_identifier: {diagnostics.get('dataset_identifier', '')}",
            f"dataset_cache_key: {diagnostics.get('dataset_cache_key', source_meta.get('dataset_cache_key', ''))}",
            f"row_count: {diagnostics.get('row_count', len(data))}",
            f"column_count: {diagnostics.get('column_count', len(data.columns))}",
            f"manual_semantic_override_count: {diagnostics.get('manual_semantic_override_count', 0)}",
            f"manual_semantic_override_summary: {diagnostics.get('manual_semantic_override_summary', '')}",
            f"trend_cache_belongs_to_current_dataset: {diagnostics.get('trend_cache_belongs_to_current_dataset', '')}",
            f"cohort_cache_belongs_to_current_dataset: {diagnostics.get('cohort_cache_belongs_to_current_dataset', '')}",
        ]
        for line in diagnostic_lines:
            st.caption(line)
        st.json(diagnostics, expanded=False)
        info_or_table(pd.DataFrame([diagnostics]), 'Active dataset diagnostics will appear here once a dataset is selected.')
    landing = pipeline['landing_summary']
    is_demo_dataset = str(source_meta.get('source_mode', '')) == 'Demo dataset'

    render_app_header(
        title=APP_TITLE,
        subtitle=BRAND_SUBTITLE,
        tagline=BRAND_TAGLINE,
        build_label=BUILD_LABEL,
        context_items=[
            ('Dataset', dataset_name),
            ('Readiness', fmt(pipeline['readiness']['readiness_score'], 'score')),
            ('Quality score', fmt(pipeline['quality']['quality_score'])),
            ('Rows in scope', fmt(pipeline['overview']['rows'])),
        ],
    )
    _render_active_dataset_status(source_meta, pipeline)
    if is_demo_dataset and st.session_state.get('product_demo_mode_enabled', True):
        st.info('Demo mode is active. Use the guided path below to move from intake and remediation through readiness review, clinical intelligence, and stakeholder-ready exports.')
    render_section_intro(
        'Clinical Readiness Workspace',
        landing.get('subheadline', 'Profile clinical datasets, resolve quality blockers, validate readiness, and package audit-ready outputs in one workflow.'),
    )
    render_badge_row(
        [
            ('Clinical data quality', 'info'),
            ('Remediation-ready', 'accent'),
            ('Audit-friendly', 'success'),
            ('Export-governed', 'warning'),
        ]
    )
    if landing.get('audience_summary') or landing.get('positioning_statement'):
        render_surface_panel(
            'Platform positioning',
            ' '.join(
                part.strip()
                for part in [
                    str(landing.get('audience_summary', '') or ''),
                    str(landing.get('positioning_statement', '') or ''),
                ]
                if part and str(part).strip()
            ),
            tone='info',
        )
    workflow_steps = [
        (str(row.get('step', 'Workflow step')), str(row.get('summary', '')))
        for row in landing.get('four_step_workflow', [])
        if isinstance(row, dict)
    ]
    if workflow_steps:
        render_subsection_header('Workflow', 'Move from intake to governed reporting without leaving the active dataset context.')
        render_workflow_steps(workflow_steps)
    summary_cols = st.columns([1.1, 0.9], gap='large')
    with summary_cols[0]:
        render_subsection_header('Teams this fits best')
        for line in landing.get('who_its_for', []):
            st.write(f'- {line}')
    with summary_cols[1]:
        render_surface_panel(
            'Why Clinverity stands out',
            str(landing.get('platform_value_summary', 'Clinical readiness, remediation, and export workflows remain aligned in one platform.')),
            tone='accent',
        )
    metric_row([(card['label'], card['value']) for card in landing.get('product_value_cards', [])[:4]])
    if landing.get('onboarding_cues'):
        with st.expander('Operator notes', expanded=False):
            for line in landing.get('onboarding_cues', [])[:3]:
                st.write(f'- {line}')
    if is_demo_dataset:
        render_subsection_header('Guided Demo Path', 'Use a shorter story arc for demos, first meetings, and quick product walkthroughs.')
        demo_flow = landing.get('startup_demo_flow', {})
        if isinstance(demo_flow, dict) and demo_flow:
            render_surface_panel(
                'Best demo dataset',
                f"{demo_flow.get('best_dataset', 'Healthcare Operations Demo')} - {demo_flow.get('dataset_reason', 'This dataset activates the strongest healthcare walkthrough.')}",
                tone='accent',
            )
            st.caption(str(demo_flow.get('intro', 'Use this guided demo path to show the strongest product flow.')))
            if demo_flow.get('estimated_demo_time'):
                st.caption(f"Estimated demo time: {demo_flow.get('estimated_demo_time')}")
            if demo_flow.get('quick_start_steps'):
                render_subsection_header('Quick start path')
                for index, line in enumerate(demo_flow.get('quick_start_steps', []), start=1):
                    st.write(f'{index}. {line}')
            render_subsection_header('Recommended tabs')
            for index, line in enumerate(demo_flow.get('recommended_tabs', []), start=1):
                st.write(f'{index}. {line}')
            render_subsection_header('Suggested AI Copilot prompts')
            for prompt in demo_flow.get('suggested_copilot_prompts', []):
                st.write(f'- {prompt}')
            render_surface_panel(
                'Best export to generate',
                f"{demo_flow.get('recommended_export', 'Executive Summary')} - {demo_flow.get('export_reason', 'Use the export flow to finish the walkthrough with a stakeholder-ready output.')}",
            )
            if demo_flow.get('demo_outcome'):
                st.caption(f"Demo outcome: {demo_flow.get('demo_outcome')}")
    render_subsection_header('Early Access & Pilot', 'Capture beta demand and keep pilot next steps attached to the current workflow.')
    beta_conversion = build_beta_conversion_panel(
        bool(st.session_state.get('product_beta_interest_enabled', True)),
        st.session_state.get('beta_interest_submissions', []),
    )
    st.write(f"**{beta_conversion.get('headline', 'Join the beta')}**")
    st.caption(str(beta_conversion.get('subheadline', '')))
    metric_row([(card['label'], card['value']) for card in beta_conversion.get('summary_cards', [])[:3]])
    cta_cols = st.columns(len(beta_conversion.get('cta_cards', [])) or 1)
    for idx, cta in enumerate(beta_conversion.get('cta_cards', [])):
        col = cta_cols[idx % len(cta_cols)]
        col.write(f"**{cta.get('label', 'Beta CTA')}**")
        col.caption(str(cta.get('note', '')))
        if cta.get('cta_summary'):
            col.write(str(cta.get('cta_summary')))
        if col.button(cta.get('label', 'Open beta flow'), key=f"beta_cta_{idx}", use_container_width=True):
            with st.spinner(f"Opening {cta.get('label', 'beta flow').lower()}..."):
                st.session_state['beta_interest_use_case'] = str(cta.get('preset_use_case', ''))
                st.session_state['beta_interest_focus_note'] = (
                    f"{cta.get('label', 'Beta CTA')} is ready below in Data Intake > Beta Interest / Early Access."
                )
                st.session_state['beta_interest_feedback'] = {
                    'status': 'success',
                    'message': f"{cta.get('label', 'Beta CTA')} is ready. Complete the short form below to save the request.",
                }
            st.rerun()
    if beta_conversion.get('conversion_steps'):
        with st.expander('Beta workflow details', expanded=False):
            for index, line in enumerate(beta_conversion.get('conversion_steps', []), start=1):
                st.write(f'{index}. {line}')
    st.caption(str(beta_conversion.get('status_note', '')))
    if beta_conversion.get('reassurance_note'):
        st.caption(str(beta_conversion.get('reassurance_note')))
    if landing.get('report_polish_cues'):
        with st.expander('Report handoff notes', expanded=False):
            for line in landing.get('report_polish_cues', [])[:3]:
                st.write(f'- {line}')
    if st.session_state.get('beta_interest_focus_note'):
        st.info(str(st.session_state.get('beta_interest_focus_note')))
    render_subsection_header('Product Narrative', 'Keep deeper positioning, pilot framing, and architecture notes available without crowding the main workflow.')
    narrative = landing.get('investor_demo_narrative', {})
    if isinstance(narrative, dict) and narrative:
        with st.expander('Product and demo narrative', expanded=False):
            st.write(f"**The problem:** {narrative.get('problem', '')}")
            st.write('**Platform workflow**')
            for line in narrative.get('workflow', []):
                st.write(f'- {line}')
            st.write('**Why healthcare teams care**')
            for line in narrative.get('value_for_teams', []):
                st.write(f'- {line}')
            st.write('**Top modules to highlight in a demo**')
            for line in narrative.get('top_modules', []):
                st.write(f'- {line}')
            st.write('**Sample outputs to show**')
            for line in narrative.get('sample_outputs', []):
                st.write(f'- {line}')
            st.write('**Recommended demo path**')
            for index, line in enumerate(narrative.get('recommended_demo_path', []), start=1):
                st.write(f'{index}. {line}')
    design_partner = landing.get('design_partner_mode', {})
    if isinstance(design_partner, dict) and design_partner:
        with st.expander('Design partner readiness', expanded=False):
            st.write('**Current capabilities**')
            for line in design_partner.get('capabilities_now', []):
                st.write(f'- {line}')
            st.write('**Configurable workflows**')
            for line in design_partner.get('configurable_workflows', []):
                st.write(f'- {line}')
            st.write('**Where integrations can plug in later**')
            for line in design_partner.get('future_integration_points', []):
                st.write(f'- {line}')
            st.write('**How a pilot can be structured**')
            for index, line in enumerate(design_partner.get('pilot_structure', []), start=1):
                st.write(f'{index}. {line}')
            st.caption(str(design_partner.get('positioning_note', '')))
    roi = landing.get('roi_value_estimation', {})
    if isinstance(roi, dict) and roi.get('cards'):
        render_subsection_header('Value signals')
        metric_row([(card['label'], card['value']) for card in roi.get('cards', [])[:4]])
        with st.expander('Value estimation details', expanded=False):
            for card in roi.get('cards', []):
                st.caption(f"{card.get('label', 'Value estimate')}: {card.get('detail', '')}")
            for line in roi.get('notes', []):
                st.write(f'- {line}')
    market_views = pipeline.get('market_solution_views', {})
    market_table = safe_df(market_views.get('market_solution_views'))
    if not market_table.empty:
        render_subsection_header('Solution views', str(market_views.get('narrative', '')))
        render_surface_panel(
            'Recommended buyer view',
            str(market_views.get('recommended_solution_view', 'Healthcare Data Readiness')),
        )
        info_or_table(
            market_table[
                [
                    'solution_view',
                    'current_fit',
                    'best_fit_workflow',
                    'best_fit_package',
                ]
            ],
            'No market-specific solution summary is available yet.',
        )
    pilot_toolkit = landing.get('pilot_readiness_toolkit', {})
    if isinstance(pilot_toolkit, dict) and pilot_toolkit:
        with st.expander('Pilot toolkit', expanded=False):
            sections = [
                ('Prepare data', 'prepare_data'),
                ('Run the first analysis', 'run_first_analysis'),
                ('Reports to generate', 'reports_to_generate'),
                ('How to interpret results', 'interpret_results'),
                ('How to evaluate success', 'evaluate_success'),
            ]
            for label, key in sections:
                st.write(f'**{label}**')
                for line in pilot_toolkit.get(key, []):
                    st.write(f'- {line}')
            st.caption(str(pilot_toolkit.get('pilot_note', '')))
    architecture = landing.get('architecture_signals', {})
    if isinstance(architecture, dict) and architecture:
        with st.expander('Platform architecture', expanded=False):
            sections = [
                ('Workflow layers', 'workflow_layers'),
                ('Product layers', 'product_layers'),
                ('Scalability signals', 'scalability_signals'),
                ('Future integration slots', 'future_integration_slots'),
            ]
            for label, key in sections:
                st.write(f'**{label}**')
                for line in architecture.get(key, []):
                    st.write(f'- {line}')
            st.caption(str(architecture.get('technical_evaluator_note', '')))
    pitch = landing.get('startup_pitch_polish', {})
    if isinstance(pitch, dict) and pitch:
        with st.expander('Executive positioning', expanded=False):
            sections = [
                ('Premium product framing', 'premium_copy'),
                ('Value statements', 'value_statements'),
                ('Use-case positioning', 'use_case_positioning'),
                ('Workflow messaging', 'workflow_messaging'),
            ]
            for label, key in sections:
                st.write(f'**{label}**')
                for line in pitch.get(key, []):
                    st.write(f'- {line}')
            st.caption(str(pitch.get('pitch_note', '')))
    differentiators = safe_df(landing.get('differentiators'))
    if not differentiators.empty:
        render_subsection_header('Why Clinverity stands out')
        for row in differentiators.to_dict(orient='records'):
            st.write(f"**{row.get('title', 'Platform capability')}** - {row.get('summary', '')}")
            st.caption(str(row.get('why_it_matters', '')))
    render_subsection_header('Recommended starting paths')
    for line in landing.get('recommended_starting_paths', []):
        st.write(f'- {line}')
    if is_demo_dataset:
        render_subsection_header('Built-in demo datasets')
        info_or_table(build_demo_dataset_cards(), 'Built-in demo dataset guidance is not available yet.')
    render_subsection_header('Platform coverage')
    metric_row([(badge['label'], badge['value']) for badge in landing.get('capability_badges', [])[:4]])
    for line in landing.get('analysis_covers', []):
        st.write(f'- {line}')
    render_subsection_header('Current system status')
    for line in landing.get('system_status', []):
        st.write(f'- {line}')
    st.caption(landing.get('synthetic_support_note', ''))
    metric_row([
        ('Rows', fmt(pipeline['overview']['rows'])),
        ('Analyzed Columns', fmt(pipeline['overview'].get('analyzed_columns', pipeline['overview']['columns']))),
        ('Data Quality Score', fmt(pipeline['quality']['quality_score'])),
        ('Analysis Readiness', fmt(pipeline['readiness']['readiness_score'], 'score')),
    ])
    if int(pipeline['overview'].get('helper_columns_added', 0)) > 0:
        st.caption(
            f"Source columns: {fmt(pipeline['overview'].get('source_columns', pipeline['overview']['columns']))}. "
            f"Analyzed columns: {fmt(pipeline['overview'].get('analyzed_columns', pipeline['overview']['columns']))}. "
            'Derived or helper fields are disclosed in the workflow notes below.'
        )

    tabs = st.tabs(TAB_LABELS)
    with tabs[0]:
        render_safely('Data Intake', render_data_intake, pipeline, dataset_name, source_meta)
    with tabs[1]:
        render_safely('Dataset Profile · Overview', render_overview, pipeline)
    with tabs[2]:
        render_safely('Dataset Profile · Column Detection', render_column_detection, pipeline, original_lookup)
    with tabs[3]:
        render_safely('Dataset Profile · Field Profiling', render_profiling, pipeline)
    with tabs[4]:
        render_safely('Data Quality · Quality Review', render_quality, pipeline)
    with tabs[5]:
        render_safely('Data Quality · Analysis Readiness', render_readiness, pipeline)
    with tabs[6]:
        render_safely('Healthcare Analytics · Healthcare Intelligence', render_healthcare, pipeline)
    with tabs[7]:
        render_safely('Healthcare Analytics · Trend Analysis', render_trend_analysis, pipeline)
    with tabs[8]:
        render_safely('Healthcare Analytics · Cohort Analysis', render_cohort_analysis, pipeline)
    with tabs[9]:
        render_safely('Insights & Export · Key Insights', render_key_insights, pipeline, dataset_name)
    with tabs[10]:
        render_safely('Insights & Export · Export Center', render_export_center, pipeline, source_meta=source_meta, dataset_name=dataset_name)
    with tabs[11]:
        render_safely('Policy Center', render_policy_center, pipeline, dataset_name, source_meta)
    with tabs[12]:
        render_safely('Admin Diagnostics', render_admin_diagnostics, pipeline, dataset_name, source_meta)


if __name__ == '__main__':
    main()


