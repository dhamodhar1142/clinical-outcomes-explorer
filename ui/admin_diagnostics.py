from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from src.logger import build_support_diagnostics
from src.export_orchestrator import build_export_execution_plan, build_export_runtime_profile, recommended_report_label
from src.storage import build_storage_backend_health
from src.ui_components import metric_row, render_section_intro, render_subsection_header, render_surface_panel
from src.validation_orchestrator import build_validation_execution_plan, build_validation_runtime_profile, run_recommended_validation
from src.versioning import current_build_label
from ui.common import info_or_table, safe_df


def render_admin_diagnostics(pipeline: dict[str, Any], dataset_name: str, source_meta: dict[str, Any]) -> None:
    render_section_intro(
        'Admin Diagnostics',
        'Review runtime health, adaptive backlog state, and execute recommended validation from a single operator surface.',
    )
    storage_health = build_storage_backend_health(st.session_state.get('storage_service'))
    support_diagnostics = build_support_diagnostics()
    evolution_summary = dict(pipeline.get('evolution_summary', {}))
    active_diagnostics = dict(st.session_state.get('active_dataset_diagnostics', {}))
    review_store = dict(st.session_state.get('dataset_review_approvals', {}))
    dataset_identifier = str(
        pipeline.get('dataset_runtime_diagnostics', {}).get('dataset_identifier', '')
        or source_meta.get('dataset_identifier', '')
        or active_diagnostics.get('dataset_identifier', '')
        or dataset_name
    )
    review_entry = dict(review_store.get(dataset_identifier, {}))
    validation_recommendations = safe_df(evolution_summary.get('validation_recommendations_table'))
    validation_execution_plan = safe_df(build_validation_execution_plan(validation_recommendations))
    validation_runtime_profile = build_validation_runtime_profile()
    plan_awareness = pipeline.get('plan_awareness', {})
    role = str(st.session_state.get('active_role') or st.session_state.get('workspace_role') or 'Analyst')
    export_allowed = bool(pipeline.get('export_summary', {}).get('available', True))
    strict_plan = bool(plan_awareness.get('strict_enforcement'))
    active_plan = str(plan_awareness.get('active_plan', 'Pro'))
    advanced_exports_allowed = export_allowed and (active_plan != 'Starter' or not strict_plan)
    governance_exports_allowed = export_allowed and (active_plan in {'Pro', 'Enterprise'} or not strict_plan)
    stakeholder_bundle_allowed = export_allowed and (active_plan != 'Starter' or not strict_plan)
    export_runtime_profile = build_export_runtime_profile(pipeline)
    export_execution_plan = safe_df(
        build_export_execution_plan(
            pipeline,
            role=role,
            export_allowed=export_allowed,
            advanced_exports_allowed=advanced_exports_allowed,
            governance_exports_allowed=governance_exports_allowed,
            stakeholder_bundle_allowed=stakeholder_bundle_allowed,
        )
    )
    backlog_summary = safe_df(evolution_summary.get('backlog_summary_table'))
    drift_alerts = safe_df(evolution_summary.get('drift_alerts_table'))
    recent_runs = safe_df(pd.DataFrame(st.session_state.get('validation_orchestration_runs', [])))

    metric_row(
        [
            ('Build', current_build_label()),
            ('Storage health', str(storage_health.get('status', 'Unknown'))),
            ('Tracked items', str(len(st.session_state.get('evolution_execution_queue', [])))),
            ('Dataset cache key', str(active_diagnostics.get('dataset_cache_key', source_meta.get('dataset_cache_key', '')))[:16] or 'Unavailable'),
        ]
    )
    render_surface_panel(
        'Runtime health',
        str(storage_health.get('detail', 'Runtime diagnostics will appear here once the storage service is initialized.')),
        tone='info' if str(storage_health.get('status', '')).lower() == 'healthy' else 'warning',
    )

    diagnostics_table = pd.DataFrame(
        [
            {'signal': 'Dataset name', 'value': dataset_name},
            {'signal': 'Source mode', 'value': str(source_meta.get('source_mode', 'Unknown'))},
            {'signal': 'Dataset identifier', 'value': dataset_identifier},
            {'signal': 'Persistence enabled', 'value': 'Yes' if st.session_state.get('application_service') is not None and st.session_state.get('application_service').enabled else 'No'},
            {'signal': 'Storage mode', 'value': str(storage_health.get('mode', 'session'))},
            {'signal': 'Storage target', 'value': str(storage_health.get('storage_target', 'session-only'))},
            {'signal': 'Release signoff', 'value': str(review_entry.get('release_signoff_status', 'Pending'))},
            {'signal': 'Review history entries', 'value': str(len(review_entry.get('review_history', [])))},
        ]
    )
    info_or_table(diagnostics_table, 'Admin diagnostics will appear here once runtime context is available.')

    render_subsection_header('Recommended validation orchestration')
    render_surface_panel(
        'Validation runtime profile',
        str(validation_runtime_profile.get('detail', 'Validation runtime guidance will appear here once the environment profile is available.')),
        tone='info',
    )
    info_or_table(
        validation_execution_plan if not validation_execution_plan.empty else validation_recommendations,
        'Recommended validation gates will appear here once Clinverity has enough trust and readiness context.',
    )
    runnable_validations = (
        validation_execution_plan.loc[validation_execution_plan['allowed'].astype(bool), 'validation'].astype(str).tolist()
        if not validation_execution_plan.empty and 'allowed' in validation_execution_plan.columns
        else validation_recommendations['validation'].astype(str).tolist() if not validation_recommendations.empty else []
    )
    if runnable_validations:
        selected_validation = st.selectbox(
            'Recommended validation',
            runnable_validations,
            key=f'admin_validation_runner::{dataset_identifier}',
        )
        if st.button('Run Recommended Validation', key=f'admin_run_validation::{dataset_identifier}', type='primary'):
            result = run_recommended_validation(selected_validation)
            runs = list(st.session_state.get('validation_orchestration_runs', []))
            runs.insert(0, result)
            st.session_state['validation_orchestration_runs'] = runs[:25]
            if result.get('status') == 'Passed':
                st.success(f"{selected_validation} completed successfully.")
            else:
                st.error(f"{selected_validation} failed. Review the execution log below.")
            st.rerun()
    elif not validation_execution_plan.empty:
        st.info('No recommended validation is runnable in the current runtime. Use a local or staging environment for the heavier suggested gates.')

    render_subsection_header('Export runtime orchestration')
    render_surface_panel(
        'Export runtime profile',
        str(export_runtime_profile.get('detail', 'Export runtime guidance will appear here once the environment profile is available.')),
        tone='info',
    )
    info_or_table(
        export_execution_plan,
        'Recommended export execution guidance will appear here once the current role, plan, and dataset context are available.',
    )
    if not export_execution_plan.empty:
        st.caption(f"Recommended report target: {recommended_report_label(role, pipeline)}")

    render_subsection_header('Adaptive backlog and drift')
    info_or_table(
        backlog_summary,
        'Backlog summary will appear here when adaptive execution work has been queued.',
    )
    info_or_table(
        drift_alerts,
        'Drift alerts will appear here when repeated dataset-family runs meaningfully diverge from prior learned patterns.',
    )

    render_subsection_header('Execution history')
    info_or_table(
        recent_runs[
            ['validation_name', 'task', 'status', 'started_at', 'finished_at', 'artifact_path', 'markdown_report']
        ] if not recent_runs.empty else recent_runs,
        'Validation orchestration history will appear here after the first recommended validation is run from the app.',
    )
    if not recent_runs.empty:
        with st.expander('Latest validation execution log', expanded=False):
            latest = dict(recent_runs.iloc[0].to_dict())
            st.code(str(latest.get('stdout', '')).strip() or 'No stdout captured.')
            if str(latest.get('stderr', '')).strip():
                st.code(str(latest.get('stderr', '')).strip())

    render_subsection_header('Support diagnostics')
    info_or_table(
        safe_df(pd.DataFrame([{'note': note} for note in list(storage_health.get('notes', []))])),
        'Storage notes will appear here when backend guidance is available.',
    )
    info_or_table(
        safe_df(support_diagnostics.get('diagnostics_table')),
        'Recent support diagnostics will appear here when platform events have been captured.',
    )
