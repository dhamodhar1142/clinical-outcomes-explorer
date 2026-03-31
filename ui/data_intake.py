from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

import src.auth as auth_module
from src.beta_interest import (
    BETA_INTEREST_STATUS_OPTIONS,
    BETA_INTEREST_STORAGE_OPTIONS,
    beta_interest_csv_bytes,
    build_beta_interest_summary,
    merge_beta_interest_submissions,
    save_beta_interest_submission,
    support_email,
    update_beta_interest_status,
    validate_beta_interest,
)
from src.collaboration_notes import append_collaboration_note, build_collaboration_note_summary, build_collaboration_notes_view, build_note_target_options
from src.data_intake_support import (
    CANONICAL_FIELDS,
    PLAN_ENFORCEMENT_OPTIONS,
    PLAN_OPTIONS,
    REPORT_MODES,
    ROLE_OPTIONS,
    TAB_LABELS,
    apply_auto_mapping_suggestions,
    apply_mapping_template,
    build_mapping_template,
    build_data_profiling_summary,
    build_deployment_support_notes,
    build_lineage_sankey,
    build_mapping_confidence_table,
    build_remap_board,
)
from src.data_loader import DEMO_DATASETS, DataLoadError, load_uploaded_file
from src.deployment_readiness import build_config_guidance, build_launch_checklist
from src.evolution_engine import (
    append_review_history,
    apply_execution_autopilot,
    build_dataset_version_diff,
    build_execution_autopilot_actions,
    build_release_readiness_summary,
    queue_execution_items,
    record_outcome_feedback,
    update_execution_item_status,
)
from src.evolution_memory_store import save_evolution_memory
from src.enterprise_features import (
    build_audit_log_view,
    build_data_lineage_view,
    build_dataset_comparison_dashboard,
    build_join_recommendation,
    build_workflow_pack_details,
    build_workflow_pack_summary,
    cohort_monitoring_over_time,
    detect_join_candidates,
    infer_linked_dataset_role,
    preview_linked_merge,
)
from src.logger import build_support_diagnostics
from src.mapping_profiles import (
    available_mapping_profiles,
    build_mapping_profile,
    build_profile_suggestion_table,
    infer_dataset_family,
    profile_mapping_for_columns,
    suggest_mapping_profile,
)
from src.modules.privacy_security import (
    REDACTION_LEVEL_OPTIONS,
    WORKSPACE_EXPORT_ACCESS_OPTIONS,
    build_export_governance_summary,
    get_export_policy_presets,
)
from src.plan_awareness import is_strict_plan_enforcement
from src.product_settings import DEFAULT_PRODUCT_SETTINGS, LARGE_DATASET_PROFILES, build_product_settings_summary
from src.profiler import build_field_profile, build_profile_cache_summary, build_quality_checks, clear_profile_cache
from src.readiness_engine import evaluate_analysis_readiness
from src.schema_detection import detect_structure
from src.semantic_mapper import CANONICAL_FIELDS, build_data_dictionary, infer_semantic_mapping
from src.session_portability import build_session_export_bundle, build_session_export_text, parse_session_import, restore_session_bundle
from src.storage import build_storage_backend_health
from src.services.workspace_service import (
    add_collaboration_note_to_workspace,
    load_snapshot_into_session,
    load_workflow_pack_into_session,
    save_snapshot_to_workspace,
    save_workflow_pack_to_workspace,
)
from src.solution_layers import build_market_specific_solution_views
from src.temporal_detection import augment_temporal_fields
from src.ui_components import metric_row, render_advanced_sections_toggle, render_badge_row, render_role_context_panel, render_section_intro, render_subsection_header, render_surface_panel, render_workflow_steps
from src.validation_orchestrator import build_validation_execution_plan, run_recommended_validation
from src.versioning import current_build_label
from src.workspace import persist_active_workspace_state
from ui.common import build_demo_dataset_cards, build_recommended_workflow_component, fmt, info_or_table, log_event, safe_df


def active_controls() -> dict[str, Any]:
    application_service = st.session_state.get('application_service')
    if application_service is not None:
        return application_service.build_active_controls(st.session_state)
    return {
        key: st.session_state.get(key)
        for key in [
            'analysis_template',
            'report_mode',
            'export_policy_name',
            'active_role',
            'accuracy_benchmark_profile',
            'accuracy_reporting_threshold_profile',
            'accuracy_reporting_min_trust_score',
            'accuracy_allow_directional_external_reporting',
            'active_plan',
            'plan_enforcement_mode',
            'workflow_action_prompt',
            'demo_dataset_name',
        ]
        if key in st.session_state
    }



def render_data_intake(pipeline: dict[str, Any], dataset_name: str, source_meta: dict[str, str]) -> None:
    is_demo_dataset = str(source_meta.get('source_mode', '')) == 'Demo dataset'
    render_section_intro(
        'Data Intake',
        'Manage ingestion, lineage, field mapping, workspace controls, and onboarding workflows from a single clinical review surface.',
    )
    render_badge_row(
        [
            ('Authoritative dataset context', 'info'),
            ('Lineage-aware', 'accent'),
            ('Workspace-governed', 'success'),
        ]
    )
    structure = pipeline['structure']
    temporal_context = pipeline.get('temporal_context', {})
    intelligence = pipeline.get('dataset_intelligence', {})
    plan_awareness = pipeline.get('plan_awareness', {})
    application_service = st.session_state.get('application_service')
    workspace_identity = st.session_state.get('workspace_identity', {})
    render_surface_panel(
        'Intake workflow',
        'Start with source selection, schema review, lineage, and workflow guidance before branching into workspace controls, linked analysis, or export preparation.',
        tone='info',
    )
    advanced_sections_enabled = render_advanced_sections_toggle(
        'data_intake',
        help_text='Expand the dense workspace, governance, and operational intake sections by default for admin-style review.',
    )
    render_subsection_header('Data ingestion wizard')
    next_step = (
        f"Address the top blocker first: {pipeline['remediation'].iloc[0]['issue']}. Recommended fix: {pipeline['remediation'].iloc[0]['recommended_fix']}"
        if not pipeline['remediation'].empty
        else 'The dataset is already in a strong enough position for guided profiling, readiness review, and stakeholder export preparation.'
    )
    render_workflow_steps(
        [
            ('Source selection', 'Choose a built-in demo or upload a CSV/Excel file. Original field names remain available for audit review.'),
            ('Schema review', f"Detected {len(structure.detection_table)} fields with an average structure confidence of {structure.confidence_score:.1%}."),
            ('Readiness summary', f"{pipeline['readiness']['available_count']} modules are fully ready and {pipeline['readiness']['partial_count']} are partially ready."),
            ('Recommended next step', next_step),
        ]
    )
    if temporal_context.get('synthetic_date_created'):
        st.info(str(temporal_context.get('note', 'A synthetic event_date was generated to support temporal analysis.')))

    workflow = build_recommended_workflow_component(pipeline)
    render_subsection_header('Recommended workflow', 'Use the recommended package and workflow as the canonical path for the current dataset.')
    metric_row([
        ('Dataset Type', workflow.get('dataset_type', 'Dataset type not classified'), 'Full inferred dataset classification used to guide the workflow.'),
        ('Recommended Workflow', workflow.get('recommended_workflow', 'Healthcare Data Readiness'), 'Best-fit walkthrough based on the current schema and healthcare support.'),
        ('Recommended Package', workflow.get('recommended_package', 'Healthcare Data Readiness'), 'Suggested stakeholder package for this dataset context.'),
    ])
    st.caption(str(workflow.get('rationale', 'Use this guided flow to move from dataset understanding to stakeholder-ready outputs.')))
    render_surface_panel(
        'Workflow guidance',
        'Follow the recommended flow below to move from intake through governed analysis and stakeholder-ready output.'
    )
    workflow_steps = [(step.split('. ', 1)[-1], 'Recommended progression for the current dataset context.') for step in workflow.get('steps', [])]
    if workflow_steps:
        render_workflow_steps(workflow_steps)

    if is_demo_dataset:
        render_subsection_header('Built-in demo datasets', 'Use the recommended workflow above as the main walkthrough path, then choose the dataset that best matches the story you want to show.')
        info_or_table(build_demo_dataset_cards(), 'Built-in demo dataset guidance appears here when the startup dataset catalog is available for walkthrough selection.')

    st.markdown('### Data Lineage Visualization')
    lineage = pipeline.get('lineage', {}) or build_data_lineage_view(
        dataset_name,
        source_meta,
        pipeline.get('semantic', {}),
        pipeline.get('readiness', {}),
        active_controls(),
    )
    lineage_chart = build_lineage_sankey(lineage)
    if lineage_chart is not None:
        st.plotly_chart(lineage_chart, width='stretch')
    metric_row([
        ('Mapped Fields', fmt(len(safe_df(lineage.get('derived_fields_table'))))),
        ('Source Mode', str(source_meta.get('source_mode', 'Unknown'))),
        ('Active Controls', fmt(len(safe_df(lineage.get('active_controls_table'))))),
        ('Readiness Blocks', fmt(len(pipeline.get('readiness', {}).get('major_blockers', [])))),
    ])
    info_or_table(safe_df(lineage.get('source_table')), 'Lineage source details will appear here when dataset source metadata is available.')
    info_or_table(safe_df(lineage.get('derived_fields_table')), 'Derived field lineage appears here when semantic mapping has identified canonical downstream roles.')
    info_or_table(safe_df(lineage.get('active_controls_table')), 'Active analysis controls appear here after the runtime initializes the current review context.')
    with st.expander('Metric lineage and approval state', expanded=False):
        info_or_table(
            safe_df(lineage.get('metric_lineage_table')),
            'Metric lineage will appear here when result-accuracy scoring has mapped module support and reporting gates.',
        )
        info_or_table(
            safe_df(lineage.get('approval_table')),
            'Approval workflow state will appear here when mapping, trust, and export gates have been reviewed for this dataset.',
        )
    for step in lineage.get('transformation_steps', [])[:6]:
        st.caption(step)

    st.markdown('### Schema Detection Confidence')
    detection_table = safe_df(structure.detection_table)
    metric_row([
        ('Structure Confidence', f"{structure.confidence_score:.1%}"),
        ('Numeric Fields', fmt(len(structure.numeric_columns))),
        ('Date Fields', fmt(len(structure.date_columns))),
        ('Identifier Fields', fmt(len(structure.identifier_columns))),
    ])
    info_or_table(
        detection_table[['column_name', 'inferred_type', 'confidence_score', 'evidence', 'null_percentage']].head(25)
        if not detection_table.empty else detection_table,
        'Schema detection details appear here once the dataset has been profiled.',
    )

    st.markdown('### Column Mapping Interface')
    mapping_table = build_mapping_confidence_table(pipeline)
    data_dictionary = build_data_dictionary(structure, pipeline.get('semantic', {}))
    metric_row([
        ('Mapped Canonical Fields', fmt(len(mapping_table))),
        ('Avg Mapping Confidence', f"{float(mapping_table['confidence_score'].mean()) if not mapping_table.empty else 0.0:.1%}"),
        ('Downstream Ready', fmt(int(mapping_table['used_downstream'].sum()) if not mapping_table.empty else 0)),
        ('Healthcare Mapping Score', f"{float(pipeline.get('semantic', {}).get('healthcare_readiness_score', 0.0) or 0.0):.1%}"),
    ])
    info_or_table(mapping_table, 'Column-to-canonical mappings appear here once semantic detection has enough signal to propose confident roles.')
    with st.expander('Data dictionary with mapping confidence'):
        info_or_table(data_dictionary, 'The data dictionary appears here when field profiling and semantic mapping are available.')

    st.markdown('### Field Remapping Studio')
    st.caption('Use this remapping board to review or override semantic mappings. The order column behaves like a drag-and-drop priority lane for this demo-safe workflow.')
    remap_key = f"remap_board::{dataset_name}"
    remap_notice_key = f"{remap_key}::notice"
    dataset_cache_key = str(
        pipeline.get('dataset_runtime_diagnostics', {}).get('dataset_cache_key', '')
        or source_meta.get('dataset_cache_key', '')
        or dataset_name
    )
    if remap_key not in st.session_state:
        st.session_state[remap_key] = build_remap_board(pipeline)
    template_store = st.session_state.setdefault('mapping_templates', {})
    profile_store = st.session_state.setdefault('semantic_mapping_profiles', {})
    available_profiles = available_mapping_profiles(profile_store)
    dataset_family = infer_dataset_family(pipeline['data'].columns, dataset_name)
    suggested_profile = suggest_mapping_profile(
        pipeline['data'].columns,
        dataset_name=dataset_name,
        user_profiles=profile_store,
    )
    remap_board = safe_df(st.session_state.get(remap_key))
    render_surface_panel(
        'Dataset-family recognition',
        (
            f"Detected family: {dataset_family.get('family_label', 'Generic Healthcare Feed')} "
            f"with suggested benchmark profile {dataset_family.get('benchmark_profile', 'Generic Healthcare')}."
        ),
        tone='info',
    )
    evolution_summary = dict(pipeline.get('evolution_summary', {}))
    if evolution_summary.get('available'):
        render_subsection_header(
            'Adaptive evolution engine',
            'Clinverity tracks recurring weak spots and turns them into ranked improvement opportunities instead of changing production logic on its own.',
        )
        metric_row([
            ('Opportunities', fmt(evolution_summary.get('opportunity_count', 0))),
            ('High priority', fmt(evolution_summary.get('high_priority_count', 0))),
            ('Recurring patterns', fmt(evolution_summary.get('recurring_pattern_count', 0))),
            ('Auto actions ready', fmt(len(safe_df(evolution_summary.get('auto_actions_table'))))),
        ])
        render_surface_panel(
            'Adaptive goal',
            (
                f"Clinverity is currently optimizing this dataset family toward {str(evolution_summary.get('goal_profile', 'Healthcare intelligence expansion')).lower()}. "
                f"{str(evolution_summary.get('goal_rationale', 'It keeps learning from recurring outcomes in this dataset family.'))}"
            ),
            tone='accent',
        )
        st.caption(str(evolution_summary.get('summary_text', '')))
        info_or_table(
            safe_df(evolution_summary.get('opportunities_table')),
            'Adaptive improvement opportunities appear here after Clinverity has enough trust, readiness, and mapping context to rank them.',
        )
        metric_row([
            ('Field remediation items', fmt(len(safe_df(evolution_summary.get('field_remediation_table'))))),
            ('Feedback records', fmt(int(dict(evolution_summary.get('outcome_feedback_summary', {})).get('feedback_count', 0)))),
            ('Helpful rate', fmt(float(dict(evolution_summary.get('outcome_feedback_summary', {})).get('helpful_rate', 0.0)), 'pct')),
            ('Suggested validation gates', fmt(len(safe_df(evolution_summary.get('validation_recommendations_table'))))),
        ])
        metric_row([
            ('Tracked backlog', fmt(int(dict(evolution_summary.get('backlog_summary', {})).get('total_items', 0)))),
            ('Release blockers', fmt(int(dict(evolution_summary.get('backlog_summary', {})).get('release_blockers', 0)))),
            ('Open high priority', fmt(int(dict(evolution_summary.get('backlog_summary', {})).get('high_priority_open', 0)))),
            ('Family run history', fmt(len(safe_df(evolution_summary.get('drift_history_table'))))),
        ])
        info_or_table(
            safe_df(evolution_summary.get('field_remediation_table')),
            'Field-level remediation suggestions appear here when Clinverity can estimate the most useful schema and mapping fixes for this dataset family.',
        )
        validation_recommendations = safe_df(evolution_summary.get('validation_recommendations_table'))
        if not validation_recommendations.empty and st.button(
            'Queue recommended validation plan',
            key=f'queue_validation_plan::{dataset_name}',
        ):
            validation_queue = list(st.session_state.get('evolution_execution_queue', []))
            for row in validation_recommendations.to_dict(orient='records'):
                validation_queue = queue_execution_items(
                    validation_queue,
                    pd.DataFrame(
                        [
                            {
                                'proposal_title': f"{row.get('validation', 'Validation')} follow-up",
                                'proposal_type': 'Validation plan',
                                'priority': row.get('priority', 'Medium'),
                                'proposed_change': row.get('why', ''),
                                'suggested_validation': row.get('validation', ''),
                            }
                        ]
                    ),
                    dataset_name=dataset_name,
                    dataset_family_key=str(evolution_summary.get('dataset_family_key', 'generic-healthcare')),
                )
            st.session_state['evolution_execution_queue'] = validation_queue
            updated_memory = dict(st.session_state.get('evolution_memory', {}))
            updated_memory['execution_queue'] = validation_queue
            st.session_state['evolution_memory'] = updated_memory
            save_evolution_memory(
                updated_memory,
                storage_service=st.session_state.get('storage_service'),
                workspace_identity=workspace_identity,
            )
            if application_service is not None:
                application_service.persist_user_settings(st.session_state)
            st.success('Queued the recommended validation follow-up plan.')
            st.rerun()
        with st.expander('Recurring patterns and safe adaptive actions', expanded=False):
            info_or_table(
                safe_df(evolution_summary.get('recurring_patterns_table')),
                'Recurring weak spots will appear here after Clinverity sees repeat patterns for this dataset family.',
            )
            info_or_table(
                safe_df(evolution_summary.get('auto_actions_table')),
                'Safe adaptive actions will appear here when Clinverity can recommend benchmark, mapping, or validation upgrades without changing production logic automatically.',
            )
            info_or_table(
                safe_df(evolution_summary.get('proposal_queue_table')),
                'Draft upgrade proposals will appear here when Clinverity can convert recurring issues into structured improvement specs.',
            )
            info_or_table(
                safe_df(evolution_summary.get('validation_recommendations_table')),
                'Adaptive validation recommendations will appear here when Clinverity can suggest the safest next validation gate for this dataset family.',
            )
            info_or_table(
                safe_df(evolution_summary.get('drift_alerts_table')),
                'Drift alerts will appear here when the current run diverges meaningfully from prior accepted patterns for this dataset family.',
            )
            info_or_table(
                safe_df(st.session_state.get('evolution_execution_queue', [])),
                'Tracked adaptive execution items will appear here after draft proposals are explicitly queued for follow-up.',
            )
            info_or_table(
                safe_df(evolution_summary.get('backlog_summary_table')),
                'Adaptive backlog summary will appear here once tracked execution work exists for the current dataset family.',
            )
            info_or_table(
                safe_df(evolution_summary.get('family_intelligence_table')),
                'Dataset-family intelligence will appear here when Clinverity has enough semantic evidence and prior observations to describe the family clearly.',
            )
            info_or_table(
                safe_df(evolution_summary.get('semantic_learning_table')),
                'Semantic learning hints will appear here when approved family profiles can guide future mapping suggestions for similar uploads.',
            )
            info_or_table(
                safe_df(evolution_summary.get('outcome_feedback_table')),
                'Outcome feedback will appear here after reviewers mark generated outputs as useful or not useful for this dataset family.',
            )
            info_or_table(
                safe_df(evolution_summary.get('drift_history_table')),
                'Family run history will appear here when Clinverity has seen repeated runs for this dataset family.',
            )
        review_store = st.session_state.setdefault('dataset_review_approvals', {})
        dataset_identifier = str(
            pipeline.get('dataset_runtime_diagnostics', {}).get('dataset_identifier', '')
            or pipeline.get('dataset_runtime_diagnostics', {}).get('dataset_cache_key', '')
            or dataset_name
        )
        review_entry = dict(review_store.get(dataset_identifier, {}))
        history_table = safe_df(pd.DataFrame(review_entry.get('review_history', [])))
        release_readiness = build_release_readiness_summary(
            execution_queue=st.session_state.get('evolution_execution_queue', []),
            review_history=review_entry.get('review_history', []),
            approval_workflow=review_entry,
        )
        info_or_table(
            release_readiness,
            'Release-readiness checkpoints will appear here once mapping, trust, export, and adaptive backlog state are available.',
        )
        action_cols = st.columns(2)
        with action_cols[0]:
            st.write('**Outcome quality feedback**')
            feedback_note = st.text_input(
                'Feedback note',
                key=f'evolution_feedback_note::{dataset_identifier}',
                placeholder='Describe whether the generated insights or exports were useful.',
            )
            feedback_cols = st.columns(2)
            if feedback_cols[0].button('Mark output helpful', key=f'evolution_feedback_helpful::{dataset_identifier}'):
                updated_memory = record_outcome_feedback(
                    st.session_state.get('evolution_memory', {}),
                    dataset_family_key=str(evolution_summary.get('dataset_family_key', 'generic-healthcare')),
                    dataset_name=dataset_name,
                    feedback='Helpful',
                    surface='Insights and export workflow',
                    notes=feedback_note,
                    reviewer=str(st.session_state.get('workspace_user_name', 'Analyst')),
                )
                st.session_state['evolution_memory'] = updated_memory
                save_evolution_memory(
                    updated_memory,
                    storage_service=st.session_state.get('storage_service'),
                    workspace_identity=workspace_identity,
                )
                if application_service is not None:
                    application_service.persist_user_settings(st.session_state)
                st.success('Recorded helpful outcome feedback for this dataset family.')
                st.rerun()
            if feedback_cols[1].button('Mark output not helpful', key=f'evolution_feedback_not_helpful::{dataset_identifier}'):
                updated_memory = record_outcome_feedback(
                    st.session_state.get('evolution_memory', {}),
                    dataset_family_key=str(evolution_summary.get('dataset_family_key', 'generic-healthcare')),
                    dataset_name=dataset_name,
                    feedback='Not_helpful',
                    surface='Insights and export workflow',
                    notes=feedback_note,
                    reviewer=str(st.session_state.get('workspace_user_name', 'Analyst')),
                )
                st.session_state['evolution_memory'] = updated_memory
                save_evolution_memory(
                    updated_memory,
                    storage_service=st.session_state.get('storage_service'),
                    workspace_identity=workspace_identity,
                )
                if application_service is not None:
                    application_service.persist_user_settings(st.session_state)
                st.warning('Recorded a not-helpful outcome signal so Clinverity can improve future ranking for this family.')
                st.rerun()
        with action_cols[1]:
            st.write('**Governance review history**')
            review_note = st.text_input(
                'Governance note',
                key=f'governance_review_note::{dataset_identifier}',
                placeholder='Capture reviewer context, blockers, or release guidance.',
            )
            review_role_default = str(
                st.session_state.get('approval_routing_rules', {}).get(
                    'governance_note',
                    st.session_state.get('governance_default_reviewer_role', st.session_state.get('active_role', 'Analyst')),
                )
            )
            review_role = st.selectbox(
                'Reviewer role',
                ROLE_OPTIONS,
                index=ROLE_OPTIONS.index(review_role_default)
                if review_role_default in ROLE_OPTIONS
                else 1,
                key=f'governance_reviewer_role::{dataset_identifier}',
            )
            history_cols = st.columns(2)
            if history_cols[0].button('Append governance note', key=f'append_governance_note::{dataset_identifier}'):
                review_store = append_review_history(
                    review_store,
                    dataset_identifier=dataset_identifier,
                    reviewer=str(st.session_state.get('workspace_user_name', 'Analyst')),
                    reviewer_role=review_role,
                    action='Governance note',
                    status='In review',
                    notes=review_note,
                )
                st.session_state['dataset_review_approvals'] = review_store
                if application_service is not None:
                    application_service.persist_user_settings(st.session_state)
                st.success('Governance review history updated.')
                st.rerun()
            if history_cols[1].button('Mark release blocker', key=f'mark_release_blocker::{dataset_identifier}'):
                review_store = append_review_history(
                    review_store,
                    dataset_identifier=dataset_identifier,
                    reviewer=str(st.session_state.get('workspace_user_name', 'Analyst')),
                    reviewer_role=review_role,
                    action='Release blocker',
                    status='Blocked',
                    notes=review_note or 'Explicit release blocker recorded for this dataset review.',
                )
                review_entry = dict(review_store.get(dataset_identifier, {}))
                review_entry['release_signoff_status'] = 'Blocked'
                review_store[dataset_identifier] = review_entry
                st.session_state['dataset_review_approvals'] = review_store
                if application_service is not None:
                    application_service.persist_user_settings(st.session_state)
                st.warning('Release blocker recorded for the current dataset review.')
                st.rerun()
        with st.expander('Adaptive admin diagnostics', expanded=False):
            storage_health = build_storage_backend_health(st.session_state.get('storage_service'))
            support_diagnostics = build_support_diagnostics()
            storage_notes = pd.DataFrame(
                [{'note': note} for note in list(storage_health.get('notes', []))]
            )
            diagnostics_table = pd.DataFrame(
                [
                    {'signal': 'Build label', 'value': current_build_label()},
                    {'signal': 'Persistence enabled', 'value': 'Yes' if application_service is not None and application_service.enabled else 'No'},
                    {'signal': 'Storage mode', 'value': str(storage_health.get('mode', 'session'))},
                    {'signal': 'Storage target', 'value': str(storage_health.get('storage_target', 'session-only'))},
                    {'signal': 'Storage health', 'value': str(storage_health.get('status', 'Unknown'))},
                    {'signal': 'Evolution memory families', 'value': fmt(len(dict(st.session_state.get('evolution_memory', {})).get('family_memory', {})))},
                    {'signal': 'Tracked execution items', 'value': fmt(len(st.session_state.get('evolution_execution_queue', [])))},
                    {'signal': 'Recent support diagnostics', 'value': fmt(len(safe_df(support_diagnostics.get('diagnostics_table'))))},
                ]
            )
            info_or_table(diagnostics_table, 'Adaptive admin diagnostics will appear here once the runtime is initialized.')
            render_surface_panel(
                'Storage and runtime health',
                str(storage_health.get('detail', 'Runtime storage diagnostics will appear here once the storage service is initialized.')),
                tone='info' if str(storage_health.get('status', '')).lower() == 'healthy' else 'warning',
            )
            info_or_table(
                storage_notes,
                'Storage notes will appear here when the runtime provides backend guidance or configuration hints.',
            )
            info_or_table(
                safe_df(support_diagnostics.get('diagnostics_table')),
                'Recent support diagnostics will appear here when recent platform events have been captured.',
            )
        with st.expander('Governance review history', expanded=False):
            info_or_table(
                history_table,
                'Governance review history will appear here after reviewers save notes, approvals, or release blockers for the dataset.',
            )
    if suggested_profile:
        st.caption(
            f"Suggested mapping profile: {suggested_profile.get('profile_name', 'Mapping profile')} "
            f"({float(suggested_profile.get('suggestion_score', 0.0)):.2f} match score)."
        )
    suggestion_table = build_profile_suggestion_table(
        pipeline['data'].columns,
        dataset_name=dataset_name,
        user_profiles=profile_store,
    )
    if not suggestion_table.empty:
        with st.expander('Dataset-family profile suggestions', expanded=False):
            info_or_table(
                suggestion_table,
                'Profile suggestions will appear here when a saved or built-in dataset-family template matches the current upload.',
            )
    if st.session_state.get(remap_notice_key):
        st.success(str(st.session_state.get(remap_notice_key)))
    editable_remap_board = st.data_editor(
        remap_board,
        width='stretch',
        hide_index=True,
        key=f'{remap_key}::editor',
        column_config={
            'display_order': st.column_config.NumberColumn('Order', min_value=1, step=1),
            'mapped_field': st.column_config.SelectboxColumn('Mapped field', options=['Not mapped', *sorted(CANONICAL_FIELDS.keys())]),
            'confidence_score': st.column_config.NumberColumn('Confidence', min_value=0.0, max_value=1.0, step=0.01, format='%.2f'),
            'source_column': st.column_config.TextColumn('Source column', disabled=True),
            'confidence_label': st.column_config.TextColumn('Confidence label', disabled=True),
            'inferred_type': st.column_config.TextColumn('Inferred type', disabled=True),
            'suggested_field': st.column_config.TextColumn('Suggested field', disabled=True),
            'suggested_confidence': st.column_config.NumberColumn('Suggested confidence', min_value=0.0, max_value=1.0, step=0.01, format='%.2f', disabled=True),
        },
        disabled=['source_column', 'confidence_label', 'inferred_type', 'suggested_field', 'suggested_confidence'],
    )
    remap_actions = st.columns(5)
    if remap_actions[0].button('Save Remap Overrides', key=f'{remap_key}::save'):
        cleaned_board = editable_remap_board.sort_values(['display_order', 'source_column']).reset_index(drop=True)
        st.session_state[remap_key] = cleaned_board
        overrides = {
            str(row.get('mapped_field', '')).strip(): str(row.get('source_column', '')).strip()
            for row in cleaned_board.to_dict(orient='records')
            if str(row.get('mapped_field', '')).strip() and str(row.get('mapped_field', '')).strip() != 'Not mapped'
        }
        st.session_state.setdefault('semantic_mapping_overrides_by_dataset', {})[dataset_cache_key] = overrides
        key_pairs = [
            f"{field} <- {column}"
            for field, column in [
                ('patient_id', overrides.get('patient_id', '')),
                ('admission_date', overrides.get('admission_date', '')),
                ('discharge_date', overrides.get('discharge_date', '')),
            ]
            if column
        ]
        st.session_state[remap_notice_key] = (
            f"Applied mapping profile: {' | '.join(key_pairs)}"
            if key_pairs
            else f"Applied mapping profile with {len(overrides)} confirmed overrides."
        )
        active_diagnostics = dict(st.session_state.get('active_dataset_diagnostics', {}))
        active_diagnostics['manual_semantic_override_count'] = len(overrides)
        active_diagnostics['manual_semantic_override_summary'] = ' | '.join(key_pairs) if key_pairs else ''
        st.session_state['active_dataset_diagnostics'] = active_diagnostics
        if application_service is not None:
            application_service.persist_user_settings(st.session_state)
        log_event('Field Remapping Saved', f"Saved {len(cleaned_board)} remapping rows for dataset '{dataset_name}'.", 'Field remapping', 'Data intake', resource_type='semantic_mapping', resource_name=dataset_name)
        st.rerun()
    if remap_actions[1].button('Apply Auto-Mapping Suggestions', key=f'{remap_key}::auto_apply'):
        updated_board = apply_auto_mapping_suggestions(editable_remap_board)
        st.session_state[remap_key] = updated_board
        log_event('Auto Mapping Applied', f"Applied semantic auto-mapping suggestions for dataset '{dataset_name}'.", 'Field remapping', 'Data intake', resource_type='semantic_mapping', resource_name=dataset_name)
        st.rerun()
    if remap_actions[2].button('Save Mapping Profile', key=f'{remap_key}::save_template'):
        dataset_type = str(intelligence.get('dataset_type_label', 'General tabular dataset'))
        template_name = f"{dataset_family.get('family_label', dataset_type)} | {dataset_name}"
        template_store[template_name] = build_mapping_template(editable_remap_board, template_name=template_name, dataset_type=dataset_type)
        profile_store[template_name] = build_mapping_profile(
            editable_remap_board,
            profile_name=template_name,
            dataset_type=dataset_type,
            family_key=str(dataset_family.get('family_key', 'generic-healthcare')),
            family_label=str(dataset_family.get('family_label', 'Generic Healthcare Feed')),
            benchmark_profile=str(dataset_family.get('benchmark_profile', 'Generic Healthcare')),
        )
        if application_service is not None:
            application_service.persist_user_settings(st.session_state)
        log_event('Mapping Profile Saved', f"Saved mapping profile '{template_name}'.", 'Field remapping', 'Data intake', resource_type='mapping_template', resource_name=template_name)
        st.success(f"Saved reusable mapping profile '{template_name}'.")
    if remap_actions[3].button('Apply Suggested Profile', key=f'{remap_key}::apply_suggested', disabled=suggested_profile is None):
        profile_template = {
            'mappings': [
                {
                    'source_column': source_column,
                    'mapped_field': canonical_field,
                    'display_order': index,
                }
                for index, (canonical_field, source_column) in enumerate(
                    dict(suggested_profile.get('resolved_mappings', {})).items(),
                    start=1,
                )
            ]
        } if suggested_profile else {'mappings': []}
        updated_board = apply_mapping_template(editable_remap_board, profile_template)
        st.session_state[remap_key] = updated_board
        st.session_state.setdefault('semantic_mapping_overrides_by_dataset', {})[dataset_cache_key] = {
            str(row.get('mapped_field', '')).strip(): str(row.get('source_column', '')).strip()
            for row in updated_board.to_dict(orient='records')
            if str(row.get('mapped_field', '')).strip() and str(row.get('mapped_field', '')).strip() != 'Not mapped'
        }
        applied_overrides = st.session_state.get('semantic_mapping_overrides_by_dataset', {}).get(dataset_cache_key, {})
        key_pairs = [
            f"{field} <- {column}"
            for field, column in [
                ('patient_id', applied_overrides.get('patient_id', '')),
                ('admission_date', applied_overrides.get('admission_date', '')),
                ('discharge_date', applied_overrides.get('discharge_date', '')),
            ]
            if column
        ]
        st.session_state[remap_notice_key] = (
            f"Applied mapping profile: {' | '.join(key_pairs)}"
            if key_pairs
            else f"Applied mapping profile with {len(applied_overrides)} confirmed overrides."
        )
        active_diagnostics = dict(st.session_state.get('active_dataset_diagnostics', {}))
        active_diagnostics['manual_semantic_override_count'] = len(applied_overrides)
        active_diagnostics['manual_semantic_override_summary'] = ' | '.join(key_pairs) if key_pairs else ''
        st.session_state['active_dataset_diagnostics'] = active_diagnostics
        st.session_state['accuracy_benchmark_profile'] = str(suggested_profile.get('benchmark_profile', 'Auto')) if suggested_profile else st.session_state.get('accuracy_benchmark_profile', 'Auto')
        if application_service is not None:
            application_service.persist_user_settings(st.session_state)
        st.rerun()
    evolution_auto_actions = safe_df(evolution_summary.get('auto_actions_table')) if evolution_summary.get('available') else pd.DataFrame()
    dataset_identifier = str(
        pipeline.get('dataset_runtime_diagnostics', {}).get('dataset_identifier', '')
        or source_meta.get('dataset_identifier', '')
        or dataset_cache_key
    )
    if not evolution_auto_actions.empty:
        render_surface_panel(
            'One-click adaptive actions',
            'Approve and apply the highest-value safe improvements for this dataset family without leaving the intake workflow.',
            tone='accent',
        )
        adaptive_cols = st.columns(7)
        can_save_family_profile = bool(
            st.session_state.get('semantic_mapping_overrides_by_dataset', {}).get(dataset_cache_key, {})
        )
        if adaptive_cols[0].button(
            'Approve & Save Family Profile',
            key=f'{remap_key}::adaptive_save_profile',
            disabled=not can_save_family_profile,
        ):
            cleaned_board = editable_remap_board.sort_values(['display_order', 'source_column']).reset_index(drop=True)
            st.session_state[remap_key] = cleaned_board
            dataset_type = str(intelligence.get('dataset_type_label', 'General tabular dataset'))
            profile_name = f"{dataset_family.get('family_label', dataset_type)} | Approved Family Profile"
            template_store[profile_name] = build_mapping_template(
                cleaned_board,
                template_name=profile_name,
                dataset_type=dataset_type,
            )
            profile_store[profile_name] = build_mapping_profile(
                cleaned_board,
                profile_name=profile_name,
                dataset_type=dataset_type,
                family_key=str(dataset_family.get('family_key', 'generic-healthcare')),
                family_label=str(dataset_family.get('family_label', 'Generic Healthcare Feed')),
                benchmark_profile=str(dataset_family.get('benchmark_profile', 'Generic Healthcare')),
            )
            review_store = st.session_state.setdefault('dataset_review_approvals', {})
            review_state = dict(review_store.get(dataset_identifier, {}))
            review_state['mapping_status'] = 'Approved'
            review_state['reviewed_by_role'] = str(
                st.session_state.get('approval_routing_rules', {}).get(
                    'mapping_approval',
                    st.session_state.get('governance_default_reviewer_role', st.session_state.get('active_role', 'Analyst')),
                )
            )
            review_state['review_notes'] = (
                f"{str(review_state.get('review_notes', '')).strip()} "
                f"Approved reusable mapping profile: {profile_name}."
            ).strip()
            review_store[dataset_identifier] = review_state
            if application_service is not None:
                application_service.persist_user_settings(st.session_state)
            st.success(f"Approved and saved reusable family profile '{profile_name}'.")
            st.rerun()
        suggested_benchmark = str(evolution_summary.get('suggested_benchmark_profile', 'Generic Healthcare'))
        current_benchmark = str(st.session_state.get('accuracy_benchmark_profile', 'Auto'))
        if adaptive_cols[1].button(
            'Approve Suggested Benchmark',
            key=f'{remap_key}::adaptive_benchmark',
            disabled=not suggested_benchmark or suggested_benchmark == current_benchmark,
        ):
            st.session_state['accuracy_benchmark_profile'] = suggested_benchmark
            review_store = st.session_state.setdefault('dataset_review_approvals', {})
            review_state = dict(review_store.get(dataset_identifier, {}))
            review_state['trust_gate_status'] = 'Approved'
            review_state['reviewed_by_role'] = str(
                st.session_state.get('approval_routing_rules', {}).get(
                    'trust_gate',
                    st.session_state.get('governance_default_reviewer_role', st.session_state.get('active_role', 'Analyst')),
                )
            )
            review_state['review_notes'] = (
                f"{str(review_state.get('review_notes', '')).strip()} "
                f"Approved benchmark profile: {suggested_benchmark}."
            ).strip()
            review_store[dataset_identifier] = review_state
            if application_service is not None:
                application_service.persist_user_settings(st.session_state)
            st.success(f"Approved benchmark profile '{suggested_benchmark}'.")
            st.rerun()
        proposal_table = safe_df(evolution_summary.get('proposal_queue_table'))
        if adaptive_cols[2].button(
            'Queue Draft Proposals',
            key=f'{remap_key}::adaptive_queue_proposals',
            disabled=proposal_table.empty,
        ):
            updated_queue = queue_execution_items(
                st.session_state.get('evolution_execution_queue', []),
                proposal_table,
                dataset_name=dataset_name,
                dataset_family_key=str(evolution_summary.get('dataset_family_key', 'generic-healthcare')),
            )
            st.session_state['evolution_execution_queue'] = updated_queue
            updated_memory = dict(st.session_state.get('evolution_memory', {}))
            updated_memory['execution_queue'] = updated_queue
            st.session_state['evolution_memory'] = updated_memory
            save_evolution_memory(
                updated_memory,
                storage_service=st.session_state.get('storage_service'),
                workspace_identity=workspace_identity,
            )
            if application_service is not None:
                application_service.persist_user_settings(st.session_state)
            st.success('Draft proposals have been promoted into the tracked adaptive execution queue.')
            st.rerun()
        execution_items = safe_df(st.session_state.get('evolution_execution_queue', []))
        autopilot_actions = build_execution_autopilot_actions(
            st.session_state.get('evolution_execution_queue', []),
            proposal_table,
            validation_recommendations,
        )
        autopilot_options = autopilot_actions['action'].astype(str).tolist() if not autopilot_actions.empty else ['No safe actions available']
        selected_autopilot_action = adaptive_cols[3].selectbox(
            'Autopilot action',
            autopilot_options,
            key=f'{remap_key}::autopilot_action',
            disabled=autopilot_actions.empty,
        )
        if adaptive_cols[4].button(
            'Run Safe Autopilot',
            key=f'{remap_key}::run_safe_autopilot',
            disabled=autopilot_actions.empty,
        ):
            updated_queue = apply_execution_autopilot(
                st.session_state.get('evolution_execution_queue', []),
                action_name=selected_autopilot_action,
                proposals=proposal_table,
                validation_recommendations=validation_recommendations,
                dataset_name=dataset_name,
                dataset_family_key=str(evolution_summary.get('dataset_family_key', 'generic-healthcare')),
                default_owner=str(st.session_state.get('governance_default_owner', st.session_state.get('active_role', 'Analyst'))),
            )
            st.session_state['evolution_execution_queue'] = updated_queue
            updated_memory = dict(st.session_state.get('evolution_memory', {}))
            updated_memory['execution_queue'] = updated_queue
            st.session_state['evolution_memory'] = updated_memory
            save_evolution_memory(
                updated_memory,
                storage_service=st.session_state.get('storage_service'),
                workspace_identity=workspace_identity,
            )
            if application_service is not None:
                application_service.persist_user_settings(st.session_state)
            st.success(f"Applied safe autopilot action: {selected_autopilot_action}.")
            st.rerun()
        adaptive_cols[3].caption(
            f"{len(autopilot_actions)} safe execution action(s) available"
        )
        execution_select_options = execution_items['proposal_title'].astype(str).tolist() if not execution_items.empty else ['No tracked items']
        selected_execution_item = adaptive_cols[5].selectbox(
            'Tracked item',
            execution_select_options,
            key=f'{remap_key}::execution_item_title',
            disabled=execution_items.empty,
        )
        if adaptive_cols[6].button(
            'Mark Tracked Item Complete',
            key=f'{remap_key}::complete_execution_item',
            disabled=execution_items.empty,
        ):
            updated_queue = update_execution_item_status(
                st.session_state.get('evolution_execution_queue', []),
                proposal_title=selected_execution_item,
                status='Completed',
            )
            st.session_state['evolution_execution_queue'] = updated_queue
            updated_memory = dict(st.session_state.get('evolution_memory', {}))
            updated_memory['execution_queue'] = updated_queue
            st.session_state['evolution_memory'] = updated_memory
            save_evolution_memory(
                updated_memory,
                storage_service=st.session_state.get('storage_service'),
                workspace_identity=workspace_identity,
            )
            if application_service is not None:
                application_service.persist_user_settings(st.session_state)
            st.success(f"Marked '{selected_execution_item}' as completed.")
            st.rerun()
        adaptive_cols[2].caption(
            f"Current benchmark: {current_benchmark} | Suggested: {suggested_benchmark or 'None'}"
        )
        if not autopilot_actions.empty:
            info_or_table(
                autopilot_actions,
                'Safe execution autopilot actions will appear here once the adaptive backlog has enough context.',
            )
        validation_execution_plan = safe_df(build_validation_execution_plan(validation_recommendations))
        runnable_validation_rows = (
            validation_execution_plan[validation_execution_plan['allowed'].astype(bool)]
            if not validation_execution_plan.empty and 'allowed' in validation_execution_plan.columns
            else validation_execution_plan
        )
        if not runnable_validation_rows.empty:
            top_validation = str(runnable_validation_rows.iloc[0].get('validation', 'Quick validation'))
            if st.button(
                f'Run Top Recommended Validation: {top_validation}',
                key=f'{remap_key}::run_top_validation',
            ):
                result = run_recommended_validation(top_validation)
                runs = list(st.session_state.get('validation_orchestration_runs', []))
                runs.insert(0, result)
                st.session_state['validation_orchestration_runs'] = runs[:25]
                if result.get('status') == 'Passed':
                    st.success(f"{top_validation} completed successfully.")
                else:
                    st.error(f"{top_validation} failed. Review Admin Diagnostics for the execution log.")
                st.rerun()
        elif not validation_execution_plan.empty:
            st.info('Recommended validation exists, but the current runtime is intentionally gating heavier checks. Use Admin Diagnostics from a local or staging environment to run them.')
        if not execution_items.empty:
            execution_editor = st.data_editor(
                execution_items,
                width='stretch',
                hide_index=True,
                key=f'{remap_key}::execution_queue_editor',
                column_config={
                    'proposal_title': st.column_config.TextColumn('Proposal', disabled=True),
                    'proposal_type': st.column_config.TextColumn('Type', disabled=True),
                    'priority': st.column_config.TextColumn('Priority', disabled=True),
                    'dataset_name': st.column_config.TextColumn('Dataset', disabled=True),
                    'owner': st.column_config.SelectboxColumn('Owner', options=['Unassigned', 'Admin', 'Analyst', 'Executive', 'Clinician', 'Researcher', 'Data Steward']),
                    'due_state': st.column_config.SelectboxColumn('Due state', options=['Planned', 'Next review', 'Before release', 'In progress', 'Deferred']),
                    'release_gate_status': st.column_config.SelectboxColumn('Release gate', options=['Advisory', 'Required before release', 'Blocked for release', 'Satisfied']),
                    'status': st.column_config.SelectboxColumn('Status', options=['Queued', 'In progress', 'Completed', 'Deferred']),
                    'proposed_change': st.column_config.TextColumn('Proposed change', disabled=True, width='large'),
                    'suggested_validation': st.column_config.TextColumn('Suggested validation', disabled=True, width='large'),
                },
                disabled=['proposal_title', 'proposal_type', 'priority', 'dataset_name', 'proposed_change', 'suggested_validation'],
            )
            if st.button('Save Execution Workflow', key=f'{remap_key}::save_execution_workflow'):
                updated_queue = execution_editor.to_dict(orient='records')
                st.session_state['evolution_execution_queue'] = updated_queue
                updated_memory = dict(st.session_state.get('evolution_memory', {}))
                updated_memory['execution_queue'] = updated_queue
                st.session_state['evolution_memory'] = updated_memory
                save_evolution_memory(
                    updated_memory,
                    storage_service=st.session_state.get('storage_service'),
                    workspace_identity=workspace_identity,
                )
                if application_service is not None:
                    application_service.persist_user_settings(st.session_state)
                st.success('Adaptive execution workflow saved.')
                st.rerun()
    available_templates = list(template_store.keys())
    selected_template = remap_actions[4].selectbox(
        'Apply template',
        ['None', *available_templates],
        key=f'{remap_key}::template_select',
    )
    if selected_template != 'None' and st.button('Apply Selected Template', key=f'{remap_key}::apply_template'):
        st.session_state[remap_key] = apply_mapping_template(editable_remap_board, template_store[selected_template])
        log_event('Mapping Template Applied', f"Applied mapping template '{selected_template}' to dataset '{dataset_name}'.", 'Field remapping', 'Data intake', resource_type='mapping_template', resource_name=selected_template)
        st.rerun()
    if st.button('Reset Remap Board', key=f'{remap_key}::reset'):
        st.session_state[remap_key] = build_remap_board(pipeline)
        st.session_state.setdefault('semantic_mapping_overrides_by_dataset', {}).pop(dataset_cache_key, None)
        st.session_state.pop(remap_notice_key, None)
        active_diagnostics = dict(st.session_state.get('active_dataset_diagnostics', {}))
        active_diagnostics['manual_semantic_override_count'] = 0
        active_diagnostics['manual_semantic_override_summary'] = ''
        st.session_state['active_dataset_diagnostics'] = active_diagnostics
        st.rerun()
    overrides = safe_df(
        pd.DataFrame(
            [
                {'canonical_field': field, 'source_column': column}
                for field, column in st.session_state.get('semantic_mapping_overrides_by_dataset', {}).get(dataset_cache_key, {}).items()
            ]
        )
    )
    if not overrides.empty:
        key_override_lookup = {
            str(row.get('canonical_field', '')).strip(): str(row.get('source_column', '')).strip()
            for row in overrides.to_dict(orient='records')
        }
        key_mapping_summaries = [
            f"{canonical_field} <- {key_override_lookup[canonical_field]}"
            for canonical_field in ['patient_id', 'admission_date', 'discharge_date']
            if key_override_lookup.get(canonical_field)
        ]
        info_or_table(overrides, 'Saved remapping overrides will appear here after the first change is recorded.')
        render_surface_panel(
            'Confirmed mapping overrides',
            f"{len(overrides)} canonical fields are now explicitly locked for this dataset family and session.",
            tone='success',
        )
        if key_mapping_summaries:
            st.caption(f"Key encounter mappings: {' | '.join(key_mapping_summaries)}")
        for override in overrides.sort_values(['canonical_field', 'source_column']).to_dict(orient='records'):
            st.markdown(
                f"- `{str(override.get('canonical_field', '')).strip()}` <- "
                f"`{str(override.get('source_column', '')).strip()}`"
            )
    version_records = application_service.list_dataset_versions(workspace_identity, dataset_name=dataset_name) if application_service is not None else []
    version_table = safe_df(pd.DataFrame(version_records))
    if not version_table.empty:
        render_subsection_header('Dataset version history', 'Compare the active dataset with prior stored versions from this workspace.')
        display_columns = [
            column
            for column in ['dataset_name', 'version_label', 'source_mode', 'row_count', 'column_count', 'file_size_mb', 'created_at', 'is_active']
            if column in version_table.columns
        ]
        metric_row([
            ('Stored versions', fmt(len(version_table))),
            ('Latest version hash', str(version_table.iloc[0].get('version_hash', ''))[:10]),
            ('Current source', str(source_meta.get('source_mode', 'Unknown'))),
            ('Workspace persistence', 'Enabled'),
        ])
        info_or_table(
            version_table[display_columns] if display_columns else version_table,
            'Dataset version history will appear here when persistence is enabled and prior versions have been stored.',
        )
        if len(version_table) >= 2:
            current_row = version_table.iloc[0]
            previous_row = version_table.iloc[1]
            version_diff = build_dataset_version_diff(
                current_row.to_dict(),
                previous_row.to_dict(),
            )
            with st.expander('Compare latest vs previous version', expanded=False):
                info_or_table(
                    safe_df(version_diff.get('summary_table')),
                    'Version comparison details appear here when at least two stored dataset versions are available.',
                )
                info_or_table(
                    safe_df(version_diff.get('added_columns_table')),
                    'Added columns will appear here when the current dataset version introduced new source fields.',
                )
                info_or_table(
                    safe_df(version_diff.get('removed_columns_table')),
                    'Removed columns will appear here when the latest version dropped fields from the prior version.',
                )
                info_or_table(
                    safe_df(version_diff.get('dtype_changes_table')),
                    'Column type changes will appear here when the same source field changed dtype between versions.',
                )

    st.markdown('### Data Profiling Summary')
    profiling_summary = build_data_profiling_summary(pipeline)
    metric_row([(card['label'], card['value']) for card in profiling_summary.get('summary_cards', [])])
    if pipeline.get('sample_info', {}).get('sampling_applied'):
        st.info('Large-dataset sampling is active for profiling and readiness review. The tables below show capped summaries so the UI stays responsive on heavy files.')
    info_or_table(safe_df(profiling_summary.get('high_risk_table')), 'High-risk profiling findings appear here when null spikes, outliers, or identifier-like columns need attention.')
    with st.expander('Numeric profiling summary'):
        info_or_table(safe_df(profiling_summary.get('numeric_summary')), 'Numeric profiling details appear here when numeric fields are available in the current dataset.')
    with st.expander('Field profile preview'):
        info_or_table(safe_df(profiling_summary.get('field_profile_preview')), 'Field profile details appear here once profiling is available for the current dataset.')

    intake_view_role = str(st.session_state.get('workspace_role') or st.session_state.get('active_role') or 'Viewer')
    intake_order = (
        ['workspace_admin', 'operations_review', 'workflow_guidance', 'collaboration_tools']
        if intake_view_role in {'Admin', 'Data Steward', 'Owner'}
        else ['workflow_guidance', 'collaboration_tools', 'operations_review', 'workspace_admin']
    )
    intake_sections = {key: st.container() for key in intake_order}
    workspace_admin_section = intake_sections['workspace_admin']
    workflow_guidance_section = intake_sections['workflow_guidance']
    operations_review_section = intake_sections['operations_review']
    collaboration_tools_section = intake_sections['collaboration_tools']

    render_role_context_panel(
        intake_view_role,
        primary_message=(
            'Workflow-first roles see guidance, collaboration, and review surfaces first. '
            'Governance roles see workspace controls and operational oversight earlier.'
        ),
        advanced_enabled=advanced_sections_enabled,
        advanced_label='Dense admin and audit sections',
    )

    with workspace_admin_section:
        st.markdown('### Workspace Access')
        auth_service = st.session_state.get('auth_service')
        application_service = st.session_state.get('application_service')
        auth_session = st.session_state.get('auth_session', auth_module.build_guest_auth_session())
        workspace_identity = st.session_state.get('workspace_identity') or auth_service.build_workspace_identity(auth_session, st.session_state.get('workspace_name'))
        workspace_cols = st.columns([1.0, 1.0, 1.2, 0.9, 0.7, 0.7])
        workspace_cols[0].text_input('Display name', key='workspace_user_name')
        workspace_cols[1].text_input('Workspace name', key='workspace_name')
        workspace_cols[2].text_input('Email', key='workspace_user_email', placeholder='name@organization.com')
        workspace_cols[3].selectbox('Workspace role', auth_module.WORKSPACE_ROLE_OPTIONS, key='workspace_role')
        hosted_config = getattr(auth_service, 'hosted_config', None)
        if hosted_config is not None and hosted_config.provider != 'guest':
            st.caption(
                f"Hosted provider prepared: {hosted_config.provider_label} | "
                f"Redirect URI: {hosted_config.redirect_uri or 'Not configured'}"
            )
        identity_cols = st.columns([1.2, 1.2, 1.0])
        identity_cols[0].text_input('OIDC subject / employee ID', key='workspace_provider_subject', placeholder='oidc subject, employee id, or external user id')
        identity_cols[1].text_input('2FA code', key='workspace_two_factor_code', placeholder='123456')
        auth_start = auth_service.build_oidc_start() if hasattr(auth_service, 'build_oidc_start') else None
        if auth_start:
            identity_cols[2].link_button('Open SSO sign-in', auth_start['authorization_url'], use_container_width=True)
        else:
            identity_cols[2].caption('SSO sign-in becomes available when hosted auth is fully configured.')
        if workspace_cols[4].button('Sign In', key='workspace_sign_in'):
            auth_session, workspace_identity = auth_service.sign_in_local(
                st.session_state.get('workspace_user_name'),
                st.session_state.get('workspace_user_email'),
                st.session_state.get('workspace_name'),
                st.session_state.get('workspace_role'),
                two_factor_code=st.session_state.get('workspace_two_factor_code'),
            )
            st.session_state['auth_session'] = auth_session
            st.session_state['workspace_identity'] = workspace_identity
            if application_service is not None:
                application_service.hydrate_workspace_state(st.session_state)
            log_event(
                'Workspace Sign In',
                f"Activated workspace '{st.session_state['workspace_identity']['workspace_name']}' for {st.session_state['workspace_identity']['display_name']} via {st.session_state['workspace_identity'].get('provider', 'local-demo')}.",
                'Workspace access',
                'Data intake',
                resource_type='workspace',
                resource_name=str(st.session_state['workspace_identity']['workspace_name']),
            )
            st.rerun()
        if hosted_config is not None and hosted_config.provider != 'guest' and st.button('Simulate Hosted Sign-In', key='workspace_sign_in_hosted'):
            auth_session, workspace_identity = auth_service.sign_in_hosted(
                display_name=st.session_state.get('workspace_user_name'),
                email=st.session_state.get('workspace_user_email'),
                provider_subject=st.session_state.get('workspace_provider_subject'),
                workspace_name=st.session_state.get('workspace_name'),
                workspace_role=st.session_state.get('workspace_role'),
                provider=hosted_config.provider,
                tenant_id=hosted_config.tenant_id,
                two_factor_code=st.session_state.get('workspace_two_factor_code'),
            )
            st.session_state['auth_session'] = auth_session
            st.session_state['workspace_identity'] = workspace_identity
            if application_service is not None:
                application_service.hydrate_workspace_state(st.session_state)
            log_event(
                'Workspace Hosted Sign In',
                f"Activated hosted workspace '{st.session_state['workspace_identity']['workspace_name']}' for {st.session_state['workspace_identity']['display_name']} via {hosted_config.provider_label}.",
                'Workspace access',
                'Data intake',
                resource_type='workspace',
                resource_name=str(st.session_state['workspace_identity']['workspace_name']),
            )
            st.rerun()
        if workspace_cols[5].button('Sign Out', key='workspace_sign_out'):
            auth_session, workspace_identity = auth_service.sign_out()
            st.session_state['auth_session'] = auth_session
            st.session_state['workspace_identity'] = workspace_identity
            st.session_state['workspace_user_name'] = str(auth_session.get('display_name', 'Guest User'))
            st.session_state['workspace_user_email'] = ''
            st.session_state['workspace_name'] = str(workspace_identity.get('workspace_name', 'Guest Demo Workspace'))
            st.session_state['workspace_role'] = 'Viewer'
            if application_service is not None:
                application_service.hydrate_workspace_state(st.session_state)
            log_event('Workspace Sign Out', 'Returned to the guest demo workspace.', 'Workspace access', 'Data intake', resource_type='workspace', resource_name='Guest Demo Workspace')
            st.rerun()
        workspace_identity = st.session_state.get('workspace_identity', workspace_identity)
        if application_service is not None:
            application_service.hydrate_workspace_state(st.session_state)
            application_service.upsert_collaboration_presence(
                workspace_identity,
                session_id=str(workspace_identity.get('session_id', st.session_state.get('platform_session_id', 'guest-session'))),
                active_section='Data Intake',
            )
        admin_ops_service = st.session_state.get('admin_ops_service')
        admin_ops_view = None
        if admin_ops_service is not None:
            admin_ops_view = admin_ops_service.build_admin_ops_view(
                workspace_identity=workspace_identity,
                plan_awareness=plan_awareness,
                analysis_log=st.session_state.get('analysis_log', []),
                run_history=st.session_state.get('run_history', []),
                visited_sections=st.session_state.get('visited_sections', []),
                generated_report_outputs=st.session_state.get('generated_report_outputs', {}),
                saved_snapshots=st.session_state.get('saved_snapshots', {}),
                workflow_packs=st.session_state.get('workflow_packs', {}),
            )
        metric_row([
            ('Workspace Status', str(workspace_identity.get('status_label', 'Guest session'))),
            ('Active Workspace', str(workspace_identity.get('workspace_name', 'Guest Demo Workspace'))),
            ('Workspace User', str(workspace_identity.get('display_name', 'Guest User'))),
            ('Workspace Role', str(workspace_identity.get('role_label', 'Viewer'))),
            ('Saved Assets', fmt(len(st.session_state.get('saved_snapshots', {})) + len(st.session_state.get('workflow_packs', {})))),
        ])
        st.caption(
            'This is a demo-safe authentication and workspace ownership layer. '
            'Guest mode stays available, while local sign-in scopes snapshots, workflow packs, notes, and usage history to the active workspace owner.'
        )
        st.caption(
            f"Auth mode: {workspace_identity.get('auth_mode', 'guest')} | "
            f"Workspace role: {workspace_identity.get('role_label', 'Viewer')} | "
            f"Workspace owner: {workspace_identity.get('owner_label', 'Guest session')} | "
            f"Storage: {getattr(getattr(auth_service, 'status', None), 'storage_target', 'session-only')}"
        )
        if workspace_identity.get('mfa_required'):
            if workspace_identity.get('mfa_verified'):
                st.success(f"Two-factor authentication verified for this session via {workspace_identity.get('provider', 'current provider')}.")
            else:
                st.warning('Two-factor authentication is required for this session. Enter a valid TOTP code before relying on production-style access controls.')

    session_rows = auth_service.list_active_sessions(workspace_identity) if hasattr(auth_service, 'list_active_sessions') else []
    session_table = safe_df(pd.DataFrame(session_rows))
    st.markdown('### Session Management')
    with st.expander('Review session activity', expanded=advanced_sections_enabled):
        info_or_table(session_table, 'Active sessions will appear here after a signed-in workspace session is created.')
        if not session_table.empty:
            revoke_session_id = st.selectbox('Revoke session', [''] + session_table['session_id'].astype(str).tolist(), key='revoke_auth_session_id')
            if revoke_session_id and st.button('Revoke Selected Session', key='revoke_auth_session'):
                auth_service.revoke_session(revoke_session_id)
                log_event(
                    'Auth Session Revoked',
                    f"Revoked auth session '{revoke_session_id}' for workspace '{workspace_identity.get('workspace_name', 'Current workspace')}'.",
                    'Session management',
                    'Data intake',
                    resource_type='auth_session',
                    resource_name=revoke_session_id,
                )
                st.rerun()

    st.markdown('### Workspace Invitations')
    with st.expander('Manage invitations and collaborators', expanded=advanced_sections_enabled):
        invite_cols = st.columns([1.5, 1.0, 0.8])
        invite_cols[0].text_input('Invite user email', key='workspace_invite_email', placeholder='teammate@organization.com')
        invite_cols[1].selectbox('Invite role', auth_module.WORKSPACE_ROLE_OPTIONS[1:], key='workspace_invite_role')
        if invite_cols[2].button('Send Invite', key='workspace_send_invite'):
            try:
                auth_module.enforce_workspace_permission(workspace_identity, 'workspace_admin', resource_label='workspace invitations')
                invitation = auth_service.create_workspace_invitation(
                    workspace_identity,
                    email=st.session_state.get('workspace_invite_email', ''),
                    role=st.session_state.get('workspace_invite_role', 'Viewer'),
                )
                if invitation is None:
                    st.info('Workspace invitations become persistent when the auth database is enabled.')
                else:
                    log_event(
                        'Workspace Invitation Created',
                        f"Invited {invitation['email']} to workspace '{workspace_identity.get('workspace_name', 'Current workspace')}' as {invitation['role']}.",
                        'Workspace sharing',
                        'Data intake',
                        resource_type='workspace_invitation',
                        resource_name=str(invitation['invitation_id']),
                    )
                    st.success(f"Invitation created for {invitation['email']}.")
            except PermissionError as error:
                st.warning(str(error))
        invitation_rows = auth_service.list_workspace_invitations(workspace_identity) if hasattr(auth_service, 'list_workspace_invitations') else []
        invitation_table = safe_df(pd.DataFrame(invitation_rows))
        info_or_table(invitation_table, 'Workspace invitations will appear here after the first invitation is created in a persistent auth environment.')
        if not invitation_table.empty and st.session_state.get('auth_session', {}).get('signed_in'):
            accept_cols = st.columns([1.5, 0.8])
            accept_invite_id = accept_cols[0].selectbox(
                'Accept invitation',
                [''] + invitation_table.loc[invitation_table['status'] == 'Pending', 'invitation_id'].astype(str).tolist(),
                key='accept_workspace_invitation_id',
            )
            if accept_invite_id and accept_cols[1].button('Accept', key='accept_workspace_invitation'):
                accepted = auth_service.accept_workspace_invitation(st.session_state.get('auth_session', {}), accept_invite_id)
                if accepted is not None:
                    log_event(
                        'Workspace Invitation Accepted',
                        f"Accepted invitation '{accept_invite_id}' for workspace '{accepted.get('workspace_name', workspace_identity.get('workspace_name', 'Current workspace'))}'.",
                        'Workspace sharing',
                        'Data intake',
                        resource_type='workspace_invitation',
                        resource_name=accept_invite_id,
                    )
                    auth_session = st.session_state.get('auth_session', {})
                    st.session_state['workspace_name'] = str(accepted.get('workspace_name', st.session_state.get('workspace_name', 'Personal Demo Workspace')))
                    st.session_state['workspace_identity'] = auth_service.build_workspace_identity(auth_session, st.session_state.get('workspace_name'))
                    if application_service is not None:
                        application_service.hydrate_workspace_state(st.session_state)
                    st.rerun()
    collaboration_presence = application_service.list_collaboration_presence(workspace_identity) if application_service is not None else []
    presence_table = safe_df(pd.DataFrame(collaboration_presence))
    if not presence_table.empty:
        with st.expander('Active collaborators', expanded=advanced_sections_enabled):
            info_or_table(
                presence_table[['display_name', 'active_section', 'presence_state', 'updated_at']].head(10),
                'Active collaborators appear here when multiple signed-in sessions share the same workspace.',
            )
    st.caption(str(workspace_identity.get('workspace_access_summary', 'Workspace access summary is not available yet.')))
    st.caption('Use this workspace area to decide who owns the current pilot review, which role is being simulated, and where saved assets should belong.')
    for note in getattr(getattr(auth_service, 'status', None), 'notes', [])[:2]:
        st.caption(note)
    st.markdown('### Security / Governance Foundation')
    with st.expander('Security and governance controls', expanded=advanced_sections_enabled):
        governance_cols = st.columns([1, 1, 1])
        governance_cols[0].selectbox(
            'Redaction level',
            REDACTION_LEVEL_OPTIONS,
            key='workspace_governance_redaction_level',
        )
        governance_cols[1].selectbox(
            'Workspace export access',
            WORKSPACE_EXPORT_ACCESS_OPTIONS,
            key='workspace_governance_export_access',
        )
        governance_cols[2].toggle(
            'Watermark sensitive exports',
            key='workspace_governance_watermark_sensitive_exports',
        )
        governance_config = {
            'redaction_level': st.session_state.get('workspace_governance_redaction_level', 'Medium'),
            'workspace_export_access': st.session_state.get('workspace_governance_export_access', 'Editors and owners'),
            'watermark_sensitive_exports': bool(st.session_state.get('workspace_governance_watermark_sensitive_exports', True)),
        }
        workspace_security = auth_module.build_workspace_security_summary(
            workspace_identity,
            getattr(auth_service, 'status', None),
        )
        export_governance = build_export_governance_summary(
            str(st.session_state.get('export_policy_name', 'Internal Review')),
            pipeline.get('privacy_review', {}),
            workspace_identity,
            governance_config,
        )
        metric_row([(card['label'], card['value']) for card in workspace_security.get('summary_cards', [])[:4]])
        metric_row([(card['label'], card['value']) for card in export_governance.get('summary_cards', [])[:4]])
        info_or_table(
            safe_df(pd.DataFrame(workspace_security.get('boundaries_table', []))),
            'Workspace security boundary guidance appears here once the current workspace identity is available.',
        )
        info_or_table(
            safe_df(export_governance.get('controls_table')),
            'Export governance guidance appears here once privacy and workspace context are available.',
        )
        with st.expander('Data classification and handling'):
            classification_table = safe_df(export_governance.get('classification_table'))
            info_or_table(
                classification_table,
                'Data classification details appear here once privacy classification has been derived for the current dataset.',
            )
        for note in workspace_security.get('notes', []):
            st.caption(note)
        for note in export_governance.get('notes', []):
            st.caption(note)

    st.markdown('### Product Plan Awareness')
    with st.expander('Plan packaging and entitlement details', expanded=advanced_sections_enabled):
        plan_cols = st.columns([1, 1, 2])
        plan_cols[0].selectbox('Active plan', PLAN_OPTIONS, key='active_plan')
        plan_cols[1].selectbox('Plan enforcement', PLAN_ENFORCEMENT_OPTIONS, key='plan_enforcement_mode')
        plan_cols[2].caption(str(plan_awareness.get('description', 'Select a demo plan to preview how workspace limits and premium capabilities would be packaged in a product environment.')))
        if plan_awareness.get('plan_story'):
            st.caption(str(plan_awareness.get('plan_story')))
        metric_row([(card['label'], card['value']) for card in plan_awareness.get('summary_cards', [])])
        for note in plan_awareness.get('guidance_notes', []):
            st.caption(note)
        info_or_table(safe_df(plan_awareness.get('limits_table')), 'Plan guidance appears here once the active demo plan and workspace limits are loaded.')
        with st.expander('Plan packaging overview'):
            info_or_table(safe_df(plan_awareness.get('plan_comparison')), 'Plan packaging guidance appears here once the active plan catalog is available.')
        with st.expander('Soft limits and entitlements'):
            info_or_table(safe_df(plan_awareness.get('entitlement_summary')), 'Entitlement guidance appears here once the active plan context is available.')
        with st.expander('Premium capability coverage'):
            info_or_table(safe_df(plan_awareness.get('premium_features')), 'Premium capability guidance appears here once the active plan context is available.')

    st.markdown('### Product Settings / Admin Panel')
    with st.expander('Product settings and cache controls', expanded=advanced_sections_enabled):
        product_cols = st.columns(3)
        product_cols[0].toggle('Demo mode guidance', key='product_demo_mode_enabled')
        product_cols[1].toggle('Enable beta interest capture', key='product_beta_interest_enabled')
        product_cols[2].selectbox('Large dataset profile', list(LARGE_DATASET_PROFILES.keys()), key='product_large_dataset_profile')
        product_cols2 = st.columns(2)
        product_cols2[0].selectbox('Sampling explanation', ['Concise', 'Detailed'], key='product_sampling_explanation_mode')
        product_cols2[1].selectbox('Default report mode', REPORT_MODES, key='product_default_report_mode')
        product_cols2 = st.columns(2)
        product_cols2[0].selectbox('Default export policy', get_export_policy_presets()['policy_name'].tolist(), key='product_default_export_policy')
        product_cols2[1].selectbox('AI Copilot style', ['Concise', 'Detailed'], key='product_copilot_response_style')
        product_cols3 = st.columns(2)
        product_cols3[0].toggle('Show Copilot workflow previews', key='product_copilot_show_workflow_preview')
        product_cols3[1].caption('Use the controls above to tune startup guidance, demo capture, and Copilot output style.')
        product_settings = build_product_settings_summary(
            {key: st.session_state.get(key) for key in DEFAULT_PRODUCT_SETTINGS},
            plan_awareness,
        )
        metric_row([(card['label'], card['value']) for card in product_settings.get('summary_cards', [])])
        for note in product_settings.get('admin_notes', []):
            st.caption(note)
        info_or_table(safe_df(product_settings.get('settings_table')), 'Product settings guidance appears here once the current demo and admin preferences are loaded.')
        cache_summary = build_profile_cache_summary(st.session_state.get('profile_cache_metrics'))
        st.write('**Dataset profile cache**')
        metric_row([(card['label'], card['value']) for card in cache_summary.get('summary_cards', [])])
        for note in cache_summary.get('notes', []):
            st.caption(note)
        info_or_table(
            safe_df(cache_summary.get('settings_table')),
            'Profile cache metrics appear here after the first dataset profile request is cached.',
        )
    st.markdown('### Managed Task Foundation')
    with st.expander('Managed task and runtime controls', expanded=advanced_sections_enabled):
        job_runtime = pipeline.get('job_runtime', {})
        metric_row([
            ('Task Mode', str(job_runtime.get('status_label', 'Synchronous fallback'))),
            ('Backend Configured', 'Yes' if bool(job_runtime.get('backend_configured')) else 'No'),
            ('Tracked Job Runs', fmt(len(st.session_state.get('job_runs', [])))),
        ])
        for note in job_runtime.get('notes', []):
            st.caption(note)
        info_or_table(
            safe_df(pipeline.get('heavy_task_catalog')),
            'Managed task guidance appears here when the platform prepares the heavy-operation catalog.',
        )
        settings_action_cols = st.columns(3)
        if settings_action_cols[0].button('Apply export defaults', key='apply_product_export_defaults'):
            if not auth_module.workspace_identity_can_access(workspace_identity, 'settings_manage'):
                st.warning('Workspace viewers and analysts can review current settings, but only owners and admins can apply shared product defaults in signed-in mode.')
            else:
                st.session_state['report_mode'] = st.session_state.get('product_default_report_mode', 'Executive Summary')
                st.session_state['export_policy_name'] = st.session_state.get('product_default_export_policy', 'Internal Review')
                log_event(
                    'Product Settings Applied',
                    f"Applied product export defaults: {st.session_state['report_mode']} with {st.session_state['export_policy_name']}.",
                    'Product settings',
                    'Data intake',
                    resource_type='product_settings',
                    resource_name='export defaults',
                )
                st.success('Export defaults applied to the active session.')
                st.rerun()
        if settings_action_cols[1].button('Restore recommended settings', key='restore_product_settings_defaults'):
            if not auth_module.workspace_identity_can_access(workspace_identity, 'settings_manage'):
                st.warning('Workspace viewers and analysts can review current settings, but only owners and admins can reset shared product defaults in signed-in mode.')
            else:
                for key, value in DEFAULT_PRODUCT_SETTINGS.items():
                    st.session_state[key] = value
                log_event(
                    'Product Settings Reset',
                    'Restored the recommended product settings for demo mode, exports, and AI Copilot guidance.',
                    'Product settings',
                    'Data intake',
                    resource_type='product_settings',
                    resource_name='recommended defaults',
                )
                st.success('Recommended product settings restored.')
                st.rerun()
        if settings_action_cols[2].button('Clear profile cache', key='clear_profile_cache'):
            if not auth_module.workspace_identity_can_access(workspace_identity, 'settings_manage'):
                st.warning('Workspace viewers and analysts can review cache metrics, but only owners and admins can clear the shared profile cache in signed-in mode.')
            else:
                clear_profile_cache(st.session_state.setdefault('profile_cache_metrics', {}))
                log_event(
                    'Profile Cache Cleared',
                    'Cleared the dataset profile cache and reset the current session cache metrics.',
                    'Product settings',
                    'Data intake',
                    resource_type='product_settings',
                    resource_name='profile cache',
                )
                st.success('Dataset profile cache cleared.')
                st.rerun()

    with workflow_guidance_section:
        render_surface_panel(
            'Workflow and solution guidance',
            'Keep this section open when you are deciding what the dataset can support, which healthcare workflow fits best, and how to frame next-step recommendations.',
            tone='info',
        )
        st.markdown('### Guided Workflow')
        demo_mode = pipeline.get('demo_mode_content', {})
        if not st.session_state.get('product_demo_mode_enabled', True):
            st.info('Guided workflow content is currently muted in Product Settings. You can still explore every tab and workflow manually.')
        elif demo_mode and is_demo_dataset:
            st.caption(str(demo_mode.get('intro', 'Use the guided flow below to walk through the strongest parts of the platform.')))
            if intelligence.get('dataset_type_label'):
                st.write(f"**Dataset type:** {intelligence.get('dataset_type_label')}")
            st.info('Use the Recommended Workflow above as the main step-by-step path for this demo dataset.')
            with st.expander('Demo highlights checklist'):
                for item in demo_mode.get('demo_highlights_checklist', []):
                    st.write(f'- {item}')
                for item in demo_mode.get('insights_to_look_for', []):
                    st.write(f'- Insight focus: {item}')
                st.caption(str(demo_mode.get('synthetic_support_note', '')))
        else:
            st.caption('The active uploaded dataset is now the only workflow context for this session. Review the recommended workflow, readiness guidance, and remediation details below.')

    with workflow_guidance_section:
        st.markdown('### Beta Interest / Early Access')
        if not st.session_state.get('product_beta_interest_enabled', True):
            st.info('Beta interest capture is currently muted in Product Settings. You can turn it back on at any time for demo follow-up collection.')
        else:
            st.caption('Capture lightweight follow-up interest during demos. Choose local, database, or API routing without leaving the current workspace.')
            feedback = st.session_state.get('beta_interest_feedback')
            if isinstance(feedback, dict) and str(feedback.get('message', '')).strip():
                status = str(feedback.get('status', 'success')).lower()
                if status == 'error':
                    st.error(str(feedback.get('message', '')))
                else:
                    st.success(str(feedback.get('message', '')))
            if st.session_state.get('beta_interest_focus_note'):
                st.info(str(st.session_state.get('beta_interest_focus_note')))
            beta_cols = st.columns(2)
            beta_cols[0].text_input('Name', key='beta_interest_name', placeholder='Example: Jamie Rivera')
            beta_cols[1].text_input('Email', key='beta_interest_email', placeholder='name@organization.com')
            beta_cols2 = st.columns(2)
            beta_cols2[0].text_input('Organization', key='beta_interest_organization', placeholder='Hospital, research team, or consulting group')
            beta_cols2[1].selectbox('Storage mode', BETA_INTEREST_STORAGE_OPTIONS, key='beta_interest_storage_mode')
            st.text_area(
                'Use case',
                key='beta_interest_use_case',
                height=100,
                placeholder='Describe the workflow, pilot, or analytics need you want to evaluate.',
            )
            if str(st.session_state.get('beta_interest_storage_mode', 'LOCAL')).upper() == 'API':
                st.text_input(
                    'API endpoint',
                    key='beta_interest_api_endpoint',
                    placeholder='https://example.crm.com/api/beta-interest',
                )
            if st.button('Save Beta Interest', key='save_beta_interest', use_container_width=True):
                name = str(st.session_state.get('beta_interest_name', '')).strip()
                email = str(st.session_state.get('beta_interest_email', '')).strip()
                organization = str(st.session_state.get('beta_interest_organization', '')).strip()
                use_case = str(st.session_state.get('beta_interest_use_case', '')).strip()
                errors = validate_beta_interest(name, email, organization, use_case)
                if errors:
                    st.session_state['beta_interest_feedback'] = {
                        'status': 'error',
                        'message': ' '.join(errors),
                    }
                else:
                    with st.spinner('Saving beta interest...'):
                        try:
                            saved = save_beta_interest_submission(
                                local_submissions=st.session_state.setdefault('beta_interest_submissions', []),
                                name=name,
                                email=email,
                                organization=organization,
                                use_case=use_case,
                                workspace_identity=workspace_identity,
                                dataset_name=dataset_name,
                                dataset_source_mode=str(source_meta.get('source_mode', 'Unknown')),
                                storage_mode=str(st.session_state.get('beta_interest_storage_mode', 'LOCAL')),
                                application_service=application_service,
                                api_url=str(st.session_state.get('beta_interest_api_endpoint', '')),
                            )
                            workspace_id = str(workspace_identity.get('workspace_id', 'guest-demo-workspace'))
                            st.session_state.setdefault('workspace_beta_interest_submissions', {})[workspace_id] = st.session_state.get('beta_interest_submissions', [])
                            persist_active_workspace_state(st.session_state)
                            log_event(
                                'Beta Interest Submitted',
                                f"Captured beta interest from {saved['name']} ({saved['email']}) for workspace '{workspace_identity.get('workspace_name', 'Guest Demo Workspace')}'.",
                                'Beta interest capture',
                                'Data intake',
                                resource_type='beta_interest',
                                resource_name=str(saved.get('email', '')),
                            )
                            st.session_state['beta_interest_feedback'] = {
                                'status': 'success',
                                'message': f"Thank you! We'll be in touch at {saved['email']}",
                            }
                            st.session_state['beta_interest_name'] = ''
                            st.session_state['beta_interest_email'] = ''
                            st.session_state['beta_interest_organization'] = ''
                            st.session_state['beta_interest_use_case'] = ''
                        except Exception:
                            st.session_state['beta_interest_feedback'] = {
                                'status': 'error',
                                'message': f"Error saving. Please try again or email {support_email()}",
                            }
                    st.rerun()
            persisted_beta_submissions = application_service.list_beta_interest_submissions(workspace_identity) if application_service is not None else []
            merged_beta_submissions = merge_beta_interest_submissions(
                st.session_state.get('beta_interest_submissions', []),
                persisted_beta_submissions,
            )
            beta_summary = build_beta_interest_summary(merged_beta_submissions)
            metric_row([(card['label'], card['value']) for card in beta_summary.get('summary_cards', [])])
            for note in beta_summary.get('notes', []):
                st.caption(note)
            beta_history = safe_df(beta_summary.get('history'))
            info_or_table(beta_history, 'Beta interest capture is enabled for this demo workspace. Add a contact above to start building a follow-up list.')
            if not beta_history.empty:
                st.download_button(
                    'Export Beta Interest CSV',
                    data=beta_interest_csv_bytes(merged_beta_submissions),
                    file_name='beta_interest_submissions.csv',
                    mime='text/csv',
                    key='beta_interest_export_csv',
                )
                status_cols = st.columns([1.6, 1.0, 0.8])
                status_options = {
                    f"{row['submitted_at']} | {row['email']} | {row['follow_up_status']}": row['submission_id']
                    for _, row in beta_history.iterrows()
                    if str(row.get('submission_id', '')).strip()
                }
                selected_submission_label = status_cols[0].selectbox(
                    'Submission',
                    list(status_options.keys()),
                    key='beta_interest_selected_submission',
                )
                selected_status = status_cols[1].selectbox(
                    'Follow-up status',
                    BETA_INTEREST_STATUS_OPTIONS,
                    key='beta_interest_selected_status',
                )
                if status_cols[2].button('Update Status', key='beta_interest_update_status', use_container_width=True):
                    submission_id = status_options[selected_submission_label]
                    updated_submission = update_beta_interest_status(
                        st.session_state.setdefault('beta_interest_submissions', []),
                        submission_id,
                        follow_up_status=selected_status,
                    )
                    if updated_submission is None:
                        st.session_state['beta_interest_submissions'] = merged_beta_submissions
                        updated_submission = update_beta_interest_status(
                            st.session_state.setdefault('beta_interest_submissions', []),
                            submission_id,
                            follow_up_status=selected_status,
                        )
                    if application_service is not None:
                        contacted_at = str(updated_submission.get('contacted_at', '')) if updated_submission is not None else ''
                        completed_at = str(updated_submission.get('completed_at', '')) if updated_submission is not None else ''
                        application_service.update_beta_interest_submission_status(
                            workspace_identity,
                            submission_id,
                            follow_up_status=selected_status,
                            contacted_at=contacted_at,
                            completed_at=completed_at,
                        )
                    workspace_id = str(workspace_identity.get('workspace_id', 'guest-demo-workspace'))
                    st.session_state.setdefault('workspace_beta_interest_submissions', {})[workspace_id] = st.session_state.get('beta_interest_submissions', [])
                    persist_active_workspace_state(st.session_state)
                    log_event(
                        'Beta Interest Status Updated',
                        f"Updated beta interest submission '{submission_id}' to {selected_status}.",
                        'Beta interest capture',
                        'Data intake',
                        resource_type='beta_interest',
                        resource_name=submission_id,
                    )
                    st.session_state['beta_interest_feedback'] = {
                        'status': 'success',
                        'message': f'Updated follow-up status to {selected_status}.',
                    }
                    st.rerun()

    with workflow_guidance_section:
        st.markdown('### Workflow Guidance')
        demo_guidance = pipeline.get('demo_guidance', {})
        if demo_guidance:
            metric_row([
                ('Detected Dataset Type', str(demo_guidance.get('detected_dataset_type', 'Not classified'))),
                ('Recommended Workflow', str(demo_guidance.get('recommended_workflow', 'Healthcare Data Readiness'))),
                ('Recommended Package', str(demo_guidance.get('recommended_package', 'Healthcare Data Readiness'))),
                ('Relevant Modules', fmt(len(demo_guidance.get('relevant_modules', [])))),
            ])
            st.caption(str(demo_guidance.get('narrative', '')))
            for line in demo_guidance.get('highlights', []):
                st.write(f'- {line}')
            st.caption('The full step-by-step flow is shown once in the Recommended Workflow section above.')
        else:
            st.info('Workflow guidance will appear here after the platform classifies the dataset and maps it to a recommended workflow.')

        st.markdown('### Healthcare Solution Layers')
        solution_layers = pipeline.get('solution_layers', {})
        cards = safe_df(solution_layers.get('solution_cards'))
        if cards.empty:
            st.info('Solution-layer guidance will appear here once the platform maps the dataset to one or more healthcare solution layers.')
        else:
            summary = solution_layers.get('summary', {})
            metric_row([
                ('Ready Now', fmt(summary.get('ready_now', 0))),
                ('Partially Supported', fmt(summary.get('partially_supported', 0))),
                ('Needs Stronger Data', fmt(summary.get('needs_stronger_source_support', 0))),
                ('Recommended Start', str(summary.get('recommended_starting_point', 'Healthcare Data Readiness'))),
            ])
            st.caption(str(solution_layers.get('narrative', '')))
            info_or_table(cards[['solution_group', 'status_label', 'who_it_is_for']], 'Solution-layer fit summaries appear here when the dataset has enough structure to map supported healthcare use cases.')
            with st.expander('Solution layer details'):
                info_or_table(cards[['solution_group', 'modules', 'outcomes_supported']], 'Solution-layer module details appear here when workflow fit has been calculated for the current dataset.')

        st.markdown('### Automatic Use-Case Detection')
        use_case = pipeline.get('use_case_detection', {})
        if use_case:
            metric_row([
                ('Detected Use Case', str(use_case.get('detected_use_case', 'Not classified'))),
                ('Recommended Workflow', str(use_case.get('recommended_workflow', 'Healthcare Data Readiness'))),
                ('Use-Case Confidence', fmt(use_case.get('detected_use_case_confidence', 0.0), 'pct')),
                ('Ready Modules', fmt(use_case.get('available_module_count', 0))),
            ])
            st.caption(str(use_case.get('narrative', '')))
            relevant_modules = use_case.get('relevant_modules', [])
            if relevant_modules:
                st.write('**Relevant modules**')
                for module in relevant_modules[:6]:
                    st.write(f'- {module}')
            unavailable = safe_df(use_case.get('unavailable_modules'))
            partial = safe_df(use_case.get('partially_available_modules'))
            if not unavailable.empty:
                info_or_table(unavailable.head(6), 'No blocked module details are available yet.')
            if not partial.empty:
                info_or_table(partial.head(6), 'No partial module details are available yet.')

        st.markdown('### Guided Solution Packages')
        solution_packages = pipeline.get('solution_packages', {})
        packages_table = safe_df(solution_packages.get('packages_table'))
        if packages_table.empty:
            st.info('Solution package guidance will appear here once the platform recommends a workflow package for the current dataset.')
        else:
            st.write(f"**Recommended package:** {solution_packages.get('recommended_package', 'Healthcare Data Readiness')}")
            info_or_table(packages_table, 'Solution package summaries appear here when the dataset has a recommended workflow package.')
            details = solution_packages.get('package_details', {})
            selected_package = st.selectbox(
                'Solution package details',
                packages_table['solution_package'].astype(str).tolist(),
                index=max(
                    0,
                    packages_table['solution_package'].astype(str).tolist().index(solution_packages.get('recommended_package'))
                    if solution_packages.get('recommended_package') in packages_table['solution_package'].astype(str).tolist()
                    else 0,
                ),
                key='solution_package_detail',
            )
            package = details.get(selected_package, {})
            if package:
                st.caption(str(package.get('description', '')))
                st.write('**Recommended steps**')
                for step in package.get('recommended_steps', []):
                    st.write(f'- {step}')
                st.write('**Relevant modules**')
                for module in package.get('relevant_modules', []):
                    st.write(f'- {module}')
                st.write('**Suggested AI Copilot prompts**')
                for prompt in package.get('suggested_prompts', []):
                    st.write(f'- {prompt}')
                st.write('**Recommended exports**')
                for export_name in package.get('recommended_exports', []):
                    st.write(f'- {export_name}')

        st.markdown('### Market-Specific Solution Views')
        market_views = pipeline.get('market_solution_views', {})
        market_table = safe_df(market_views.get('market_solution_views'))
        if market_table.empty:
            st.info('Market-specific solution views will appear here once the current dataset is mapped to likely buyer or pilot contexts.')
        else:
            st.write(f"**Recommended buyer view:** {market_views.get('recommended_solution_view', 'Healthcare Data Readiness')}")
            st.caption(str(market_views.get('narrative', '')))
            info_or_table(
                market_table[
                    [
                        'solution_view',
                        'best_fit_workflow',
                        'best_fit_package',
                        'current_fit',
                        'recommended_now',
                    ]
                ],
                'Market-specific solution summaries appear here when the platform can map the dataset to hospital, research, readiness, or consulting views.',
            )

    with operations_review_section:
        render_surface_panel(
            'Operational review and launch readiness',
            'Use this area for lineage, run history, workspace operations, product diagnostics, and launch-readiness checks.',
        )
        st.markdown('### Data Lineage')
        info_or_table(safe_df(pipeline['lineage'].get('source_table')), 'Lineage source details appear here when source tracking is available for the current dataset.')
        info_or_table(safe_df(pipeline['lineage'].get('derived_fields_table')), 'Derived field lineage appears here when helper fields or transformations are part of the current analysis.')
        with st.expander('Transformation steps'):
            for line in pipeline['lineage'].get('transformation_steps', []):
                st.write(f'- {line}')

        st.markdown('### Analysis Log')
        info_or_table(build_audit_log_view(st.session_state.get('analysis_log', [])), 'Analysis actions will appear here as you move through workflow steps, exports, and Copilot prompts.')
        st.markdown('### Run History Summary')
        audit_bundle = pipeline.get('audit_summary_bundle', {})
        info_or_table(safe_df(audit_bundle.get('audit_summary')), 'Run history will populate here after datasets are analyzed in this workspace.')
        if audit_bundle.get('audit_summary_text'):
            st.caption(str(audit_bundle['audit_summary_text']))

        st.markdown('### Usage Analytics')
        usage = (admin_ops_view or {}).get('usage', {})
        metric_row([(card['label'], card['value']) for card in usage.get('summary_cards', [])])
        info_or_table(safe_df(usage.get('activity_table')), 'Usage analytics will appear here as workflows, exports, and Copilot actions are used in this session.')
        with st.expander('Module and dataset usage details'):
            info_or_table(safe_df(usage.get('module_table')), 'Module visit details appear here after the main analysis sections are opened.')
            info_or_table(safe_df(usage.get('dataset_runs')), 'Dataset run history appears here after one or more datasets are analyzed in this workspace.')
        for note in usage.get('notes', []):
            st.caption(note)
        st.markdown('### Customer Success Summary')
        success_summary = (admin_ops_view or {}).get('customer_success', {})
        metric_row([(card['label'], card['value']) for card in success_summary.get('summary_cards', [])])
        info_or_table(
            safe_df(success_summary.get('value_table')),
            'Customer success indicators will appear here after datasets, workflows, and reports move through this workspace.',
        )
        for note in success_summary.get('notes', []):
            st.caption(note)

        st.markdown('### Product Admin / Customer Ops')
        product_admin = (admin_ops_view or {}).get('product_admin', {})
        metric_row([(card['label'], card['value']) for card in product_admin.get('summary_cards', [])])
        admin_cols = st.columns(2)
        with admin_cols[0]:
            st.write('**Workspace and plan summary**')
            info_or_table(
                safe_df(product_admin.get('workspace_table')),
                'Workspace operations guidance appears here once workspace context is available.',
            )
            info_or_table(
                safe_df(product_admin.get('plan_table')),
                'Plan operations guidance appears here once the active plan is loaded.',
            )
        with admin_cols[1]:
            st.write('**Usage and report operations**')
            info_or_table(
                safe_df(product_admin.get('usage_table')),
                'Usage operations guidance appears here once activity is captured in the workspace.',
            )
            info_or_table(
                safe_df(product_admin.get('reports_table')),
                'Report operations guidance appears here after reports or bundles are generated.',
            )
        with st.expander('Customer ops detail'):
            info_or_table(
                safe_df(product_admin.get('customer_success_table')),
                'Customer success indicators will appear here as datasets, workflows, and reports move through this workspace.',
            )
            info_or_table(
                safe_df(product_admin.get('dataset_ops_table')),
                'Persisted dataset metadata will appear here once workspace persistence is configured.',
            )
            diagnostics_cards = product_admin.get('support_diagnostics_summary_cards', [])
            if diagnostics_cards:
                metric_row([(card['label'], card['value']) for card in diagnostics_cards])
            info_or_table(
                safe_df(product_admin.get('support_diagnostics_table')),
                'Support diagnostics will appear here as operational events and protected errors are captured.',
            )
            info_or_table(
                safe_df(product_admin.get('support_error_frequency_table')),
                'Error frequency by operation will appear here after recurring failures are observed.',
            )
            info_or_table(
                safe_df(product_admin.get('support_recurring_error_table')),
                'Recurring error alerts will appear here when the same operation keeps failing.',
            )
            with st.expander('Retained error history (30 days)'):
                info_or_table(
                    safe_df(product_admin.get('support_recent_error_table')),
                    'Retained error history will appear here once protected or runtime failures are captured.',
                )
        for note in product_admin.get('notes', []):
            st.caption(note)
        landing_summary = pipeline.get('landing_summary', {})
        for note in landing_summary.get('workspace_handoff_cues', [])[:3]:
            st.caption(note)

        st.markdown('### Production Hardening')
        preflight = pipeline.get('preflight', {})
        column_validation = pipeline.get('column_validation', {})
        for warning in preflight.get('warnings', []):
            st.warning(warning)
        info_or_table(safe_df(preflight.get('checks_table')), 'Preflight guardrails appear here after file size, memory, and structural checks are evaluated.')
        if column_validation:
            st.caption(f"{str(column_validation.get('summary', ''))} Use the 'View column health details' summary above for the active empty-column alert.")
            info_or_table(safe_df(column_validation.get('checks_table')), 'Column validation details appear here when header, empty-column, or low-signal checks are relevant.')
            info_or_table(safe_df(column_validation.get('issue_samples')), 'Column issue samples appear here when the platform finds headers or fields that need review.')
        info_or_table(safe_df(pipeline.get('deployment_health_checks')), 'Deployment health checks appear here after the current environment and demo assets are evaluated.')
        info_or_table(safe_df(pipeline.get('performance_diagnostics')), 'Performance diagnostics appear here once dataset size, sampling, and runtime conditions are assessed.')
        with st.expander('Deployment support notes'):
            info_or_table(build_deployment_support_notes(), 'Deployment support notes appear here when the environment guidance pack is available.')
        st.markdown('### Deployment & Launch Readiness')
        startup_readiness = pipeline.get('startup_readiness', {})
        if startup_readiness:
            metric_row([(card['label'], card['value']) for card in startup_readiness.get('summary_cards', [])])
            for note in startup_readiness.get('notes', []):
                st.caption(note)
        environment_checks = pipeline.get('environment_checks', {})
        info_or_table(safe_df(environment_checks.get('checks_table')), 'Deployment environment checks appear here once startup readiness has been evaluated.')
        for note in environment_checks.get('notes', []):
            st.caption(note)
        info_or_table(build_launch_checklist(), 'Launch checklist guidance appears here when deployment readiness support is loaded.')
        with st.expander('Config and secrets guidance'):
            info_or_table(build_config_guidance(), 'Configuration guidance appears here once secrets and environment assumptions are loaded.')

        if is_demo_dataset:
            st.markdown('### Demo Dataset Onboarding')
        else:
            st.markdown('### Uploaded Dataset Onboarding')
        onboarding = pipeline.get('dataset_onboarding', {})
        onboarding_summary = onboarding.get('dataset_onboarding_summary', {})
        if onboarding_summary:
            st.write(str(onboarding_summary.get('suitability', 'This dataset is ready for schema-flexible analytics and walkthrough review.')))
            if intelligence.get('dataset_type_label'):
                st.write(f"**Dataset intelligence type:** {intelligence.get('dataset_type_label')}")
            st.caption(str(onboarding_summary.get('synthetic_note', '')))
        info_or_table(safe_df(onboarding.get('module_unlock_guide')), 'Module unlock guidance appears here when the platform can map the current fields to supported healthcare workflows.')
        info_or_table(safe_df(onboarding.get('data_upgrade_suggestions')), 'Data upgrade suggestions appear here when stronger source fields would unlock additional analytics.')

    with collaboration_tools_section:
        render_surface_panel(
            'Collaboration, portability, and comparison tools',
            'Save analysis state, package repeatable workflows, add handoff notes, and compare related datasets without losing the active dataset context.',
            tone='accent',
        )
        st.markdown('### Save Analysis Snapshots')
        snapshot_name = st.text_input('Snapshot name', key='snapshot_name')
        cols = st.columns(2)
        if cols[0].button('Save Analysis Snapshot', key='save_snapshot') and snapshot_name.strip():
            try:
                save_snapshot_to_workspace(
                    st.session_state,
                    snapshot_name=snapshot_name.strip(),
                    dataset_name=dataset_name,
                    controls=active_controls(),
                    workspace_identity=workspace_identity,
                )
                if application_service is not None:
                    dataset_versions = application_service.list_dataset_versions(workspace_identity, dataset_name)
                    active_version_id = str(dataset_versions[0]['version_id']) if dataset_versions else ''
                    application_service.save_workspace_snapshot_record(
                        workspace_identity,
                        snapshot_name=snapshot_name.strip(),
                        dataset_name=dataset_name,
                        dataset_version_id=active_version_id,
                        snapshot_payload=dict(st.session_state.get('saved_snapshots', {}).get(snapshot_name.strip(), {})),
                    )
                    application_service.persist_workspace_state(st.session_state)
                log_event(
                    'Snapshot Saved',
                    f"Saved analysis snapshot '{snapshot_name.strip()}' in workspace '{workspace_identity.get('workspace_name', 'Guest Demo Workspace')}'.",
                    'Snapshot management',
                    'Data intake',
                    resource_type='snapshot',
                    resource_name=snapshot_name.strip(),
                )
                st.success('Analysis snapshot saved for this session.')
            except PermissionError as error:
                st.warning(str(error))
        snapshots = [''] + sorted(st.session_state.get('saved_snapshots', {}).keys())
        selected_snapshot = cols[1].selectbox('Reload snapshot', snapshots, key='selected_snapshot')
        if selected_snapshot and st.button('Load Snapshot', key='load_snapshot'):
            try:
                load_snapshot_into_session(st.session_state, selected_snapshot, workspace_identity)
                log_event(
                    'Snapshot Loaded',
                    f"Loaded analysis snapshot '{selected_snapshot}' from workspace '{workspace_identity.get('workspace_name', 'Guest Demo Workspace')}'.",
                    'Snapshot management',
                    'Data intake',
                    resource_type='snapshot',
                    resource_name=selected_snapshot,
                )
                st.rerun()
            except PermissionError as error:
                st.warning(str(error))
        if application_service is not None:
            persisted_snapshot_records = safe_df(pd.DataFrame(application_service.list_workspace_snapshot_records(workspace_identity)))
            if not persisted_snapshot_records.empty:
                with st.expander('Persistent snapshot restore points'):
                    info_or_table(
                        persisted_snapshot_records[['snapshot_name', 'dataset_name', 'dataset_version_id', 'created_by_user_id', 'created_at']],
                        'Persistent snapshot restore points appear here when database-backed workspace snapshots are available.',
                    )

    with collaboration_tools_section:
        st.markdown('### Workflow Packs')
        workflow_name = st.text_input('Workflow pack name', key='workflow_pack_name')
        strict_plan = bool(plan_awareness.get('strict_enforcement'))
        workflow_limit_reached = bool(plan_awareness.get('workflow_pack_limit_reached'))
        save_pack_disabled = strict_plan and workflow_limit_reached
        if save_pack_disabled:
            st.info('Workflow pack saving is paused because the active plan is at its current limit. Switch to Demo-safe mode or a higher plan to continue saving additional packs.')
        if st.button('Save Workflow Pack', key='save_workflow_pack', disabled=save_pack_disabled) and workflow_name.strip():
            try:
                details = build_workflow_pack_details(active_controls(), st.session_state.get('analysis_template', 'General Review'), dataset_context={'dataset_source_mode': source_meta.get('source_mode'), 'demo_dataset_name': dataset_name if source_meta.get('source_mode') == 'Demo dataset' else None})
                save_workflow_pack_to_workspace(
                    st.session_state,
                    workflow_name=workflow_name.strip(),
                    details=details,
                    summary=build_workflow_pack_summary(active_controls(), st.session_state.get('analysis_template', 'General Review')),
                    controls=active_controls(),
                    workspace_identity=workspace_identity,
                )
                if application_service is not None:
                    application_service.persist_workspace_state(st.session_state)
                log_event(
                    'Workflow Pack Saved',
                    f"Saved workflow pack '{workflow_name.strip()}' in workspace '{workspace_identity.get('workspace_name', 'Guest Demo Workspace')}'.",
                    'Workflow packs',
                    'Data intake',
                    resource_type='workflow_pack',
                    resource_name=workflow_name.strip(),
                )
                st.success('Workflow pack saved for this session.')
            except PermissionError as error:
                st.warning(str(error))
        packs = [''] + sorted(st.session_state.get('workflow_packs', {}).keys())
        selected_pack = st.selectbox('Reload workflow pack', packs, key='selected_workflow_pack')
        if selected_pack:
            details = st.session_state['workflow_packs'][selected_pack].get('details', {})
            st.write(details.get('summary', ''))
            for line in details.get('highlighted_controls', []):
                st.write(f'- {line}')
            if st.button('Load Workflow Pack', key='load_workflow_pack'):
                try:
                    load_workflow_pack_into_session(st.session_state, selected_pack, workspace_identity)
                    log_event(
                        'Workflow Pack Loaded',
                        f"Loaded workflow pack '{selected_pack}' from workspace '{workspace_identity.get('workspace_name', 'Guest Demo Workspace')}'.",
                        'Workflow packs',
                        'Data intake',
                        resource_type='workflow_pack',
                        resource_name=selected_pack,
                    )
                    st.rerun()
                except PermissionError as error:
                    st.warning(str(error))

    with collaboration_tools_section:
        st.markdown('### Session Export / Import')
        session_bundle = build_session_export_bundle(
            st.session_state,
            dataset_name,
            source_meta,
            workspace_identity,
        )
        session_bundle_bytes = build_session_export_text(session_bundle)
        storage_service = st.session_state.get('storage_service')
        if storage_service is not None and bool(getattr(storage_service, 'enabled', False)):
            stored_bundle_keys = st.session_state.setdefault('stored_session_bundle_artifacts', {})
            storage_key = (
                str(workspace_identity.get('workspace_id', 'guest-demo-workspace')),
                str(dataset_name or 'current-dataset'),
                str(source_meta.get('source_mode', 'session')),
            )
            if storage_key not in stored_bundle_keys:
                stored_bundle_keys[storage_key] = storage_service.save_session_bundle(
                    workspace_identity,
                    dataset_name=dataset_name or 'current-dataset',
                    file_name='smart_dataset_analyzer_session.json',
                    payload=session_bundle_bytes,
                )
        st.download_button(
            'Download Session Bundle',
            data=session_bundle_bytes,
            file_name='smart_dataset_analyzer_session.json',
            mime='application/json',
        )
        imported_session = st.file_uploader('Import session bundle', type=['json'], key='import_session_bundle')
        if imported_session is not None and st.button('Restore Session Bundle', key='restore_session_bundle'):
            try:
                imported_payload = parse_session_import(imported_session.getvalue().decode('utf-8'))
                restore_result = restore_session_bundle(
                    imported_payload,
                    st.session_state,
                    {
                        'dataset_source_mode': ['Built-in example dataset', 'Uploaded dataset'],
                        'demo_dataset_name': list(DEMO_DATASETS.keys()),
                        'analysis_template': ['General Review', 'Hospital Readmission Analysis', 'Healthcare Cost Analysis', 'Population Health Monitoring', 'Clinical Outcomes Analysis', 'Executive Review', 'Operations Review'],
                        'report_mode': REPORT_MODES,
                        'export_policy_name': get_export_policy_presets()['policy_name'].tolist(),
                        'active_role': ROLE_OPTIONS,
                        'active_plan': PLAN_OPTIONS,
                        'plan_enforcement_mode': PLAN_ENFORCEMENT_OPTIONS,
                    },
                )
                workspace_id = str(workspace_identity.get('workspace_id', 'guest-demo-workspace'))
                st.session_state.setdefault('workspace_saved_snapshots', {})[workspace_id] = st.session_state.get('saved_snapshots', {})
                st.session_state.setdefault('workspace_workflow_packs', {})[workspace_id] = st.session_state.get('workflow_packs', {})
                if application_service is not None:
                    application_service.persist_workspace_state(st.session_state)
                    application_service.hydrate_workspace_state(st.session_state)
                log_event(
                    'Session Bundle Restored',
                    f"Restored a portable session bundle with {len(restore_result.get('applied_keys', []))} applied values.",
                    'Session portability',
                    'Data intake',
                )
                st.success('Session bundle restored for the active workspace.')
                for note in restore_result.get('notes', []):
                    st.caption(note)
                st.rerun()
            except Exception as error:
                st.error('The selected session bundle could not be restored.')
                st.caption('Support detail: the bundle format or settings were not compatible with the current session. Try a newer bundle, or restore into a workspace with the same demo/source context.')

    with collaboration_tools_section:
        st.markdown('### Collaboration Notes')
        note_summary = build_collaboration_note_summary(st.session_state.get('collaboration_notes', []))
        st.caption('Capture findings, handoff notes, and open questions here so reviewers can quickly see what matters across datasets, workflow packs, reports, and analysis sections.')
        metric_row([(card['label'], card['value']) for card in note_summary.get('summary_cards', [])])
        note_targets = build_note_target_options(
            dataset_name,
            sorted(st.session_state.get('workflow_packs', {}).keys()),
            REPORT_MODES,
            TAB_LABELS,
        )
        note_target_labels = ['All Targets'] + [option['label'] for option in note_targets]
        note_cols = st.columns([1.4, 1.2])
        selected_note_target = note_cols[0].selectbox('Annotation target', note_target_labels[1:] or ['Dataset | Current dataset'], key='collaboration_note_target')
        note_text = note_cols[1].text_area('Add a note', key='collaboration_note_text', height=90, placeholder='Capture a finding, a question for review, or a handoff note for this workspace.')
        if st.button('Save Note', key='save_collaboration_note') and note_text.strip():
            target = next((option for option in note_targets if option['label'] == selected_note_target), {'target_type': 'Dataset', 'target_name': dataset_name or 'Current dataset'})
            try:
                add_collaboration_note_to_workspace(
                    st.session_state,
                    target_type=target['target_type'],
                    target_name=target['target_name'],
                    note_text=note_text,
                    workspace_identity=workspace_identity,
                    section_name=target['target_name'],
                )
                workspace_id = str(workspace_identity.get('workspace_id', 'guest-demo-workspace'))
                st.session_state.setdefault('workspace_collaboration_notes', {})[workspace_id] = st.session_state.get('collaboration_notes', [])
                if application_service is not None:
                    application_service.persist_workspace_state(st.session_state)
                log_event(
                    'Collaboration Note Saved',
                    f"Saved a note for {target['target_type'].lower()} '{target['target_name']}' in workspace '{workspace_identity.get('workspace_name', 'Guest Demo Workspace')}'.",
                    'Collaboration notes',
                    'Data intake',
                    resource_type='collaboration_note',
                    resource_name=target['target_name'],
                )
                st.success('Collaboration note saved for this workspace.')
                st.rerun()
            except PermissionError as error:
                st.warning(str(error))
        recent_notes = safe_df(note_summary.get('recent_notes'))
        top_targets = safe_df(note_summary.get('top_targets'))
        discovery_cols = st.columns(2)
        with discovery_cols[0]:
            st.write('**Recent notes**')
            info_or_table(
                recent_notes,
                'Recent notes will appear here after the first annotation is saved in this workspace.',
            )
        with discovery_cols[1]:
            st.write('**Most active note targets**')
            info_or_table(
                top_targets,
                'Most-active note targets will appear here once multiple annotations are saved.',
            )
        note_filter = st.selectbox('Note history view', note_target_labels, key='collaboration_note_filter')
        filtered_notes = build_collaboration_notes_view(st.session_state.get('collaboration_notes', []), note_filter)
        if note_filter != 'All Targets':
            st.caption(f'Currently reviewing notes for {note_filter}.')
        info_or_table(
            filtered_notes,
            'No collaboration notes have been saved in this workspace yet.',
        )
        with st.expander('Note history by target'):
            info_or_table(safe_df(note_summary.get('history')), 'No note history summary is available yet.')

    with collaboration_tools_section:
        st.markdown('### Multi-file Linked Analysis')
        related = st.file_uploader('Upload a related file to preview a safe join', type=['csv', 'xlsx', 'xlsm', 'xls'], key='related_file')
        if related is not None:
            try:
                related_df, _ = load_uploaded_file(related.name, related.getvalue())
                related_df, related_temporal = augment_temporal_fields(related_df)
                related_structure = detect_structure(related_df)
                related_semantic = infer_semantic_mapping(related_df, related_structure)
                join_candidates = detect_join_candidates(pipeline['data'], pipeline['structure'], related_df, related_structure)
                st.write(build_join_recommendation(join_candidates, infer_linked_dataset_role(pipeline['semantic']), infer_linked_dataset_role(related_semantic)))
                if related_temporal.get('synthetic_date_created'):
                    st.caption(str(related_temporal.get('note', 'A synthetic event_date was generated for the related dataset preview.')))
                info_or_table(join_candidates.head(10), 'No strong automatic join candidates were detected.')
                if not join_candidates.empty:
                    top = join_candidates.iloc[0]
                    left_keys = sorted(join_candidates['left_key'].unique().tolist())
                    right_keys = sorted(join_candidates['right_key'].unique().tolist())
                    left_key = st.selectbox('Primary key', left_keys, index=left_keys.index(top['left_key']), key='linked_left_key')
                    right_key = st.selectbox('Related key', right_keys, index=right_keys.index(top['right_key']), key='linked_right_key')
                    merge_preview = preview_linked_merge(pipeline['data'], related_df, left_key, right_key)
                    if merge_preview.get('available'):
                        info_or_table(safe_df(merge_preview.get('preview')), 'No merge preview is available.')
            except DataLoadError as error:
                st.error(str(error))
        else:
            st.info('Upload a related patient, treatment, facility, or cost file to preview linked analysis safely.')

        st.markdown('### Comparative Dataset Analysis')
        uploads = st.file_uploader('Upload additional datasets for side-by-side comparison', type=['csv', 'xlsx', 'xlsm', 'xls'], accept_multiple_files=True, key='comparison_uploads')
        rows = [
            {
                'dataset_name': 'Current Dataset',
                'source_mode': 'Active session',
                'rows': len(pipeline['data']),
                'columns': len(pipeline['data'].columns),
                'missing_values': int(pipeline['data'].isna().sum().sum()),
                'missing_rate': float(pipeline['data'].isna().mean().mean()),
                'schema_coverage': pipeline['semantic']['semantic_confidence_score'],
                'readiness_score': pipeline['readiness']['readiness_score'],
                'quality_score': pipeline['quality']['quality_score'],
                'ready_modules': pipeline['readiness']['available_count'],
                'structure_confidence': pipeline['structure'].confidence_score,
                'semantic_confidence': pipeline['semantic']['semantic_confidence_score'],
            }
        ]
        for uploaded in uploads or []:
            try:
                df, _ = load_uploaded_file(uploaded.name, uploaded.getvalue())
                df, comparison_temporal = augment_temporal_fields(df)
                structure2 = detect_structure(df)
                profile2 = build_field_profile(df, structure2)
                quality2 = build_quality_checks(df, structure2, profile2)
                semantic2 = infer_semantic_mapping(df, structure2)
                readiness2 = evaluate_analysis_readiness(semantic2['canonical_map'])
                rows.append({'dataset_name': uploaded.name, 'source_mode': 'Uploaded comparison' + (' (synthetic event_date)' if comparison_temporal.get('synthetic_date_created') else ''), 'rows': len(df), 'columns': len(df.columns), 'missing_values': int(df.isna().sum().sum()), 'missing_rate': float(df.isna().mean().mean()), 'schema_coverage': semantic2['semantic_confidence_score'], 'readiness_score': readiness2['readiness_score'], 'quality_score': quality2['quality_score'], 'ready_modules': readiness2['available_count'], 'structure_confidence': structure2.confidence_score, 'semantic_confidence': semantic2['semantic_confidence_score']})
            except DataLoadError:
                continue
        dashboard = build_dataset_comparison_dashboard(rows)
        if dashboard.get('available'):
            info_or_table(safe_df(dashboard.get('summary_table')), 'No comparison summary is available.')
        else:
            st.info(dashboard.get('reason', 'Add more datasets to compare them side by side.'))
