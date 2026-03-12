from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from src.ai_copilot import append_copilot_message, initialize_copilot_memory, plan_workflow_action, run_copilot_question
from src.data_loader import DEMO_DATASETS, DataLoadError, estimate_memory_mb, list_excel_sheets, load_demo_dataset, load_uploaded_file
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
from src.export_utils import (
    apply_export_policy,
    apply_role_based_redaction,
    build_audience_report_text,
    build_compliance_dashboard_csv,
    build_compliance_dashboard_payload,
    build_compliance_handoff_payload,
    build_compliance_summary_text,
    build_compliance_support_csv,
    build_executive_summary_text,
    build_readmission_summary_text,
    build_governance_review_csv,
    build_governance_review_payload,
    build_governance_review_text,
    build_policy_aware_bundle_profile,
    build_policy_note_text,
    build_report_support_csv,
    build_role_export_bundle_manifest,
    build_role_export_bundle_text,
    build_text_report,
    dataframe_to_csv_bytes,
    json_bytes,
    normalize_report_mode,
    recommended_report_mode_for_role,
)
from src.healthcare_analysis import build_cohort_summary, build_readmission_cohort_review, operational_alerts, plan_readmission_intervention, run_healthcare_analysis
from src.insights_engine import build_action_recommendations, build_automated_insight_board, build_key_insights
from src.modules.audit import log_audit_event
from src.modules.privacy_security import evaluate_export_policy, get_export_policy_presets, run_privacy_security_review
from src.modules.rbac import can_access
from src.modeling_studio import build_model_comparison_studio, build_model_fairness_review, build_prediction_explainability, build_predictive_model, default_modeling_selection
from src.ops_hardening import (
    build_deployment_health_checks,
    build_deployment_support_notes,
    build_performance_diagnostics,
    build_preflight_guardrails,
)
from src.presentation_support import (
    build_audit_summary,
    build_compliance_governance_summary,
    build_executive_report_pack,
    build_landing_summary,
    build_printable_reports,
    build_run_history_entry,
    build_stakeholder_export_bundle,
    update_run_history,
)
from src.portfolio_support import (
    build_app_metadata,
    build_dataset_onboarding_summary,
    build_demo_mode_content,
    build_documentation_support,
    build_screenshot_support,
)
from src.decision_support import (
    build_executive_summary,
    build_intervention_recommendations,
    build_kpi_benchmarking_layer,
    build_prioritized_insights,
    build_scenario_simulation_studio,
)
from src.profiler import analysis_sample_info, build_dataset_overview, build_field_profile, build_numeric_summary, build_quality_checks
from src.readiness_engine import evaluate_analysis_readiness
from src.remediation_engine import apply_remediation_augmentations
from src.schema_detection import detect_structure
from src.semantic_mapper import build_data_dictionary, build_data_remediation_assistant, build_dataset_improvement_plan, infer_semantic_mapping
from src.standards_validator import apply_standards_mapping_overrides, build_standards_override_catalog, build_terminology_override_catalog, validate_healthcare_standards
from src.temporal_detection import augment_temporal_fields
from src.ui_components import apply_theme, metric_row, plot_bar, plot_correlation, plot_missingness, plot_numeric_box, plot_numeric_distribution, plot_time_trend, plot_top_categories

APP_TITLE = 'Smart Dataset Analyzer'
TAB_LABELS = [
    'Data Intake',
    'Dataset Profile · Overview',
    'Dataset Profile · Column Detection',
    'Dataset Profile · Field Profiling',
    'Data Quality · Quality Review',
    'Data Quality · Analysis Readiness',
    'Healthcare Analytics · Healthcare Intelligence',
    'Healthcare Analytics · Trend Analysis',
    'Healthcare Analytics · Cohort Analysis',
    'Insights & Export · Key Insights',
    'Insights & Export · Export Center',
]
ROLE_OPTIONS = ['Admin', 'Analyst', 'Researcher', 'Viewer']
REPORT_MODES = ['Analyst Report', 'Operational Report', 'Executive Summary', 'Clinical Report']


def init_state() -> None:
    defaults = {
        'analysis_log': [],
        'saved_snapshots': {},
        'workflow_packs': {},
        'analysis_template': 'General Review',
        'report_mode': 'Executive Summary',
        'export_policy_name': 'Internal Review',
        'active_role': 'Analyst',
        'run_history': [],
        'demo_synthetic_helper_mode': 'Auto',
        'demo_bmi_remediation_mode': 'median',
        'demo_synthetic_cost_mode': 'Auto',
        'demo_synthetic_readmission_mode': 'Auto',
        'demo_executive_summary_verbosity': 'Concise',
        'demo_scenario_simulation_mode': 'Basic',
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def log_event(event_type: str, details: str, user_interaction: str = 'User action', analysis_step: str = 'Session activity') -> None:
    st.session_state['analysis_log'] = log_audit_event(
        st.session_state.get('analysis_log', []),
        event_type,
        details,
        user_interaction=user_interaction,
        analysis_step=analysis_step,
    )


def safe_df(table: Any, columns: list[str] | None = None) -> pd.DataFrame:
    return table if isinstance(table, pd.DataFrame) else pd.DataFrame(columns=columns or [])


def info_or_table(table: pd.DataFrame, message: str) -> None:
    if table.empty:
        st.info(message)
    else:
        st.dataframe(table, width='stretch')


def info_or_chart(fig, message: str) -> None:
    if fig is None:
        st.info(message)
    else:
        st.plotly_chart(fig, width='stretch')


def render_safely(section_name: str, render_fn, *args, **kwargs) -> None:
    try:
        render_fn(*args, **kwargs)
    except Exception as error:
        st.error(f'{section_name} could not be rendered for the current dataset.')
        st.info('The rest of the platform is still available. Try a smaller extract, review the data quality checks, or revisit the standards/privacy guidance before re-running this section.')
        st.caption(f'Technical summary: {type(error).__name__}: {error}')


def fmt(value: Any, kind: str = 'int') -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return 'Not available'
    if kind == 'pct':
        return f'{float(value):.1%}'
    if kind == 'score':
        return f'{float(value) * 100:.0f}/100'
    if kind == 'float':
        return f'{float(value):,.2f}'
    if kind == 'money':
        return f'${float(value):,.2f}'
    return f'{int(value):,}' if isinstance(value, (int, float)) else str(value)


def active_controls() -> dict[str, object]:
    keys = ['analysis_template', 'report_mode', 'export_policy_name', 'active_role', 'workflow_action_prompt', 'demo_dataset_name']
    return {key: st.session_state.get(key) for key in keys if key in st.session_state}


def load_primary_dataset(source_mode: str) -> tuple[pd.DataFrame | None, dict[str, str], str, dict[str, str]]:
    original_lookup: dict[str, str] = {}
    dataset_name = ''
    source_meta = {'source_mode': source_mode, 'description': '', 'best_for': '', 'file_size_mb': 0.0}
    if source_mode == 'Built-in example dataset':
        dataset_name = st.sidebar.selectbox('Example dataset', list(DEMO_DATASETS.keys()), key='demo_dataset_name')
        data, original_lookup = load_demo_dataset(dataset_name)
        demo = DEMO_DATASETS[dataset_name]
        demo_path = demo['path']
        file_size_mb = (Path(demo_path).stat().st_size / (1024 ** 2)) if Path(demo_path).exists() else 0.0
        source_meta = {'source_mode': 'Demo dataset', 'description': demo['description'], 'best_for': demo['best_for'], 'file_size_mb': file_size_mb}
        return data, original_lookup, dataset_name, source_meta

    uploaded = st.sidebar.file_uploader('Upload a dataset', type=['csv', 'xlsx', 'xlsm', 'xls'], key='primary_upload')
    if uploaded is None:
        st.info('Upload a CSV or Excel file, or switch to a built-in example dataset to start exploring the platform.')
        return None, original_lookup, dataset_name, source_meta

    file_bytes = uploaded.getvalue()
    sheet_name = None
    suffix = uploaded.name.lower().rsplit('.', 1)[-1]
    if suffix in {'xlsx', 'xlsm', 'xls'}:
        try:
            sheets = list_excel_sheets(file_bytes, suffix=f'.{suffix}')
            if len(sheets) > 1:
                sheet_name = st.sidebar.selectbox('Excel sheet', sheets, key='primary_sheet_name')
        except DataLoadError as error:
            st.error(str(error))
            return None, original_lookup, dataset_name, source_meta
    try:
        data, original_lookup = load_uploaded_file(uploaded.name, file_bytes, sheet_name=sheet_name)
    except DataLoadError as error:
        st.error(str(error))
        return None, original_lookup, uploaded.name, source_meta
    source_meta = {
        'source_mode': source_mode,
        'description': 'Uploaded source dataset provided by the current user session.',
        'best_for': 'Schema-flexible profiling, readiness review, and conditional healthcare analysis.',
        'file_size_mb': len(file_bytes) / (1024 ** 2),
    }
    return data, original_lookup, uploaded.name, source_meta


def run_pipeline(
    data: pd.DataFrame,
    dataset_name: str,
    source_meta: dict[str, str],
    demo_config: dict[str, str] | None = None,
    progress_callback=None,
) -> dict[str, Any]:
    demo_config = demo_config or {}
    analysis_data, temporal_context = augment_temporal_fields(data)
    analysis_data, remediation_context = apply_remediation_augmentations(
        analysis_data,
        bmi_mode=str(demo_config.get('bmi_remediation_mode', 'median')).lower(),
        helper_mode=str(demo_config.get('synthetic_helper_mode', 'Auto')),
        synthetic_cost_mode=str(demo_config.get('synthetic_cost_mode', 'Auto')),
        synthetic_readmission_mode=str(demo_config.get('synthetic_readmission_mode', 'Auto')),
    )
    source_meta = dict(source_meta)
    if temporal_context.get('synthetic_date_created'):
        source_meta['temporal_note'] = str(temporal_context.get('note', 'Synthetic event_date generated for temporal analysis.'))
        source_meta['best_for'] = f"{source_meta.get('best_for', 'General analysis')} Temporal trend modules are enabled with a synthetic event_date derived from existing date parts."
    if progress_callback:
        progress_callback(0.10, 'Detecting structure and preparing profile inputs...')
    structure = detect_structure(analysis_data)
    if progress_callback:
        progress_callback(0.25, 'Building field profiling and quality diagnostics...')
    field_profile = build_field_profile(analysis_data, structure)
    quality = build_quality_checks(analysis_data, structure, field_profile)
    if progress_callback:
        progress_callback(0.40, 'Summarizing dataset overview and semantic mappings...')
    overview = build_dataset_overview(analysis_data, estimate_memory_mb(analysis_data))
    semantic = infer_semantic_mapping(analysis_data, structure)
    helper_fields = set(
        remediation_context.get('helper_fields', pd.DataFrame())
        .loc[lambda df: df.get('helper_type', pd.Series(dtype=str)).isin(['synthetic', 'derived']), 'helper_field']
        .astype(str)
        .tolist()
    ) if isinstance(remediation_context.get('helper_fields'), pd.DataFrame) else set()
    if progress_callback:
        progress_callback(0.55, 'Evaluating readiness and healthcare analytics...')
    readiness = evaluate_analysis_readiness(semantic['canonical_map'], synthetic_fields=helper_fields)
    healthcare = run_healthcare_analysis(analysis_data, semantic['canonical_map'], synthetic_fields=helper_fields)
    if progress_callback:
        progress_callback(0.72, 'Running standards, privacy, and governance checks...')
    standards = validate_healthcare_standards(analysis_data, structure, semantic)
    privacy_review = run_privacy_security_review(analysis_data)
    data_dictionary = build_data_dictionary(structure, semantic)
    remediation = build_data_remediation_assistant(structure, semantic, readiness)
    improvement_plan = build_dataset_improvement_plan(structure, semantic, readiness)
    if progress_callback:
        progress_callback(0.88, 'Preparing rule engine, insights, and action guidance...')
    rule_engine = build_quality_rule_engine(analysis_data, semantic['canonical_map'])
    insights = build_key_insights(overview, field_profile, quality, readiness, semantic, healthcare, structure)
    action_recommendations = build_action_recommendations(quality, readiness, semantic, healthcare)
    insight_board = build_automated_insight_board(overview, readiness, healthcare, insights, action_recommendations)
    intervention_recommendations = build_intervention_recommendations(
        healthcare,
        quality,
        readiness,
        remediation_context,
        model_comparison=None,
    )
    executive_summary = build_executive_summary(
        dataset_name,
        overview,
        readiness,
        healthcare,
        action_recommendations,
        intervention_recommendations,
        remediation_context,
        verbosity=str(demo_config.get('executive_summary_verbosity', 'Concise')).lower(),
    )
    kpi_benchmarking = build_kpi_benchmarking_layer(healthcare, quality, remediation_context)
    scenario_studio = build_scenario_simulation_studio(
        healthcare,
        quality,
        remediation_context,
        scenario_mode=str(demo_config.get('scenario_simulation_mode', 'Basic')).lower(),
    )
    prioritized_insights = build_prioritized_insights(
        overview,
        quality,
        readiness,
        healthcare,
        action_recommendations,
        intervention_recommendations,
        model_comparison=None,
    )
    lineage = build_data_lineage_view(dataset_name, source_meta, semantic, readiness, active_controls())
    if not remediation_context.get('helper_fields', pd.DataFrame()).empty:
        helper_rows = remediation_context['helper_fields'].rename(columns={'helper_field': 'source_column', 'helper_type': 'derived_role', 'note': 'business_meaning'})
        derived_fields = safe_df(lineage.get('derived_fields_table'))
        lineage['derived_fields_table'] = pd.concat([derived_fields, helper_rows], ignore_index=True, sort=False)
        steps = list(lineage.get('transformation_steps', []))
        for key in ['bmi_remediation', 'synthetic_cost', 'synthetic_clinical', 'synthetic_readmission']:
            note = remediation_context.get(key, {}).get('lineage_note')
            if note:
                steps.append(note)
        lineage['transformation_steps'] = steps
    sample_info = analysis_sample_info(analysis_data)
    compliance_governance_summary = build_compliance_governance_summary(
        standards,
        privacy_review,
        lineage,
        remediation_context,
        readiness,
    )
    executive_report_pack = build_executive_report_pack(
        dataset_name,
        overview,
        quality,
        readiness,
        healthcare,
        executive_summary,
        action_recommendations,
        intervention_recommendations,
        kpi_benchmarking,
        scenario_studio,
        prioritized_insights,
        remediation_context,
        demo_config,
    )
    printable_reports = build_printable_reports(executive_report_pack, compliance_governance_summary)
    stakeholder_bundle = build_stakeholder_export_bundle(
        executive_report_pack,
        kpi_benchmarking,
        intervention_recommendations,
        healthcare.get('explainability_fairness', {}),
        healthcare.get('readmission', {}),
        quality,
        compliance_governance_summary,
    )
    dataset_onboarding = build_dataset_onboarding_summary(dataset_name, source_meta, {'readiness': readiness, 'remediation': remediation, 'remediation_context': remediation_context})
    demo_mode_content = build_demo_mode_content(dataset_name, source_meta, {'readiness': readiness, 'remediation_context': remediation_context}, demo_config)
    documentation_support = build_documentation_support(dataset_name, {'readiness': readiness, 'healthcare': healthcare, 'remediation_context': remediation_context})
    screenshot_support = build_screenshot_support(dataset_name, {'readiness': readiness, 'remediation_context': remediation_context})
    app_metadata = build_app_metadata({'readiness': readiness, 'remediation_context': remediation_context})
    if progress_callback:
        progress_callback(1.0, 'Analysis preparation complete.')
    return {
        'data': analysis_data,
        'overview': overview,
        'structure': structure,
        'field_profile': field_profile,
        'quality': quality,
        'semantic': semantic,
        'readiness': readiness,
        'healthcare': healthcare,
        'standards': standards,
        'privacy_review': privacy_review,
        'data_dictionary': data_dictionary,
        'remediation': remediation,
        'improvement_plan': improvement_plan,
        'rule_engine': rule_engine,
        'insights': insights,
        'action_recommendations': action_recommendations,
        'insight_board': insight_board,
        'intervention_recommendations': intervention_recommendations,
        'executive_summary': executive_summary,
        'kpi_benchmarking': kpi_benchmarking,
        'scenario_studio': scenario_studio,
        'prioritized_insights': prioritized_insights,
        'compliance_governance_summary': compliance_governance_summary,
        'executive_report_pack': executive_report_pack,
        'printable_reports': printable_reports,
        'stakeholder_export_bundle': stakeholder_bundle,
        'dataset_onboarding': dataset_onboarding,
        'demo_mode_content': demo_mode_content,
        'documentation_support': documentation_support,
        'screenshot_support': screenshot_support,
        'app_metadata': app_metadata,
        'lineage': lineage,
        'sample_info': sample_info,
        'temporal_context': temporal_context,
        'remediation_context': remediation_context,
        'demo_config': demo_config,
    }


def get_demo_config() -> dict[str, str]:
    return {
        'synthetic_helper_mode': st.session_state.get('demo_synthetic_helper_mode', 'Auto'),
        'bmi_remediation_mode': st.session_state.get('demo_bmi_remediation_mode', 'median'),
        'synthetic_cost_mode': st.session_state.get('demo_synthetic_cost_mode', 'Auto'),
        'synthetic_readmission_mode': st.session_state.get('demo_synthetic_readmission_mode', 'Auto'),
        'executive_summary_verbosity': st.session_state.get('demo_executive_summary_verbosity', 'Concise'),
        'scenario_simulation_mode': st.session_state.get('demo_scenario_simulation_mode', 'Basic'),
    }


def compliance_snapshot(pipeline: dict[str, Any]) -> pd.DataFrame:
    standards = pipeline['standards']
    privacy = pipeline['privacy_review']
    lineage = pipeline['lineage']
    return pd.DataFrame([
        {'dimension': 'Standards readiness', 'status': standards.get('badge_text', 'Not assessed')},
        {'dimension': 'Privacy posture', 'status': privacy.get('hipaa', {}).get('risk_level', 'Low')},
        {'dimension': 'Governance depth', 'status': f"{len(lineage.get('transformation_steps', []))} tracked steps"},
        {'dimension': 'Export context', 'status': f"{st.session_state.get('active_role', 'Analyst')} ? {st.session_state.get('export_policy_name', 'Internal Review')}"},
    ])


def _normalize_choice(value: Any) -> str:
    if value in (None, '', 'Auto-detect'):
        return ''
    return str(value)


def _override_key(prefix: str, target: str) -> str:
    safe = ''.join(char.lower() if char.isalnum() else '_' for char in str(target))
    return f'{prefix}_{safe}'


def _apply_mapping_suggestions(rows: pd.DataFrame, key_column: str, source_column: str, prefix: str, target_filter=None) -> int:
    if rows.empty:
        return 0
    applied = 0
    for _, row in rows.iterrows():
        target = str(row.get(key_column, ''))
        source = str(row.get(source_column, ''))
        if not target or not source:
            continue
        if target_filter and not target_filter(target):
            continue
        st.session_state[_override_key(prefix, target)] = source
        applied += 1
    return applied


def _render_standards_mapping_overrides(pipeline: dict[str, Any], key_prefix: str) -> None:
    standards = pipeline['standards']
    if not standards.get('available'):
        return
    columns = sorted(str(column) for column in pipeline['data'].columns)
    options = ['Auto-detect'] + columns
    override_catalog = build_standards_override_catalog(standards)
    terminology_catalog = build_terminology_override_catalog(standards)

    with st.expander('Standards mapping overrides'):
        st.caption('Confirm or override suggested CDISC, FHIR, and terminology mappings when you want a more directed readiness review. Auto-detected values remain the default until you change them.')
        if st.button('Reset standards mapping overrides', key=f'{key_prefix}_reset_standards_overrides'):
            for key in list(st.session_state.keys()):
                if key.startswith('standards_override_') or key.startswith('terminology_override_'):
                    del st.session_state[key]
            st.rerun()

        if not override_catalog.empty:
            st.markdown('#### Standards mapping review')
            for group_name, group in override_catalog.groupby('mapping_group'):
                st.write(f'**{group_name}**')
                for row in group.itertuples(index=False):
                    widget_key = _override_key(f'{key_prefix}_standards_override', row.target_field)
                    default_choice = row.default_source if row.default_source in columns else 'Auto-detect'
                    if widget_key not in st.session_state:
                        st.session_state[widget_key] = default_choice
                    st.selectbox(
                        row.target_field,
                        options,
                        key=widget_key,
                        help=row.help_text,
                    )

        if not terminology_catalog.empty:
            st.markdown('#### Terminology confirmation')
            for row in terminology_catalog.itertuples(index=False):
                widget_key = _override_key(f'{key_prefix}_terminology_override', row.terminology_type)
                default_choice = row.default_source if row.default_source in columns else 'Auto-detect'
                if widget_key not in st.session_state:
                    st.session_state[widget_key] = default_choice
                st.selectbox(
                    row.terminology_type,
                    options,
                    key=widget_key,
                    help=row.help_text,
                )

    overrides = {
        row.target_field: _normalize_choice(st.session_state.get(_override_key(f'{key_prefix}_standards_override', row.target_field)))
        for row in override_catalog.itertuples(index=False)
    }
    terminology_overrides = {
        row.terminology_type: _normalize_choice(st.session_state.get(_override_key(f'{key_prefix}_terminology_override', row.terminology_type)))
        for row in terminology_catalog.itertuples(index=False)
    }
    pipeline['standards'] = apply_standards_mapping_overrides(standards, overrides, terminology_overrides)


def _build_standards_prefill_actions(pipeline: dict[str, Any]) -> list[dict[str, str]]:
    standards = pipeline['standards']
    if not standards.get('available'):
        return []
    actions: list[dict[str, str]] = []
    cdisc = standards.get('cdisc_report', {})
    interop = standards.get('interoperability_report', {})
    cdisc_map = safe_df(cdisc.get('mapping_suggestions'))
    interop_map = safe_df(interop.get('mapping_suggestions'))
    terminology = safe_df(interop.get('terminology_validation'))
    if not cdisc_map.empty:
        actions.append({'label': 'Prefill CDISC mappings', 'action': 'apply_detected_cdisc_mappings'})
        core_targets = {'STUDYID', 'USUBJID', 'SUBJID', 'VISIT', 'VISITNUM', 'DOMAIN'}
        if set(cdisc_map.get('cdisc_field', pd.Series(dtype=str)).astype(str)).intersection(core_targets):
            actions.append({'label': 'Prefill CDISC core fields', 'action': 'apply_detected_cdisc_core_mappings'})
        actions.append({'label': 'Prepare CDISC readiness review', 'action': 'prepare_cdisc_readiness_review'})
    if not interop_map.empty:
        actions.append({'label': 'Prefill interoperability mappings', 'action': 'apply_detected_interop_mappings'})
        fhir_prefixes = ('Patient.', 'Encounter.', 'Condition.', 'Procedure.', 'Observation.', 'Medication')
        targets = interop_map.get('reference_model_target', pd.Series(dtype=str)).astype(str).tolist()
        if any(target.startswith(fhir_prefixes) for target in targets):
            actions.append({'label': 'Prefill FHIR mappings', 'action': 'apply_detected_fhir_mappings'})
    if not terminology.empty:
        actions.append({'label': 'Prefill terminology confirmations', 'action': 'apply_detected_terminology'})
    return actions


def _run_standards_prefill_action(pipeline: dict[str, Any], action_name: str) -> str:
    standards = pipeline['standards']
    cdisc_map = safe_df(standards.get('cdisc_report', {}).get('mapping_suggestions'))
    interop_map = safe_df(standards.get('interoperability_report', {}).get('mapping_suggestions'))
    terminology = safe_df(standards.get('interoperability_report', {}).get('terminology_validation'))

    if action_name == 'apply_detected_cdisc_mappings':
        applied = _apply_mapping_suggestions(cdisc_map, 'cdisc_field', 'suggested_source_column', 'standards_override')
        return f'Prefilled {applied} CDISC mapping suggestions for review.'
    if action_name == 'apply_detected_cdisc_core_mappings':
        core_targets = {'STUDYID', 'USUBJID', 'SUBJID', 'VISIT', 'VISITNUM', 'DOMAIN'}
        applied = _apply_mapping_suggestions(cdisc_map, 'cdisc_field', 'suggested_source_column', 'standards_override', lambda target: target in core_targets)
        return f'Prefilled {applied} core CDISC fields for review.'
    if action_name == 'prepare_cdisc_readiness_review':
        st.session_state['analysis_template'] = 'General Review'
        st.session_state['report_mode'] = 'Analyst Report'
        st.session_state['export_policy_name'] = 'Internal Review'
        return 'Prepared the workspace for a CDISC-focused readiness review.'
    if action_name == 'apply_detected_interop_mappings':
        applied = _apply_mapping_suggestions(interop_map, 'reference_model_target', 'suggested_source_column', 'standards_override')
        return f'Prefilled {applied} interoperability mappings for review.'
    if action_name == 'apply_detected_fhir_mappings':
        prefixes = ('Patient.', 'Encounter.', 'Condition.', 'Procedure.', 'Observation.', 'Medication')
        applied = _apply_mapping_suggestions(interop_map, 'reference_model_target', 'suggested_source_column', 'standards_override', lambda target: str(target).startswith(prefixes))
        return f'Prefilled {applied} FHIR-style mappings for review.'
    if action_name == 'apply_detected_terminology':
        applied = 0
        if not terminology.empty:
            for row in terminology.itertuples(index=False):
                term_type = str(getattr(row, 'terminology_type', ''))
                column_name = str(getattr(row, 'column_name', ''))
                if term_type and column_name:
                    st.session_state[_override_key('terminology_override', term_type)] = column_name
                    applied += 1
        return f'Prefilled {applied} terminology confirmations for review.'
    return 'No standards prefill action was applied.'


def _build_ehr_onboarding_actions(pipeline: dict[str, Any]) -> list[dict[str, str]]:
    interop = pipeline['standards'].get('interoperability_report', {})
    ehr_patterns = safe_df(interop.get('ehr_export_patterns'))
    hl7_patterns = safe_df(interop.get('hl7_patterns'))
    actions: list[dict[str, str]] = []
    if not ehr_patterns.empty and 'export_pattern' in ehr_patterns.columns:
        detected = set(ehr_patterns['export_pattern'].astype(str))
        if 'Epic-like export' in detected:
            actions.append({'label': 'Prepare Epic-style onboarding', 'action': 'prepare_epic_onboarding'})
        if 'Cerner-like export' in detected:
            actions.append({'label': 'Prepare Cerner-style onboarding', 'action': 'prepare_cerner_onboarding'})
    if not hl7_patterns.empty:
        actions.append({'label': 'Prepare HL7-style onboarding', 'action': 'prepare_hl7_onboarding'})
    return actions


def _run_ehr_onboarding_action(action_name: str) -> str:
    st.session_state['analysis_template'] = 'General Review'
    st.session_state['export_policy_name'] = 'Internal Review'
    if action_name == 'prepare_epic_onboarding':
        st.session_state['report_mode'] = 'Analyst Report'
        return 'Prepared an Epic-style onboarding workflow with analyst-oriented review settings.'
    if action_name == 'prepare_cerner_onboarding':
        st.session_state['report_mode'] = 'Operational Report'
        return 'Prepared a Cerner-style onboarding workflow with operational review settings.'
    if action_name == 'prepare_hl7_onboarding':
        st.session_state['report_mode'] = 'Analyst Report'
        return 'Prepared an HL7-style onboarding workflow for structure and standards review.'
    return 'No onboarding preset was applied.'


def _build_ehr_onboarding_explanation(pipeline: dict[str, Any]) -> pd.DataFrame:
    interop = pipeline['standards'].get('interoperability_report', {})
    rows: list[dict[str, object]] = []
    ehr_patterns = safe_df(interop.get('ehr_export_patterns'))
    if not ehr_patterns.empty:
        for row in ehr_patterns.itertuples(index=False):
            rows.append({
                'detected_signal': getattr(row, 'export_pattern', 'EHR export pattern'),
                'status': getattr(row, 'status', 'Detected'),
                'why_it_appeared': getattr(row, 'matched_signals', 'Column naming pattern and export structure heuristics'),
            })
    hl7_patterns = safe_df(interop.get('hl7_patterns'))
    if not hl7_patterns.empty:
        for row in hl7_patterns.itertuples(index=False):
            rows.append({
                'detected_signal': f"HL7 segment {getattr(row, 'segment', '')}",
                'status': getattr(row, 'status', 'Detected'),
                'why_it_appeared': getattr(row, 'matched_signals', 'Segment-like naming and field pattern signals'),
            })
    return pd.DataFrame(rows)


def _build_compliance_quick_actions(pipeline: dict[str, Any]) -> list[dict[str, str]]:
    actions: list[dict[str, str]] = []
    standards = pipeline['standards']
    privacy = pipeline['privacy_review']
    if standards.get('available') and standards.get('cdisc_report', {}).get('available'):
        actions.append({'label': 'Prepare Clinical Outcomes workflow', 'action': 'prepare_clinical_outcomes'})
    if standards.get('interoperability_report', {}).get('available'):
        actions.append({'label': 'Prepare Analyst handoff', 'action': 'prepare_analyst_handoff'})
    hipaa_risk = privacy.get('hipaa', {}).get('risk_level', 'Low')
    if hipaa_risk in {'Moderate', 'High'}:
        actions.append({'label': 'Apply HIPAA-style export posture', 'action': 'apply_hipaa_export'})
        actions.append({'label': 'Apply research-safe export posture', 'action': 'apply_research_export'})
    return actions


def _run_compliance_quick_action(action_name: str) -> str:
    if action_name == 'prepare_clinical_outcomes':
        st.session_state['analysis_template'] = 'Clinical Outcomes Analysis'
        st.session_state['report_mode'] = 'Clinical Report'
        return 'Prepared the workspace for a clinical outcomes review.'
    if action_name == 'prepare_analyst_handoff':
        st.session_state['analysis_template'] = 'General Review'
        st.session_state['report_mode'] = 'Analyst Report'
        return 'Prepared the workspace for an analyst-oriented handoff review.'
    if action_name == 'apply_hipaa_export':
        st.session_state['export_policy_name'] = 'HIPAA-style Limited Dataset'
        return 'Applied the HIPAA-style limited dataset export posture.'
    if action_name == 'apply_research_export':
        st.session_state['export_policy_name'] = 'Research-safe Extract'
        return 'Applied the research-safe export posture.'
    return 'No compliance action was applied.'


def _build_next_compliance_action(pipeline: dict[str, Any]) -> dict[str, str] | None:
    standards = pipeline['standards']
    privacy = pipeline['privacy_review']
    if privacy.get('hipaa', {}).get('risk_level') in {'Moderate', 'High'}:
        return {
            'title': 'Reduce privacy exposure before sharing',
            'detail': 'Sensitive columns were detected. Use a stricter export policy or review the de-identification preview before handing the dataset to a wider audience.',
        }
    if standards.get('available') and standards.get('combined_readiness_score', 0) < 45:
        return {
            'title': 'Confirm standards mappings',
            'detail': 'Use the standards mapping overrides to confirm trial or interoperability fields so downstream healthcare reviews can rely on clearer structure.',
        }
    if not standards.get('available'):
        return {
            'title': 'Start with schema and remediation review',
            'detail': 'The dataset does not yet resemble a strong standards-aligned structure. Use the remediation assistant to fill the highest-impact gaps first.',
        }
    return {
        'title': 'Prepare a governed handoff bundle',
        'detail': 'The dataset is in a reasonable position for standards and privacy review. Generate a compliance handoff or governance packet before sharing results.',
    }


def render_standards(pipeline: dict[str, Any], key_prefix: str = 'standards') -> None:
    standards = pipeline['standards']
    st.markdown('### Standards Compliance')
    if not standards.get('available'):
        st.info(standards.get('reason', 'Standards readiness is not available for the current dataset.'))
        info_or_table(safe_df(standards.get('summary_table')), 'No standards summary is available yet.')
        return
    _render_standards_mapping_overrides(pipeline, key_prefix)
    standards = pipeline['standards']
    metric_row([
        ('Combined Readiness', fmt(standards.get('combined_readiness_score', 0.0), 'float')),
        ('Detected Standard', standards.get('detected_standard', 'Not detected')),
        ('Confidence', standards.get('confidence_label', 'Low')),
        ('Missing Required Fields', fmt(standards.get('missing_required_fields', 0))),
    ])
    st.caption(standards.get('note', 'Readiness-oriented validator only; not a formal certification engine.'))
    info_or_table(safe_df(standards.get('summary_table')), 'No standards summary is available yet.')
    summary_table = _build_standards_override_summary(standards)
    if not summary_table.empty:
        st.markdown('#### Mapping confirmation snapshot')
        info_or_table(summary_table, 'No mapping confirmation snapshot is available yet.')
    effective = safe_df(standards.get('effective_mappings'))
    if not effective.empty:
        st.markdown('#### Effective standards mappings')
        info_or_table(effective, 'No effective mappings are available yet.')
    cols = st.columns(2)
    cdisc = standards.get('cdisc_report', {})
    interop = standards.get('interoperability_report', {})
    with cols[0]:
        st.markdown('#### CDISC validation')
        if cdisc.get('available'):
            info_or_table(safe_df(cdisc.get('validation_report')), 'No CDISC validation details are available.')
            info_or_table(safe_df(cdisc.get('effective_mappings', cdisc.get('mapping_suggestions'))), 'No CDISC mapping suggestions were detected.')
            info_or_table(safe_df(cdisc.get('domain_templates')), 'No SDTM domain-template guidance is available yet.')
        else:
            st.info('The current dataset does not look trial-oriented enough for a useful CDISC review yet.')
    with cols[1]:
        st.markdown('#### FHIR / HL7 interoperability')
        if interop.get('available'):
            info_or_table(safe_df(interop.get('validation_report')), 'No interoperability validation details are available.')
            info_or_table(safe_df(interop.get('effective_mappings', interop.get('mapping_suggestions'))), 'No interoperability mapping suggestions were detected.')
            info_or_table(safe_df(interop.get('fhir_resources')), 'No FHIR resource signals were detected.')
            info_or_table(safe_df(interop.get('hl7_patterns')), 'No HL7-like patterns were detected.')
            info_or_table(safe_df(interop.get('ehr_export_patterns')), 'No likely Epic-like or Cerner-like export pattern was detected.')
            info_or_table(safe_df(interop.get('effective_terminology', interop.get('terminology_validation', pd.DataFrame()))), 'No terminology-like fields were detected.')
        else:
            st.info('The current dataset does not yet look interoperable enough for a useful FHIR or HL7 readiness review.')


def render_privacy(pipeline: dict[str, Any]) -> None:
    role = st.session_state.get('active_role', 'Analyst')
    st.markdown('### Privacy & Security Review')
    if not can_access(role, 'sensitive_review'):
        st.info(f'The active {role} role can view standards guidance, but privacy review details are limited for this role.')
        return
    privacy = pipeline['privacy_review']
    sensitive = safe_df(privacy.get('sensitive_fields'))
    hipaa = privacy.get('hipaa', {})
    metric_row([
        ('Sensitive Columns', fmt(len(sensitive))),
        ('HIPAA-style Risk', hipaa.get('risk_level', 'Low')),
        ('Direct Identifiers', fmt(hipaa.get('direct_identifier_count', 0))),
        ('Safe Harbor', 'Ready' if hipaa.get('safe_harbor_ready') else 'Needs review'),
    ])
    info_or_table(sensitive, 'No likely PHI or PII columns were detected in the current dataset sample.')
    info_or_table(safe_df(privacy.get('deidentification_preview')), 'No de-identification preview is available.')
    info_or_table(safe_df(privacy.get('gdpr_impact')), 'No GDPR-style impact notes are available.')
    info_or_table(safe_df(privacy.get('privacy_rule_pack')), 'No privacy rule pack details are available.')


def render_data_intake(pipeline: dict[str, Any], dataset_name: str, source_meta: dict[str, str]) -> None:
    st.subheader('Data Intake')
    structure = pipeline['structure']
    temporal_context = pipeline.get('temporal_context', {})
    st.caption('Start with onboarding guidance, lineage, workflow packs, snapshots, linked analysis, and comparison tools.')
    st.markdown('### Data Ingestion Wizard')
    st.write('**1. Source selection**')
    st.write('Choose a built-in demo or upload a CSV/Excel file. The platform standardizes columns internally and keeps original names available for review.')
    st.write('**2. Schema review**')
    st.write(f"Detected {len(structure.detection_table)} fields with an average structure confidence of {structure.confidence_score:.1%}.")
    st.write('**3. Readiness summary**')
    st.write(f"{pipeline['readiness']['available_count']} modules are fully ready and {pipeline['readiness']['partial_count']} are partially ready.")
    st.write('**4. Recommended next step**')
    if not pipeline['remediation'].empty:
        top = pipeline['remediation'].iloc[0]
        st.write(f"Address the top blocker first: {top['issue']} Recommended fix: {top['recommended_fix']}")
    else:
        st.write('The dataset is already in a strong enough position for guided profiling, readiness review, and stakeholder export preparation.')
    if temporal_context.get('synthetic_date_created'):
        st.info(str(temporal_context.get('note', 'A synthetic event_date was generated to support temporal analysis.')))

    st.markdown('### Guided Demo Mode')
    demo_mode = pipeline.get('demo_mode_content', {})
    if demo_mode:
        st.caption(str(demo_mode.get('intro', 'Use the guided flow below to walk through the strongest parts of the platform.')))
        for step in demo_mode.get('recommended_flow', []):
            st.write(f'- {step}')
        with st.expander('Demo highlights checklist'):
            for item in demo_mode.get('demo_highlights_checklist', []):
                st.write(f'- {item}')
            for item in demo_mode.get('insights_to_look_for', []):
                st.write(f'- Insight focus: {item}')
            st.caption(str(demo_mode.get('synthetic_support_note', '')))

    st.markdown('### Data Lineage')
    info_or_table(safe_df(pipeline['lineage'].get('source_table')), 'No lineage source summary is available yet.')
    info_or_table(safe_df(pipeline['lineage'].get('derived_fields_table')), 'No derived field lineage is available yet.')
    with st.expander('Transformation steps'):
        for line in pipeline['lineage'].get('transformation_steps', []):
            st.write(f'- {line}')

    st.markdown('### Analysis Log')
    info_or_table(build_audit_log_view(st.session_state.get('analysis_log', [])), 'No analysis actions have been logged yet in this session.')
    st.markdown('### Run History Summary')
    audit_bundle = pipeline.get('audit_summary_bundle', {})
    info_or_table(safe_df(audit_bundle.get('audit_summary')), 'No run history summary is available yet.')
    if audit_bundle.get('audit_summary_text'):
        st.caption(str(audit_bundle['audit_summary_text']))

    st.markdown('### Production Hardening')
    preflight = pipeline.get('preflight', {})
    for warning in preflight.get('warnings', []):
        st.warning(warning)
    info_or_table(safe_df(preflight.get('checks_table')), 'No preflight guardrail details are available yet.')
    info_or_table(safe_df(pipeline.get('deployment_health_checks')), 'No deployment health checks are available yet.')
    info_or_table(safe_df(pipeline.get('performance_diagnostics')), 'No performance diagnostics are available yet.')
    with st.expander('Deployment support notes'):
        info_or_table(build_deployment_support_notes(), 'No deployment support notes are available yet.')

    st.markdown('### Demo Dataset Onboarding')
    onboarding = pipeline.get('dataset_onboarding', {})
    onboarding_summary = onboarding.get('dataset_onboarding_summary', {})
    if onboarding_summary:
        st.write(str(onboarding_summary.get('suitability', 'This dataset is ready for schema-flexible analytics and walkthrough review.')))
        st.caption(str(onboarding_summary.get('synthetic_note', '')))
    info_or_table(safe_df(onboarding.get('module_unlock_guide')), 'No module unlock guide is available yet.')
    info_or_table(safe_df(onboarding.get('data_upgrade_suggestions')), 'No data upgrade suggestions are available yet.')

    st.markdown('### Save Analysis Snapshots')
    snapshot_name = st.text_input('Snapshot name', key='snapshot_name')
    cols = st.columns(2)
    if cols[0].button('Save Analysis Snapshot', key='save_snapshot') and snapshot_name.strip():
        st.session_state['saved_snapshots'][snapshot_name.strip()] = {'dataset_name': dataset_name, 'controls': active_controls()}
        log_event('Snapshot Saved', f"Saved analysis snapshot '{snapshot_name.strip()}'.", 'Snapshot management', 'Data intake')
        st.success('Analysis snapshot saved for this session.')
    snapshots = [''] + sorted(st.session_state.get('saved_snapshots', {}).keys())
    selected_snapshot = cols[1].selectbox('Reload snapshot', snapshots, key='selected_snapshot')
    if selected_snapshot and st.button('Load Snapshot', key='load_snapshot'):
        for key, value in st.session_state['saved_snapshots'][selected_snapshot].get('controls', {}).items():
            st.session_state[key] = value
        log_event('Snapshot Loaded', f"Loaded analysis snapshot '{selected_snapshot}'.", 'Snapshot management', 'Data intake')
        st.rerun()

    st.markdown('### Workflow Packs')
    workflow_name = st.text_input('Workflow pack name', key='workflow_pack_name')
    if st.button('Save Workflow Pack', key='save_workflow_pack') and workflow_name.strip():
        details = build_workflow_pack_details(active_controls(), st.session_state.get('analysis_template', 'General Review'), dataset_context={'dataset_source_mode': source_meta.get('source_mode'), 'demo_dataset_name': dataset_name if source_meta.get('source_mode') == 'Demo dataset' else None})
        st.session_state['workflow_packs'][workflow_name.strip()] = {'summary': build_workflow_pack_summary(active_controls(), st.session_state.get('analysis_template', 'General Review')), 'details': details, 'controls': active_controls()}
        log_event('Workflow Pack Saved', f"Saved workflow pack '{workflow_name.strip()}'.", 'Workflow packs', 'Data intake')
        st.success('Workflow pack saved for this session.')
    packs = [''] + sorted(st.session_state.get('workflow_packs', {}).keys())
    selected_pack = st.selectbox('Reload workflow pack', packs, key='selected_workflow_pack')
    if selected_pack:
        details = st.session_state['workflow_packs'][selected_pack].get('details', {})
        st.write(details.get('summary', ''))
        for line in details.get('highlighted_controls', []):
            st.write(f'- {line}')
        if st.button('Load Workflow Pack', key='load_workflow_pack'):
            for key, value in st.session_state['workflow_packs'][selected_pack].get('controls', {}).items():
                st.session_state[key] = value
            log_event('Workflow Pack Loaded', f"Loaded workflow pack '{selected_pack}'.", 'Workflow packs', 'Data intake')
            st.rerun()

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


def render_overview(pipeline: dict[str, Any]) -> None:
    overview = pipeline['overview']
    structure = pipeline['structure']
    sample_info = pipeline['sample_info']
    temporal_context = pipeline.get('temporal_context', {})
    remediation = pipeline.get('remediation_context', {})
    st.subheader('Dataset Overview')
    metric_row([
        ('Rows', fmt(overview['rows'])),
        ('Columns', fmt(overview['columns'])),
        ('Duplicate Rows', fmt(overview['duplicate_rows'])),
        ('Missing Values', fmt(overview['missing_values'])),
    ])
    metric_row([
        ('Structure Confidence', fmt(structure.confidence_score, 'pct')),
        ('Healthcare Readiness', fmt(pipeline['healthcare']['healthcare_readiness_score'], 'pct')),
        ('Data Quality Score', fmt(pipeline['quality']['quality_score'])),
        ('Memory (MB)', fmt(overview['memory_mb'], 'float')),
    ])
    if sample_info['sampling_applied']:
        st.info(f"Large dataset detected. Profiling uses {sample_info['profile_sample_rows']:,} rows and quality review uses {sample_info['quality_sample_rows']:,} rows while overview metrics still reflect all {sample_info['total_rows']:,} rows.")
    if temporal_context.get('synthetic_date_created'):
        st.caption(str(temporal_context.get('note', 'Synthetic temporal support is active for this dataset.')))
    bmi_summary = remediation.get('bmi_remediation', {})
    if bmi_summary.get('available') and bmi_summary.get('total_bmi_outliers', 0):
        st.caption(
            f"BMI remediation is active in {bmi_summary.get('remediation_mode', 'median')} mode for "
            f"{int(bmi_summary.get('total_bmi_outliers', 0)):,} rows ({float(bmi_summary.get('outlier_pct', 0.0)):.1%})."
        )
    if remediation.get('synthetic_cost', {}).get('available'):
        st.caption('Estimated cost is synthetic and demo-derived because no native cost field was available.')
    if remediation.get('synthetic_clinical', {}).get('available'):
        st.caption('Diagnosis labels and risk labels are derived approximations to support demo-safe clinical segmentation.')
    if remediation.get('synthetic_readmission', {}).get('available'):
        st.caption('Readmission support is synthetic and intended for workflow demonstration rather than clinical truth.')
    st.dataframe(pipeline['data'].head(20), width='stretch')
    st.markdown('### Compliance handoff snapshot')
    info_or_table(compliance_snapshot(pipeline), 'Compliance handoff details are not available yet.')
    st.markdown('### Compliance & Governance Summary')
    compliance_summary = pipeline.get('compliance_governance_summary', {})
    cards = compliance_summary.get('compliance_snapshot_cards', [])
    if cards:
        metric_row([(card['label'], card['value']) for card in cards[:4]])
    info_or_table(safe_df(compliance_summary.get('summary_table')), 'Compliance and governance summary is not available yet.')
    for note in compliance_summary.get('governance_notes', []):
        st.caption(note)
    st.markdown('### Best sharing mode')
    info_or_table(_build_best_sharing_mode_card(pipeline), 'Best sharing mode guidance is not available yet.')
    executive_summary = pipeline.get('executive_summary', {})
    if executive_summary:
        st.markdown('### Executive Snapshot')
        for bullet in executive_summary.get('stakeholder_summary_bullets', [])[:3]:
            st.write(f'- {bullet}')
    st.markdown('### About Smart Dataset Analyzer')
    app_metadata = pipeline.get('app_metadata', {})
    if app_metadata:
        st.write(str(app_metadata.get('tagline', '')))
        st.write(f"**Built for:** {app_metadata.get('best_for', 'Healthcare analytics, quality review, and stakeholder reporting')}")
        st.write(f"**Works best with:** {app_metadata.get('best_data', 'Healthcare, operational, and general tabular datasets')}")
        st.caption(str(app_metadata.get('synthetic_support', '')))
        st.caption(str(app_metadata.get('maturity', '')))

def render_column_detection(pipeline: dict[str, Any], original_lookup: dict[str, str]) -> None:
    st.subheader('Column Detection')
    detection = pipeline['structure'].detection_table.copy()
    if original_lookup:
        detection.insert(1, 'original_name', detection['column_name'].map(original_lookup).fillna(detection['column_name']))
    info_or_table(detection, 'Column detection details are not available.')
    st.markdown('### Semantic Mapping Review')
    info_or_table(pipeline['semantic'].get('mapping_table', pd.DataFrame()), 'No strong semantic mappings were detected yet.')
    st.markdown('### Data Dictionary')
    info_or_table(pipeline['data_dictionary'], 'Data dictionary details are not available.')
    render_standards(pipeline, key_prefix='column_detection_standards')


def render_profiling(pipeline: dict[str, Any]) -> None:
    profile = pipeline['field_profile']
    st.subheader('Field Profiling')
    metric_row([
        ('Near-constant Fields', fmt(int(profile['is_near_constant'].sum()) if 'is_near_constant' in profile.columns else 0)),
        ('High-cardinality IDs', fmt(int(profile['is_high_cardinality_identifier'].sum()) if 'is_high_cardinality_identifier' in profile.columns else 0)),
        ('Potential Categoricals', fmt(int(profile['looks_potentially_categorical'].sum()) if 'looks_potentially_categorical' in profile.columns else 0)),
        ('Numeric Outlier Signals', fmt(int(profile['has_numeric_outlier_signal'].sum()) if 'has_numeric_outlier_signal' in profile.columns else 0)),
    ])
    info_or_table(profile, 'Field profile details are not available.')
    st.markdown('### Numeric Summary')
    info_or_table(build_numeric_summary(profile), 'No numeric summary is available for the current dataset.')
    if not profile.empty:
        selected = st.selectbox('Selected field deep dive', profile['column_name'].tolist(), key='profile_selected_column')
        row = profile[profile['column_name'] == selected].head(1)
        info_or_table(row, 'No profile details are available for the selected field.')
        inferred = str(row['inferred_type'].iloc[0])
        left_fig = plot_numeric_distribution(pipeline['data'], selected) if inferred == 'numeric' else plot_top_categories(pipeline['data'], selected)
        right_fig = plot_numeric_box(pipeline['data'], selected) if inferred == 'numeric' else None
        cols = st.columns(2)
        with cols[0]:
            info_or_chart(left_fig, 'No distribution chart is available for the selected field.')
        with cols[1]:
            if right_fig is not None:
                st.plotly_chart(right_fig, width='stretch')
            else:
                st.info('No additional deep-dive chart is available for the selected field.')


def render_quality(pipeline: dict[str, Any]) -> None:
    quality = pipeline['quality']
    remediation = pipeline.get('remediation_context', {})
    st.subheader('Data Quality Review')
    metric_row([
        ('Quality Score', fmt(quality['quality_score'])),
        ('High Missingness Fields', fmt(len(quality['high_missing']))),
        ('Mixed-type Suspicions', fmt(len(quality['mixed_type_suspicions']))),
        ('Numeric Outlier Fields', fmt(len(quality['numeric_outliers']))),
    ])
    info_or_chart(plot_missingness(safe_df(quality['high_missing'])), 'No major missingness signal was detected in the current dataset.')
    for label, table, msg in [
        ('High missingness', safe_df(quality['high_missing']), 'No high-missingness fields were detected.'),
        ('Near-constant fields', safe_df(quality['near_constant']), 'No near-constant fields were detected.'),
        ('Suspicious numeric-as-text', safe_df(quality['suspicious_numeric_text']), 'No suspicious numeric-as-text fields were detected.'),
        ('Mixed-type suspicions', safe_df(quality['mixed_type_suspicions']), 'No mixed-type suspicions were detected.'),
        ('Duplicate identifiers', safe_df(quality['duplicate_identifiers']), 'No duplicate-heavy identifier fields were detected.'),
        ('Numeric outliers', safe_df(quality['numeric_outliers']), 'No numeric outlier fields were detected.'),
    ]:
        st.markdown(f'#### {label}')
        info_or_table(table, msg)
    bmi_summary = remediation.get('bmi_remediation', {})
    if bmi_summary.get('available'):
        st.markdown('### BMI Remediation Summary')
        info_or_table(
            pd.DataFrame([
                {
                    'rows_checked': bmi_summary.get('total_rows_checked', 0),
                    'bmi_outliers': bmi_summary.get('total_bmi_outliers', 0),
                    'outlier_pct': bmi_summary.get('outlier_pct', 0.0),
                    'remediation_mode': bmi_summary.get('remediation_mode', 'median'),
                    'replacement_value_if_used': bmi_summary.get('replacement_value_if_used'),
                }
            ]),
            'No BMI remediation summary is available.',
        )
    st.markdown('### Data Quality Rule Engine')
    rule_engine = pipeline['rule_engine']
    if not rule_engine.get('available'):
        st.info(rule_engine.get('reason', 'The rule engine is not available for the current dataset.'))
    else:
        overview = rule_engine.get('overview', {})
        metric_row([
            ('Rules Checked', fmt(overview.get('rules_checked', 0))),
            ('Failed Rules', fmt(overview.get('failed_rules', 0))),
            ('Fields with Failures', fmt(overview.get('fields_with_failures', 0))),
            ('Skipped Fields', fmt(len(overview.get('skipped_fields', [])))),
        ])
        severity_summary = safe_df(rule_engine.get('severity_summary'))
        if not severity_summary.empty:
            st.markdown('#### Rule severity summary')
            info_or_table(severity_summary, 'No severity summary is available for the current rule checks.')
        if rule_engine.get('passed'):
            st.success(rule_engine.get('reason', 'No quality rule failures were detected for the currently mapped fields.'))
        info_or_table(safe_df(rule_engine.get('summary_table')), 'No rule failures were detected for the current dataset.')
        info_or_table(safe_df(rule_engine.get('detail_table')), 'No sample violating rows are available.')


def render_readiness(pipeline: dict[str, Any]) -> None:
    readiness = pipeline['readiness']
    remediation = pipeline.get('remediation_context', {})
    st.subheader('Analysis Readiness')
    metric_row([
        ('Ready Modules', fmt(readiness['available_count'])),
        ('Partially Ready', fmt(readiness['partial_count'])),
        ('Readiness Score', fmt(readiness['readiness_score'], 'score')),
        ('Healthcare Type', pipeline['healthcare']['likely_dataset_type']),
    ])
    next_action = _build_next_compliance_action(pipeline)
    if next_action:
        st.caption(f"Recommended next compliance action: {next_action['title']} ? {next_action['detail']}")
    info_or_table(safe_df(readiness['readiness_table']), 'No readiness detail is available for the current dataset.')
    helper_summary = pd.DataFrame([
        {
            'native_fields': remediation.get('native_field_count', 0),
            'synthetic_helper_fields': remediation.get('synthetic_field_count', 0),
            'derived_helper_fields': remediation.get('derived_field_count', 0),
            'remediated_fields': remediation.get('remediated_field_count', 0),
        }
    ])
    st.markdown('### Remediation and helper-field transparency')
    info_or_table(helper_summary, 'No remediation transparency summary is available yet.')
    helper_notes = []
    for key in ['bmi_remediation', 'synthetic_cost', 'synthetic_clinical', 'synthetic_readmission']:
        note = remediation.get(key, {}).get('lineage_note')
        if note:
            helper_notes.append({'support_type': key.replace('_', ' ').title(), 'note': note})
    info_or_table(pd.DataFrame(helper_notes), 'No remediation or helper-field notes are available yet.')

    standards_actions = _build_standards_prefill_actions(pipeline)
    if standards_actions:
        st.markdown('### Standards prefill actions')
        action_cols = st.columns(min(3, len(standards_actions)))
        for idx, action in enumerate(standards_actions):
            if action_cols[idx % len(action_cols)].button(action['label'], key=f"standards_action_{action['action']}"):
                message = _run_standards_prefill_action(pipeline, action['action'])
                log_event('Standards Prefill Action', message, 'Standards mapping', 'Analysis readiness')
                st.success(message)
                st.rerun()

    ehr_actions = _build_ehr_onboarding_actions(pipeline)
    if ehr_actions:
        st.markdown('### EHR onboarding presets')
        preset_cols = st.columns(min(3, len(ehr_actions)))
        for idx, action in enumerate(ehr_actions):
            if preset_cols[idx % len(preset_cols)].button(action['label'], key=f"ehr_action_{action['action']}"):
                message = _run_ehr_onboarding_action(action['action'])
                log_event('EHR Onboarding Preset', message, 'EHR onboarding', 'Analysis readiness')
                st.success(message)
                st.rerun()
        info_or_table(_build_ehr_onboarding_explanation(pipeline), 'No supporting onboarding signals are available yet.')

    quick_actions = _build_compliance_quick_actions(pipeline)
    if quick_actions:
        st.markdown('### Quick remediation actions')
        quick_cols = st.columns(min(2, len(quick_actions)))
        for idx, action in enumerate(quick_actions):
            if quick_cols[idx % len(quick_cols)].button(action['label'], key=f"quick_action_{action['action']}"):
                message = _run_compliance_quick_action(action['action'])
                log_event('Compliance Quick Action', message, 'Readiness action', 'Analysis readiness')
                st.success(message)
                st.rerun()

    st.markdown('### Dataset Improvement Plan')
    info_or_table(safe_df(pipeline['improvement_plan']), 'The dataset is already in a strong position for the currently enabled modules.')
    st.markdown('### Data Remediation Assistant')
    info_or_table(safe_df(pipeline['remediation']), 'No remediation steps are needed right now.')
    st.markdown('### Module visibility guide')
    info_or_table(pd.DataFrame([
        {'module': 'Data Remediation Assistant', 'location': 'Data Quality ? Analysis Readiness', 'status': 'Available'},
        {'module': 'Patient Cohort Builder', 'location': 'Healthcare Analytics ? Cohort Analysis', 'status': 'Available' if pipeline['healthcare']['healthcare_readiness_score'] >= 0.45 else 'Limited'},
        {'module': 'Analysis Log', 'location': 'Data Intake', 'status': 'Available'},
        {'module': 'Healthcare Standards Validator', 'location': 'Column Detection and Healthcare Intelligence', 'status': 'Available' if pipeline['standards'].get('available') else 'Limited'},
        {'module': 'Workflow Packs', 'location': 'Data Intake', 'status': 'Available'},
        {'module': 'Data Lineage', 'location': 'Data Intake', 'status': 'Available'},
    ]), 'No module visibility guidance is available yet.')
    render_privacy(pipeline)


def render_healthcare(pipeline: dict[str, Any]) -> None:
    healthcare = pipeline['healthcare']
    remediation = pipeline.get('remediation_context', {})
    st.subheader('Healthcare Intelligence')
    metric_row([
        ('Healthcare Readiness', fmt(healthcare['healthcare_readiness_score'], 'pct')),
        ('Likely Dataset Type', healthcare['likely_dataset_type']),
        ('Matched Healthcare Fields', fmt(len(healthcare.get('matched_healthcare_fields', [])))),
        ('Risk Segmentation', 'Available' if healthcare.get('risk_segmentation', {}).get('available') else 'Limited'),
    ])
    synthetic_notes = []
    if remediation.get('synthetic_cost', {}).get('available'):
        synthetic_notes.append('Estimated cost is synthetic/demo-derived and should be interpreted as an exploratory financial placeholder.')
    if remediation.get('synthetic_clinical', {}).get('available'):
        synthetic_notes.append('Diagnosis and clinical-risk labels are derived approximations, not billing-grade clinical codes.')
    if remediation.get('synthetic_readmission', {}).get('available'):
        synthetic_notes.append('Readmission analytics are enabled with a deterministic synthetic flag for workflow demonstration.')
    for note in synthetic_notes:
        st.caption(note)
    for key, table_key, metric_col, title in [
        ('utilization', 'monthly_utilization', 'event_count', 'Monthly utilization trend'),
        ('cost', 'by_segment', 'total_cost', 'Top cost drivers'),
        ('provider', 'table', 'volume', 'Provider or facility volume'),
        ('diagnosis', 'table', 'volume', 'Diagnosis or procedure volume'),
        ('risk_segmentation', 'segment_table', 'patient_count', 'Risk segmentation'),
    ]:
        st.markdown(f'### {title}')
        section = healthcare.get(key, {})
        if not section.get('available'):
            st.info(section.get('reason', 'This healthcare module is not available for the current dataset.'))
        else:
            table = safe_df(section.get(table_key))
            info_or_table(table, f'No table is available for {title.lower()}.')
            if not table.empty:
                if 'month' in table.columns:
                    info_or_chart(plot_time_trend(table, 'month', metric_col, title), f'No trend chart is available for {title.lower()}.')
                else:
                    info_or_chart(plot_bar(table, table.columns[0], metric_col, title), f'No chart is available for {title.lower()}.')
    st.markdown('### Segment Discovery 2.0')
    segments = healthcare.get('segment_discovery', {})
    if not segments.get('available'):
        st.info(segments.get('reason', 'Segment discovery is not available for the current dataset.'))
    else:
        st.caption(segments.get('summary', 'The platform is highlighting the segments that stand out most strongly on risk, outcome, or duration signals.'))
        top_segment = segments.get('top_segment', {})
        metric_row([
            ('Discovered Segments', fmt(len(safe_df(segments.get('segment_table'))))),
            ('Top Signal', str(top_segment.get('dominant_signal', 'Not available'))),
            ('Top Priority', str(top_segment.get('priority_band', 'Not available'))),
            ('Top Segment Size', fmt(top_segment.get('record_count', 0))),
        ])
        info_or_table(safe_df(segments.get('segment_table')), 'No standout segments are available for the current dataset.')
        if not safe_df(segments.get('segment_table')).empty:
            chart_table = safe_df(segments.get('segment_table')).head(8).copy()
            info_or_chart(
                plot_bar(chart_table, 'discovered_segment', 'standout_score', 'Top Segment Discovery Signals'),
                'No segment discovery chart is available.',
            )
        st.markdown('#### Segment Pattern Summary')
        info_or_table(safe_df(segments.get('metric_leaders')), 'No segment pattern summary is available for the current dataset.')
        if top_segment.get('suggested_follow_up_analysis'):
            st.caption(f"Suggested follow-up: {top_segment['suggested_follow_up_analysis']}")
        if top_segment.get('review_question'):
            st.caption(f"Review question: {top_segment['review_question']}")
    st.markdown('### Hospital Readmission Risk Analytics')
    _render_readmission_overview(healthcare.get('readmission', {}))
    render_predictive_modeling_studio(pipeline)
    st.markdown('### Clinical Pathway and Outcome Intelligence')
    pathway = healthcare.get('care_pathway', {})
    if not pathway.get('available'):
        st.info(pathway.get('reason', 'Clinical pathway intelligence is not available for the current dataset.'))
    else:
        st.caption(pathway.get('summary', 'The platform is summarizing treatment pathways using observed duration and outcome patterns.'))
        stage_table = safe_df(pathway.get('stage_table'))
        treatment_table = safe_df(pathway.get('treatment_table'))
        pathway_table = safe_df(pathway.get('pathway_table'))
        bottleneck_table = safe_df(pathway.get('bottleneck_summary'))
        poor_outcome_table = safe_df(pathway.get('poor_outcome_pathways'))
        metric_row([
            ('Stage Pathways', fmt(len(stage_table))),
            ('Treatment Pathways', fmt(len(treatment_table))),
            ('Combined Pathways', fmt(len(pathway_table))),
            ('Bottleneck Signals', fmt(len(bottleneck_table))),
        ])
        st.markdown('#### Average Duration by Stage')
        info_or_table(stage_table, 'No stage-level pathway summary is available.')
        if not stage_table.empty and 'average_treatment_duration_days' in stage_table.columns:
            info_or_chart(
                plot_bar(stage_table, pathway.get('stage_column'), 'average_treatment_duration_days', 'Average Treatment Duration by Stage'),
                'No stage duration chart is available.',
            )
        st.markdown('#### Average Duration by Treatment')
        info_or_table(treatment_table, 'No treatment-level pathway summary is available.')
        if not treatment_table.empty and 'average_treatment_duration_days' in treatment_table.columns:
            info_or_chart(
                plot_bar(treatment_table, pathway.get('treatment_column'), 'average_treatment_duration_days', 'Average Treatment Duration by Treatment'),
                'No treatment duration chart is available.',
            )
        st.markdown('#### Outcome by Pathway')
        info_or_table(pathway_table, 'No combined pathway summary is available.')
        if not pathway_table.empty and 'survival_rate' in pathway_table.columns:
            chart_frame = pathway_table.head(8).copy()
            chart_frame['pathway_label'] = chart_frame[pathway.get('stage_column')].astype(str) + ' -> ' + chart_frame[pathway.get('treatment_column')].astype(str)
            info_or_chart(
                plot_bar(chart_frame, 'pathway_label', 'survival_rate', 'Survival by Pathway'),
                'No pathway outcome chart is available.',
            )
        st.markdown('#### Pathway Bottlenecks to Review')
        info_or_table(bottleneck_table, 'No pathway bottlenecks were flagged for the current dataset.')
        st.markdown('#### Poor-Outcome Pathways')
        info_or_table(poor_outcome_table, 'No poor-outcome pathways were identified for the current dataset.')
    render_standards(pipeline, key_prefix='healthcare_standards')
    render_privacy(pipeline)


def render_cohort_analysis(pipeline: dict[str, Any]) -> None:
    st.subheader('Cohort Analysis')
    data = pipeline['data']
    canonical_map = pipeline['semantic']['canonical_map']
    risk_table = pipeline['healthcare'].get('risk_segmentation', {}).get('segment_table', pd.DataFrame())
    age_col = canonical_map.get('age')
    gender_col = canonical_map.get('gender')
    diagnosis_col = canonical_map.get('diagnosis_code')
    treatment_col = canonical_map.get('treatment_type')
    stage_col = canonical_map.get('cancer_stage')
    filters: dict[str, Any] = {}
    cols = st.columns(3)
    if age_col and age_col in data.columns:
        ages = pd.to_numeric(data[age_col], errors='coerce').dropna()
        if not ages.empty:
            filters['age_range'] = cols[0].slider('Age range', int(ages.min()), int(ages.max()), (int(ages.min()), int(ages.max())), key='cohort_age_range')
    if gender_col and gender_col in data.columns:
        filters['gender'] = cols[1].multiselect('Gender', sorted(data[gender_col].dropna().astype(str).unique().tolist()), key='cohort_gender')
    if stage_col and stage_col in data.columns:
        filters['cancer_stage'] = cols[2].multiselect('Cancer stage', sorted(data[stage_col].dropna().astype(str).unique().tolist())[:12], key='cohort_stage')
    cols2 = st.columns(3)
    if diagnosis_col and diagnosis_col in data.columns:
        filters['diagnosis'] = cols2[0].multiselect('Diagnosis', sorted(data[diagnosis_col].dropna().astype(str).value_counts().head(12).index.tolist()), key='cohort_diagnosis')
    if treatment_col and treatment_col in data.columns:
        filters['treatment'] = cols2[1].multiselect('Treatment', sorted(data[treatment_col].dropna().astype(str).unique().tolist())[:12], key='cohort_treatment')
    risk_options = risk_table['risk_segment'].tolist() if isinstance(risk_table, pd.DataFrame) and 'risk_segment' in risk_table.columns else []
    if risk_options:
        filters['risk_segment'] = cols2[2].multiselect('Risk segment', risk_options, key='cohort_risk_segment')
    cohort = build_cohort_summary(
        data,
        canonical_map,
        age_range=filters.get('age_range'),
        genders=filters.get('gender'),
        diagnoses=filters.get('diagnosis'),
        treatments=filters.get('treatment'),
        cancer_stages=filters.get('cancer_stage'),
        risk_segments=filters.get('risk_segment'),
    )
    if not cohort.get('available'):
        st.info(cohort.get('reason', 'Cohort building is not available for the current dataset.'))
        return
    summary = cohort.get('summary', {})
    metric_row([
        ('Cohort Size', fmt(summary.get('cohort_size', 0))),
        ('Survival Rate', fmt(summary.get('survival_rate'), 'pct') if summary.get('survival_rate') is not None else 'Not available'),
        ('High-Risk Share', fmt(summary.get('high_risk_share'), 'pct') if summary.get('high_risk_share') is not None else 'Not available'),
        ('Avg Treatment Duration', fmt(summary.get('average_treatment_duration_days'), 'float') if summary.get('average_treatment_duration_days') is not None else 'Not available'),
    ])
    info_or_table(safe_df(cohort.get('risk_distribution')), 'No risk distribution is available for the current cohort.')
    info_or_table(safe_df(cohort.get('outcome_metrics_table', cohort.get('subgroup_metrics'))), 'No outcome metrics are available for the current cohort.')
    trend = safe_df(cohort.get('cohort_trend_table'))
    if not trend.empty and 'month' in trend.columns:
        info_or_chart(plot_time_trend(trend, 'month', 'record_count', 'Cohort trend over time'), 'No cohort trend chart is available.')
    info_or_table(safe_df(cohort.get('preview')), 'No cohort preview is available.')
    st.markdown('### Cohort Monitoring Over Time')
    monitoring = cohort_monitoring_over_time(data, canonical_map)
    if monitoring.get('available'):
        info_or_table(safe_df(monitoring.get('trend_table')), 'No cohort monitoring trend table is available.')
    else:
        st.info(monitoring.get('reason', 'Cohort monitoring is not available for the current dataset.'))
    st.markdown('### Readmission Cohort Builder')
    readmission = pipeline['healthcare'].get('readmission', {})
    if not readmission.get('available'):
        _render_readmission_readiness(readmission)
        st.info(readmission.get('reason', 'Readmission cohort analysis is not available for the current dataset.'))
        return
    cohort_options = ['Overall Population', 'Older Adults (65+)', 'High LOS Patients (6+ days)']
    if not safe_df(readmission.get('by_diagnosis')).empty:
        cohort_options.append('Top Diagnosis Group')
    if not safe_df(readmission.get('high_risk_segments')).empty:
        cohort_options.append('Highest Readmission Segment')
    selected_readmission_cohort = st.selectbox('Readmission cohort focus', cohort_options, key='readmission_cohort_focus')
    readmission_cohort = build_readmission_cohort_review(readmission, selected_readmission_cohort)
    if readmission_cohort.get('available'):
        cohort_summary = readmission_cohort.get('summary', {})
        metric_row([
            ('Readmission Cohort Size', fmt(cohort_summary.get('cohort_size', 0))),
            ('Cohort Readmission Rate', fmt(cohort_summary.get('readmission_rate'), 'pct')),
            ('Overall Population Rate', fmt(cohort_summary.get('overall_population_rate'), 'pct')),
            ('Gap vs Overall', fmt(cohort_summary.get('gap_vs_overall'), 'pct')),
        ])
        st.caption(readmission_cohort.get('suggested_next_action', ''))
        info_or_table(safe_df(readmission_cohort.get('preview')), 'No row-level readmission cohort preview is available.')
        st.markdown('#### Readmission Intervention Planning')
        planner_cols = st.columns(4)
        follow_up = planner_cols[0].slider('Enhanced discharge follow-up (%)', 0, 50, 15, key='readmit_follow_up')
        case_mgmt = planner_cols[1].slider('Targeted case management (%)', 0, 50, 10, key='readmit_case_mgmt')
        los_reduction = planner_cols[2].slider('LOS reduction (days)', 0.0, 3.0, 1.0, 0.5, key='readmit_los_reduction')
        early_follow_up = planner_cols[3].slider('Early follow-up improvement (%)', 0, 50, 10, key='readmit_early_follow_up')
        intervention = plan_readmission_intervention(readmission, selected_readmission_cohort, follow_up, case_mgmt, los_reduction, early_follow_up)
        if intervention.get('available'):
            metric_row([
                ('Baseline Rate', fmt(intervention.get('baseline_readmission_rate'), 'pct')),
                ('Projected Rate', fmt(intervention.get('projected_readmission_rate'), 'pct')),
                ('Projected Overall Rate', fmt(intervention.get('projected_overall_readmission_rate'), 'pct')),
                ('Estimated Readmissions Avoided', fmt(intervention.get('estimated_readmissions_avoided', 0), 'float')),
            ])
            info_or_table(safe_df(intervention.get('summary_table')), 'No intervention summary is available.')
            st.write('Assumptions:')
            for line in intervention.get('assumptions', []):
                st.write(f'- {line}')
        else:
            st.info(intervention.get('reason', 'Readmission intervention planning is not available for the selected cohort.'))
    else:
        st.info(readmission_cohort.get('reason', 'Readmission cohort review is not available for the selected cohort.'))

def render_trend_analysis(pipeline: dict[str, Any]) -> None:
    st.subheader('Trend Analysis')
    structure = pipeline['structure']
    data = pipeline['data']
    if not structure.default_date_column:
        st.info('No reliable date field was detected, so time-series analysis is limited. Add or map a usable date field to unlock stronger longitudinal review.')
        return
    date_col = structure.default_date_column
    trend_source = data[[date_col]].copy()
    trend_source[date_col] = pd.to_datetime(trend_source[date_col], errors='coerce')
    trend_source = trend_source.dropna(subset=[date_col])
    if trend_source.empty:
        st.info('No usable rows remain after parsing the default date field.')
    else:
        monthly = trend_source.assign(month=trend_source[date_col].dt.to_period('M').dt.to_timestamp()).groupby('month').size().reset_index(name='record_count')
        info_or_chart(plot_time_trend(monthly, 'month', 'record_count', 'Record volume over time'), 'No record trend is available yet.')
        info_or_table(monthly, 'No trend table is available yet.')
    survival = pipeline['healthcare'].get('survival_outcomes', {})
    if survival.get('available'):
        for label, table_key in [('Survival by stage', 'stage_table'), ('Survival by treatment', 'treatment_table'), ('Treatment duration distribution', 'duration_distribution'), ('Outcome trend over time', 'outcome_trend'), ('Treatment duration trend', 'treatment_duration_trend'), ('Disease progression timeline', 'progression_timeline')]:
            st.markdown(f'### {label}')
            table = safe_df(survival.get(table_key))
            info_or_table(table, f'{label} is not available for the current dataset.')
    readmission = pipeline['healthcare'].get('readmission', {})
    st.markdown('### Readmission Trend')
    if readmission.get('available') and not safe_df(readmission.get('trend')).empty:
        trend = safe_df(readmission.get('trend'))
        info_or_chart(plot_time_trend(trend, 'month', 'readmission_rate', 'Readmission rate over time'), 'No readmission trend chart is available.')
        info_or_table(trend, 'No readmission trend table is available.')
    else:
        _render_readmission_readiness(readmission)
        st.info(readmission.get('reason', 'Readmission trend analysis is not available for the current dataset.'))
    corr = data[structure.numeric_columns].corr(numeric_only=True) if structure.numeric_columns else pd.DataFrame()
    st.markdown('### Correlation Heatmap')
    info_or_chart(plot_correlation(corr), 'No correlation heatmap is available because the dataset lacks sufficient numeric fields.')


def render_automated_insight_board(pipeline: dict[str, Any]) -> None:
    board = pipeline.get('insight_board', {})
    st.markdown('### Automated Insight Board')
    if not board.get('available'):
        st.info(board.get('reason', 'The current dataset does not support a strong automated insight board yet. Continue with readiness review and remediation to unlock more executive signals.'))
        return

    cards = board.get('kpi_cards', [])
    if cards:
        metric_row([(card['label'], card['value']) for card in cards])
        for card in cards:
            st.caption(f"{card['label']}: {card['description']}")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('#### Top Findings')
        for line in board.get('top_findings', []):
            st.write(f'- {line}')
        st.markdown('#### Top Risks')
        for line in board.get('top_risks', []):
            st.write(f'- {line}')
    with col2:
        st.markdown('#### Top Recommendations')
        for line in board.get('top_recommendations', []):
            st.write(f'- {line}')
        if board.get('benchmark_gaps'):
            st.markdown('#### Key Benchmark Gaps')
            for line in board.get('benchmark_gaps', []):
                st.write(f'- {line}')

    anomaly_summary = safe_df(board.get('key_anomaly_summary'))
    if not anomaly_summary.empty:
        st.markdown('#### Key Anomaly Summary')
        info_or_table(anomaly_summary, 'No anomaly summary is available for the current board.')

    chart_spec = board.get('key_chart')
    if isinstance(chart_spec, dict) and isinstance(chart_spec.get('data'), pd.DataFrame) and not chart_spec['data'].empty:
        st.markdown('#### Executive Chart')
        if chart_spec.get('kind') == 'time':
            info_or_chart(plot_time_trend(chart_spec['data'], chart_spec['x'], chart_spec['y'], chart_spec['title']), 'No board chart is available right now.')
        else:
            info_or_chart(plot_bar(chart_spec['data'], chart_spec['x'], chart_spec['y'], chart_spec['title']), 'No board chart is available right now.')

    intervention = board.get('intervention_summary')
    if isinstance(intervention, dict):
        st.markdown(f"#### {intervention.get('title', 'Intervention Summary')}")
        st.info(f"{intervention.get('headline', '')} {intervention.get('detail', '')}".strip())


def render_predictive_modeling_studio(pipeline: dict[str, Any]) -> None:
    st.markdown('### Predictive Modeling Studio')
    defaults = default_modeling_selection(pipeline['data'], pipeline['semantic']['canonical_map'])
    if not defaults.get('available'):
        st.info(defaults.get('reason', 'Predictive modeling is not available for the current dataset.'))
        return

    candidate_table = safe_df(defaults.get('candidate_table'))
    target_options = candidate_table['source_column'].astype(str).tolist() if not candidate_table.empty else []
    if not target_options:
        st.info('No suitable target variable is available for guided modeling.')
        return

    default_target = defaults.get('default_target', target_options[0])
    default_index = target_options.index(default_target) if default_target in target_options else 0
    target_column = st.selectbox(
        'Target variable',
        target_options,
        index=default_index,
        key='modeling_target',
        help='Choose a binary or low-cardinality outcome field such as readmission or survived.',
    )

    feature_options = [column for column in pipeline['data'].columns if column != target_column]
    default_features = [column for column in defaults.get('default_features', []) if column in feature_options]
    selected_features = st.multiselect(
        'Feature fields',
        feature_options,
        default=default_features,
        key='modeling_features',
        help='Start with clinically meaningful fields such as age, diagnosis, LOS, treatment, or smoking status.',
    )
    model_type = st.selectbox(
        'Model type',
        ['Logistic Regression', 'Random Forest', 'Gradient Boosting', 'XGBoost'],
        key='modeling_type',
        help='Use logistic regression for a more interpretable baseline or random forest for a flexible non-linear model.',
    )

    comparison = build_model_comparison_studio(
        pipeline['data'],
        pipeline['semantic']['canonical_map'],
        target_column,
        selected_features,
    )
    st.markdown('#### Model Comparison Studio')
    if comparison.get('available'):
        best = comparison.get('best_model_summary', {})
        metric_row([
            ('Best Model', best.get('model_name', 'Not available')),
            ('Best ROC AUC', fmt(best.get('roc_auc'), 'float') if best.get('roc_auc') is not None else 'Not available'),
            ('Best F1', fmt(best.get('f1'), 'float') if best.get('f1') is not None else 'Not available'),
            ('Synthetic Features Used', 'Yes' if best.get('synthetic_features_used') else 'No'),
        ])
        st.caption(best.get('why_it_won', 'The strongest available model is shown below.'))
        info_or_table(safe_df(comparison.get('model_comparison_table')), 'No model comparison table is available.')
        for note in comparison.get('comparison_notes', []):
            st.write(f'- {note}')
    else:
        st.info(comparison.get('reason', 'Model comparison is not available for the current modeling selection.'))

    result = build_predictive_model(pipeline['data'], pipeline['semantic']['canonical_map'], target_column, selected_features, model_type)
    if not result.get('available'):
        st.info(result.get('reason', 'The current target and feature selection is not sufficient for guided modeling.'))
        return

    st.caption('This guided modeling workflow is intended for exploratory analytics and education. It is not a production clinical decision engine.')
    train_test_summary = safe_df(result.get('train_test_summary'))
    if not train_test_summary.empty:
        row = train_test_summary.iloc[0]
        metric_row([
            ('Train Rows', fmt(row.get('train_rows', 0))),
            ('Test Rows', fmt(row.get('test_rows', 0))),
            ('Features', fmt(row.get('feature_count', 0))),
            ('Accuracy', fmt(result.get('accuracy'), 'float')),
            ('Precision', fmt(result.get('precision'), 'float')),
            ('Recall', fmt(result.get('recall'), 'float')),
            ('F1', fmt(result.get('f1'), 'float')),
            ('ROC AUC', fmt(result.get('roc_auc'), 'float') if result.get('roc_auc') is not None else 'Not available'),
        ])
        info_or_table(train_test_summary, 'No train/test summary is available.')
    if result.get('weak_target_note'):
        st.warning(str(result['weak_target_note']))
    if result.get('synthetic_features_used'):
        st.caption('Synthetic helper fields are participating in this model run: ' + ', '.join(map(str, result['synthetic_features_used'])))

    transformations = safe_df(result.get('feature_transformations'))
    if not transformations.empty:
        st.markdown('#### Feature Compatibility Layer')
        st.caption('Interval-style bins and datetime fields were converted into model-safe numeric representations before training.')
        info_or_table(transformations, 'No feature compatibility transforms were required for the current model run.')

    st.markdown('#### Confusion Matrix')
    info_or_table(safe_df(result.get('confusion_matrix')), 'No confusion matrix is available for the current model run.')

    importance = safe_df(result.get('feature_importance'))
    st.markdown('#### Feature Importance')
    info_or_table(importance, 'No feature importance summary is available.')
    if not importance.empty:
        info_or_chart(plot_bar(importance, 'feature', 'importance', 'Top Feature Importance'), 'No feature importance chart is available.')

    distribution = safe_df(result.get('prediction_distribution'))
    st.markdown('#### Prediction Distribution')
    info_or_table(distribution, 'No prediction distribution is available for the current model run.')
    if not distribution.empty:
        distribution_axis = 'probability_band_label' if 'probability_band_label' in distribution.columns else 'probability_band'
        info_or_chart(plot_bar(distribution, distribution_axis, 'record_count', 'Predicted Probability Distribution'), 'No prediction distribution chart is available.')

    st.markdown('#### Highest Predicted Risk Rows')
    info_or_table(safe_df(result.get('high_risk_rows')), 'No high-risk prediction preview is available for the current model run.')

    explainability = build_prediction_explainability(result)
    st.markdown('#### Explainable Prediction Layer')
    if not explainability.get('available'):
        st.info(explainability.get('reason', 'Prediction explainability is not available for the current model run.'))
        return

    for line in explainability.get('narrative', []):
        st.write(f'- {line}')
    st.markdown('##### Top Model Drivers')
    info_or_table(safe_df(explainability.get('driver_table')), 'No driver explanation table is available.')
    if not safe_df(explainability.get('driver_table')).empty:
        info_or_chart(plot_bar(safe_df(explainability.get('driver_table')), 'feature', 'importance', 'Top Prediction Drivers'), 'No prediction driver chart is available.')
    st.markdown('##### High-Risk Row Explanations')
    info_or_table(safe_df(explainability.get('row_explanations')), 'No row-level explanation preview is available.')
    st.markdown('##### High-Risk Segment Explanations')
    info_or_table(safe_df(explainability.get('segment_explanations')), 'No segment-level explanation summary is available.')

    fairness = build_model_fairness_review(result, pipeline['data'], pipeline['semantic']['canonical_map'])
    st.markdown('#### Fairness & Bias Review')
    if not fairness.get('available'):
        st.info(fairness.get('reason', 'Model fairness review is not available for the current model output.'))
        return
    st.caption(fairness.get('narrative', 'This fairness view is intended as a transparent screening aid, not a formal bias audit.'))
    summary = safe_df(fairness.get('fairness_summary'))
    if not summary.empty:
        info_or_table(summary, 'No fairness summary is available.')
    info_or_table(safe_df(fairness.get('comparison_table')), 'No subgroup comparison table is available for the current model output.')
    info_or_table(safe_df(fairness.get('flags')), 'No material subgroup gaps were flagged in the current model output.')
    for note in fairness.get('fairness_limitations', []):
        st.caption(note)


def render_key_insights(pipeline: dict[str, Any], dataset_name: str) -> None:
    st.subheader('Key Insights')
    render_automated_insight_board(pipeline)
    executive_summary = pipeline.get('executive_summary', {})
    if executive_summary:
        st.markdown('### Executive Summary Generator')
        for bullet in executive_summary.get('stakeholder_summary_bullets', []):
            st.write(f'- {bullet}')
        recruiter_summary = executive_summary.get('recruiter_demo_summary')
        if recruiter_summary:
            st.caption(recruiter_summary)
    for line in pipeline['insights'].get('summary_lines', []):
        st.write(f'- {line}')
    prioritized = pipeline.get('prioritized_insights', {})
    if prioritized.get('available'):
        st.markdown('### Prioritized Insights')
        info_or_table(safe_df(prioritized.get('critical_findings')), 'No critical findings are available right now.')
        info_or_table(safe_df(prioritized.get('watchlist_findings')), 'No watchlist findings are available right now.')
        info_or_table(safe_df(prioritized.get('remediation_opportunities')), 'No remediation opportunities were prioritized right now.')
    st.markdown('### Action Recommendations')
    info_or_table(safe_df(pipeline['action_recommendations']), 'No action recommendations are available yet. Continue with profiling and readiness review to surface next-step guidance.')
    st.markdown('### Intervention Recommendation Engine')
    info_or_table(safe_df(pipeline.get('intervention_recommendations')), 'No intervention recommendations are available yet.')
    benchmarking = pipeline.get('kpi_benchmarking', {})
    st.markdown('### KPI Benchmarking Layer')
    if benchmarking.get('available'):
        metric_row([(card['label'], card['value']) for card in benchmarking.get('kpi_cards', [])[:4]])
        info_or_table(safe_df(benchmarking.get('benchmark_table')), 'No internal benchmarking table is available.')
        for signal in benchmarking.get('standout_positive_signals', []):
            st.write(f'- Positive signal: {signal}')
        for signal in benchmarking.get('standout_risk_signals', []):
            st.write(f'- Risk signal: {signal}')
    else:
        st.info(benchmarking.get('reason', 'KPI benchmarking is not available for the current dataset.'))
    scenarios = pipeline.get('scenario_studio', {})
    st.markdown('### Scenario Simulation Studio')
    if scenarios.get('available'):
        st.caption(scenarios.get('summary', 'Directional scenarios are available for stakeholder review.'))
        info_or_table(safe_df(scenarios.get('scenario_table')), 'No scenario table is available.')
    else:
        st.info(scenarios.get('reason', 'Scenario simulation is not available for the current dataset.'))
    st.markdown('### Explainability & Fairness')
    fairness = pipeline['healthcare'].get('explainability_fairness', {})
    if fairness.get('available'):
        if fairness.get('high_risk_segment_explanation'):
            st.write(fairness['high_risk_segment_explanation'])
        info_or_table(safe_df(fairness.get('factor_table')), 'No driver table is available.')
        info_or_table(safe_df(fairness.get('comparison_table')), 'No subgroup comparison table is available.')
        info_or_table(safe_df(fairness.get('fairness_flags')), 'No subgroup gap flags were detected.')
    else:
        st.info(fairness.get('reason', 'Explainability and fairness review is not available for the current dataset.'))
    st.markdown('### Operational Alerts')
    alerts = pipeline['healthcare'].get('operational_alerts')
    if not isinstance(alerts, dict):
        alerts = operational_alerts(pipeline['data'], pipeline['semantic']['canonical_map'], pipeline['healthcare'], 0.70, 0.25, 5, 60.0, 0.10)
        pipeline['healthcare']['operational_alerts'] = alerts
    if alerts.get('available'):
        info_or_table(safe_df(alerts.get('alerts_table')), 'No operational alerts are active.')
    else:
        st.info(alerts.get('reason', 'Operational alerts are not available yet.'))
    st.markdown('### Readmission Focus')
    readmission = pipeline['healthcare'].get('readmission', {})
    if readmission.get('available'):
        overview = readmission.get('overview', {})
        st.write(f"The current population shows a readmission rate of {fmt(overview.get('overall_readmission_rate'), 'pct')} across {fmt(overview.get('records_in_scope', 0))} records.")
        info_or_table(safe_df(readmission.get('high_risk_segments')).head(5), 'No standout readmission segments are available.')
    else:
        st.info(readmission.get('reason', 'Readmission-focused review is not available for the current dataset.'))
    st.markdown('### AI Copilot')
    messages = initialize_copilot_memory()
    if not messages:
        with st.chat_message('assistant'):
            st.write('Ask about dataset summary, average cost, average length of stay, readmission by department, highest readmission risk, readmission drivers, or the top diagnosis by cost.')
    for message in messages:
        with st.chat_message(message.get('role', 'assistant')):
            st.write(message.get('content', ''))
            table = message.get('table_data')
            if isinstance(table, pd.DataFrame) and not table.empty:
                st.dataframe(table, width='stretch')
            chart = message.get('chart_figure')
            if chart is not None:
                st.plotly_chart(chart, width='stretch')
    prompt = st.chat_input('Ask the AI Copilot a question about the current dataset', key='copilot_chat_input')
    if prompt:
        append_copilot_message('user', prompt)
        log_event('AI Copilot Question', f'Asked: {prompt}', 'AI copilot', 'Insight exploration')
        response = run_copilot_question(prompt, pipeline['data'], {'matched_schema': pipeline['semantic']['canonical_map'], 'dataset_name': dataset_name})
        append_copilot_message('assistant', response.get('answer', ''), response)
        st.rerun()
    st.markdown('### AI Copilot Workflow Actions')
    workflow_prompt = st.text_input('Ask the copilot to prepare a workflow action', key='workflow_action_prompt')
    result = plan_workflow_action(
        workflow_prompt,
        pipeline['data'],
        pipeline['semantic']['canonical_map'],
        pipeline['readiness'],
        pipeline['healthcare'],
        pipeline['remediation'],
    )
    planned_action = result.get('planned_action')
    if planned_action:
        st.caption(f'Planned action: {planned_action}')
    st.write(result.get('message', ''))
    if result.get('recommended_section'):
        st.caption(f"Recommended section: {result['recommended_section']}")
    preview_table = result.get('preview_table')
    if isinstance(preview_table, pd.DataFrame) and not preview_table.empty:
        info_or_table(preview_table, 'No workflow preview is available yet.')
    preview_chart = result.get('preview_chart')
    if preview_chart is not None:
        info_or_chart(preview_chart, 'No workflow preview chart is available yet.')
    prompts = result.get('suggested_prompts', [])
    if prompts:
        st.caption('Suggested follow-up prompts:')
        for prompt_text in prompts:
            st.write(f'- {prompt_text}')
    if result.get('widget_updates'):
        info_or_table(pd.DataFrame([{'setting': k, 'value': v} for k, v in result['widget_updates'].items()]), 'No workflow updates are pending.')
        if st.button('Apply workflow guidance', key='apply_workflow_guidance'):
            for key, value in result['widget_updates'].items():
                st.session_state[key] = value
            log_event('Workflow Action', result.get('message', 'Applied workflow guidance.'), 'Workflow copilot', 'Guided workflow')
            st.rerun()


def _build_standards_override_summary(standards: dict[str, Any]) -> pd.DataFrame:
    effective = safe_df(standards.get('effective_mappings'))
    interop = standards.get('interoperability_report', {}) if isinstance(standards, dict) else {}
    terminology = safe_df(interop.get('effective_terminology', interop.get('terminology_validation', pd.DataFrame())))
    rows: list[dict[str, object]] = []
    if not effective.empty:
        mapped_count = int((effective['source_column'].astype(str) != '').sum()) if 'source_column' in effective.columns else len(effective)
        manual_count = int((effective.get('mapping_source', pd.Series(dtype=str)).astype(str) == 'Manual override').sum()) if 'mapping_source' in effective.columns else 0
        rows.append({
            'focus_area': 'Standards mappings',
            'status': f'{mapped_count} effective mappings',
            'detail': f'{manual_count} manually confirmed' if manual_count else 'Using auto-detected mappings',
        })
    if not terminology.empty:
        confirmed = int((terminology.get('status', pd.Series(dtype=str)).astype(str) == 'Manually confirmed').sum()) if 'status' in terminology.columns else 0
        rows.append({
            'focus_area': 'Terminology confirmation',
            'status': f'{len(terminology)} terminology fields',
            'detail': f'{confirmed} manually confirmed' if confirmed else 'Using detected coded-field signals',
        })
    return pd.DataFrame(rows)


def _build_best_sharing_mode_card(pipeline: dict[str, Any]) -> pd.DataFrame:
    privacy = pipeline['privacy_review']
    standards = pipeline['standards']
    risk_level = privacy.get('hipaa', {}).get('risk_level', 'Low')
    policy_name = st.session_state.get('export_policy_name', 'Internal Review')
    role = st.session_state.get('active_role', 'Analyst')
    if risk_level == 'High':
        recommended_mode = 'Research-safe Extract'
        rationale = 'High-sensitivity columns were detected, so broader sharing should use the strictest masking posture.'
    elif risk_level == 'Moderate':
        recommended_mode = 'HIPAA-style Limited Dataset'
        rationale = 'Moderate identifier exposure suggests a limited-dataset sharing posture before wider operational handoff.'
    elif standards.get('available') and standards.get('combined_readiness_score', 0) >= 60:
        recommended_mode = 'Internal Review'
        rationale = 'The dataset is in a stronger governed state, so internal review is appropriate for analyst or operational handoff.'
    else:
        recommended_mode = 'Internal Review'
        rationale = 'Use internal review first while standards and remediation checks are still maturing.'
    return pd.DataFrame([
        {
            'focus_area': 'Best sharing mode',
            'status': recommended_mode,
            'detail': rationale,
        },
        {
            'focus_area': 'Current sharing posture',
            'status': f'{role} ? {policy_name}',
            'detail': f'HIPAA-style risk is currently {risk_level.lower()}.',
        },
    ])


def _render_readmission_readiness(readmission: dict[str, Any]) -> None:
    readiness = readmission.get('readiness', {})
    missing = readiness.get('missing_fields', [])
    available = readiness.get('available_analysis', [])
    st.caption(readiness.get('badge_text', 'Readmission analytics readiness is being assessed from the current dataset.'))
    if missing:
        st.info(
            'Full readmission analytics still needs: '
            + ', '.join(str(item).replace('_', ' ') for item in missing)
            + '.'
        )
    if available:
        st.write('What can still be analyzed now:')
        for item in available:
            st.write(f'- {item}')
    extra = readiness.get('additional_fields_to_unlock_full_analysis', [])
    if extra:
        st.caption(
            'Add or map these fields to unlock the full readmission workflow: '
            + ', '.join(str(item).replace('_', ' ') for item in extra)
            + '.'
        )


def _render_readmission_overview(readmission: dict[str, Any]) -> None:
    if not readmission.get('available'):
        _render_readmission_readiness(readmission)
        st.info(readmission.get('reason', 'Readmission-focused analytics are not available for the current dataset.'))
        return

    overview = readmission.get('overview', {})
    metric_row([
        ('Overall Readmission Rate', fmt(overview.get('overall_readmission_rate'), 'pct')),
        ('Readmissions in Scope', fmt(overview.get('readmission_count', 0))),
        ('Records Reviewed', fmt(overview.get('records_in_scope', 0))),
        ('Workflow Status', readmission.get('readiness', {}).get('badge_text', 'Available')),
    ])
    if readmission.get('note'):
        st.caption(str(readmission['note']))

    for title, table_key, metric_col in [
        ('Readmission by Department', 'by_department', 'readmission_rate'),
        ('Readmission by Diagnosis', 'by_diagnosis', 'readmission_rate'),
        ('Readmission by Age Band', 'by_age_band', 'readmission_rate'),
        ('Readmission by Length-of-Stay Band', 'by_los_band', 'readmission_rate'),
    ]:
        st.markdown(f'#### {title}')
        table = safe_df(readmission.get(table_key))
        info_or_table(table, f'{title} is not available for the current dataset.')
        if not table.empty:
            info_or_chart(plot_bar(table, table.columns[0], metric_col, title), f'No chart is available for {title.lower()}.')

    st.markdown('#### High-Risk Readmission Segments')
    info_or_table(safe_df(readmission.get('high_risk_segments')), 'No high-risk readmission segments were identified from the current rules.')
    st.markdown('#### Readmission Driver Analysis')
    driver_table = safe_df(readmission.get('driver_table'))
    if not driver_table.empty:
        st.write(readmission.get('driver_interpretation', ''))
        info_or_table(driver_table, 'No readmission driver table is available.')
        info_or_chart(plot_bar(driver_table, 'factor', 'gap_vs_overall', 'Readmission Driver Gaps'), 'No readmission driver chart is available.')
    else:
        st.info(readmission.get('driver_interpretation', 'Readmission driver analysis is limited for the current dataset.'))
    st.markdown('#### High-Risk Patients or Rows')
    info_or_table(safe_df(readmission.get('high_risk_patients')), 'No high-risk patient or encounter rows are available for readmission review.')


def _build_export_workflow_presets(pipeline: dict[str, Any], role: str) -> list[dict[str, str]]:
    interop = pipeline['standards'].get('interoperability_report', {})
    ehr_patterns = safe_df(interop.get('ehr_export_patterns'))
    actions: list[dict[str, str]] = []
    if ehr_patterns.empty or 'export_pattern' not in ehr_patterns.columns:
        return actions
    detected = set(ehr_patterns['export_pattern'].astype(str))
    if 'Epic-like export' in detected:
        actions.append({
            'label': 'Apply Epic-oriented export preset',
            'action': 'epic_export_preset',
            'why': 'Epic-like encounter and result patterns were detected, so an operational review bundle is likely to be the best first handoff.',
        })
    if 'Cerner-like export' in detected:
        actions.append({
            'label': 'Apply Cerner-oriented export preset',
            'action': 'cerner_export_preset',
            'why': 'Cerner-like event and catalog patterns were detected, so a more analyst-oriented handoff is likely to be useful.',
        })
    cdisc = pipeline['standards'].get('cdisc_report', {})
    if cdisc.get('available') and str(cdisc.get('likely_dataset_type', '')).startswith('CDISC'):
        actions.append({
            'label': 'Apply CDISC-oriented export preset',
            'action': 'cdisc_export_preset',
            'why': 'Trial-style CDISC signals were detected, so an analyst-oriented export bundle with internal review posture is a safer first handoff.',
        })
    hipaa_risk = pipeline['privacy_review'].get('hipaa', {}).get('risk_level', 'Low')
    if hipaa_risk in {'Moderate', 'High'}:
        actions.append({
            'label': 'Apply HIPAA-style sharing preset',
            'action': 'hipaa_sharing_preset',
            'why': 'Elevated identifier risk was detected, so a limited-dataset posture is a safer operational export starting point.',
        })
    if hipaa_risk == 'High':
        actions.append({
            'label': 'Apply research-safe sharing preset',
            'action': 'research_sharing_preset',
            'why': 'High identifier sensitivity suggests the strongest masking posture before broader review or research-oriented sharing.',
        })
    if role in {'Researcher', 'Viewer'} and actions:
        for action in actions:
            action['why'] += ' The current role will still keep the configured policy-aware protections in place.'
    return actions


def _run_export_workflow_preset(action_name: str) -> str:
    st.session_state['export_policy_name'] = 'Internal Review'
    if action_name == 'epic_export_preset':
        st.session_state['report_mode'] = 'Operational Report'
        return 'Applied an Epic-oriented export preset with an operational report focus.'
    if action_name == 'cerner_export_preset':
        st.session_state['report_mode'] = 'Analyst Report'
        return 'Applied a Cerner-oriented export preset with an analyst report focus.'
    if action_name == 'cdisc_export_preset':
        st.session_state['report_mode'] = 'Analyst Report'
        return 'Applied a CDISC-oriented export preset with an analyst report focus.'
    if action_name == 'hipaa_sharing_preset':
        st.session_state['export_policy_name'] = 'HIPAA-style Limited Dataset'
        return 'Applied a HIPAA-style sharing preset with limited-dataset export protections.'
    if action_name == 'research_sharing_preset':
        st.session_state['export_policy_name'] = 'Research-safe Extract'
        return 'Applied a research-safe sharing preset with stronger masking protections.'
    return 'No export preset was applied.'


def _build_export_workflow_preset_table(actions: list[dict[str, str]]) -> pd.DataFrame:
    if not actions:
        return pd.DataFrame(columns=['preset', 'why_this_is_recommended'])
    return pd.DataFrame([
        {'preset': action['label'], 'why_this_is_recommended': action['why']}
        for action in actions
    ])


def render_export_center(pipeline: dict[str, Any], dataset_name: str, source_meta: dict[str, str]) -> None:
    st.subheader('Export Center')
    role = st.selectbox('Role', ROLE_OPTIONS, index=ROLE_OPTIONS.index(st.session_state.get('active_role', 'Analyst')), key='active_role')
    policies = get_export_policy_presets()
    policy_name = st.selectbox('Export policy', policies['policy_name'].tolist(), index=policies['policy_name'].tolist().index(st.session_state.get('export_policy_name', 'Internal Review')), key='export_policy_name')
    report_mode = st.selectbox('Report mode', REPORT_MODES, index=REPORT_MODES.index(normalize_report_mode(st.session_state.get('report_mode', 'Executive Summary'))), key='report_mode')
    privacy = pipeline['privacy_review']
    policy_eval = evaluate_export_policy(policy_name, privacy)
    export_allowed = can_access(role, 'exports')
    metric_row([
        ('Export Role', role),
        ('Policy Readiness', policy_eval.get('sharing_readiness', 'Internal only')),
        ('Redaction Level', policy_eval.get('redaction_level', 'Low')),
        ('Exports Enabled', 'Yes' if export_allowed else 'Limited'),
    ])

    text_report = build_text_report(dataset_name, pipeline['overview'], pipeline['structure'], pipeline['field_profile'], pipeline['quality'], pipeline['semantic'], pipeline['readiness'], pipeline['healthcare'], pipeline['insights'])
    executive_report = build_executive_summary_text(dataset_name, pipeline['overview'], pipeline['healthcare'], pipeline['insights'])
    readmission_report = build_readmission_summary_text(dataset_name, pipeline['healthcare'].get('readmission', {}), safe_df(pipeline.get('action_recommendations')))
    audience_report = build_audience_report_text(report_mode, dataset_name, pipeline['overview'], pipeline['structure'], pipeline['quality'], pipeline['semantic'], pipeline['readiness'], pipeline['healthcare'], pipeline['insights'], pipeline['action_recommendations'])
    support_csv = build_report_support_csv(report_mode, pipeline['overview'], pipeline['quality'], pipeline['readiness'], pipeline['healthcare'], pipeline['action_recommendations'])
    compliance_summary = build_compliance_summary_text(dataset_name, pipeline['standards'], privacy, role=role)
    compliance_csv = build_compliance_support_csv(pipeline['standards'], privacy, role=role)
    compliance_payload = build_compliance_dashboard_payload(pipeline['standards'], privacy, role=role, policy_name=policy_name)
    audit_log = build_audit_log_view(st.session_state.get('analysis_log', []))
    governance_payload = build_governance_review_payload(dataset_name, source_meta, pipeline['lineage'], audit_log, pipeline['standards'], privacy)
    governance_text = build_governance_review_text(dataset_name, source_meta, pipeline['lineage'], audit_log, pipeline['standards'], privacy, role=role, policy_name=policy_name)
    governance_csv = build_governance_review_csv(pipeline['lineage'], audit_log, pipeline['standards'], privacy, role=role, policy_name=policy_name)
    policy_note = build_policy_note_text(policy_name, role, privacy)
    compliance_handoff = build_compliance_handoff_payload(dataset_name, pipeline['standards'], privacy, pipeline['lineage'], role, policy_name)
    bundle_text = build_role_export_bundle_text(role, policy_name, export_allowed, report_mode, privacy)
    bundle_manifest = build_role_export_bundle_manifest(role, policy_name, export_allowed, report_mode, privacy)
    bundle_title, bundle_table = build_policy_aware_bundle_profile(role, report_mode, policy_name, export_allowed, privacy)
    export_presets = _build_export_workflow_presets(pipeline, role)
    remediation = pipeline.get('remediation_context', {})

    def protect(text_bytes: bytes) -> bytes:
        return apply_role_based_redaction(apply_export_policy(policy_note + b'\n\n' + text_bytes, policy_name, privacy), role, privacy)

    st.markdown('### Recommended export bundle')
    st.write(f"Recommended report for **{role}**: **{recommended_report_mode_for_role(role)}**")
    info_or_table(bundle_manifest, 'No bundle manifest is available yet.')
    st.download_button('Download bundle manifest CSV', data=dataframe_to_csv_bytes(bundle_manifest), file_name='role_export_bundle_manifest.csv', mime='text/csv', disabled=not export_allowed)
    st.download_button('Download bundle guide TXT', data=protect(bundle_text), file_name='role_export_bundle_guide.txt', mime='text/plain', disabled=not export_allowed)
    st.markdown('### Policy-aware bundle recommendation')
    st.write(bundle_title)
    info_or_table(bundle_table, 'No policy-aware bundle recommendation is available yet.')

    if export_presets:
        st.markdown('### EHR export presets')
        preset_cols = st.columns(min(2, len(export_presets)))
        for idx, action in enumerate(export_presets):
            if preset_cols[idx % len(preset_cols)].button(action['label'], key=f"export_preset_{action['action']}"):
                message = _run_export_workflow_preset(action['action'])
                log_event('Export Preset Applied', message, 'Export workflow preset', 'Export center')
                st.success(message)
                st.rerun()
        info_or_table(_build_export_workflow_preset_table(export_presets), 'No EHR export preset guidance is available yet.')

    st.markdown('### Report exports')
    executive_summary = pipeline.get('executive_summary', {})
    if executive_summary:
        st.caption('Executive summary preview')
        for bullet in executive_summary.get('stakeholder_summary_bullets', [])[:3]:
            st.write(f'- {bullet}')
    executive_pack = pipeline.get('executive_report_pack', {})
    printable = pipeline.get('printable_reports', {})
    stakeholder_bundle = pipeline.get('stakeholder_export_bundle', {})
    st.markdown('### Executive Report Pack')
    if executive_pack:
        info_or_table(
            pd.DataFrame([{'section': key, 'summary': value} for key, value in executive_pack.get('executive_report_sections', {}).items()]),
            'Executive report pack is not available yet.',
        )
        st.download_button('Download executive report pack TXT', data=protect(executive_pack.get('executive_report_text', '').encode('utf-8')), file_name='executive_report_pack.txt', mime='text/plain', disabled=not export_allowed)
        st.download_button('Download executive report pack Markdown', data=protect(executive_pack.get('executive_report_markdown', '').encode('utf-8')), file_name='executive_report_pack.md', mime='text/markdown', disabled=not export_allowed)
    st.markdown('### Print-Friendly Reports')
    if printable:
        st.download_button('Download print-friendly executive report', data=protect(printable.get('printable_executive_report', '').encode('utf-8')), file_name='printable_executive_report.txt', mime='text/plain', disabled=not export_allowed)
        st.download_button('Download print-friendly compliance summary', data=protect(printable.get('printable_compliance_summary', '').encode('utf-8')), file_name='printable_compliance_summary.txt', mime='text/plain', disabled=not export_allowed)
    st.markdown('### Stakeholder Export Bundle')
    info_or_table(safe_df(stakeholder_bundle.get('export_bundle_manifest')), 'No stakeholder export bundle is available yet.')
    for note in stakeholder_bundle.get('export_bundle_notes', []):
        st.caption(note)
    if not safe_df(stakeholder_bundle.get('export_bundle_manifest')).empty:
        st.download_button('Download stakeholder bundle manifest CSV', data=dataframe_to_csv_bytes(safe_df(stakeholder_bundle.get('export_bundle_manifest'))), file_name='stakeholder_export_bundle_manifest.csv', mime='text/csv', disabled=not export_allowed)
    helper_export_notes = []
    if remediation.get('bmi_remediation', {}).get('available'):
        helper_export_notes.append('BMI values were remediated before downstream analysis and reporting.')
    if remediation.get('synthetic_cost', {}).get('available'):
        helper_export_notes.append('Synthetic estimated cost is included as a demo-only support field where financial analytics required it.')
    if remediation.get('synthetic_clinical', {}).get('available'):
        helper_export_notes.append('Derived diagnosis and clinical-risk labels support blocked clinical modules in demo mode.')
    if remediation.get('synthetic_readmission', {}).get('available'):
        helper_export_notes.append('Synthetic readmission support is included for workflow demonstrations when native readmission fields are absent.')
    for note in helper_export_notes:
        st.caption(note)
    st.download_button('Download summary report', data=protect(text_report), file_name='smart_dataset_summary.txt', mime='text/plain', disabled=not export_allowed)
    st.download_button('Download executive summary', data=protect(executive_report), file_name='executive_summary.txt', mime='text/plain', disabled=not export_allowed)
    st.download_button('Download readmission summary', data=protect(readmission_report), file_name='readmission_summary.txt', mime='text/plain', disabled=not export_allowed)
    st.download_button('Download audience report', data=protect(audience_report), file_name=f"{normalize_report_mode(report_mode).replace(' ', '_').lower()}.txt", mime='text/plain', disabled=not export_allowed)
    st.download_button('Download support tables CSV', data=support_csv, file_name='report_support_tables.csv', mime='text/csv', disabled=not export_allowed)

    st.markdown('### Compliance dashboard pack')
    st.download_button('Download compliance summary', data=protect(compliance_summary), file_name='compliance_summary.txt', mime='text/plain', disabled=not export_allowed)
    st.download_button('Download compliance review CSV', data=compliance_csv, file_name='compliance_review.csv', mime='text/csv', disabled=not export_allowed)
    st.download_button('Download compliance dashboard JSON', data=json_bytes(compliance_payload), file_name='compliance_dashboard.json', mime='application/json', disabled=not export_allowed)
    st.download_button('Download compliance dashboard CSV', data=build_compliance_dashboard_csv(pipeline['standards'], privacy, role=role), file_name='compliance_dashboard.csv', mime='text/csv', disabled=not export_allowed)
    st.download_button('Download compliance handoff JSON', data=json_bytes(compliance_handoff), file_name='compliance_handoff.json', mime='application/json', disabled=not export_allowed)

    st.markdown('### Governance and audit packet')
    st.download_button('Download governance review TXT', data=protect(governance_text), file_name='governance_review.txt', mime='text/plain', disabled=not export_allowed)
    st.download_button('Download governance review CSV', data=governance_csv, file_name='governance_review.csv', mime='text/csv', disabled=not export_allowed)
    st.download_button('Download governance review JSON', data=json_bytes(governance_payload), file_name='governance_review.json', mime='application/json', disabled=not export_allowed)

    docs = pipeline.get('documentation_support', {})
    screenshots = pipeline.get('screenshot_support', {})
    st.markdown('### Documentation and Portfolio Support')
    if docs:
        with st.expander('README-ready project summary'):
            for key, value in docs.get('readme_sections', {}).items():
                st.write(f"**{key.replace('_', ' ').title()}**")
                if isinstance(value, list):
                    for item in value:
                        st.write(f'- {item}')
                else:
                    st.write(str(value))
        st.download_button('Download portfolio project summary', data=str(docs.get('portfolio_project_summary', '')).encode('utf-8'), file_name='portfolio_project_summary.txt', mime='text/plain', disabled=not export_allowed)
        st.download_button('Download demo walkthrough', data=str(docs.get('demo_walkthrough_text', '')).encode('utf-8'), file_name='demo_walkthrough.txt', mime='text/plain', disabled=not export_allowed)
    if screenshots:
        with st.expander('Screenshot plan and recruiter callouts'):
            info_or_table(safe_df(screenshots.get('screenshot_plan')), 'No screenshot plan is available yet.')
            for line in screenshots.get('recruiter_callouts', []):
                st.write(f'- {line}')


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout='wide')
    apply_theme()
    init_state()
    st.sidebar.title(APP_TITLE)
    source_mode = st.sidebar.radio('Dataset source', ['Built-in example dataset', 'Uploaded dataset'], key='dataset_source_mode')
    st.sidebar.caption('This demo supports general tabular datasets and activates healthcare, standards, privacy, and governance modules only when the data can support them safely.')
    with st.sidebar.expander('Demo Configuration'):
        st.selectbox('Synthetic helper fields', ['Auto', 'On', 'Off'], key='demo_synthetic_helper_mode')
        st.selectbox('BMI remediation mode', ['median', 'clip', 'null', 'none'], key='demo_bmi_remediation_mode')
        st.selectbox('Synthetic cost support', ['Auto', 'On', 'Off'], key='demo_synthetic_cost_mode')
        st.selectbox('Synthetic readmission support', ['Auto', 'On', 'Off'], key='demo_synthetic_readmission_mode')
        st.selectbox('Executive summary verbosity', ['Concise', 'Detailed'], key='demo_executive_summary_verbosity')
        st.selectbox('Scenario simulation mode', ['Basic', 'Expanded'], key='demo_scenario_simulation_mode')
        st.caption('These settings control demo-safe helper behavior and summary presentation without changing the source data.')
    data, original_lookup, dataset_name, source_meta = load_primary_dataset(source_mode)
    if data is None:
        return
    preflight = build_preflight_guardrails(source_meta, estimate_memory_mb(data), len(data), len(data.columns))
    if preflight.get('blocked'):
        st.error(preflight.get('block_reason', 'This dataset exceeds the safe interactive limits for the current demo environment.'))
        info_or_table(safe_df(preflight.get('checks_table')), 'No preflight details are available.')
        st.info('Try a smaller extract, fewer columns, or a narrower date range before re-running the analysis.')
        return
    log_event('Dataset Selected', f"Selected dataset '{dataset_name}'.", 'Dataset selection', 'Data intake')
    progress_placeholder = st.empty()
    progress_bar = st.progress(0, text='Preparing analysis context...')
    demo_config = get_demo_config()

    def _progress_update(value: float, message: str) -> None:
        progress_bar.progress(int(max(0.0, min(1.0, value)) * 100), text=message)
        progress_placeholder.caption(message)

    try:
        pipeline = run_pipeline(data, dataset_name, source_meta, demo_config=demo_config, progress_callback=_progress_update)
    except Exception as error:
        progress_bar.empty()
        progress_placeholder.empty()
        st.error('The platform could not complete a full analysis for this dataset. Review the production hardening guidance below and try a smaller or cleaner extract.')
        info_or_table(safe_df(preflight.get('checks_table')), 'No preflight details are available.')
        st.info(f'Technical summary: {type(error).__name__}: {error}')
        return
    progress_bar.empty()
    progress_placeholder.caption('Analysis context is ready. Use the tabs below to explore the current dataset.')
    pipeline['preflight'] = preflight
    pipeline['deployment_health_checks'] = build_deployment_health_checks(pipeline, source_meta)
    pipeline['performance_diagnostics'] = build_performance_diagnostics(pipeline['overview'], pipeline['sample_info'], source_meta)
    run_entry = build_run_history_entry(dataset_name, pipeline, demo_config)
    st.session_state['run_history'] = update_run_history(st.session_state.get('run_history', []), run_entry)
    pipeline['audit_summary_bundle'] = build_audit_summary(st.session_state.get('run_history', []), st.session_state.get('analysis_log', []))
    landing = build_landing_summary(pipeline, demo_config, dataset_name)
    pipeline['landing_summary'] = landing

    st.title(APP_TITLE)
    st.caption(landing.get('subheadline', 'Schema-flexible healthcare and general analytics with readiness, compliance, governance, and decision-support workflows.'))
    st.markdown('### What this platform covers')
    metric_row([(badge['label'], badge['value']) for badge in landing.get('capability_badges', [])[:4]])
    for line in landing.get('analysis_covers', []):
        st.write(f'- {line}')
    st.markdown('### Current system status')
    for line in landing.get('system_status', []):
        st.write(f'- {line}')
    st.caption(landing.get('platform_value_summary', ''))
    st.caption(landing.get('synthetic_support_note', ''))
    metric_row([
        ('Rows', fmt(pipeline['overview']['rows'])),
        ('Columns', fmt(pipeline['overview']['columns'])),
        ('Data Quality Score', fmt(pipeline['quality']['quality_score'])),
        ('Analysis Readiness', fmt(pipeline['readiness']['readiness_score'], 'score')),
    ])

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


if __name__ == '__main__':
    main()


