from __future__ import annotations

import time
from typing import Any

import pandas as pd
import streamlit as st

import src.auth as auth_module
from src.ai_copilot import build_copilot_panel_config, initialize_copilot_memory
from src.audience_modes import build_audience_mode_guidance
from src.collaboration_notes import build_collaboration_notes_view
from src.decision_support import build_executive_summary, build_intervention_recommendations, build_prioritized_insights, build_scenario_simulation_studio
from src.enterprise_features import build_audit_log_view
from src.export_utils import (
    apply_export_policy,
    apply_role_based_redaction,
    build_audience_report_text,
    build_compliance_dashboard_csv,
    build_compliance_dashboard_payload,
    build_compliance_handoff_payload,
    build_compliance_summary_text,
    build_compliance_support_csv,
    build_cross_setting_reporting_profile,
    build_executive_summary_text,
    build_generated_report_text,
    build_governance_review_csv,
    build_governance_review_payload,
    build_governance_review_text,
    build_policy_note_text,
    build_readmission_summary_text,
    build_report_support_csv,
    build_shared_report_bundles,
    build_shared_report_bundle_text,
    build_text_report,
    dataframe_to_csv_bytes,
    json_bytes,
    normalize_report_mode,
    recommended_report_mode_for_role,
)
from src.healthcare_analysis import operational_alerts
from src.jobs import build_job_status_view, build_job_user_message, get_job_result, get_job_status, submit_job
from src.modules.privacy_security import apply_export_watermark, build_export_governance_summary, evaluate_export_policy, get_export_policy_presets
from src.modules.rbac import can_access
from src.ops_hardening import build_export_safety_note
from src.plan_awareness import plan_feature_enabled
from src.services.copilot_service import execute_copilot_prompt, plan_copilot_workflow
from src.services.export_service import generate_export_report_output, prepare_policy_aware_export_bundle, record_export_bundle_metadata_once
from src.ui_components import metric_row, render_advanced_sections_toggle, render_badge_row, render_role_context_panel, render_section_intro, render_surface_panel, render_subsection_header, render_workflow_steps

from ui.common import info_or_chart, info_or_table, log_event, safe_df, tracked_download_button
from ui.healthcare_analytics import render_automated_insight_board

ROLE_OPTIONS = ['Admin', 'Analyst', 'Executive', 'Clinician', 'Researcher', 'Data Steward', 'Viewer']
REPORT_MODES = ['Analyst Report', 'Operational Report', 'Executive Summary', 'Clinical Report', 'Data Readiness Review', 'Population Health Summary']


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

def render_key_insights(pipeline: dict[str, Any], dataset_name: str) -> None:
    render_section_intro(
        'Key Insights',
        'Summarize priority findings, operational implications, and guided Copilot responses in a cleaner stakeholder-ready insight layer.',
    )
    render_badge_row(
        [
            ('Stakeholder-ready', 'info'),
            ('Explainable', 'accent'),
            ('Governed', 'success'),
        ]
    )
    render_surface_panel(
        'Insight workflow',
        'Use this surface to review prioritized findings, intervention signals, explainability notes, and governed narrative output for the active dataset.',
    )
    render_workflow_steps(
        [
            ('Prioritize', 'Review critical findings, remediation opportunities, and watchlist items.'),
            ('Explain', 'Assess drivers, fairness signals, and operational alert context.'),
            ('Act', 'Translate quality and clinical patterns into recommendations and summaries.'),
            ('Share', 'Move approved narratives and bundles into governed export workflows.'),
        ]
    )
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
        with st.expander('Watchlist and remediation opportunities', expanded=False):
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
        with st.expander('Benchmark details and signals', expanded=False):
            info_or_table(safe_df(benchmarking.get('benchmark_table')), 'Internal benchmark comparisons appear here when the dataset supports cohort-relative KPI review.')
            for signal in benchmarking.get('standout_positive_signals', []):
                st.write(f'- Positive signal: {signal}')
            for signal in benchmarking.get('standout_risk_signals', []):
                st.write(f'- Risk signal: {signal}')
    else:
        st.info(benchmarking.get('reason', 'This feature unlocks when healthcare-specific fields are detected for cohort-relative KPI benchmarking.'))
    scenarios = pipeline.get('scenario_studio', {})
    st.markdown('### Scenario Simulation Studio')
    if scenarios.get('available'):
        st.caption(scenarios.get('summary', 'Directional scenarios are available for stakeholder review.'))
        info_or_table(safe_df(scenarios.get('scenario_table')), 'No scenario table is available.')
    else:
        st.info(scenarios.get('reason', 'This feature unlocks when the dataset supports directional scenario simulation with enough operational or clinical signal.'))
    st.markdown('### Explainability & Fairness')
    fairness = pipeline['healthcare'].get('explainability_fairness', {})
    if fairness.get('available'):
        if fairness.get('high_risk_segment_explanation'):
            st.write(fairness['high_risk_segment_explanation'])
        with st.expander('Explainability and fairness details', expanded=False):
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
    copilot_style = str(st.session_state.get('product_copilot_response_style', 'Concise'))
    show_workflow_preview = bool(st.session_state.get('product_copilot_show_workflow_preview', True))
    copilot_panel = build_copilot_panel_config(
        pipeline['data'],
        {'matched_schema': pipeline['semantic']['canonical_map'], 'dataset_name': dataset_name},
    )
    st.caption(f"{copilot_panel.get('title', 'AI Copilot Panel')} | {copilot_panel.get('mode_label', 'Rule-based fallback')}")
    st.caption(str(copilot_panel.get('mode_note', '')))
    messages = initialize_copilot_memory()
    suggested_prompts = list(copilot_panel.get('suggested_prompts', []))
    queued_follow_up = str(st.session_state.pop('copilot_follow_up_prompt', '')).strip()
    if suggested_prompts:
        st.write('**Suggested questions**')
        suggestion_cols = st.columns(min(3, len(suggested_prompts)))
        selected_prompt = None
        for idx, prompt_text in enumerate(suggested_prompts):
            if suggestion_cols[idx % len(suggestion_cols)].button(prompt_text, key=f'copilot_suggested_{idx}'):
                selected_prompt = prompt_text
    else:
        selected_prompt = None
    prompt_cols = st.columns([4, 1])
    manual_prompt = prompt_cols[0].text_input(
        'Ask the AI Copilot',
        key='copilot_prompt_input',
        placeholder='Summarize the dataset, ask about high-risk patients, or request key healthcare insights.',
    )
    ask_clicked = prompt_cols[1].button('Ask Copilot', key='run_copilot_prompt')
    prompt = queued_follow_up or selected_prompt or (manual_prompt.strip() if ask_clicked and manual_prompt.strip() else '')
    latest_response = None
    if prompt:
        log_event('AI Copilot Question', f'Asked: {prompt}', 'AI copilot', 'Insight exploration')
        with st.spinner('Generating AI copilot response...'):
            latest_response = execute_copilot_prompt(
                prompt,
                data=pipeline['data'],
                schema_context={'matched_schema': pipeline['semantic']['canonical_map'], 'dataset_name': dataset_name},
            )
        messages = latest_response.get('messages', initialize_copilot_memory())
    assistant_messages = [message for message in messages if str(message.get('role', '')).lower() == 'assistant']
    latest_assistant = assistant_messages[-1] if assistant_messages else None
    st.write('**Copilot response**')
    response_source = latest_response or latest_assistant
    if response_source:
        st.info(str(response_source.get('content', response_source.get('answer', ''))))
        if response_source.get('llm_provider'):
            provider_caption = f"LLM provider: {response_source.get('llm_provider')}"
            if response_source.get('cached_response'):
                provider_caption += ' | cached response'
            st.caption(provider_caption)
        elif response_source.get('cached_response'):
            st.caption('Cached deterministic response')
        explanation = str(response_source.get('explanation', '')).strip()
        if explanation:
            st.caption(explanation)
        response_table = response_source.get('table_data')
        if isinstance(response_table, pd.DataFrame) and not response_table.empty:
            st.dataframe(response_table, width='stretch')
        response_chart = response_source.get('chart_figure')
        if response_chart is not None:
            st.plotly_chart(response_chart, width='stretch')
        citations = response_source.get('citations', [])
        if citations:
            with st.expander('Citations and grounding'):
                for citation in citations:
                    st.write(f"**{citation.get('title', 'Grounding detail')}**")
                    st.write(str(citation.get('detail', '')))
        follow_ups = response_source.get('follow_ups', [])
        if follow_ups:
            st.caption('Suggested follow-up questions:')
            follow_up_cols = st.columns(min(3, len(follow_ups)))
            for idx, follow_up in enumerate(follow_ups[:3]):
                if follow_up_cols[idx % len(follow_up_cols)].button(follow_up, key=f'copilot_follow_up_{idx}'):
                    st.session_state['copilot_follow_up_prompt'] = follow_up
                    st.rerun()
    else:
        if copilot_style == 'Detailed':
            st.info('Ask about dataset summary, average cost, average length of stay, readmission by department, highest readmission risk, readmission drivers, top diagnosis by cost, or use workflow actions to prepare exports and focused reviews.')
        else:
            st.info('Ask about dataset summary, average cost, average length of stay, readmission by department, highest readmission risk, readmission drivers, or the top diagnosis by cost.')
    with st.expander('Conversation history'):
        for message in messages:
            role = str(message.get('role', 'assistant')).title()
            st.write(f"**{role}:** {message.get('content', '')}")
    st.markdown('### AI Copilot Workflow Actions')
    workflow_prompt = st.text_input('Ask the copilot to prepare a workflow action', key='workflow_action_prompt')
    result = plan_copilot_workflow(
        workflow_prompt,
        data=pipeline['data'],
        canonical_map=pipeline['semantic']['canonical_map'],
        readiness=pipeline['readiness'],
        healthcare=pipeline['healthcare'],
        remediation=pipeline['remediation'],
    )
    planned_action = result.get('planned_action')
    if planned_action:
        st.caption(f'Planned action: {planned_action}')
    st.write(result.get('message', ''))
    if result.get('recommended_section'):
        st.caption(f"Recommended section: {result['recommended_section']}")
    preview_table = result.get('preview_table')
    if show_workflow_preview and isinstance(preview_table, pd.DataFrame) and not preview_table.empty:
        info_or_table(preview_table, 'No workflow preview is available yet.')
    preview_chart = result.get('preview_chart')
    if show_workflow_preview and preview_chart is not None:
        info_or_chart(preview_chart, 'No workflow preview chart is available yet.')
    elif not show_workflow_preview and (isinstance(preview_table, pd.DataFrame) and not preview_table.empty or preview_chart is not None):
        st.caption('Workflow previews are hidden by the current product settings. Turn them back on in Product Settings / Admin Panel if you want richer Copilot previews.')
    prompts = result.get('suggested_prompts', [])
    if prompts:
        st.caption('Suggested follow-up prompts:')
        prompt_limit = len(prompts) if copilot_style == 'Detailed' else min(3, len(prompts))
        for prompt_text in prompts[:prompt_limit]:
            st.write(f'- {prompt_text}')
    if result.get('widget_updates'):
        info_or_table(pd.DataFrame([{'setting': k, 'value': v} for k, v in result['widget_updates'].items()]), 'No workflow updates are pending.')
        if st.button('Apply workflow guidance', key='apply_workflow_guidance'):
            for key, value in result['widget_updates'].items():
                st.session_state[key] = value
            log_event('Workflow Action', result.get('message', 'Applied workflow guidance.'), 'Workflow copilot', 'Guided workflow')
            st.rerun()

def render_export_center(pipeline: dict[str, Any], dataset_name: str, source_meta: dict[str, str]) -> None:
    render_section_intro(
        'Export Center',
        'Package Clinverity outputs for analyst, operational, executive, and governed audit handoff workflows without losing policy context or remediation disclosures.',
    )
    render_badge_row(
        [
            ('Policy-aware', 'info'),
            ('Audit-ready', 'accent'),
            ('Role-governed', 'success'),
        ]
    )
    render_surface_panel(
        'Export workflow guidance',
        'Choose the smallest approved handoff that fits the audience: one primary report, the necessary support tables, and governance material only when the review requires it.',
    )
    advanced_export_sections_enabled = render_advanced_sections_toggle(
        'export_center',
        help_text='Expand bundle, governance, and portfolio export sections by default for admin or steward review.',
    )
    export_view_role = str(st.session_state.get('active_role') or st.session_state.get('workspace_role') or 'Analyst')
    export_order = (
        ['strategy', 'governance_support', 'report_generation', 'supporting_bundles']
        if export_view_role in {'Admin', 'Data Steward', 'Owner'}
        else ['strategy', 'report_generation', 'supporting_bundles', 'governance_support']
    )
    export_sections = {key: st.container() for key in export_order}
    plan_awareness = pipeline.get('plan_awareness', {})
    export_safety_note = build_export_safety_note(
        int(pipeline.get('overview', {}).get('rows', 0)),
        pipeline.get('sample_info', {}),
        pipeline.get('large_dataset_profile', {}),
    )
    role = st.selectbox('Role', ROLE_OPTIONS, index=ROLE_OPTIONS.index(st.session_state.get('active_role', 'Analyst')), key='active_role')
    policies = get_export_policy_presets()
    policy_name = st.selectbox('Export policy', policies['policy_name'].tolist(), index=policies['policy_name'].tolist().index(st.session_state.get('export_policy_name', 'Internal Review')), key='export_policy_name')
    report_mode = st.selectbox('Report mode', REPORT_MODES, index=REPORT_MODES.index(normalize_report_mode(st.session_state.get('report_mode', 'Executive Summary'))), key='report_mode')
    accuracy_cols = st.columns(4)
    benchmark_pack_store = st.session_state.setdefault('organization_benchmark_packs', {})
    benchmark_options = ['Auto', 'Hospital Encounters', 'Payer Claims', 'Clinical Registry', 'Generic Healthcare', *sorted(benchmark_pack_store.keys())]
    threshold_options = ['Role default', 'Conservative', 'Standard', 'Permissive']
    selected_benchmark = str(st.session_state.get('accuracy_benchmark_profile', 'Auto'))
    selected_threshold = str(st.session_state.get('accuracy_reporting_threshold_profile', 'Role default'))
    accuracy_cols[0].selectbox(
        'Benchmark profile',
        benchmark_options,
        index=benchmark_options.index(selected_benchmark) if selected_benchmark in benchmark_options else 0,
        key='accuracy_benchmark_profile',
    )
    accuracy_cols[1].selectbox(
        'Reporting threshold',
        threshold_options,
        index=threshold_options.index(selected_threshold) if selected_threshold in threshold_options else 0,
        key='accuracy_reporting_threshold_profile',
    )
    accuracy_cols[2].slider(
        'Min trust for external',
        min_value=0.50,
        max_value=0.95,
        value=float(st.session_state.get('accuracy_reporting_min_trust_score', 0.76)),
        step=0.01,
        key='accuracy_reporting_min_trust_score',
    )
    accuracy_cols[3].toggle(
        'Allow directional external',
        value=bool(st.session_state.get('accuracy_allow_directional_external_reporting', False)),
        key='accuracy_allow_directional_external_reporting',
    )
    active_pack_options = ['None', *sorted(benchmark_pack_store.keys())]
    st.selectbox(
        'Org benchmark pack',
        active_pack_options,
        index=active_pack_options.index(str(st.session_state.get('active_benchmark_pack_name', 'None'))) if str(st.session_state.get('active_benchmark_pack_name', 'None')) in active_pack_options else 0,
        key='active_benchmark_pack_name',
    )
    privacy = pipeline['privacy_review']
    workspace_identity = st.session_state.get('workspace_identity', {})
    governance_config = {
        'redaction_level': st.session_state.get('workspace_governance_redaction_level', 'Medium'),
        'workspace_export_access': st.session_state.get('workspace_governance_export_access', 'Editors and owners'),
        'watermark_sensitive_exports': bool(st.session_state.get('workspace_governance_watermark_sensitive_exports', True)),
    }
    policy_eval = evaluate_export_policy(policy_name, privacy, workspace_identity, governance_config)
    governance_summary = build_export_governance_summary(policy_name, privacy, workspace_identity, governance_config)
    workspace_export_allowed = auth_module.workspace_identity_can_access(workspace_identity, 'export_download') and bool(policy_eval.get('workspace_export_allowed', True))
    export_allowed = can_access(role, 'exports') and workspace_export_allowed
    strict_plan = bool(plan_awareness.get('strict_enforcement'))
    advanced_exports_allowed = export_allowed and (plan_feature_enabled(str(plan_awareness.get('active_plan', 'Pro')), 'advanced_exports') or not strict_plan)
    governance_exports_allowed = export_allowed and (plan_feature_enabled(str(plan_awareness.get('active_plan', 'Pro')), 'governance_exports') or not strict_plan)
    stakeholder_bundle_allowed = export_allowed and (plan_feature_enabled(str(plan_awareness.get('active_plan', 'Pro')), 'stakeholder_bundle') or not strict_plan)
    metric_row([
        ('Export Role', role),
        ('Plan', str(plan_awareness.get('active_plan', 'Pro'))),
        ('Policy Readiness', policy_eval.get('sharing_readiness', 'Internal only')),
        ('Redaction Level', policy_eval.get('redaction_level', 'Low')),
        ('Exports Enabled', 'Yes' if export_allowed else 'Limited'),
    ])
    trust_summary = pipeline.get('analysis_trust_summary', {})
    accuracy_summary = pipeline.get('result_accuracy_summary', {})
    dataset_identifier = str(
        pipeline.get('dataset_runtime_diagnostics', {}).get('dataset_identifier', '')
        or pipeline.get('source_meta', {}).get('dataset_identifier', '')
        or dataset_name
    )
    application_service = st.session_state.get('application_service')
    render_role_context_panel(
        export_view_role,
        primary_message=(
            'Analyst and executive views prioritize report generation and concise handoff guidance. '
            'Admin and data-steward views prioritize governance, policy, and compliance material earlier.'
        ),
        advanced_enabled=advanced_export_sections_enabled,
        advanced_label='Governance-heavy export sections',
    )
    if not workspace_export_allowed:
        st.info(
            'The active workspace export policy restricts this action for the current signed-in role. '
            'Guest/demo mode stays export-friendly, while signed-in workspaces now respect workspace-level export access controls.'
        )
    st.caption('Use this export surface to turn the current review into a stakeholder-ready handoff, not just a raw file download.')
    st.caption(str(policy_eval.get('watermark_label', '')))
    if strict_plan and not plan_feature_enabled(str(plan_awareness.get('active_plan', 'Pro')), 'advanced_exports'):
        st.info('Advanced export bundles are limited by the active plan while strict enforcement is enabled. Standard summary exports remain available.')
    elif not plan_feature_enabled(str(plan_awareness.get('active_plan', 'Pro')), 'advanced_exports'):
        st.caption('Advanced export bundles are packaged as a premium capability. They remain visible here in demo-safe mode so the export flow can still be reviewed.')
    if export_safety_note:
        st.caption(export_safety_note)
    if trust_summary.get('available'):
        render_subsection_header('Result trust disclosure')
        metric_row([
            ('Trust Level', str(trust_summary.get('trust_level', 'Unknown'))),
            ('Trust Score', f"{float(trust_summary.get('trust_score', 0.0)):.0%}"),
            ('Sampling Mode', 'Sampled' if trust_summary.get('sampling_active') else 'Full'),
            ('Interpretation', str(trust_summary.get('interpretation_mode', 'Not available'))),
        ])
        metric_row([
            ('Benchmark Profile', str(accuracy_summary.get('benchmark_profile', {}).get('profile_name', 'Auto'))),
            ('Reporting Threshold', str(accuracy_summary.get('reporting_policy', {}).get('profile_name', 'Role default'))),
            ('Min Trust Gate', f"{float(accuracy_summary.get('reporting_policy', {}).get('minimum_trust_score', 0.0)):.0%}"),
            ('Directional External', 'Allowed' if accuracy_summary.get('reporting_policy', {}).get('allow_directional_external_reporting') else 'Blocked'),
        ])
        st.caption(str(trust_summary.get('summary_text', '')))
        for note in accuracy_summary.get('uncertainty_narrative', {}).get('notes', [])[:3]:
            st.caption(note)
        with st.expander('Trust guidance for exports', expanded=False):
            info_or_table(
                safe_df(trust_summary.get('disclosure_table')),
                'Trust disclosures are not available for the current export workflow.',
            )
            for item in trust_summary.get('recommended_uses', []):
                st.write(f'- Recommended: {item}')
            for item in trust_summary.get('restricted_uses', []):
                st.write(f'- Caution: {item}')
            info_or_table(
                safe_df(accuracy_summary.get('module_reporting_gates')),
                'Module-level export/reporting gates are not available for the current dataset.',
            )
            info_or_table(
                safe_df(accuracy_summary.get('benchmark_calibration_table')),
                'Benchmark calibration checks are not available for the current export workflow.',
            )
            info_or_table(
                safe_df(accuracy_summary.get('metric_lineage_table')),
                'Metric-level lineage will appear here when result accuracy scoring has mapped support types and driving fields.',
            )

    review_store = st.session_state.setdefault('dataset_review_approvals', {})
    current_review = review_store.get(dataset_identifier, {})
    render_subsection_header('Review approval workflow')
    review_cols = st.columns(4)
    mapping_status = review_cols[0].selectbox(
        'Mapping review',
        ['Pending', 'Needs changes', 'Approved'],
        index=['Pending', 'Needs changes', 'Approved'].index(str(current_review.get('mapping_status', 'Pending'))) if str(current_review.get('mapping_status', 'Pending')) in ['Pending', 'Needs changes', 'Approved'] else 0,
        key=f'review_mapping::{dataset_identifier}',
    )
    trust_status = review_cols[1].selectbox(
        'Trust gate',
        ['Pending', 'Directional only', 'Approved'],
        index=['Pending', 'Directional only', 'Approved'].index(str(current_review.get('trust_gate_status', 'Pending'))) if str(current_review.get('trust_gate_status', 'Pending')) in ['Pending', 'Directional only', 'Approved'] else 0,
        key=f'review_trust::{dataset_identifier}',
    )
    export_status = review_cols[2].selectbox(
        'Export eligibility',
        ['Pending', 'Internal only', 'Approved'],
        index=['Pending', 'Internal only', 'Approved'].index(str(current_review.get('export_eligibility_status', 'Pending'))) if str(current_review.get('export_eligibility_status', 'Pending')) in ['Pending', 'Internal only', 'Approved'] else 0,
        key=f'review_export::{dataset_identifier}',
    )
    reviewer_role = review_cols[3].selectbox(
        'Reviewer role',
        ROLE_OPTIONS,
        index=ROLE_OPTIONS.index(str(current_review.get('reviewed_by_role', role))) if str(current_review.get('reviewed_by_role', role)) in ROLE_OPTIONS else ROLE_OPTIONS.index(role),
        key=f'review_role::{dataset_identifier}',
    )
    review_notes = st.text_area(
        'Review notes',
        value=str(current_review.get('review_notes', '')),
        key=f'review_notes::{dataset_identifier}',
        height=90,
        placeholder='Capture mapping approvals, trust caveats, and export restrictions for this dataset.',
    )
    review_actions = st.columns(2)
    if review_actions[0].button('Save approval workflow', key=f'save_review::{dataset_identifier}'):
        review_store[dataset_identifier] = {
            'mapping_status': mapping_status,
            'trust_gate_status': trust_status,
            'export_eligibility_status': export_status,
            'reviewed_by_role': reviewer_role,
            'review_notes': review_notes,
        }
        if application_service is not None:
            application_service.persist_user_settings(st.session_state)
        st.success('Approval workflow state saved for the current dataset.')
        st.rerun()
    if review_actions[1].button('Save benchmark pack from current context', key=f'save_pack::{dataset_identifier}'):
        benchmark_name = f"{dataset_name} benchmark pack"
        observed = safe_df(accuracy_summary.get('benchmark_calibration_table'))
        numeric_bands = {
            'Average length of stay': (2.0, 9.0),
            'Average cost': (1200.0, 60000.0),
        }
        rate_bands = {
            'Readmission rate': (0.08, 0.25),
            'High-risk share': (0.08, 0.45),
        }
        if not observed.empty:
            for _, benchmark_row in observed.iterrows():
                metric = str(benchmark_row.get('metric', ''))
                value = pd.to_numeric(pd.Series([benchmark_row.get('observed_value')]), errors='coerce').iloc[0]
                if pd.isna(value):
                    continue
                if metric in {'Readmission rate', 'High-risk share'}:
                    rate_bands[metric] = (max(float(value) - 0.12, 0.0), min(float(value) + 0.12, 1.0))
                elif metric == 'Average length of stay':
                    numeric_bands[metric] = (max(float(value) - 3.0, 0.0), float(value) + 3.0)
                elif metric == 'Average cost':
                    numeric_bands[metric] = (max(float(value) * 0.6, 0.0), float(value) * 1.4)
        benchmark_pack_store[benchmark_name] = {
            'profile_family': 'organization-pack',
            'rate_bands': rate_bands,
            'numeric_bands': numeric_bands,
            'detail_note': f'Organization-specific benchmark pack derived from {dataset_name}.',
        }
        st.session_state['active_benchmark_pack_name'] = benchmark_name
        if application_service is not None:
            application_service.persist_user_settings(st.session_state)
        st.success(f"Saved organization benchmark pack '{benchmark_name}'.")
        st.rerun()

    text_report = build_text_report(dataset_name, pipeline['overview'], pipeline['structure'], pipeline['field_profile'], pipeline['quality'], pipeline['semantic'], pipeline['readiness'], pipeline['healthcare'], pipeline['insights'])
    executive_report = build_executive_summary_text(
        dataset_name,
        pipeline['overview'],
        pipeline['healthcare'],
        pipeline['insights'],
        trust_summary,
        accuracy_summary,
    )
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
    export_bundle = prepare_policy_aware_export_bundle(
        role=role,
        report_mode=report_mode,
        policy_name=policy_name,
        export_allowed=export_allowed,
        privacy_review=privacy,
        workspace_identity=workspace_identity,
        governance_config=governance_config,
    )
    bundle_text = export_bundle['bundle_text']
    bundle_manifest = export_bundle['bundle_manifest']
    bundle_title = export_bundle['bundle_title']
    bundle_table = export_bundle['bundle_table']
    export_presets = _build_export_workflow_presets(pipeline, role)
    remediation = pipeline.get('remediation_context', {})
    audience_guidance = build_audience_mode_guidance(
        role,
        pipeline.get('use_case_detection', {}),
        pipeline.get('solution_packages', {}),
        pipeline.get('dataset_intelligence', {}),
        pipeline.get('readiness', {}),
        pipeline.get('healthcare', {}),
    )

    def protect(text_bytes: bytes) -> bytes:
        protected = apply_role_based_redaction(apply_export_policy(policy_note + b'\n\n' + text_bytes, policy_name, privacy), role, privacy)
        return apply_export_watermark(protected, str(policy_eval.get('watermark_label', 'Internal export')))

    with export_sections['strategy']:
        render_subsection_header('Recommended export bundle')
        st.write(f"Recommended report for **{role}**: **{recommended_report_mode_for_role(role)}**")
        render_subsection_header('Audience output guidance')
        st.write(str(audience_guidance.get('help_text', 'Audience-specific output guidance is not available yet.')))
        info_or_table(safe_df(audience_guidance.get('recommended_outputs')), 'Audience-specific output guidance appears here once the current role and dataset context are in scope.')
        info_or_table(safe_df(audience_guidance.get('recommended_sections')), 'Audience-specific review sections appear here when the current role has a recommended path through the platform.')
        prompt_table = safe_df(audience_guidance.get('suggested_prompts'))
        if not prompt_table.empty:
            st.caption('Suggested AI Copilot prompts for this audience mode:')
            for _, row in prompt_table.iterrows():
                st.write(f"- {row['suggested_prompt']}")
        render_subsection_header('Cross-setting reporting')
        st.write('Choose the report mode that matches the healthcare review context you need today.')
        cross_setting = build_cross_setting_reporting_profile(
            pipeline.get('dataset_intelligence', {}),
            pipeline.get('use_case_detection', {}),
            pipeline.get('solution_packages', {}),
            report_mode,
        )
        info_or_table(cross_setting, 'Cross-setting reporting guidance appears here when the current dataset maps cleanly to one or more review contexts.')
        info_or_table(bundle_manifest, 'Bundle manifest details appear here after the export bundle profile is prepared for the selected role.')
        tracked_download_button('Download bundle manifest CSV', data=dataframe_to_csv_bytes(bundle_manifest), file_name='role_export_bundle_manifest.csv', mime='text/csv', disabled=not advanced_exports_allowed, event_detail='Downloaded the role export bundle manifest CSV.')
        tracked_download_button('Download bundle guide TXT', data=protect(bundle_text), file_name='role_export_bundle_guide.txt', mime='text/plain', disabled=not advanced_exports_allowed, event_detail='Downloaded the role export bundle guide TXT.')
        application_service = st.session_state.get('application_service')
        if not bundle_manifest.empty:
            record_export_bundle_metadata_once(
                st.session_state,
                application_service,
                workspace_identity,
                dataset_name=dataset_name,
                bundle_label='Role Export Bundle Manifest',
                file_name='role_export_bundle_manifest.csv',
            )
        render_subsection_header('Policy-aware bundle recommendation')
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

        st.markdown('### Report Notes')
        report_notes = build_collaboration_notes_view(
            st.session_state.get('collaboration_notes', []),
            f'Report | {report_mode}',
        )
        info_or_table(report_notes[['timestamp', 'author', 'note_text']] if not report_notes.empty else report_notes, 'Report notes appear here after reviewers add handoff comments or stakeholder guidance for this report mode.')

    executive_summary = pipeline.get('executive_summary', {})
    executive_pack = pipeline.get('executive_report_pack', {})
    printable = pipeline.get('printable_reports', {})
    stakeholder_bundle = pipeline.get('stakeholder_export_bundle', {})
    shared_bundles = build_shared_report_bundles(
        pipeline.get('dataset_intelligence', {}),
        pipeline.get('use_case_detection', {}),
        pipeline.get('solution_packages', {}),
        pipeline.get('healthcare', {}),
        pipeline.get('readiness', {}),
    )
    intelligence = pipeline.get('dataset_intelligence', {})
    with export_sections['supporting_bundles']:
        with st.expander('Executive pack and dataset intelligence', expanded=advanced_export_sections_enabled):
            st.markdown('### Executive Report Pack')
            if executive_pack:
                info_or_table(
                    pd.DataFrame([{'section': key, 'summary': value} for key, value in executive_pack.get('executive_report_sections', {}).items()]),
                    'Executive report sections appear here once the current dataset has enough summary content for a stakeholder-ready pack.',
                )
                tracked_download_button('Download executive report pack TXT', data=protect(executive_pack.get('executive_report_text', '').encode('utf-8')), file_name='executive_report_pack.txt', mime='text/plain', disabled=not advanced_exports_allowed, event_detail='Downloaded the executive report pack TXT.')
                tracked_download_button('Download executive report pack Markdown', data=protect(executive_pack.get('executive_report_markdown', '').encode('utf-8')), file_name='executive_report_pack.md', mime='text/markdown', disabled=not advanced_exports_allowed, event_detail='Downloaded the executive report pack Markdown.')
            if intelligence:
                st.markdown('### Dataset Intelligence Summary')
                info_or_table(safe_df(intelligence.get('analytics_capability_matrix')).head(8), 'Dataset intelligence highlights appear here when the capability matrix is available for the current dataset.')
                info_or_table(safe_df(intelligence.get('highest_impact_upgrades')), 'Data upgrade guidance appears here when the platform identifies high-impact fields to add next.')
        st.markdown('### Print-Friendly Reports')
        if printable:
            tracked_download_button('Download print-friendly executive report', data=protect(printable.get('printable_executive_report', '').encode('utf-8')), file_name='printable_executive_report.txt', mime='text/plain', disabled=not advanced_exports_allowed, event_detail='Downloaded the print-friendly executive report.')
            tracked_download_button('Download print-friendly compliance summary', data=protect(printable.get('printable_compliance_summary', '').encode('utf-8')), file_name='printable_compliance_summary.txt', mime='text/plain', disabled=not advanced_exports_allowed, event_detail='Downloaded the print-friendly compliance summary.')
        with st.expander('Bundle manifests and shared handoffs', expanded=advanced_export_sections_enabled):
            st.markdown('### Stakeholder Export Bundle')
            info_or_table(safe_df(stakeholder_bundle.get('export_bundle_manifest')), 'Stakeholder bundle details appear here once the current dataset supports a grouped export handoff.')
            for note in stakeholder_bundle.get('export_bundle_notes', []):
                st.caption(note)
            if not safe_df(stakeholder_bundle.get('export_bundle_manifest')).empty:
                record_export_bundle_metadata_once(
                    st.session_state,
                    application_service,
                    workspace_identity,
                    dataset_name=dataset_name,
                    bundle_label='Stakeholder Export Bundle',
                    file_name='stakeholder_export_bundle_manifest.csv',
                )
                tracked_download_button('Download stakeholder bundle manifest CSV', data=dataframe_to_csv_bytes(safe_df(stakeholder_bundle.get('export_bundle_manifest'))), file_name='stakeholder_export_bundle_manifest.csv', mime='text/csv', disabled=not stakeholder_bundle_allowed, event_detail='Downloaded the stakeholder export bundle manifest CSV.')
            st.markdown('### Shared Report Bundles')
            info_or_table(safe_df(shared_bundles.get('bundle_manifest')), 'Shared report bundles appear here when the current workflow maps to executive, analyst, clinical, or operations handoffs.')
            for note in shared_bundles.get('bundle_notes', []):
                st.caption(note)
            if not safe_df(shared_bundles.get('bundle_manifest')).empty:
                record_export_bundle_metadata_once(
                    st.session_state,
                    application_service,
                    workspace_identity,
                    dataset_name=dataset_name,
                    bundle_label='Shared Report Bundle',
                    file_name='shared_report_bundle_manifest.csv',
                )
                tracked_download_button(
                    'Download shared bundle manifest CSV',
                    data=dataframe_to_csv_bytes(safe_df(shared_bundles.get('bundle_manifest'))),
                    file_name='shared_report_bundle_manifest.csv',
                    mime='text/csv',
                    disabled=not advanced_exports_allowed,
                    event_detail='Downloaded the shared report bundle manifest CSV.',
                )
                tracked_download_button(
                    'Download shared bundle guide TXT',
                    data=protect(build_shared_report_bundle_text(safe_df(shared_bundles.get('bundle_manifest')))),
                    file_name='shared_report_bundle_guide.txt',
                    mime='text/plain',
                    disabled=not advanced_exports_allowed,
                    event_detail='Downloaded the shared report bundle guide TXT.',
                )
    with export_sections['report_generation']:
        st.markdown('### Report exports')
        st.caption('Choose the smallest set of exports that matches the audience for this review cycle: one primary report, supporting tables when needed, and governance material only when the handoff calls for it.')
        if executive_summary:
            st.caption('Executive summary preview')
            for bullet in executive_summary.get('stakeholder_summary_bullets', [])[:3]:
                st.write(f'- {bullet}')
        st.markdown('### Generated Reports')
        st.caption('Generate a structured report deliverable for the current dataset and review context, then use the preview to confirm the wording before sharing it with pilot stakeholders.')
        job_runtime = pipeline.get('job_runtime', {})
        st.caption(build_job_user_message(job_runtime, 'report_generation'))
        generated_reports = st.session_state.setdefault('generated_report_outputs', {})
        stored_report_artifacts = st.session_state.setdefault('stored_report_artifacts', {})
        report_actions = [
            ('Generate Analyst Report', 'Analyst Report'),
            ('Generate Executive Report', 'Executive Report'),
            ('Generate Data Readiness Report', 'Data Readiness Report'),
            ('Generate Clinical Summary', 'Clinical Summary'),
            ('Generate Readmission Report', 'Readmission Report'),
        ]
        report_status = st.empty()
        report_progress = st.empty()
        export_started_at = 0.0

        def _job_progress(value: float, message: str) -> None:
            bounded = max(0.0, min(1.0, value))
            report_status.caption(f'Managed task progress: {int(bounded * 100)}% | {message}')
            if export_started_at and (time.monotonic() - export_started_at) >= 5.0:
                report_progress.progress(bounded, text=f'Export in progress: {message}')

        report_cols = st.columns(5)
        for idx, (button_label, report_label) in enumerate(report_actions):
            if report_cols[idx].button(button_label, key=f"generate_{report_label.lower().replace(' ', '_')}"):
                try:
                    export_started_at = time.monotonic()
                    report_progress.empty()
                    report_result = generate_export_report_output(
                        st.session_state,
                        job_runtime=job_runtime,
                        report_label=report_label,
                        dataset_name=dataset_name,
                        pipeline=pipeline,
                        workspace_identity=workspace_identity,
                        role=role,
                        progress_callback=_job_progress,
                        application_service=application_service,
                        storage_service=st.session_state.get('storage_service'),
                        policy_name=policy_name,
                        privacy_review=privacy,
                        governance_config=governance_config,
                    )
                    generated_reports[report_label] = report_result
                    report_run_status = report_result['status']
                    report_status.caption(
                        f"Managed task status: {report_run_status.get('status', 'completed').title()} | "
                        f'{report_label} is ready.'
                    )
                    report_progress.empty()
                    if report_result.get('artifact') is not None:
                        stored_report_artifacts[report_label] = report_result['artifact']
                    log_event(
                        'Export Generated',
                        f"Generated {report_label} for dataset '{dataset_name}'.",
                        'Export generation',
                        'Export center',
                        resource_type='generated_report',
                        resource_name=f"{report_label} | {policy_name} | {policy_eval.get('workspace_export_access', 'Editors and owners')}",
                    )
                    st.success(f'{report_label} is ready for preview and download.')
                except PermissionError as error:
                    report_progress.empty()
                    report_status.caption('Managed task status: Blocked | Role-aware export protections prevented this action.')
                    st.warning(str(error))
                except Exception:
                    report_progress.empty()
                    report_status.caption('Managed task status: Failed | Export generation could not complete.')
                    raise
        if generated_reports:
            selected_generated_report = st.selectbox(
                'Generated report preview',
                list(generated_reports.keys()),
                key='selected_generated_report_preview',
            )
            active_generated_report = generated_reports.get(selected_generated_report, {})
            active_report_text = active_generated_report.get('report_bytes', b'')
            bundle_manifest = safe_df(active_generated_report.get('bundle_manifest'))
            export_strategy = active_generated_report.get('export_strategy', {})
            st.text_area(
                'Generated report text',
                value=active_report_text.decode('utf-8') if isinstance(active_report_text, (bytes, bytearray)) else str(active_report_text),
                height=320,
                key='generated_report_preview_text',
            )
            if export_strategy:
                st.caption(str(export_strategy.get('strategy_note', '')))
            if active_generated_report.get('governance_summary'):
                st.caption(str(active_generated_report['governance_summary'].get('policy_evaluation', {}).get('watermark_label', '')))
            info_or_table(bundle_manifest, 'Export artifact details will appear here after a generated report finishes.')
            tracked_download_button(
                f'Download {selected_generated_report} TXT',
                data=active_report_text if isinstance(active_report_text, bytes) else str(active_report_text).encode('utf-8'),
                file_name=f"{selected_generated_report.replace(' ', '_').lower()}.txt",
                mime='text/plain',
                disabled=not export_allowed,
                event_detail=f'Downloaded the generated {selected_generated_report}.',
            )
            tracked_download_button(
                f'Download {selected_generated_report} PDF',
                data=active_generated_report.get('pdf_bytes', b''),
                file_name=f"{selected_generated_report.replace(' ', '_').lower()}.pdf",
                mime='application/pdf',
                disabled=not export_allowed,
                event_detail=f'Downloaded the generated {selected_generated_report} PDF.',
            )
            tracked_download_button(
                f'Download {selected_generated_report} Excel',
                data=active_generated_report.get('excel_bytes', b''),
                file_name=f"{selected_generated_report.replace(' ', '_').lower()}.xlsx",
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                disabled=not export_allowed,
                event_detail=f'Downloaded the generated {selected_generated_report} Excel workbook.',
            )
            tracked_download_button(
                f'Download {selected_generated_report} JSON',
                data=active_generated_report.get('json_bytes', b''),
                file_name=f"{selected_generated_report.replace(' ', '_').lower()}.json",
                mime='application/json',
                disabled=not export_allowed,
                event_detail=f'Downloaded the generated {selected_generated_report} JSON payload.',
            )
            tracked_download_button(
                f'Download {selected_generated_report} ZIP Bundle',
                data=active_generated_report.get('zip_bytes', b''),
                file_name=f"{selected_generated_report.replace(' ', '_').lower()}_bundle.zip",
                mime='application/zip',
                disabled=not export_allowed,
                event_detail=f'Downloaded the generated {selected_generated_report} ZIP export bundle.',
            )
        else:
            st.info('Generate an analyst, executive, or data readiness report to create stakeholder-ready TXT, PDF, Excel, JSON, and ZIP deliverables for this dataset.')
        recent_jobs = build_job_status_view(st.session_state.get('job_runs', [])[:5])
        if not recent_jobs.empty:
            st.markdown('### Recent Managed Tasks')
            info_or_table(recent_jobs, 'Managed task history will appear here after reports or other heavy tasks run.')
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
        tracked_download_button('Download Dataset Summary Report', data=protect(text_report), file_name='smart_dataset_summary.txt', mime='text/plain', disabled=not export_allowed, event_detail='Downloaded the dataset summary report.')
        tracked_download_button('Download Executive Summary Report', data=protect(executive_report), file_name='executive_summary.txt', mime='text/plain', disabled=not export_allowed, event_detail='Downloaded the executive summary report.')
        tracked_download_button('Download Readmission Summary Report', data=protect(readmission_report), file_name='readmission_summary.txt', mime='text/plain', disabled=not export_allowed, event_detail='Downloaded the readmission summary report.')
        tracked_download_button('Download Audience-Specific Report', data=protect(audience_report), file_name=f"{normalize_report_mode(report_mode).replace(' ', '_').lower()}.txt", mime='text/plain', disabled=not export_allowed, event_detail=f'Downloaded the {normalize_report_mode(report_mode)} audience-specific report.')
        tracked_download_button('Download Report Support Tables CSV', data=support_csv, file_name='report_support_tables.csv', mime='text/csv', disabled=not export_allowed, event_detail='Downloaded the report support tables CSV.')

    with export_sections['governance_support']:
        with st.expander('Compliance, governance, and portfolio support', expanded=advanced_export_sections_enabled):
            st.markdown('### Compliance dashboard pack')
            tracked_download_button('Download compliance summary', data=protect(compliance_summary), file_name='compliance_summary.txt', mime='text/plain', disabled=not export_allowed, event_detail='Downloaded the compliance summary.')
            tracked_download_button('Download compliance review CSV', data=compliance_csv, file_name='compliance_review.csv', mime='text/csv', disabled=not export_allowed, event_detail='Downloaded the compliance review CSV.')
            tracked_download_button('Download compliance dashboard JSON', data=json_bytes(compliance_payload), file_name='compliance_dashboard.json', mime='application/json', disabled=not advanced_exports_allowed, event_detail='Downloaded the compliance dashboard JSON.')
            tracked_download_button('Download compliance dashboard CSV', data=build_compliance_dashboard_csv(pipeline['standards'], privacy, role=role), file_name='compliance_dashboard.csv', mime='text/csv', disabled=not advanced_exports_allowed, event_detail='Downloaded the compliance dashboard CSV.')
            tracked_download_button('Download compliance handoff JSON', data=json_bytes(compliance_handoff), file_name='compliance_handoff.json', mime='application/json', disabled=not advanced_exports_allowed, event_detail='Downloaded the compliance handoff JSON.')

            st.markdown('### Governance and audit packet')
            tracked_download_button('Download governance review TXT', data=protect(governance_text), file_name='governance_review.txt', mime='text/plain', disabled=not governance_exports_allowed, event_detail='Downloaded the governance review TXT.')
            tracked_download_button('Download governance review CSV', data=governance_csv, file_name='governance_review.csv', mime='text/csv', disabled=not governance_exports_allowed, event_detail='Downloaded the governance review CSV.')
            tracked_download_button('Download governance review JSON', data=json_bytes(governance_payload), file_name='governance_review.json', mime='application/json', disabled=not governance_exports_allowed, event_detail='Downloaded the governance review JSON.')

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
                tracked_download_button('Download portfolio project summary', data=str(docs.get('portfolio_project_summary', '')).encode('utf-8'), file_name='portfolio_project_summary.txt', mime='text/plain', disabled=not export_allowed, event_detail='Downloaded the portfolio project summary.')
                tracked_download_button('Download demo walkthrough', data=str(docs.get('demo_walkthrough_text', '')).encode('utf-8'), file_name='demo_walkthrough.txt', mime='text/plain', disabled=not export_allowed, event_detail='Downloaded the demo walkthrough.')
            if screenshots:
                with st.expander('Screenshot plan and recruiter callouts'):
                    info_or_table(safe_df(screenshots.get('screenshot_plan')), 'No screenshot plan is available yet.')
                    for line in screenshots.get('recruiter_callouts', []):
                        st.write(f'- {line}')

