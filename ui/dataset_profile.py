from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from src.audience_modes import build_audience_mode_guidance
from src.profiler import build_numeric_summary
from src.semantic_mapper import CANONICAL_FIELDS, FIELD_DESCRIPTIONS
from src.ui_components import metric_row, render_badge_row, render_section_intro, render_subsection_header, render_surface_panel, plot_numeric_box, plot_numeric_distribution, plot_top_categories

from ui.common import fmt, info_or_chart, info_or_table, safe_df
from ui.standards import compliance_snapshot, render_standards


PRIORITY_MAPPING_FIELDS = [
    'patient_id',
    'member_id',
    'claim_id',
    'encounter_id',
    'admission_date',
    'discharge_date',
    'service_date',
    'encounter_status_code',
    'encounter_status',
    'encounter_type_code',
    'encounter_type',
    'room_id',
    'diagnosis_code',
    'procedure_code',
    'department',
    'provider_id',
    'provider_name',
    'facility',
    'payer',
    'cost_amount',
    'length_of_stay',
    'readmission',
]


def _render_manual_mapping_editor(pipeline: dict[str, Any]) -> None:
    source_meta = pipeline.get('source_meta', {})
    dataset_name = str(source_meta.get('dataset_name', '') or pipeline.get('dataset_name', '') or 'current dataset')
    dataset_cache_key = str(
        source_meta.get('dataset_cache_key', '')
        or st.session_state.get('active_dataset_diagnostics', {}).get('dataset_cache_key', '')
        or dataset_name
    )
    semantic = pipeline.get('semantic', {})
    canonical_map = dict(semantic.get('canonical_map', {}))
    manual_overrides = dict(semantic.get('manual_overrides_applied', {}) or {})
    source_columns = [str(column) for column in pipeline.get('data', pd.DataFrame()).columns]
    override_store = st.session_state.setdefault('semantic_mapping_overrides_by_dataset', {})
    field_options = [field for field in PRIORITY_MAPPING_FIELDS if field in CANONICAL_FIELDS]

    required_primitives = ('form', 'selectbox', 'form_submit_button', 'session_state', 'success', 'rerun', 'columns')
    if any(not hasattr(st, primitive) for primitive in required_primitives):
        return

    def _render_mapping_controls() -> None:
        st.caption(
            'Confirm or correct ambiguous field mappings once for this dataset. Saved overrides are applied before readiness, healthcare analytics, and insights are recomputed.'
        )
        metric_row([
            ('Manual Overrides', fmt(len(manual_overrides))),
            ('Strong Mappings', fmt(len(canonical_map))),
            ('Source Columns', fmt(len(source_columns))),
            ('Dataset Coverage', fmt(semantic.get('column_accounting_summary', {}).get('field_coverage_score', 0.0), 'pct')),
        ])
        with st.form(f'manual_mapping_form::{dataset_cache_key}'):
            updated_overrides: dict[str, str] = {}
            options = ['-- Auto detect --'] + source_columns
            for field in field_options:
                auto_value = str(canonical_map.get(field, '') or '')
                current_manual = str(manual_overrides.get(field, '') or '')
                default_value = current_manual or auto_value
                default_index = options.index(default_value) if default_value in options else 0
                label = str(field).replace('_', ' ').title()
                help_text = FIELD_DESCRIPTIONS.get(field, 'Manual override for a canonical field.')
                selected = st.selectbox(
                    label,
                    options=options,
                    index=default_index,
                    key=f'manual_mapping::{dataset_cache_key}::{field}',
                    help=help_text,
                )
                if selected != '-- Auto detect --' and selected != auto_value:
                    updated_overrides[field] = selected
            apply_clicked = st.form_submit_button('Apply mapping overrides')
            clear_clicked = st.form_submit_button('Clear manual overrides')
        if apply_clicked:
            override_store[dataset_cache_key] = updated_overrides
            st.session_state['semantic_mapping_overrides_by_dataset'] = override_store
            st.success('Manual mapping overrides saved. Recomputing the dataset workflow with the confirmed field mappings.')
            st.rerun()
        if clear_clicked:
            override_store.pop(dataset_cache_key, None)
            st.session_state['semantic_mapping_overrides_by_dataset'] = override_store
            st.success('Manual mapping overrides cleared. Recomputing with auto-detected mappings.')
            st.rerun()
        if manual_overrides:
            info_or_table(
                pd.DataFrame(
                    [{'canonical_field': field, 'source_column': column} for field, column in manual_overrides.items()]
                ),
                'No manual mapping overrides are active for this dataset.',
            )

    if hasattr(st, 'expander'):
        with st.expander('Manual mapping overrides', expanded=False):
            _render_mapping_controls()
    else:
        _render_mapping_controls()

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
            'status': f'{role} · {policy_name}',
            'detail': f'HIPAA-style risk is currently {risk_level.lower()}.',
        },
    ])

def _build_audience_guidance_frame(guidance: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame([
        {'guidance': 'Recommended workflow', 'detail': guidance.get('recommended_workflow', 'Healthcare Data Readiness')},
        {'guidance': 'Recommended package', 'detail': guidance.get('recommended_package', 'Healthcare Data Readiness')},
        {'guidance': 'Recommended report', 'detail': guidance.get('recommended_report', 'Executive Summary')},
        {'guidance': 'Guidance status', 'detail': guidance.get('status_label', 'Needs stronger source support')},
    ])

def render_overview(pipeline: dict[str, Any]) -> None:
    overview = pipeline['overview']
    structure = pipeline['structure']
    sample_info = pipeline['sample_info']
    trust_summary = pipeline.get('analysis_trust_summary', {})
    accuracy_summary = pipeline.get('result_accuracy_summary', {})
    temporal_context = pipeline.get('temporal_context', {})
    remediation = pipeline.get('remediation_context', {})
    semantic = pipeline.get('semantic', {})
    render_section_intro(
        'Overview',
        'Review the current dataset footprint, structure confidence, remediation disclosures, and solution-fit summary before moving into deeper quality or analytics workflows.',
    )
    render_badge_row(
        [
            ('Profiled', 'info'),
            ('Schema-aware', 'accent'),
            ('Governance-ready', 'success'),
        ]
    )
    render_surface_panel(
        'Overview guidance',
        'This page is the fastest way to confirm footprint, remediation disclosures, dataset intelligence, and stakeholder-facing readiness before moving deeper into the workflow.',
    )
    metric_row([
        ('Rows', fmt(overview['rows'])),
        ('Analyzed Columns', fmt(overview.get('analyzed_columns', overview['columns']))),
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
    source_columns = int(overview.get('source_columns', overview['columns']))
    analyzed_columns = int(overview.get('analyzed_columns', overview['columns']))
    helper_columns_added = int(overview.get('helper_columns_added', 0))
    if helper_columns_added > 0:
        st.caption(
            f"Column count disclosure: the source dataset loaded with {source_columns} columns and the analyzed dataset now contains {analyzed_columns} columns after adding {helper_columns_added} derived or helper fields."
        )
    else:
        st.caption(f"Column count disclosure: the source dataset and analyzed dataset both currently contain {analyzed_columns} columns.")
    accounting = semantic.get('column_accounting_summary', {})
    if accounting:
        render_subsection_header('Source field accounting')
        metric_row([
            ('Source Columns', fmt(accounting.get('total_source_columns', 0))),
            ('Strongly Mapped', fmt(accounting.get('strongly_mapped_columns', 0))),
            ('Suggestion-only', fmt(accounting.get('suggestion_only_columns', 0))),
            ('Field Coverage', fmt(accounting.get('field_coverage_score', 0.0), 'pct')),
        ])
        if semantic.get('manual_overrides_applied'):
            st.caption(f"Manual semantic overrides active: {len(semantic.get('manual_overrides_applied', {}))}.")
        unresolved_columns = accounting.get('unresolved_columns', [])
        if unresolved_columns:
            st.caption('Columns still not mapped strongly enough for downstream use: ' + ', '.join(unresolved_columns))
        else:
            st.caption('All current source columns are accounted for with strong downstream mappings.')
    if trust_summary.get('available'):
        render_subsection_header('Result trust and interpretation')
        metric_row([
            ('Trust Level', str(trust_summary.get('trust_level', 'Unknown'))),
            ('Trust Score', fmt(trust_summary.get('trust_score', 0.0), 'pct')),
            ('Interpretation Mode', str(trust_summary.get('interpretation_mode', 'Not available'))),
            (
                'Authoritative / Analyzed Rows',
                f"{fmt(trust_summary.get('authoritative_row_count', 0))} / {fmt(trust_summary.get('analyzed_row_count', 0))}",
            ),
        ])
        st.caption(str(trust_summary.get('summary_text', '')))
        for note in trust_summary.get('notes', [])[:3]:
            st.caption(note)
        with st.expander('Trust disclosures and use guidance', expanded=False):
            info_or_table(
                safe_df(trust_summary.get('disclosure_table')),
                'Trust disclosures are not available for the current dataset.',
            )
            recommended = trust_summary.get('recommended_uses', [])
            restricted = trust_summary.get('restricted_uses', [])
            if recommended:
                st.write('**Recommended uses**')
                for item in recommended:
                    st.write(f'- {item}')
            if restricted:
                st.write('**Use with caution for**')
                for item in restricted:
                    st.write(f'- {item}')
            info_or_table(
                safe_df(accuracy_summary.get('metric_confidence_table')),
                'Metric-level confidence details are not available for the current dataset.',
            )
            info_or_table(
                safe_df(accuracy_summary.get('field_uncertainty_table')),
                'Field-level uncertainty details are not available for the current dataset.',
            )
            info_or_table(
                safe_df(accuracy_summary.get('benchmark_calibration_table')),
                'Benchmark calibration checks are not available for the current dataset.',
            )
    with st.expander('Dataset preview', expanded=False):
        st.dataframe(pipeline['data'].head(20), width='stretch')
    intelligence = pipeline.get('dataset_intelligence', {})
    if intelligence:
        summary = intelligence.get('dataset_intelligence_summary', {})
        render_subsection_header('Dataset intelligence summary')
        metric_row([
            ('Dataset Type', str(summary.get('dataset_type', 'Not classified')), str(intelligence.get('dataset_type_rationale', 'Full dataset classification for the current file.'))),
            ('Healthcare Coverage', str(summary.get('healthcare_coverage', 'Not available')), 'How strongly the current dataset supports healthcare-specific analytics.'),
            ('Analytics Readiness', str(summary.get('analytics_readiness', 'Not available')), 'Overall readiness across the mapped analytics workflow.'),
            ('Enabled / Blocked', f"{summary.get('enabled_modules', 0)} / {summary.get('blocked_modules', 0)}", 'Count of currently enabled modules versus blocked modules.'),
        ])
        st.caption(str(intelligence.get('dataset_type_rationale', '')))
        st.caption(str(intelligence.get('support_disclosure_note', '')))
        render_subsection_header('Analytics capability matrix')
        info_or_table(safe_df(intelligence.get('analytics_capability_matrix')), 'This capability view fills in once schema, readiness, and healthcare signals are available for the current dataset.')
        render_subsection_header('Why some analytics are blocked')
        blocked = safe_df(intelligence.get('blocker_explanations'))
        partial = safe_df(intelligence.get('partial_support_explanations'))
        if blocked.empty and partial.empty:
            st.info('The current dataset has no major blocked or partially supported analytics in scope.')
        else:
            if not blocked.empty:
                info_or_table(blocked, 'No blocked analytics explanations are available.')
            if not partial.empty:
                info_or_table(partial, 'No partial-support explanations are available.')
        render_subsection_header('Highest-impact data upgrades')
        info_or_table(safe_df(intelligence.get('highest_impact_upgrades')), 'Data upgrade guidance will appear here when the platform identifies blocked or partially supported analytics.')
    solution_layers = pipeline.get('solution_layers', {})
    cards = safe_df(solution_layers.get('solution_cards'))
    if not cards.empty:
        st.markdown('### Solution Layer Fit')
        summary = solution_layers.get('summary', {})
        metric_row([
            ('Ready Solution Layers', fmt(summary.get('ready_now', 0))),
            ('Partial Solution Layers', fmt(summary.get('partially_supported', 0))),
            ('Recommended Start', str(summary.get('recommended_starting_point', 'Healthcare Data Readiness'))),
            ('Healthcare Readiness', fmt(solution_layers.get('healthcare_readiness_score', 0.0), 'pct')),
        ])
        st.caption(str(solution_layers.get('narrative', '')))
        info_or_table(cards[['solution_group', 'status_label', 'who_it_is_for']], 'No solution-layer fit summary is available yet.')
    audience_guidance = build_audience_mode_guidance(
        st.session_state.get('active_role', 'Analyst'),
        pipeline.get('use_case_detection', {}),
        pipeline.get('solution_packages', {}),
        pipeline.get('dataset_intelligence', {}),
        pipeline.get('readiness', {}),
        pipeline.get('healthcare', {}),
    )
    st.markdown('### Audience Guidance')
    metric_row([
        ('Audience Mode', str(audience_guidance.get('audience_label', 'Analyst'))),
        ('Recommended Package', str(audience_guidance.get('recommended_package', 'Healthcare Data Readiness'))),
        ('Recommended Report', str(audience_guidance.get('recommended_report', 'Executive Summary'))),
        ('Guidance Status', str(audience_guidance.get('status_label', 'Needs stronger source support'))),
    ])
    st.caption(str(audience_guidance.get('narrative', '')))
    st.caption('Change the active role in Export Center to tailor this guidance for different healthcare audiences.')
    with st.expander('Audience guidance details', expanded=False):
        info_or_table(_build_audience_guidance_frame(audience_guidance), 'Audience guidance is not available yet.')
        info_or_table(safe_df(audience_guidance.get('summary_emphasis')), 'No audience-specific summary emphasis is available yet.')
        info_or_table(safe_df(audience_guidance.get('recommended_sections')), 'No audience-specific starting sections are available yet.')
        info_or_table(safe_df(audience_guidance.get('recommended_modules')), 'No audience-specific module guidance is available yet.')
        info_or_table(safe_df(audience_guidance.get('limited_modules')), 'No limited modules are currently flagged for this audience mode.')
    st.markdown('### Compliance handoff snapshot')
    with st.expander('Compliance handoff details', expanded=False):
        info_or_table(compliance_snapshot(pipeline), 'Compliance handoff details are not available yet.')
    st.markdown('### Compliance & Governance Summary')
    compliance_summary = pipeline.get('compliance_governance_summary', {})
    cards = compliance_summary.get('compliance_snapshot_cards', [])
    if cards:
        metric_row([(card['label'], card['value']) for card in cards[:4]])
    with st.expander('Compliance and governance details', expanded=False):
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
    st.markdown('### About Clinverity')
    app_metadata = pipeline.get('app_metadata', {})
    if app_metadata:
        st.write(str(app_metadata.get('tagline', '')))
        st.write(f"**Built for:** {app_metadata.get('best_for', 'Healthcare analytics, quality review, and stakeholder reporting')}")
        st.write(f"**Works best with:** {app_metadata.get('best_data', 'Healthcare, operational, and general tabular datasets')}")
        st.caption(str(app_metadata.get('synthetic_support', '')))
        st.caption(str(app_metadata.get('maturity', '')))

def render_column_detection(pipeline: dict[str, Any], original_lookup: dict[str, str]) -> None:
    render_section_intro(
        'Column Detection',
        'Inspect inferred field types, semantic mappings, and standards-facing metadata so clinical validation and downstream auditability stay transparent.',
    )
    detection = pipeline['structure'].detection_table.copy()
    if original_lookup:
        detection.insert(1, 'original_name', detection['column_name'].map(original_lookup).fillna(detection['column_name']))
    if detection.empty:
        info_or_table(detection, 'Column detection details are not available.')
    else:
        total_rows = len(detection)
        page_size = st.selectbox(
            'Column detection page size',
            options=[25, 50, 100, 250],
            index=1,
            key='column_detection_page_size',
        )
        total_pages = max(1, (total_rows + page_size - 1) // page_size)
        page_number = st.number_input(
            'Column detection page',
            min_value=1,
            max_value=total_pages,
            value=1,
            step=1,
            key='column_detection_page_number',
        )
        start = (int(page_number) - 1) * int(page_size)
        end = min(start + int(page_size), total_rows)
        st.caption(f'Showing columns {start + 1:,}-{end:,} of {total_rows:,}.')
        info_or_table(detection.iloc[start:end].reset_index(drop=True), 'Column detection details are not available.')
    st.markdown('### Semantic Mapping Review')
    info_or_table(pipeline['semantic'].get('mapping_table', pd.DataFrame()), 'No strong semantic mappings were detected yet.')
    _render_manual_mapping_editor(pipeline)
    st.markdown('### Data Dictionary')
    info_or_table(pipeline['data_dictionary'], 'Data dictionary details are not available.')
    render_standards(pipeline, key_prefix='column_detection_standards')

def render_profiling(pipeline: dict[str, Any]) -> None:
    profile = pipeline['field_profile']
    render_section_intro(
        'Field Profiling',
        'Explore field-level distributions, identifier patterns, and numeric behavior that drive remediation, mapping confidence, and analysis readiness.',
    )
    metric_row([
        ('Near-constant Fields', fmt(int(profile['is_near_constant'].sum()) if 'is_near_constant' in profile.columns else 0)),
        ('High-cardinality IDs', fmt(int(profile['is_high_cardinality_identifier'].sum()) if 'is_high_cardinality_identifier' in profile.columns else 0)),
        ('Potential Categoricals', fmt(int(profile['looks_potentially_categorical'].sum()) if 'looks_potentially_categorical' in profile.columns else 0)),
        ('Numeric Outlier Signals', fmt(int(profile['has_numeric_outlier_signal'].sum()) if 'has_numeric_outlier_signal' in profile.columns else 0)),
    ])
    info_or_table(profile, 'Field profile details are not available.')
    with st.expander('Numeric summary', expanded=False):
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

