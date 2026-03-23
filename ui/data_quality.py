from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from src.ui_components import metric_row, plot_missingness, render_badge_row, render_section_intro, render_surface_panel, render_subsection_header

from ui.common import fmt, info_or_chart, info_or_table, log_event, safe_df
from ui.standards import (
    _build_compliance_quick_actions,
    _build_ehr_onboarding_actions,
    _build_ehr_onboarding_explanation,
    _build_next_compliance_action,
    _build_standards_prefill_actions,
    _run_compliance_quick_action,
    _run_ehr_onboarding_action,
    _run_standards_prefill_action,
    render_privacy,
)

def render_quality(pipeline: dict[str, Any]) -> None:
    quality = pipeline['quality']
    remediation = pipeline.get('remediation_context', {})
    render_section_intro(
        'Data Quality Review',
        'Prioritize missingness, anomaly signals, remediation outcomes, and rule failures with an audit-friendly review of what changed and what still needs attention.',
    )
    render_badge_row(
        [
            ('Audit-friendly review', 'info'),
            ('Remediation-aware', 'accent'),
            ('Rule-driven', 'warning'),
        ]
    )
    render_surface_panel(
        'Quality review guidance',
        'Use this surface to separate source-data defects, remediation outcomes, helper-field disclosures, and manual-review items before downstream analytics or export.',
        tone='warning',
    )
    metric_row([
        ('Quality Score', fmt(quality['quality_score'])),
        ('High Missingness Fields', fmt(len(quality['high_missing']))),
        ('Mixed-type Suspicions', fmt(len(quality['mixed_type_suspicions']))),
        ('Numeric Outlier Fields', fmt(len(quality['numeric_outliers']))),
    ])
    info_or_chart(plot_missingness(safe_df(quality['high_missing'])), 'No major missingness signal was detected in the current dataset.')
    suppressed_helper_missing = safe_df(quality.get('suppressed_helper_missing'))
    if not suppressed_helper_missing.empty:
        st.caption(
            f"{len(suppressed_helper_missing)} helper or audit fields were excluded from the main high-missingness severity view because they were generated during remediation or demo support."
        )
        with st.expander('Helper / audit field missingness (excluded from severity)'):
            info_or_table(
                suppressed_helper_missing,
                'No helper or audit fields were excluded from severity scoring.',
            )
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
    anomaly_summary = pipeline.get('healthcare', {}).get('anomaly_detection', {})
    if anomaly_summary.get('available'):
        render_subsection_header('Clinical anomaly investigation')
        info_or_table(
            safe_df(anomaly_summary.get('classification_table')),
            'No anomaly classification summary is available.',
        )
        with st.expander('Anomaly remediation details', expanded=False):
            info_or_table(
                safe_df(anomaly_summary.get('remediation_action_log')),
                'No remediation action log is available.',
            )
            info_or_table(
                safe_df(anomaly_summary.get('validation_report')),
                'No anomaly validation report is available.',
            )
            info_or_table(
                safe_df(anomaly_summary.get('updated_records_preview')),
                'No auto-corrected anomaly preview is available.',
            )
        review_export = safe_df(anomaly_summary.get('review_export'))
        if not review_export.empty:
            with st.expander('Manual review export preview', expanded=False):
                info_or_table(review_export.head(25), 'No anomaly review export rows are available.')
    bmi_summary = remediation.get('bmi_remediation', {})
    if bmi_summary.get('available'):
        render_subsection_header('BMI remediation summary')
        info_or_table(
            pd.DataFrame([
                {
                    'rows_checked': bmi_summary.get('total_rows_checked', 0),
                    'bmi_outliers': bmi_summary.get('total_bmi_outliers', 0),
                    'outlier_pct': bmi_summary.get('outlier_pct', 0.0),
                    'remediation_mode': bmi_summary.get('remediation_mode', 'median'),
                    'replacement_value_if_used': bmi_summary.get('replacement_value_if_used'),
                    'high_confidence_share': bmi_summary.get('high_confidence_share', 0.0),
                    'target_met': bmi_summary.get('target_met', False),
                }
            ]),
            'No BMI remediation summary is available.',
        )
        with st.expander('BMI confidence, validation, and impact', expanded=False):
            info_or_table(
                safe_df(bmi_summary.get('confidence_distribution')),
                'No BMI confidence distribution is available.',
            )
            info_or_table(
                safe_df(bmi_summary.get('calculation_method_breakdown')),
                'No BMI calculation method breakdown is available.',
            )
            info_or_table(
                safe_df(bmi_summary.get('validation_error_report')),
                'No BMI validation error report is available.',
            )
            info_or_table(
                safe_df(bmi_summary.get('downstream_impact_report')),
                'No BMI downstream impact report is available.',
            )
        audit_preview = safe_df(bmi_summary.get('mapping_audit_table'))
        if not audit_preview.empty:
            with st.expander('BMI remapping audit preview', expanded=False):
                info_or_table(
                    audit_preview.head(20),
                    'No BMI remapping audit rows are available.',
                )
    secondary_summary = remediation.get('secondary_diagnosis_remediation', {})
    if secondary_summary.get('available'):
        render_subsection_header('Secondary diagnosis remediation summary')
        info_or_table(
            pd.DataFrame([
                {
                    'rows_checked': secondary_summary.get('total_rows_checked', 0),
                    'original_missing_count': secondary_summary.get('original_missing_count', 0),
                    'original_missing_pct': secondary_summary.get('original_missing_pct', 0.0),
                    'post_remediation_missing_count': secondary_summary.get('post_remediation_missing_count', 0),
                    'manual_review_count': secondary_summary.get('manual_review_count', 0),
                }
            ]),
            'No secondary diagnosis remediation summary is available.',
        )
        with st.expander('Secondary diagnosis method and confidence details', expanded=False):
            info_or_table(
                safe_df(secondary_summary.get('method_breakdown')),
                'No secondary diagnosis method breakdown is available.',
            )
            info_or_table(
                safe_df(secondary_summary.get('confidence_distribution')),
                'No secondary diagnosis confidence distribution is available.',
            )
            info_or_table(
                safe_df(secondary_summary.get('validation_error_report')),
                'No secondary diagnosis validation report is available.',
            )
        with st.expander('Secondary diagnosis distribution before vs after'):
            info_or_table(
                safe_df(secondary_summary.get('original_distribution')),
                'No original secondary diagnosis distribution is available.',
            )
            info_or_table(
                safe_df(secondary_summary.get('cleaned_distribution')),
                'No cleaned secondary diagnosis distribution is available.',
            )
        audit_preview = safe_df(secondary_summary.get('mapping_audit_table'))
        if not audit_preview.empty:
            with st.expander('Secondary diagnosis audit preview', expanded=False):
                info_or_table(
                    audit_preview.head(20),
                    'No secondary diagnosis audit rows are available.',
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
            with st.expander('Rule severity summary', expanded=False):
                info_or_table(severity_summary, 'No severity summary is available for the current rule checks.')
        if rule_engine.get('passed'):
            st.success(rule_engine.get('reason', 'No quality rule failures were detected for the currently mapped fields.'))
        info_or_table(safe_df(rule_engine.get('summary_table')), 'No rule failures were detected for the current dataset.')
        with st.expander('Rule detail samples', expanded=False):
            info_or_table(safe_df(rule_engine.get('detail_table')), 'No sample violating rows are available.')

def render_readiness(pipeline: dict[str, Any]) -> None:
    readiness = pipeline['readiness']
    remediation = pipeline.get('remediation_context', {})
    use_case = pipeline.get('use_case_detection', {})
    render_section_intro(
        'Analysis Readiness',
        'See which clinical and operational modules are ready now, what remains partially supported, and which remediations will unlock the next highest-value workflows.',
    )
    metric_row([
        ('Ready Modules', fmt(readiness['available_count']), 'Modules with enough mapped support to run cleanly now.'),
        ('Partially Ready', fmt(readiness['partial_count']), 'Modules that can run in a limited or guidance-first mode.'),
        ('Readiness Score', fmt(readiness['readiness_score'], 'score'), 'Overall readiness score across the current workflow.'),
        ('Healthcare Type', pipeline['healthcare']['likely_dataset_type'], 'Full inferred healthcare dataset classification for the current file.'),
    ])
    next_action = _build_next_compliance_action(pipeline)
    if next_action:
        st.caption(f"Recommended next compliance action: {next_action['title']} - {next_action['detail']}")
    if use_case:
        st.write(
            f"**{use_case.get('detected_use_case', 'Dataset type not classified')}** is best aligned to "
            f"**{use_case.get('recommended_workflow', 'Healthcare Data Readiness')}**."
        )
        st.caption(f"{str(use_case.get('narrative', ''))} Use the Recommended Workflow in Data Intake as the canonical guided path.")
        unavailable = safe_df(use_case.get('unavailable_modules'))
        if not unavailable.empty:
            info_or_table(unavailable.head(5), 'No blocked workflow details are available yet.')
    solution_packages = pipeline.get('solution_packages', {})
    package_table = safe_df(solution_packages.get('packages_table'))
    if not package_table.empty:
        st.write(f"**Recommended package:** {solution_packages.get('recommended_package', 'Healthcare Data Readiness')}")
        st.caption('The full package walkthrough is shown in Data Intake. This summary keeps the current readiness view aligned with that recommendation.')
        info_or_table(package_table[['solution_package', 'status']], 'No solution package overview is available yet.')
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
    for key in ['bmi_remediation', 'secondary_diagnosis_remediation', 'synthetic_cost', 'synthetic_clinical', 'synthetic_readmission']:
        note = remediation.get(key, {}).get('lineage_note')
        if note:
            helper_notes.append({'support_type': key.replace('_', ' ').title(), 'note': note})
    with st.expander('Remediation lineage notes', expanded=False):
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
    st.caption("Use this assistant to move from today's readiness baseline to a stronger healthcare workflow state. Each fix shows the likely lift, the blocker it removes, and which analytics unlock next.")
    remediation_table = safe_df(pipeline['remediation'])
    with st.expander('Detailed remediation plan', expanded=False):
        if not remediation_table.empty:
            preferred_columns = [
                'priority_field',
                'current_readiness',
                'projected_readiness',
                'blockers',
                'recommended_fix',
                'impact',
                'modules_unlocked_after_remediation',
                'estimated_readiness_improvement',
            ]
            visible_columns = [column for column in preferred_columns if column in remediation_table.columns]
            info_or_table(remediation_table[visible_columns], 'No remediation steps are needed right now.')
        else:
            info_or_table(remediation_table, 'No remediation steps are needed right now.')
    st.markdown('### Module visibility guide')
    info_or_table(pd.DataFrame([
        {'module': 'Data Remediation Assistant', 'location': 'Data Quality · Analysis Readiness', 'status': 'Available'},
        {'module': 'Patient Cohort Builder', 'location': 'Healthcare Analytics · Cohort Analysis', 'status': 'Available' if pipeline['healthcare']['healthcare_readiness_score'] >= 0.45 else 'Limited'},
        {'module': 'Analysis Log', 'location': 'Data Intake', 'status': 'Available'},
        {'module': 'Healthcare Standards Validator', 'location': 'Dataset Profile · Column Detection and Healthcare Analytics · Healthcare Intelligence', 'status': 'Available' if pipeline['standards'].get('available') else 'Limited'},
        {'module': 'Workflow Packs', 'location': 'Data Intake', 'status': 'Available'},
        {'module': 'Data Lineage', 'location': 'Data Intake', 'status': 'Available'},
    ]), 'No module visibility guidance is available yet.')
    render_privacy(pipeline)

