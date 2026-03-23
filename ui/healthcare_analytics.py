from __future__ import annotations

import json
import time
from typing import Any

import pandas as pd
import streamlit as st

from src.decision_support import build_kpi_benchmarking_layer
from src.enterprise_features import cohort_monitoring_over_time
from src.healthcare_analysis import build_cohort_summary, build_readmission_cohort_review, operational_alerts, plan_readmission_intervention
from src.jobs import build_job_status_view, build_job_user_message, get_job_result, get_job_status, submit_job
from src.modeling_studio import build_model_comparison_studio, build_model_fairness_review, build_prediction_explainability, build_predictive_model, default_modeling_selection
from src.plan_awareness import plan_feature_enabled
from src.ui_components import metric_row, plot_bar, plot_correlation, plot_time_trend, render_badge_row, render_section_intro, render_surface_panel
from src.visualization_performance import (
    build_visual_cache_diagnostics,
    build_cohort_analysis_payload,
    build_trend_analysis_payload,
    build_visual_cache_key,
    get_cached_visual_payload,
    resolve_debounced_filters,
    store_visual_payload_for_pipeline,
)

from ui.common import build_cohort_placeholder_frame, fmt, info_or_chart, info_or_table, safe_df
from ui.standards import render_privacy, render_standards


def _show_skeleton_loader(label: str, blocks: int = 3):
    placeholder = st.empty()
    skeleton = ''.join(
        '<div style="height:18px;margin:10px 0;border-radius:8px;'
        'background:linear-gradient(90deg,#eef2f7 25%,#f8fafc 50%,#eef2f7 75%);"></div>'
        for _ in range(blocks)
    )
    placeholder.markdown(
        (
            '<div style="padding:0.75rem 0;">'
            f'<div style="font-size:0.9rem;color:#5b6472;margin-bottom:0.5rem;">{label}</div>'
            f'{skeleton}'
            '</div>'
        ),
        unsafe_allow_html=True,
    )
    return placeholder


def _lazy_chart_enabled(label: str, key: str, data: pd.DataFrame) -> bool:
    default_enabled = len(data) <= 150_000
    return st.toggle(label, value=default_enabled, key=key)

def _render_readmission_readiness(readmission: dict[str, Any]) -> None:
    readiness = readmission.get('readiness', {})
    missing = readiness.get('missing_fields', [])
    available = readiness.get('available_analysis', [])
    support_level = str(readiness.get('support_level', 'Unavailable'))
    support_score = readiness.get('support_score')
    st.caption(readiness.get('badge_text', 'Readmission analytics readiness is being assessed from the current dataset.'))
    metric_row([
        ('Support Level', support_level),
        ('Support Score', fmt(support_score, 'pct') if support_score is not None else 'Not available'),
        ('Missing Fields', fmt(len(missing))),
        ('Available Readmission Views', fmt(len(available))),
    ])
    if readmission.get('solution_summary_cards'):
        metric_row([(card['label'], card['value']) for card in readmission.get('solution_summary_cards', [])])
    if readiness.get('blocker_summary'):
        st.info(str(readiness.get('blocker_summary')))
    info_or_table(
        safe_df(readiness.get('support_table')),
        'Readmission readiness guidance will appear here when the platform assesses the current dataset structure.',
    )
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
    for note in readiness.get('guidance_notes', []):
        st.caption(str(note))

def _render_readmission_overview(readmission: dict[str, Any]) -> None:
    if not readmission.get('available'):
        _render_readmission_readiness(readmission)
        if readmission.get('solution_story'):
            st.caption(str(readmission.get('solution_story')))
        if readmission.get('next_best_action'):
            st.caption(f"Next best action: {readmission.get('next_best_action')}")
        st.info(readmission.get('reason', 'Readmission-focused analytics are not available for the current dataset.'))
        return

    overview = readmission.get('overview', {})
    st.caption('Hospital Readmission Risk Analytics is the platform’s readmission-focused solution workflow for pilot teams who need a clear burden view, high-risk populations, likely drivers, and intervention-ready next steps.')
    st.caption('This overview summarizes the current readmission burden, the highest-variance groupings, and the longitudinal pattern when encounter dates are available.')
    metric_row([
        ('Overall Readmission Rate', fmt(overview.get('overall_readmission_rate'), 'pct')),
        ('Readmissions in Scope', fmt(overview.get('readmission_count', 0))),
        ('Records Reviewed', fmt(overview.get('records_in_scope', 0))),
        ('Workflow Status', readmission.get('readiness', {}).get('badge_text', 'Available')),
    ])
    metric_row([(card['label'], card['value']) for card in readmission.get('solution_summary_cards', [])])
    st.caption(str(readmission.get('solution_story', '')))
    if readmission.get('note'):
        st.caption(str(readmission['note']))
    source_label = str(readmission.get('source', 'native')).strip().lower()
    if source_label == 'synthetic':
        st.caption('Readmission disclosure: this workflow is using a deterministic synthetic readmission flag for demonstration support. Rates and segments are useful for workflow review, but they are not source-grade clinical outcomes.')
    elif source_label == 'derived':
        st.caption('Readmission disclosure: this workflow is deriving a 30-day readmission flag from repeat encounters. It remains useful for operational review, but a native readmission field would strengthen trust and reporting.')

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

    st.markdown('#### Readmission Trend Over Time')
    trend_table = safe_df(readmission.get('trend'))
    if not trend_table.empty and 'month' in trend_table.columns:
        info_or_chart(
            plot_time_trend(trend_table, 'month', 'readmission_rate', 'Readmission Rate Trend Over Time'),
            'No readmission trend chart is available for the current dataset.',
        )
        info_or_table(trend_table, 'No readmission trend table is available for the current dataset.')
    else:
        st.info('Readmission trend over time unlocks when the dataset includes admission, discharge, service, or event-style dates.')

    st.markdown('#### High-Risk Readmission Segments')
    segments = safe_df(readmission.get('high_risk_segments'))
    if not segments.empty:
        st.write(str(readmission.get('top_high_risk_summary', '')))
    info_or_table(segments, 'No high-risk readmission segments were identified from the current rules.')
    st.markdown('#### Readmission Driver Analysis')
    driver_table = safe_df(readmission.get('driver_table'))
    if not driver_table.empty:
        st.write(readmission.get('driver_interpretation', ''))
        info_or_table(driver_table, 'No readmission driver table is available.')
        info_or_chart(plot_bar(driver_table, 'factor', 'gap_vs_overall', 'Readmission Driver Gaps'), 'No readmission driver chart is available.')
    else:
        st.info(readmission.get('driver_interpretation', 'Readmission driver analysis is limited for the current dataset.'))
    st.markdown('#### High-Risk Patients or Rows')
    high_risk_patients = safe_df(readmission.get('high_risk_patients'))
    if not high_risk_patients.empty:
        metric_row([
            ('High-Risk Rows to Review', fmt(len(high_risk_patients))),
            ('Highest Risk Score', fmt(high_risk_patients['readmission_risk_score'].max(), 'float')),
            ('High-Risk Segment Rows', fmt((high_risk_patients['readmission_risk_segment'] == 'High Risk').sum())),
            ('Next Best Action', str(readmission.get('next_best_action', 'Review the highest-gap readmission cohort first.'))),
        ])
        st.caption('These rows are directional follow-up priorities for discharge planning, case management, or post-discharge outreach review.')
    info_or_table(high_risk_patients, 'No high-risk patient or encounter rows are available for readmission review.')
    st.markdown('#### Readmission Intervention Recommendations')
    recommendations = safe_df(readmission.get('intervention_recommendations'))
    if not recommendations.empty:
        info_or_table(recommendations, 'No readmission intervention recommendations are available.')
    else:
        st.info('Intervention recommendations appear here when the dataset surfaces at least one high-gap readmission segment or driver.')
    st.caption('Export Center can generate a readmission-specific summary once you are ready to share the overview, drivers, interventions, and recommended next steps.')


def _render_readmission_model_snapshot(readmission: dict[str, Any]) -> None:
    st.markdown('#### Readmission Risk Model Snapshot')
    if not readmission.get('available'):
        st.info(readmission.get('reason', 'Readmission risk modeling is not available for the current dataset.'))
        return
    high_risk_patients = safe_df(readmission.get('high_risk_patients'))
    driver_table = safe_df(readmission.get('driver_table'))
    if high_risk_patients.empty and driver_table.empty:
        st.info('Readmission risk modeling needs enough patient-level or segment-level signal to rank likely risk contributors.')
        return
    if not high_risk_patients.empty:
        metric_row([
            ('Model rows in scope', fmt(len(high_risk_patients))),
            ('Average risk score', fmt(high_risk_patients['readmission_risk_score'].mean(), 'float')),
            ('High-risk share', fmt((high_risk_patients['readmission_risk_segment'] == 'High Risk').mean(), 'pct')),
            ('Top risk segment', str(high_risk_patients.iloc[0].get('readmission_risk_segment', 'High Risk'))),
        ])
        info_or_table(
            high_risk_patients.head(15),
            'No readmission risk rows are available for the current dataset.',
        )
    if not driver_table.empty:
        st.caption(str(readmission.get('driver_interpretation', 'This model snapshot highlights the factors with the strongest observed readmission gap versus the overall population.')))
        info_or_table(driver_table, 'No readmission driver summary is available.')
        info_or_chart(
            plot_bar(driver_table.head(8), 'factor', 'gap_vs_overall', 'Readmission Risk Factors'),
            'No readmission risk-factor chart is available.',
        )


def _render_length_of_stay_prediction(section: dict[str, Any]) -> None:
    st.markdown('### Length-of-Stay Prediction')
    if not section.get('available'):
        st.info(section.get('reason', 'Length-of-stay prediction is not available for the current dataset.'))
        return
    summary = section.get('summary', {})
    metric_row([
        ('Avg Actual LOS', fmt(summary.get('average_actual_length_of_stay'), 'float')),
        ('Avg Predicted LOS', fmt(summary.get('average_predicted_length_of_stay'), 'float')),
        ('Predicted High-LOS Rows', fmt(summary.get('predicted_high_los_rows', 0))),
        ('P90 Predicted LOS', fmt(summary.get('p90_predicted_length_of_stay'), 'float')),
    ])
    st.caption(str(section.get('narrative', '')))
    info_or_table(safe_df(section.get('factor_table')), 'LOS prediction factors are not available yet.')
    info_or_table(safe_df(section.get('group_table')), 'LOS benchmark groups are not available yet.')
    high_los_rows = safe_df(section.get('high_los_rows'))
    info_or_table(high_los_rows, 'No high-LOS rows are available for review.')
    if not high_los_rows.empty and 'predicted_length_of_stay' in high_los_rows.columns:
        chart_source = safe_df(section.get('group_table'))
        if not chart_source.empty and 'average_predicted_length_of_stay' in chart_source.columns:
            info_or_chart(
                plot_bar(chart_source.head(10), 'segment', 'average_predicted_length_of_stay', 'Predicted LOS by Segment'),
                'No LOS benchmark chart is available.',
            )


def _render_mortality_adverse_event_indicators(section: dict[str, Any]) -> None:
    st.markdown('### Mortality and Adverse Event Indicators')
    if not section.get('available'):
        st.info(section.get('reason', 'Mortality and adverse-event indicators are not available for the current dataset.'))
        return
    st.caption(str(section.get('narrative', '')))
    indicator_table = safe_df(section.get('indicator_table'))
    info_or_table(indicator_table, 'No mortality or adverse-event indicators are available.')
    group_table = safe_df(section.get('group_table'))
    info_or_table(group_table, 'No grouped adverse-event indicators are available.')
    flagged_rows = safe_df(section.get('flagged_rows'))
    info_or_table(flagged_rows, 'No flagged adverse-event rows are available.')
    if not group_table.empty and 'average_adverse_event_score' in group_table.columns:
        info_or_chart(
            plot_bar(group_table.head(10), 'segment', 'average_adverse_event_score', 'Adverse Event Burden by Segment'),
            'No adverse-event burden chart is available.',
        )


def _render_population_health(section: dict[str, Any]) -> None:
    st.markdown('### Population Health Analytics')
    if not section.get('available'):
        st.info(section.get('reason', 'Population health analytics are not available for the current dataset.'))
        return
    metric_row([(card['label'], card['value']) for card in section.get('summary_cards', [])])
    st.caption(str(section.get('narrative', '')))
    for line in section.get('insights', []):
        st.write(f'- {line}')
    prevalence_table = safe_df(section.get('prevalence_table'))
    info_or_table(prevalence_table, 'No population health prevalence table is available.')
    if not prevalence_table.empty:
        info_or_chart(
            plot_bar(prevalence_table.head(10), 'segment', 'population_share', 'Population Share by Segment'),
            'No population-health chart is available.',
        )


def _render_clinical_outcome_benchmarks(pipeline: dict[str, Any]) -> None:
    st.markdown('### Clinical Outcome Benchmarks')
    outcome_benchmarks = pipeline['healthcare'].get('clinical_outcome_benchmarks', {})
    kpi_benchmarks = pipeline.get('kpi_benchmarking', {})
    if not outcome_benchmarks.get('available') and not kpi_benchmarks.get('available'):
        st.info('Clinical outcome benchmarks need stage, smoking, risk, cost, or readmission signals to compare the current population against internal reference groups.')
        return
    if outcome_benchmarks.get('available'):
        metric_row([(card['label'], card['value']) for card in outcome_benchmarks.get('summary_cards', [])])
        st.caption(str(outcome_benchmarks.get('narrative', '')))
        info_or_table(safe_df(outcome_benchmarks.get('summary_table')), 'No clinical outcome benchmark summary is available.')
        for note in outcome_benchmarks.get('notes', []):
            st.caption(str(note))
    if kpi_benchmarks.get('available'):
        st.markdown('#### Internal KPI Benchmark Layer')
        metric_row([(card['label'], card['value']) for card in kpi_benchmarks.get('kpi_cards', [])[:4]])
        info_or_table(safe_df(kpi_benchmarks.get('benchmark_table')), 'No KPI benchmark table is available.')
        for line in kpi_benchmarks.get('standout_positive_signals', []):
            st.write(f'- Positive: {line}')
        for line in kpi_benchmarks.get('standout_risk_signals', []):
            st.write(f'- Risk: {line}')


def _render_standards_compliance_snapshot(pipeline: dict[str, Any]) -> None:
    st.markdown('### SDTM / ADaM Standards Compliance')
    standards = pipeline.get('standards', {})
    profiles = safe_df(standards.get('standards_profiles'))
    if profiles.empty:
        st.info('SDTM and ADaM compliance checking becomes available when the dataset resembles study, domain, or analysis-ready trial structures.')
        return
    cdisc_profiles = profiles[profiles['standard_type'].astype(str).str.contains('CDISC SDTM|CDISC ADaM', case=False, regex=True)]
    if cdisc_profiles.empty:
        st.info('The current dataset is not strongly trial-oriented, so SDTM and ADaM checks remain informational for now.')
        return
    metric_row([
        ('Best CDISC Match', str(cdisc_profiles.iloc[0].get('standard_type', 'CDISC'))),
        ('Top Confidence', fmt(cdisc_profiles.iloc[0].get('confidence_score'), 'float')),
        ('Standards Readiness', fmt(cdisc_profiles.iloc[0].get('standards_readiness_score'), 'float')),
        ('Missing Required Fields', fmt(cdisc_profiles.iloc[0].get('missing_required_fields', 0))),
    ])
    info_or_table(cdisc_profiles, 'No SDTM or ADaM standards profile summary is available.')
    requirements = safe_df(standards.get('required_field_review'))
    if not requirements.empty:
        cdisc_requirements = requirements[requirements['standard_type'].astype(str).str.contains('CDISC SDTM|CDISC ADaM', case=False, regex=True)]
        info_or_table(cdisc_requirements, 'No SDTM or ADaM required-field review is available.')

def render_healthcare(pipeline: dict[str, Any]) -> None:
    healthcare = pipeline['healthcare']
    remediation = pipeline.get('remediation_context', {})
    accuracy_summary = pipeline.get('result_accuracy_summary', {})
    render_section_intro(
        'Healthcare Intelligence',
        'Review clinically relevant indicators, operational risk patterns, and standards-aware signals in a product-grade intelligence workspace.',
    )
    render_badge_row(
        [
            ('Clinical signals', 'info'),
            ('Operational risk', 'accent'),
            ('Standards-aware', 'success'),
        ]
    )
    render_surface_panel(
        'Healthcare workflow guidance',
        'Stay in this surface for domain-specific signals, readmission workflows, pathway review, and decision-support outputs grounded in the active dataset.',
    )
    metric_row([
        ('Healthcare Readiness', fmt(healthcare['healthcare_readiness_score'], 'pct')),
        ('Likely Dataset Type', healthcare['likely_dataset_type']),
        ('Matched Healthcare Fields', fmt(len(healthcare.get('matched_healthcare_fields', [])))),
        ('Risk Segmentation', 'Available' if healthcare.get('risk_segmentation', {}).get('available') else 'Limited'),
    ])
    if accuracy_summary.get('available'):
        st.caption('Healthcare outputs now include native-vs-derived confidence, external-reporting gates, and benchmark calibration checks.')
        with st.expander('Accuracy, calibration, and reporting gates', expanded=False):
            info_or_table(
                safe_df(accuracy_summary.get('metric_confidence_table')),
                'Metric-level confidence details are not available for the current healthcare workflow.',
            )
            info_or_table(
                safe_df(accuracy_summary.get('benchmark_calibration_table')),
                'Benchmark calibration checks are not available for the current healthcare workflow.',
            )
            info_or_table(
                safe_df(accuracy_summary.get('module_reporting_gates')),
                'Module-level reporting gates are not available for the current healthcare workflow.',
            )
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
    _render_readmission_model_snapshot(healthcare.get('readmission', {}))
    _render_length_of_stay_prediction(healthcare.get('length_of_stay_prediction', {}))
    _render_mortality_adverse_event_indicators(healthcare.get('mortality_adverse_events', {}))
    _render_population_health(healthcare.get('population_health', {}))
    _render_clinical_outcome_benchmarks(pipeline)
    _render_standards_compliance_snapshot(pipeline)
    render_predictive_modeling_studio(pipeline)
    st.markdown('### Clinical Pathway and Outcome Intelligence')
    st.caption('This workflow brings together outcome, duration, and pathway signals so you can see which clinical groups are underperforming and what to review next.')
    survival = healthcare.get('survival_outcomes', {})
    st.markdown('#### Outcome Intelligence Snapshot')
    if not survival.get('available'):
        st.info(survival.get('reason', 'Outcome intelligence is not available for the current dataset.'))
    else:
        st.caption(survival.get('interpretation', survival.get('summary', 'Outcome intelligence is available for the current dataset.')))
        metric_row([
            ('Outcome Support', str(survival.get('support_level', 'Available'))),
            ('Overall Survival Rate', fmt(survival.get('overall_survival_rate'), 'pct') if survival.get('overall_survival_rate') is not None else 'Not available'),
            ('Stage Groups', fmt(len(safe_df(survival.get('stage_table'))))),
            ('Treatment Groups', fmt(len(safe_df(survival.get('treatment_table'))))),
        ])
        duration_summary = survival.get('duration_summary') or {}
        if duration_summary:
            metric_row([
                ('Avg Duration', fmt(duration_summary.get('average_duration_days'), 'float')),
                ('Median Duration', fmt(duration_summary.get('median_duration_days'), 'float')),
                ('P90 Duration', fmt(duration_summary.get('p90_duration_days'), 'float')),
                ('Duration Records', fmt(duration_summary.get('records_with_duration', 0))),
            ])
        outcome_snapshot_tables = []
        if not safe_df(survival.get('stage_table')).empty:
            outcome_snapshot_tables.append(safe_df(survival.get('stage_table')).head(5))
        if not safe_df(survival.get('treatment_table')).empty:
            outcome_snapshot_tables.append(safe_df(survival.get('treatment_table')).head(5))
        if outcome_snapshot_tables:
            info_or_table(pd.concat(outcome_snapshot_tables, ignore_index=True, sort=False), 'No stage or treatment outcome snapshot is available yet.')
        else:
            st.info(str(survival.get('what_unlocks_next', 'Add stronger stage, treatment, or date detail to unlock richer outcome comparisons.')))
        duration_by_cohort = safe_df(survival.get('duration_by_cohort'))
        if not duration_by_cohort.empty:
            st.markdown('#### Duration by Cohort')
            info_or_table(duration_by_cohort, 'No cohort-level duration summary is available yet.')
        if survival.get('what_unlocks_next'):
            st.caption(f"Next unlock: {survival.get('what_unlocks_next')}")
    pathway = healthcare.get('care_pathway', {})
    if not pathway.get('available'):
        st.info(pathway.get('reason', 'Clinical pathway intelligence is not available for the current dataset.'))
    else:
        st.caption(pathway.get('summary', 'The platform is summarizing treatment pathways using observed duration and outcome patterns.'))
        stage_table = safe_df(pathway.get('stage_table'))
        treatment_table = safe_df(pathway.get('treatment_table'))
        pathway_table = safe_df(pathway.get('pathway_table'))
        pathway_summary_table = safe_df(pathway.get('pathway_summary_table'))
        average_duration_by_pathway = safe_df(pathway.get('average_duration_by_pathway'))
        bottleneck_table = safe_df(pathway.get('bottleneck_summary'))
        poor_outcome_table = safe_df(pathway.get('poor_outcome_pathways'))
        metric_row([
            ('Stage Pathways', fmt(len(stage_table))),
            ('Treatment Pathways', fmt(len(treatment_table))),
            ('Combined Pathways', fmt(len(pathway_table))),
            ('Bottleneck Signals', fmt(len(bottleneck_table))),
        ])
        if not average_duration_by_pathway.empty and 'average_treatment_duration_days' in average_duration_by_pathway.columns:
            metric_row([
                ('Avg Pathway Duration', fmt(average_duration_by_pathway['average_treatment_duration_days'].mean(), 'float')),
                ('Longest-Running Pathway', str(average_duration_by_pathway.sort_values('average_treatment_duration_days', ascending=False).iloc[0].get('pathway_label', 'Not available'))),
                ('Underperforming Pathways', fmt(len(poor_outcome_table))),
                ('Suggested Pathway Actions', fmt(len(pathway.get('action_recommendations', [])))),
            ])
        st.markdown('#### Pathway Performance Summary')
        info_or_table(pathway_summary_table, 'No pathway performance summary is available for the current dataset.')
        if not average_duration_by_pathway.empty and 'average_treatment_duration_days' in average_duration_by_pathway.columns:
            info_or_chart(
                plot_bar(average_duration_by_pathway.head(10), 'pathway_label', 'average_treatment_duration_days', 'Average Duration by Pathway'),
                'No pathway duration chart is available.',
            )
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
        st.markdown('#### Poor-Outcome Pathways to Review')
        info_or_table(poor_outcome_table, 'No poor-outcome pathways were identified for the current dataset.')
        st.markdown('#### Suggested Pathway Actions')
        if pathway.get('action_recommendations'):
            for line in pathway.get('action_recommendations', []):
                st.write(f'- {line}')
        else:
            st.info('Suggested pathway actions appear here when pathway duration or outcome patterns highlight a review priority.')
    render_standards(pipeline, key_prefix='healthcare_standards')
    render_privacy(pipeline)

def render_cohort_analysis(pipeline: dict[str, Any]) -> None:
    render_section_intro(
        'Cohort Analysis',
        'Compare focused patient populations against the broader dataset, examine outcome differences, and preserve exportable cohort definitions for follow-up review.',
    )
    render_surface_panel(
        'Cohort workflow',
        'Use cohort filters to isolate a population, compare it with the overall dataset, and keep the resulting definition ready for export or follow-up review.',
        tone='accent',
    )
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
    cols3 = st.columns(2)
    smoking_col = canonical_map.get('smoking_status')
    if smoking_col and smoking_col in data.columns:
        filters['smoking_status'] = cols3[0].multiselect('Smoking status', sorted(data[smoking_col].dropna().astype(str).unique().tolist())[:12], key='cohort_smoking')
    comorbidity_col = canonical_map.get('comorbidities')
    if comorbidity_col and comorbidity_col in data.columns:
        filters['comorbidity'] = cols3[1].multiselect('Comorbidity burden', ['Present', 'Absent'], key='cohort_comorbidity')
    debounced = resolve_debounced_filters(st.session_state, 'cohort_analysis_filters', filters)
    applied_filters = debounced.applied_filters
    selected_filter_count = sum(
        1
        for value in applied_filters.values()
        if value is not None and value != () and value != [] and value != ''
    )
    if selected_filter_count == 0:
        st.info('Select a diagnosis to explore risk segments. You can also start with treatment, cancer stage, gender, or risk segment filters to focus the cohort review.')
        placeholder_frame, placeholder_title = build_cohort_placeholder_frame(
            data,
            diagnosis_col=diagnosis_col,
            treatment_col=treatment_col,
            risk_table=risk_table,
        )
        if not placeholder_frame.empty:
            placeholder_x = next((column for column in ['diagnosis', 'treatment', 'risk_segment'] if column in placeholder_frame.columns), placeholder_frame.columns[0])
            placeholder_y = next((column for column in ['record_count', 'patient_count'] if column in placeholder_frame.columns), placeholder_frame.columns[-1])
            info_or_chart(
                plot_bar(placeholder_frame, placeholder_x, placeholder_y, placeholder_title),
                'Cohort starter chart will appear here once the current dataset has at least one usable grouping field.',
            )
            info_or_table(placeholder_frame, 'Cohort starter table will appear here when the platform finds diagnosis, treatment, or risk-group fields.')
    cohort_cache_key = build_visual_cache_key(
        pipeline,
        'cohort_analysis',
        extra=str(tuple(sorted((key, str(value)) for key, value in applied_filters.items()))),
    )
    payload = get_cached_visual_payload(st.session_state, cohort_cache_key)
    if payload is None:
        skeleton = _show_skeleton_loader('Preparing cohort comparisons and monitoring trends...', blocks=4)
        payload = store_visual_payload_for_pipeline(st.session_state, pipeline, cohort_cache_key, build_cohort_analysis_payload(pipeline, applied_filters))
        skeleton.empty()
    cache_diagnostics = build_visual_cache_diagnostics(st.session_state, pipeline, cohort_cache_key)
    st.session_state.setdefault('active_dataset_diagnostics', {}).update(
        {
            'cohort_cache_key': cohort_cache_key,
            'cohort_cache_belongs_to_current_dataset': bool(cache_diagnostics.get('cache_belongs_to_current_dataset')),
        }
    )
    if debounced.pending:
        st.caption('Updating cohort filters...')
        time.sleep(min(debounced.remaining_seconds, 0.35))
        st.rerun()
    cohort = payload.get('cohort', {})
    if not cohort.get('available'):
        st.info(cohort.get('reason', 'Cohort building is not available for the current dataset.'))
        return
    summary = cohort.get('summary', {})
    if selected_filter_count == 0:
        st.caption('You are currently reviewing the overall population. Add one or more cohort filters above to compare a narrower group against the full dataset.')
    if cohort.get('summary_narrative'):
        st.caption(str(cohort.get('summary_narrative')))
    metric_row([
        ('Cohort Size', fmt(summary.get('cohort_size', 0))),
        ('Survival Rate', fmt(summary.get('survival_rate'), 'pct') if summary.get('survival_rate') is not None else 'Not available'),
        ('Readmission Rate', fmt(summary.get('readmission_rate'), 'pct') if summary.get('readmission_rate') is not None else 'Not available'),
        ('High-Risk Share', fmt(summary.get('high_risk_share'), 'pct') if summary.get('high_risk_share') is not None else 'Not available'),
        ('Avg Treatment Duration', fmt(summary.get('average_treatment_duration_days'), 'float') if summary.get('average_treatment_duration_days') is not None else 'Not available'),
    ])
    st.markdown('#### Cohort vs Overall Population')
    comparison_table = safe_df(cohort.get('comparison_table'))
    info_or_table(comparison_table, 'Cohort comparison appears here when the dataset supports side-by-side risk, outcome, or duration benchmarking.')
    if show_charts := _lazy_chart_enabled('Render cohort charts', 'cohort_lazy_charts', data):
        comparison_chart = comparison_table[
            comparison_table['metric'].isin(['Readmission Rate', 'Survival Rate', 'High-Risk Share'])
        ] if not comparison_table.empty else pd.DataFrame()
        if not comparison_chart.empty:
            chart_frame = comparison_chart.copy()
            chart_frame['gap_vs_overall_pct'] = chart_frame['gap_vs_overall'] * 100.0
            info_or_chart(
                plot_bar(chart_frame, 'metric', 'gap_vs_overall_pct', 'Cohort Gap vs Overall Population'),
                'No cohort comparison chart is available.',
            )
    st.markdown('#### Cohort Risk Summary')
    info_or_table(safe_df(cohort.get('risk_distribution', cohort.get('risk_distribution_table'))), 'Risk distribution appears here when the dataset includes risk segmentation support.')
    st.markdown('#### Demographic Breakdown')
    demographic_breakdown = safe_df(cohort.get('demographic_breakdown_table'))
    info_or_table(demographic_breakdown, 'Demographic breakdown appears here once the cohort includes age, gender, smoking, stage, treatment, or comorbidity signals.')
    if show_charts and not demographic_breakdown.empty:
        demo_chart = demographic_breakdown.sort_values(['cohort_share', 'cohort_count'], ascending=[False, False]).head(10).copy()
        demo_chart['cohort_share_pct'] = demo_chart['cohort_share'] * 100.0
        info_or_chart(
            plot_bar(demo_chart, 'segment', 'cohort_share_pct', 'Top Cohort Demographic Segments'),
            'No cohort demographic chart is available.',
        )
    st.markdown('#### Cohort Outcome Summary')
    info_or_table(safe_df(cohort.get('outcome_metrics_table', cohort.get('subgroup_metrics'))), 'Outcome summary appears here when the dataset includes outcome or treatment support.')
    st.markdown('#### Outcome Differences vs Overall Population')
    outcome_difference_table = safe_df(cohort.get('outcome_difference_table'))
    info_or_table(outcome_difference_table, 'Outcome difference summary appears here when the cohort has measurable outcome, risk, or duration signals.')
    if show_charts and not outcome_difference_table.empty:
        outcome_chart = outcome_difference_table.copy()
        outcome_chart['gap_vs_overall_pct'] = outcome_chart['gap_vs_overall'].apply(lambda value: float(value) * 100.0 if pd.notna(value) else value)
        info_or_chart(
            plot_bar(outcome_chart, 'metric', 'gap_vs_overall_pct', 'Outcome Difference vs Overall Population'),
            'No outcome-difference chart is available.',
        )
    trend = safe_df(cohort.get('cohort_trend_table'))
    if not trend.empty and 'month' in trend.columns:
        st.markdown('#### Cohort Trend Summary')
        for line in cohort.get('trend_summary', []):
            st.write(f'- {line}')
        if show_charts:
            info_or_chart(plot_time_trend(trend, 'month', 'record_count', 'Selected Cohort Volume Over Time'), 'No cohort trend chart is available.')
            if 'survival_rate' in trend.columns:
                info_or_chart(plot_time_trend(trend, 'month', 'survival_rate', 'Selected Cohort Outcome Trend'), 'No cohort outcome trend chart is available.')
            if 'high_risk_share' in trend.columns:
                info_or_chart(plot_time_trend(trend, 'month', 'high_risk_share', 'Selected Cohort High-Risk Trend'), 'No cohort risk trend chart is available.')
        else:
            st.caption('Charts are lazy-loaded for faster cohort tab switching. Enable chart rendering when you want the visuals.')
        info_or_table(trend, 'No cohort timeline table is available.')
    else:
        st.info('Cohort trend summaries unlock when the dataset includes diagnosis, admission, service, or treatment-end dates.')
    if cohort.get('suggested_next_steps'):
        st.markdown('#### Suggested Cohort Actions')
        for line in cohort.get('suggested_next_steps', []):
            st.write(f'- {line}')
    else:
        st.info('Suggested cohort actions appear here when the current filters reveal a meaningful risk, outcome, or duration difference versus the overall population.')
    info_or_table(safe_df(cohort.get('preview')), 'No cohort preview is available.')
    st.markdown('### Export Cohort Definition')
    cohort_definition = cohort.get('cohort_definition', {})
    info_or_table(safe_df(cohort_definition.get('filter_table')), 'Active cohort filters will appear here after one or more cohort filters are selected.')
    export_cols = st.columns(2)
    definition_json = json.dumps(
        {
            'dataset_name': pipeline.get('overview', {}).get('dataset_name', 'Current dataset'),
            'active_filters': cohort_definition.get('active_filters', {}),
            'summary': cohort_definition.get('summary', {}),
            'baseline_summary': cohort_definition.get('baseline_summary', {}),
        },
        indent=2,
        default=str,
    )
    export_cols[0].download_button(
        'Download Cohort Definition (JSON)',
        data=definition_json.encode('utf-8'),
        file_name='cohort_definition.json',
        mime='application/json',
        key='download_cohort_definition_json',
    )
    export_cols[1].download_button(
        'Download Cohort Definition (CSV)',
        data=safe_df(cohort_definition.get('filter_table')).to_csv(index=False).encode('utf-8'),
        file_name='cohort_definition.csv',
        mime='text/csv',
        key='download_cohort_definition_csv',
    )
    st.markdown('### Cohort Monitoring Over Time')
    monitoring = payload.get('monitoring', {})
    if monitoring.get('available'):
        monitoring_table = safe_df(monitoring.get('trend_table'))
        info_or_table(monitoring_table, 'No cohort monitoring trend table is available.')
        if show_charts and not monitoring_table.empty and 'month' in monitoring_table.columns:
            info_or_chart(
                plot_time_trend(monitoring_table, 'month', 'record_count', 'Cohort vs Overall Timeline'),
                'No cohort monitoring chart is available.',
            )
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
            st.caption(f"Target cohort: {intervention.get('target_cohort', selected_readmission_cohort)}")
            info_or_table(safe_df(intervention.get('summary_table')), 'No intervention summary is available.')
            st.write('Assumptions:')
            for line in intervention.get('assumptions', []):
                st.write(f'- {line}')
        else:
            st.info(intervention.get('reason', 'Readmission intervention planning is not available for the selected cohort.'))
    else:
        st.info(readmission_cohort.get('reason', 'Readmission cohort review is not available for the selected cohort.'))

def render_trend_analysis(pipeline: dict[str, Any]) -> None:
    render_section_intro(
        'Trend Analysis',
        'Track time-based clinical and operational patterns with a lighter, faster workspace tuned for repeated analysis and executive-ready chart review.',
    )
    render_surface_panel(
        'Trend workspace',
        'Trend Analysis keeps cached payloads and chart loading lightweight so repeated navigation remains fast, especially on large datasets.',
        tone='info',
    )
    data = pipeline['data']
    trend_cache_key = build_visual_cache_key(pipeline, 'trend_analysis')
    payload = get_cached_visual_payload(st.session_state, trend_cache_key)
    if payload is None:
        skeleton = _show_skeleton_loader('Preparing trend summaries and chart inputs...', blocks=4)
        payload = store_visual_payload_for_pipeline(st.session_state, pipeline, trend_cache_key, build_trend_analysis_payload(pipeline))
        skeleton.empty()
    cache_diagnostics = build_visual_cache_diagnostics(st.session_state, pipeline, trend_cache_key)
    st.session_state.setdefault('active_dataset_diagnostics', {}).update(
        {
            'trend_cache_key': trend_cache_key,
            'trend_cache_belongs_to_current_dataset': bool(cache_diagnostics.get('cache_belongs_to_current_dataset')),
        }
    )
    if not payload.get('date_col'):
        st.info('No reliable date field was detected, so time-series analysis is limited. Add or map a usable date field to unlock stronger longitudinal review.')
        return
    monthly = safe_df(payload.get('monthly'))
    show_charts = _lazy_chart_enabled('Render trend charts', 'trend_lazy_charts', data)
    if monthly.empty:
        st.info('No usable rows remain after parsing the default date field.')
    else:
        if show_charts:
            info_or_chart(plot_time_trend(monthly, 'month', 'record_count', 'Dataset Record Volume Over Time'), 'No record trend is available yet.')
        else:
            st.caption('Charts are lazy-loaded for faster trend tab switching. Enable chart rendering when you want the visuals.')
        info_or_table(monthly, 'No trend table is available yet.')
    if payload.get('survival_available'):
        for label, table_key in [('Survival by stage', 'stage_table'), ('Survival by treatment', 'treatment_table'), ('Treatment duration distribution', 'duration_distribution'), ('Outcome trend over time', 'outcome_trend'), ('Treatment duration trend', 'treatment_duration_trend'), ('Disease progression timeline', 'progression_timeline')]:
            st.markdown(f'### {label}')
            table = safe_df(payload.get('survival_tables', {}).get(table_key))
            info_or_table(table, f'{label} is not available for the current dataset.')
    readmission = pipeline['healthcare'].get('readmission', {})
    st.markdown('### Readmission Trend')
    readmission_trend = safe_df(payload.get('readmission_trend'))
    if payload.get('readmission_available') and not readmission_trend.empty:
        if show_charts:
            info_or_chart(plot_time_trend(readmission_trend, 'month', 'readmission_rate', 'Readmission Rate Over Time'), 'No readmission trend chart is available.')
        info_or_table(readmission_trend, 'No readmission trend table is available.')
    else:
        _render_readmission_readiness(readmission)
        st.info(readmission.get('reason', 'Readmission trend analysis is not available for the current dataset.'))
    st.markdown('### Correlation Heatmap')
    correlation = safe_df(payload.get('correlation'))
    if show_charts:
        info_or_chart(plot_correlation(correlation), 'No correlation heatmap is available because the dataset lacks sufficient numeric fields.')
    else:
        st.caption('Correlation heatmap will render on demand when chart loading is enabled.')

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
    job_runtime = pipeline.get('job_runtime', {})
    st.caption(build_job_user_message(job_runtime, 'predictive_modeling'))
    plan_awareness = pipeline.get('plan_awareness', {})
    if not plan_feature_enabled(str(plan_awareness.get('active_plan', 'Pro')), 'predictive_modeling'):
        st.caption('Predictive Modeling Studio is packaged as a premium capability. It remains visible in demo-safe mode so you can review the workflow before enabling it in a paid plan.')
        if bool(plan_awareness.get('strict_enforcement')):
            st.info('Predictive Modeling Studio is unavailable under the current plan while strict enforcement is enabled. Switch to Demo-safe mode or upgrade the plan to continue.')
            return
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

    modeling_status = st.empty()

    def _job_progress(value: float, message: str) -> None:
        modeling_status.caption(f'Managed task progress: {int(max(0.0, min(1.0, value)) * 100)}% · {message}')

    comparison_job = submit_job(
        st.session_state,
        job_runtime,
        task_key='predictive_modeling',
        task_label='Predictive modeling comparison',
        detail='Prepared comparison-ready candidate models for the current target and feature selection.',
        stage_messages=[
            'Preparing modeling inputs...',
            'Evaluating candidate models...',
            'Model comparison is ready.',
        ],
        progress_callback=_job_progress,
        runner=lambda: build_model_comparison_studio(
            pipeline['data'],
            pipeline['semantic']['canonical_map'],
            target_column,
            selected_features,
        ),
    )
    comparison = get_job_result(st.session_state, comparison_job['job_id'])
    comparison_status = get_job_status(st.session_state, comparison_job['job_id'])
    modeling_status.caption(f"Managed task status: {comparison_status.get('status', 'completed').title()} · Model comparison output is ready for review.")
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

    model_job = submit_job(
        st.session_state,
        job_runtime,
        task_key='predictive_modeling',
        task_label='Predictive model build',
        detail=f'Built {model_type} for {target_column}.',
        stage_messages=[
            'Preparing model features...',
            'Training model and computing metrics...',
            'Predictive model output is ready.',
        ],
        progress_callback=_job_progress,
        runner=lambda: build_predictive_model(pipeline['data'], pipeline['semantic']['canonical_map'], target_column, selected_features, model_type),
    )
    result = get_job_result(st.session_state, model_job['job_id'])
    model_status = get_job_status(st.session_state, model_job['job_id'])
    modeling_status.caption('Managed task progress: 100% · Predictive modeling outputs are ready for review.')
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
    recent_jobs = build_job_status_view(st.session_state.get('job_runs', [])[:3])
    if not recent_jobs.empty:
        st.markdown('#### Recent Managed Tasks')
        info_or_table(recent_jobs, 'Managed task history will appear here after predictive modeling or export jobs complete.')

