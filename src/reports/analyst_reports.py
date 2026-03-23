from __future__ import annotations

from io import StringIO

import pandas as pd

from src.reports.common import (
    _analyzed_columns,
    _combine_report_tables,
    _safe_df,
    _source_columns,
    dataframe_to_csv_bytes,
    normalize_report_mode,
)


def build_cross_setting_reporting_profile(
    dataset_intelligence: dict[str, object],
    use_case_detection: dict[str, object],
    solution_packages: dict[str, object],
    selected_report_mode: str,
) -> pd.DataFrame:
    capability = _safe_df(dataset_intelligence.get('analytics_capability_matrix'))
    recommended_workflow = str(use_case_detection.get('recommended_workflow', 'Healthcare Data Readiness'))
    recommended_package = str(solution_packages.get('recommended_package', 'Healthcare Data Readiness'))
    selected_report_mode = normalize_report_mode(selected_report_mode)

    rows = [
        {
            'review_type': 'Operational review',
            'recommended_report_mode': 'Operational Report',
            'best_for': 'Utilization, alerts, provider concentration, and operational planning.',
            'workflow_fit': 'Hospital Operations Analytics',
            'required_modules': ['Provider / Facility Volume', 'Trend Analysis', 'Readmission Analytics'],
        },
        {
            'review_type': 'Clinical review',
            'recommended_report_mode': 'Clinical Report',
            'best_for': 'Outcome patterns, pathway summaries, subgroup gaps, and readmission context.',
            'workflow_fit': 'Clinical Outcome Analytics',
            'required_modules': ['Risk Segmentation', 'Cohort Analysis', 'Care Pathway Intelligence'],
        },
        {
            'review_type': 'Data readiness review',
            'recommended_report_mode': 'Data Readiness Review',
            'best_for': 'Readiness blockers, quality gaps, remediation priorities, and governance follow-up.',
            'workflow_fit': 'Healthcare Data Readiness',
            'required_modules': ['Data Profiling', 'Data Quality Review', 'Standards / Governance Review'],
        },
        {
            'review_type': 'Population health summary',
            'recommended_report_mode': 'Population Health Summary',
            'best_for': 'Segment risk mix, disparity signals, population trends, and outreach-style planning.',
            'workflow_fit': 'Population Health Intelligence',
            'required_modules': ['Risk Segmentation', 'Cohort Monitoring Over Time', 'Trend Analysis'],
        },
    ]

    if not capability.empty and {'analytics_module', 'status', 'support', 'rationale'}.issubset(capability.columns):
        lookup = capability.set_index('analytics_module')
        for row in rows:
            matched = lookup.reindex(row['required_modules']).dropna(how='all')
            statuses = matched['status'].astype(str).str.lower().tolist() if not matched.empty else []
            supports = matched['support'].astype(str).str.lower().tolist() if not matched.empty else []
            if any(status == 'enabled' for status in statuses):
                status_label = 'Ready now'
            elif any(status == 'partial' for status in statuses):
                status_label = 'Partially supported'
            else:
                status_label = 'Needs stronger source support'

            support_label = 'native'
            if any('synthetic' in support for support in supports):
                support_label = 'synthetic-assisted'
            elif any('inferred' in support for support in supports):
                support_label = 'inferred'
            elif not supports:
                support_label = 'unavailable'

            blocked_row = matched[matched['status'].astype(str).str.lower() != 'enabled'].head(1)
            rationale = (
                str(blocked_row.iloc[0]['rationale'])
                if not blocked_row.empty and 'rationale' in blocked_row.columns
                else 'The core workflow modules are currently available.'
            )
            row['status'] = status_label
            row['support_type'] = support_label
            row['why_this_matters'] = rationale
    else:
        for row in rows:
            row['status'] = 'Needs stronger source support'
            row['support_type'] = 'unavailable'
            row['why_this_matters'] = 'Capability evidence is limited for this dataset.'

    for row in rows:
        row['workflow_alignment'] = 'Recommended now' if row['workflow_fit'] == recommended_workflow else 'Available option'
        row['selected_report'] = 'Current selection' if normalize_report_mode(row['recommended_report_mode']) == selected_report_mode else ''
        row['recommended_package'] = recommended_package if row['workflow_fit'] == recommended_workflow else ''

    return pd.DataFrame(rows)[[
        'review_type',
        'recommended_report_mode',
        'status',
        'support_type',
        'workflow_alignment',
        'best_for',
        'why_this_matters',
        'selected_report',
        'recommended_package',
    ]]


def build_report_support_tables(report_mode: str, overview: dict[str, object], quality: dict[str, object], readiness: dict[str, object], healthcare: dict[str, object], action_recommendations: pd.DataFrame) -> dict[str, pd.DataFrame]:
    normalized_mode = normalize_report_mode(report_mode)
    overview_table = pd.DataFrame([
        {'metric': 'Rows', 'value': overview['rows']},
        {'metric': 'Analyzed columns', 'value': _analyzed_columns(overview)},
        {'metric': 'Duplicate rows', 'value': overview['duplicate_rows']},
        {'metric': 'Missing values', 'value': overview['missing_values']},
    ])

    tables: dict[str, pd.DataFrame] = {'Dataset Overview': overview_table}

    if normalized_mode == 'Analyst Report':
        if not quality.get('high_missing', pd.DataFrame()).empty:
            tables['High Missingness Fields'] = quality['high_missing'].head(10)
        if not readiness.get('readiness_table', pd.DataFrame()).empty:
            tables['Analysis Readiness'] = readiness['readiness_table']
        anomaly = healthcare.get('anomaly_detection', {})
        if anomaly.get('available') and not anomaly.get('summary_table', pd.DataFrame()).empty:
            tables['Anomaly Summary'] = anomaly['summary_table'].head(10)
    elif normalized_mode == 'Data Readiness Review':
        if not readiness.get('readiness_table', pd.DataFrame()).empty:
            tables['Analysis Readiness'] = readiness['readiness_table']
        if not quality.get('high_missing', pd.DataFrame()).empty:
            tables['High Missingness Fields'] = quality['high_missing'].head(10)
        if not action_recommendations.empty:
            tables['Recommended Actions'] = action_recommendations.head(8)
    elif normalized_mode == 'Operational Report':
        alerts = healthcare.get('operational_alerts_preview', {})
        if alerts.get('available') and not alerts.get('alerts_table', pd.DataFrame()).empty:
            tables['Operational Alerts'] = alerts['alerts_table'].head(10)
        risk = healthcare.get('risk_segmentation', {})
        if risk.get('available') and not risk.get('segment_table', pd.DataFrame()).empty:
            tables['Risk Segment Mix'] = risk['segment_table']
        if not action_recommendations.empty:
            tables['Recommended Actions'] = action_recommendations.head(8)
    elif normalized_mode == 'Executive Summary':
        risk = healthcare.get('risk_segmentation', {})
        if risk.get('available') and not risk.get('segment_table', pd.DataFrame()).empty:
            tables['Headline Risk Summary'] = risk['segment_table'].head(5)
        if not action_recommendations.empty:
            tables['Priority Actions'] = action_recommendations.head(5)
    elif normalized_mode == 'Population Health Summary':
        risk = healthcare.get('risk_segmentation', {})
        if risk.get('available') and not risk.get('segment_table', pd.DataFrame()).empty:
            tables['Population Risk Mix'] = risk['segment_table']
        fairness = healthcare.get('explainability_fairness', {})
        if fairness.get('available') and not fairness.get('fairness_flags', pd.DataFrame()).empty:
            tables['Population Gap Flags'] = fairness['fairness_flags'].head(10)
        if not action_recommendations.empty:
            tables['Recommended Actions'] = action_recommendations.head(6)
    else:
        survival = healthcare.get('survival_outcomes', {})
        if survival.get('available') and not survival.get('stage_table', pd.DataFrame()).empty:
            tables['Survival by Stage'] = survival['stage_table']
        if survival.get('available') and not survival.get('treatment_table', pd.DataFrame()).empty:
            tables['Survival by Treatment'] = survival['treatment_table']
        fairness = healthcare.get('explainability_fairness', {})
        if fairness.get('available') and not fairness.get('fairness_flags', pd.DataFrame()).empty:
            tables['Subgroup Gap Flags'] = fairness['fairness_flags']

    return tables


def build_report_support_csv(report_mode: str, overview: dict[str, object], quality: dict[str, object], readiness: dict[str, object], healthcare: dict[str, object], action_recommendations: pd.DataFrame) -> bytes:
    tables = build_report_support_tables(report_mode, overview, quality, readiness, healthcare, action_recommendations)
    combined = _combine_report_tables(tables)
    return dataframe_to_csv_bytes(combined)


def build_audience_report_text(
    report_mode: str,
    dataset_name: str,
    overview: dict[str, object],
    structure,
    quality: dict[str, object],
    semantic: dict[str, object],
    readiness: dict[str, object],
    healthcare: dict[str, object],
    insights: dict[str, object],
    action_recommendations: pd.DataFrame,
    trust_summary: dict[str, object] | None = None,
    accuracy_summary: dict[str, object] | None = None,
) -> bytes:
    normalized_mode = normalize_report_mode(report_mode)
    trust_summary = trust_summary or {}
    accuracy_summary = accuracy_summary or {}
    uncertainty = accuracy_summary.get('uncertainty_narrative', {}) if isinstance(accuracy_summary, dict) else {}
    reporting_policy = accuracy_summary.get('reporting_policy', {}) if isinstance(accuracy_summary, dict) else {}
    benchmark_profile = accuracy_summary.get('benchmark_profile', {}) if isinstance(accuracy_summary, dict) else {}
    buffer = StringIO()
    buffer.write(f'{normalized_mode}\n')
    buffer.write('=' * len(normalized_mode) + '\n\n')
    buffer.write(f'Dataset: {dataset_name}\n')
    buffer.write(f"Rows: {overview['rows']:,} | Analyzed columns: {_analyzed_columns(overview):,} | Duplicate rows: {overview['duplicate_rows']:,}\n")
    if _source_columns(overview) != _analyzed_columns(overview):
        buffer.write(f"Source columns: {_source_columns(overview):,}\n")
    buffer.write(f"Missing values: {overview['missing_values']:,} | Analysis readiness: {readiness['readiness_score']:.0%} | Healthcare readiness: {healthcare['healthcare_readiness_score']:.0%}\n\n")
    if trust_summary:
        buffer.write('Accuracy and Disclosure Context\n')
        buffer.write('-' * 30 + '\n')
        buffer.write(f"Trust level: {trust_summary.get('trust_level', 'Unknown')} | Interpretation: {trust_summary.get('interpretation_mode', 'Unknown')}\n")
        buffer.write(f"Benchmark profile: {benchmark_profile.get('profile_name', 'Generic Healthcare')} | Reporting policy: {reporting_policy.get('profile_name', 'Standard')}\n")
        buffer.write(f"{uncertainty.get('headline', trust_summary.get('summary_text', 'Directional caveats may still apply to parts of this workflow.'))}\n\n")
    buffer.write('Collaboration Notes\n')
    buffer.write('-' * 19 + '\n')
    buffer.write('This version is tailored to support cross-functional review and can be shared directly with the intended audience.\n\n')

    key_findings: list[str] = []
    appendix_lines: list[str] = []

    if normalized_mode == 'Analyst Report':
        buffer.write('Analyst Focus\n')
        buffer.write('-' * 13 + '\n')
        buffer.write(f"Detected numeric fields: {len(structure.numeric_columns)}\n")
        buffer.write(f"Detected date fields: {len(structure.date_columns)}\n")
        buffer.write(f"Semantic mappings in use: {len(semantic['canonical_map'])}\n")
        buffer.write(f"Ready analysis modules: {int((readiness['readiness_table']['status'] == 'Available').sum())}\n")
        if not quality['high_missing'].empty:
            top_missing = quality['high_missing'].iloc[0]
            finding = f"Highest missingness is in {top_missing['column_name']} at {top_missing['null_percentage']:.1%}."
            buffer.write(finding + '\n')
            key_findings.append(finding)
        anomaly = healthcare.get('anomaly_detection', {})
        if anomaly.get('available') and not anomaly['summary_table'].empty:
            top_anomaly = anomaly['summary_table'].iloc[0]
            finding = f"Top anomaly field is {top_anomaly['field']} with {int(top_anomaly['anomaly_count'])} flagged values."
            buffer.write(finding + '\n')
            key_findings.append(finding)
            appendix_lines.append(f"Anomaly summary leader: {top_anomaly['field']} ({int(top_anomaly['anomaly_count'])} flags)")
        mixed_type_suspects = quality.get('mixed_type_suspects', pd.DataFrame())
        if not mixed_type_suspects.empty:
            suspect = mixed_type_suspects.iloc[0]
            finding = f"Mixed-type watch item: {suspect['column_name']}."
            buffer.write(finding + '\n')
            key_findings.append(finding)
        appendix_lines.append(
            f"Detected field types: {len(getattr(structure, 'numeric_columns', []))} numeric, "
            f"{len(getattr(structure, 'date_columns', []))} date, "
            f"{len(getattr(structure, 'categorical_columns', []))} categorical"
        )
    elif normalized_mode == 'Data Readiness Review':
        buffer.write('Data Readiness Focus\n')
        buffer.write('-' * 20 + '\n')
        ready_modules = int((readiness['readiness_table']['status'] == 'Available').sum()) if not readiness.get('readiness_table', pd.DataFrame()).empty else 0
        buffer.write(f"Ready analysis modules: {ready_modules}\n")
        buffer.write(f"Semantic mappings in use: {len(semantic['canonical_map'])}\n")
        if not quality['high_missing'].empty:
            top_missing = quality['high_missing'].iloc[0]
            finding = f"Highest-impact readiness gap is {top_missing['column_name']} at {top_missing['null_percentage']:.1%} missing."
            buffer.write(finding + '\n')
            key_findings.append(finding)
        if not readiness.get('readiness_table', pd.DataFrame()).empty:
            blocked = readiness['readiness_table'][readiness['readiness_table']['status'].astype(str) != 'Available']
            if not blocked.empty:
                finding = f"{len(blocked)} modules still need stronger source support or remediation."
                buffer.write(finding + '\n')
                key_findings.append(finding)
                appendix_lines.append(f"Blocked or partial modules: {len(blocked)}")
        appendix_lines.append(f"Readiness score: {readiness['readiness_score']:.0%}")
    elif normalized_mode == 'Operational Report':
        buffer.write('Manager Focus\n')
        buffer.write('-' * 13 + '\n')
        util = healthcare.get('utilization', {})
        if util.get('available'):
            finding = f"Average events per entity: {util['average_events_per_entity']:.2f}."
            buffer.write(finding + '\n')
            key_findings.append(finding)
        risk = healthcare.get('risk_segmentation', {})
        if risk.get('available') and not risk['segment_table'].empty:
            top_segment = risk['segment_table'].sort_values('patient_count', ascending=False).iloc[0]
            finding = f"Largest risk segment: {top_segment['risk_segment']} ({int(top_segment['patient_count'])} records)."
            buffer.write(finding + '\n')
            key_findings.append(finding)
            appendix_lines.append(f"Largest risk segment share: {float(top_segment['percentage']):.1%}")
        monitoring = healthcare.get('operational_alerts_preview', {})
        if monitoring.get('available') and not monitoring['alerts_table'].empty:
            finding = f"Active operational alerts: {len(monitoring['alerts_table'])}."
            buffer.write(finding + '\n')
            key_findings.append(finding)
            appendix_lines.append(f"Top alert: {monitoring['alerts_table'].iloc[0]['alert_title']}")
        if not action_recommendations.empty:
            buffer.write(f"Priority action: {action_recommendations.iloc[0]['recommendation_title']}\n")
    elif normalized_mode == 'Executive Summary':
        buffer.write('Executive Focus\n')
        buffer.write('-' * 15 + '\n')
        risk = healthcare.get('risk_segmentation', {})
        if risk.get('available') and risk.get('survival_rate') is not None:
            finding = f"Overall survival rate is {risk['survival_rate']:.1%}."
            buffer.write(finding + '\n')
            key_findings.append(finding)
        if not action_recommendations.empty:
            first = action_recommendations.iloc[0]
            finding = f"Top action: {first['recommendation_title']} ({first['priority']})."
            buffer.write(finding + '\n')
            key_findings.append(finding)
        for line in insights['summary_lines'][:3]:
            buffer.write(f"- {line}\n")
            key_findings.append(line)
        if not action_recommendations.empty:
            appendix_lines.append(f"Priority recommendation count: {len(action_recommendations)}")
    elif normalized_mode == 'Population Health Summary':
        buffer.write('Population Health Focus\n')
        buffer.write('-' * 23 + '\n')
        risk = healthcare.get('risk_segmentation', {})
        if risk.get('available') and not risk.get('segment_table', pd.DataFrame()).empty:
            top_segment = risk['segment_table'].sort_values('patient_count', ascending=False).iloc[0]
            finding = f"Largest monitored segment is {top_segment['risk_segment']} with {int(top_segment['patient_count'])} records."
            buffer.write(finding + '\n')
            key_findings.append(finding)
        fairness = healthcare.get('explainability_fairness', {})
        if fairness.get('available') and not fairness.get('fairness_flags', pd.DataFrame()).empty:
            finding = f"Population disparity flags identified: {len(fairness['fairness_flags'])}."
            buffer.write(finding + '\n')
            key_findings.append(finding)
            appendix_lines.append(f"Fairness or disparity flags: {len(fairness['fairness_flags'])}")
        for line in healthcare.get('ai_insight_summary', [])[:2]:
            buffer.write(f"- {line}\n")
            key_findings.append(line)
    else:
        buffer.write('Clinical Focus\n')
        buffer.write('-' * 14 + '\n')
        survival = healthcare.get('survival_outcomes', {})
        if survival.get('available') and not survival.get('stage_table', pd.DataFrame()).empty:
            weakest = survival['stage_table'].iloc[0]
            finding = f"Lowest observed stage-level survival: {weakest[survival['stage_column']]} at {weakest['survival_rate']:.1%}."
            buffer.write(finding + '\n')
            key_findings.append(finding)
            appendix_lines.append(f"Stage rows reviewed: {len(survival['stage_table'])}")
        if survival.get('available') and not survival.get('treatment_table', pd.DataFrame()).empty:
            strongest = survival['treatment_table'].iloc[0]
            finding = f"Best observed treatment outcome: {strongest[survival['treatment_column']]} at {strongest['survival_rate']:.1%}."
            buffer.write(finding + '\n')
            key_findings.append(finding)
            appendix_lines.append(f"Treatment rows reviewed: {len(survival['treatment_table'])}")
        fairness = healthcare.get('explainability_fairness', {})
        if fairness.get('available') and not fairness.get('fairness_flags', pd.DataFrame()).empty:
            finding = f"Material subgroup gaps flagged: {len(fairness['fairness_flags'])}."
            buffer.write(finding + '\n')
            key_findings.append(finding)
        for line in healthcare.get('ai_insight_summary', [])[:3]:
            buffer.write(f"- {line}\n")
            key_findings.append(line)

    buffer.write('\nKey Findings\n')
    buffer.write('-' * 12 + '\n')
    if key_findings:
        for item in key_findings[:5]:
            buffer.write(f"- {item}\n")
    else:
        for line in insights['summary_lines'][:4]:
            buffer.write(f"- {line}\n")

    buffer.write('\nShared Review Priorities\n')
    buffer.write('-' * 23 + '\n')
    if not action_recommendations.empty:
        for _, row in action_recommendations.head(4).iterrows():
            buffer.write(f"- [{row['priority']}] {row['recommendation_title']}: {row['rationale']}\n")
    else:
        for line in insights['recommendations'][:4]:
            buffer.write(f"- {line}\n")

    buffer.write('\nAppendix-Style Supporting Notes\n')
    buffer.write('-' * 31 + '\n')
    if appendix_lines:
        for item in appendix_lines[:5]:
            buffer.write(f"- {item}\n")
    else:
        buffer.write(f"- Semantic mappings in use: {len(semantic['canonical_map'])}\n")
        buffer.write(f"- Ready analysis modules: {int((readiness['readiness_table']['status'] == 'Available').sum())}\n")
        buffer.write(f"- Duplicate rows: {overview['duplicate_rows']:,}\n")

    buffer.write('\nSuggested Next Review Step\n')
    buffer.write('-' * 26 + '\n')
    if normalized_mode == 'Analyst Report':
        buffer.write('Use the Data Quality Review and Semantic Mapping sections to validate the data foundation before sharing results more broadly.\n')
    elif normalized_mode == 'Data Readiness Review':
        buffer.write('Use Analysis Readiness, Data Remediation Assistant, and Standards / Governance Review to close the highest-impact blockers first.\n')
    elif normalized_mode == 'Operational Report':
        buffer.write('Use Benchmarking and Cohort Monitoring Over Time to compare the current population against baseline performance.\n')
    elif normalized_mode == 'Executive Summary':
        buffer.write('Use Presentation Mode and Action Recommendations to support a concise stakeholder discussion on priorities and trade-offs.\n')
    elif normalized_mode == 'Population Health Summary':
        buffer.write('Use Cohort Analysis and Trend Analysis to follow the largest at-risk population segments over time and prioritize outreach-style reviews.\n')
    else:
        buffer.write('Use Survival & Outcome Analysis together with Explainability & Fairness to review the clinical implications of the current outcomes.\n')

    return buffer.getvalue().encode('utf-8')


def build_generated_report_text(
    report_label: str,
    dataset_name: str,
    overview: dict[str, object],
    quality: dict[str, object],
    readiness: dict[str, object],
    healthcare: dict[str, object],
    insights: dict[str, object],
    action_recommendations: pd.DataFrame,
    trust_summary: dict[str, object] | None = None,
    accuracy_summary: dict[str, object] | None = None,
) -> bytes:
    buffer = StringIO()
    trust_summary = trust_summary or {}
    accuracy_summary = accuracy_summary or {}
    uncertainty = accuracy_summary.get('uncertainty_narrative', {}) if isinstance(accuracy_summary, dict) else {}
    reporting_policy = accuracy_summary.get('reporting_policy', {}) if isinstance(accuracy_summary, dict) else {}
    benchmark_profile = accuracy_summary.get('benchmark_profile', {}) if isinstance(accuracy_summary, dict) else {}
    normalized_label = str(report_label).strip() or 'Generated Report'
    quality_score = float(quality.get('quality_score', 0.0) or 0.0)
    readiness_score = float(readiness.get('readiness_score', 0.0) or 0.0)
    readiness_table = _safe_df(readiness.get('readiness_table'))
    top_missing = _safe_df(quality.get('high_missing')).head(3)
    key_insights = list(insights.get('summary_lines', []))
    recommendations = _safe_df(action_recommendations).head(4)
    default_cohort = healthcare.get('default_cohort_summary', {}) if isinstance(healthcare, dict) else {}
    readmission = healthcare.get('readmission', {}) if isinstance(healthcare, dict) else {}
    utilization = healthcare.get('utilization', {}) if isinstance(healthcare, dict) else {}
    survival = healthcare.get('survival_outcomes', {}) if isinstance(healthcare, dict) else {}

    buffer.write(f'{normalized_label}\n')
    buffer.write('=' * len(normalized_label) + '\n\n')
    buffer.write('Dataset Summary\n')
    buffer.write('-' * 15 + '\n')
    buffer.write(f'Dataset: {dataset_name}\n')
    buffer.write(f"Rows: {int(overview.get('rows', 0)):,}\n")
    buffer.write(f"Analyzed columns: {_analyzed_columns(overview):,}\n")
    if _source_columns(overview) != _analyzed_columns(overview):
        buffer.write(f"Source columns: {_source_columns(overview):,}\n")
    buffer.write(f"Duplicate rows: {int(overview.get('duplicate_rows', 0)):,}\n")
    buffer.write(f"Missing values: {int(overview.get('missing_values', 0)):,}\n\n")
    if trust_summary:
        buffer.write('Accuracy and Reporting Guardrails\n')
        buffer.write('-' * 32 + '\n')
        buffer.write(f"Trust level: {trust_summary.get('trust_level', 'Unknown')}\n")
        buffer.write(f"Interpretation mode: {trust_summary.get('interpretation_mode', 'Unknown')}\n")
        buffer.write(f"Benchmark profile: {benchmark_profile.get('profile_name', 'Generic Healthcare')}\n")
        buffer.write(f"Reporting policy: {reporting_policy.get('profile_name', 'Standard')}\n")
        buffer.write(f"{uncertainty.get('headline', trust_summary.get('summary_text', 'Use governance review before broader external sharing.'))}\n\n")

    buffer.write('Quality Metrics\n')
    buffer.write('-' * 15 + '\n')
    buffer.write(f'Quality score: {quality_score:.1f}\n')
    if not top_missing.empty:
        for _, row in top_missing.iterrows():
            pct = float(row.get('null_percentage', 0.0) or 0.0)
            buffer.write(f"- Missingness watch: {row.get('column_name', 'Unknown field')} at {pct:.1%}\n")
    else:
        buffer.write('- No major missingness spikes were flagged in the current view.\n')
    buffer.write('\n')

    buffer.write('Readiness Score\n')
    buffer.write('-' * 15 + '\n')
    buffer.write(f'Analysis readiness: {readiness_score:.0%}\n')
    if not readiness_table.empty:
        ready_count = int((readiness_table['status'].astype(str) == 'Available').sum()) if 'status' in readiness_table.columns else 0
        buffer.write(f'Ready modules: {ready_count}\n')
        blocked = readiness_table[readiness_table['status'].astype(str) != 'Available'] if 'status' in readiness_table.columns else pd.DataFrame()
        if not blocked.empty and 'analysis_module' in blocked.columns:
            buffer.write(f"Needs attention: {', '.join(blocked['analysis_module'].astype(str).head(4).tolist())}\n")
    buffer.write('\n')

    buffer.write('Key Insights\n')
    buffer.write('-' * 12 + '\n')
    if key_insights:
        for line in key_insights[:5]:
            buffer.write(f'- {line}\n')
    else:
        buffer.write('- Key insight generation is waiting on stronger analytic signal from the current dataset.\n')
    buffer.write('\n')

    buffer.write('Cohort Findings\n')
    buffer.write('-' * 15 + '\n')
    if isinstance(default_cohort, dict) and default_cohort.get('available'):
        summary = default_cohort.get('summary', {})
        buffer.write(f"Cohort size: {int(summary.get('cohort_size', 0)):,}\n")
        if summary.get('survival_rate') is not None:
            buffer.write(f"Survival rate: {float(summary.get('survival_rate', 0.0)):.1%}\n")
        if summary.get('average_treatment_duration_days') is not None:
            buffer.write(f"Average treatment duration: {float(summary.get('average_treatment_duration_days', 0.0)):.1f} days\n")
    elif isinstance(readmission, dict) and readmission.get('available'):
        segments = _safe_df(readmission.get('high_risk_segments')).head(3)
        if not segments.empty:
            for _, row in segments.iterrows():
                buffer.write(
                    f"- {row.get('segment_type', 'Segment')} = {row.get('segment_value', 'Unknown')} | "
                    f"readmission rate {float(row.get('readmission_rate', 0.0) or 0.0):.1%}\n"
                )
        else:
            buffer.write('- Cohort review is available, but no standout segment summary was generated for the current slice.\n')
    else:
        buffer.write('- Cohort findings will appear here when the dataset supports cohort selection or readmission segmentation.\n')
    buffer.write('\n')

    buffer.write('Trend Summaries\n')
    buffer.write('-' * 15 + '\n')
    monthly_util = _safe_df(utilization.get('monthly_utilization')).tail(3)
    readmit_trend = _safe_df(readmission.get('trend')).tail(3)
    outcome_trend = _safe_df(survival.get('outcome_trend')).tail(3)
    if not monthly_util.empty:
        latest = monthly_util.iloc[-1]
        buffer.write(f"Latest utilization month: {latest.get('month')} with {int(latest.get('event_count', 0)):,} events\n")
    elif not readmit_trend.empty:
        latest = readmit_trend.iloc[-1]
        buffer.write(f"Latest readmission month: {latest.get('month')} at {float(latest.get('readmission_rate', 0.0) or 0.0):.1%}\n")
    elif not outcome_trend.empty:
        buffer.write(f"Latest outcome trend point: {outcome_trend.iloc[-1].to_dict()}\n")
    else:
        buffer.write('- Trend summaries will appear here when event-style dates support longitudinal review.\n')
    buffer.write('\n')

    buffer.write('Recommended Actions\n')
    buffer.write('-' * 19 + '\n')
    if not recommendations.empty:
        for _, row in recommendations.iterrows():
            buffer.write(f"- {row.get('recommendation_title', 'Next step')}\n")
    else:
        buffer.write('- Continue with data quality review and healthcare analytics to surface stronger actions.\n')

    return buffer.getvalue().encode('utf-8')
