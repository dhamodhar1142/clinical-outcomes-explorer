from __future__ import annotations

from io import StringIO

from src.reports.common import _analyzed_columns, _source_columns


def build_executive_summary_text(
    dataset_name: str,
    overview: dict[str, object],
    healthcare: dict[str, object],
    insights: dict[str, object],
    trust_summary: dict[str, object] | None = None,
    accuracy_summary: dict[str, object] | None = None,
) -> bytes:
    trust_summary = trust_summary or {}
    accuracy_summary = accuracy_summary or {}
    uncertainty = accuracy_summary.get('uncertainty_narrative', {}) if isinstance(accuracy_summary, dict) else {}
    reporting_policy = accuracy_summary.get('reporting_policy', {}) if isinstance(accuracy_summary, dict) else {}
    benchmark_profile = accuracy_summary.get('benchmark_profile', {}) if isinstance(accuracy_summary, dict) else {}
    buffer = StringIO()
    buffer.write('Executive Summary\n')
    buffer.write('=' * 18 + '\n\n')
    buffer.write(f'Dataset: {dataset_name}\n')
    buffer.write(f"Rows reviewed: {overview['rows']:,}\n")
    buffer.write(f"Analyzed columns reviewed: {_analyzed_columns(overview):,}\n")
    if _source_columns(overview) != _analyzed_columns(overview):
        buffer.write(f"Source columns loaded: {_source_columns(overview):,}\n")
    buffer.write(f"Duplicate rows: {overview['duplicate_rows']:,}\n")
    buffer.write(f"Missing values: {overview['missing_values']:,}\n\n")
    if trust_summary:
        buffer.write('Result Accuracy and Trust\n')
        buffer.write('-' * 25 + '\n')
        buffer.write(f"Trust level: {trust_summary.get('trust_level', 'Unknown')}\n")
        buffer.write(f"Interpretation mode: {trust_summary.get('interpretation_mode', 'Unknown')}\n")
        buffer.write(f"Reporting threshold: {reporting_policy.get('profile_name', 'Standard')}\n")
        buffer.write(f"Benchmark profile: {benchmark_profile.get('profile_name', 'Generic Healthcare')}\n")
        buffer.write(f"{uncertainty.get('headline', trust_summary.get('summary_text', 'Use directional caution when sharing this workflow externally.'))}\n\n")

    risk = healthcare.get('risk_segmentation', {})
    ai_insights = healthcare.get('ai_insight_summary', [])
    scenario = healthcare.get('scenario', {})
    cohort = healthcare.get('default_cohort_summary', {})

    buffer.write('Survival and Outcome Summary\n')
    buffer.write('-' * 27 + '\n')
    if risk.get('available') and risk.get('survival_rate') is not None:
        buffer.write(f"Overall survival rate: {risk['survival_rate']:.1%}\n")
    else:
        buffer.write('Overall survival rate is not available for this dataset.\n')
    buffer.write('\n')

    buffer.write('Key Risk Groups\n')
    buffer.write('-' * 15 + '\n')
    if risk.get('available') and not risk['segment_table'].empty:
        top_row = risk['segment_table'].sort_values('patient_count', ascending=False).iloc[0]
        buffer.write(f"Largest risk segment: {top_row['risk_segment']} ({int(top_row['patient_count'])} records).\n")
    else:
        buffer.write('Risk segmentation could not be generated from the current fields.\n')
    buffer.write('\n')

    buffer.write('Treatment Insights\n')
    buffer.write('-' * 18 + '\n')
    if any('treatment' in line.lower() for line in ai_insights):
        for line in ai_insights:
            if 'treatment' in line.lower():
                buffer.write(f"- {line}\n")
    else:
        buffer.write('Treatment-specific insight is not available for this dataset.\n')
    buffer.write('\n')

    buffer.write('Recommended Next Steps\n')
    buffer.write('-' * 23 + '\n')
    if insights['recommendations']:
        for line in insights['recommendations'][:4]:
            buffer.write(f"- {line}\n")
    else:
        buffer.write('- Continue with data quality review and segmentation analysis.\n')

    if scenario.get('available'):
        buffer.write('\nScenario Highlight\n')
        buffer.write('-' * 18 + '\n')
        buffer.write(f"Simulated survival rate: {scenario['simulated_survival_rate']:.1%} versus baseline {scenario['baseline_survival_rate']:.1%}.\n")

    if cohort.get('available'):
        buffer.write('\nCohort Snapshot\n')
        buffer.write('-' * 15 + '\n')
        buffer.write(f"Current cohort size: {cohort['summary']['cohort_size']:,}\n")

    return buffer.getvalue().encode('utf-8')
