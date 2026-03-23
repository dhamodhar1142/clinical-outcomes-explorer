from __future__ import annotations

import pandas as pd

from src.result_accuracy import weight_recommendation_frame


def build_key_insights(overview: dict[str, object], field_profile: pd.DataFrame, quality: dict[str, object], readiness: dict[str, object], semantic: dict[str, object], healthcare: dict[str, object], structure) -> dict[str, object]:
    summary_lines: list[str] = []
    recommendations: list[str] = []

    if not quality['high_missing'].empty:
        top_missing = quality['high_missing'].iloc[0]
        summary_lines.append(f"{top_missing['column_name']} has the highest missingness at {top_missing['null_percentage']:.1%}.")
        recommendations.append('Review high-missingness fields before using them in comparisons or trend reporting.')
    if structure.numeric_columns:
        summary_lines.append(f"The dataset includes {len(structure.numeric_columns)} numeric fields, which supports distribution analysis and benchmark-style review.")
    if structure.categorical_columns:
        summary_lines.append(f"The dataset includes {len(structure.categorical_columns)} categorical fields, creating useful segmentation opportunities.")
    if structure.default_date_column:
        summary_lines.append(f"Time-series analysis is possible using {structure.default_date_column} as the default date field.")
    else:
        summary_lines.append('No reliable date field was detected, so time-series analysis is limited.')
        recommendations.append('Add or map a reliable date field to unlock stronger trend analysis.')

    if semantic['canonical_map']:
        summary_lines.append(f"The semantic mapping engine identified {len(semantic['canonical_map'])} canonical business fields.")
    else:
        summary_lines.append('Semantic mapping confidence is limited, so downstream analysis stays mostly generic.')
        recommendations.append('Use clearer business-friendly column names to improve semantic mapping confidence.')

    if readiness['available_count']:
        summary_lines.append(f"{readiness['available_count']} advanced analysis modules are fully available right now.")
    if healthcare['healthcare_readiness_score'] >= 0.45:
        summary_lines.append(f"This dataset appears healthcare-related: {healthcare['likely_dataset_type']}.")
        if healthcare['readmission'].get('available'):
            readmission = healthcare['readmission']
            rate = readmission.get('approximate_readmission_rate')
            if rate is None:
                rate = readmission.get('overview', {}).get('overall_readmission_rate')
            if rate is not None:
                summary_lines.append(f"Approximate 30-day readmission-style rate is {float(rate):.1%} based on the detected encounter timing logic.")
    else:
        summary_lines.append('The dataset does not strongly resemble a healthcare claims or encounter file, so healthcare modules remain selective.')

    if not recommendations:
        recommendations.append('The dataset is in good shape for exploratory analysis and exportable summary reporting.')

    suitability = []
    if structure.numeric_columns:
        suitability.append('numeric profiling')
    if structure.categorical_columns:
        suitability.append('segmentation analysis')
    if structure.default_date_column:
        suitability.append('trend analysis')
    if healthcare['healthcare_readiness_score'] >= 0.45:
        suitability.append('healthcare operations review')

    return {
        'summary_lines': summary_lines,
        'recommendations': recommendations[:6],
        'suitability': suitability,
    }


def build_action_recommendations(quality: dict[str, object], readiness: dict[str, object], semantic: dict[str, object], healthcare: dict[str, object]) -> pd.DataFrame:
    recommendations: list[dict[str, str]] = []

    risk = healthcare.get('risk_segmentation', {})
    if risk.get('available') and not risk['segment_table'].empty:
        high_risk = risk['segment_table'][risk['segment_table']['risk_segment'] == 'High Risk']
        if not high_risk.empty:
            row = high_risk.iloc[0]
            recommendations.append({
                'priority': 'High',
                'recommendation_title': 'Review high-risk patient cohorts',
                'rationale': f"{int(row['patient_count'])} records fall into the High Risk segment. Start with older smokers or Stage III/IV patients to target outreach or care management review.",
            })

    anomaly = healthcare.get('anomaly_detection', {})
    if anomaly.get('available') and not anomaly['summary_table'].empty:
        top_anomaly = anomaly['summary_table'].iloc[0]
        recommendations.append({
            'priority': 'High',
            'recommendation_title': 'Validate unusual clinical measurements',
            'rationale': f"{top_anomaly['field']} contains {int(top_anomaly['anomaly_count'])} unusual values. Confirm these outliers before using them for benchmarking or risk review.",
        })

    if not quality['high_missing'].empty:
        top_missing = quality['high_missing'].iloc[0]
        recommendations.append({
            'priority': 'Medium',
            'recommendation_title': 'Address high-missingness fields before comparison work',
            'rationale': f"{top_missing['column_name']} is missing in {top_missing['null_percentage']:.1%} of records. Cleaning this field will improve downstream segmentation and reporting confidence.",
        })

    missing_cost = all(field not in semantic['canonical_map'] for field in ['cost_amount', 'paid_amount', 'allowed_amount', 'billed_amount'])
    if missing_cost:
        recommendations.append({
            'priority': 'Medium',
            'recommendation_title': 'Map a cost field to unlock financial analysis',
            'rationale': 'A usable cost or payment field has not been mapped yet. Adding one would turn on spend, outlier, and provider cost comparisons.',
        })

    if readiness['partial_count'] or readiness['available_count'] < len(readiness['readiness_table']):
        unavailable = readiness['readiness_table'][readiness['readiness_table']['status'] == 'Unavailable']
        if not unavailable.empty:
            first_gap = unavailable.iloc[0]
            recommendations.append({
                'priority': 'Medium',
                'recommendation_title': 'Close the highest-value analysis gap',
                'rationale': f"{first_gap['analysis_module']} is still unavailable. Adding {first_gap['missing_prerequisites']} would unlock a richer view of the dataset.",
            })

    survival = healthcare.get('survival_outcomes', {})
    if survival.get('available') and not survival.get('stage_table', pd.DataFrame()).empty:
        weakest = survival['stage_table'].iloc[0]
        recommendations.append({
            'priority': 'Medium',
            'recommendation_title': 'Compare outcomes for advanced-stage patients',
            'rationale': f"{weakest[survival['stage_column']]} currently has the weakest observed survival rate. Use treatment comparisons to understand whether outcomes differ for this stage group.",
        })

    if healthcare.get('ai_insight_summary'):
        recommendations.append({
            'priority': 'Low',
            'recommendation_title': 'Turn the strongest insights into a stakeholder briefing',
            'rationale': 'Use the Export Center to create an executive or clinical summary once the main risk, treatment, and quality findings have been reviewed.',
        })

    if not recommendations:
        recommendations.append({
            'priority': 'Low',
            'recommendation_title': 'Continue with guided exploratory review',
            'rationale': 'The dataset is in a good position for profiling and trend review. Start with Field Profiling, then move into Benchmarking or Export Center outputs.',
        })

    priority_order = {'High': 0, 'Medium': 1, 'Low': 2}
    frame = pd.DataFrame(recommendations).drop_duplicates(subset=['recommendation_title'])
    frame['priority_rank'] = frame['priority'].map(priority_order)
    frame['base_priority_score'] = frame['priority_rank'].map({0: 100, 1: 75, 2: 50}).fillna(40)
    frame = weight_recommendation_frame(
        frame,
        title_col='recommendation_title',
        rationale_col='rationale',
        base_priority_col='base_priority_score',
        healthcare=healthcare,
        remediation_context={},
    )
    frame = frame.sort_values(['priority_rank', 'weighted_priority_score', 'recommendation_title'], ascending=[True, False, True]).drop(columns=['priority_rank']).reset_index(drop=True)
    return frame.head(8)


def build_automated_insight_board(
    overview: dict[str, object],
    readiness: dict[str, object],
    healthcare: dict[str, object],
    insights: dict[str, object],
    action_recommendations: pd.DataFrame,
) -> dict[str, object]:
    risk = healthcare.get('risk_segmentation', {})
    readmission = healthcare.get('readmission', {})
    survival = healthcare.get('survival_outcomes', {})
    anomaly = healthcare.get('anomaly_detection', {})
    fairness = healthcare.get('explainability_fairness', {})
    scenario = healthcare.get('scenario', {})
    cost = healthcare.get('cost', {})

    findings: list[str] = []
    risks: list[str] = []
    recommendations: list[str] = []
    benchmark_gaps: list[str] = []
    kpis: list[dict[str, str]] = []
    stage_table = survival.get('stage_table', pd.DataFrame())
    stage_column = survival.get('stage_column')
    if isinstance(stage_table, pd.DataFrame) and not stage_table.empty and not stage_column:
        stage_column = next((column for column in stage_table.columns if column not in {'record_count', 'survival_rate', 'average_treatment_duration_days'}), None)

    kpis.append({'label': 'Rows in Scope', 'value': f"{int(overview.get('rows', 0)):,}", 'description': 'Records included in the current review.'})
    kpis.append({'label': 'Analysis Readiness', 'value': f"{float(readiness.get('readiness_score', 0.0)):.0%}", 'description': 'Advanced modules currently supported by the dataset.'})
    kpis.append({'label': 'Ready Modules', 'value': f"{int(readiness.get('available_count', 0))}", 'description': 'Analytics modules available right now.'})
    kpis.append({'label': 'Healthcare Readiness', 'value': f"{float(healthcare.get('healthcare_readiness_score', 0.0)):.0%}", 'description': 'How strongly the dataset supports healthcare-specific review.'})

    if readmission.get('available'):
        overview_row = readmission.get('overview', {})
        kpis.append({'label': 'Readmission Rate', 'value': f"{float(overview_row.get('overall_readmission_rate', 0.0)):.1%}", 'description': 'Overall readmission-style rate in the current scope.'})
        findings.append(
            f"Overall readmission rate is {float(overview_row.get('overall_readmission_rate', 0.0)):.1%} across {int(overview_row.get('records_in_scope', 0)):,} records."
        )
        segments = readmission.get('high_risk_segments', pd.DataFrame())
        if isinstance(segments, pd.DataFrame) and not segments.empty:
            top_segment = segments.iloc[0]
            risks.append(
                f"{top_segment['segment_type']} = {top_segment['segment_value']} has the highest visible readmission exposure at {float(top_segment['readmission_rate']):.1%}."
            )

    if risk.get('available') and not risk.get('segment_table', pd.DataFrame()).empty:
        segment_table = risk['segment_table']
        high_risk = segment_table[segment_table['risk_segment'].astype(str) == 'High Risk']
        if not high_risk.empty:
            row = high_risk.iloc[0]
            kpis.append({'label': 'High-Risk Share', 'value': f"{float(row.get('percentage', 0.0)):.1%}", 'description': 'Share of records in the highest rule-based risk segment.'})
            risks.append(f"{int(row['patient_count'])} records fall into the High Risk segment, representing {float(row['percentage']):.1%} of the reviewed population.")

    if survival.get('available') and isinstance(stage_table, pd.DataFrame) and not stage_table.empty and stage_column:
        weakest = stage_table.sort_values('survival_rate').iloc[0]
        findings.append(f"{weakest[stage_column]} currently has the weakest observed survival rate at {float(weakest['survival_rate']):.1%}.")

    if anomaly.get('available') and not anomaly.get('summary_table', pd.DataFrame()).empty:
        top_anomaly = anomaly['summary_table'].iloc[0]
        risks.append(f"{top_anomaly['field']} has {int(top_anomaly['anomaly_count'])} flagged anomalies and should be validated before deeper comparisons.")

    if fairness.get('available') and not fairness.get('fairness_flags', pd.DataFrame()).empty:
        top_gap = fairness['fairness_flags'].iloc[0]
        grouping_label = top_gap.get('grouping_dimension', top_gap.get('group_dimension', top_gap.get('subgroup', 'A subgroup')))
        gap_value = top_gap.get('gap_value', top_gap.get('gap_size', 0.0))
        metric_label = top_gap.get('metric', top_gap.get('flag_type', 'performance'))
        benchmark_gaps.append(
            f"{grouping_label} shows a {float(gap_value):.1%} gap for {metric_label} in the current review."
        )

    if cost.get('available'):
        summary = cost.get('summary', {})
        kpis.append({'label': 'Average Cost', 'value': f"${float(summary.get('average_cost', 0.0)):,.0f}", 'description': 'Average cost across usable cost records.'})
        if not cost.get('by_segment', pd.DataFrame()).empty:
            top_cost = cost['by_segment'].iloc[0]
            findings.append(
                f"{top_cost.iloc[0]} is the leading cost segment with ${float(top_cost['total_cost']):,.0f} in total cost."
            )

    for line in insights.get('summary_lines', []):
        if line not in findings:
            findings.append(line)
    for _, row in action_recommendations.head(3).iterrows():
        recommendations.append(f"{row['recommendation_title']}: {row['rationale']}")

    chart_spec = None
    if risk.get('available') and not risk.get('segment_table', pd.DataFrame()).empty:
        chart_spec = {
            'kind': 'bar',
            'data': risk['segment_table'].head(6),
            'x': 'risk_segment',
            'y': 'patient_count',
            'title': 'Risk Segment Mix',
        }
    elif readmission.get('available') and not readmission.get('by_department', pd.DataFrame()).empty:
        table = readmission['by_department'].head(8)
        chart_spec = {
            'kind': 'bar',
            'data': table,
            'x': table.columns[0],
            'y': 'readmission_rate',
            'title': 'Readmission Rate by Department',
        }
    elif survival.get('available') and isinstance(stage_table, pd.DataFrame) and not stage_table.empty and stage_column:
        chart_spec = {
            'kind': 'bar',
            'data': stage_table.head(8),
            'x': stage_column,
            'y': 'survival_rate',
            'title': 'Survival by Stage',
        }
    elif anomaly.get('available') and not anomaly.get('summary_table', pd.DataFrame()).empty:
        chart_spec = {
            'kind': 'bar',
            'data': anomaly['summary_table'].head(8),
            'x': 'field',
            'y': 'anomaly_count',
            'title': 'Top Anomaly Signals',
        }

    intervention_summary = None
    if scenario.get('available'):
        intervention_summary = {
            'title': 'Intervention Preview',
            'headline': f"Simulated survival improves from {float(scenario.get('baseline_survival_rate', 0.0)):.1%} to {float(scenario.get('simulated_survival_rate', 0.0)):.1%}.",
            'detail': f"That reflects a modeled improvement of {float(scenario.get('improvement', 0.0)):.1%} under the current scenario assumptions.",
        }
    elif readmission.get('available'):
        intervention_summary = {
            'title': 'Readmission Intervention Preview',
            'headline': 'Readmission intervention planning is available for the current dataset.',
            'detail': 'Use the Readmission Cohort Builder to test discharge follow-up, case management, LOS reduction, and early follow-up scenarios.',
        }

    if not findings and not risks and not recommendations and not benchmark_gaps and chart_spec is None:
        return {
            'available': False,
            'reason': 'The current dataset does not yet support a strong executive insight board. Start with Analysis Readiness, profiling, and remediation guidance to strengthen the available signals.',
        }

    return {
        'available': True,
        'kpi_cards': kpis[:6],
        'top_findings': findings[:3],
        'top_risks': risks[:3],
        'top_recommendations': recommendations[:3],
        'benchmark_gaps': benchmark_gaps[:3],
        'key_anomaly_summary': anomaly.get('summary_table', pd.DataFrame()).head(3) if anomaly.get('available') else pd.DataFrame(),
        'key_chart': chart_spec,
        'intervention_summary': intervention_summary,
    }
