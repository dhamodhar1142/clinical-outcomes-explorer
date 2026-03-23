from __future__ import annotations

from typing import Any

import pandas as pd

from src.result_accuracy import weight_recommendation_frame


def _safe_df(table: Any) -> pd.DataFrame:
    return table if isinstance(table, pd.DataFrame) else pd.DataFrame()


def _priority_rank(value: str) -> int:
    return {'Immediate': 0, 'Near-Term': 1, 'Strategic': 2}.get(str(value), 3)


def build_intervention_recommendations(
    healthcare: dict[str, object],
    quality: dict[str, object],
    readiness: dict[str, object],
    remediation_context: dict[str, object],
    model_comparison: dict[str, object] | None = None,
) -> pd.DataFrame:
    recommendations: list[dict[str, object]] = []

    risk = healthcare.get('risk_segmentation', {})
    if risk.get('available'):
        segment_table = _safe_df(risk.get('segment_table'))
        high_risk = segment_table[segment_table.get('risk_segment', pd.Series(dtype=str)).astype(str) == 'High Risk']
        if not high_risk.empty:
            row = high_risk.iloc[0]
            recommendations.append(
                {
                    'recommendation_title': 'Prioritize outreach for the highest-risk cohort',
                    'recommendation_type': 'care management prioritization',
                    'target_population': 'High Risk segment',
                    'why_it_matters': f"{int(row.get('patient_count', 0)):,} records fall into the highest current risk tier.",
                    'expected_operational_value': 'Improves care-management focus and follow-up planning.',
                    'implementation_difficulty': 'Medium',
                    'supporting_signals': 'Rule-based risk segmentation',
                    'confidence_level': 'High',
                    'priority_bucket': 'Immediate',
                }
            )

    readmission = healthcare.get('readmission', {})
    if readmission.get('available'):
        segments = _safe_df(readmission.get('high_risk_segments'))
        if not segments.empty:
            top_segment = segments.iloc[0]
            recommendations.append(
                {
                    'recommendation_title': 'Focus readmission prevention on the highest-exposure group',
                    'recommendation_type': 'workflow monitoring action',
                    'target_population': f"{top_segment.get('segment_type', 'segment')} = {top_segment.get('segment_value', 'Unknown')}",
                    'why_it_matters': f"Readmission exposure is {float(top_segment.get('readmission_rate', 0.0)):.1%} for the highest-risk visible segment.",
                    'expected_operational_value': 'Supports discharge planning and follow-up prioritization.',
                    'implementation_difficulty': 'Medium',
                    'supporting_signals': 'Readmission analytics',
                    'confidence_level': 'Medium' if readmission.get('source') == 'synthetic' else 'High',
                    'priority_bucket': 'Immediate',
                }
            )

    anomaly = healthcare.get('anomaly_detection', {})
    anomaly_table = _safe_df(anomaly.get('summary_table'))
    if anomaly.get('available') and not anomaly_table.empty:
        top_anomaly = anomaly_table.iloc[0]
        recommendations.append(
            {
                'recommendation_title': 'Review the highest-impact anomaly field',
                'recommendation_type': 'data remediation action',
                'target_population': str(top_anomaly.get('field', 'flagged records')),
                'why_it_matters': f"{int(top_anomaly.get('anomaly_count', 0)):,} unusual values were detected and may weaken downstream analytics.",
                'expected_operational_value': 'Improves model stability, benchmarking validity, and cohort integrity.',
                'implementation_difficulty': 'Low',
                'supporting_signals': 'Anomaly detection',
                'confidence_level': 'High',
                'priority_bucket': 'Immediate',
            }
        )

    synthetic_cost = remediation_context.get('synthetic_cost', {})
    if synthetic_cost.get('available'):
        recommendations.append(
            {
                'recommendation_title': 'Replace synthetic cost support with a native cost field',
                'recommendation_type': 'data remediation action',
                'target_population': 'Financial analytics layer',
                'why_it_matters': 'Cost analytics are currently supported by a deterministic synthetic estimate rather than a native financial field.',
                'expected_operational_value': 'Unlocks stronger benchmarking, outlier detection, and spend comparisons.',
                'implementation_difficulty': 'Medium',
                'supporting_signals': 'Synthetic cost augmentation',
                'confidence_level': 'High',
                'priority_bucket': 'Near-Term',
            }
        )

    bmi_summary = remediation_context.get('bmi_remediation', {})
    if bmi_summary.get('available') and bmi_summary.get('total_bmi_outliers', 0):
        recommendations.append(
            {
                'recommendation_title': 'Investigate BMI data quality before clinical comparison work',
                'recommendation_type': 'high-BMI follow-up review',
                'target_population': 'Rows flagged in BMI remediation',
                'why_it_matters': f"{int(bmi_summary.get('total_bmi_outliers', 0)):,} BMI values required remediation or review.",
                'expected_operational_value': 'Improves segment credibility and model input quality.',
                'implementation_difficulty': 'Low',
                'supporting_signals': 'BMI remediation summary',
                'confidence_level': 'High',
                'priority_bucket': 'Near-Term',
            }
        )

    if not _safe_df(quality.get('high_missing')).empty:
        top_missing = _safe_df(quality.get('high_missing')).iloc[0]
        recommendations.append(
            {
                'recommendation_title': 'Improve completeness for the highest-missing field',
                'recommendation_type': 'data remediation action',
                'target_population': str(top_missing.get('column_name', 'high-missing field')),
                'why_it_matters': f"This field is missing in {float(top_missing.get('null_percentage', 0.0)):.1%} of records.",
                'expected_operational_value': 'Improves readiness for more advanced modules and reporting.',
                'implementation_difficulty': 'Medium',
                'supporting_signals': 'Data quality review',
                'confidence_level': 'High',
                'priority_bucket': 'Near-Term',
            }
        )

    if model_comparison and model_comparison.get('available'):
        best = model_comparison.get('best_model_summary', {})
        if best:
            recommendations.append(
                {
                    'recommendation_title': 'Operationalize the best-performing demo model for review workflows',
                    'recommendation_type': 'workflow monitoring action',
                    'target_population': str(best.get('target_variable', 'current predictive target')),
                    'why_it_matters': f"{best.get('model_name', 'The leading model')} currently leads model comparison on the selected target.",
                    'expected_operational_value': 'Supports repeatable scorecards and guided model review.',
                    'implementation_difficulty': 'Medium',
                    'supporting_signals': 'Model comparison studio',
                    'confidence_level': 'Medium',
                    'priority_bucket': 'Strategic',
                }
            )

    if readiness.get('synthetic_supported_modules'):
        recommendations.append(
            {
                'recommendation_title': 'Replace synthetic helper fields with native source data',
                'recommendation_type': 'data remediation action',
                'target_population': 'Modules currently running in synthetic support mode',
                'why_it_matters': 'Several modules are running with helper fields rather than source-grade data.',
                'expected_operational_value': 'Improves trust, reduces caveats, and increases stakeholder confidence.',
                'implementation_difficulty': 'Strategic',
                'supporting_signals': 'Readiness support-type analysis',
                'confidence_level': 'Medium',
                'priority_bucket': 'Strategic',
            }
        )

    frame = pd.DataFrame(recommendations).drop_duplicates(subset=['recommendation_title'])
    if frame.empty:
        return pd.DataFrame(
            columns=[
                'recommendation_title',
                'recommendation_type',
                'target_population',
                'why_it_matters',
                'expected_operational_value',
                'implementation_difficulty',
                'supporting_signals',
                'confidence_level',
                'priority_bucket',
            ]
        )
    frame['priority_rank'] = frame['priority_bucket'].map(_priority_rank)
    frame['base_priority_score'] = frame['priority_rank'].map({0: 100, 1: 80, 2: 60}).fillna(40)
    frame = weight_recommendation_frame(
        frame,
        title_col='recommendation_title',
        rationale_col='why_it_matters',
        base_priority_col='base_priority_score',
        healthcare=healthcare,
        remediation_context=remediation_context,
    )
    frame = frame.sort_values(
        ['priority_rank', 'weighted_priority_score', 'recommendation_title'],
        ascending=[True, False, True],
    ).drop(columns=['priority_rank']).reset_index(drop=True)
    return frame.head(10)


def build_executive_summary(
    dataset_name: str,
    overview: dict[str, object],
    readiness: dict[str, object],
    healthcare: dict[str, object],
    action_recommendations: pd.DataFrame,
    intervention_recommendations: pd.DataFrame,
    remediation_context: dict[str, object],
    trust_summary: dict[str, object] | None = None,
    accuracy_summary: dict[str, object] | None = None,
    verbosity: str = 'concise',
) -> dict[str, object]:
    trust_summary = trust_summary or {}
    accuracy_summary = accuracy_summary or {}
    active_modules = int(readiness.get('available_count', 0))
    analyzed_columns = int(overview.get('analyzed_columns', overview.get('columns', 0)))
    source_columns = int(overview.get('source_columns', analyzed_columns))
    helper_columns_added = int(overview.get('helper_columns_added', max(analyzed_columns - source_columns, 0)))
    blocked = _safe_df(readiness.get('readiness_table'))
    blocked_count = int((blocked.get('status', pd.Series(dtype=str)) == 'Unavailable').sum()) if not blocked.empty else 0
    uncertainty_narrative = accuracy_summary.get('uncertainty_narrative', {}) if isinstance(accuracy_summary, dict) else {}
    reporting_policy = accuracy_summary.get('reporting_policy', {}) if isinstance(accuracy_summary, dict) else {}
    benchmark_profile = accuracy_summary.get('benchmark_profile', {}) if isinstance(accuracy_summary, dict) else {}
    sections = {
        'Dataset Overview': f"{dataset_name} contains {int(overview.get('rows', 0)):,} rows and {analyzed_columns:,} analyzed columns for the current review.",
        'Current Readiness': f"{float(readiness.get('readiness_score', 0.0)):.0%} readiness with {active_modules} fully available modules and {blocked_count} blocked modules.",
        'Top Findings': '; '.join(_safe_df(healthcare.get('readmission', {}).get('high_risk_segments')).head(1).astype(str).agg(' | '.join, axis=1).tolist()) or 'Core healthcare analytics are available for operational review.',
        'Top Data Issues': f"{int(overview.get('duplicate_rows', 0)):,} duplicate rows and {int(overview.get('missing_values', 0)):,} missing values are currently in scope.",
        'Recommended Next Actions': '; '.join(intervention_recommendations.head(3)['recommendation_title'].tolist()) if not intervention_recommendations.empty else '; '.join(action_recommendations.head(3)['recommendation_title'].tolist()),
        'Result Trust': str(
            trust_summary.get(
                'summary_text',
                'Result trust disclosures are not yet available for the current workflow.',
            )
        ),
        'Accuracy and Reporting Guardrails': (
            f"{str(uncertainty_narrative.get('headline', 'Accuracy guardrails are not available yet.'))} "
            f"Benchmark profile: {benchmark_profile.get('profile_name', 'Generic Healthcare')}. "
            f"Reporting policy: {reporting_policy.get('profile_name', 'Standard')}."
        ),
        'Synthetic Support Disclosure': (
            f"Synthetic or derived helper support added {helper_columns_added} analyzed columns on top of {source_columns:,} source columns."
            if helper_columns_added > 0
            else 'No synthetic helper fields are currently required.'
        ),
    }
    bullets = [
        sections['Dataset Overview'],
        sections['Current Readiness'],
        sections['Recommended Next Actions'],
    ]
    if str(verbosity).lower() == 'detailed':
        bullets.extend([
            sections['Result Trust'],
            sections['Accuracy and Reporting Guardrails'],
            sections['Synthetic Support Disclosure'],
        ])
    recruiter_summary = (
        f"{dataset_name} is currently running a healthcare-aware analytics workflow with {active_modules} active modules, "
        f"a readiness score of {float(readiness.get('readiness_score', 0.0)):.0%}, and explicit governance around synthetic support."
    )
    text = '\n'.join(f"{title}: {body}" for title, body in sections.items())
    return {
        'executive_summary_text': text,
        'executive_summary_sections': sections,
        'stakeholder_summary_bullets': bullets,
        'recruiter_demo_summary': recruiter_summary,
    }


def build_kpi_benchmarking_layer(
    healthcare: dict[str, object],
    quality: dict[str, object],
    remediation_context: dict[str, object],
) -> dict[str, object]:
    cards: list[dict[str, object]] = []
    rows: list[dict[str, object]] = []
    positives: list[str] = []
    risks: list[str] = []

    risk = healthcare.get('risk_segmentation', {})
    segment_table = _safe_df(risk.get('segment_table'))
    if risk.get('available') and not segment_table.empty:
        high = segment_table[segment_table.get('risk_segment', pd.Series(dtype=str)).astype(str) == 'High Risk']
        low = segment_table[segment_table.get('risk_segment', pd.Series(dtype=str)).astype(str) == 'Low Risk']
        if not high.empty:
            high_share = float(high.iloc[0].get('percentage', 0.0))
            cards.append({'label': 'High-Risk Share', 'value': f'{high_share:.1%}', 'description': 'Internal risk benchmark based on the current dataset.'})
            rows.append({'metric': 'High-Risk Share', 'current_value': high_share, 'benchmark_basis': 'Internal segment mix', 'interpretation': 'Higher values increase operational risk exposure.'})
            risks.append(f'High-risk share currently stands at {high_share:.1%}.')
        if not low.empty:
            positives.append(f"Low-risk share covers {float(low.iloc[0].get('percentage', 0.0)):.1%} of the current population.")

    cost = healthcare.get('cost', {})
    cost_summary = cost.get('summary', {}) if isinstance(cost, dict) else {}
    if cost.get('available'):
        avg_cost = float(cost_summary.get('average_cost', 0.0))
        cards.append({'label': 'Average Cost', 'value': f'${avg_cost:,.0f}', 'description': 'Internal dataset-relative cost benchmark.'})
        rows.append({'metric': 'Average Cost', 'current_value': avg_cost, 'benchmark_basis': 'Current dataset average', 'interpretation': 'Use for internal cohort and segment comparison only.'})

    bmi = remediation_context.get('bmi_remediation', {})
    if bmi.get('available'):
        outlier_pct = float(bmi.get('outlier_pct', 0.0))
        cards.append({'label': 'BMI Anomaly Rate', 'value': f'{outlier_pct:.1%}', 'description': 'Share of rows requiring BMI review or remediation.'})
        rows.append({'metric': 'BMI Anomaly Rate', 'current_value': outlier_pct, 'benchmark_basis': 'Internal quality benchmark', 'interpretation': 'Higher rates reduce trust in BMI-based segmentation.'})
        if outlier_pct < 0.02:
            positives.append('BMI anomaly burden is low relative to the full dataset.')
        elif outlier_pct >= 0.05:
            risks.append('BMI anomaly burden remains high enough to affect downstream modeling and segmentation.')

    readmission = healthcare.get('readmission', {})
    if readmission.get('available'):
        rate = float(readmission.get('overview', {}).get('overall_readmission_rate', 0.0))
        cards.append({'label': 'Readmission Proxy Rate', 'value': f'{rate:.1%}', 'description': 'Internal readmission-style benchmark for the current population.'})
        rows.append({'metric': 'Readmission Proxy Rate', 'current_value': rate, 'benchmark_basis': 'Current dataset average', 'interpretation': 'Use as an internal benchmark only, especially in synthetic-support mode.'})
        if readmission.get('source') == 'synthetic':
            risks.append('Readmission analytics are currently supported by a synthetic proxy rather than native encounter outcomes.')

    if not cards:
        return {
            'available': False,
            'reason': 'Internal KPI benchmarking needs at least one stable operational or clinical signal.',
            'kpi_cards': [],
            'benchmark_table': pd.DataFrame(),
            'standout_positive_signals': [],
            'standout_risk_signals': [],
        }
    return {
        'available': True,
        'kpi_cards': cards[:6],
        'benchmark_table': pd.DataFrame(rows),
        'standout_positive_signals': positives[:4],
        'standout_risk_signals': risks[:4],
    }


def build_scenario_simulation_studio(
    healthcare: dict[str, object],
    quality: dict[str, object],
    remediation_context: dict[str, object],
    scenario_mode: str = 'basic',
) -> dict[str, object]:
    scenarios: list[dict[str, object]] = []
    risk = healthcare.get('risk_segmentation', {})
    risk_table = _safe_df(risk.get('segment_table'))
    if risk.get('available') and not risk_table.empty:
        high = risk_table[risk_table.get('risk_segment', pd.Series(dtype=str)).astype(str) == 'High Risk']
        if not high.empty:
            baseline = float(high.iloc[0].get('percentage', 0.0))
            projected = max(baseline - 0.05, 0.0)
            scenarios.append({
                'scenario_name': 'Reduce current smoking prevalence',
                'baseline_metric': baseline,
                'projected_metric': projected,
                'delta': projected - baseline,
                'assumptions': 'Assumes targeted smoking-risk outreach reduces the high-risk share by five percentage points.',
                'limitation_note': 'Directional scenario only. This is not a causal forecast.',
            })

    bmi = remediation_context.get('bmi_remediation', {})
    if bmi.get('available'):
        baseline = float(bmi.get('outlier_pct', 0.0))
        projected = max(baseline * 0.5, 0.0)
        scenarios.append({
            'scenario_name': 'Improve BMI distribution quality',
            'baseline_metric': baseline,
            'projected_metric': projected,
            'delta': projected - baseline,
            'assumptions': 'Assumes half of current BMI anomalies are corrected or excluded from downstream analysis.',
            'limitation_note': 'Improves analytic credibility rather than directly changing outcomes.',
        })

    readmission = healthcare.get('readmission', {})
    if readmission.get('available'):
        baseline = float(readmission.get('overview', {}).get('overall_readmission_rate', 0.0))
        projected = max(baseline - 0.03, 0.0)
        scenarios.append({
            'scenario_name': 'Reduce readmission proxy threshold exceedance',
            'baseline_metric': baseline,
            'projected_metric': projected,
            'delta': projected - baseline,
            'assumptions': 'Assumes better follow-up and case management reduce the proxy rate by three percentage points.',
            'limitation_note': 'Readmission proxy may be synthetic or heuristic depending on source data.',
        })

    if not _safe_df(quality.get('high_missing')).empty:
        scenarios.append({
            'scenario_name': 'Improve blocked-module data completeness',
            'baseline_metric': float(len(_safe_df(quality.get('high_missing')))),
            'projected_metric': max(float(len(_safe_df(quality.get('high_missing')))) - 2.0, 0.0),
            'delta': -min(2.0, float(len(_safe_df(quality.get('high_missing'))))),
            'assumptions': 'Assumes two of the highest-missing fields are materially improved.',
            'limitation_note': 'This scenario changes readiness posture rather than direct clinical outcomes.',
        })

    if not scenarios:
        return {'available': False, 'reason': 'Scenario simulation needs at least one stable risk, quality, or readmission signal.', 'scenario_table': pd.DataFrame()}
    scenario_table = pd.DataFrame(scenarios)
    if str(scenario_mode).lower() != 'expanded':
        scenario_table = scenario_table.head(3).reset_index(drop=True)
    return {
        'available': True,
        'scenario_table': scenario_table,
        'summary': f"{len(scenario_table)} directional scenarios are available for stakeholder discussion.",
    }


def build_prioritized_insights(
    overview: dict[str, object],
    quality: dict[str, object],
    readiness: dict[str, object],
    healthcare: dict[str, object],
    action_recommendations: pd.DataFrame,
    intervention_recommendations: pd.DataFrame,
    model_comparison: dict[str, object] | None = None,
) -> dict[str, object]:
    rows: list[dict[str, object]] = []

    if not _safe_df(quality.get('high_missing')).empty:
        top_missing = _safe_df(quality.get('high_missing')).iloc[0]
        rows.append({
            'category': 'Data Quality',
            'finding': f"{top_missing.get('column_name', 'A field')} has the highest missingness in scope.",
            'priority_score': 90,
            'bucket': 'critical',
            'why_it_matters': 'High missingness limits reliable comparisons and can block downstream modules.',
        })

    anomaly = healthcare.get('anomaly_detection', {})
    anomaly_table = _safe_df(anomaly.get('summary_table'))
    if anomaly.get('available') and not anomaly_table.empty:
        row = anomaly_table.iloc[0]
        rows.append({
            'category': 'Clinical Quality',
            'finding': f"{row.get('field', 'A clinical field')} has the strongest anomaly burden.",
            'priority_score': 88,
            'bucket': 'critical',
            'why_it_matters': 'Anomalies weaken segmentation, benchmarking, and model credibility.',
        })

    readmission = healthcare.get('readmission', {})
    if readmission.get('available'):
        rows.append({
            'category': 'Readmission',
            'finding': f"Readmission rate is {float(readmission.get('overview', {}).get('overall_readmission_rate', 0.0)):.1%}.",
            'priority_score': 85 if readmission.get('source') != 'synthetic' else 72,
            'bucket': 'critical',
            'why_it_matters': 'Readmission-style risk is a high-visibility operational signal.',
        })

    risk = healthcare.get('risk_segmentation', {})
    risk_table = _safe_df(risk.get('segment_table'))
    if risk.get('available') and not risk_table.empty:
        high = risk_table[risk_table.get('risk_segment', pd.Series(dtype=str)).astype(str) == 'High Risk']
        if not high.empty:
            row = high.iloc[0]
            rows.append({
                'category': 'Risk',
                'finding': f"High-risk population share is {float(row.get('percentage', 0.0)):.1%}.",
                'priority_score': 80,
                'bucket': 'critical',
                'why_it_matters': 'This defines the most immediate operational review population.',
            })

    synthetic_supported = readiness.get('synthetic_supported_modules')
    synthetic_supported_count = len(synthetic_supported) if isinstance(synthetic_supported, (list, tuple, set)) else int(synthetic_supported or 0)
    if synthetic_supported_count:
        rows.append({
            'category': 'Readiness',
            'finding': f"{synthetic_supported_count} modules are running with synthetic support.",
            'priority_score': 68,
            'bucket': 'watchlist',
            'why_it_matters': 'Synthetic helper fields improve demos, but native fields would improve trust and auditability.',
        })

    if model_comparison and model_comparison.get('available'):
        best = model_comparison.get('best_model_summary', {})
        rows.append({
            'category': 'Modeling',
            'finding': f"{best.get('model_name', 'The leading model')} is currently the top-performing model for {best.get('target_variable', 'the selected target')}.",
            'priority_score': 64,
            'bucket': 'watchlist',
            'why_it_matters': 'Model comparison improves confidence in the current analytic framing.',
        })

    if not intervention_recommendations.empty:
        top = intervention_recommendations.iloc[0]
        rows.append({
            'category': 'Action',
            'finding': str(top.get('recommendation_title', 'A near-term intervention opportunity is available.')),
            'priority_score': 76,
            'bucket': 'remediation',
            'why_it_matters': str(top.get('why_it_matters', '')),
        })

    frame = pd.DataFrame(rows)
    if frame.empty:
        return {
            'available': False,
            'reason': 'Not enough validated signals are available to prioritize insights confidently.',
            'prioritized_insights': pd.DataFrame(),
            'critical_findings': pd.DataFrame(),
            'watchlist_findings': pd.DataFrame(),
            'remediation_opportunities': pd.DataFrame(),
        }

    frame = weight_recommendation_frame(
        frame,
        title_col='finding',
        rationale_col='why_it_matters',
        base_priority_col='priority_score',
        healthcare=healthcare,
        remediation_context={},
    )
    frame = frame.sort_values(['weighted_priority_score', 'category'], ascending=[False, True]).reset_index(drop=True)
    return {
        'available': True,
        'prioritized_insights': frame,
        'critical_findings': frame[frame['bucket'] == 'critical'].reset_index(drop=True),
        'watchlist_findings': frame[frame['bucket'] == 'watchlist'].reset_index(drop=True),
        'remediation_opportunities': frame[frame['bucket'] == 'remediation'].reset_index(drop=True),
    }
