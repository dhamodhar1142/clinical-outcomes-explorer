from __future__ import annotations

from io import StringIO

import pandas as pd


def build_readmission_summary_text(dataset_name: str, readmission: dict[str, object], action_recommendations: pd.DataFrame) -> bytes:
    buffer = StringIO()
    buffer.write('Readmission Risk Summary\n')
    buffer.write('=' * 24 + '\n\n')
    buffer.write(f'Dataset: {dataset_name}\n')
    readiness = readmission.get('readiness', {}) if isinstance(readmission, dict) else {}
    buffer.write(f"Workflow: {readiness.get('workflow_title', 'Hospital Readmission Risk Analytics')}\n")
    if readiness.get('support_level'):
        buffer.write(f"Support level: {readiness.get('support_level')}\n")
    if readiness.get('badge_text'):
        buffer.write(f"Workflow status: {readiness.get('badge_text')}\n")
    buffer.write('\n')

    if not readmission.get('available'):
        buffer.write('Readmission-focused analytics are not fully available for this dataset.\n')
        if readiness.get('blocker_summary'):
            buffer.write(str(readiness.get('blocker_summary')) + '\n')
        if readmission.get('solution_story'):
            buffer.write(str(readmission.get('solution_story')) + '\n')
        missing = readiness.get('missing_fields', readmission.get('missing_fields', []))
        if missing:
            buffer.write('Missing or weak fields:\n')
            for item in missing:
                buffer.write(f"- {str(item).replace('_', ' ')}\n")
        available = readiness.get('available_analysis', readmission.get('available_analysis', []))
        if available:
            buffer.write('\nWhat can still be reviewed now:\n')
            for item in available:
                buffer.write(f"- {item}\n")
        extra = readiness.get('additional_fields_to_unlock_full_analysis', [])
        if extra:
            buffer.write('\nAdd these next to unlock the full workflow:\n')
            for item in extra:
                buffer.write(f"- {str(item).replace('_', ' ')}\n")
        for note in readiness.get('guidance_notes', []):
            buffer.write(f"- {note}\n")
        if readmission.get('next_best_action'):
            buffer.write('\nRecommended next action:\n')
            buffer.write(f"- {readmission.get('next_best_action')}\n")
        return buffer.getvalue().encode('utf-8')

    overview = readmission.get('overview', {})
    if readmission.get('solution_story'):
        buffer.write(str(readmission.get('solution_story')) + '\n\n')
    if readmission.get('top_high_risk_summary'):
        buffer.write(str(readmission.get('top_high_risk_summary')) + '\n\n')
    buffer.write('\nReadmission Overview\n')
    buffer.write('-' * 20 + '\n')
    buffer.write(f"Overall readmission rate: {float(overview.get('overall_readmission_rate', 0.0)):.1%}\n")
    buffer.write(f"Readmissions in scope: {int(overview.get('readmission_count', 0)):,}\n")
    buffer.write(f"Records reviewed: {int(overview.get('records_in_scope', 0)):,}\n\n")
    if str(readmission.get('source', '')).lower() == 'synthetic' or 'synthetic' in str(readmission.get('note', '')).lower():
        buffer.write('Readmission signal note: the current workflow is using synthetic or demo-derived support rather than a native readmission field.\n\n')

    buffer.write('Key Risk Segments\n')
    buffer.write('-' * 18 + '\n')
    segments = readmission.get('high_risk_segments', pd.DataFrame())
    if isinstance(segments, pd.DataFrame) and not segments.empty:
        for _, row in segments.head(3).iterrows():
            buffer.write(f"- {row['segment_type']}: {row['segment_value']} at {float(row['readmission_rate']):.1%} ({int(row['record_count'])} records)\n")
    else:
        buffer.write('- No standout readmission segments were detected.\n')

    buffer.write('\nReadmission Drivers\n')
    buffer.write('-' * 19 + '\n')
    drivers = readmission.get('driver_table', pd.DataFrame())
    if isinstance(drivers, pd.DataFrame) and not drivers.empty:
        for _, row in drivers.head(3).iterrows():
            buffer.write(f"- {row['factor']}: {row['driver_group']} is {float(row['gap_vs_overall']):.1%} above the overall rate.\n")
    else:
        buffer.write('- Driver analysis is limited for the current dataset.\n')

    buffer.write('\nIntervention Recommendations\n')
    buffer.write('-' * 28 + '\n')
    recommendations = readmission.get('intervention_recommendations', pd.DataFrame())
    if isinstance(recommendations, pd.DataFrame) and not recommendations.empty:
        for _, row in recommendations.head(3).iterrows():
            buffer.write(
                f"- {row.get('intervention', 'Intervention')}: {row.get('target_cohort', 'Target cohort')} | "
                f"{row.get('why_it_matters', 'Focus this intervention on the highest-gap readmission cohort.')}\n"
            )
    elif not action_recommendations.empty:
        for _, row in action_recommendations.head(3).iterrows():
            buffer.write(f"- {row.get('recommendation_title', 'Recommendation')}: {row.get('rationale', '')}\n")
    else:
        buffer.write('- Review discharge follow-up, longer-stay patients, and high-risk diagnosis groups for near-term readmission reduction opportunities.\n')

    buffer.write('\nRecommended Next Steps\n')
    buffer.write('-' * 22 + '\n')
    if readmission.get('next_best_action'):
        buffer.write(f"- Next best action: {readmission.get('next_best_action')}.\n")
    buffer.write('- Validate the readmission flag or encounter timing logic before sharing the results broadly.\n')
    buffer.write('- Focus case-management review on the highest-gap segment first.\n')
    buffer.write('- Use the readmission cohort builder to compare targeted groups against the overall population.\n')
    return buffer.getvalue().encode('utf-8')
