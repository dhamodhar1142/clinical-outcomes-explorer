from __future__ import annotations

from io import StringIO
import json

import pandas as pd


REPORT_MODE_ALIASES = {
    'Analyst Summary': 'Analyst Report',
    'Manager Summary': 'Operational Report',
    'Operations Summary': 'Operational Report',
    'Operational Review': 'Operational Report',
    'Clinical Summary': 'Clinical Report',
    'Clinical Review': 'Clinical Report',
}


def dataframe_to_csv_bytes(data: pd.DataFrame) -> bytes:
    return data.to_csv(index=False).encode('utf-8')


def json_bytes(payload: dict[str, object]) -> bytes:
    return json.dumps(payload, indent=2, default=str).encode('utf-8')


def normalize_report_mode(report_mode: str) -> str:
    return REPORT_MODE_ALIASES.get(report_mode, report_mode)


def _combine_report_tables(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for section, table in tables.items():
        if isinstance(table, pd.DataFrame) and not table.empty:
            frame = table.copy()
            frame.insert(0, 'report_section', section)
            frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=['report_section'])
    return pd.concat(frames, ignore_index=True, sort=False)


def build_report_support_tables(report_mode: str, overview: dict[str, object], quality: dict[str, object], readiness: dict[str, object], healthcare: dict[str, object], action_recommendations: pd.DataFrame) -> dict[str, pd.DataFrame]:
    normalized_mode = normalize_report_mode(report_mode)
    overview_table = pd.DataFrame([
        {'metric': 'Rows', 'value': overview['rows']},
        {'metric': 'Columns', 'value': overview['columns']},
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


def build_text_report(dataset_name: str, overview: dict[str, object], structure, field_profile: pd.DataFrame, quality: dict[str, object], semantic: dict[str, object], readiness: dict[str, object], healthcare: dict[str, object], insights: dict[str, object]) -> bytes:
    buffer = StringIO()
    buffer.write('Smart Dataset Analyzer Summary\n')
    buffer.write('=' * 32 + '\n\n')
    buffer.write(f'Dataset: {dataset_name}\n')
    buffer.write(f"Rows: {overview['rows']:,}\n")
    buffer.write(f"Columns: {overview['columns']:,}\n")
    buffer.write(f"Duplicate rows: {overview['duplicate_rows']:,}\n")
    buffer.write(f"Missing values: {overview['missing_values']:,}\n")
    buffer.write(f"Memory estimate: {overview['memory_mb']:.2f} MB\n\n")

    buffer.write('Detected Column Types\n')
    buffer.write('-' * 22 + '\n')
    for _, row in structure.detection_table.iterrows():
        buffer.write(f"{row['column_name']}: {row['inferred_type']} ({row['confidence_score']:.2f})\n")
    buffer.write('\n')

    buffer.write('Semantic Mappings\n')
    buffer.write('-' * 18 + '\n')
    mapping_table = semantic['mapping_table']
    if mapping_table.empty:
        buffer.write('No strong semantic mappings were detected.\n')
    else:
        for _, row in mapping_table.iterrows():
            buffer.write(f"{row['original_column']} -> {row['semantic_label']} [{row['confidence_label']}]\n")
    buffer.write('\n')

    buffer.write('Data Quality Notes\n')
    buffer.write('-' * 18 + '\n')
    if not quality['high_missing'].empty:
        buffer.write('High missingness columns:\n')
        for _, row in quality['high_missing'].head(5).iterrows():
            buffer.write(f"- {row['column_name']}: {row['null_percentage']:.1%}\n")
    else:
        buffer.write('No major missingness issues were detected.\n')
    buffer.write('\n')

    buffer.write('Analysis Readiness\n')
    buffer.write('-' * 18 + '\n')
    for _, row in readiness['readiness_table'].iterrows():
        buffer.write(f"- {row['analysis_module']}: {row['status']}\n")
    buffer.write('\n')

    buffer.write('Key Insights\n')
    buffer.write('-' * 12 + '\n')
    for line in insights['summary_lines']:
        buffer.write(f"- {line}\n")
    buffer.write('\nRecommendations\n')
    buffer.write('-' * 15 + '\n')
    for line in insights['recommendations']:
        buffer.write(f"- {line}\n")
    if 'bmi_original_value' in field_profile['column_name'].astype(str).tolist() or 'bmi' in field_profile['column_name'].astype(str).tolist():
        buffer.write('\nRemediation Notes\n')
        buffer.write('-' * 17 + '\n')
        if 'cost_amount' in semantic.get('canonical_map', {}):
            buffer.write('- Financial analysis may include a synthetic cost field when no native source cost was available.\n')
        if 'diagnosis_code' in semantic.get('canonical_map', {}):
            buffer.write('- Clinical segmentation may include derived diagnosis labels for demo-safe analytics support.\n')
        if 'readmission' in semantic.get('canonical_map', {}):
            buffer.write('- Readmission analytics may rely on a synthetic or inferred support field when native readmission flags are unavailable.\n')

    return buffer.getvalue().encode('utf-8')


def build_executive_summary_text(dataset_name: str, overview: dict[str, object], healthcare: dict[str, object], insights: dict[str, object]) -> bytes:
    buffer = StringIO()
    buffer.write('Executive Summary\n')
    buffer.write('=' * 18 + '\n\n')
    buffer.write(f'Dataset: {dataset_name}\n')
    buffer.write(f"Rows reviewed: {overview['rows']:,}\n")
    buffer.write(f"Columns reviewed: {overview['columns']:,}\n")
    buffer.write(f"Duplicate rows: {overview['duplicate_rows']:,}\n")
    buffer.write(f"Missing values: {overview['missing_values']:,}\n\n")

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


def build_audience_report_text(report_mode: str, dataset_name: str, overview: dict[str, object], structure, quality: dict[str, object], semantic: dict[str, object], readiness: dict[str, object], healthcare: dict[str, object], insights: dict[str, object], action_recommendations: pd.DataFrame) -> bytes:
    normalized_mode = normalize_report_mode(report_mode)
    buffer = StringIO()
    buffer.write(f'{normalized_mode}\n')
    buffer.write('=' * len(normalized_mode) + '\n\n')
    buffer.write(f'Dataset: {dataset_name}\n')
    buffer.write(f"Rows: {overview['rows']:,} | Columns: {overview['columns']:,} | Duplicate rows: {overview['duplicate_rows']:,}\n")
    buffer.write(f"Missing values: {overview['missing_values']:,} | Analysis readiness: {readiness['readiness_score']:.0%} | Healthcare readiness: {healthcare['healthcare_readiness_score']:.0%}\n\n")

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
        numeric_count = len(getattr(structure, 'numeric_columns', []))
        date_count = len(getattr(structure, 'date_columns', []))
        categorical_count = len(getattr(structure, 'categorical_columns', []))
        appendix_lines.append(f"Detected field types: {numeric_count} numeric, {date_count} date, {categorical_count} categorical")
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
            first = action_recommendations.iloc[0]
            buffer.write(f"Priority action: {first['recommendation_title']}\n")
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
    elif normalized_mode == 'Operational Report':
        buffer.write('Use Benchmarking and Cohort Monitoring Over Time to compare the current population against baseline performance.\n')
    elif normalized_mode == 'Executive Summary':
        buffer.write('Use Presentation Mode and Action Recommendations to support a concise stakeholder discussion on priorities and trade-offs.\n')
    else:
        buffer.write('Use Survival & Outcome Analysis together with Explainability & Fairness to review the clinical implications of the current outcomes.\n')

    return buffer.getvalue().encode('utf-8')



def _privacy_redaction_note(role: str, privacy_review: dict[str, object]) -> str:
    sensitive_fields = privacy_review.get('sensitive_fields', pd.DataFrame()) if isinstance(privacy_review, dict) else pd.DataFrame()
    if sensitive_fields.empty:
        return ''
    if role in {'Viewer', 'Researcher'}:
        return 'Privacy note: sensitive column names and detailed identifier guidance are partially redacted in this export for the current role.\n\n'
    return 'Privacy note: this export may reference sensitive-field findings. Review the Privacy & Security Review panel before broader sharing.\n\n'


def apply_role_based_redaction(text_bytes: bytes, role: str, privacy_review: dict[str, object]) -> bytes:
    text = text_bytes.decode('utf-8') if isinstance(text_bytes, (bytes, bytearray)) else str(text_bytes)
    sensitive_fields = privacy_review.get('sensitive_fields', pd.DataFrame()) if isinstance(privacy_review, dict) else pd.DataFrame()
    if isinstance(sensitive_fields, pd.DataFrame) and not sensitive_fields.empty and role in {'Viewer', 'Researcher'}:
        for column_name in sensitive_fields['column_name'].astype(str).tolist():
            text = text.replace(column_name, '[sensitive field]')
    note = _privacy_redaction_note(role, privacy_review)
    return (note + text).encode('utf-8')


def build_compliance_summary_text(dataset_name: str, standards_validation: dict[str, object], privacy_review: dict[str, object], role: str = 'Analyst') -> bytes:
    buffer = StringIO()
    buffer.write('Compliance Readiness Summary\n')
    buffer.write('=' * 28 + '\n\n')
    buffer.write(f'Dataset: {dataset_name}\n')
    buffer.write(f'Role context: {role}\n\n')

    if standards_validation.get('available'):
        buffer.write('Healthcare Standards Review\n')
        buffer.write('-' * 27 + '\n')
        buffer.write(f"Detected standard: {standards_validation.get('detected_standard', 'None')}\n")
        buffer.write(f"Standards readiness: {standards_validation.get('combined_readiness_score', 0):.0f}/100\n")
        buffer.write(f"Badge: {standards_validation.get('badge_text', 'Early readiness')}\n")
        buffer.write(f"Missing required fields: {standards_validation.get('missing_required_fields', 0)}\n")
        recommendations = standards_validation.get('recommendations', pd.DataFrame())
        if isinstance(recommendations, pd.DataFrame) and not recommendations.empty:
            buffer.write('Top remediation items:\n')
            for _, row in recommendations.head(5).iterrows():
                buffer.write(f"- {row['missing_field']}: {row['recommended_mapping']}\n")
        buffer.write('\n')
    else:
        buffer.write('Healthcare Standards Review\n')
        buffer.write('-' * 27 + '\n')
        buffer.write(str(standards_validation.get('reason', 'Standards review is not available for this dataset.')) + '\n\n')

    if privacy_review.get('available'):
        hipaa = privacy_review.get('hipaa', {})
        buffer.write('Privacy & Security Review\n')
        buffer.write('-' * 25 + '\n')
        buffer.write(f"HIPAA-style risk: {hipaa.get('risk_level', 'Low')} ({hipaa.get('risk_score', 0)}/100)\n")
        buffer.write(f"Direct identifiers detected: {hipaa.get('direct_identifier_count', 0)}\n")
        buffer.write(f"Safe Harbor indicator: {'Ready' if hipaa.get('safe_harbor_ready') else 'Needs review'}\n")
        rule_pack = privacy_review.get('privacy_rule_pack', pd.DataFrame())
        if isinstance(rule_pack, pd.DataFrame) and not rule_pack.empty:
            buffer.write('Priority privacy actions:\n')
            for _, row in rule_pack.head(5).iterrows():
                buffer.write(f"- {row['rule_name']}: {row['recommended_action']}\n")
        buffer.write('\n')
    else:
        buffer.write('Privacy & Security Review\n')
        buffer.write('-' * 25 + '\n')
        buffer.write(str(privacy_review.get('reason', 'Privacy review is not available for this dataset.')) + '\n\n')

    output = buffer.getvalue().encode('utf-8')
    return apply_role_based_redaction(output, role, privacy_review)


def build_compliance_support_csv(standards_validation: dict[str, object], privacy_review: dict[str, object], role: str = 'Analyst') -> bytes:
    tables: dict[str, pd.DataFrame] = {}
    standards_summary = standards_validation.get('summary_table', pd.DataFrame()) if isinstance(standards_validation, dict) else pd.DataFrame()
    if isinstance(standards_summary, pd.DataFrame) and not standards_summary.empty:
        tables['Standards Summary'] = standards_summary
    standards_recommendations = standards_validation.get('recommendations', pd.DataFrame()) if isinstance(standards_validation, dict) else pd.DataFrame()
    if isinstance(standards_recommendations, pd.DataFrame) and not standards_recommendations.empty:
        tables['Standards Recommendations'] = standards_recommendations
    sensitive_fields = privacy_review.get('sensitive_fields', pd.DataFrame()) if isinstance(privacy_review, dict) else pd.DataFrame()
    if isinstance(sensitive_fields, pd.DataFrame) and not sensitive_fields.empty:
        sensitive_export = sensitive_fields.copy()
        if role in {'Viewer', 'Researcher'} and 'column_name' in sensitive_export.columns:
            sensitive_export['column_name'] = '[sensitive field]'
        tables['Sensitive Fields'] = sensitive_export
    rule_pack = privacy_review.get('privacy_rule_pack', pd.DataFrame()) if isinstance(privacy_review, dict) else pd.DataFrame()
    if isinstance(rule_pack, pd.DataFrame) and not rule_pack.empty:
        tables['Privacy Rule Pack'] = rule_pack
    return dataframe_to_csv_bytes(_combine_report_tables(tables))


def build_compliance_dashboard_payload(standards_validation: dict[str, object], privacy_review: dict[str, object], role: str = 'Analyst', policy_name: str = 'Internal Review') -> dict[str, object]:
    cdisc_report = standards_validation.get('cdisc_report', {}) if isinstance(standards_validation, dict) else {}
    interoperability_report = standards_validation.get('interoperability_report', {}) if isinstance(standards_validation, dict) else {}
    hipaa = privacy_review.get('hipaa', {}) if isinstance(privacy_review, dict) else {}
    return {
        'role_context': role,
        'export_policy': policy_name,
        'standards_overview': {
            'available': bool(standards_validation.get('available')),
            'detected_standard': standards_validation.get('detected_standard', 'None'),
            'combined_readiness_score': standards_validation.get('combined_readiness_score', 0),
            'badge_text': standards_validation.get('badge_text', 'Early readiness'),
            'missing_required_fields': standards_validation.get('missing_required_fields', 0),
        },
        'cdisc_overview': {
            'available': bool(cdisc_report.get('available')),
            'readiness_score': cdisc_report.get('readiness_score', 0),
            'badge_text': cdisc_report.get('badge_text', 'Early CDISC Readiness'),
            'likely_dataset_type': cdisc_report.get('likely_dataset_type', 'Not trial-oriented'),
            'missing_required_fields': cdisc_report.get('missing_required_fields', []),
        },
        'interoperability_overview': {
            'available': bool(interoperability_report.get('available')),
            'readiness_score': interoperability_report.get('readiness_score', 0),
            'badge_text': interoperability_report.get('badge_text', 'Early Interoperability Readiness'),
            'fhir_resources_detected': len(interoperability_report.get('fhir_resources', pd.DataFrame())),
            'hl7_patterns_detected': len(interoperability_report.get('hl7_patterns', pd.DataFrame())),
            'terminology_fields_detected': len(interoperability_report.get('terminology_validation', pd.DataFrame())),
        },
        'privacy_overview': {
            'available': bool(privacy_review.get('available')),
            'hipaa_risk_level': hipaa.get('risk_level', 'Low'),
            'hipaa_risk_score': hipaa.get('risk_score', 0),
            'direct_identifier_count': hipaa.get('direct_identifier_count', 0),
            'safe_harbor_ready': hipaa.get('safe_harbor_ready', False),
            'sensitive_column_count': len(privacy_review.get('sensitive_fields', pd.DataFrame())),
        },
    }


def build_compliance_dashboard_csv(standards_validation: dict[str, object], privacy_review: dict[str, object], role: str = 'Analyst') -> bytes:
    tables: dict[str, pd.DataFrame] = {}
    summary = build_compliance_dashboard_payload(standards_validation, privacy_review, role)
    tables['Compliance Overview'] = pd.DataFrame([
        {'section': 'Standards', 'metric': 'Detected standard', 'value': summary['standards_overview']['detected_standard']},
        {'section': 'Standards', 'metric': 'Combined readiness score', 'value': summary['standards_overview']['combined_readiness_score']},
        {'section': 'CDISC', 'metric': 'Readiness score', 'value': summary['cdisc_overview']['readiness_score']},
        {'section': 'Interoperability', 'metric': 'Readiness score', 'value': summary['interoperability_overview']['readiness_score']},
        {'section': 'Privacy', 'metric': 'HIPAA risk level', 'value': summary['privacy_overview']['hipaa_risk_level']},
        {'section': 'Privacy', 'metric': 'Sensitive column count', 'value': summary['privacy_overview']['sensitive_column_count']},
    ])
    standards_summary = standards_validation.get('summary_table', pd.DataFrame()) if isinstance(standards_validation, dict) else pd.DataFrame()
    if isinstance(standards_summary, pd.DataFrame) and not standards_summary.empty:
        tables['Standards Summary'] = standards_summary
    cdisc_validation = standards_validation.get('cdisc_report', {}).get('validation_report', pd.DataFrame()) if isinstance(standards_validation, dict) else pd.DataFrame()
    if isinstance(cdisc_validation, pd.DataFrame) and not cdisc_validation.empty:
        tables['CDISC Validation'] = cdisc_validation
    terminology = standards_validation.get('interoperability_report', {}).get('effective_terminology', pd.DataFrame()) if isinstance(standards_validation, dict) else pd.DataFrame()
    if isinstance(terminology, pd.DataFrame) and not terminology.empty:
        tables['Terminology Confirmation'] = terminology
    rule_pack = privacy_review.get('privacy_rule_pack', pd.DataFrame()) if isinstance(privacy_review, dict) else pd.DataFrame()
    if isinstance(rule_pack, pd.DataFrame) and not rule_pack.empty:
        tables['Privacy Rule Pack'] = rule_pack
    return dataframe_to_csv_bytes(_combine_report_tables(tables))




def build_governance_review_payload(
    dataset_name: str,
    source_meta: dict[str, str],
    data_lineage: dict[str, pd.DataFrame],
    audit_log: pd.DataFrame,
    standards_validation: dict[str, object],
    privacy_review: dict[str, object],
    role: str = 'Analyst',
    policy_name: str = 'Internal Review',
) -> dict[str, object]:
    compliance = build_compliance_dashboard_payload(standards_validation, privacy_review, role=role, policy_name=policy_name)
    lineage_overview = data_lineage.get('source_summary', pd.DataFrame()) if isinstance(data_lineage, dict) else pd.DataFrame()
    transformations = data_lineage.get('transformation_steps', pd.DataFrame()) if isinstance(data_lineage, dict) else pd.DataFrame()
    controls = data_lineage.get('active_controls', pd.DataFrame()) if isinstance(data_lineage, dict) else pd.DataFrame()
    return {
        'dataset_name': dataset_name,
        'source_mode': source_meta.get('source_mode', 'Unknown') if isinstance(source_meta, dict) else 'Unknown',
        'role_context': role,
        'export_policy': policy_name,
        'lineage_overview': lineage_overview.to_dict(orient='records') if isinstance(lineage_overview, pd.DataFrame) else [],
        'transformation_count': len(transformations) if isinstance(transformations, pd.DataFrame) else 0,
        'active_control_count': len(controls) if isinstance(controls, pd.DataFrame) else 0,
        'audit_event_count': len(audit_log) if isinstance(audit_log, pd.DataFrame) else 0,
        'compliance': compliance,
    }


def build_governance_review_text(
    dataset_name: str,
    source_meta: dict[str, str],
    data_lineage: dict[str, pd.DataFrame],
    audit_log: pd.DataFrame,
    standards_validation: dict[str, object],
    privacy_review: dict[str, object],
    role: str = 'Analyst',
    policy_name: str = 'Internal Review',
) -> bytes:
    buffer = StringIO()
    buffer.write('Governance & Audit Review Packet\n')
    buffer.write('=' * 30 + '\n\n')
    buffer.write(f'Dataset: {dataset_name}\n')
    buffer.write(f"Source mode: {source_meta.get('source_mode', 'Unknown') if isinstance(source_meta, dict) else 'Unknown'}\n")
    buffer.write(f'Role context: {role}\n')
    buffer.write(f'Export policy: {policy_name}\n\n')

    source_summary = data_lineage.get('source_summary', pd.DataFrame()) if isinstance(data_lineage, dict) else pd.DataFrame()
    transformation_steps = data_lineage.get('transformation_steps', pd.DataFrame()) if isinstance(data_lineage, dict) else pd.DataFrame()
    active_controls = data_lineage.get('active_controls', pd.DataFrame()) if isinstance(data_lineage, dict) else pd.DataFrame()
    if isinstance(source_summary, pd.DataFrame) and not source_summary.empty:
        buffer.write('Data Lineage Overview\n')
        buffer.write('-' * 21 + '\n')
        for _, row in source_summary.head(6).iterrows():
            buffer.write(f"- {row.get('attribute', 'Attribute')}: {row.get('value', '')}\n")
        buffer.write('\n')

    if isinstance(transformation_steps, pd.DataFrame) and not transformation_steps.empty:
        buffer.write('Transformation Steps\n')
        buffer.write('-' * 20 + '\n')
        for _, row in transformation_steps.head(8).iterrows():
            buffer.write(f"- {row.get('step', 'Step')}: {row.get('description', '')}\n")
        buffer.write('\n')

    if isinstance(active_controls, pd.DataFrame) and not active_controls.empty:
        buffer.write('Active Controls\n')
        buffer.write('-' * 15 + '\n')
        for _, row in active_controls.head(8).iterrows():
            buffer.write(f"- {row.get('control_name', 'Control')}: {row.get('control_value', '')}\n")
        buffer.write('\n')

    buffer.write('Compliance Summary\n')
    buffer.write('-' * 18 + '\n')
    compliance_payload = build_compliance_dashboard_payload(standards_validation, privacy_review, role=role, policy_name=policy_name)
    buffer.write(f"Detected standard: {compliance_payload['standards_overview']['detected_standard']}\n")
    buffer.write(f"Combined readiness score: {compliance_payload['standards_overview']['combined_readiness_score']}\n")
    buffer.write(f"HIPAA-style risk: {compliance_payload['privacy_overview']['hipaa_risk_level']}\n")
    buffer.write(f"Sensitive columns detected: {compliance_payload['privacy_overview']['sensitive_column_count']}\n\n")

    if isinstance(audit_log, pd.DataFrame) and not audit_log.empty:
        buffer.write('Recent Audit Events\n')
        buffer.write('-' * 18 + '\n')
        for _, row in audit_log.head(10).iterrows():
            buffer.write(f"- {row.get('timestamp', '')} | {row.get('action', row.get('action_type', 'Action'))}: {row.get('detail', '')}\n")
        buffer.write('\n')
    else:
        buffer.write('Recent Audit Events\n')
        buffer.write('-' * 18 + '\n')
        buffer.write('No audit events have been captured in the current session yet.\n\n')

    buffer.write('This packet is designed for governance-style review and onboarding. It supports readiness and control review, but it is not a formal certification artifact.\n')
    return buffer.getvalue().encode('utf-8')


def build_readmission_summary_text(dataset_name: str, readmission: dict[str, object], action_recommendations: pd.DataFrame) -> bytes:
    buffer = StringIO()
    buffer.write('Readmission Risk Summary\n')
    buffer.write('=' * 24 + '\n\n')
    buffer.write(f'Dataset: {dataset_name}\n')

    if not readmission.get('available'):
        buffer.write('Readmission-focused analytics are not fully available for this dataset.\n')
        missing = readmission.get('readiness', {}).get('missing_fields', readmission.get('missing_fields', []))
        if missing:
            buffer.write('Missing or weak fields:\n')
            for item in missing:
                buffer.write(f"- {str(item).replace('_', ' ')}\n")
        available = readmission.get('readiness', {}).get('available_analysis', readmission.get('available_analysis', []))
        if available:
            buffer.write('\nWhat can still be reviewed now:\n')
            for item in available:
                buffer.write(f"- {item}\n")
        return buffer.getvalue().encode('utf-8')

    overview = readmission.get('overview', {})
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

    buffer.write('\nIntervention Ideas\n')
    buffer.write('-' * 18 + '\n')
    if not action_recommendations.empty:
        for _, row in action_recommendations.head(3).iterrows():
            title = str(row.get('recommendation_title', 'Recommendation'))
            rationale = str(row.get('rationale', ''))
            buffer.write(f"- {title}: {rationale}\n")
    else:
        buffer.write('- Review discharge follow-up, longer-stay patients, and high-risk diagnosis groups for near-term readmission reduction opportunities.\n')

    buffer.write('\nRecommended Next Steps\n')
    buffer.write('-' * 22 + '\n')
    buffer.write('- Validate the readmission flag or encounter timing logic before sharing the results broadly.\n')
    buffer.write('- Focus case-management review on the highest-gap segment first.\n')
    buffer.write('- Use the readmission cohort builder to compare targeted groups against the overall population.\n')
    return buffer.getvalue().encode('utf-8')


def build_governance_review_csv(
    data_lineage: dict[str, pd.DataFrame],
    audit_log: pd.DataFrame,
    standards_validation: dict[str, object],
    privacy_review: dict[str, object],
    role: str = 'Analyst',
    policy_name: str = 'Internal Review',
) -> bytes:
    tables: dict[str, pd.DataFrame] = {}
    source_summary = data_lineage.get('source_summary', pd.DataFrame()) if isinstance(data_lineage, dict) else pd.DataFrame()
    transformations = data_lineage.get('transformation_steps', pd.DataFrame()) if isinstance(data_lineage, dict) else pd.DataFrame()
    controls = data_lineage.get('active_controls', pd.DataFrame()) if isinstance(data_lineage, dict) else pd.DataFrame()
    derived_fields = data_lineage.get('derived_fields', pd.DataFrame()) if isinstance(data_lineage, dict) else pd.DataFrame()
    if isinstance(source_summary, pd.DataFrame) and not source_summary.empty:
        tables['Lineage Source Summary'] = source_summary
    if isinstance(transformations, pd.DataFrame) and not transformations.empty:
        tables['Transformation Steps'] = transformations
    if isinstance(controls, pd.DataFrame) and not controls.empty:
        tables['Active Controls'] = controls
    if isinstance(derived_fields, pd.DataFrame) and not derived_fields.empty:
        tables['Derived Fields'] = derived_fields
    if isinstance(audit_log, pd.DataFrame) and not audit_log.empty:
        tables['Audit Log'] = audit_log
    summary = build_compliance_dashboard_payload(standards_validation, privacy_review, role=role, policy_name=policy_name)
    tables['Governance Compliance Overview'] = pd.DataFrame([
        {'section': 'Standards', 'metric': 'Detected standard', 'value': summary['standards_overview']['detected_standard']},
        {'section': 'Standards', 'metric': 'Combined readiness score', 'value': summary['standards_overview']['combined_readiness_score']},
        {'section': 'Privacy', 'metric': 'HIPAA risk level', 'value': summary['privacy_overview']['hipaa_risk_level']},
        {'section': 'Privacy', 'metric': 'Sensitive column count', 'value': summary['privacy_overview']['sensitive_column_count']},
    ])
    return dataframe_to_csv_bytes(_combine_report_tables(tables))

def build_compliance_handoff_payload(
    dataset_name: str,
    standards_validation: dict[str, object],
    privacy_review: dict[str, object],
    data_lineage: dict[str, pd.DataFrame],
    role: str = 'Analyst',
    policy_name: str = 'Internal Review',
) -> dict[str, object]:
    hipaa = privacy_review.get('hipaa', {}) if isinstance(privacy_review, dict) else {}
    transformation_steps = data_lineage.get('transformation_steps', pd.DataFrame()) if isinstance(data_lineage, dict) else pd.DataFrame()
    active_controls = data_lineage.get('active_controls', pd.DataFrame()) if isinstance(data_lineage, dict) else pd.DataFrame()
    sensitive_fields = privacy_review.get('sensitive_fields', pd.DataFrame()) if isinstance(privacy_review, dict) else pd.DataFrame()
    summary_table = [
        {
            'focus_area': 'Standards readiness',
            'status': standards_validation.get('badge_text', 'Early readiness'),
            'detail': f"Detected standard: {standards_validation.get('detected_standard', 'None')}",
        },
        {
            'focus_area': 'Privacy posture',
            'status': hipaa.get('risk_level', 'Low'),
            'detail': f"Sensitive columns detected: {len(sensitive_fields) if isinstance(sensitive_fields, pd.DataFrame) else 0}",
        },
        {
            'focus_area': 'Governance depth',
            'status': 'Tracked' if (len(transformation_steps) if isinstance(transformation_steps, pd.DataFrame) else 0) > 0 else 'Basic',
            'detail': f"Transformations: {len(transformation_steps) if isinstance(transformation_steps, pd.DataFrame) else 0} | Active controls: {len(active_controls) if isinstance(active_controls, pd.DataFrame) else 0}",
        },
        {
            'focus_area': 'Export posture',
            'status': policy_name,
            'detail': f"Role context: {role}",
        },
    ]
    return {
        'dataset_name': dataset_name,
        'role_context': role,
        'export_policy': policy_name,
        'standards_readiness_score': float(standards_validation.get('combined_readiness_score', 0.0)),
        'privacy_risk': hipaa.get('risk_level', 'Low'),
        'governance_step_count': len(transformation_steps) if isinstance(transformation_steps, pd.DataFrame) else 0,
        'export_posture': policy_name,
        'summary_table': summary_table,
    }
def recommended_report_mode_for_role(role: str) -> str:
    if role == 'Admin':
        return 'Analyst Report'
    if role == 'Analyst':
        return 'Analyst Report'
    if role == 'Researcher':
        return 'Clinical Review'
    return 'Executive Summary'


def build_role_export_bundle_manifest(role: str, policy_name: str, export_allowed: bool, report_mode: str, privacy_review: dict[str, object]) -> pd.DataFrame:
    if role == 'Admin':
        artifacts = [
            ('Summary report', 'TXT', 'Core cross-functional summary of the current dataset review.'),
            (report_mode, 'TXT', 'Audience-focused report for deeper review.'),
            ('Compliance summary', 'TXT', 'Standards, privacy, and readiness summary.'),
            ('Compliance review CSV', 'CSV', 'Supporting standards and privacy review tables.'),
            ('Semantic mapping JSON', 'JSON', 'Canonical field mapping summary for technical handoff.'),
        ]
    elif role == 'Analyst':
        artifacts = [
            ('Summary report', 'TXT', 'Core analytical summary for handoff or review.'),
            (report_mode, 'TXT', 'Audience-focused report for detailed review.'),
            ('Field profile CSV', 'CSV', 'Column-level profiling output for follow-up analysis.'),
            ('Compliance summary', 'TXT', 'Readiness, standards, and privacy notes.'),
        ]
    elif role == 'Researcher':
        artifacts = [
            ('Clinical Review', 'TXT', 'Outcome-oriented summary suited to research review.'),
            ('Compliance summary', 'TXT', 'Privacy and standards context for data-sharing review.'),
            ('Compliance review CSV', 'CSV', 'Supporting compliance tables with role-aware protections.'),
        ]
    else:
        artifacts = [
            ('Executive Summary', 'TXT', 'Board-style high-level summary for non-technical review.'),
            ('Compliance summary', 'TXT', 'Concise note on sharing readiness and protections.'),
        ]

    rows = [
        {
            'artifact': artifact,
            'format': fmt,
            'included_for_role': 'Yes' if export_allowed else 'Limited',
            'policy_context': policy_name,
            'why_it_matters': note,
        }
        for artifact, fmt, note in artifacts
    ]
    sensitive_fields = privacy_review.get('sensitive_fields', pd.DataFrame()) if isinstance(privacy_review, dict) else pd.DataFrame()
    if isinstance(sensitive_fields, pd.DataFrame) and not sensitive_fields.empty:
        rows.append({
            'artifact': 'Role-aware masking note',
            'format': 'TXT',
            'included_for_role': 'Yes',
            'policy_context': policy_name,
            'why_it_matters': 'Documents how sensitive field names and content are handled for the current role and export policy.',
        })
    return pd.DataFrame(rows)



def build_policy_aware_bundle_profile(role: str, report_mode: str, policy_name: str, export_allowed: bool, privacy_review: dict[str, object]) -> tuple[str, pd.DataFrame]:
    normalized_report = normalize_report_mode(report_mode)
    rows: list[dict[str, object]] = []
    rows.append({
        'bundle_component': normalized_report,
        'status': 'Recommended' if export_allowed else 'Limited',
        'why_this_is_included': 'Primary audience-facing report aligned to the current review context.',
    })
    rows.append({
        'bundle_component': 'Compliance summary',
        'status': 'Recommended',
        'why_this_is_included': 'Documents standards, privacy, and sharing readiness in a compact handoff note.',
    })

    if role in {'Admin', 'Analyst'}:
        rows.append({
            'bundle_component': 'Field profile CSV',
            'status': 'Recommended',
            'why_this_is_included': 'Supports deeper validation, profiling, and technical follow-up.',
        })
    if role == 'Admin':
        rows.append({
            'bundle_component': 'Governance and audit review packet',
            'status': 'Recommended',
            'why_this_is_included': 'Adds lineage, controls, and audit context for governance-style review.',
        })
    elif role in {'Researcher', 'Viewer'}:
        rows.append({
            'bundle_component': 'Executive summary',
            'status': 'Optional',
            'why_this_is_included': 'Provides a concise narrative for low-friction sharing with non-technical stakeholders.',
        })

    sensitive_fields = privacy_review.get('sensitive_fields', pd.DataFrame()) if isinstance(privacy_review, dict) else pd.DataFrame()
    risk_level = privacy_review.get('hipaa', {}).get('risk_level', 'Low') if isinstance(privacy_review, dict) else 'Low'
    if isinstance(sensitive_fields, pd.DataFrame) and not sensitive_fields.empty:
        rows.append({
            'bundle_component': 'Privacy-aware export policy note',
            'status': 'Recommended',
            'why_this_is_included': f'Supports sharing under the {policy_name} posture with {risk_level.lower()}-to-high sensitivity review signals.',
        })

    if not export_allowed:
        summary = f'{role} access currently limits direct file downloads, so this bundle is guidance-first under the {policy_name} policy.'
    elif policy_name == 'Research-safe Extract':
        summary = f'This bundle emphasizes masked, research-safe sharing for the {normalized_report} review path.'
    elif policy_name == 'HIPAA-style Limited Dataset':
        summary = f'This bundle emphasizes limited-dataset sharing controls while keeping the {normalized_report} review path available.'
    else:
        summary = f'This bundle balances operational usability and governance context for the {normalized_report} review path.'

    return summary, pd.DataFrame(rows)
def apply_export_policy(text_bytes: bytes, policy_name: str, privacy_review: dict[str, object]) -> bytes:
    text = text_bytes.decode('utf-8') if isinstance(text_bytes, (bytes, bytearray)) else str(text_bytes)
    sensitive_fields = privacy_review.get('sensitive_fields', pd.DataFrame()) if isinstance(privacy_review, dict) else pd.DataFrame()
    if not isinstance(sensitive_fields, pd.DataFrame) or sensitive_fields.empty:
        return text.encode('utf-8')

    policy = str(policy_name or 'Internal Review')
    direct_identifier_labels = {'Name', 'First Name', 'Last Name', 'Ssn', 'Email', 'Phone', 'Address', 'Medical Record Number'}
    quasi_identifier_labels = {'Date Of Birth', 'Zip', 'Member Id', 'Insurance Id'}

    fields_to_redact = sensitive_fields.copy()
    if policy == 'HIPAA-style Limited Dataset':
        fields_to_redact = sensitive_fields[sensitive_fields['sensitive_type'].isin(direct_identifier_labels)]
    elif policy == 'Research-safe Extract':
        fields_to_redact = sensitive_fields[sensitive_fields['sensitive_type'].isin(direct_identifier_labels.union(quasi_identifier_labels))]

    if not fields_to_redact.empty and 'column_name' in fields_to_redact.columns:
        for column_name in fields_to_redact['column_name'].astype(str).tolist():
            text = text.replace(column_name, '[protected field]')

    if policy == 'HIPAA-style Limited Dataset':
        prefix = 'Export policy: HIPAA-style Limited Dataset. Direct identifiers are masked in this export.\n\n'
    elif policy == 'Research-safe Extract':
        prefix = 'Export policy: Research-safe Extract. Direct identifiers and major quasi-identifiers are masked in this export.\n\n'
    else:
        prefix = 'Export policy: Internal Review. Role-aware protections remain active for this export.\n\n'
    return (prefix + text).encode('utf-8')


def build_policy_note_text(policy_name: str, role: str, privacy_review: dict[str, object]) -> bytes:
    hipaa = privacy_review.get('hipaa', {}) if isinstance(privacy_review, dict) else {}
    sensitive_fields = privacy_review.get('sensitive_fields', pd.DataFrame()) if isinstance(privacy_review, dict) else pd.DataFrame()
    direct_count = hipaa.get('direct_identifier_count', 0)
    risk_level = hipaa.get('risk_level', 'Low')
    policy = str(policy_name or 'Internal Review')

    lines = [
        'Privacy & Sharing Note',
        '----------------------',
        f'Active export policy: {policy}',
        f'Active role: {role}',
        f'HIPAA-style risk: {risk_level}',
        f'Sensitive columns detected: {len(sensitive_fields) if isinstance(sensitive_fields, pd.DataFrame) else 0}',
    ]

    if policy == 'HIPAA-style Limited Dataset':
        lines.append(f'Direct identifiers prioritized for masking: {direct_count}')
        lines.append('Use this version for broader operational review after masking direct identifiers.')
    elif policy == 'Research-safe Extract':
        lines.append('This version is prepared for lower-risk research-style sharing with stronger masking.')
    else:
        lines.append('This version is intended for controlled internal review under the current role settings.')

    lines.append('These protections are onboarding-oriented safeguards, not formal legal certification controls.')
    return ('\n'.join(lines) + '\n\n').encode('utf-8')


def build_role_export_bundle_text(role: str, policy_name: str, export_allowed: bool, report_mode: str, privacy_review: dict[str, object]) -> bytes:
    buffer = StringIO()
    buffer.write('Role-aware Export Bundle Guide\n')
    buffer.write('=' * 29 + '\n\n')
    buffer.write(f'Role: {role}\n')
    buffer.write(f'Export policy: {policy_name}\n')
    buffer.write(f"Exports enabled: {'Yes' if export_allowed else 'No'}\n\n")
    buffer.write('Recommended bundle contents\n')
    buffer.write('-' * 27 + '\n')

    if role == 'Admin':
        items = ['Summary report', report_mode, 'Compliance summary', 'Compliance review CSV', 'Semantic mapping JSON']
    elif role == 'Analyst':
        items = ['Summary report', report_mode, 'Field profile CSV', 'Compliance summary']
    elif role == 'Researcher':
        items = ['Clinical or analyst report', 'Compliance summary', 'Compliance review CSV']
    else:
        items = ['Executive summary', 'Compliance summary']

    for item in items:
        buffer.write(f'- {item}\n')

    sensitive_fields = privacy_review.get('sensitive_fields', pd.DataFrame()) if isinstance(privacy_review, dict) else pd.DataFrame()
    if isinstance(sensitive_fields, pd.DataFrame) and not sensitive_fields.empty:
        buffer.write('\nProtection notes\n')
        buffer.write('-' * 16 + '\n')
        buffer.write(f"- Sensitive fields detected: {len(sensitive_fields)}\n")
        if role in {'Viewer', 'Researcher'}:
            buffer.write('- Detailed sensitive-field names are redacted for this role.\n')

    return buffer.getvalue().encode('utf-8')



