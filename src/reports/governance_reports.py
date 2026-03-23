from __future__ import annotations

from io import StringIO

import pandas as pd

from src.reports.common import _combine_report_tables, apply_role_based_redaction, dataframe_to_csv_bytes


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

    return apply_role_based_redaction(buffer.getvalue().encode('utf-8'), role, privacy_review)


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

    compliance_payload = build_compliance_dashboard_payload(standards_validation, privacy_review, role=role, policy_name=policy_name)
    buffer.write('Compliance Summary\n')
    buffer.write('-' * 18 + '\n')
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
        {'focus_area': 'Standards readiness', 'status': standards_validation.get('badge_text', 'Early readiness'), 'detail': f"Detected standard: {standards_validation.get('detected_standard', 'None')}"},
        {'focus_area': 'Privacy posture', 'status': hipaa.get('risk_level', 'Low'), 'detail': f"Sensitive columns detected: {len(sensitive_fields) if isinstance(sensitive_fields, pd.DataFrame) else 0}"},
        {'focus_area': 'Governance depth', 'status': 'Tracked' if (len(transformation_steps) if isinstance(transformation_steps, pd.DataFrame) else 0) > 0 else 'Basic', 'detail': f"Transformations: {len(transformation_steps) if isinstance(transformation_steps, pd.DataFrame) else 0} | Active controls: {len(active_controls) if isinstance(active_controls, pd.DataFrame) else 0}"},
        {'focus_area': 'Export posture', 'status': policy_name, 'detail': f"Role context: {role}"},
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
