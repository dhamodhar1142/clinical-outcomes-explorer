from __future__ import annotations

from io import StringIO

import pandas as pd

from src.reports.common import _safe_df, normalize_report_mode


def recommended_report_mode_for_role(role: str) -> str:
    if role == 'Admin':
        return 'Analyst Report'
    if role == 'Analyst':
        return 'Analyst Report'
    if role == 'Executive':
        return 'Executive Summary'
    if role == 'Clinician':
        return 'Clinical Review'
    if role == 'Researcher':
        return 'Clinical Review'
    if role == 'Data Steward':
        return 'Data Readiness Review'
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
    elif role == 'Executive':
        artifacts = [
            ('Executive Summary', 'TXT', 'Concise leadership-facing summary of operational value, readiness, and next actions.'),
            ('Executive report pack', 'TXT', 'Structured board-style summary for stakeholder walkthroughs.'),
            ('Compliance summary', 'TXT', 'High-level readiness and sharing posture note.'),
        ]
    elif role == 'Clinician':
        artifacts = [
            ('Clinical Review', 'TXT', 'Outcome-focused summary suited to clinical review.'),
            ('Readmission summary', 'TXT', 'Focused readmission-oriented review where available.'),
            ('Compliance summary', 'TXT', 'Concise note on sharing posture and synthetic support.'),
        ]
    elif role == 'Researcher':
        artifacts = [
            ('Clinical Review', 'TXT', 'Outcome-oriented summary suited to research review.'),
            ('Compliance summary', 'TXT', 'Privacy and standards context for data-sharing review.'),
            ('Compliance review CSV', 'CSV', 'Supporting compliance tables with role-aware protections.'),
        ]
    elif role == 'Data Steward':
        artifacts = [
            ('Data Readiness Review', 'TXT', 'Structured readiness and data-quality summary for stewardship review.'),
            ('Compliance summary', 'TXT', 'Standards, privacy, and governance posture note.'),
            ('Governance and audit review packet', 'TXT', 'Lineage, controls, and audit-style summary for governed handoff.'),
            ('Compliance review CSV', 'CSV', 'Supporting standards and privacy tables for remediation follow-up.'),
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
    rows: list[dict[str, object]] = [
        {
            'bundle_component': normalized_report,
            'status': 'Recommended' if export_allowed else 'Limited',
            'why_this_is_included': 'Primary audience-facing report aligned to the current review context.',
        },
        {
            'bundle_component': 'Compliance summary',
            'status': 'Recommended',
            'why_this_is_included': 'Documents standards, privacy, and sharing readiness in a compact handoff note.',
        },
    ]

    if role in {'Admin', 'Analyst', 'Data Steward'}:
        rows.append({
            'bundle_component': 'Field profile CSV',
            'status': 'Recommended',
            'why_this_is_included': 'Supports deeper validation, profiling, and technical follow-up.',
        })
    if role in {'Admin', 'Data Steward'}:
        rows.append({
            'bundle_component': 'Governance and audit review packet',
            'status': 'Recommended',
            'why_this_is_included': 'Adds lineage, controls, and audit context for governance-style review.',
        })
    elif role in {'Researcher', 'Viewer', 'Executive'}:
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


def build_shared_report_bundles(
    dataset_intelligence: dict[str, object],
    use_case_detection: dict[str, object],
    solution_packages: dict[str, object],
    healthcare: dict[str, object],
    readiness: dict[str, object],
) -> dict[str, object]:
    capability = _safe_df(dataset_intelligence.get('analytics_capability_matrix'))
    package_lookup = solution_packages.get('package_details', {}) if isinstance(solution_packages, dict) else {}
    recommended_workflow = str(use_case_detection.get('recommended_workflow', 'Healthcare Data Readiness'))

    def _module_status(modules: list[str]) -> tuple[str, str]:
        if capability.empty or 'analytics_module' not in capability.columns:
            return 'Partial', 'Capability evidence is still building for this dataset.'
        matched = capability[capability['analytics_module'].isin(modules)]
        if matched.empty:
            return 'Partial', 'The current dataset has limited direct evidence for this bundle.'
        statuses = matched['status'].astype(str).str.lower().tolist()
        supports = matched['support'].astype(str).str.lower().tolist() if 'support' in matched.columns else []
        if all(status == 'enabled' for status in statuses):
            status = 'Ready now'
        elif any(status == 'enabled' for status in statuses) or any(status == 'partial' for status in statuses):
            status = 'Partially supported'
        else:
            status = 'Needs stronger source support'
        if any('synthetic' in support for support in supports):
            detail = 'Some bundle elements rely on synthetic-assisted support.'
        elif any('inferred' in support for support in supports):
            detail = 'Some bundle elements rely on inferred support from the current dataset.'
        else:
            detail = 'The core bundle components are supported by the current dataset.'
        return status, detail

    bundles = [
        {
            'bundle_name': 'Executive Bundle',
            'primary_report_mode': 'Executive Summary',
            'best_for': 'Leadership walkthroughs, stakeholder reviews, and concise decision framing.',
            'includes': 'Executive Summary; Executive Report Pack; KPI Benchmarking; Scenario highlights; Compliance summary',
            'modules': ['Export / Executive Reporting', 'Decision Support', 'Risk Segmentation'],
            'solution_fit': 'Decision Support',
            'package_hint': package_lookup.get('Hospital Operations Review', {}),
        },
        {
            'bundle_name': 'Analyst Bundle',
            'primary_report_mode': 'Analyst Report',
            'best_for': 'Technical review, profiling, readiness remediation, and detailed handoff.',
            'includes': 'Analyst Report; Support tables CSV; Dataset Intelligence Summary; Data Quality Review; Compliance summary',
            'modules': ['Data Profiling', 'Data Quality Review', 'Predictive Modeling', 'Standards / Governance Review'],
            'solution_fit': 'Healthcare Data Readiness',
            'package_hint': package_lookup.get('Healthcare Data Readiness', {}),
        },
        {
            'bundle_name': 'Clinical Bundle',
            'primary_report_mode': 'Clinical Report',
            'best_for': 'Outcome review, readmission follow-up, pathway analysis, and fairness review.',
            'includes': 'Clinical Report; Readmission Summary; Care Pathway summary; Fairness snapshot; Priority actions',
            'modules': ['Clinical Outcome Analytics', 'Readmission Analytics', 'Care Pathway Intelligence', 'Cohort Analysis'],
            'solution_fit': 'Clinical Outcome Analytics',
            'package_hint': package_lookup.get('Clinical Outcomes Review', {}),
        },
        {
            'bundle_name': 'Operations Bundle',
            'primary_report_mode': 'Operational Report',
            'best_for': 'Utilization review, operational alerts, trend tracking, and planning.',
            'includes': 'Operational Report; Operational alerts; Readmission overview; Trend highlights; Action recommendations',
            'modules': ['Hospital Operations Analytics', 'Trend Analysis', 'Readmission Analytics', 'Provider / Facility Volume'],
            'solution_fit': 'Hospital Operations Analytics',
            'package_hint': package_lookup.get('Hospital Operations Review', {}),
        },
    ]

    rows: list[dict[str, object]] = []
    notes: list[str] = []
    for bundle in bundles:
        status, support_note = _module_status(bundle['modules'])
        recommended_steps = bundle.get('package_hint', {}).get('recommended_steps', [])
        rows.append(
            {
                'bundle_name': bundle['bundle_name'],
                'status': status,
                'primary_report_mode': bundle['primary_report_mode'],
                'best_for': bundle['best_for'],
                'included_outputs': bundle['includes'],
                'solution_fit': bundle['solution_fit'],
                'current_fit': 'Recommended now' if bundle['solution_fit'] == recommended_workflow else 'Available option',
                'support_note': support_note,
                'recommended_steps': ' | '.join(recommended_steps[:3]) if recommended_steps else 'Use the recommended workflow steps already shown in Data Intake.',
            }
        )
        if status != 'Ready now':
            notes.append(f"{bundle['bundle_name']} is {status.lower()} because some supporting modules are not fully available for the current dataset.")

    if not notes:
        notes.append('All shared report bundles are ready to support different stakeholder conversations with the current dataset.')

    return {'bundle_manifest': pd.DataFrame(rows), 'bundle_notes': notes}


def build_shared_report_bundle_text(bundle_manifest: pd.DataFrame) -> bytes:
    buffer = StringIO()
    buffer.write('Shared Report Bundles\n')
    buffer.write('=' * 21 + '\n\n')
    if bundle_manifest.empty:
        buffer.write('No shared report bundles are available for the current dataset.\n')
        return buffer.getvalue().encode('utf-8')
    for _, row in bundle_manifest.iterrows():
        buffer.write(f"{row['bundle_name']}\n")
        buffer.write('-' * len(str(row['bundle_name'])) + '\n')
        buffer.write(f"Status: {row['status']}\n")
        buffer.write(f"Primary report: {row['primary_report_mode']}\n")
        buffer.write(f"Best for: {row['best_for']}\n")
        buffer.write(f"Included outputs: {row['included_outputs']}\n")
        buffer.write(f"Current fit: {row['current_fit']}\n")
        buffer.write(f"Support note: {row['support_note']}\n\n")
    return buffer.getvalue().encode('utf-8')


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
    elif role == 'Executive':
        items = ['Executive Summary', 'Executive report pack', 'Compliance summary']
    elif role == 'Clinician':
        items = ['Clinical Review', 'Readmission summary', 'Compliance summary']
    elif role == 'Researcher':
        items = ['Clinical or analyst report', 'Compliance summary', 'Compliance review CSV']
    elif role == 'Data Steward':
        items = ['Data Readiness Review', 'Compliance summary', 'Governance and audit review packet', 'Compliance review CSV']
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
