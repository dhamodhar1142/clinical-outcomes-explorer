from __future__ import annotations

from typing import Any

import pandas as pd

from src.export_utils import recommended_report_mode_for_role


def _safe_df(value: Any) -> pd.DataFrame:
    return value if isinstance(value, pd.DataFrame) else pd.DataFrame()


MODULE_SECTION_MAP = {
    'Dataset Intelligence Summary': 'Dataset Profile · Overview',
    'Executive Snapshot': 'Dataset Profile · Overview',
    'Data Quality Review': 'Data Quality · Quality Review',
    'Analysis Readiness': 'Data Quality · Analysis Readiness',
    'Field Profiling': 'Dataset Profile · Field Profiling',
    'Healthcare Intelligence': 'Healthcare Analytics · Healthcare Intelligence',
    'Cohort Analysis': 'Healthcare Analytics · Cohort Analysis',
    'Survival & Outcome Analysis': 'Healthcare Analytics · Healthcare Intelligence',
    'Care Pathway Intelligence': 'Healthcare Analytics · Healthcare Intelligence',
    'Readmission Analytics': 'Healthcare Analytics · Healthcare Intelligence',
    'Predictive Modeling Studio': 'Healthcare Analytics · Healthcare Intelligence',
    'Explainability & Fairness': 'Healthcare Analytics · Healthcare Intelligence',
    'Standards / Governance Review': 'Dataset Profile · Column Detection',
    'Healthcare Standards Validator': 'Healthcare Analytics · Healthcare Intelligence',
    'Data Remediation Assistant': 'Data Quality · Analysis Readiness',
    'Data Lineage': 'Data Intake',
    'Analysis Log': 'Data Intake',
    'Export Center': 'Insights & Export · Export Center',
    'Automated Insight Board': 'Insights & Export · Key Insights',
    'Action Recommendations': 'Insights & Export · Key Insights',
}


def _recommended_sections(modules: list[str]) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for module in modules:
        section = MODULE_SECTION_MAP.get(module)
        if not section or section in seen:
            continue
        seen.add(section)
        rows.append({'recommended_section': section, 'why_start_here': f'Best place to review {module.lower()} first.'})
    return pd.DataFrame(rows)


def build_audience_mode_guidance(
    role: str,
    use_case_detection: dict[str, Any] | None,
    solution_packages: dict[str, Any] | None,
    dataset_intelligence: dict[str, Any] | None,
    readiness: dict[str, Any] | None,
    healthcare: dict[str, Any] | None,
) -> dict[str, object]:
    use_case_detection = use_case_detection or {}
    solution_packages = solution_packages or {}
    dataset_intelligence = dataset_intelligence or {}
    readiness = readiness or {}
    healthcare = healthcare or {}

    package_name = str(solution_packages.get('recommended_package', 'Healthcare Data Readiness'))
    package_details = solution_packages.get('package_details', {}).get(package_name, {})
    capability = _safe_df(dataset_intelligence.get('analytics_capability_matrix'))
    use_case = str(use_case_detection.get('detected_use_case', 'Generic healthcare dataset'))
    recommended_workflow = str(use_case_detection.get('recommended_workflow', 'Healthcare Data Readiness'))
    report_mode = recommended_report_mode_for_role(role)
    readiness_score = float(readiness.get('readiness_score', 0.0) or 0.0)
    healthcare_score = float(healthcare.get('healthcare_readiness_score', 0.0) or 0.0)

    role_map: dict[str, dict[str, object]] = {
        'Analyst': {
            'focus': 'Validate data quality, interpret analytics carefully, and prepare strong technical handoff materials.',
            'recommended_modules': ['Dataset Intelligence Summary', 'Data Quality Review', 'Analysis Readiness', 'Field Profiling', 'Healthcare Intelligence'],
            'recommended_outputs': ['Analyst Report', 'Compliance summary', 'Stakeholder export bundle manifest'],
            'help_text': 'Best for users who need to understand the data deeply before sharing insights more broadly.',
            'summary_emphasis': ['Readiness and quality first', 'Validate synthetic support disclosures', 'Prepare technical handoff outputs'],
        },
        'Executive': {
            'focus': 'Review headline risk, operational readiness, major blockers, and the clearest next actions.',
            'recommended_modules': ['Executive Snapshot', 'Dataset Intelligence Summary', 'Automated Insight Board', 'Action Recommendations', 'Export Center'],
            'recommended_outputs': ['Executive Summary', 'Executive report pack', 'Print-friendly executive report'],
            'help_text': 'Best for concise stakeholder walkthroughs that emphasize business value and next-step prioritization.',
            'summary_emphasis': ['Headline risk and blockers', 'Top actions to prioritize', 'Concise export-ready summaries'],
        },
        'Clinician': {
            'focus': 'Review cohort differences, outcome patterns, pathway performance, and clinically meaningful risk signals.',
            'recommended_modules': ['Healthcare Intelligence', 'Cohort Analysis', 'Survival & Outcome Analysis', 'Care Pathway Intelligence', 'Readmission Analytics'],
            'recommended_outputs': ['Clinical Report', 'Readmission summary', 'Executive Summary'],
            'help_text': 'Best for care teams or clinical reviewers who want outcome-oriented interpretation without leaving the current workflow.',
            'summary_emphasis': ['Outcome patterns by cohort', 'High-risk and readmission signals', 'Pathway friction and follow-up priorities'],
        },
        'Researcher': {
            'focus': 'Review cohort design, model transparency, subgroup differences, and standards/privacy context for research-style use.',
            'recommended_modules': ['Cohort Analysis', 'Predictive Modeling Studio', 'Explainability & Fairness', 'Standards / Governance Review', 'Dataset Intelligence Summary'],
            'recommended_outputs': ['Clinical Review', 'Compliance summary', 'Governance and audit review packet'],
            'help_text': 'Best for users working on cohort comparisons, model interpretation, and governed data-sharing review.',
            'summary_emphasis': ['Cohort definitions and subgroup differences', 'Model transparency and limitations', 'Governed sharing posture'],
        },
        'Data Steward': {
            'focus': 'Strengthen readiness, governance, mapping confidence, and controlled sharing posture before downstream use.',
            'recommended_modules': ['Data Quality Review', 'Analysis Readiness', 'Data Remediation Assistant', 'Healthcare Standards Validator', 'Data Lineage'],
            'recommended_outputs': ['Analyst Report', 'Governance and audit review packet', 'Compliance dashboard pack'],
            'help_text': 'Best for owners of data quality, standards alignment, provenance, and safe operational handoff.',
            'summary_emphasis': ['Fix blockers before deeper analytics', 'Track provenance and standards confidence', 'Choose controlled export posture'],
        },
        'Admin': {
            'focus': 'Coordinate technical review, governance, and export posture across teams without losing operational context.',
            'recommended_modules': ['Dataset Intelligence Summary', 'Data Lineage', 'Export Center', 'Analysis Log', 'Healthcare Standards Validator'],
            'recommended_outputs': ['Analyst Report', 'Governance and audit review packet', 'Compliance dashboard pack'],
            'help_text': 'Best for broad platform review and cross-functional oversight.',
            'summary_emphasis': ['Cross-team readiness snapshot', 'Governance and audit posture', 'Controlled export planning'],
        },
        'Viewer': {
            'focus': 'Use concise summaries and guided exports to understand what the dataset supports without diving into technical detail.',
            'recommended_modules': ['Executive Snapshot', 'Dataset Intelligence Summary', 'Automated Insight Board', 'Export Center'],
            'recommended_outputs': ['Executive Summary', 'Print-friendly executive report', 'Compliance summary'],
            'help_text': 'Best for lightweight stakeholder review with controlled detail.',
            'summary_emphasis': ['What the dataset supports now', 'Top findings and actions', 'Simple guided exports'],
        },
    }

    role_profile = role_map.get(role, role_map['Analyst'])
    package_modules = [str(item) for item in package_details.get('relevant_modules', [])]
    package_prompts = [str(item) for item in package_details.get('suggested_prompts', [])]

    if not capability.empty and 'analytics_module' in capability.columns and 'status' in capability.columns:
        lookup = capability.set_index('analytics_module')
        available_for_role = [
            module
            for module in role_profile['recommended_modules']
            if module not in lookup.index or str(lookup.loc[module, 'status']).lower() != 'blocked'
        ]
        limited_rows = capability[
            capability['analytics_module'].isin(role_profile['recommended_modules'])
            & capability['status'].astype(str).isin(['partial', 'blocked'])
        ][['analytics_module', 'status', 'rationale']].rename(
            columns={
                'analytics_module': 'module',
                'status': 'support_status',
                'rationale': 'why_limited',
            }
        )
    else:
        available_for_role = list(role_profile['recommended_modules'])
        limited_rows = pd.DataFrame(columns=['module', 'support_status', 'why_limited'])

    if readiness_score >= 0.7:
        status_label = 'Ready now'
    elif readiness_score >= 0.4 or healthcare_score >= 0.4:
        status_label = 'Partially supported'
    else:
        status_label = 'Needs stronger source support'

    narrative = (
        f"For the {role.lower()} audience, the strongest starting point is {recommended_workflow.lower()} "
        f"through the {package_name} package. This dataset looks most like a {use_case.lower()}, "
        f"so the guidance emphasizes {role_profile['focus'].lower()}"
    )

    outputs_table = pd.DataFrame(
        [{'recommended_output': item, 'why_it_matters': role_profile['focus']} for item in role_profile['recommended_outputs']]
    )
    modules_table = pd.DataFrame(
        [{'recommended_module': item} for item in available_for_role]
    )
    prompt_table = pd.DataFrame(
        [{'suggested_prompt': item} for item in package_prompts[:4]]
    )
    section_table = _recommended_sections(available_for_role)
    emphasis_table = pd.DataFrame([{'priority_focus': item} for item in role_profile.get('summary_emphasis', [])])

    return {
        'audience_label': role,
        'status_label': status_label,
        'recommended_workflow': recommended_workflow,
        'recommended_package': package_name,
        'recommended_report': report_mode,
        'focus_summary': str(role_profile['focus']),
        'help_text': str(role_profile['help_text']),
        'recommended_modules': modules_table,
        'recommended_outputs': outputs_table,
        'suggested_prompts': prompt_table,
        'limited_modules': limited_rows,
        'recommended_sections': section_table,
        'summary_emphasis': emphasis_table,
        'package_modules': package_modules,
        'narrative': narrative,
    }
