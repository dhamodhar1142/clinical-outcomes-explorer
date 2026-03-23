from __future__ import annotations

from typing import Any

import pandas as pd


def _safe_df(value: Any) -> pd.DataFrame:
    return value if isinstance(value, pd.DataFrame) else pd.DataFrame()


def _status_for_modules(capability: pd.DataFrame, modules: list[str]) -> str:
    if capability.empty or 'analytics_module' not in capability.columns or 'status' not in capability.columns:
        return 'limited'
    subset = capability[capability['analytics_module'].isin(modules)]
    if subset.empty:
        return 'limited'
    statuses = set(subset['status'].astype(str).str.lower())
    if 'enabled' in statuses:
        return 'enabled'
    if 'partial' in statuses:
        return 'partial'
    return 'limited'


def _solution_card(
    title: str,
    audience: str,
    modules: list[str],
    outcomes: list[str],
    capability: pd.DataFrame,
) -> dict[str, object]:
    status = _status_for_modules(capability, modules)
    status_label = {
        'enabled': 'Ready now',
        'partial': 'Partially supported',
        'limited': 'Needs stronger source support',
    }[status]
    return {
        'solution_group': title,
        'status': status,
        'status_label': status_label,
        'who_it_is_for': audience,
        'modules': modules,
        'outcomes_supported': outcomes,
    }


def build_solution_layer_guidance(
    dataset_intelligence: dict[str, Any] | None,
    healthcare: dict[str, Any] | None,
    readiness: dict[str, Any] | None,
) -> dict[str, object]:
    dataset_intelligence = dataset_intelligence or {}
    healthcare = healthcare or {}
    readiness = readiness or {}
    capability = _safe_df(dataset_intelligence.get('analytics_capability_matrix'))

    solutions = [
        _solution_card(
            'Healthcare Data Readiness',
            'Data stewards, implementation teams, and analysts preparing healthcare datasets for broader use.',
            ['Data Profiling', 'Data Quality Review', 'Standards / Governance Review', 'Export / Executive Reporting'],
            [
                'Profile incoming healthcare data quickly',
                'Clarify blockers, governance posture, and remediation paths',
                'Prepare onboarding-ready summaries for downstream teams',
            ],
            capability,
        ),
        _solution_card(
            'Hospital Operations Analytics',
            'Hospital operations, service-line, and quality improvement teams reviewing utilization and workflow performance.',
            ['Readmission Analytics', 'Cost Driver Analysis', 'Provider / Facility Volume', 'Trend Analysis', 'Export / Executive Reporting'],
            [
                'Track operational risk, utilization, and service concentration',
                'Review readmission, cost, and provider-style patterns',
                'Support operational reporting and follow-up planning',
            ],
            capability,
        ),
        _solution_card(
            'Clinical Outcome Analytics',
            'Clinical analysts, researchers, and care teams reviewing outcome, cohort, and pathway patterns.',
            ['Risk Segmentation', 'Cohort Analysis', 'Predictive Modeling', 'Care Pathway Intelligence', 'Readmission Analytics'],
            [
                'Review outcome, pathway, and high-risk patient patterns',
                'Compare clinical cohorts and intervention scenarios',
                'Support explainable predictive and pathway reviews',
            ],
            capability,
        ),
        _solution_card(
            'Population Health Intelligence',
            'Population health, quality, and preventive health teams looking for segment-level opportunity areas.',
            ['Cohort Analysis', 'Predictive Modeling', 'Trend Analysis', 'Cohort Monitoring Over Time', 'Risk Segmentation'],
            [
                'Monitor segment-level risk and disparities over time',
                'Identify preventable or escalating risk patterns',
                'Support cohort-focused outreach and benchmarking',
            ],
            capability,
        ),
        _solution_card(
            'Decision Support',
            'Decision-makers, analytics managers, and cross-functional teams needing concise action guidance.',
            ['Predictive Modeling', 'Readmission Analytics', 'Export / Executive Reporting', 'Standards / Governance Review'],
            [
                'Prioritize interventions, exports, and operational next steps',
                'Translate analytics into executive-facing recommendations',
                'Support structured walkthroughs for stakeholders and recruiters',
            ],
            capability,
        ),
    ]

    status_counts = {
        'enabled': sum(1 for item in solutions if item['status'] == 'enabled'),
        'partial': sum(1 for item in solutions if item['status'] == 'partial'),
        'limited': sum(1 for item in solutions if item['status'] == 'limited'),
    }
    recommended_start = next((item['solution_group'] for item in solutions if item['status'] == 'enabled'), None)
    if recommended_start is None:
        recommended_start = next((item['solution_group'] for item in solutions if item['status'] == 'partial'), 'Healthcare Data Readiness')

    summary = {
        'solution_layers': len(solutions),
        'ready_now': status_counts['enabled'],
        'partially_supported': status_counts['partial'],
        'needs_stronger_source_support': status_counts['limited'],
        'recommended_starting_point': recommended_start,
    }
    narrative = (
        f"This dataset is currently strongest for {recommended_start}. "
        f"{status_counts['enabled']} solution group(s) are ready now, "
        f"{status_counts['partial']} are partially supported, and "
        f"{status_counts['limited']} need stronger source data."
    )

    return {
        'summary': summary,
        'solution_cards': pd.DataFrame(solutions),
        'narrative': narrative,
        'healthcare_readiness_score': healthcare.get('healthcare_readiness_score', 0.0),
        'available_modules': int(readiness.get('available_count', 0)),
    }


def build_use_case_detection(
    dataset_intelligence: dict[str, Any] | None,
    readiness: dict[str, Any] | None,
    healthcare: dict[str, Any] | None,
) -> dict[str, object]:
    dataset_intelligence = dataset_intelligence or {}
    readiness = readiness or {}
    healthcare = healthcare or {}
    capability = _safe_df(dataset_intelligence.get('analytics_capability_matrix'))
    dataset_type = str(dataset_intelligence.get('dataset_type_label', 'Generic tabular dataset'))
    dataset_conf = float(dataset_intelligence.get('dataset_type_confidence', 0.0) or 0.0)
    rationale = str(dataset_intelligence.get('dataset_type_rationale', 'The current dataset has limited structural evidence.'))

    use_case = 'Generic healthcare dataset'
    workflow = 'Healthcare Data Readiness'
    relevant_modules = ['Data Profiling', 'Data Quality Review', 'Standards / Governance Review', 'Export / Executive Reporting']

    if 'Claims / encounter-level' in dataset_type:
        use_case = 'Cost/claims-style dataset'
        workflow = 'Hospital Operations Analytics'
        relevant_modules = ['Readmission Analytics', 'Cost Driver Analysis', 'Provider / Facility Volume', 'Trend Analysis']
    elif 'Clinical / patient-level' in dataset_type:
        use_case = 'Patient-level clinical dataset'
        workflow = 'Clinical Outcome Analytics'
        relevant_modules = ['Risk Segmentation', 'Cohort Analysis', 'Predictive Modeling', 'Care Pathway Intelligence']
    elif 'Trial / research-oriented' in dataset_type:
        use_case = 'Patient-level clinical dataset'
        workflow = 'Clinical Outcome Analytics'
        relevant_modules = ['Cohort Analysis', 'Predictive Modeling', 'Standards / Governance Review', 'Export / Executive Reporting']
    elif 'Mixed healthcare operational' in dataset_type:
        use_case = 'Hospital reporting dataset'
        workflow = 'Hospital Operations Analytics'
        relevant_modules = ['Provider / Facility Volume', 'Trend Analysis', 'Export / Executive Reporting', 'Standards / Governance Review']
    elif 'Healthcare-related' in dataset_type:
        use_case = 'Generic healthcare dataset'
        workflow = 'Healthcare Data Readiness'
        relevant_modules = ['Data Profiling', 'Data Quality Review', 'Risk Segmentation', 'Export / Executive Reporting']

    if not capability.empty and 'analytics_module' in capability.columns:
        relevant_rows = capability[capability['analytics_module'].isin(relevant_modules)].copy()
        blocked_rows = capability[capability['status'].astype(str).eq('blocked')].copy()
        blocked_rows = blocked_rows.copy()
        blocked_rows['rationale'] = blocked_rows.get('rationale', pd.Series(index=blocked_rows.index, dtype=str)).fillna('Additional source-grade fields are needed to enable this module.')
        unavailable_rows = blocked_rows[['analytics_module', 'rationale']].rename(
            columns={'analytics_module': 'module', 'rationale': 'why_unavailable'}
        )
        partial_rows = capability[capability['status'].astype(str).eq('partial')].copy()
        partial_rows['rationale'] = partial_rows.get('rationale', pd.Series(index=partial_rows.index, dtype=str)).fillna('This module is available in a limited mode with the current dataset.')
        partially_available_rows = partial_rows[['analytics_module', 'rationale']].rename(
            columns={'analytics_module': 'module', 'rationale': 'why_limited'}
        )
        ready_modules = relevant_rows[relevant_rows['status'].astype(str).eq('enabled')]['analytics_module'].astype(str).tolist()
        if ready_modules:
            relevant_modules = ready_modules + [m for m in relevant_modules if m not in ready_modules]
    else:
        unavailable_rows = pd.DataFrame()
        partially_available_rows = pd.DataFrame()

    narrative = (
        f"The dataset looks most like a {use_case.lower()} and is best matched to the "
        f"{workflow} workflow. {rationale}"
    )

    return {
        'detected_use_case': use_case,
        'detected_use_case_confidence': round(dataset_conf, 2),
        'detected_use_case_rationale': rationale,
        'recommended_workflow': workflow,
        'relevant_modules': relevant_modules,
        'unavailable_modules': unavailable_rows,
        'partially_available_modules': partially_available_rows,
        'narrative': narrative,
        'healthcare_readiness_score': healthcare.get('healthcare_readiness_score', 0.0),
        'available_module_count': int(readiness.get('available_count', 0)),
    }


def build_solution_packages(
    dataset_intelligence: dict[str, Any] | None,
    use_case_detection: dict[str, Any] | None,
) -> dict[str, object]:
    dataset_intelligence = dataset_intelligence or {}
    use_case_detection = use_case_detection or {}
    capability = _safe_df(dataset_intelligence.get('analytics_capability_matrix'))

    packages = [
        {
            'package_name': 'Healthcare Data Readiness',
            'description': 'Best for onboarding new healthcare data, reviewing blockers, and preparing governance-ready summaries.',
            'recommended_steps': [
                'Start in Dataset Profile · Overview to understand the dataset type and capability posture.',
                'Use Data Quality · Analysis Readiness to review blockers, remediation, and standards/privacy posture.',
                'Finish in Export Center with an analyst or executive readiness summary.',
            ],
            'relevant_modules': ['Data Profiling', 'Data Quality Review', 'Standards / Governance Review', 'Export / Executive Reporting'],
            'suggested_prompts': [
                'Show remediation suggestions',
                'Summarize this dataset',
                'Generate analyst report',
            ],
            'recommended_exports': ['Analyst Report', 'Executive Summary'],
        },
        {
            'package_name': 'Hospital Operations Review',
            'description': 'Best for utilization, provider/service concentration, readmission, and operational planning workflows.',
            'recommended_steps': [
                'Review Healthcare Intelligence for volume, readmission, and operational risk signals.',
                'Use Trend Analysis and Cohort Analysis to understand variation over time and by segment.',
                'Generate an operational or executive export for leadership review.',
            ],
            'relevant_modules': ['Readmission Analytics', 'Provider / Facility Volume', 'Trend Analysis', 'Cost Driver Analysis', 'Export / Executive Reporting'],
            'suggested_prompts': [
                'Show readmission by department',
                'Compare treatments',
                'Generate operational report',
            ],
            'recommended_exports': ['Operational Report', 'Executive Summary'],
        },
        {
            'package_name': 'Clinical Outcomes Review',
            'description': 'Best for patient-level outcome, cohort, pathway, and predictive modeling review.',
            'recommended_steps': [
                'Start in Healthcare Intelligence to review risk, pathway, and outcome-focused signals.',
                'Use Cohort Analysis to compare the highest-risk or most clinically important groups.',
                'Use Predictive Modeling and Export Center for explainable clinical review outputs.',
            ],
            'relevant_modules': ['Risk Segmentation', 'Cohort Analysis', 'Predictive Modeling', 'Care Pathway Intelligence', 'Readmission Analytics'],
            'suggested_prompts': [
                'Which patients are highest readmission risk?',
                'What factors drive readmission?',
                'Generate clinical report',
            ],
            'recommended_exports': ['Clinical Report', 'Executive Summary'],
        },
        {
            'package_name': 'Population Health Review',
            'description': 'Best for segment-level variation, cohort monitoring, predictive risk, and preventive opportunity review.',
            'recommended_steps': [
                'Review cohort and segment patterns in Healthcare Intelligence and Cohort Analysis.',
                'Use Trend Analysis and benchmarking to understand disparities and risk concentration over time.',
                'Finish with an executive or operational export for action planning.',
            ],
            'relevant_modules': ['Cohort Analysis', 'Risk Segmentation', 'Trend Analysis', 'Cohort Monitoring Over Time', 'Predictive Modeling'],
            'suggested_prompts': [
                'Compare smoking segments',
                'Build cohort for female stage iii patients',
                'Generate executive summary',
            ],
            'recommended_exports': ['Executive Summary', 'Operational Report'],
        },
    ]

    if not capability.empty and 'analytics_module' in capability.columns and 'status' in capability.columns:
        lookup = capability.set_index('analytics_module')['status'].astype(str)
        for package in packages:
            relevant = [lookup.get(module, 'blocked') for module in package['relevant_modules']]
            if any(status == 'enabled' for status in relevant):
                package['status'] = 'Ready now'
            elif any(status == 'partial' for status in relevant):
                package['status'] = 'Partially supported'
            else:
                package['status'] = 'Needs stronger source data'
    else:
        for package in packages:
            package['status'] = 'Needs stronger source data'

    workflow_to_package = {
        'Healthcare Data Readiness': 'Healthcare Data Readiness',
        'Hospital Operations Analytics': 'Hospital Operations Review',
        'Clinical Outcome Analytics': 'Clinical Outcomes Review',
        'Population Health Intelligence': 'Population Health Review',
        'Decision Support': 'Clinical Outcomes Review',
    }
    recommended_package = workflow_to_package.get(
        str(use_case_detection.get('recommended_workflow', 'Healthcare Data Readiness')),
        'Healthcare Data Readiness',
    )

    return {
        'packages_table': pd.DataFrame(
            [
                {
                    'solution_package': package['package_name'],
                    'status': package['status'],
                    'description': package['description'],
                }
                for package in packages
            ]
        ),
        'package_details': {package['package_name']: package for package in packages},
        'recommended_package': recommended_package,
    }


def build_demo_guidance_system(
    dataset_intelligence: dict[str, Any] | None,
    use_case_detection: dict[str, Any] | None,
    solution_packages: dict[str, Any] | None,
) -> dict[str, object]:
    dataset_intelligence = dataset_intelligence or {}
    use_case_detection = use_case_detection or {}
    solution_packages = solution_packages or {}

    dataset_type = str(dataset_intelligence.get('dataset_type_label', 'Dataset type not classified'))
    workflow = str(use_case_detection.get('recommended_workflow', 'Healthcare Data Readiness'))
    package_name = str(solution_packages.get('recommended_package', 'Healthcare Data Readiness'))
    rationale = str(
        use_case_detection.get(
            'detected_use_case_rationale',
            dataset_intelligence.get('dataset_type_rationale', 'The current dataset is being assessed from its available structure and mapped fields.'),
        )
    )

    package_details = (solution_packages.get('package_details') or {}).get(package_name, {})
    steps = list(package_details.get('recommended_steps', []))
    if not steps:
        steps = [
            'Start in Data Quality · Analysis Readiness to review blockers and readiness.',
            'Review Healthcare Analytics modules that are currently enabled.',
            'Finish in Export Center to create a stakeholder-facing summary.',
        ]

    relevant_modules = list(use_case_detection.get('relevant_modules', []))[:5]
    unavailable = _safe_df(use_case_detection.get('unavailable_modules'))
    partial = _safe_df(use_case_detection.get('partially_available_modules'))
    unavailable_count = len(unavailable)
    partial_count = len(partial)

    narrative = (
        f"Dataset detected: {dataset_type}. Recommended workflow: {workflow}. "
        f"This path is recommended because {rationale[:1].lower() + rationale[1:] if rationale else 'the current schema best matches that workflow.'}"
    )

    highlight_lines = [
        f"Focus on {workflow} first to get the highest-value insights quickly.",
        f"{partial_count} module(s) are partially supported and {unavailable_count} are currently blocked for this workflow." if (partial_count or unavailable_count) else 'The core recommended modules are currently available for this workflow.',
        f"Recommended package: {package_name}.",
    ]

    return {
        'detected_dataset_type': dataset_type,
        'recommended_workflow': workflow,
        'recommended_package': package_name,
        'rationale': rationale,
        'recommended_steps': steps,
        'relevant_modules': relevant_modules,
        'unavailable_modules': unavailable,
        'partially_available_modules': partial,
        'narrative': narrative,
        'highlights': highlight_lines,
    }


def build_market_specific_solution_views(
    dataset_intelligence: dict[str, Any] | None,
    use_case_detection: dict[str, Any] | None,
    solution_packages: dict[str, Any] | None,
    solution_layers: dict[str, Any] | None = None,
) -> dict[str, object]:
    dataset_intelligence = dataset_intelligence or {}
    use_case_detection = use_case_detection or {}
    solution_packages = solution_packages or {}
    solution_layers = solution_layers or {}

    package_details = solution_packages.get('package_details') or {}
    package_table = _safe_df(solution_packages.get('packages_table'))
    package_status_lookup = (
        package_table.set_index('solution_package')['status'].astype(str).to_dict()
        if not package_table.empty and 'solution_package' in package_table.columns and 'status' in package_table.columns
        else {}
    )
    recommended_package = str(solution_packages.get('recommended_package', 'Healthcare Data Readiness'))
    recommended_workflow = str(use_case_detection.get('recommended_workflow', 'Healthcare Data Readiness'))
    rationale = str(use_case_detection.get('detected_use_case_rationale', dataset_intelligence.get('dataset_type_rationale', 'The current schema is being matched to the strongest-fit workflow.')))

    views = [
        {
            'solution_view': 'Hospital Operations',
            'who_it_supports': 'Hospital operators, service-line leaders, and quality teams focused on utilization and throughput.',
            'best_fit_workflow': 'Hospital Operations Analytics',
            'best_fit_package': 'Hospital Operations Review',
            'typical_outputs': 'Operational report, readmission review, provider/facility utilization summaries',
            'why_it_works': 'Best when the dataset supports volume, readmission, trend, or cost-oriented operational workflows.',
        },
        {
            'solution_view': 'Clinical Research',
            'who_it_supports': 'Clinical researchers, outcomes analysts, and teams reviewing cohorts, risk, and pathways.',
            'best_fit_workflow': 'Clinical Outcome Analytics',
            'best_fit_package': 'Clinical Outcomes Review',
            'typical_outputs': 'Clinical report, cohort review, predictive/fairness summaries',
            'why_it_works': 'Best when the dataset has patient-level signals, outcomes, or pathway-style structure.',
        },
        {
            'solution_view': 'Healthcare Data Readiness',
            'who_it_supports': 'Data stewards, platform teams, and implementation analysts preparing healthcare data for wider use.',
            'best_fit_workflow': 'Healthcare Data Readiness',
            'best_fit_package': 'Healthcare Data Readiness',
            'typical_outputs': 'Readiness review, remediation plan, governance snapshot, analyst report',
            'why_it_works': 'Best when the immediate need is to understand blockers, mappings, and governance posture.',
        },
        {
            'solution_view': 'Population Health',
            'who_it_supports': 'Population health teams, preventive care programs, and quality leaders reviewing segment-level variation.',
            'best_fit_workflow': 'Population Health Intelligence',
            'best_fit_package': 'Population Health Review',
            'typical_outputs': 'Executive summary, benchmark review, cohort monitoring, intervention ideas',
            'why_it_works': 'Best when the dataset can support risk tiering, cohort monitoring, and disparity-aware segment analysis.',
        },
        {
            'solution_view': 'Analytics Consulting Teams',
            'who_it_supports': 'Consulting teams and analytics partners who need fast dataset triage plus stakeholder-ready deliverables.',
            'best_fit_workflow': 'Decision Support',
            'best_fit_package': 'Healthcare Data Readiness',
            'typical_outputs': 'Executive report pack, stakeholder export bundle, remediation and roadmap summaries',
            'why_it_works': 'Best when the goal is to assess an unfamiliar dataset quickly and package findings into a client-ready review.',
        },
    ]

    for item in views:
        package_name = item['best_fit_package']
        item['current_fit'] = package_status_lookup.get(package_name, 'Needs stronger source data')
        item['recommended_now'] = 'Yes' if (
            item['best_fit_package'] == recommended_package or item['best_fit_workflow'] == recommended_workflow
        ) else 'No'
        item['fit_rationale'] = (
            f"Recommended now because the dataset is currently best matched to {recommended_workflow.lower()}."
            if item['recommended_now'] == 'Yes'
            else f"Available as a secondary path if the team wants to extend beyond {recommended_package.lower()}."
        )
        details = package_details.get(package_name, {})
        item['relevant_modules'] = ', '.join(details.get('relevant_modules', [])[:4]) if details else 'Uses the existing cross-setting workflow modules.'

    recommended_view = next((item['solution_view'] for item in views if item['recommended_now'] == 'Yes'), views[0]['solution_view'])
    recommended_view_row = next((item for item in views if item['solution_view'] == recommended_view), views[0])
    narrative = (
        f"The platform can support multiple healthcare buyer types from the same workflow foundation. "
        f"For the current dataset, the strongest immediate fit is {recommended_view} via the "
        f"{recommended_view_row.get('best_fit_workflow', 'recommended workflow')} path, because "
        f"{rationale[:1].lower() + rationale[1:] if rationale else 'the detected structure aligns with that operating model.'}"
    )

    return {
        'market_solution_views': pd.DataFrame(views),
        'recommended_solution_view': recommended_view,
        'narrative': narrative,
    }
