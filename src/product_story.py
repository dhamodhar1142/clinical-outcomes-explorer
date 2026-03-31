from __future__ import annotations

import pandas as pd


PRODUCT_ONE_LINER = (
    'Clinverity helps healthcare teams turn raw clinical or operational datasets into '
    'trusted readiness assessments, actionable analytics, and stakeholder-ready reports.'
)

CORE_OUTCOMES = [
    {
        'outcome': 'Make dataset risk visible fast',
        'description': 'Surface data quality, readiness, trust, and reporting blockers before teams waste time on unreliable analysis.',
    },
    {
        'outcome': 'Turn healthcare data into decisions',
        'description': 'Translate encounter, cohort, trend, and remediation signals into clear next-step recommendations for operational and analytical teams.',
    },
    {
        'outcome': 'Package proof for stakeholders',
        'description': 'Generate executive, analyst, and governed export outputs without losing lineage, context, or disclosure.',
    },
]

DEMO_WALKTHROUGH = [
    {
        'step': '1. Load the demo or upload a file',
        'focus': 'Use the healthcare operations demo or a real CSV/XLSX extract.',
        'why_it_matters': 'Shows Clinverity can start from messy source data, not only pre-modeled dashboards.',
    },
    {
        'step': '2. Confirm dataset readiness',
        'focus': 'Review row counts, schema coverage, readiness, trust, and quality blockers.',
        'why_it_matters': 'Makes the value obvious in under a minute: Clinverity tells teams what the dataset can safely support right now.',
    },
    {
        'step': '3. Explore healthcare intelligence',
        'focus': 'Walk through cohort, trend, risk, and recommendations using the active dataset.',
        'why_it_matters': 'Shows that analytics stay tied to the uploaded dataset context and not a generic demo shell.',
    },
    {
        'step': '4. Export the stakeholder handoff',
        'focus': 'Generate an executive summary, operational review, or readiness report.',
        'why_it_matters': 'Closes the loop from raw file to usable decision artifact.',
    },
]

ROLE_PITCHES = {
    'Recruiter': {
        'audience': 'Recruiter / hiring manager',
        'pitch': (
            'Clinverity is a production-style healthcare data product that proves I can build more than dashboards: '
            'it handles uploaded datasets reliably, validates readiness, runs domain-aware analytics, and packages results into governed exports.'
        ),
    },
    'Hospital Operations': {
        'audience': 'Hospital operations leader',
        'pitch': (
            'Clinverity helps operations teams assess whether a hospital extract is trustworthy enough for action, '
            'then turns it into readiness signals, utilization insight, operational recommendations, and exportable summaries.'
        ),
    },
    'Healthcare Analytics Client': {
        'audience': 'Healthcare analytics client',
        'pitch': (
            'Clinverity shortens the path from raw healthcare data to decision-ready analysis by combining data quality, semantic mapping, '
            'trust scoring, cohort and trend insight, and stakeholder-ready reporting in one workflow.'
        ),
    },
}

DEMO_SCRIPT_STEPS = [
    'Open Clinverity and show that the product starts in a guided, demo-ready state.',
    'Load the healthcare operations demo or upload a healthcare extract and call out that the uploaded dataset remains authoritative.',
    'Use the Overview and Readiness surfaces to explain health score, readiness, trust, and key blockers.',
    'Move into Healthcare Intelligence, Trend Analysis, or Cohort Analysis to show domain-aware analytics tied to the active dataset.',
    'Close in Export Center by generating an executive summary or readiness review to prove the handoff workflow.',
]


def build_core_outcomes_table() -> pd.DataFrame:
    return pd.DataFrame(CORE_OUTCOMES)


def build_demo_walkthrough_table() -> pd.DataFrame:
    return pd.DataFrame(DEMO_WALKTHROUGH)


def build_role_pitch_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                'audience': details['audience'],
                'pitch': details['pitch'],
            }
            for details in ROLE_PITCHES.values()
        ]
    )


def build_demo_script_text() -> str:
    return '\n'.join(f'{index}. {step}' for index, step in enumerate(DEMO_SCRIPT_STEPS, start=1))


def build_professional_export_template(report_mode: str) -> dict[str, list[str] | str]:
    mode = str(report_mode or 'Executive Summary')
    sections = [
        '1. Audience and decision context',
        '2. Dataset scope and source posture',
        '3. Readiness, trust, and risk summary',
        '4. Key findings and implications',
        '5. Recommendations and next steps',
        '6. Disclosure, lineage, and reporting guardrails',
    ]
    if mode in {'Operational Report', 'Clinical Report', 'Population Health Summary'}:
        sections.insert(4, '5. Cohort, trend, or operational detail')
        sections[-1] = '7. Disclosure, lineage, and reporting guardrails'
    elif mode == 'Data Readiness Review':
        sections[3] = '4. Quality blockers and remediation priorities'
        sections[4] = '5. Recommended data upgrades'
    return {
        'report_mode': mode,
        'title': f'Clinverity professional {mode.lower()} template',
        'sections': sections,
        'principles': [
            'Lead with business or operational meaning, not raw metrics.',
            'State readiness and trust before sharing downstream insight.',
            'Keep recommendations specific, prioritized, and tied to detected issues.',
            'Always disclose synthetic, inferred, or directional support when relevant.',
        ],
    }
