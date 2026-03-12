from __future__ import annotations

from typing import Any

import pandas as pd


def _safe_df(table: Any) -> pd.DataFrame:
    return table if isinstance(table, pd.DataFrame) else pd.DataFrame()


def build_demo_mode_content(
    dataset_name: str,
    source_meta: dict[str, str],
    pipeline: dict[str, object],
    demo_config: dict[str, object],
) -> dict[str, object]:
    remediation = pipeline.get('remediation_context', {})
    synthetic_count = int(remediation.get('synthetic_field_count', 0))
    readiness = pipeline.get('readiness', {})
    demo_flow = [
        'Start in Dataset Profile · Overview to understand scale, readiness, and platform coverage.',
        'Move to Data Quality · Analysis Readiness to review blockers, remediation opportunities, and standards/privacy posture.',
        'Use Healthcare Analytics · Healthcare Intelligence to review risk, readmission, pathway, and cohort findings.',
        'Finish in Insights & Export · Export Center to generate stakeholder-facing summaries and governance packs.',
    ]
    insights_to_look_for = [
        'Where native source fields are strong enough to support healthcare analytics directly.',
        'Which synthetic helper fields were added to unlock demo-safe analytics workflows.',
        'Which recommendations, benchmarks, and scenarios matter most for the current dataset.',
    ]
    checklist = [
        'Review readiness and quality before presenting advanced findings.',
        'Check whether synthetic support is active and disclose it when relevant.',
        'Capture one overview screen, one healthcare analytics screen, and one export/report screen.',
    ]
    support_note = (
        'Synthetic helper fields are enabled in a transparent demo-safe mode for this walkthrough.'
        if synthetic_count
        else 'This walkthrough relies primarily on native source fields.'
    )
    mode = 'Guided demo mode is most useful on the built-in examples.' if source_meta.get('source_mode') == 'Demo dataset' else 'Guided demo mode is also useful on uploaded datasets once readiness has been reviewed.'
    return {
        'title': f'How to explore {dataset_name}',
        'intro': f"{mode} The current dataset is {float(readiness.get('readiness_score', 0.0)):.0%} ready with {int(readiness.get('available_count', 0))} active modules.",
        'recommended_flow': demo_flow,
        'insights_to_look_for': insights_to_look_for,
        'demo_highlights_checklist': checklist,
        'synthetic_support_note': support_note,
        'scenario_mode': str(demo_config.get('scenario_simulation_mode', 'Basic')),
    }


def build_dataset_onboarding_summary(
    dataset_name: str,
    source_meta: dict[str, str],
    pipeline: dict[str, object],
) -> dict[str, object]:
    readiness = pipeline.get('readiness', {})
    remediation = _safe_df(pipeline.get('remediation'))
    helper_fields = _safe_df(pipeline.get('remediation_context', {}).get('helper_fields'))
    unlock_guide = _safe_df(readiness.get('readiness_table'))
    available = unlock_guide[unlock_guide.get('status', pd.Series(dtype=str)).astype(str) == 'Available']['analysis_module'].astype(str).tolist()
    blocked = unlock_guide[unlock_guide.get('status', pd.Series(dtype=str)).astype(str) == 'Unavailable']['analysis_module'].astype(str).tolist()
    top_fix = remediation.iloc[0]['recommended_fix'] if not remediation.empty and 'recommended_fix' in remediation.columns else 'Review the top readiness blockers and complete the suggested field mapping/remediation steps.'
    synthetic_note = (
        f"{len(helper_fields)} helper fields were added to unlock demo-safe analytics."
        if not helper_fields.empty
        else 'No synthetic helper fields were needed for the current dataset.'
    )
    upgrade_suggestions = remediation.head(3)[['issue', 'recommended_fix', 'modules_unlocked']] if not remediation.empty else pd.DataFrame()
    return {
        'dataset_onboarding_summary': {
            'dataset_name': dataset_name,
            'suitability': source_meta.get('best_for', 'General schema-flexible analytics and healthcare readiness review.'),
            'readiness_score': readiness.get('readiness_score', 0.0),
            'top_fix': top_fix,
            'synthetic_note': synthetic_note,
        },
        'module_unlock_guide': unlock_guide[['analysis_module', 'status', 'missing_prerequisites']] if not unlock_guide.empty else pd.DataFrame(),
        'data_upgrade_suggestions': upgrade_suggestions,
        'available_modules': available,
        'blocked_modules': blocked,
    }


def build_documentation_support(
    dataset_name: str,
    pipeline: dict[str, object],
) -> dict[str, object]:
    readiness = pipeline.get('readiness', {})
    healthcare = pipeline.get('healthcare', {})
    remediation = pipeline.get('remediation_context', {})
    helper_count = int(remediation.get('synthetic_field_count', 0))
    feature_list = [
        'Dataset profiling, quality review, and semantic mapping',
        'Healthcare standards, privacy, and governance readiness',
        'Cohort, risk, readmission, and pathway analytics',
        'Predictive modeling, fairness review, and intervention recommendations',
        'Stakeholder-ready exports, governance packs, and executive summaries',
    ]
    architecture = [
        'Streamlit UI entrypoint in app.py',
        'Modular analytics, remediation, standards, and export services under src/',
        'Deterministic helper-field augmentation for demo-safe workflow unlocks',
        'Session-scoped governance, run history, and stakeholder export support',
    ]
    demo_flow = [
        'Open the overview to understand dataset shape, readiness, and capability coverage.',
        'Review Analysis Readiness and Data Quality to understand blockers and remediation steps.',
        'Use Healthcare Intelligence and Key Insights for the core decision-support narrative.',
        'Finish in Export Center for executive, governance, and portfolio-ready outputs.',
    ]
    limitations = [
        'Synthetic helper fields support demo-safe analytics but are not substitutes for source-grade clinical data.',
        'Compliance and privacy outputs are readiness aids, not legal or certification determinations.',
        'The platform is optimized for explainable analytics and walkthroughs rather than production orchestration.',
    ]
    sections = {
        'project_overview': f'Smart Dataset Analyzer is a schema-flexible healthcare analytics platform for profiling, readiness review, and decision-support across clinical, operational, and governance workflows. The current dataset {dataset_name} is {float(readiness.get("readiness_score", 0.0)):.0%} ready with {int(readiness.get("available_count", 0))} active modules.',
        'feature_list': feature_list,
        'architecture_summary': architecture,
        'demo_flow': demo_flow,
        'key_modules': available_modules_text(readiness),
        'synthetic_support_explanation': f'The platform currently tracks {helper_count} synthetic helper fields transparently in lineage, exports, and readiness summaries.' if helper_count else 'The current dataset does not rely on synthetic helper fields.',
        'limitations': limitations,
        'future_enhancements': [
            'Deeper collaboration and workflow automation',
            'Broader standards coverage and operational integrations',
            'More advanced predictive and benchmarking workflows',
        ],
    }
    portfolio_summary = f'Smart Dataset Analyzer demonstrates end-to-end healthcare analytics product thinking: profiling, remediation, standards review, predictive intelligence, governance, and stakeholder-ready reporting in a single Streamlit platform.'
    technical_summary = f'The platform uses modular Python services for profiling, remediation, readiness scoring, healthcare analytics, decision support, and exports, with Streamlit as the presentation layer and deterministic helper-field augmentation where source data is incomplete.'
    walkthrough_text = '\n'.join(f'- {step}' for step in demo_flow)
    return {
        'readme_sections': sections,
        'portfolio_project_summary': portfolio_summary,
        'technical_project_summary': technical_summary,
        'demo_walkthrough_text': walkthrough_text,
    }


def available_modules_text(readiness: dict[str, object]) -> list[str]:
    table = _safe_df(readiness.get('readiness_table'))
    if table.empty or 'analysis_module' not in table.columns:
        return ['Analysis modules depend on the current dataset structure and readiness state.']
    return table[table.get('status', pd.Series(dtype=str)).astype(str) == 'Available']['analysis_module'].astype(str).head(8).tolist()


def build_screenshot_support(
    dataset_name: str,
    pipeline: dict[str, object],
) -> dict[str, object]:
    readiness = pipeline.get('readiness', {})
    synthetic_count = int(pipeline.get('remediation_context', {}).get('synthetic_field_count', 0))
    plan_rows = [
        {'screen': 'Overview', 'demonstrates': 'Platform framing, readiness, and governance posture', 'caption': f'Overview of {dataset_name} with readiness, governance, and stakeholder-facing analytics context.'},
        {'screen': 'Healthcare Intelligence', 'demonstrates': 'Risk, cohort, readmission, and pathway analytics', 'caption': 'Healthcare Intelligence showing explainable risk, readmission, and pathway-focused decision support.'},
        {'screen': 'Export Center', 'demonstrates': 'Executive and governance reporting outputs', 'caption': 'Export Center with executive report packs, governance outputs, and stakeholder bundles.'},
    ]
    captions = [row['caption'] for row in plan_rows]
    callouts = [
        f'The platform currently supports {int(readiness.get("available_count", 0))} active modules with a readiness score of {float(readiness.get("readiness_score", 0.0)):.0%}.',
        'Synthetic support is tracked transparently in lineage and executive summaries.' if synthetic_count else 'Native source fields are strong enough to support the core workflow without synthetic augmentation.',
        'The app combines analytics, governance, and stakeholder-ready exports in one interface.',
    ]
    return {
        'screenshot_plan': pd.DataFrame(plan_rows),
        'screenshot_captions': captions,
        'recruiter_callouts': callouts,
    }


def build_app_metadata(pipeline: dict[str, object]) -> dict[str, object]:
    helper_count = int(pipeline.get('remediation_context', {}).get('synthetic_field_count', 0))
    readiness = pipeline.get('readiness', {})
    metadata = {
        'product_name': 'Smart Dataset Analyzer',
        'tagline': 'Healthcare analytics, readiness review, and stakeholder reporting for real-world tabular datasets.',
        'best_for': 'Students, analysts, healthcare operators, research teams, and portfolio walkthroughs.',
        'best_data': 'Operational healthcare, claims-like, clinical, and general tabular datasets with partial or complete schema support.',
        'synthetic_support': 'Synthetic helper fields are used only to unlock demo-safe analytics when native fields are missing.' if helper_count else 'The current workflow relies primarily on native source fields.',
        'maturity': f"Portfolio-ready analytics platform with {int(readiness.get('available_count', 0))} active modules and transparent governance support.",
        'version': 'Demo Release 1.0',
    }
    return metadata
