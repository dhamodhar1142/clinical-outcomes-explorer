from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd


def _safe_df(table: Any) -> pd.DataFrame:
    return table if isinstance(table, pd.DataFrame) else pd.DataFrame()


def build_executive_report_pack(
    dataset_name: str,
    overview: dict[str, object],
    quality: dict[str, object],
    readiness: dict[str, object],
    healthcare: dict[str, object],
    executive_summary: dict[str, object],
    action_recommendations: pd.DataFrame,
    intervention_recommendations: pd.DataFrame,
    kpi_benchmarking: dict[str, object],
    scenario_studio: dict[str, object],
    prioritized_insights: dict[str, object],
    remediation_context: dict[str, object],
    demo_config: dict[str, object],
) -> dict[str, object]:
    active_modules = int(readiness.get('available_count', 0))
    blocked = _safe_df(readiness.get('readiness_table'))
    blocked_modules = blocked[blocked.get('status', pd.Series(dtype=str)).astype(str) == 'Unavailable']['analysis_module'].astype(str).tolist()
    fairness = healthcare.get('explainability_fairness', {})
    synthetic_notes: list[str] = []
    if remediation_context.get('synthetic_cost', {}).get('available'):
        synthetic_notes.append('Estimated cost is synthetic and demo-derived.')
    if remediation_context.get('synthetic_clinical', {}).get('available'):
        synthetic_notes.append('Diagnosis labels and risk labels are derived approximations.')
    if remediation_context.get('synthetic_readmission', {}).get('available'):
        synthetic_notes.append('Readmission support is synthetic/demo-derived.')
    if remediation_context.get('bmi_remediation', {}).get('available') and remediation_context.get('bmi_remediation', {}).get('total_bmi_outliers', 0):
        synthetic_notes.append(f"BMI values were remediated using {remediation_context['bmi_remediation'].get('remediation_mode', 'median')} mode.")

    sections = {
        'Report Title': f'Smart Dataset Analyzer Executive Report Pack — {dataset_name}',
        'Dataset Overview': f"{int(overview.get('rows', 0)):,} rows, {int(overview.get('columns', 0)):,} columns, and {int(overview.get('duplicate_rows', 0)):,} duplicate rows were reviewed.",
        'Data Quality Snapshot': f"Data quality score is {float(quality.get('quality_score', 0.0)):.1f} with {int(overview.get('missing_values', 0)):,} missing values in scope.",
        'Analysis Readiness Snapshot': f"Readiness is {float(readiness.get('readiness_score', 0.0)):.0%} with {active_modules} active modules.",
        'Healthcare Capability Snapshot': f"Healthcare readiness is {float(healthcare.get('healthcare_readiness_score', 0.0)):.0%} for {healthcare.get('likely_dataset_type', 'the current dataset')}.",
        'Active Modules': f'{active_modules} analytics modules are currently active.',
        'Key Findings': '; '.join(executive_summary.get('stakeholder_summary_bullets', [])[:3]) or 'Key findings are limited for the current dataset.',
        'Priority Recommendations': '; '.join(intervention_recommendations.head(3)['recommendation_title'].astype(str).tolist()) if not intervention_recommendations.empty else '; '.join(action_recommendations.head(3)['recommendation_title'].astype(str).tolist()),
        'KPI Summary': '; '.join(f"{card['label']}: {card['value']}" for card in kpi_benchmarking.get('kpi_cards', [])[:4]) if kpi_benchmarking.get('available') else 'No internal KPI benchmark set is available.',
        'Scenario Highlights': scenario_studio.get('summary', 'No scenario highlights are available.') if scenario_studio.get('available') else 'Scenario simulation is not available for the current dataset.',
        'Fairness Review Summary': fairness.get('high_risk_segment_explanation', fairness.get('reason', 'Fairness review is limited for the current dataset.')),
        'Synthetic Support Disclosures': '; '.join(synthetic_notes) if synthetic_notes else 'No synthetic helper fields are currently required.',
        'Remaining Blockers': ', '.join(blocked_modules[:6]) if blocked_modules else 'No major blockers remain for the active module set.',
        'Next-Step Roadmap': '; '.join(intervention_recommendations.head(3)['recommendation_title'].astype(str).tolist()) if not intervention_recommendations.empty else 'Continue with quality review, benchmarking, and stakeholder export preparation.',
    }
    markdown = '\n\n'.join(f"## {title}\n{body}" for title, body in sections.items())
    text = '\n'.join(f"{title}: {body}" for title, body in sections.items())
    return {
        'executive_report_pack': sections,
        'executive_report_sections': sections,
        'executive_report_markdown': markdown,
        'executive_report_text': text,
    }


def build_printable_reports(
    executive_report_pack: dict[str, object],
    compliance_governance_summary: dict[str, object],
) -> dict[str, object]:
    exec_sections = executive_report_pack.get('executive_report_sections', {})
    printable_exec = '\n\n'.join(f"# {k}\n{v}" for k, v in exec_sections.items())
    compliance_sections = compliance_governance_summary.get('sections', {})
    printable_compliance = '\n\n'.join(f"# {k}\n{v}" for k, v in compliance_sections.items())
    return {
        'printable_executive_report': printable_exec,
        'printable_compliance_summary': printable_compliance,
    }


def build_stakeholder_export_bundle(
    executive_report_pack: dict[str, object],
    kpi_benchmarking: dict[str, object],
    intervention_recommendations: pd.DataFrame,
    fairness_review: dict[str, object],
    readmission: dict[str, object],
    quality: dict[str, object],
    compliance_governance_summary: dict[str, object],
) -> dict[str, object]:
    manifest_rows = [
        {'bundle_item': 'Executive Summary', 'status': 'Ready', 'note': 'Uses the current executive report pack.'},
        {'bundle_item': 'KPI Summary', 'status': 'Ready' if kpi_benchmarking.get('available') else 'Limited', 'note': 'Internal dataset-relative benchmarks only.'},
        {'bundle_item': 'Recommendation Summary', 'status': 'Ready' if not intervention_recommendations.empty else 'Limited', 'note': 'Priority recommendations are deterministic and stakeholder-friendly.'},
        {'bundle_item': 'Fairness Snapshot', 'status': 'Ready' if fairness_review.get('available') else 'Limited', 'note': 'Transparent subgroup comparison, not a formal bias audit.'},
        {'bundle_item': 'Readmission Summary', 'status': 'Ready' if readmission.get('available') else 'Limited', 'note': 'May rely on synthetic support when native fields are unavailable.'},
        {'bundle_item': 'Data Quality & Remediation Summary', 'status': 'Ready', 'note': 'Uses current quality diagnostics and remediation context.'},
        {'bundle_item': 'Compliance / Governance Snapshot', 'status': 'Ready', 'note': 'Readiness and governance support, not legal certification.'},
    ]
    notes = [
        'Bundle is assembled from current pipeline outputs without recomputing analysis.',
        'Synthetic and inferred support is disclosed where relevant.',
    ]
    return {
        'stakeholder_export_bundle': {
            'executive_report': executive_report_pack.get('executive_report_text', ''),
            'kpi_summary': kpi_benchmarking.get('benchmark_table', pd.DataFrame()),
            'recommendations': intervention_recommendations,
            'fairness_snapshot': fairness_review.get('comparison_table', pd.DataFrame()),
            'readmission_summary': _safe_df(readmission.get('high_risk_segments')),
            'quality_summary': _safe_df(quality.get('high_missing')),
            'compliance_governance': compliance_governance_summary.get('summary_table', pd.DataFrame()),
        },
        'export_bundle_manifest': pd.DataFrame(manifest_rows),
        'export_bundle_notes': notes,
    }


def build_run_history_entry(
    dataset_name: str,
    pipeline: dict[str, object],
    demo_config: dict[str, object],
    model_comparison: dict[str, object] | None = None,
) -> dict[str, object]:
    remediation = pipeline.get('remediation_context', {})
    helper_fields = _safe_df(remediation.get('helper_fields'))
    readiness = pipeline.get('readiness', {})
    readiness_table = _safe_df(readiness.get('readiness_table'))
    entry = {
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'dataset_name': dataset_name,
        'row_count': int(pipeline['overview'].get('rows', 0)),
        'column_count': int(pipeline['overview'].get('columns', 0)),
        'active_native_fields': int(remediation.get('native_field_count', 0)),
        'synthetic_helper_fields_added': int(remediation.get('synthetic_field_count', 0)),
        'derived_fields_added': int(remediation.get('derived_field_count', 0)),
        'remediation_actions_performed': ', '.join(helper_fields.get('helper_field', pd.Series(dtype=str)).astype(str).tolist()[:6]),
        'models_compared': int(len(_safe_df(model_comparison.get('model_comparison_table')))) if model_comparison else 0,
        'best_model_selected': str(model_comparison.get('best_model_name', '')) if model_comparison else '',
        'fairness_review_mode': str(model_comparison.get('fairness_review_mode', 'not_run')) if model_comparison else 'not_run',
        'scenario_set_executed': str(demo_config.get('scenario_simulation_mode', 'basic')),
        'major_blockers_detected': ', '.join(readiness_table.loc[lambda df: df.get('status', pd.Series(dtype=str)) == 'Unavailable', 'analysis_module'].astype(str).tolist()[:5]) if not readiness_table.empty else '',
        'config_signature': f"{demo_config.get('synthetic_helper_mode')}|{demo_config.get('bmi_remediation_mode')}|{demo_config.get('executive_summary_verbosity')}",
    }
    return entry


def update_run_history(existing: list[dict[str, object]], entry: dict[str, object]) -> list[dict[str, object]]:
    history = list(existing or [])
    signature = (entry.get('dataset_name'), entry.get('row_count'), entry.get('column_count'), entry.get('config_signature'))
    if history:
        last = history[-1]
        last_sig = (last.get('dataset_name'), last.get('row_count'), last.get('column_count'), last.get('config_signature'))
        if last_sig == signature:
            history[-1] = entry
            return history
    history.append(entry)
    return history[-12:]


def build_audit_summary(run_history: list[dict[str, object]], analysis_log: list[dict[str, object]]) -> dict[str, object]:
    history_df = pd.DataFrame(run_history)
    log_df = pd.DataFrame(analysis_log)
    summary_table = pd.DataFrame([
        {'metric': 'Runs this session', 'value': f"{int(len(history_df)):,}"},
        {'metric': 'Audit events logged', 'value': f"{int(len(log_df)):,}"},
        {'metric': 'Most recent dataset', 'value': str(history_df.iloc[-1]['dataset_name']) if not history_df.empty else 'None'},
    ])
    text = f"Current session includes {len(history_df)} tracked analysis runs and {len(log_df)} audit events."
    return {
        'run_history_entry': history_df.iloc[-1].to_dict() if not history_df.empty else {},
        'audit_summary': summary_table,
        'audit_summary_text': text,
    }


def build_compliance_governance_summary(
    standards: dict[str, object],
    privacy_review: dict[str, object],
    lineage: dict[str, object],
    remediation_context: dict[str, object],
    readiness: dict[str, object],
) -> dict[str, object]:
    direct_identifiers = int(privacy_review.get('hipaa', {}).get('direct_identifier_count', 0))
    synthetic_count = int(remediation_context.get('synthetic_field_count', 0))
    sections = {
        'Standards Readiness': f"{float(standards.get('combined_readiness_score', 0.0)):.1f}/100 with badge {standards.get('badge_text', 'Not assessed')}.",
        'Privacy Posture': f"HIPAA-style risk is {privacy_review.get('hipaa', {}).get('risk_level', 'Low')} with {direct_identifiers} direct identifiers detected.",
        'Governance Depth': f"{len(lineage.get('transformation_steps', []))} transformation steps and {synthetic_count} synthetic helper fields are currently tracked.",
        'Disclosure Summary': 'Synthetic and inferred support is disclosed in readiness, export, and lineage views.' if synthetic_count else 'Current workflow does not rely on synthetic helper fields.',
    }
    cards = [
        {'label': 'Standards', 'value': standards.get('badge_text', 'Not assessed')},
        {'label': 'Privacy Risk', 'value': privacy_review.get('hipaa', {}).get('risk_level', 'Low')},
        {'label': 'Lineage Steps', 'value': str(len(lineage.get('transformation_steps', [])))},
        {'label': 'Synthetic Fields', 'value': str(synthetic_count)},
    ]
    notes = [
        'This is a readiness and governance support layer, not a formal compliance certification.',
    ]
    if synthetic_count:
        notes.append('Synthetic helper fields are improving readiness but should not be treated as native source-grade fields.')
    return {
        'compliance_governance_summary': sections,
        'compliance_snapshot_cards': cards,
        'governance_notes': notes,
        'disclosure_summary': sections['Disclosure Summary'],
        'sections': sections,
        'summary_table': pd.DataFrame([{'focus_area': key, 'status': value} for key, value in sections.items()]),
    }


def build_landing_summary(
    pipeline: dict[str, object],
    demo_config: dict[str, object],
    dataset_name: str,
) -> dict[str, object]:
    readiness = pipeline['readiness']
    remediation = pipeline.get('remediation_context', {})
    synthetic_count = int(remediation.get('synthetic_field_count', 0))
    capability_badges = [
        {'label': 'Healthcare Analytics', 'value': f"{float(pipeline['healthcare'].get('healthcare_readiness_score', 0.0)):.0%} ready"},
        {'label': 'Compliance & Governance', 'value': pipeline['standards'].get('badge_text', 'In review')},
        {'label': 'Modeling', 'value': 'Enabled' if readiness.get('available_count', 0) else 'Selective'},
        {'label': 'Exports', 'value': 'Stakeholder-ready'},
    ]
    covers = [
        'Quality review and remediation guidance',
        'Healthcare intelligence, risk, and cohort analytics',
        'Standards, privacy, and governance readiness',
        'Stakeholder reporting and export preparation',
    ]
    system_status = [
        f"Dataset: {dataset_name}",
        f"Readiness: {float(readiness.get('readiness_score', 0.0)):.0%}",
        f"Active modules: {int(readiness.get('available_count', 0))}",
    ]
    value_summary = 'Designed for students, analysts, and healthcare teams who need a polished analytics workflow with transparent readiness, governance, and decision-support layers.'
    synthetic_note = 'Synthetic helper fields are enabled in a transparent demo-safe mode.' if synthetic_count else 'Current analysis relies primarily on native fields.'
    return {
        'headline': 'Enterprise-ready healthcare analytics for messy real-world datasets',
        'subheadline': 'Profile, remediate, benchmark, and explain healthcare data with transparent standards, governance, and decision-support workflows.',
        'capability_badges': capability_badges,
        'analysis_covers': covers,
        'system_status': system_status,
        'platform_value_summary': value_summary,
        'synthetic_support_note': synthetic_note,
    }
