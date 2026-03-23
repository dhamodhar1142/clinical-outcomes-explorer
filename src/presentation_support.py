from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd

from src.data_loader import DEMO_DATASETS


def _safe_df(table: Any) -> pd.DataFrame:
    return table if isinstance(table, pd.DataFrame) else pd.DataFrame()


def _build_dataset_specific_differentiators(
    dataset_type_label: str,
    recommended_workflow: str,
    healthcare_readiness_score: float,
) -> pd.DataFrame:
    normalized_type = str(dataset_type_label).lower()
    normalized_workflow = str(recommended_workflow).lower()
    is_healthcare = 'healthcare' in normalized_type or 'clinical' in normalized_type or 'claims' in normalized_type or 'encounter' in normalized_type
    is_business = 'generic' in normalized_type or 'tabular' in normalized_type or 'business' in normalized_type

    common_rows = [
        {
            "title": "Dataset Intelligence",
            "summary": f"Classifies the current dataset as {dataset_type_label} and adapts the workflow guidance automatically.",
            "why_it_matters": "The product updates its value story when dataset characteristics change instead of forcing every file through the same pitch.",
        },
        {
            "title": "Readiness and Remediation",
            "summary": "Turns schema gaps, quality blockers, and field-mapping issues into specific next-step improvements.",
            "why_it_matters": "Teams can move from unfamiliar raw files to usable analytics faster instead of losing time in manual triage.",
        },
        {
            "title": "Decision Support and Reporting",
            "summary": "Packages analysis into stakeholder-ready guidance, reports, and export flows without making users rebuild the narrative manually.",
            "why_it_matters": "That shortens the path from analysis to a usable business or operational handoff.",
        },
    ]

    if is_healthcare or healthcare_readiness_score >= 0.35:
        domain_row = {
            "title": "Healthcare-Aware Advantage",
            "summary": "Combines readiness, healthcare analytics, governance, and reporting in one workflow for operations, outcomes, cohort, and readmission review.",
            "why_it_matters": "Healthcare teams get domain-aware guidance and transparent native versus synthetic support instead of generic BI dashboards with no data-readiness layer.",
        }
        competitive_row = {
            "title": "Why it stands out for healthcare teams",
            "summary": f"Recommended workflow: {recommended_workflow}. The platform links dataset onboarding directly to cohort, risk, pathway, governance, and stakeholder-reporting workflows.",
            "why_it_matters": "That gives healthcare analysts and pilot teams one product surface for dataset triage, analytics, and exportable next steps.",
        }
    elif is_business or 'operations' in normalized_workflow or 'readiness' in normalized_workflow:
        domain_row = {
            "title": "Business Dataset Advantage",
            "summary": "Adapts the same product shell for general business and operations data by emphasizing data readiness, profiling, quality, trend review, and stakeholder reporting.",
            "why_it_matters": "Business users still get clear onboarding, issue discovery, and export-ready summaries even when healthcare-specific modules are not the main story.",
        }
        competitive_row = {
            "title": "Why it stands out for business workflows",
            "summary": f"Recommended workflow: {recommended_workflow}. The product combines dataset intelligence, remediation guidance, trend analysis, and reporting instead of acting like a charts-only layer.",
            "why_it_matters": "That makes it more useful than a generic dashboard shell when teams first need to understand whether the dataset is trustworthy and reusable.",
        }
    else:
        domain_row = {
            "title": "Adaptive Workflow Advantage",
            "summary": f"Recommended workflow: {recommended_workflow}. The platform shifts emphasis based on the current dataset instead of assuming one fixed use case.",
            "why_it_matters": "Users get a clearer story for mixed or partially classified files while the product stays transparent about what is and is not supported yet.",
        }
        competitive_row = {
            "title": "Why it stands out across dataset types",
            "summary": "Brings together classification, readiness, remediation, analytics, and reporting in one guided flow.",
            "why_it_matters": "That helps both healthcare and non-healthcare users get to a credible next step faster.",
        }

    return pd.DataFrame(common_rows[:2] + [domain_row] + [competitive_row] + common_rows[2:])


def build_demo_dataset_cards() -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for name, meta in DEMO_DATASETS.items():
        rows.append(
            {
                "dataset_name": str(name),
                "description": str(meta.get("description", "")),
                "best_for": str(meta.get("best_for", "")),
            }
        )
    return pd.DataFrame(rows)


def build_executive_report_pack(
    dataset_name: str,
    overview: dict[str, object],
    quality: dict[str, object],
    readiness: dict[str, object],
    healthcare: dict[str, object],
    dataset_intelligence: dict[str, object],
    executive_summary: dict[str, object],
    action_recommendations: pd.DataFrame,
    intervention_recommendations: pd.DataFrame,
    kpi_benchmarking: dict[str, object],
    scenario_studio: dict[str, object],
    prioritized_insights: dict[str, object],
    remediation_context: dict[str, object],
    demo_config: dict[str, object],
) -> dict[str, object]:
    active_modules = int(readiness.get("available_count", 0))
    analyzed_columns = int(overview.get("analyzed_columns", overview.get("columns", 0)))
    source_columns = int(overview.get("source_columns", analyzed_columns))
    helper_columns_added = int(overview.get("helper_columns_added", max(analyzed_columns - source_columns, 0)))
    blocked = _safe_df(readiness.get("readiness_table"))
    blocked_modules = (
        blocked[blocked.get("status", pd.Series(dtype=str)).astype(str) == "Unavailable"]["analysis_module"]
        .astype(str)
        .tolist()
    )
    fairness = healthcare.get("explainability_fairness", {})
    synthetic_notes: list[str] = []
    if remediation_context.get("synthetic_cost", {}).get("available"):
        synthetic_notes.append("Estimated cost is synthetic and demo-derived.")
    if remediation_context.get("synthetic_clinical", {}).get("available"):
        synthetic_notes.append("Diagnosis labels and risk labels are derived approximations.")
    if remediation_context.get("synthetic_readmission", {}).get("available"):
        synthetic_notes.append("Readmission support is synthetic/demo-derived.")
    if (
        remediation_context.get("bmi_remediation", {}).get("available")
        and remediation_context.get("bmi_remediation", {}).get("total_bmi_outliers", 0)
    ):
        synthetic_notes.append(
            f"BMI values were remediated using {remediation_context['bmi_remediation'].get('remediation_mode', 'median')} mode."
        )

    sections = {
        "Report Title": f"Clinverity Executive Report Pack - {dataset_name}",
        "Dataset Overview": f"{int(overview.get('rows', 0)):,} rows, {analyzed_columns:,} analyzed columns, and {int(overview.get('duplicate_rows', 0)):,} duplicate rows were reviewed.",
        "Dataset Intelligence Summary": (
            f"{dataset_intelligence.get('dataset_type_label', 'Dataset type not classified')} with "
            f"{int(len(dataset_intelligence.get('enabled_analytics', []))):,} enabled analytics modules and "
            f"{int(len(dataset_intelligence.get('blocked_analytics', []))):,} blocked modules."
        ),
        "Data Quality Snapshot": f"Data quality score is {float(quality.get('quality_score', 0.0)):.1f} with {int(overview.get('missing_values', 0)):,} missing values in scope.",
        "Analysis Readiness Snapshot": f"Readiness is {float(readiness.get('readiness_score', 0.0)):.0%} with {active_modules} active modules.",
        "Healthcare Capability Snapshot": f"Healthcare readiness is {float(healthcare.get('healthcare_readiness_score', 0.0)):.0%} for {healthcare.get('likely_dataset_type', 'the current dataset')}.",
        "Analytics Capability Overview": "; ".join(dataset_intelligence.get("enabled_analytics", [])[:4]) or "Capability coverage is still developing for the current dataset.",
        "Active Modules": f"{active_modules} analytics modules are currently active.",
        "Key Findings": "; ".join(executive_summary.get("stakeholder_summary_bullets", [])[:3]) or "Key findings are limited for the current dataset.",
        "Result Trust": str(
            executive_summary.get("executive_summary_sections", {}).get(
                "Result Trust",
                "Result trust disclosures are not available for the current dataset.",
            )
        ),
        "Priority Recommendations": "; ".join(intervention_recommendations.head(3)["recommendation_title"].astype(str).tolist()) if not intervention_recommendations.empty else "; ".join(action_recommendations.head(3)["recommendation_title"].astype(str).tolist()),
        "KPI Summary": "; ".join(f"{card['label']}: {card['value']}" for card in kpi_benchmarking.get("kpi_cards", [])[:4]) if kpi_benchmarking.get("available") else "No internal KPI benchmark set is available.",
        "Scenario Highlights": scenario_studio.get("summary", "No scenario highlights are available.") if scenario_studio.get("available") else "Scenario simulation is not available for the current dataset.",
        "Fairness Review Summary": fairness.get("high_risk_segment_explanation", fairness.get("reason", "Fairness review is limited for the current dataset.")),
        "Synthetic Support Disclosures": "; ".join(synthetic_notes + ([f'The analyzed dataset includes {helper_columns_added} helper or derived columns on top of {source_columns:,} source columns.'] if helper_columns_added > 0 else [])) if synthetic_notes or helper_columns_added > 0 else "No synthetic helper fields are currently required.",
        "Remaining Blockers": ", ".join(blocked_modules[:6]) if blocked_modules else "No major blockers remain for the active module set.",
        "Data Upgrade Recommendations": "; ".join(dataset_intelligence.get("next_best_actions", [])[:3]) or "Continue strengthening source-grade timestamps, outcomes, and coded clinical fields where possible.",
        "Next-Step Roadmap": "; ".join(intervention_recommendations.head(3)["recommendation_title"].astype(str).tolist()) if not intervention_recommendations.empty else "Continue with quality review, benchmarking, and stakeholder export preparation.",
    }
    markdown = "\n\n".join(f"## {title}\n{body}" for title, body in sections.items())
    text = "\n".join(f"{title}: {body}" for title, body in sections.items())
    return {
        "executive_report_pack": sections,
        "executive_report_sections": sections,
        "executive_report_markdown": markdown,
        "executive_report_text": text,
    }


def build_printable_reports(
    executive_report_pack: dict[str, object],
    compliance_governance_summary: dict[str, object],
) -> dict[str, object]:
    exec_sections = executive_report_pack.get("executive_report_sections", {})
    printable_exec = "\n\n".join(f"# {k}\n{v}" for k, v in exec_sections.items())
    compliance_sections = compliance_governance_summary.get("sections", {})
    printable_compliance = "\n\n".join(f"# {k}\n{v}" for k, v in compliance_sections.items())
    return {
        "printable_executive_report": printable_exec,
        "printable_compliance_summary": printable_compliance,
    }


def build_stakeholder_export_bundle(
    executive_report_pack: dict[str, object],
    dataset_intelligence: dict[str, object],
    kpi_benchmarking: dict[str, object],
    intervention_recommendations: pd.DataFrame,
    fairness_review: dict[str, object],
    readmission: dict[str, object],
    quality: dict[str, object],
    compliance_governance_summary: dict[str, object],
) -> dict[str, object]:
    manifest_rows = [
        {"bundle_item": "Executive Summary", "status": "Ready", "note": "Uses the current executive report pack."},
        {"bundle_item": "KPI Summary", "status": "Ready" if kpi_benchmarking.get("available") else "Limited", "note": "Internal dataset-relative benchmarks only."},
        {"bundle_item": "Recommendation Summary", "status": "Ready" if not intervention_recommendations.empty else "Limited", "note": "Priority recommendations are deterministic and stakeholder-friendly."},
        {"bundle_item": "Fairness Snapshot", "status": "Ready" if fairness_review.get("available") else "Limited", "note": "Transparent subgroup comparison, not a formal bias audit."},
        {"bundle_item": "Readmission Summary", "status": "Ready" if readmission.get("available") else "Limited", "note": "May rely on synthetic support when native fields are unavailable."},
        {"bundle_item": "Dataset Intelligence Summary", "status": "Ready", "note": "Summarizes dataset type, capability coverage, blockers, and highest-impact upgrades."},
        {"bundle_item": "Data Quality & Remediation Summary", "status": "Ready", "note": "Uses current quality diagnostics and remediation context."},
        {"bundle_item": "Compliance / Governance Snapshot", "status": "Ready", "note": "Readiness and governance support, not legal certification."},
    ]
    notes = [
        "Bundle is assembled from current pipeline outputs without recomputing analysis.",
        "Synthetic and inferred support is disclosed where relevant.",
    ]
    return {
        "stakeholder_export_bundle": {
            "executive_report": executive_report_pack.get("executive_report_text", ""),
            "kpi_summary": kpi_benchmarking.get("benchmark_table", pd.DataFrame()),
            "recommendations": intervention_recommendations,
            "fairness_snapshot": fairness_review.get("comparison_table", pd.DataFrame()),
            "readmission_summary": _safe_df(readmission.get("high_risk_segments")),
            "dataset_intelligence_summary": _safe_df(dataset_intelligence.get("analytics_capability_matrix")),
            "quality_summary": _safe_df(quality.get("high_missing")),
            "compliance_governance": compliance_governance_summary.get("summary_table", pd.DataFrame()),
        },
        "export_bundle_manifest": pd.DataFrame(manifest_rows),
        "export_bundle_notes": notes,
    }


def build_run_history_entry(
    dataset_name: str,
    pipeline: dict[str, object],
    demo_config: dict[str, object],
    model_comparison: dict[str, object] | None = None,
) -> dict[str, object]:
    remediation = pipeline.get("remediation_context", {})
    helper_fields = _safe_df(remediation.get("helper_fields"))
    readiness = pipeline.get("readiness", {})
    readiness_table = _safe_df(readiness.get("readiness_table"))
    entry = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "dataset_name": dataset_name,
        "row_count": int(pipeline["overview"].get("rows", 0)),
        "column_count": int(pipeline["overview"].get("analyzed_columns", pipeline["overview"].get("columns", 0))),
        "active_native_fields": int(remediation.get("native_field_count", 0)),
        "synthetic_helper_fields_added": int(remediation.get("synthetic_field_count", 0)),
        "derived_fields_added": int(remediation.get("derived_field_count", 0)),
        "remediation_actions_performed": ", ".join(helper_fields.get("helper_field", pd.Series(dtype=str)).astype(str).tolist()[:6]),
        "models_compared": int(len(_safe_df(model_comparison.get("model_comparison_table")))) if model_comparison else 0,
        "best_model_selected": str(model_comparison.get("best_model_name", "")) if model_comparison else "",
        "fairness_review_mode": str(model_comparison.get("fairness_review_mode", "not_run")) if model_comparison else "not_run",
        "scenario_set_executed": str(demo_config.get("scenario_simulation_mode", "basic")),
        "major_blockers_detected": ", ".join(readiness_table.loc[lambda df: df.get("status", pd.Series(dtype=str)) == "Unavailable", "analysis_module"].astype(str).tolist()[:5]) if not readiness_table.empty else "",
        "config_signature": f"{demo_config.get('synthetic_helper_mode')}|{demo_config.get('bmi_remediation_mode')}|{demo_config.get('executive_summary_verbosity')}",
    }
    return entry


def update_run_history(existing: list[dict[str, object]], entry: dict[str, object]) -> list[dict[str, object]]:
    history = list(existing or [])
    signature = (entry.get("dataset_name"), entry.get("row_count"), entry.get("column_count"), entry.get("config_signature"))
    if history:
        last = history[-1]
        last_sig = (last.get("dataset_name"), last.get("row_count"), last.get("column_count"), last.get("config_signature"))
        if last_sig == signature:
            history[-1] = entry
            return history
    history.append(entry)
    return history[-12:]


def build_audit_summary(run_history: list[dict[str, object]], analysis_log: list[dict[str, object]]) -> dict[str, object]:
    history_df = pd.DataFrame(run_history)
    log_df = pd.DataFrame(analysis_log)
    summary_table = pd.DataFrame([
        {"metric": "Runs this session", "value": f"{int(len(history_df)):,}"},
        {"metric": "Audit events logged", "value": f"{int(len(log_df)):,}"},
        {"metric": "Most recent dataset", "value": str(history_df.iloc[-1]["dataset_name"]) if not history_df.empty else "None"},
    ])
    text = f"Current session includes {len(history_df)} tracked analysis runs and {len(log_df)} audit events."
    return {
        "run_history_entry": history_df.iloc[-1].to_dict() if not history_df.empty else {},
        "audit_summary": summary_table,
        "audit_summary_text": text,
    }


def build_compliance_governance_summary(
    standards: dict[str, object],
    privacy_review: dict[str, object],
    lineage: dict[str, object],
    remediation_context: dict[str, object],
    readiness: dict[str, object],
) -> dict[str, object]:
    direct_identifiers = int(privacy_review.get("hipaa", {}).get("direct_identifier_count", 0))
    synthetic_count = int(remediation_context.get("synthetic_field_count", 0))
    sections = {
        "Standards Readiness": f"{float(standards.get('combined_readiness_score', 0.0)):.1f}/100 with badge {standards.get('badge_text', 'Not assessed')}.",
        "Privacy Posture": f"HIPAA-style risk is {privacy_review.get('hipaa', {}).get('risk_level', 'Low')} with {direct_identifiers} direct identifiers detected.",
        "Governance Depth": f"{len(lineage.get('transformation_steps', []))} transformation steps and {synthetic_count} synthetic helper fields are currently tracked.",
        "Disclosure Summary": "Synthetic and inferred support is disclosed in readiness, export, and lineage views." if synthetic_count else "Current workflow does not rely on synthetic helper fields.",
    }
    cards = [
        {"label": "Standards", "value": standards.get("badge_text", "Not assessed")},
        {"label": "Privacy Risk", "value": privacy_review.get("hipaa", {}).get("risk_level", "Low")},
        {"label": "Lineage Steps", "value": str(len(lineage.get("transformation_steps", [])))},
        {"label": "Synthetic Fields", "value": str(synthetic_count)},
    ]
    notes = [
        "This is a readiness and governance support layer, not a formal compliance certification.",
    ]
    if synthetic_count:
        notes.append("Synthetic helper fields are improving readiness but should not be treated as native source-grade fields.")
    return {
        "compliance_governance_summary": sections,
        "compliance_snapshot_cards": cards,
        "governance_notes": notes,
        "disclosure_summary": sections["Disclosure Summary"],
        "sections": sections,
        "summary_table": pd.DataFrame([{"focus_area": key, "status": value} for key, value in sections.items()]),
    }


def build_landing_summary(
    pipeline: dict[str, object],
    demo_config: dict[str, object],
    dataset_name: str,
) -> dict[str, object]:
    source_meta = pipeline.get("source_meta", {})
    source_mode = str(source_meta.get("source_mode", ""))
    is_demo_dataset = source_mode == "Demo dataset"
    readiness = pipeline["readiness"]
    remediation = pipeline.get("remediation_context", {})
    intelligence = pipeline.get("dataset_intelligence", {})
    use_case = pipeline.get("use_case_detection", {})
    solution_packages = pipeline.get("solution_packages", {})
    synthetic_count = int(remediation.get("synthetic_field_count", 0))
    dataset_type_label = str(intelligence.get("dataset_type_label", "Not classified"))
    recommended_workflow = str(use_case.get("recommended_workflow", "Healthcare Data Readiness"))
    healthcare_readiness_score = float(pipeline['healthcare'].get('healthcare_readiness_score', 0.0))
    capability_badges = [
        {"label": "Healthcare Analytics", "value": f"{healthcare_readiness_score:.0%} ready"},
        {"label": "Compliance & Governance", "value": pipeline["standards"].get("badge_text", "In review")},
        {"label": "Modeling", "value": "Enabled" if readiness.get("available_count", 0) else "Selective"},
        {"label": "Exports", "value": "Stakeholder-ready"},
    ]
    product_value_cards = [
        {"label": "Dataset Type", "value": dataset_type_label},
        {"label": "Recommended Workflow", "value": recommended_workflow},
        {"label": "Readiness", "value": f"{float(readiness.get('readiness_score', 0.0)):.0%}"},
        {"label": "Active Modules", "value": f"{int(readiness.get('available_count', 0))}"},
    ]
    differentiators = _build_dataset_specific_differentiators(
        dataset_type_label,
        recommended_workflow,
        healthcare_readiness_score,
    )
    covers = [
        "Quality review and remediation guidance",
        "Healthcare intelligence, risk, and cohort analytics",
        "Standards, privacy, and governance readiness",
        "Stakeholder reporting and export preparation",
    ]
    who_its_for = [
        "Healthcare analysts who need to understand dataset readiness, detect issues, and move into analysis quickly.",
        "Research teams and consulting teams who need explainable cohort, outcome, benchmarking, and reporting workflows.",
        "Small healthcare organizations that need a practical healthcare data readiness and analytics copilot without a heavy analytics stack.",
    ]
    four_step_workflow = [
        {
            "step": "1. Upload",
            "summary": (
                "Load a built-in demo or upload a healthcare dataset to start the readiness workflow."
                if is_demo_dataset
                else "Load the active uploaded dataset and start the readiness workflow with the current session context."
            ),
        },
        {
            "step": "2. Assess",
            "summary": "Review dataset intelligence, readiness scoring, profiling, and blocker explanations to understand what the data can support now.",
        },
        {
            "step": "3. Analyze",
            "summary": "Use healthcare analytics, cohort views, risk workflows, and AI Copilot guidance where the dataset has support.",
        },
        {
            "step": "4. Report",
            "summary": "Generate stakeholder-ready reports, exports, and next-step recommendations in minutes.",
        },
    ]
    starting_paths = [
        "Start with Readiness to review blockers, remediation priorities, and what the dataset can support now.",
        "Move to Healthcare Intelligence for cohort, risk, readmission, and pathway signals with the strongest mapped support.",
        "Finish in Export Center for executive, governance, and audit-ready handoff outputs.",
    ]
    if solution_packages.get("recommended_package"):
        starting_paths.insert(0, f"Recommended package: {solution_packages.get('recommended_package')}.")
    system_status = [
        f"Dataset: {dataset_name}",
        f"Readiness: {float(readiness.get('readiness_score', 0.0)):.0%}",
        f"Active modules: {int(readiness.get('available_count', 0))}",
    ]
    value_summary = "Profile clinical datasets, surface remediation needs, validate readiness, and package audit-ready outputs in one platform."
    positioning = (
        "Clinical data quality, remediation, and audit platform for healthcare analysts, research teams, consulting teams, and operational data teams."
    )
    investor_demo_narrative = {
        "problem": "Clinical teams often receive source extracts without a clear view of what is usable, what is missing, and what must be remediated before downstream review.",
        "workflow": [
            "Start with dataset intelligence, readiness, and remediation guidance to establish audit-ready context.",
            "Move into clinical intelligence only where mapped fields and validation rules support trustworthy review.",
            "Finish with decision support and export packaging that explains both findings and remaining caveats.",
        ],
        "value_for_teams": [
            "Reduces time spent classifying and validating unfamiliar clinical datasets.",
            "Turns blockers into concrete remediation steps instead of dead-end failures.",
            "Connects quality, governance, analytics, and reporting in one controlled workflow.",
        ],
        "top_modules": [
            "Dataset Intelligence Report",
            "Healthcare Data Readiness",
            "Healthcare Intelligence",
            "AI Copilot",
            "Export Center",
        ],
        "sample_outputs": [
            "Capability matrix and blocker explanations",
            "Readiness and remediation guidance",
            "Risk, cohort, benchmark, and pathway insights",
            "Executive report pack and stakeholder bundle",
        ],
        "recommended_demo_path": [
            "Open Dataset Profile · Overview for the dataset intelligence summary.",
            "Use Data Quality · Analysis Readiness to show blockers, remediation, and recommended workflow.",
            "Show Healthcare Analytics · Healthcare Intelligence for outcome, risk, and decision-support modules.",
            "End in Insights & Export · Export Center with the executive report pack and stakeholder bundle.",
        ],
    }
    design_partner_mode = {
        "capabilities_now": [
            "Supports dataset intelligence, quality review, remediation guidance, governance, and clinical analytics in one workflow.",
            "Adapts to native, inferred, and synthetic-assisted support transparently so pilot teams can start before every source field is perfect.",
            "Produces stakeholder-ready exports for operations, clinical review, readiness, and executive audiences.",
        ],
        "configurable_workflows": [
            "Guided solution packages align workflows to hospital operations, clinical outcomes, population health, and readiness review.",
            "Audience guidance helps analysts, clinicians, executives, researchers, and data stewards start in the right place.",
            "Workspace state, workflow packs, and collaboration notes support repeatable pilot reviews without a heavy backend rewrite.",
        ],
        "future_integration_points": [
            "Source-system ingestion and mapping services can connect into the standards, lineage, and dataset intelligence layers.",
            "Workspace, plan, and usage layers can extend into a multi-tenant SaaS backend later.",
            "Reporting and export surfaces are structured so pilot-specific deliverables can be added without reworking the analytics core.",
        ],
        "pilot_structure": [
            "Begin with one representative dataset and use readiness plus remediation to identify the fastest path to value.",
            "Run one guided solution package aligned to the partner's use case, such as operations, outcomes, or population health.",
            "Close the pilot cycle with stakeholder reports, blocker summaries, recommended upgrades, and a success checklist for the next iteration.",
        ],
        "positioning_note": "Design-partner mode is meant to show how the platform can support real healthcare datasets and configurable workflows without implying a fully custom production deployment on day one.",
    }
    missing_values = int(pipeline.get("overview", {}).get("missing_values", 0))
    duplicates = int(pipeline.get("overview", {}).get("duplicate_rows", 0))
    active_modules = int(readiness.get("available_count", 0))
    readiness_score = float(readiness.get("readiness_score", 0.0))
    intervention_count = int(len(pipeline.get("intervention_recommendations", pd.DataFrame())))
    value_cards = [
        {
            "label": "Onboarding Time Saved",
            "value": "High" if readiness_score >= 0.65 else "Moderate",
            "detail": "Dataset intelligence, readiness scoring, and remediation guidance shorten early dataset triage.",
        },
        {
            "label": "Issue Detection Speed",
            "value": "High" if (missing_values > 0 or duplicates > 0) else "Moderate",
            "detail": "Quality checks, blocker explanations, and remediation paths surface issues before deeper analysis starts.",
        },
        {
            "label": "Reporting Turnaround",
            "value": "High" if active_modules >= 5 else "Moderate",
            "detail": "Executive, stakeholder, and governance-ready exports reduce the time to produce a usable review pack.",
        },
        {
            "label": "Workflow Efficiency",
            "value": "High" if intervention_count >= 3 else "Moderate",
            "detail": "Guided workflows and decision-support outputs reduce back-and-forth between profiling, analytics, and reporting.",
        },
    ]
    roi_notes = [
        "Value estimates are directional and dataset-relative, not financial guarantees.",
        "They are based on the current dataset's readiness, issue profile, and reporting coverage.",
    ]
    if synthetic_count:
        roi_notes.append("Some unlocked value depends on synthetic helper support and should be validated with stronger source-grade fields later.")
    roi_value_estimation = {
        "cards": value_cards,
        "notes": roi_notes,
    }
    pilot_readiness_toolkit = {
        "prepare_data": [
            "Start with one representative dataset extract that has clear headers and as few duplicate columns as possible.",
            "Review dataset intelligence, readiness, and remediation before expecting deeper clinical modules to activate.",
            "Keep source-grade timestamps, coded clinical fields, and native outcome signals where possible for the strongest pilot results.",
        ],
        "run_first_analysis": [
            "Use the recommended workflow and solution package instead of exploring every module at once.",
            "Begin with Readiness, then move to Healthcare Intelligence and Export Center.",
            "Use AI Copilot prompts after the initial readiness review so the guidance is grounded in the active dataset.",
        ],
        "reports_to_generate": [
            "Generate an Analyst or Data Readiness review first to align on blockers and remediation priorities.",
            "Follow with an Executive or Operational summary once the strongest workflow modules are active.",
            "Use stakeholder export bundles to package findings, blockers, and next-step actions for pilot review meetings.",
        ],
        "interpret_results": [
            "Treat native fields as the strongest support and synthetic helpers as transparent pilot-enablement layers.",
            "Use remediation guidance to separate source data issues from analytics limitations.",
            "Frame standards, privacy, and governance outputs as readiness support rather than certification.",
        ],
        "evaluate_success": [
            "Confirm that the dataset intelligence report correctly classifies the dataset and recommended workflow.",
            "Track whether readiness improves after remediation and whether the target solution package becomes more usable.",
            "Measure success by faster onboarding, clearer blocker visibility, and stronger stakeholder-ready outputs.",
        ],
        "pilot_note": "Pilot readiness guidance is designed for practical onboarding conversations with healthcare teams, analytics leaders, and design partners.",
    }
    architecture_signals = {
        "workflow_layers": [
            "Dataset intelligence and readiness layer for classification, blockers, and remediation paths.",
            "Clinical analytics layer for cohort, risk, readmission, survival, pathway, and benchmarking workflows.",
            "Decision-support layer for recommendations, intervention planning, alerts, and scenario summaries.",
        ],
        "product_layers": [
            "Reporting and export layer for executive, operational, clinical, readiness, and stakeholder outputs.",
            "Workspace and product layer for settings, packs, sessions, notes, usage tracking, and plan-aware behavior.",
        ],
        "scalability_signals": [
            "Modular services are separated under src/ so analytics, governance, reporting, and product features can evolve independently.",
            "Synthetic, inferred, and native support are tracked separately to keep the platform transparent as workflows scale.",
            "The current architecture is ready for future backends such as authentication, billing, persistent workspaces, and external data integrations.",
        ],
        "future_integration_slots": [
            "Source-system ingestion and mapping services",
            "Persistent workspace/backend services",
            "External auth, billing, and team collaboration services",
            "Workflow automation and scheduled report orchestration",
        ],
        "technical_evaluator_note": "These architecture signals are meant to show modular product layers and future integration fit without implying an overbuilt enterprise backend today.",
    }
    startup_pitch_polish = {
        "premium_copy": [
            "A clinical data quality platform that helps teams move from unknown source extracts to decision-ready outputs faster.",
            "Built to combine readiness, remediation, analytics, and reporting in one transparent workflow instead of scattered point solutions.",
        ],
        "value_statements": [
            "Accelerates dataset onboarding and issue discovery for healthcare analysts and implementation teams.",
            "Turns blocked analytics into actionable remediation paths rather than dead-end failures.",
            "Packages findings into stakeholder-ready outputs for operations, research, and executive audiences.",
        ],
        "use_case_positioning": [
            "Investor demo: shows a differentiated workflow from messy healthcare data to insight and reporting.",
            "Customer demo: shows how the same platform adapts to readiness, operations, outcomes, and population-health reviews.",
            "Recruiter / portfolio demo: shows product thinking, healthcare domain depth, and modular engineering discipline.",
        ],
        "workflow_messaging": [
            "Start with dataset intelligence and readiness to establish trust in the data.",
            "Move into healthcare-aware analytics only where the dataset truly supports them.",
            "Finish with decision support, reports, and stakeholder bundles that explain both findings and next actions.",
        ],
        "pitch_note": "The product is designed to feel premium and enterprise-ready while staying transparent about native, inferred, and synthetic support.",
    }
    recommended_report = "Executive Summary"
    recommended_workflow_name = str(use_case.get("recommended_workflow", "Healthcare Data Readiness"))
    if "Operations" in recommended_workflow_name:
        recommended_report = "Operational Report"
    elif "Clinical" in recommended_workflow_name or "Outcome" in recommended_workflow_name:
        recommended_report = "Clinical Report"
    elif "Readiness" in recommended_workflow_name:
        recommended_report = "Analyst Report"
    startup_demo_flow = (
        {
            "headline": "Startup Demo Flow",
            "intro": "Use this concise walkthrough to show the strongest product story for first-time users, investors, and pilot stakeholders.",
            "best_dataset": "Healthcare Operations Demo",
            "dataset_reason": "It activates readiness, clinical analytics, cohort review, readmission, AI Copilot, and reporting in one walkthrough.",
            "estimated_demo_time": "5-7 minutes",
            "quick_start_steps": [
                "Load the Healthcare Operations Demo from the built-in dataset selector.",
                "Open Overview to frame the dataset type and coverage story.",
                "Use Readiness to explain blockers and what the file supports now.",
                "Finish with Healthcare Intelligence, AI Copilot, and Export Center for the decision-ready handoff.",
            ],
            "recommended_tabs": [
                "Overview",
                "Readiness",
                "Healthcare Intelligence",
                "Export Center",
            ],
            "suggested_copilot_prompts": [
                "Summarize the dataset",
                "Identify high-risk groups",
                "Explain readiness blockers",
                "Show healthcare insights",
            ],
            "recommended_export": recommended_report,
            "export_reason": "This export gives the clearest stakeholder-ready summary after the recommended walkthrough is complete.",
            "demo_outcome": "The audience should leave understanding what the dataset can support, where the biggest blockers are, which insights are trustworthy now, and what report to share next.",
        }
        if is_demo_dataset
        else {}
    )
    onboarding_cues = [
        "Start with one representative dataset rather than a full historical export for the first pilot review.",
        "Use the recommended workflow and guided demo path before exploring every tab in depth.",
        "Close each review by generating one report and one next-step recommendation set for stakeholders.",
    ]
    workspace_handoff_cues = [
        "Use snapshots for point-in-time review states and workflow packs for repeatable review patterns.",
        "Capture collaboration notes on datasets, reports, and workflow packs so pilot follow-up is easier to review.",
        "Use the Product Admin / Customer Ops summary to monitor pilot activity, reports, and workspace value signals.",
    ]
    report_polish_cues = [
        "Choose one primary audience-facing report for each review cycle instead of generating every export at once.",
        "Pair the main report with the compliance or governance pack when a broader handoff is needed.",
        "Call out any synthetic or helper-backed support explicitly before sharing findings outside the working session.",
    ]
    synthetic_note = "Synthetic helper fields are enabled in a transparent demo-safe mode." if synthetic_count else "Current analysis relies primarily on native fields."
    return {
        "headline": "Clinverity",
        "subheadline": "Clinical Data Quality, Remediation & Audit Platform",
        "audience_summary": "Built for healthcare analysts, research teams, consulting teams, and small healthcare organizations.",
        "capability_badges": capability_badges,
        "product_value_cards": product_value_cards,
        "differentiators": differentiators,
        "analysis_covers": covers,
        "who_its_for": who_its_for,
        "four_step_workflow": four_step_workflow,
        "recommended_starting_paths": starting_paths,
        "system_status": system_status,
        "platform_value_summary": value_summary,
        "positioning_statement": positioning,
        "investor_demo_narrative": investor_demo_narrative,
        "design_partner_mode": design_partner_mode,
        "roi_value_estimation": roi_value_estimation,
        "pilot_readiness_toolkit": pilot_readiness_toolkit,
        "architecture_signals": architecture_signals,
        "startup_pitch_polish": startup_pitch_polish,
        "startup_demo_flow": startup_demo_flow,
        "onboarding_cues": onboarding_cues,
        "workspace_handoff_cues": workspace_handoff_cues,
        "report_polish_cues": report_polish_cues,
        "synthetic_support_note": synthetic_note,
    }
