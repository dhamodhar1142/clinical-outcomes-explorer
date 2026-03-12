from __future__ import annotations

from collections import Counter
from typing import Any

import pandas as pd


def _safe_df(table: Any) -> pd.DataFrame:
    return table if isinstance(table, pd.DataFrame) else pd.DataFrame()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _safe_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, pd.Series):
        return [str(item) for item in value.dropna().astype(str).tolist() if str(item).strip()]
    return []


def _canonical_fields(semantic: dict[str, object]) -> set[str]:
    canonical_map = semantic.get("canonical_map", {})
    return {str(key) for key in canonical_map.keys()}


def _helper_field_summary(remediation_context: dict[str, object]) -> tuple[list[str], list[str], list[str]]:
    helper_fields = _safe_df(remediation_context.get("helper_fields"))
    if helper_fields.empty or "helper_field" not in helper_fields.columns:
        return [], [], []
    native = []
    inferred = helper_fields.loc[
        helper_fields.get("helper_type", pd.Series(dtype=str)).astype(str).eq("inferred"),
        "helper_field",
    ].astype(str).tolist() if "helper_type" in helper_fields.columns else []
    synthetic = helper_fields.loc[
        helper_fields.get("helper_type", pd.Series(dtype=str)).astype(str).isin(["synthetic", "derived"]),
        "helper_field",
    ].astype(str).tolist() if "helper_type" in helper_fields.columns else []
    return native, inferred, synthetic


def _support_for_fields(fields: list[str], canonical_map: dict[str, str], synthetic_fields: set[str], inferred_fields: set[str]) -> str:
    if not fields:
        return "unavailable"
    matches = [field for field in fields if field in canonical_map]
    if not matches:
        return "unavailable"
    if any(field in synthetic_fields for field in matches):
        return "synthetic-assisted"
    if any(field in inferred_fields for field in matches):
        return "inferred"
    return "native"


def _dataset_type_classification(
    semantic: dict[str, object],
    readiness: dict[str, object],
    healthcare: dict[str, object],
    standards: dict[str, object],
) -> dict[str, object]:
    canonical = _canonical_fields(semantic)
    readiness_score = _safe_float(readiness.get("readiness_score"))
    healthcare_score = _safe_float(healthcare.get("healthcare_readiness_score"))
    structure_conf = _safe_float(semantic.get("semantic_confidence_score"))
    cdisc = standards.get("cdisc_report", {})

    diagnosis_or_proc = bool({"diagnosis_code", "procedure_code"} & canonical)
    patient_context = bool({"patient_id", "member_id", "encounter_id"} & canonical)
    encounter_context = bool({"admission_date", "discharge_date", "service_date", "event_date"} & canonical)
    operational_context = bool({"provider_id", "provider_name", "facility", "department", "payer", "plan"} & canonical)
    trial_context = bool({"diagnosis_date", "end_treatment_date", "survived", "treatment_type", "cancer_stage"} & canonical)

    label = "Generic tabular dataset"
    rationale = "The uploaded data currently behaves like a general-purpose tabular dataset with limited healthcare structure."
    confidence = 0.45

    if cdisc.get("available") and _safe_float(cdisc.get("readiness_score")) >= 45:
        label = "Trial / research-oriented dataset"
        rationale = "The dataset shows trial-style identifiers, visit structure, or research outcome fields that align with SDTM or ADaM-style review."
        confidence = min(0.95, 0.55 + _safe_float(cdisc.get("readiness_score")) / 120)
    elif patient_context and encounter_context and operational_context:
        label = "Claims / encounter-level healthcare dataset"
        rationale = "The dataset includes encounter-style timing, patient linking, and operational or financial fields that support claims or utilization analysis."
        confidence = min(0.95, 0.55 + healthcare_score * 0.35)
    elif patient_context and (diagnosis_or_proc or trial_context):
        label = "Clinical / patient-level dataset"
        rationale = "The dataset includes patient-level fields plus clinical outcomes, diagnosis, or treatment context suitable for cohort and risk analytics."
        confidence = min(0.92, 0.50 + healthcare_score * 0.35)
    elif operational_context and healthcare_score >= 0.35:
        label = "Mixed healthcare operational dataset"
        rationale = "The dataset has meaningful healthcare and operational context, but some clinical or encounter fields remain incomplete."
        confidence = min(0.90, 0.45 + healthcare_score * 0.30)
    elif healthcare_score >= 0.20 or diagnosis_or_proc:
        label = "Healthcare-related dataset"
        rationale = "The dataset contains healthcare-oriented fields, but current structure supports only selective healthcare analytics."
        confidence = min(0.85, 0.40 + max(healthcare_score, readiness_score) * 0.25)
    elif structure_conf >= 0.45:
        confidence = 0.55

    return {
        "dataset_type_label": label,
        "dataset_type_confidence": round(confidence, 2),
        "dataset_type_rationale": rationale,
    }


def _module_reason(module_name: str, healthcare: dict[str, object], readiness_table: pd.DataFrame) -> str:
    module_lookup = {
        "Risk Segmentation": healthcare.get("risk_segmentation", {}),
        "Cohort Analysis": healthcare.get("default_cohort_summary", {}),
        "Readmission Analytics": healthcare.get("readmission", {}),
        "Cost Driver Analysis": healthcare.get("cost", {}),
        "Diagnosis / Procedure Analysis": healthcare.get("diagnosis", {}),
        "Provider / Facility Volume": healthcare.get("provider", {}),
        "Care Pathway Intelligence": healthcare.get("care_pathway", {}),
    }
    module_payload = module_lookup.get(module_name, {})
    if isinstance(module_payload, dict):
        if module_payload.get("reason"):
            return str(module_payload["reason"])
        if module_name == "Readmission Analytics":
            readiness_info = module_payload.get("readiness", {})
            missing = _safe_list(readiness_info.get("missing_fields"))
            if missing:
                return f"Additional encounter fields are needed: {', '.join(missing[:4])}."

    if not readiness_table.empty and "analysis_module" in readiness_table.columns:
        matched = readiness_table[readiness_table["analysis_module"].astype(str) == module_name]
        if not matched.empty:
            missing = str(matched.iloc[0].get("missing_prerequisites", "")).strip()
            if missing and missing != "-":
                return f"Current support is limited by missing prerequisites: {missing}."
    return "Current dataset coverage is not yet strong enough for this workflow."


def _capability_matrix(
    semantic: dict[str, object],
    readiness: dict[str, object],
    healthcare: dict[str, object],
    standards: dict[str, object],
    privacy_review: dict[str, object],
    remediation_context: dict[str, object],
) -> pd.DataFrame:
    canonical_map = semantic.get("canonical_map", {})
    canonical = set(canonical_map.keys())
    _, inferred_list, synthetic_list = _helper_field_summary(remediation_context)
    inferred_fields = set(inferred_list)
    synthetic_fields = set(synthetic_list)
    readiness_table = _safe_df(readiness.get("readiness_table"))

    modules: list[dict[str, object]] = []

    def add(module: str, status: str, support: str, rationale: str) -> None:
        modules.append(
            {
                "analytics_module": module,
                "status": status,
                "support": support,
                "rationale": rationale,
            }
        )

    add("Data Profiling", "enabled", "native", "Core profiling and summary views are available for all tabular datasets.")
    add("Data Quality Review", "enabled", "native", "Data quality diagnostics and rule checks are active for the current dataset.")

    risk_payload = healthcare.get("risk_segmentation", {})
    if risk_payload.get("available"):
        support = "synthetic-assisted" if remediation_context.get("synthetic_clinical", {}).get("available") else "native"
        add("Risk Segmentation", "enabled", support, "Risk segmentation is available from current clinical and demographic support.")
    else:
        add("Risk Segmentation", "blocked", "unavailable", _module_reason("Risk Segmentation", healthcare, readiness_table))

    cohort_payload = healthcare.get("default_cohort_summary", {})
    if cohort_payload.get("available"):
        support = "synthetic-assisted" if remediation_context.get("synthetic_clinical", {}).get("available") else "native"
        add("Cohort Analysis", "enabled", support, "Cohort analysis is supported with the current patient-level fields.")
    else:
        add("Cohort Analysis", "blocked", "unavailable", _module_reason("Cohort Analysis", healthcare, readiness_table))

    readmission_payload = healthcare.get("readmission", {})
    if readmission_payload.get("available"):
        support = "synthetic-assisted" if str(readmission_payload.get("source", "")).lower().startswith("synthetic") or remediation_context.get("synthetic_readmission", {}).get("available") else "native"
        status = "partial" if support == "synthetic-assisted" else "enabled"
        rationale = (
            "Readmission review is synthetic-assisted because native readmission support is incomplete."
            if status == "partial"
            else "Readmission review is available from current encounter and outcome fields."
        )
        add("Readmission Analytics", status, support, rationale)
    else:
        add("Readmission Analytics", "blocked", "unavailable", _module_reason("Readmission Analytics", healthcare, readiness_table))

    modeling_target_fields = [field for field in ["readmission", "survived"] if field in canonical]
    modeling_feature_fields = [field for field in ["age", "bmi", "smoking_status", "length_of_stay", "comorbidities", "diagnosis_code", "department"] if field in canonical]
    if modeling_target_fields and len(modeling_feature_fields) >= 2:
        support = "synthetic-assisted" if any(field in synthetic_fields for field in modeling_target_fields + modeling_feature_fields) else "native"
        status = "partial" if support == "synthetic-assisted" else "enabled"
        rationale = "Predictive modeling can run with the current target and feature coverage."
        if status == "partial":
            rationale = "Predictive modeling is available, but some usable features or targets rely on synthetic helper support."
        add("Predictive Modeling", status, support, rationale)
    elif modeling_target_fields or len(modeling_feature_fields) >= 2:
        add("Predictive Modeling", "partial", "unavailable", "A suitable target or broader feature set is still needed for guided predictive modeling.")
    else:
        add("Predictive Modeling", "blocked", "unavailable", "A supported outcome field and feature set are needed before predictive modeling can run.")

    cost_payload = healthcare.get("cost", {})
    if cost_payload.get("available"):
        support = "synthetic-assisted" if remediation_context.get("synthetic_cost", {}).get("available") else "native"
        status = "partial" if support == "synthetic-assisted" else "enabled"
        rationale = "Cost analytics are available."
        if status == "partial":
            rationale = "Cost analytics are synthetic-assisted because estimated_cost is demo-derived."
        add("Cost Driver Analysis", status, support, rationale)
    else:
        add("Cost Driver Analysis", "blocked", "unavailable", _module_reason("Cost Driver Analysis", healthcare, readiness_table))

    diagnosis_payload = healthcare.get("diagnosis", {})
    if diagnosis_payload.get("available"):
        support = "synthetic-assisted" if remediation_context.get("synthetic_clinical", {}).get("available") else "native"
        status = "partial" if support == "synthetic-assisted" else "enabled"
        rationale = "Diagnosis and procedure segmentation is available."
        if status == "partial":
            rationale = "Diagnosis segmentation is synthetic-assisted because clinical groupings were derived for demo use."
        add("Diagnosis / Procedure Analysis", status, support, rationale)
    else:
        add("Diagnosis / Procedure Analysis", "blocked", "unavailable", _module_reason("Diagnosis / Procedure Analysis", healthcare, readiness_table))

    provider_payload = healthcare.get("provider", {})
    if provider_payload.get("available"):
        add("Provider / Facility Volume", "enabled", "native", "Provider or facility grouping fields support operational volume review.")
    else:
        add("Provider / Facility Volume", "blocked", "unavailable", _module_reason("Provider / Facility Volume", healthcare, readiness_table))

    trend_row = readiness_table[readiness_table.get("analysis_module", pd.Series(dtype=str)).astype(str) == "Trend Analysis"]
    if not trend_row.empty:
        row = trend_row.iloc[0]
        status = str(row.get("status", "Unavailable")).strip().lower()
        support = str(row.get("support_type", "Native")).strip().lower().replace("native", "native").replace("synthetic-assisted", "synthetic-assisted")
        mapped_status = {"available": "enabled", "partial": "partial", "unavailable": "blocked"}.get(status, "blocked")
        rationale = "Trend analysis is available from the current temporal fields."
        if mapped_status == "partial":
            rationale = f"Trend analysis is limited: {row.get('missing_prerequisites', '-')}"
        elif mapped_status == "blocked":
            rationale = "Trend analysis is blocked because no event-style date field is available."
        elif support == "synthetic-assisted":
            rationale = "Trend analysis is synthetic-assisted because event_date was generated from an existing year or date-part field."
        add("Trend Analysis", mapped_status, support if mapped_status != "blocked" else "unavailable", rationale)
    else:
        add("Trend Analysis", "blocked", "unavailable", "Trend analysis is blocked because no event-style date field is available.")

    monitoring_available = bool(readiness_table[readiness_table.get("analysis_module", pd.Series(dtype=str)).astype(str) == "Trend Analysis"].shape[0]) and not _safe_df(healthcare.get("default_cohort_summary", {}).get("trend_table")).empty
    if monitoring_available:
        add("Cohort Monitoring Over Time", "enabled", "native", "Cohort monitoring over time is available with the current temporal and cohort support.")
    elif trend_row.empty or (not trend_row.empty and str(trend_row.iloc[0].get("status")) == "Unavailable"):
        add("Cohort Monitoring Over Time", "blocked", "unavailable", "Cohort monitoring needs a usable event-style date field.")
    else:
        add("Cohort Monitoring Over Time", "partial", "native", "Basic cohort logic is available, but the current dataset needs stronger longitudinal support for full monitoring.")

    pathway_payload = healthcare.get("care_pathway", {})
    if pathway_payload.get("available"):
        add("Care Pathway Intelligence", "enabled", "native", "Care pathway analysis is available from current treatment and outcome timing support.")
    else:
        add("Care Pathway Intelligence", "blocked", "unavailable", _module_reason("Care Pathway Intelligence", healthcare, readiness_table))

    standards_available = standards.get("available") or not _safe_df(privacy_review.get("sensitive_fields")).empty
    if standards_available:
        support = "synthetic-assisted" if synthetic_fields else "native"
        add("Standards / Governance Review", "enabled", support, "Standards, privacy, and governance review is available for the current dataset.")
    else:
        add("Standards / Governance Review", "partial", "unavailable", "Governance review is available, but stronger healthcare structure would improve standards coverage.")

    add("Export / Executive Reporting", "enabled", "native", "Executive reporting and stakeholder exports are available from the current pipeline outputs.")

    return pd.DataFrame(modules)


def _build_explanations(capability_matrix: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    blocked = capability_matrix[capability_matrix["status"] == "blocked"].copy()
    partial = capability_matrix[capability_matrix["status"] == "partial"].copy()
    if not blocked.empty:
        blocked = blocked.rename(columns={"analytics_module": "analytics_area", "rationale": "why_blocked"})[
            ["analytics_area", "why_blocked", "support"]
        ]
    if not partial.empty:
        partial = partial.rename(columns={"analytics_module": "analytics_area", "rationale": "why_partial"})[
            ["analytics_area", "why_partial", "support"]
        ]
    return blocked, partial


def _recommendation_priority(module_count: int, governance_value: int) -> tuple[str, int]:
    score = module_count * 3 + governance_value
    if score >= 8:
        return "High", score
    if score >= 4:
        return "Medium", score
    return "Low", score


def _build_upgrade_recommendations(
    capability_matrix: pd.DataFrame,
    healthcare: dict[str, object],
    remediation: pd.DataFrame,
    remediation_context: dict[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    field_unlocks: dict[str, dict[str, Any]] = {
        "encounter_date": {"modules": ["Trend Analysis", "Cohort Monitoring Over Time", "Readmission Analytics"], "reason": "Unlock encounter sequencing and time-based monitoring.", "governance": 2},
        "admission_date": {"modules": ["Readmission Analytics", "Trend Analysis", "Cohort Monitoring Over Time"], "reason": "Strengthen encounter timing and longitudinal analysis.", "governance": 2},
        "discharge_date": {"modules": ["Readmission Analytics", "Care Pathway Intelligence"], "reason": "Complete encounter structure and treatment duration logic.", "governance": 2},
        "diagnosis_code": {"modules": ["Diagnosis / Procedure Analysis", "Cohort Analysis", "Care Pathway Intelligence"], "reason": "Improve clinical segmentation and pathway review.", "governance": 2},
        "procedure_code": {"modules": ["Diagnosis / Procedure Analysis", "Care Pathway Intelligence"], "reason": "Enable richer procedure and treatment pattern analysis.", "governance": 1},
        "payer": {"modules": ["Cost Driver Analysis", "Provider / Facility Volume"], "reason": "Improve financial benchmarking and payer segmentation.", "governance": 2},
        "provider_id": {"modules": ["Provider / Facility Volume", "Cost Driver Analysis"], "reason": "Unlock provider benchmarking and operational review.", "governance": 2},
        "facility_id": {"modules": ["Provider / Facility Volume", "Cost Driver Analysis"], "reason": "Improve facility benchmarking and governance traceability.", "governance": 2},
        "native readmission flag": {"modules": ["Readmission Analytics", "Predictive Modeling"], "reason": "Move readmission workflows from synthetic support toward source-grade analysis.", "governance": 3},
        "outcome fields": {"modules": ["Predictive Modeling", "Care Pathway Intelligence"], "reason": "Strengthen outcome-based modeling and pathway evaluation.", "governance": 2},
        "lab/safety/efficacy fields": {"modules": ["Risk Segmentation", "Predictive Modeling", "Care Pathway Intelligence"], "reason": "Improve patient-level severity and response measurement.", "governance": 1},
    }
    blocked_or_partial = set(
        capability_matrix.loc[capability_matrix["status"].isin(["blocked", "partial"]), "analytics_module"].astype(str).tolist()
    )
    rows: list[dict[str, object]] = []
    for field_name, meta in field_unlocks.items():
        affected = [module for module in meta["modules"] if module in blocked_or_partial]
        if not affected:
            continue
        priority, score = _recommendation_priority(len(affected), int(meta["governance"]))
        rows.append(
            {
                "recommended_field": field_name,
                "priority": priority,
                "priority_score": score,
                "unlock_impact": len(affected),
                "analytics_unlocked": ", ".join(affected),
                "why_it_matters": meta["reason"],
                "governance_value": int(meta["governance"]),
            }
        )

    if not remediation.empty:
        for row in remediation.head(5).itertuples(index=False):
            field_name = str(getattr(row, "priority_field", ""))
            if not field_name or any(existing["recommended_field"] == field_name for existing in rows):
                continue
            modules = _safe_list(getattr(row, "modules_unlocked", []))
            priority, score = _recommendation_priority(len(modules), 1)
            rows.append(
                {
                    "recommended_field": field_name,
                    "priority": priority,
                    "priority_score": score,
                    "unlock_impact": len(modules),
                    "analytics_unlocked": ", ".join(modules),
                    "why_it_matters": str(getattr(row, "why_it_matters", getattr(row, "impact_on_analytics", ""))),
                    "governance_value": 1,
                }
            )

    recommendations = pd.DataFrame(rows).sort_values(["priority_score", "unlock_impact"], ascending=[False, False]).reset_index(drop=True) if rows else pd.DataFrame(columns=["recommended_field", "priority", "priority_score", "unlock_impact", "analytics_unlocked", "why_it_matters", "governance_value"])
    source_improvements = pd.DataFrame(
        [
            {
                "source_improvement": "Standardize encounter-level timestamps",
                "value": "Improves trend, cohort monitoring, readmission, and pathway workflows.",
            },
            {
                "source_improvement": "Add native coded diagnosis/procedure fields",
                "value": "Improves clinical segmentation, pathway review, and governance quality.",
            },
            {
                "source_improvement": "Add native financial or payer support",
                "value": "Improves cost benchmarking and provider/facility review.",
            },
        ]
    )
    highest_impact = recommendations.head(3).copy()

    next_actions: list[str] = []
    if remediation_context.get("synthetic_cost", {}).get("available"):
        next_actions.append("Replace synthetic estimated cost with a source-grade cost or payment field for stronger financial analysis.")
    if remediation_context.get("synthetic_readmission", {}).get("available"):
        next_actions.append("Add a native readmission flag or encounter dates to move readmission analytics beyond demo-derived support.")
    if remediation_context.get("synthetic_clinical", {}).get("available"):
        next_actions.append("Add native diagnosis or procedure fields to replace derived clinical groupings.")
    if recommendations.shape[0]:
        next_actions.extend(
            [f"Add {row.recommended_field} to unlock {row.analytics_unlocked}." for row in recommendations.head(2).itertuples(index=False)]
        )
    return recommendations, source_improvements, highest_impact, next_actions[:5]


def build_dataset_intelligence_report(
    data: pd.DataFrame,
    structure,
    semantic: dict[str, object],
    readiness: dict[str, object],
    healthcare: dict[str, object],
    quality: dict[str, object],
    remediation: pd.DataFrame,
    remediation_context: dict[str, object],
    standards: dict[str, object],
    privacy_review: dict[str, object],
    compliance_governance_summary: dict[str, object],
) -> dict[str, object]:
    dataset_type = _dataset_type_classification(semantic, readiness, healthcare, standards)
    capability_matrix = _capability_matrix(semantic, readiness, healthcare, standards, privacy_review, remediation_context)
    blocked_explanations, partial_explanations = _build_explanations(capability_matrix)
    recommendations, source_improvements, highest_impact, next_actions = _build_upgrade_recommendations(
        capability_matrix,
        healthcare,
        remediation,
        remediation_context,
    )

    canonical_map = semantic.get("canonical_map", {})
    native_fields = sorted(str(field) for field in canonical_map.keys() if field not in {"event_date", "estimated_cost", "diagnosis_label", "diagnosis_code", "clinical_risk_label", "readmission_flag"})
    _, inferred_fields, synthetic_fields = _helper_field_summary(remediation_context)
    synthetic_fields = sorted(set(synthetic_fields))
    inferred_fields = sorted(set(inferred_fields))

    enabled = capability_matrix.loc[capability_matrix["status"] == "enabled", "analytics_module"].astype(str).tolist()
    partial = capability_matrix.loc[capability_matrix["status"] == "partial", "analytics_module"].astype(str).tolist()
    blocked = capability_matrix.loc[capability_matrix["status"] == "blocked", "analytics_module"].astype(str).tolist()

    summary = {
        "dataset_type": dataset_type["dataset_type_label"],
        "healthcare_coverage": f"{_safe_float(healthcare.get('healthcare_readiness_score')):.0%}",
        "analytics_readiness": f"{_safe_float(readiness.get('readiness_score')):.0%}",
        "data_quality_score": f"{_safe_float(quality.get('quality_score')):.1f}",
        "structure_confidence": f"{_safe_float(getattr(structure, 'confidence_score', 0.0)):.0%}",
        "enabled_modules": len(enabled),
        "blocked_modules": len(blocked),
    }

    support_note_parts = []
    if synthetic_fields:
        support_note_parts.append("Some advanced workflows currently rely on synthetic helper fields created for demo-safe analysis.")
    if inferred_fields:
        support_note_parts.append("Several analytics use inferred semantic mappings that should be confirmed for production-grade use.")
    if not support_note_parts:
        support_note_parts.append("Current analytics rely primarily on native source structure and confirmed mappings.")

    governance_notes = _safe_list(compliance_governance_summary.get("governance_notes"))
    if not governance_notes:
        governance_notes = ["Governance coverage is summarized from standards, privacy, and lineage outputs."]

    return {
        "dataset_intelligence_summary": summary,
        **dataset_type,
        "healthcare_coverage": _safe_float(healthcare.get("healthcare_readiness_score")),
        "analytics_readiness": _safe_float(readiness.get("readiness_score")),
        "data_quality_score": _safe_float(quality.get("quality_score")),
        "structure_confidence": _safe_float(getattr(structure, "confidence_score", 0.0)),
        "enabled_analytics": enabled,
        "partially_enabled_analytics": partial,
        "blocked_analytics": blocked,
        "analytics_capability_matrix": capability_matrix,
        "blocker_explanations": blocked_explanations,
        "partial_support_explanations": partial_explanations,
        "recommended_fields_to_add": recommendations,
        "recommended_source_improvements": source_improvements,
        "highest_impact_upgrades": highest_impact,
        "native_support_summary": native_fields[:12],
        "inferred_support_summary": inferred_fields[:12],
        "synthetic_support_summary": synthetic_fields[:12],
        "support_disclosure_note": " ".join(support_note_parts),
        "governance_notes": governance_notes,
        "next_best_actions": next_actions,
    }
