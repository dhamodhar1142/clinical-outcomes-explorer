from __future__ import annotations

import re
from difflib import SequenceMatcher

import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype


DATASET_MODES = [
    "patient_level_clinical",
    "hospital_measure_reporting",
    "generic_tabular",
]

EXPECTED_FIELDS = [
    "admissions",
    "age",
    "gender",
    "diagnosis",
    "department",
    "length_of_stay",
    "cost",
    "readmission",
    "date",
    "comorbidity_score",
    "prior_admissions_12m",
    "hospital_name",
    "provider_id",
    "measure_name",
    "measure_id",
    "score",
    "compared_to_national",
    "denominator",
]

FIELD_LABELS = {
    "admissions": "Admissions",
    "age": "Age",
    "gender": "Gender",
    "diagnosis": "Diagnosis",
    "department": "Department",
    "length_of_stay": "Length of Stay",
    "cost": "Cost",
    "readmission": "Readmission",
    "date": "Admission Date",
    "comorbidity_score": "Comorbidity Score",
    "prior_admissions_12m": "Prior Admissions (12 Months)",
    "hospital_name": "Hospital Name",
    "provider_id": "Provider ID",
    "measure_name": "Measure Name",
    "measure_id": "Measure ID",
    "score": "Score",
    "compared_to_national": "Compared To National",
    "denominator": "Denominator",
}

ALIASES = {
    "admissions": ["admission_id", "encounter_id", "visit_id", "patient_id", "admissions", "admit_id", "encounter_number"],
    "age": ["age", "patient_age", "age_years"],
    "gender": ["gender", "sex", "patient_gender"],
    "diagnosis": ["diagnosis", "primary_diagnosis", "dx", "dx_code", "condition", "disease", "diagnosis_group"],
    "department": ["department", "dept", "service_line", "unit", "division", "ward", "specialty"],
    "length_of_stay": ["length_of_stay", "los", "los_days", "stay_days", "hospital_days", "stay_length", "days_admitted", "time_in_hospital"],
    "cost": ["cost", "charge", "charges", "total_charge", "total_cost", "billing_amount", "hospital_cost", "treatment_cost"],
    "readmission": ["readmission", "readmitted", "readmit_flag", "is_readmission", "readmission_flag", "is_readmitted"],
    "date": ["admission_date", "admit_date", "date", "visit_date", "encounter_date", "service_date", "measure_start_date", "measure_end_date"],
    "comorbidity_score": ["comorbidity_score", "comorbidity_index", "charlson_score", "risk_score"],
    "prior_admissions_12m": ["prior_admissions_12m", "prior_admissions", "admissions_last_12m", "previous_admissions"],
    "hospital_name": ["hospital_name", "facility_name", "provider_name", "organization_name"],
    "provider_id": ["provider_id", "facility_id", "hospital_id", "cms_certification_num", "ccn"],
    "measure_name": ["measure_name", "measure", "metric_name", "quality_measure"],
    "measure_id": ["measure_id", "measure_code", "metric_id", "quality_measure_id"],
    "score": ["score", "rate", "value", "performance_score", "measure_score"],
    "compared_to_national": ["compared_to_national", "comparison_to_national", "national_comparison", "benchmark_comparison"],
    "denominator": ["denominator", "case_count", "sample_size", "eligible_cases"],
}

FIELD_KEYWORDS = {field: [normalize for normalize in aliases] for field, aliases in ALIASES.items()}
REQUIRED_FOR_KPI = ["readmission", "length_of_stay", "cost"]
REQUIRED_FOR_COHORT = ["age", "diagnosis", "department", "gender"]
REQUIRED_FOR_COST = ["cost", "length_of_stay", "department", "diagnosis"]
REQUIRED_FOR_ML = ["age", "comorbidity_score", "prior_admissions_12m", "length_of_stay", "readmission"]
FUZZY_THRESHOLD = 0.60
PATIENT_MODE_FIELDS = ["age", "gender", "diagnosis", "readmission", "length_of_stay", "cost", "admissions"]
MEASURE_MODE_FIELDS = ["hospital_name", "provider_id", "measure_name", "measure_id", "score", "compared_to_national", "denominator"]


def normalize_column_name(column_name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", str(column_name).strip().lower())
    return re.sub(r"_+", "_", cleaned).strip("_")


def normalize_columns(data: pd.DataFrame) -> pd.DataFrame:
    normalized = data.copy()
    normalized.columns = [normalize_column_name(column) for column in normalized.columns]
    return normalized


def _similarity(left: str, right: str) -> float:
    return float(SequenceMatcher(None, normalize_column_name(left), normalize_column_name(right)).ratio())


def _sample_series(series: pd.Series, limit: int = 250) -> pd.Series:
    non_null = series.dropna()
    if len(non_null) > limit:
        return non_null.head(limit)
    return non_null


def _is_binary_like(series: pd.Series) -> bool:
    sample = _sample_series(series).astype(str).str.strip().str.lower()
    if sample.empty:
        return False
    allowed = {"0", "1", "true", "false", "yes", "no", "y", "n", "readmitted", "not readmitted"}
    return set(sample.unique()).issubset(allowed) or sample.nunique() <= 2


def _is_date_like(series: pd.Series) -> bool:
    sample = _sample_series(series)
    if sample.empty:
        return False
    parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
    return float(parsed.notna().mean()) >= 0.7


def _is_small_category(series: pd.Series, max_unique: int = 30) -> bool:
    sample = _sample_series(series).astype(str)
    return not sample.empty and sample.nunique() <= max_unique


def _numeric_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _name_score(column_name: str, field: str) -> tuple[float, str]:
    normalized = normalize_column_name(column_name)
    normalized_tokens = set(token for token in normalized.split("_") if token)
    best_score = 0.0
    best_alias = ""
    for alias in ALIASES.get(field, []):
        alias_norm = normalize_column_name(alias)
        alias_tokens = set(token for token in alias_norm.split("_") if token)
        if normalized == alias_norm:
            return 1.0, f"exact synonym match: {alias}"
        token_overlap = len(normalized_tokens & alias_tokens) / max(len(normalized_tokens | alias_tokens), 1)
        if token_overlap >= 0.75:
            score = 0.86
        elif token_overlap >= 0.5:
            score = 0.58
        else:
            score = _similarity(normalized, alias_norm)
        if score > best_score:
            best_score = score
            best_alias = alias
    if best_score >= FUZZY_THRESHOLD:
        return best_score, f"column-name synonym match: {best_alias}"
    return best_score, "column name has weak synonym similarity"


def _value_score(series: pd.Series, field: str) -> tuple[float, str]:
    sample = _sample_series(series)
    if sample.empty:
        return 0.0, "column has no non-null sample values"
    normalized_text = sample.astype(str).str.strip().str.lower()
    numeric = _numeric_series(sample)
    unique_ratio = float(sample.nunique(dropna=True) / max(len(sample), 1))

    if field == "age":
        valid = numeric.dropna()
        if not valid.empty and float(((valid >= 0) & (valid <= 120)).mean()) >= 0.9:
            return 0.9, "numeric values mostly fall within a realistic age range"
    if field == "gender":
        allowed = {"m", "f", "male", "female", "other", "unknown", "u"}
        if set(normalized_text.unique()).issubset(allowed) and normalized_text.nunique() <= 6:
            return 0.92, "categorical values look like gender codes"
    if field == "readmission":
        if _is_binary_like(sample):
            return 0.93, "binary values are consistent with a readmission flag"
    if field == "date":
        if _is_date_like(sample):
            return 0.95, "sample values parse cleanly as dates"
    if field == "length_of_stay":
        valid = numeric.dropna()
        if not valid.empty and float(((valid >= 0) & (valid <= 90)).mean()) >= 0.8:
            return 0.85, "numeric values look like hospital stay duration"
    if field == "cost":
        valid = numeric.dropna()
        if not valid.empty and float((valid >= 0).mean()) >= 0.95 and valid.nunique() > 5:
            return 0.78, "non-negative numeric values look like financial amounts"
    if field == "comorbidity_score":
        valid = numeric.dropna()
        if not valid.empty and float(((valid >= 0) & (valid <= 20)).mean()) >= 0.8:
            return 0.8, "numeric values look like a bounded comorbidity score"
    if field == "prior_admissions_12m":
        valid = numeric.dropna()
        if not valid.empty and float(((valid >= 0) & (valid <= 20)).mean()) >= 0.8:
            return 0.84, "numeric values look like prior admission counts"
    if field in {"diagnosis", "department", "hospital_name", "measure_name", "compared_to_national"}:
        if _is_small_category(sample, 40):
            return 0.62, "categorical/text values fit the expected field structure"
    if field in {"admissions", "provider_id", "measure_id"}:
        if unique_ratio >= 0.7:
            return 0.7, "values are highly unique and resemble identifier fields"
    if field in {"score", "denominator"}:
        valid = numeric.dropna()
        if not valid.empty:
            return 0.72, "numeric values match the expected reporting metric structure"
    return 0.0, "value pattern does not strongly support this field"


def _mode_bonus(field: str, mode: str) -> tuple[float, str]:
    if mode == "patient_level_clinical" and field in PATIENT_MODE_FIELDS + ["comorbidity_score", "prior_admissions_12m"]:
        return 0.08, "whole-dataset pattern supports patient-level clinical mapping"
    if mode == "hospital_measure_reporting" and field in MEASURE_MODE_FIELDS:
        return 0.08, "whole-dataset pattern supports hospital measure reporting"
    if mode == "generic_tabular":
        return 0.0, "generic tabular mode provides no additional field bias"
    return 0.0, "dataset mode does not add confidence for this field"


def _classify_dataset_mode(normalized: pd.DataFrame) -> dict[str, object]:
    columns = list(normalized.columns)
    patient_score = 0.0
    measure_score = 0.0
    patient_reasons: list[str] = []
    measure_reasons: list[str] = []

    def has_alias(field: str) -> bool:
        return any(_name_score(column, field)[0] >= FUZZY_THRESHOLD for column in columns)

    if any(has_alias(field) for field in ["age", "gender", "diagnosis", "readmission", "length_of_stay"]):
        patient_score += 0.45
        patient_reasons.append("age/gender/diagnosis/readmission/LOS-like columns detected")
    if any(has_alias(field) for field in ["admissions", "date"]):
        patient_score += 0.2
        patient_reasons.append("encounter or admission-style identifiers/dates detected")
    if any(has_alias(field) for field in ["provider_id", "hospital_name"]):
        measure_score += 0.25
        measure_reasons.append("provider or hospital identifier columns detected")
    if any(has_alias(field) for field in ["measure_name", "measure_id", "score", "denominator", "compared_to_national"]):
        measure_score += 0.55
        measure_reasons.append("quality measure reporting columns detected")
    if any(has_alias(field) for field in ["date"]):
        measure_score += 0.1
        measure_reasons.append("reporting date columns detected")

    if len(normalized) > 0:
        if any(normalize_column_name(column).endswith("id") for column in columns):
            patient_score += 0.05
            measure_score += 0.05
        if len(normalized.columns) >= 8:
            patient_score += 0.05
            measure_score += 0.05

    scores = {
        "patient_level_clinical": min(patient_score, 1.0),
        "hospital_measure_reporting": min(measure_score, 1.0),
        "generic_tabular": 0.35,
    }
    top_mode = max(scores, key=scores.get)
    top_score = scores[top_mode]
    if top_score < 0.55:
        top_mode = "generic_tabular"
        top_score = max(0.55, scores["generic_tabular"])
        reason = "No healthcare-specific pattern was strong enough, so the dataset is treated as generic tabular data."
    elif top_mode == "patient_level_clinical":
        reason = "; ".join(patient_reasons) or "Patient-level clinical indicators were strongest."
    else:
        reason = "; ".join(measure_reasons) or "Hospital quality reporting indicators were strongest."

    return {
        "dataset_mode": top_mode,
        "mode_confidence": round(top_score, 3),
        "mode_reason": reason,
        "mode_scores": scores,
    }


def _score_column_for_field(column_name: str, series: pd.Series, field: str, mode: str) -> dict[str, object]:
    name_score, name_reason = _name_score(column_name, field)
    value_score, value_reason = _value_score(series, field)
    mode_bonus, mode_reason = _mode_bonus(field, mode)
    confidence = round(min((0.62 * name_score) + (0.30 * value_score) + mode_bonus, 0.99), 3)

    reasons = []
    if name_score >= FUZZY_THRESHOLD:
        reasons.append(name_reason)
    if value_score > 0:
        reasons.append(value_reason)
    if mode_bonus > 0:
        reasons.append(mode_reason)
    if not reasons:
        reasons.append("no strong multi-signal evidence")

    return {
        "field": field,
        "column": column_name,
        "confidence": confidence,
        "reason": " + ".join(reasons),
        "name_score": round(name_score, 3),
        "value_score": round(value_score, 3),
        "mode_bonus": round(mode_bonus, 3),
    }


def _infer_fields(normalized: pd.DataFrame, mode: str, threshold: float) -> tuple[dict[str, str], pd.DataFrame, list[str], list[str]]:
    candidates: list[dict[str, object]] = []
    for field in EXPECTED_FIELDS:
        for column in normalized.columns:
            candidates.append(_score_column_for_field(column, normalized[column], field, mode))

    candidates_df = pd.DataFrame(candidates).sort_values(["confidence", "name_score", "value_score"], ascending=[False, False, False]).reset_index(drop=True)
    matched_schema: dict[str, str] = {}
    used_columns: set[str] = set()
    used_fields: set[str] = set()
    chosen_rows: list[dict[str, object]] = []

    for _, row in candidates_df.iterrows():
        if float(row["confidence"]) < threshold:
            break
        field = str(row["field"])
        column_name = str(row["column"])
        if field in used_fields or column_name in used_columns:
            continue
        matched_schema[field] = column_name
        used_fields.add(field)
        used_columns.add(column_name)
        chosen_rows.append({
            "field": field,
            "matched_column": column_name,
            "confidence_score": float(row["confidence"]),
            "similarity_score": float(row["confidence"]),
            "reason": row["reason"],
            "match_type": "inferred",
            "name_score": float(row["name_score"]),
            "value_score": float(row["value_score"]),
            "mode_bonus": float(row["mode_bonus"]),
        })

    unmatched_columns = [column for column in normalized.columns if column not in used_columns]
    missing_fields = [field for field in EXPECTED_FIELDS if field not in matched_schema]
    return matched_schema, pd.DataFrame(chosen_rows), unmatched_columns, missing_fields


def detect_schema(data: pd.DataFrame, threshold: float = FUZZY_THRESHOLD) -> dict[str, object]:
    normalized = normalize_columns(data)
    mode_info = _classify_dataset_mode(normalized)
    matched_schema, match_details, unmatched_columns, missing_fields = _infer_fields(normalized, mode_info["dataset_mode"], threshold)
    return {
        "normalized_data": normalized,
        "matched_schema": matched_schema,
        "unmatched_columns": unmatched_columns,
        "missing_fields": missing_fields,
        "match_details": match_details,
        "threshold": threshold,
        "dataset_mode": mode_info["dataset_mode"],
        "mode_confidence": mode_info["mode_confidence"],
        "mode_reason": mode_info["mode_reason"],
        "mode_scores": mode_info["mode_scores"],
    }


def apply_detected_schema(data: pd.DataFrame, matched_schema: dict[str, str]) -> pd.DataFrame:
    normalized = normalize_columns(data)
    rename_map = {source_column: target_field for target_field, source_column in matched_schema.items() if source_column in normalized.columns}
    return normalized.rename(columns=rename_map)


def schema_coverage_percent(matched_schema: dict[str, str]) -> float:
    return (len(matched_schema) / len(EXPECTED_FIELDS)) if EXPECTED_FIELDS else 0.0


def dataset_capabilities(schema_input: dict[str, object] | dict[str, str]) -> dict[str, bool]:
    if "matched_schema" in schema_input:
        matched_schema = schema_input["matched_schema"]
        dataset_mode = schema_input.get("dataset_mode", "generic_tabular")
        mode_confidence = float(schema_input.get("mode_confidence", 0.0))
    else:
        matched_schema = schema_input
        dataset_mode = "patient_level_clinical"
        mode_confidence = 1.0
    available = set(matched_schema.keys())
    patient_healthcare = dataset_mode == "patient_level_clinical" and mode_confidence >= 0.55
    hospital_measure = dataset_mode == "hospital_measure_reporting" and mode_confidence >= 0.55
    return {
        "patient_level_mode": patient_healthcare,
        "hospital_measure_mode": hospital_measure,
        "generic_mode": dataset_mode == "generic_tabular",
        "kpi_analysis": patient_healthcare and all(field in available for field in REQUIRED_FOR_KPI),
        "cohort_analysis": patient_healthcare and all(field in available for field in REQUIRED_FOR_COHORT),
        "cost_analysis": patient_healthcare and all(field in available for field in REQUIRED_FOR_COST),
        "machine_learning": patient_healthcare and all(field in available for field in REQUIRED_FOR_ML),
    }


def build_mapping_table(matched_schema: dict[str, str], match_details: pd.DataFrame | None = None) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    detail_lookup = {}
    if match_details is not None and not match_details.empty:
        detail_lookup = match_details.set_index("field").to_dict("index")
    for field in EXPECTED_FIELDS:
        detail = detail_lookup.get(field, {})
        rows.append({
            "field_label": FIELD_LABELS.get(field, field.replace("_", " ").title()),
            "internal_field": field,
            "mapped_column": matched_schema.get(field, "Not mapped"),
            "match_type": detail.get("match_type", "missing"),
            "similarity_score": detail.get("similarity_score", 0.0),
            "confidence_score": detail.get("confidence_score", 0.0),
            "reason": detail.get("reason", "No strong mapping signal detected."),
        })
    return pd.DataFrame(rows)
