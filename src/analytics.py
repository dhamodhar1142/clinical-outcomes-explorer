from __future__ import annotations

import io

import pandas as pd

from src.metrics import (
    add_derived_buckets,
    binary_rate,
    calculate_average_cost,
    calculate_average_cost_per_day,
    calculate_average_length_of_stay,
    calculate_readmission_count,
    calculate_readmission_rate,
    calculate_total_admissions,
    fill_missing_category,
    get_key_metrics,
    get_risk_segment_summary,
    resolve_column,
    to_numeric,
)
from src.schema_detection import dataset_capabilities, schema_coverage_percent


COHORT_FIELDS = {
    "age_group": "Age Group",
    "diagnosis": "Diagnosis",
    "department": "Department",
    "gender": "Gender",
    "comorbidity_bucket": "Comorbidity Bucket",
}

PERFORMANCE_FLAG_ORDER = ["High Concern", "Needs Review", "Stable"]


def build_cohort_summary(data: pd.DataFrame, cohort_key: str) -> pd.DataFrame:
    enriched = add_derived_buckets(data)
    cohort_column = cohort_key if cohort_key in enriched.columns else resolve_column(enriched, cohort_key)
    readmission_column = resolve_column(enriched, "readmission")
    cost_column = resolve_column(enriched, "cost")
    los_column = resolve_column(enriched, "length_of_stay")
    admission_column = resolve_column(enriched, "admissions", required=False)

    analysis_columns = [cohort_column, readmission_column, cost_column, los_column]
    if admission_column:
        analysis_columns.append(admission_column)

    analysis = enriched[analysis_columns].copy()
    analysis[cohort_column] = fill_missing_category(analysis[cohort_column], "Unknown")
    analysis[cost_column] = to_numeric(analysis[cost_column])
    analysis[los_column] = to_numeric(analysis[los_column])

    if analysis.empty:
        return pd.DataFrame(columns=[cohort_column, "total_admissions", "readmission_rate", "average_cost", "average_length_of_stay"])

    grouped = analysis.groupby(cohort_column, dropna=False, observed=False)
    summary = pd.DataFrame({cohort_column: grouped.size().index})
    summary["total_admissions"] = grouped[admission_column].nunique().values if admission_column else grouped.size().values
    summary["readmission_rate"] = grouped[readmission_column].apply(binary_rate).values
    summary["average_cost"] = grouped[cost_column].mean().values
    summary["average_length_of_stay"] = grouped[los_column].mean().values
    return summary.sort_values(["readmission_rate", "total_admissions"], ascending=[False, False]).reset_index(drop=True)


def get_all_cohort_summaries(data: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {cohort_key: build_cohort_summary(data, cohort_key) for cohort_key in COHORT_FIELDS}


def get_top_diagnosis_groups_by_total_cost(data: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    diagnosis_column = resolve_column(data, "diagnosis")
    cost_column = resolve_column(data, "cost")
    analysis = data[[diagnosis_column, cost_column]].copy()
    analysis[diagnosis_column] = fill_missing_category(analysis[diagnosis_column], "Unknown")
    analysis[cost_column] = to_numeric(analysis[cost_column])
    analysis = analysis.dropna(subset=[cost_column])
    if analysis.empty:
        return pd.DataFrame(columns=[diagnosis_column, "total_cost"])
    return analysis.groupby(diagnosis_column, dropna=False)[cost_column].sum().reset_index(name="total_cost").sort_values("total_cost", ascending=False).head(top_n)


def get_cost_per_day_by_diagnosis(data: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    diagnosis_column = resolve_column(data, "diagnosis")
    cost_column = resolve_column(data, "cost")
    los_column = resolve_column(data, "length_of_stay")
    analysis = data[[diagnosis_column, cost_column, los_column]].copy()
    analysis[diagnosis_column] = fill_missing_category(analysis[diagnosis_column], "Unknown")
    analysis[cost_column] = to_numeric(analysis[cost_column])
    analysis[los_column] = to_numeric(analysis[los_column])
    analysis = analysis[(analysis[los_column] > 0) & analysis[cost_column].notna()]
    if analysis.empty:
        return pd.DataFrame(columns=[diagnosis_column, "cost_per_day"])
    analysis["cost_per_day"] = analysis[cost_column] / analysis[los_column]
    return analysis.groupby(diagnosis_column, dropna=False)["cost_per_day"].mean().reset_index().sort_values("cost_per_day", ascending=False).head(top_n)


def get_highest_cost_departments(data: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    department_column = resolve_column(data, "department")
    cost_column = resolve_column(data, "cost")
    analysis = data[[department_column, cost_column]].copy()
    analysis[department_column] = fill_missing_category(analysis[department_column], "Unknown")
    analysis[cost_column] = to_numeric(analysis[cost_column])
    analysis = analysis.dropna(subset=[cost_column])
    if analysis.empty:
        return pd.DataFrame(columns=[department_column, "total_cost"])
    return analysis.groupby(department_column, dropna=False)[cost_column].sum().reset_index(name="total_cost").sort_values("total_cost", ascending=False).head(top_n)


def calculate_average_readmission_rate(data: pd.DataFrame) -> float:
    return float(binary_rate(data[resolve_column(data, "readmission")]))


def build_department_scorecard(data: pd.DataFrame, scored_data: pd.DataFrame | None = None) -> pd.DataFrame:
    department_column = resolve_column(data, "department")
    readmission_column = resolve_column(data, "readmission")
    cost_column = resolve_column(data, "cost")
    los_column = resolve_column(data, "length_of_stay")
    admission_column = resolve_column(data, "admissions", required=False)

    analysis_columns = [department_column, readmission_column, cost_column, los_column]
    if admission_column:
        analysis_columns.append(admission_column)

    analysis = data[analysis_columns].copy()
    analysis[department_column] = fill_missing_category(analysis[department_column], "Unknown")
    analysis[cost_column] = to_numeric(analysis[cost_column])
    analysis[los_column] = to_numeric(analysis[los_column])
    if analysis.empty:
        return pd.DataFrame(columns=[department_column, "total_admissions", "average_length_of_stay", "readmission_rate", "average_cost", "average_predicted_readmission_risk", "performance_flag"])

    grouped = analysis.groupby(department_column, dropna=False)
    scorecard = pd.DataFrame({department_column: grouped.size().index})
    scorecard["total_admissions"] = grouped[admission_column].nunique().values if admission_column else grouped.size().values
    scorecard["average_length_of_stay"] = grouped[los_column].mean().values
    scorecard["readmission_rate"] = grouped[readmission_column].apply(binary_rate).values
    scorecard["average_cost"] = grouped[cost_column].mean().values

    if scored_data is not None and not scored_data.empty:
        risk_department_column = resolve_column(scored_data, "department")
        risk_frame = scored_data.copy()
        risk_frame[risk_department_column] = fill_missing_category(risk_frame[risk_department_column], "Unknown")
        risk_summary = risk_frame.groupby(risk_department_column, dropna=False)["predicted_readmission_risk"].mean().reset_index(name="average_predicted_readmission_risk")
        scorecard = scorecard.merge(risk_summary, left_on=department_column, right_on=risk_department_column, how="left")
        if risk_department_column != department_column:
            scorecard = scorecard.drop(columns=[risk_department_column])
    else:
        scorecard["average_predicted_readmission_risk"] = pd.NA

    overall_readmission_rate = calculate_average_readmission_rate(data)
    overall_average_cost = calculate_average_cost(data)

    def _performance_flag(row: pd.Series) -> str:
        high_readmission = pd.notna(row["readmission_rate"]) and row["readmission_rate"] > overall_readmission_rate
        high_cost = pd.notna(row["average_cost"]) and row["average_cost"] > overall_average_cost
        if high_readmission and high_cost:
            return "High Concern"
        if high_readmission or high_cost:
            return "Needs Review"
        return "Stable"

    scorecard["performance_flag"] = scorecard.apply(_performance_flag, axis=1)
    scorecard["performance_flag"] = pd.Categorical(scorecard["performance_flag"], categories=PERFORMANCE_FLAG_ORDER, ordered=True)
    return scorecard.sort_values(["performance_flag", "readmission_rate", "average_cost"], ascending=[True, False, False]).reset_index(drop=True)


def simulate_intervention(data: pd.DataFrame, los_reduction_days: float, readmission_reduction_pct: float) -> dict[str, float]:
    metrics = get_key_metrics(data)
    projected_cost_savings = los_reduction_days * metrics["average_cost_per_day"] * metrics["total_admissions"]
    projected_readmissions_avoided = readmission_reduction_pct * metrics["readmission_count"]
    return {
        "current_average_length_of_stay": metrics["average_length_of_stay"],
        "projected_average_length_of_stay": max(metrics["average_length_of_stay"] - los_reduction_days, 0),
        "current_readmission_count": metrics["readmission_count"],
        "projected_readmission_count": max(metrics["readmission_count"] - projected_readmissions_avoided, 0),
        "projected_cost_savings": projected_cost_savings,
        "projected_readmissions_avoided": projected_readmissions_avoided,
    }


def compare_scenarios(data: pd.DataFrame, scenario_inputs: dict[str, dict[str, float]]) -> pd.DataFrame:
    rows = []
    for scenario_name, settings in scenario_inputs.items():
        result = simulate_intervention(data, settings["los_reduction"], settings["readmission_reduction"])
        rows.append({
            "scenario": scenario_name,
            "projected_cost_savings": result["projected_cost_savings"],
            "projected_readmissions_avoided": result["projected_readmissions_avoided"],
            "projected_average_length_of_stay": result["projected_average_length_of_stay"],
        })
    return pd.DataFrame(rows)


def build_executive_summary(data: pd.DataFrame, scored_data: pd.DataFrame | None = None) -> dict[str, object]:
    metrics = get_key_metrics(data)
    diagnosis_column = resolve_column(data, "diagnosis")
    department_column = resolve_column(data, "department")
    cost_column = resolve_column(data, "cost")
    readmission_column = resolve_column(data, "readmission")
    executive_frame = data.copy()
    executive_frame[department_column] = fill_missing_category(executive_frame[department_column], "Unknown")
    executive_frame[diagnosis_column] = fill_missing_category(executive_frame[diagnosis_column], "Unknown")

    highest_cost_department = executive_frame.assign(_cost=to_numeric(executive_frame[cost_column])).dropna(subset=["_cost"]).groupby(department_column, dropna=False)["_cost"].mean().sort_values(ascending=False)
    highest_risk_diagnosis = executive_frame.groupby(diagnosis_column, dropna=False)[readmission_column].apply(binary_rate).sort_values(ascending=False)

    top_cost_department = highest_cost_department.index[0] if not highest_cost_department.empty else "No valid cost data"
    top_risk_diagnosis = highest_risk_diagnosis.index[0] if not highest_risk_diagnosis.empty else "No valid readmission data"

    high_risk_patients = 0
    critical_risk_patients = 0
    preventable_readmissions = metrics["readmission_count"] * 0.05
    if scored_data is not None and not scored_data.empty:
        risk_summary = get_risk_segment_summary(scored_data).set_index("risk_segment")
        high_risk_patients = int(risk_summary.loc["High Risk", "patient_count"])
        critical_risk_patients = int(risk_summary.loc["Critical Risk", "patient_count"])
        preventable_readmissions = (high_risk_patients + critical_risk_patients) * metrics["readmission_rate"]

    savings_opportunity = simulate_intervention(data, los_reduction_days=1, readmission_reduction_pct=0.0)["projected_cost_savings"]
    narrative = (
        f"This filtered population includes {metrics['total_admissions']:,} admissions with a current readmission rate of {metrics['readmission_rate']:.1%}. "
        f"{top_risk_diagnosis} shows the highest observed readmission risk, while {top_cost_department} carries the highest average cost burden. "
        f"The current cost base is ${metrics['total_cost']:,.0f}, highlighting a practical opportunity for operational savings through targeted readmission and length-of-stay improvement efforts."
    )
    return {
        "cards": {
            "total_admissions": metrics["total_admissions"],
            "total_estimated_readmissions": metrics["readmission_count"],
            "high_risk_patients": high_risk_patients + critical_risk_patients,
            "total_cost": metrics["total_cost"],
            "estimated_preventable_readmissions": preventable_readmissions,
            "estimated_savings_opportunity": savings_opportunity,
        },
        "narrative": narrative,
    }


def generate_key_insights(data: pd.DataFrame, scored_data: pd.DataFrame | None = None) -> list[str]:
    department_column = resolve_column(data, "department")
    diagnosis_column = resolve_column(data, "diagnosis")
    cost_column = resolve_column(data, "cost")
    los_column = resolve_column(data, "length_of_stay")
    readmission_column = resolve_column(data, "readmission")
    insight_frame = data.copy()
    insight_frame[department_column] = fill_missing_category(insight_frame[department_column], "Unknown")
    insight_frame[diagnosis_column] = fill_missing_category(insight_frame[diagnosis_column], "Unknown")

    cost_by_department = insight_frame.assign(_cost=to_numeric(insight_frame[cost_column])).dropna(subset=["_cost"]).groupby(department_column, dropna=False)["_cost"].mean().sort_values(ascending=False)
    readmission_by_diagnosis = insight_frame.groupby(diagnosis_column, dropna=False)[readmission_column].apply(binary_rate).sort_values(ascending=False)
    los_by_department = insight_frame.assign(_los=to_numeric(insight_frame[los_column])).dropna(subset=["_los"]).groupby(department_column, dropna=False)["_los"].mean().sort_values(ascending=False)
    cost_per_day = get_cost_per_day_by_diagnosis(data)

    if cost_by_department.empty or readmission_by_diagnosis.empty or los_by_department.empty:
        raise ValueError("not enough valid cost, length-of-stay, or readmission data to summarize the current filter selection")

    cohort_summaries = get_all_cohort_summaries(data)
    top_cohort_label = "Not available"
    top_cohort_value = 0.0
    for cohort_key, summary in cohort_summaries.items():
        if summary.empty:
            continue
        top_row = summary.sort_values("readmission_rate", ascending=False).iloc[0]
        if pd.notna(top_row["readmission_rate"]) and float(top_row["readmission_rate"]) > top_cohort_value:
            top_cohort_value = float(top_row["readmission_rate"])
            top_cohort_label = f"{COHORT_FIELDS[cohort_key]}: {top_row.iloc[0]}"

    risk_text = "Model-based High Risk and Critical Risk share is unavailable because the logistic regression model could not be trained for the current filter selection."
    if scored_data is not None and not scored_data.empty:
        risk_summary = get_risk_segment_summary(scored_data).set_index("risk_segment")
        high_and_critical_pct = risk_summary.loc[["High Risk", "Critical Risk"], "percentage"].sum()
        risk_text = f"{high_and_critical_pct:.1%} of scored patients are currently in the High Risk or Critical Risk segments."

    insights = [
        f"{cost_by_department.index[0]} has the highest average cost at ${cost_by_department.iloc[0]:,.2f} per admission.",
        f"{readmission_by_diagnosis.index[0]} has the highest readmission rate at {readmission_by_diagnosis.iloc[0]:.1%}.",
        f"The highest-risk cohort is {top_cohort_label} with a readmission rate of {top_cohort_value:.1%}.",
        f"{cost_per_day.iloc[0, 0]} has the highest average cost per day at ${cost_per_day.iloc[0]['cost_per_day']:,.2f}." if not cost_per_day.empty else "Cost-per-day analysis is unavailable because no valid positive length-of-stay records remain after filtering.",
        risk_text,
        f"{los_by_department.index[0]} has the longest average length of stay at {los_by_department.iloc[0]:.1f} days.",
        f"The filtered cohort averages ${calculate_average_cost_per_day(data):,.2f} in cost per inpatient day.",
    ]
    return insights[:8]


def assess_data_quality(data: pd.DataFrame) -> dict[str, object]:
    missing_values = data.isna().sum().reset_index()
    missing_values.columns = ["column", "missing_values"]
    admission_column = resolve_column(data, "admissions", required=False)
    duplicate_admission_count = int(data[admission_column].duplicated().sum()) if admission_column else int(data.duplicated().sum())
    los_column = resolve_column(data, "length_of_stay")
    cost_column = resolve_column(data, "cost")
    invalid_los_count = int((to_numeric(data[los_column]).fillna(0) <= 0).sum())
    invalid_cost_count = int((to_numeric(data[cost_column]).fillna(0) <= 0).sum())
    row_count = max(len(data), 1)
    cell_count = max(len(data.columns) * len(data), 1)
    missing_ratio = missing_values["missing_values"].sum() / cell_count
    duplicate_ratio = duplicate_admission_count / row_count
    invalid_los_ratio = invalid_los_count / row_count
    invalid_cost_ratio = invalid_cost_count / row_count
    score = max(int(round(100 - min(30, missing_ratio * 100) - min(25, duplicate_ratio * 100) - min(25, invalid_los_ratio * 100) - min(20, invalid_cost_ratio * 100))), 0)
    interpretation = "Excellent" if score >= 90 else "Good" if score >= 75 else "Needs Review"
    return {
        "missing_values": missing_values.sort_values(["missing_values", "column"], ascending=[False, True]),
        "duplicate_admission_count": duplicate_admission_count,
        "invalid_los_count": invalid_los_count,
        "invalid_cost_count": invalid_cost_count,
        "quality_score": score,
        "interpretation": interpretation,
    }


def build_auto_analysis_summary(data: pd.DataFrame, matched_schema: dict[str, str]) -> dict[str, object]:
    capabilities = dataset_capabilities(matched_schema)
    coverage = schema_coverage_percent(matched_schema)
    recommendations: list[str] = []
    if capabilities["kpi_analysis"] and capabilities["cohort_analysis"] and capabilities["cost_analysis"]:
        recommendations.append("Sufficient for dashboard analytics.")
    if not capabilities["machine_learning"]:
        recommendations.append("Missing readmission or modeling fields for predictive modeling.")
    if not capabilities["cost_analysis"]:
        recommendations.append("Missing cost or length-of-stay fields for cost driver analysis.")
    if not recommendations:
        recommendations.append("Schema mapping looks strong for interactive hospital analytics.")
    return {
        "row_count": len(data),
        "column_count": len(data.columns),
        "coverage_percent": coverage,
        "capabilities": capabilities,
        "recommendations": recommendations,
    }


def build_monthly_trends(data: pd.DataFrame) -> pd.DataFrame:
    date_column = resolve_column(data, "date", required=False)
    if not date_column:
        return pd.DataFrame(columns=["month", "admissions", "average_cost", "readmission_rate", "average_length_of_stay"])

    trend_frame = data.copy()
    trend_frame[date_column] = pd.to_datetime(trend_frame[date_column], errors="coerce")
    trend_frame = trend_frame.dropna(subset=[date_column])
    if trend_frame.empty:
        return pd.DataFrame(columns=["month", "admissions", "average_cost", "readmission_rate", "average_length_of_stay"])

    trend_frame["month"] = trend_frame[date_column].dt.to_period("M").dt.to_timestamp()
    admission_column = resolve_column(trend_frame, "admissions", required=False)
    cost_column = resolve_column(trend_frame, "cost")
    los_column = resolve_column(trend_frame, "length_of_stay")
    readmission_column = resolve_column(trend_frame, "readmission")
    grouped = trend_frame.groupby("month", dropna=False)
    monthly = pd.DataFrame({"month": grouped.size().index})
    monthly["admissions"] = grouped[admission_column].nunique().values if admission_column else grouped.size().values
    monthly["average_cost"] = grouped[cost_column].apply(lambda s: to_numeric(s).mean()).values
    monthly["readmission_rate"] = grouped[readmission_column].apply(binary_rate).values
    monthly["average_length_of_stay"] = grouped[los_column].apply(lambda s: to_numeric(s).mean()).values
    return monthly.sort_values("month")


def build_benchmarking_summary(filtered_data: pd.DataFrame, overall_data: pd.DataFrame, scored_filtered: pd.DataFrame | None = None, scored_overall: pd.DataFrame | None = None) -> pd.DataFrame:
    rows = []
    comparisons = {
        "Readmission Rate": (calculate_readmission_rate(filtered_data), calculate_readmission_rate(overall_data)),
        "Average Cost": (calculate_average_cost(filtered_data), calculate_average_cost(overall_data)),
        "Average Length of Stay": (calculate_average_length_of_stay(filtered_data), calculate_average_length_of_stay(overall_data)),
    }
    if scored_filtered is not None and not scored_filtered.empty and scored_overall is not None and not scored_overall.empty:
        comparisons["Average Predicted Risk"] = (float(scored_filtered["predicted_readmission_risk"].mean()), float(scored_overall["predicted_readmission_risk"].mean()))
    for metric_name, (filtered_value, overall_value) in comparisons.items():
        direction = "Above hospital average" if filtered_value > overall_value else "Below hospital average" if filtered_value < overall_value else "In line with hospital average"
        rows.append({"metric": metric_name, "filtered_selection": filtered_value, "overall_dataset": overall_value, "benchmark_status": direction})
    return pd.DataFrame(rows)


def generate_operational_alerts(filtered_data: pd.DataFrame, overall_data: pd.DataFrame, data_quality: dict[str, object], scored_filtered: pd.DataFrame | None = None, scored_overall: pd.DataFrame | None = None) -> list[dict[str, str]]:
    alerts: list[dict[str, str]] = []
    filtered_metrics = get_key_metrics(filtered_data)
    overall_metrics = get_key_metrics(overall_data)
    if filtered_metrics["readmission_rate"] > overall_metrics["readmission_rate"] * 1.1:
        alerts.append({"level": "error", "title": "High Readmission Alert", "message": "The filtered selection is running meaningfully above the overall dataset readmission rate."})
    if filtered_metrics["average_cost"] > overall_metrics["average_cost"] * 1.1:
        alerts.append({"level": "warning", "title": "High Cost Alert", "message": "Average cost per admission is above the overall dataset average."})
    if filtered_metrics["average_length_of_stay"] > overall_metrics["average_length_of_stay"] * 1.1:
        alerts.append({"level": "warning", "title": "Extended Length of Stay Alert", "message": "Average length of stay is above the overall dataset average."})
    if data_quality["quality_score"] < 75:
        alerts.append({"level": "info", "title": "Data Quality Alert", "message": "Data quality checks suggest the filtered dataset needs review before using results for strong conclusions."})
    if scored_filtered is not None and not scored_filtered.empty and scored_overall is not None and not scored_overall.empty:
        if scored_filtered["predicted_readmission_risk"].mean() > scored_overall["predicted_readmission_risk"].mean() * 1.1:
            alerts.append({"level": "error", "title": "Elevated Predicted Risk Alert", "message": "Model-estimated readmission risk is above the overall dataset average."})
    return alerts


def build_custom_cohort(data: pd.DataFrame, rules: dict[str, object]) -> pd.DataFrame:
    cohort = data.copy()
    age_column = resolve_column(cohort, "age", required=False)
    gender_column = resolve_column(cohort, "gender", required=False)
    diagnosis_column = resolve_column(cohort, "diagnosis", required=False)
    department_column = resolve_column(cohort, "department", required=False)
    comorbidity_column = resolve_column(cohort, "comorbidity_score", required=False)

    if age_column and rules.get("min_age") is not None:
        cohort[age_column] = to_numeric(cohort[age_column])
        cohort = cohort[cohort[age_column] >= rules["min_age"]]
    if age_column and rules.get("max_age") is not None:
        cohort = cohort[cohort[age_column] <= rules["max_age"]]
    if gender_column and rules.get("gender"):
        cohort = cohort[cohort[gender_column].astype(str).isin(rules["gender"])]
    if diagnosis_column and rules.get("diagnosis"):
        cohort = cohort[cohort[diagnosis_column].astype(str).isin(rules["diagnosis"])]
    if department_column and rules.get("department"):
        cohort = cohort[cohort[department_column].astype(str).isin(rules["department"])]
    if comorbidity_column and rules.get("comorbidity_min") is not None:
        cohort[comorbidity_column] = to_numeric(cohort[comorbidity_column])
        cohort = cohort[cohort[comorbidity_column] >= rules["comorbidity_min"]]
    return cohort


def build_risk_drilldown(scored_data: pd.DataFrame, selected_segment: str) -> dict[str, object]:
    segment_data = scored_data[scored_data["risk_segment"] == selected_segment].copy()
    if segment_data.empty:
        return {"segment_data": segment_data, "patient_count": 0, "average_cost": 0.0, "average_length_of_stay": 0.0, "top_diagnoses": pd.DataFrame(), "top_departments": pd.DataFrame()}
    diagnosis_column = resolve_column(segment_data, "diagnosis", required=False)
    department_column = resolve_column(segment_data, "department", required=False)
    top_diagnoses = segment_data[diagnosis_column].value_counts(dropna=False).reset_index() if diagnosis_column else pd.DataFrame()
    top_departments = segment_data[department_column].value_counts(dropna=False).reset_index() if department_column else pd.DataFrame()
    if not top_diagnoses.empty:
        top_diagnoses.columns = [diagnosis_column, "patient_count"]
    if not top_departments.empty:
        top_departments.columns = [department_column, "patient_count"]
    return {
        "segment_data": segment_data,
        "patient_count": len(segment_data),
        "average_cost": calculate_average_cost(segment_data) if "cost" in segment_data.columns else 0.0,
        "average_length_of_stay": calculate_average_length_of_stay(segment_data) if "length_of_stay" in segment_data.columns else 0.0,
        "top_diagnoses": top_diagnoses.head(10),
        "top_departments": top_departments.head(10),
    }


def build_model_evaluation(scored_data: pd.DataFrame, threshold: float) -> dict[str, object]:
    if scored_data.empty or "actual_readmission" not in scored_data.columns:
        return {"roc_auc": None, "precision": None, "recall": None, "confusion_matrix": pd.DataFrame()}
    from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_auc_score
    actual = scored_data["actual_readmission"].astype(int)
    predicted = (scored_data["predicted_readmission_risk"] >= threshold).astype(int)
    matrix = confusion_matrix(actual, predicted, labels=[0, 1])
    confusion = pd.DataFrame(matrix, index=["Actual Negative", "Actual Positive"], columns=["Predicted Negative", "Predicted Positive"])
    return {
        "roc_auc": float(roc_auc_score(actual, scored_data["predicted_readmission_risk"])),
        "precision": float(precision_score(actual, predicted, zero_division=0)),
        "recall": float(recall_score(actual, predicted, zero_division=0)),
        "confusion_matrix": confusion,
    }



def build_executive_report(data: pd.DataFrame, schema_info: dict[str, object], selected_filters: dict[str, object], insights: list[str], alerts: list[dict[str, str]], metrics: dict[str, float | int] | None = None, executive_summary: dict[str, object] | None = None) -> bytes:
    capabilities = dataset_capabilities(schema_info)
    coverage = schema_coverage_percent(schema_info.get("matched_schema", {}))
    metrics = metrics or (get_key_metrics(data) if {"readmission", "length_of_stay", "cost"}.issubset(data.columns) else {})
    executive_summary = executive_summary or {}
    missing_total = int(data.isna().sum().sum())
    duplicate_rows = int(data.duplicated().sum())
    top_missing = data.isna().sum().sort_values(ascending=False)
    top_missing = top_missing[top_missing > 0].head(3)

    next_actions: list[str] = []
    if not capabilities.get("kpi_analysis"):
        next_actions.append("Map readmission, cost, and length of stay to unlock the KPI and executive summary views.")
    if not capabilities.get("cost_analysis"):
        next_actions.append("Add diagnosis and department fields to improve cost and utilization analysis.")
    if not capabilities.get("machine_learning"):
        next_actions.append("Map age, comorbidity score, and prior admissions if you want to explore predictive risk modeling.")
    if not next_actions:
        next_actions.append("Review the Executive Overview and Cost & Performance sections to prioritize operational opportunities.")
        next_actions.append("Use Export Center outputs to share a filtered summary with instructors, peers, or stakeholders.")

    buffer = io.StringIO()
    buffer.write("Clinical Outcomes Explorer Executive Report\n")
    buffer.write("=" * 42 + "\n\n")
    buffer.write("Dataset Overview\n")
    buffer.write("-" * 16 + "\n")
    buffer.write(f"Rows in scope: {len(data):,}\n")
    buffer.write(f"Columns in scope: {len(data.columns):,}\n")
    if selected_filters:
        buffer.write("Applied filters:\n")
        for key, value in selected_filters.items():
            buffer.write(f"- {key}: {value}\n")
    else:
        buffer.write("Applied filters: none\n")
    buffer.write("\n")

    buffer.write("Schema and Readiness Summary\n")
    buffer.write("-" * 28 + "\n")
    buffer.write(f"Detected dataset mode: {str(schema_info.get('dataset_mode', 'generic_tabular')).replace('_', ' ').title()}\n")
    buffer.write(f"Schema coverage: {coverage:.0%}\n")
    buffer.write(f"Mapped fields: {len(schema_info.get('matched_schema', {}))}/{len(schema_info.get('matched_schema', {})) + len(schema_info.get('missing_fields', []))}\n")
    buffer.write(f"Readiness note: {schema_info.get('mode_reason', 'Schema readiness details are unavailable.')}\n")
    if schema_info.get("missing_fields"):
        missing_labels = [field.replace("_", " ").title() for field in schema_info.get("missing_fields", [])[:6]]
        buffer.write(f"Fields that would unlock more features: {', '.join(missing_labels)}\n")
    buffer.write("\n")

    buffer.write("Data Quality Findings\n")
    buffer.write("-" * 21 + "\n")
    buffer.write(f"Total missing values: {missing_total:,}\n")
    buffer.write(f"Duplicate rows: {duplicate_rows:,}\n")
    if not top_missing.empty:
        buffer.write("Top columns with missing data:\n")
        for column_name, count in top_missing.items():
            buffer.write(f"- {column_name}: {int(count):,}\n")
    else:
        buffer.write("No missing-value hotspots were detected in the current filtered data.\n")
    buffer.write("\n")

    buffer.write("KPI Summary\n")
    buffer.write("-" * 11 + "\n")
    if metrics:
        if "total_admissions" in metrics:
            buffer.write(f"Admissions in scope: {int(metrics['total_admissions']):,}\n")
        if "readmission_rate" in metrics:
            buffer.write(f"Readmission rate: {float(metrics['readmission_rate']):.1%}\n")
        if "average_length_of_stay" in metrics:
            buffer.write(f"Average length of stay: {float(metrics['average_length_of_stay']):.1f} days\n")
        if "average_cost" in metrics:
            buffer.write(f"Average cost per admission: ${float(metrics['average_cost']):,.2f}\n")
        if "total_cost" in metrics:
            buffer.write(f"Total cost in scope: ${float(metrics['total_cost']):,.0f}\n")
    else:
        buffer.write("Healthcare KPI calculations are limited for this dataset because the required mapped fields are not yet available.\n")
    if executive_summary and executive_summary.get("narrative"):
        buffer.write("\n")
        buffer.write(f"Executive narrative: {executive_summary['narrative']}\n")
    buffer.write("\n")

    buffer.write("Top Findings\n")
    buffer.write("-" * 12 + "\n")
    if insights:
        for insight in insights[:5]:
            buffer.write(f"- {insight}\n")
    elif alerts:
        for alert in alerts[:3]:
            buffer.write(f"- {alert['title']}: {alert['message']}\n")
    else:
        buffer.write("- The current filtered dataset is ready for exploratory review, but no additional healthcare-specific findings were generated.\n")
    buffer.write("\n")

    buffer.write("Recommended Next Actions\n")
    buffer.write("-" * 24 + "\n")
    for action in next_actions[:4]:
        buffer.write(f"- {action}\n")

    return buffer.getvalue().encode("utf-8")

def build_stakeholder_summary(metrics: dict[str, float | int], insights: list[str], alerts: list[dict[str, str]], scenario_table: pd.DataFrame) -> bytes:
    top_cost_driver = insights[0] if insights else "Top cost driver unavailable."
    highest_risk_cohort = next((insight for insight in insights if "highest-risk cohort" in insight.lower()), "Highest-risk cohort unavailable.")
    biggest_alert = alerts[0]["title"] if alerts else "No active alerts"
    best_scenario = scenario_table.sort_values("projected_cost_savings", ascending=False).iloc[0] if not scenario_table.empty else None
    intervention_line = f"{best_scenario['scenario']} offers the largest modeled cost savings at ${best_scenario['projected_cost_savings']:,.0f}." if best_scenario is not None else "Intervention comparison unavailable."
    buffer = io.StringIO()
    buffer.write("Clinical Outcomes Explorer Stakeholder Summary\n")
    buffer.write("=" * 46 + "\n\n")
    buffer.write(f"Admissions in Scope: {metrics['total_admissions']:,}\n")
    buffer.write(f"Readmission Rate: {metrics['readmission_rate']:.1%}\n")
    buffer.write(f"Average Cost per Admission: ${metrics['average_cost']:,.2f}\n")
    buffer.write(f"Average Length of Stay: {metrics['average_length_of_stay']:.1f} days\n\n")
    buffer.write(f"Top Cost Driver: {top_cost_driver}\n")
    buffer.write(f"Highest-Risk Cohort: {highest_risk_cohort}\n")
    buffer.write(f"Biggest Alert: {biggest_alert}\n")
    buffer.write(f"Intervention Opportunity: {intervention_line}\n")
    return buffer.getvalue().encode("utf-8")


def dataframe_to_csv_bytes(data: pd.DataFrame) -> bytes:
    return data.to_csv(index=False).encode("utf-8")


def build_summary_text(metrics: dict[str, float | int], insights: list[str], selected_filters: dict[str, object]) -> bytes:
    buffer = io.StringIO()
    buffer.write("Clinical Outcomes Explorer Summary\n")
    buffer.write("=" * 36 + "\n\n")
    buffer.write("Selected Filters\n")
    for filter_name, value in selected_filters.items():
        buffer.write(f"- {filter_name}: {value}\n")
    buffer.write("\nKPI Values\n")
    buffer.write(f"- Total Admissions: {metrics['total_admissions']:,}\n")
    buffer.write(f"- Readmission Rate: {metrics['readmission_rate']:.1%}\n")
    buffer.write(f"- Estimated Readmissions: {metrics['readmission_count']:,}\n")
    buffer.write(f"- Average Length of Stay: {metrics['average_length_of_stay']:.1f} days\n")
    buffer.write(f"- Average Cost: ${metrics['average_cost']:,.2f}\n")
    buffer.write(f"- Total Cost: ${metrics['total_cost']:,.2f}\n")
    buffer.write("\nKey Insights\n")
    for insight in insights:
        buffer.write(f"- {insight}\n")
    return buffer.getvalue().encode("utf-8")

