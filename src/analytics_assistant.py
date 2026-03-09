from __future__ import annotations

import pandas as pd

from src.analytics import build_cohort_summary, build_department_scorecard, get_cost_per_day_by_diagnosis, get_top_diagnosis_groups_by_total_cost
from src.metrics import calculate_average_length_of_stay, calculate_average_cost, calculate_readmission_rate, get_high_risk_patients, resolve_column, to_numeric


SUPPORTED_PATTERNS = [
    "Which department has the highest cost?",
    "Show top diagnoses by readmission rate",
    "What is the average LOS?",
    "What is the average cost?",
    "Which department has the highest readmission risk?",
    "What is the most expensive diagnosis?",
    "Which cohort has the longest LOS?",
    "Show high risk patients",
]


def _safe_group_mean(data: pd.DataFrame, group_column: str, value_column: str) -> pd.DataFrame:
    grouped = data[[group_column, value_column]].copy()
    grouped[value_column] = to_numeric(grouped[value_column])
    grouped = grouped.dropna(subset=[value_column])
    if grouped.empty:
        return pd.DataFrame(columns=[group_column, value_column])
    return grouped.groupby(group_column, dropna=False)[value_column].mean().reset_index().sort_values(value_column, ascending=False)


def answer_business_question(question: str, filtered_data: pd.DataFrame, scored_data: pd.DataFrame | None = None) -> dict[str, object]:
    question_text = question.strip().lower()
    if not question_text:
        return {"answer": "Type a question about cost, readmissions, length of stay, departments, diagnoses, or patient risk.", "table": None}

    if "department" in question_text and "cost" in question_text and {"department", "cost"}.issubset(filtered_data.columns):
        summary = _safe_group_mean(filtered_data, "department", "cost")
        if summary.empty:
            return {"answer": "No department-level cost summary is available for the current selection.", "table": None}
        top_row = summary.iloc[0]
        return {"answer": f"{top_row['department']} has the highest average cost at ${top_row['cost']:,.2f} per record.", "table": summary.head(10)}

    if "diagnosis" in question_text and "readmission" in question_text and {"diagnosis", "readmission"}.issubset(filtered_data.columns):
        diagnosis_column = resolve_column(filtered_data, "diagnosis")
        readmission_column = resolve_column(filtered_data, "readmission")
        summary = filtered_data.groupby(diagnosis_column)[readmission_column].apply(lambda series: series.astype(str).str.lower().isin(["yes", "true", "1", "y"]).mean()).reset_index(name="readmission_rate")
        if summary.empty:
            return {"answer": "No diagnosis-level readmission summary is available for the current selection.", "table": None}
        top_row = summary.sort_values("readmission_rate", ascending=False).iloc[0]
        return {"answer": f"{top_row[diagnosis_column]} has the highest diagnosis-level readmission rate at {top_row['readmission_rate']:.1%}.", "table": summary.sort_values("readmission_rate", ascending=False).head(10)}

    if "department" in question_text and "readmission" in question_text and {"department", "readmission", "cost", "length_of_stay"}.issubset(filtered_data.columns):
        scorecard = build_department_scorecard(filtered_data, scored_data)
        if scorecard.empty:
            return {"answer": "No department-level readmission summary is available for the current selection.", "table": None}
        top_row = scorecard.sort_values("readmission_rate", ascending=False).iloc[0]
        return {"answer": f"{top_row.iloc[0]} has the highest department readmission rate at {top_row['readmission_rate']:.1%}.", "table": scorecard.head(10)}

    if ("most expensive" in question_text or "highest cost" in question_text) and "diagnosis" in question_text and {"diagnosis", "cost"}.issubset(filtered_data.columns):
        summary = get_top_diagnosis_groups_by_total_cost(filtered_data)
        if summary.empty:
            return {"answer": "No diagnosis-level cost summary is available for the current selection.", "table": None}
        top_row = summary.iloc[0]
        return {"answer": f"{top_row.iloc[0]} is the most expensive diagnosis group with total cost of ${top_row['total_cost']:,.2f}.", "table": summary}

    if "cost per day" in question_text and {"diagnosis", "cost", "length_of_stay"}.issubset(filtered_data.columns):
        summary = get_cost_per_day_by_diagnosis(filtered_data)
        if summary.empty:
            return {"answer": "No cost-per-day analysis is available for the current selection.", "table": None}
        top_row = summary.iloc[0]
        return {"answer": f"{top_row.iloc[0]} has the highest average cost per day at ${top_row['cost_per_day']:,.2f}.", "table": summary}

    if "longest" in question_text and "cohort" in question_text and {"age", "readmission", "cost", "length_of_stay"}.issubset(filtered_data.columns):
        cohort_summary = build_cohort_summary(filtered_data, "age_group")
        if cohort_summary.empty:
            return {"answer": "No age-cohort summary is available for the current selection.", "table": None}
        top_row = cohort_summary.sort_values("average_length_of_stay", ascending=False).iloc[0]
        return {"answer": f"The {top_row.iloc[0]} cohort has the longest average length of stay at {top_row['average_length_of_stay']:.1f} days.", "table": cohort_summary}

    if "high risk" in question_text and scored_data is not None:
        table = get_high_risk_patients(scored_data, selected_segments=["High Risk", "Critical Risk"])
        return {"answer": f"There are {len(table):,} high-risk or critical-risk patients in the current results table.", "table": table}

    if "average los" in question_text or "average length of stay" in question_text:
        if "length_of_stay" in filtered_data.columns:
            return {"answer": f"The current filtered selection has an average length of stay of {calculate_average_length_of_stay(filtered_data):.1f} days.", "table": None}
    if "average cost" in question_text:
        if "cost" in filtered_data.columns:
            return {"answer": f"The current filtered selection has an average cost of ${calculate_average_cost(filtered_data):,.2f} per record.", "table": None}
    if "readmission rate" in question_text and "readmission" in filtered_data.columns:
        return {"answer": f"The current filtered selection has a readmission rate of {calculate_readmission_rate(filtered_data):.1%}.", "table": None}

    return {"answer": "Try asking about department cost, diagnosis readmission rate, average LOS, average cost, longest cohort LOS, or high-risk patients.", "table": None}
