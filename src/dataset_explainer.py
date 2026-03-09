from __future__ import annotations

import pandas as pd

from src.generic_profile import profile_dataset
from src.schema_detection import dataset_capabilities, schema_coverage_percent


THEME_KEYWORDS = {
    "healthcare": ["diagnosis", "readmission", "los", "length_of_stay", "department", "patient", "admit", "hospital", "age", "gender"],
    "finance": ["revenue", "expense", "balance", "cost", "price", "profit", "account", "payment"],
    "sales": ["sales", "order", "customer", "product", "quantity", "region", "channel"],
    "logistics": ["shipment", "delivery", "warehouse", "route", "carrier", "inventory", "fleet"],
    "survey": ["response", "question", "rating", "score", "feedback", "survey"],
}


def _resolve_schema_column(data: pd.DataFrame, matched_schema: dict[str, str], internal_field: str) -> str | None:
    if internal_field in data.columns:
        return internal_field
    mapped_column = matched_schema.get(internal_field)
    if mapped_column and mapped_column in data.columns:
        return mapped_column
    return None


def infer_dataset_theme(columns: list[str], matched_schema: dict[str, str]) -> str:
    if schema_coverage_percent(matched_schema) >= 0.35:
        return "healthcare"
    lowered = [column.lower() for column in columns]
    scores: dict[str, int] = {theme: 0 for theme in THEME_KEYWORDS}
    for theme, keywords in THEME_KEYWORDS.items():
        for keyword in keywords:
            scores[theme] += sum(keyword in column for column in lowered)
    best_theme = max(scores, key=scores.get)
    return best_theme if scores[best_theme] > 0 else "general dataset"


def _class_imbalance_text(data: pd.DataFrame, matched_schema: dict[str, str]) -> str:
    column_name = _resolve_schema_column(data, matched_schema, "readmission")
    if not column_name:
        return "Class imbalance analysis is not available because no target-like readmission column was detected."
    series = data[column_name].astype(str).str.lower().str.strip()
    if series.empty:
        return "Class imbalance analysis is not available because the detected readmission column does not contain usable values."
    positive_ratio = series.isin(["1", "true", "yes", "y"]).mean()
    return f"The detected readmission field shows a positive-class rate of {positive_ratio:.1%}, which is important to consider for modeling and benchmark interpretation."


def explain_dataset(data: pd.DataFrame, matched_schema: dict[str, str]) -> dict[str, object]:
    profile = profile_dataset(data)
    capabilities = dataset_capabilities(matched_schema)
    theme = infer_dataset_theme(profile["column_names"], matched_schema)

    summary_lines = [
        f"The filtered dataset currently contains {profile['row_count']:,} rows across {profile['column_count']:,} columns.",
        f"It includes {len(profile['numeric_columns'])} numeric columns, {len(profile['categorical_columns'])} categorical or text columns, and {len(profile['date_columns'])} detected date columns.",
    ]

    observations: list[str] = []
    if not profile["missing_values"].empty:
        top_missing = profile["missing_values"].iloc[0]
        if int(top_missing["missing_values"]) > 0:
            observations.append(f"{top_missing['column_name']} has the highest missing-value count at {int(top_missing['missing_values']):,} rows.")
    if not profile["unique_values"].empty:
        top_unique = profile["unique_values"].iloc[0]
        observations.append(f"{top_unique['column_name']} is the most unique column, with {int(top_unique['unique_values']):,} distinct values.")
    if not profile["correlation_matrix"].empty:
        corr = profile["correlation_matrix"].copy()
        corr.values[[range(len(corr))] * 2] = 0
        strongest = corr.abs().stack().sort_values(ascending=False)
        if not strongest.empty:
            (left, right), value = strongest.index[0], strongest.iloc[0]
            observations.append(f"The strongest numeric relationship is between {left} and {right}, with an absolute correlation of {value:.2f}.")
    if profile["date_columns"]:
        observations.append(f"Date coverage is available through {', '.join(profile['date_columns'][:2])}, enabling time-trend analysis.")
    observations.append(_class_imbalance_text(data, matched_schema))

    if not observations:
        observations.append("The dataset appears structurally clean enough for exploratory profiling, although deeper interpretation depends on business context.")

    next_steps: list[str] = []
    if profile["numeric_columns"]:
        next_steps.append("Review numeric distributions, outliers, and trends to identify concentration and volatility patterns.")
    if profile["categorical_columns"]:
        next_steps.append("Examine top categories and grouped summaries to understand which segments drive volume or performance differences.")
    if profile["date_columns"]:
        next_steps.append("Use the detected date fields to review monthly changes in volume and average numeric measures.")
    if theme == "healthcare" and capabilities["kpi_analysis"]:
        next_steps.append("Use the healthcare mode tabs to evaluate readmissions, cost, length of stay, and department performance.")
    elif theme == "healthcare":
        next_steps.append("Improve schema coverage for diagnosis, department, cost, LOS, and readmission fields to unlock advanced healthcare analytics.")
    else:
        next_steps.append("Use the generic auto-generated dashboard to prioritize the most informative columns for follow-up analysis.")

    if theme == "healthcare" and capabilities["cost_analysis"]:
        theme_text = "The dataset appears healthcare-oriented and contains enough mapped operational fields to support hospital performance analysis."
    elif theme == "healthcare":
        theme_text = "The dataset appears healthcare-oriented, but the mapped schema is only partially complete for advanced clinical analytics."
    else:
        theme_text = f"The dataset most closely resembles a {theme} use case based on its column structure."

    return {
        "summary": summary_lines,
        "theme": theme_text,
        "observations": observations[:5],
        "next_steps": next_steps[:4],
        "profile": profile,
    }
