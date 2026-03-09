from __future__ import annotations

from typing import Iterable

import pandas as pd


COLUMN_ALIASES = {
    "admissions": ("admission_id", "encounter_id", "patient_id", "visit_id", "admissions"),
    "readmission": ("readmitted", "readmission", "is_readmitted", "readmission_flag"),
    "length_of_stay": ("length_of_stay", "los", "stay_length", "days_admitted"),
    "cost": ("cost", "total_cost", "treatment_cost", "charges"),
    "department": ("department", "service_line", "unit"),
    "diagnosis": ("diagnosis", "primary_diagnosis", "condition", "diagnosis_group", "dx"),
    "gender": ("gender", "sex", "patient_gender"),
    "age": ("age", "patient_age"),
    "comorbidity_score": ("comorbidity_score", "comorbidity_index"),
    "prior_admissions_12m": ("prior_admissions_12m", "prior_admissions", "admissions_last_12m"),
    "date": ("date", "admission_date", "admit_date", "encounter_date"),
}


DISPLAY_NAMES = {
    "admissions": "Admission",
    "readmission": "Readmission",
    "length_of_stay": "Length of Stay",
    "cost": "Cost",
    "department": "Department",
    "diagnosis": "Diagnosis",
    "gender": "Gender",
    "age": "Age",
    "comorbidity_score": "Comorbidity Score",
    "prior_admissions_12m": "Prior Admissions (12M)",
    "age_group": "Age Group",
    "comorbidity_bucket": "Comorbidity Bucket",
    "predicted_readmission_risk": "Predicted Readmission Risk",
}


MODEL_FEATURES = ["age", "comorbidity_score", "prior_admissions_12m", "length_of_stay"]
RISK_SEGMENTS = ["Low Risk", "Medium Risk", "High Risk", "Critical Risk"]


def resolve_column(data: pd.DataFrame, metric_name: str, required: bool = True) -> str | None:
    candidates: Iterable[str] = COLUMN_ALIASES.get(metric_name, (metric_name,))
    for candidate in candidates:
        if candidate in data.columns:
            return candidate
    if required:
        expected = ", ".join(candidates)
        raise KeyError(f"Missing required column for '{metric_name}'. Expected one of: {expected}.")
    return None


def get_display_name(metric_name: str) -> str:
    return DISPLAY_NAMES.get(metric_name, metric_name.replace("_", " ").title())


def to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def fill_missing_category(series: pd.Series, fill_value: str = "Unknown") -> pd.Series:
    if pd.api.types.is_categorical_dtype(series):
        updated = series
        if fill_value not in updated.cat.categories:
            updated = updated.cat.add_categories([fill_value])
        return updated.fillna(fill_value)
    return series.fillna(fill_value)


def normalize_binary_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype("Int64")
    if pd.api.types.is_numeric_dtype(series):
        numeric_series = to_numeric(series)
        normalized = pd.Series(pd.NA, index=series.index, dtype="Int64")
        normalized.loc[numeric_series.notna()] = (numeric_series.loc[numeric_series.notna()] > 0).astype(int)
        return normalized
    normalized = (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map({"yes": 1, "true": 1, "1": 1, "y": 1, "no": 0, "false": 0, "0": 0, "n": 0})
    )
    return normalized.astype("Int64")


def binary_rate(series: pd.Series) -> float:
    normalized = normalize_binary_series(series).dropna()
    if normalized.empty:
        return 0.0
    return float(normalized.mean())


def calculate_total_admissions(data: pd.DataFrame) -> int:
    admission_column = resolve_column(data, "admissions", required=False)
    if admission_column:
        return int(data[admission_column].nunique())
    return int(len(data))


def calculate_readmission_rate(data: pd.DataFrame) -> float:
    return binary_rate(data[resolve_column(data, "readmission")])


def calculate_readmission_count(data: pd.DataFrame) -> int:
    normalized = normalize_binary_series(data[resolve_column(data, "readmission")]).dropna()
    return int(normalized.sum()) if not normalized.empty else 0


def calculate_average_length_of_stay(data: pd.DataFrame) -> float:
    return float(to_numeric(data[resolve_column(data, "length_of_stay")]).mean())


def calculate_average_cost(data: pd.DataFrame) -> float:
    return float(to_numeric(data[resolve_column(data, "cost")]).mean())


def calculate_total_cost(data: pd.DataFrame) -> float:
    return float(to_numeric(data[resolve_column(data, "cost")]).sum())


def calculate_average_cost_per_day(data: pd.DataFrame) -> float:
    cost_series = to_numeric(data[resolve_column(data, "cost")]).fillna(0)
    los_series = to_numeric(data[resolve_column(data, "length_of_stay")]).fillna(0)
    total_los = float(los_series[los_series > 0].sum())
    if total_los == 0:
        return 0.0
    return float(cost_series.sum() / total_los)


def get_key_metrics(data: pd.DataFrame) -> dict[str, float | int]:
    return {
        "total_admissions": calculate_total_admissions(data),
        "readmission_rate": calculate_readmission_rate(data),
        "readmission_count": calculate_readmission_count(data),
        "average_length_of_stay": calculate_average_length_of_stay(data),
        "average_cost": calculate_average_cost(data),
        "total_cost": calculate_total_cost(data),
        "average_cost_per_day": calculate_average_cost_per_day(data),
    }


def get_age_bounds(data: pd.DataFrame) -> tuple[int, int]:
    age_column = resolve_column(data, "age")
    ages = to_numeric(data[age_column]).dropna()
    if ages.empty:
        raise ValueError("Age column does not contain usable numeric values.")
    return int(ages.min()), int(ages.max())


def add_derived_buckets(data: pd.DataFrame) -> pd.DataFrame:
    enriched = data.copy()
    age_column = resolve_column(enriched, "age")
    comorbidity_column = resolve_column(enriched, "comorbidity_score", required=False)
    enriched[age_column] = to_numeric(enriched[age_column])
    enriched["age_group"] = pd.cut(
        enriched[age_column], bins=[17, 35, 50, 65, float("inf")], labels=["18-35", "36-50", "51-65", "66+"], include_lowest=True
    )
    if comorbidity_column:
        enriched[comorbidity_column] = to_numeric(enriched[comorbidity_column])
        enriched["comorbidity_bucket"] = pd.cut(
            enriched[comorbidity_column], bins=[-1, 1, 3, float("inf")], labels=["0-1", "2-3", "4+"], include_lowest=True
        )
    else:
        enriched["comorbidity_bucket"] = pd.Series(pd.NA, index=enriched.index)
    return enriched


def train_readmission_model(data: pd.DataFrame) -> dict[str, object]:
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as error:
        raise ImportError("scikit-learn is required for the machine learning section.") from error

    feature_columns = {feature: resolve_column(data, feature) for feature in MODEL_FEATURES}
    target_column = resolve_column(data, "readmission")

    model_data = data[list(feature_columns.values()) + [target_column]].copy()
    for column_name in feature_columns.values():
        model_data[column_name] = to_numeric(model_data[column_name])
    model_data["target"] = normalize_binary_series(model_data[target_column])
    model_data = model_data.dropna()

    if len(model_data) < 10:
        raise ValueError("At least 10 complete records are needed to train the model.")
    if model_data["target"].nunique() < 2:
        raise ValueError("The filtered data must include both readmitted and non-readmitted cases.")

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000)),
    ])
    feature_frame = model_data[list(feature_columns.values())]
    pipeline.fit(feature_frame, model_data["target"])
    probabilities = pipeline.predict_proba(feature_frame)[:, 1]

    scored_data = data.loc[model_data.index].copy()
    scored_data["predicted_readmission_risk"] = probabilities
    scored_data["actual_readmission"] = model_data["target"].astype(int).values

    risk_values = scored_data["predicted_readmission_risk"]
    scored_data["risk_segment"] = pd.Series("Low Risk", index=scored_data.index)
    scored_data.loc[(risk_values >= 0.30) & (risk_values < 0.60), "risk_segment"] = "Medium Risk"
    scored_data.loc[(risk_values >= 0.60) & (risk_values < 0.85), "risk_segment"] = "High Risk"
    scored_data.loc[risk_values >= 0.85, "risk_segment"] = "Critical Risk"
    scored_data["risk_segment"] = pd.Categorical(scored_data["risk_segment"], categories=RISK_SEGMENTS, ordered=True)

    classifier = pipeline.named_steps["classifier"]
    explainability = pd.DataFrame({
        "feature_name": [get_display_name(feature) for feature in MODEL_FEATURES],
        "coefficient": classifier.coef_[0],
    })
    explainability["absolute_importance"] = explainability["coefficient"].abs()
    explainability["direction_of_impact"] = explainability["coefficient"].apply(lambda value: "Increases risk" if value > 0 else "Decreases risk")
    explainability = explainability.sort_values("absolute_importance", ascending=False).reset_index(drop=True)

    return {
        "pipeline": pipeline,
        "feature_columns": feature_columns,
        "scored_data": scored_data.sort_values("predicted_readmission_risk", ascending=False),
        "explainability": explainability,
        "roc_auc": float(roc_auc_score(model_data["target"], probabilities)),
    }


def get_risk_segment_summary(scored_data: pd.DataFrame) -> pd.DataFrame:
    summary = scored_data["risk_segment"].value_counts().reindex(RISK_SEGMENTS, fill_value=0).rename_axis("risk_segment").reset_index(name="patient_count")
    total_patients = int(summary["patient_count"].sum())
    summary["percentage"] = summary["patient_count"].apply(lambda value: (value / total_patients) if total_patients else 0.0)
    return summary


def get_high_risk_patients(scored_data: pd.DataFrame, selected_segments: list[str] | None = None, top_n: int = 20, threshold: float | None = None) -> pd.DataFrame:
    filtered = scored_data.copy()
    if threshold is not None:
        filtered = filtered[filtered["predicted_readmission_risk"] >= threshold]
    else:
        selected_segments = selected_segments or ["High Risk", "Critical Risk"]
        filtered = filtered[filtered["risk_segment"].isin(selected_segments)]
    columns: list[str] = []
    for metric_name in ["admissions", "age", "diagnosis", "department", "gender", "comorbidity_score"]:
        column_name = resolve_column(filtered, metric_name, required=False)
        if column_name and column_name not in columns:
            columns.append(column_name)
    columns.extend(["risk_segment", "predicted_readmission_risk"])
    return filtered[columns].head(top_n)
