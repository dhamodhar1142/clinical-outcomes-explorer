from __future__ import annotations

import pandas as pd
import plotly.express as px
from pandas.api.types import is_bool_dtype, is_datetime64_any_dtype, is_numeric_dtype


DARK_LAYOUT = {
    "template": "plotly_dark",
    "paper_bgcolor": "#0b1727",
    "plot_bgcolor": "#122338",
    "font": {"color": "#e7f0fb"},
}


def _style_figure(figure, yaxis_title: str, xaxis_title: str | None = None) -> None:
    figure.update_layout(**DARK_LAYOUT, margin=dict(l=20, r=20, t=60, b=20), yaxis_title=yaxis_title, xaxis_title=xaxis_title, legend_title_text="")


def _safe_series(data: pd.DataFrame, column_name: str) -> pd.Series:
    return data[column_name] if column_name in data.columns else pd.Series(dtype="object")


def detect_date_columns(data: pd.DataFrame, threshold: float = 0.7) -> list[str]:
    detected: list[str] = []
    for column in data.columns:
        series = _safe_series(data, column)
        non_null = series.dropna()
        if non_null.empty:
            continue
        if is_datetime64_any_dtype(series):
            detected.append(column)
            continue
        sample = non_null.astype(str).head(250)
        parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
        parse_ratio = float(parsed.notna().mean()) if len(sample) else 0.0
        name_hint = any(token in column.lower() for token in ["date", "time", "month", "year"])
        if parse_ratio >= threshold or (name_hint and parse_ratio >= 0.4):
            detected.append(column)
    return detected


def infer_column_type(series: pd.Series, date_columns: list[str], column_name: str) -> str:
    if column_name in date_columns or is_datetime64_any_dtype(series):
        return "datetime"
    if is_bool_dtype(series):
        return "boolean"
    if is_numeric_dtype(series):
        return "numeric"
    unique_ratio = (series.nunique(dropna=True) / max(series.dropna().shape[0], 1)) if not series.dropna().empty else 0.0
    return "categorical" if unique_ratio <= 0.5 else "text"


def profile_dataset(data: pd.DataFrame) -> dict[str, object]:
    if len(data) > 50000:
        analysis_df = data.sample(20000, random_state=42)
        is_sampled = True
    else:
        analysis_df = data
        is_sampled = False

    date_columns = detect_date_columns(data)
    column_profile_rows: list[dict[str, object]] = []
    numeric_columns: list[str] = []
    categorical_columns: list[str] = []

    for column in data.columns:
        series = _safe_series(data, column)
        inferred_type = infer_column_type(series, date_columns, column)
        if inferred_type == "numeric":
            numeric_columns.append(column)
        elif inferred_type in {"categorical", "text", "boolean"}:
            categorical_columns.append(column)
        column_profile_rows.append({
            "column_name": column,
            "inferred_type": inferred_type,
            "missing_values": int(series.isna().sum()),
            "unique_values": int(series.nunique(dropna=True)),
        })

    column_profile = pd.DataFrame(column_profile_rows)
    missing_values = column_profile[["column_name", "missing_values"]].sort_values(["missing_values", "column_name"], ascending=[False, True]).reset_index(drop=True)
    unique_values = column_profile[["column_name", "unique_values"]].sort_values(["unique_values", "column_name"], ascending=[False, True]).reset_index(drop=True)

    numeric_summary = (
        analysis_df[numeric_columns].apply(pd.to_numeric, errors="coerce").describe().transpose().reset_index().rename(columns={"index": "column_name"})
        if numeric_columns
        else pd.DataFrame(columns=["column_name", "count", "mean", "std", "min", "25%", "50%", "75%", "max"])
    )

    top_categories: dict[str, pd.DataFrame] = {}
    for column in categorical_columns[:6]:
        top_categories[column] = (
            analysis_df[column]
            .fillna("Unknown")
            .astype(str)
            .value_counts(dropna=False)
            .head(10)
            .rename_axis(column)
            .reset_index(name="count")
        )

    correlation_columns = numeric_columns[:20]
    if len(correlation_columns) >= 2:
        correlation_matrix = analysis_df[correlation_columns].apply(pd.to_numeric, errors="coerce").corr(numeric_only=True)
    else:
        correlation_matrix = pd.DataFrame()

    outlier_rows: list[dict[str, object]] = []
    for column in numeric_columns:
        values = pd.to_numeric(analysis_df[column], errors="coerce").dropna()
        if values.empty:
            continue
        q1 = float(values.quantile(0.25))
        q3 = float(values.quantile(0.75))
        iqr = q3 - q1
        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)
        outlier_count = int(((values < lower_bound) | (values > upper_bound)).sum())
        outlier_rows.append({
            "column_name": column,
            "outlier_count": outlier_count,
            "outlier_pct": outlier_count / max(len(values), 1),
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
        })
    outlier_summary = pd.DataFrame(outlier_rows).sort_values(["outlier_count", "column_name"], ascending=[False, True]).reset_index(drop=True) if outlier_rows else pd.DataFrame(columns=["column_name", "outlier_count", "outlier_pct", "lower_bound", "upper_bound"])

    grouped_summary = pd.DataFrame()
    if numeric_columns and categorical_columns:
        category_column = categorical_columns[0]
        numeric_column = numeric_columns[0]
        grouped = analysis_df[[category_column, numeric_column]].copy()
        grouped[category_column] = grouped[category_column].fillna("Unknown").astype(str)
        grouped[numeric_column] = pd.to_numeric(grouped[numeric_column], errors="coerce")
        grouped = grouped.dropna(subset=[numeric_column])
        if not grouped.empty:
            grouped_summary = (
                grouped.groupby(category_column, dropna=False)[numeric_column]
                .agg(record_count="count", average_value="mean")
                .reset_index()
                .sort_values("average_value", ascending=False)
                .head(10)
            )

    trend_data = pd.DataFrame()
    if date_columns:
        date_column = date_columns[0]
        trend_frame = data.copy()
        trend_frame[date_column] = pd.to_datetime(trend_frame[date_column], errors="coerce")
        trend_frame = trend_frame.dropna(subset=[date_column])
        if not trend_frame.empty:
            trend_frame["month"] = trend_frame[date_column].dt.to_period("M").dt.to_timestamp()
            trend_data = trend_frame.groupby("month", dropna=False).size().reset_index(name="row_count")
            for numeric_column in numeric_columns[:3]:
                trend_data[f"avg_{numeric_column}"] = trend_frame.groupby("month", dropna=False)[numeric_column].apply(lambda s: pd.to_numeric(s, errors="coerce").mean()).values

    return {
        "row_count": len(data),
        "column_count": len(data.columns),
        "column_names": list(data.columns),
        "date_columns": date_columns,
        "duplicate_row_count": int(data.duplicated().sum()),
        "column_profile": column_profile,
        "missing_values": missing_values,
        "unique_values": unique_values,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "numeric_summary": numeric_summary,
        "top_categories": top_categories,
        "correlation_matrix": correlation_matrix,
        "outlier_summary": outlier_summary,
        "grouped_summary": grouped_summary,
        "trend_data": trend_data,
        "analysis_row_count": len(analysis_df),
        "is_sampled": is_sampled,
    }


def quick_profile_dataset(data: pd.DataFrame) -> dict[str, object]:
    if len(data) > 50000:
        analysis_df = data.sample(min(len(data), 10000), random_state=42)
        is_sampled = True
    else:
        analysis_df = data
        is_sampled = False

    date_columns = detect_date_columns(analysis_df)
    column_profile_rows: list[dict[str, object]] = []
    numeric_columns: list[str] = []
    categorical_columns: list[str] = []

    for column in data.columns:
        series = _safe_series(data, column)
        inferred_type = infer_column_type(series, date_columns, column)
        if inferred_type == "numeric":
            numeric_columns.append(column)
        elif inferred_type in {"categorical", "text", "boolean"}:
            categorical_columns.append(column)
        column_profile_rows.append({
            "column_name": column,
            "inferred_type": inferred_type,
            "missing_values": int(series.isna().sum()),
            "unique_values": int(series.nunique(dropna=True)),
        })

    column_profile = pd.DataFrame(column_profile_rows)
    missing_values = column_profile[["column_name", "missing_values"]].sort_values(["missing_values", "column_name"], ascending=[False, True]).reset_index(drop=True)
    unique_values = column_profile[["column_name", "unique_values"]].sort_values(["unique_values", "column_name"], ascending=[False, True]).reset_index(drop=True)

    numeric_summary_columns = numeric_columns[:10]
    numeric_summary = (
        analysis_df[numeric_summary_columns]
        .apply(pd.to_numeric, errors="coerce")
        .describe()
        .transpose()
        .reset_index()
        .rename(columns={"index": "column_name"})
        if numeric_summary_columns
        else pd.DataFrame(columns=["column_name", "count", "mean", "std", "min", "25%", "50%", "75%", "max"])
    )

    top_categories: dict[str, pd.DataFrame] = {}
    for column in categorical_columns[:3]:
        top_categories[column] = (
            analysis_df[column]
            .fillna("Unknown")
            .astype(str)
            .value_counts(dropna=False)
            .head(10)
            .rename_axis(column)
            .reset_index(name="count")
        )

    correlation_matrix = pd.DataFrame()
    if len(data) < 20000 and len(numeric_columns) <= 10 and len(numeric_columns) >= 2:
        correlation_columns = numeric_columns[:10]
        correlation_matrix = analysis_df[correlation_columns].apply(pd.to_numeric, errors="coerce").corr(numeric_only=True)

    return {
        "row_count": len(data),
        "column_count": len(data.columns),
        "column_names": list(data.columns),
        "date_columns": date_columns,
        "duplicate_row_count": int(data.duplicated().sum()),
        "column_profile": column_profile,
        "missing_values": missing_values,
        "unique_values": unique_values,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "numeric_summary": numeric_summary,
        "top_categories": top_categories,
        "correlation_matrix": correlation_matrix,
        "outlier_summary": pd.DataFrame(columns=["column_name", "outlier_count", "outlier_pct", "lower_bound", "upper_bound"]),
        "grouped_summary": pd.DataFrame(),
        "trend_data": pd.DataFrame(),
        "analysis_row_count": len(analysis_df),
        "is_sampled": is_sampled,
        "profile_mode": "quick",
    }
def create_numeric_histogram(data: pd.DataFrame, column_name: str):
    if column_name not in data.columns:
        return None
    chart_data = pd.to_numeric(data[column_name], errors="coerce").dropna()
    if len(chart_data) > 5000:
        chart_data = chart_data.sample(min(len(chart_data), 5000), random_state=42)
    if chart_data.empty:
        return None
    figure = px.histogram(chart_data.to_frame(name=column_name), x=column_name, nbins=25, title=f"Distribution of {column_name.replace('_', ' ').title()}")
    figure.update_traces(marker_color="#4cc9f0")
    _style_figure(figure, "Rows", column_name.replace("_", " ").title())
    return figure

def create_category_bar(top_categories: pd.DataFrame, column_name: str):
    if top_categories.empty:
        return None
    figure = px.bar(top_categories, x=column_name, y="count", color="count", color_continuous_scale="Tealgrn", title=f"Top Values for {column_name.replace('_', ' ').title()}")
    figure.update_layout(coloraxis_showscale=False)
    _style_figure(figure, "Rows", column_name.replace("_", " ").title())
    return figure


def create_correlation_heatmap(correlation_matrix: pd.DataFrame):
    if correlation_matrix.empty:
        return None
    figure = px.imshow(correlation_matrix, text_auto=".2f", aspect="auto", color_continuous_scale="Teal", title="Correlation Matrix")
    _style_figure(figure, "Numeric Columns", "Numeric Columns")
    return figure


def create_grouped_summary_chart(grouped_summary: pd.DataFrame):
    if grouped_summary.empty:
        return None
    category_column = grouped_summary.columns[0]
    figure = px.bar(grouped_summary, x=category_column, y="average_value", color="record_count", color_continuous_scale="Sunset", title=f"Average Value by {category_column.replace('_', ' ').title()}")
    figure.update_layout(coloraxis_showscale=False)
    _style_figure(figure, "Average Value", category_column.replace("_", " ").title())
    return figure


def create_generic_trend_chart(trend_data: pd.DataFrame, value_column: str, title: str, yaxis_title: str):
    if trend_data.empty or value_column not in trend_data.columns:
        return None
    figure = px.line(trend_data, x="month", y=value_column, markers=True, title=title)
    _style_figure(figure, yaxis_title, "Month")
    return figure




def build_quality_insights(data: pd.DataFrame, profile: dict[str, object] | None = None) -> dict[str, object]:
    profile = profile or profile_dataset(data)
    invalid_numeric_rows = 0
    constant_columns: list[str] = []
    duplication_rows: list[dict[str, object]] = []

    for column in profile["numeric_columns"]:
        numeric_values = pd.to_numeric(data[column], errors="coerce")
        invalid_numeric_rows += int(data[column].notna().sum() - numeric_values.notna().sum())
    for column in data.columns:
        if data[column].nunique(dropna=False) <= 1:
            constant_columns.append(column)
        duplicate_pct = 1 - (data[column].nunique(dropna=False) / max(len(data), 1))
        duplication_rows.append({"column_name": column, "duplication_pct": max(duplicate_pct, 0.0)})

    missing_ratio = profile["missing_values"]["missing_values"].sum() / max(len(data) * max(len(data.columns), 1), 1)
    duplicate_ratio = profile["duplicate_row_count"] / max(len(data), 1)
    outlier_ratio = profile["outlier_summary"]["outlier_count"].sum() / max(len(data) * max(len(profile["numeric_columns"]), 1), 1) if not profile["outlier_summary"].empty else 0.0
    invalid_ratio = invalid_numeric_rows / max(len(data) * max(len(profile["numeric_columns"]), 1), 1) if profile["numeric_columns"] else 0.0
    constant_ratio = len(constant_columns) / max(len(data.columns), 1)

    score = 100
    score -= min(int(missing_ratio * 100), 30)
    score -= min(int(duplicate_ratio * 100), 20)
    score -= min(int(outlier_ratio * 100), 20)
    score -= min(int(invalid_ratio * 100), 15)
    score -= min(int(constant_ratio * 100), 15)
    score = max(score, 0)

    duplication_summary = pd.DataFrame(duplication_rows).sort_values("duplication_pct", ascending=False).head(10)
    return {
        "quality_score": score,
        "missing_values": profile["missing_values"].head(10),
        "outlier_summary": profile["outlier_summary"].head(10),
        "duplication_summary": duplication_summary,
        "constant_columns": constant_columns,
        "invalid_numeric_values": invalid_numeric_rows,
        "duplicate_row_count": profile["duplicate_row_count"],
    }


def build_auto_generated_dashboard_spec(data: pd.DataFrame, profile: dict[str, object] | None = None) -> list[dict[str, str]]:
    profile = profile or profile_dataset(data)
    charts: list[dict[str, str]] = []
    for column in profile["numeric_columns"][:3]:
        charts.append({"chart_type": "histogram", "column": column, "title": f"Distribution of {column.replace('_', ' ').title()}"})
    for column in profile["categorical_columns"][:3]:
        charts.append({"chart_type": "category", "column": column, "title": f"Top Values for {column.replace('_', ' ').title()}"})
    if not profile["correlation_matrix"].empty:
        charts.append({"chart_type": "correlation", "column": "correlation_matrix", "title": "Correlation Matrix"})
    if not profile["trend_data"].empty:
        charts.append({"chart_type": "trend", "column": "row_count", "title": "Rows Over Time"})
        avg_columns = [column for column in profile["trend_data"].columns if column.startswith("avg_")]
        if avg_columns:
            charts.append({"chart_type": "trend", "column": avg_columns[0], "title": f"Average {avg_columns[0][4:].replace('_', ' ').title()} Over Time"})
    numeric_columns = profile["numeric_columns"]
    if len(numeric_columns) >= 2:
        charts.append({"chart_type": "scatter", "column": f"{numeric_columns[0]}|{numeric_columns[1]}", "title": f"{numeric_columns[0].replace('_', ' ').title()} vs {numeric_columns[1].replace('_', ' ').title()}"})
    return charts[:10]


def create_scatter_plot(data: pd.DataFrame, x_column: str, y_column: str):
    if x_column not in data.columns or y_column not in data.columns:
        return None
    chart_data = data[[x_column, y_column]].copy()
    if len(chart_data) > 5000:
        chart_data = chart_data.sample(min(len(chart_data), 5000), random_state=42)
    chart_data[x_column] = pd.to_numeric(chart_data[x_column], errors="coerce")
    chart_data[y_column] = pd.to_numeric(chart_data[y_column], errors="coerce")
    chart_data = chart_data.dropna()
    if chart_data.empty:
        return None
    figure = px.scatter(chart_data, x=x_column, y=y_column, title=f"{x_column.replace('_', ' ').title()} vs {y_column.replace('_', ' ').title()}")
    _style_figure(figure, y_column.replace("_", " ").title(), x_column.replace("_", " ").title())
    return figure

