from __future__ import annotations

import pandas as pd
import plotly.express as px

from src.analytics import get_cost_per_day_by_diagnosis, get_highest_cost_departments, get_top_diagnosis_groups_by_total_cost
from src.metrics import RISK_SEGMENTS, binary_rate, fill_missing_category, get_display_name, resolve_column


DARK_LAYOUT = {
    "template": "plotly_dark",
    "paper_bgcolor": "#0b1727",
    "plot_bgcolor": "#122338",
    "font": {"color": "#e7f0fb"},
}


def _style_figure(figure, yaxis_title: str, xaxis_title: str | None = None) -> None:
    figure.update_layout(**DARK_LAYOUT, margin=dict(l=20, r=20, t=60, b=20), yaxis_title=yaxis_title, xaxis_title=xaxis_title, legend_title_text="")


def create_cost_by_department_chart(data: pd.DataFrame):
    department_column = resolve_column(data, "department")
    cost_column = resolve_column(data, "cost")
    chart_data = data[[department_column, cost_column]].copy()
    chart_data[department_column] = fill_missing_category(chart_data[department_column], "Unknown")
    chart_data[cost_column] = pd.to_numeric(chart_data[cost_column], errors="coerce")
    chart_data = chart_data.dropna(subset=[cost_column])
    if chart_data.empty:
        return None
    chart_data = chart_data.groupby(department_column, dropna=False)[cost_column].mean().reset_index().sort_values(cost_column, ascending=False)
    figure = px.bar(chart_data, x=department_column, y=cost_column, color=cost_column, color_continuous_scale="Tealgrn", title="Average Cost by Department", labels={department_column: "Department", cost_column: "Average Cost"})
    figure.update_traces(hovertemplate="%{x}<br>Average Cost: $%{y:,.2f}<extra></extra>")
    figure.update_layout(coloraxis_showscale=False)
    _style_figure(figure, "Average Cost", "Department")
    return figure


def create_readmission_by_diagnosis_chart(data: pd.DataFrame):
    diagnosis_column = resolve_column(data, "diagnosis")
    readmission_column = resolve_column(data, "readmission")
    chart_data = data.copy()
    chart_data[diagnosis_column] = fill_missing_category(chart_data[diagnosis_column], "Unknown")
    chart_data = chart_data.groupby(diagnosis_column, dropna=False)[readmission_column].apply(binary_rate).reset_index(name="readmission_rate").sort_values("readmission_rate", ascending=False)
    if chart_data.empty:
        return None
    figure = px.bar(chart_data, x=diagnosis_column, y="readmission_rate", color="readmission_rate", color_continuous_scale="Sunset", title="Readmission Rate by Diagnosis", labels={diagnosis_column: "Diagnosis", "readmission_rate": "Readmission Rate"})
    figure.update_traces(hovertemplate="%{x}<br>Readmission Rate: %{y:.1%}<extra></extra>")
    figure.update_layout(coloraxis_showscale=False)
    figure.update_yaxes(tickformat=".0%")
    _style_figure(figure, "Readmission Rate", "Diagnosis")
    return figure


def create_cohort_metric_chart(summary: pd.DataFrame, cohort_column: str, metric_column: str, title: str, yaxis_title: str):
    if summary.empty:
        return None
    figure = px.bar(summary, x=cohort_column, y=metric_column, color=metric_column, color_continuous_scale="Teal", title=title, labels={cohort_column: get_display_name(cohort_column), metric_column: yaxis_title})
    hover_format = "%{y:.1%}" if "rate" in metric_column else "$%{y:,.2f}" if "cost" in metric_column else "%{y:.1f}"
    figure.update_traces(hovertemplate=f"%{{x}}<br>{yaxis_title}: {hover_format}<extra></extra>")
    figure.update_layout(coloraxis_showscale=False)
    if "rate" in metric_column:
        figure.update_yaxes(tickformat=".0%")
    _style_figure(figure, yaxis_title, get_display_name(cohort_column))
    return figure


def create_risk_distribution_chart(summary: pd.DataFrame):
    if summary.empty:
        return None
    figure = px.bar(summary, x="risk_segment", y="patient_count", color="risk_segment", category_orders={"risk_segment": RISK_SEGMENTS}, title="Patient Risk Segment Distribution", labels={"risk_segment": "Risk Segment", "patient_count": "Patients"}, color_discrete_map={"Low Risk": "#2a9d8f", "Medium Risk": "#e9c46a", "High Risk": "#f4a261", "Critical Risk": "#e76f51"})
    figure.update_traces(showlegend=False)
    _style_figure(figure, "Patients", "Risk Segment")
    return figure


def create_probability_histogram(scored_data: pd.DataFrame):
    if scored_data.empty:
        return None
    figure = px.histogram(scored_data, x="predicted_readmission_risk", nbins=20, title="Predicted Readmission Probability Histogram", labels={"predicted_readmission_risk": "Predicted Probability", "count": "Patients"})
    figure.update_traces(marker_color="#4cc9f0", hovertemplate="Probability: %{x:.2f}<br>Patients: %{y}<extra></extra>")
    figure.update_xaxes(tickformat=".0%")
    _style_figure(figure, "Patients", "Predicted Probability")
    return figure


def create_explainability_chart(explainability: pd.DataFrame):
    if explainability.empty:
        return None
    chart_data = explainability.sort_values("absolute_importance", ascending=True)
    figure = px.bar(chart_data, x="absolute_importance", y="feature_name", color="direction_of_impact", orientation="h", title="Top Factors Driving Readmission", labels={"absolute_importance": "Absolute Importance", "feature_name": "Feature"}, color_discrete_map={"Increases risk": "#ff6b6b", "Decreases risk": "#2ec4b6"})
    figure.update_traces(hovertemplate="Feature: %{y}<br>Absolute Importance: %{x:.3f}<extra></extra>")
    _style_figure(figure, "Feature", "Absolute Importance")
    return figure


def create_top_diagnosis_cost_chart(data: pd.DataFrame):
    chart_data = get_top_diagnosis_groups_by_total_cost(data)
    if chart_data.empty:
        return None
    diagnosis_column = resolve_column(data, "diagnosis")
    figure = px.bar(chart_data, x=diagnosis_column, y="total_cost", color="total_cost", color_continuous_scale="Blues", title="Top Diagnosis Groups by Total Cost", labels={diagnosis_column: "Diagnosis Group", "total_cost": "Total Cost"})
    figure.update_traces(hovertemplate="%{x}<br>Total Cost: $%{y:,.2f}<extra></extra>")
    figure.update_layout(coloraxis_showscale=False)
    _style_figure(figure, "Total Cost", "Diagnosis Group")
    return figure


def create_cost_per_day_chart(data: pd.DataFrame):
    chart_data = get_cost_per_day_by_diagnosis(data)
    if chart_data.empty:
        return None
    diagnosis_column = resolve_column(data, "diagnosis")
    figure = px.bar(chart_data, x=diagnosis_column, y="cost_per_day", color="cost_per_day", color_continuous_scale="Agsunset", title="Cost per Length of Stay Day", labels={diagnosis_column: "Diagnosis Group", "cost_per_day": "Cost per Day"})
    figure.update_traces(hovertemplate="%{x}<br>Cost per Day: $%{y:,.2f}<extra></extra>")
    figure.update_layout(coloraxis_showscale=False)
    _style_figure(figure, "Cost per Day", "Diagnosis Group")
    return figure


def create_highest_cost_departments_chart(data: pd.DataFrame):
    chart_data = get_highest_cost_departments(data)
    if chart_data.empty:
        return None
    department_column = resolve_column(data, "department")
    figure = px.bar(chart_data, x=department_column, y="total_cost", color="total_cost", color_continuous_scale="Teal", title="Highest Cost Departments", labels={department_column: "Department", "total_cost": "Total Cost"})
    figure.update_traces(hovertemplate="%{x}<br>Total Cost: $%{y:,.2f}<extra></extra>")
    figure.update_layout(coloraxis_showscale=False)
    _style_figure(figure, "Total Cost", "Department")
    return figure


def create_department_metric_chart(scorecard: pd.DataFrame, metric_column: str, title: str, yaxis_title: str):
    if scorecard.empty:
        return None
    department_column = scorecard.columns[0]
    figure = px.bar(scorecard, x=department_column, y=metric_column, color=metric_column, color_continuous_scale="Sunsetdark", title=title, labels={department_column: "Department", metric_column: yaxis_title})
    figure.update_layout(coloraxis_showscale=False)
    if "rate" in metric_column or "risk" in metric_column:
        figure.update_yaxes(tickformat=".0%")
    _style_figure(figure, yaxis_title, "Department")
    return figure


def create_monthly_trend_chart(trend_data: pd.DataFrame, metric_column: str, title: str, yaxis_title: str):
    if trend_data.empty:
        return None
    figure = px.line(trend_data, x="month", y=metric_column, markers=True, title=title, labels={"month": "Month", metric_column: yaxis_title})
    if "rate" in metric_column:
        figure.update_yaxes(tickformat=".0%")
    if "cost" in metric_column:
        figure.update_traces(hovertemplate="%{x|%Y-%m}<br>Value: $%{y:,.2f}<extra></extra>")
    _style_figure(figure, yaxis_title, "Month")
    return figure


def create_comparison_chart(comparison_data: pd.DataFrame):
    if comparison_data.empty:
        return None
    figure = px.bar(comparison_data, x="metric", y=["filtered_selection", "overall_dataset"], barmode="group", title="Filtered Selection vs Overall Dataset", labels={"value": "Value", "metric": "Metric", "variable": "Population"})
    _style_figure(figure, "Value", "Metric")
    return figure


def create_risk_drilldown_chart(table: pd.DataFrame, x_column: str, title: str):
    if table.empty:
        return None
    figure = px.bar(table, x=x_column, y="patient_count", color="patient_count", color_continuous_scale="Teal", title=title, labels={x_column: x_column.replace("_", " ").title(), "patient_count": "Patients"})
    figure.update_layout(coloraxis_showscale=False)
    _style_figure(figure, "Patients", x_column.replace("_", " ").title())
    return figure
