from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st


CARD_STYLE = """
<style>
    .stApp {
        background: linear-gradient(180deg, #061321 0%, #0b1d30 58%, #10263a 100%);
        color: #e8f1fb;
    }
    [data-testid='stSidebar'] { background: #0b1625; }
    div[data-testid='stMetric'] {
        background: rgba(20, 37, 58, 0.95);
        border: 1px solid rgba(113, 184, 228, 0.20);
        border-radius: 14px;
        padding: 0.9rem 1rem;
        box-shadow: 0 10px 26px rgba(0, 0, 0, 0.16);
    }
</style>
"""

PLOT_LAYOUT = {
    'template': 'plotly_dark',
    'paper_bgcolor': '#0b1625',
    'plot_bgcolor': '#10263a',
    'font': {'color': '#e8f1fb'},
    'margin': dict(l=30, r=20, t=60, b=30),
}


def apply_theme() -> None:
    st.markdown(CARD_STYLE, unsafe_allow_html=True)


def metric_row(items: list[tuple[str, str]]) -> None:
    cols = st.columns(len(items))
    for col, (label, value) in zip(cols, items):
        col.metric(label, value)


def style_figure(fig, xaxis_title: str | None = None, yaxis_title: str | None = None):
    fig.update_layout(**PLOT_LAYOUT, xaxis_title=xaxis_title, yaxis_title=yaxis_title)
    return fig


def plot_missingness(missing_df: pd.DataFrame):
    if missing_df.empty:
        return None
    figure = px.bar(missing_df.head(15), x='column_name', y='null_percentage', title='Highest Missingness by Column', color='null_percentage', color_continuous_scale='Tealgrn')
    figure.update_layout(coloraxis_showscale=False)
    return style_figure(figure, 'Column', 'Null Percentage')


def plot_numeric_distribution(data: pd.DataFrame, column: str):
    if column not in data.columns:
        return None
    series = pd.to_numeric(data[column], errors='coerce').dropna()
    if len(series) > 5000:
        series = series.sample(5000, random_state=42)
    if series.empty:
        return None
    figure = px.histogram(series.to_frame(name=column), x=column, nbins=30, title=f'Distribution of {column}')
    figure.update_traces(marker_color='#4cc9f0')
    return style_figure(figure, column, 'Rows')


def plot_top_categories(data: pd.DataFrame, column: str):
    if column not in data.columns:
        return None
    summary = data[column].fillna('Missing').astype(str).value_counts().head(12).rename_axis(column).reset_index(name='count')
    if summary.empty:
        return None
    figure = px.bar(summary, x=column, y='count', title=f'Top Values for {column}', color='count', color_continuous_scale='Sunset')
    figure.update_layout(coloraxis_showscale=False)
    return style_figure(figure, column, 'Rows')


def plot_correlation(corr: pd.DataFrame):
    if corr.empty:
        return None
    figure = px.imshow(corr, text_auto='.2f', aspect='auto', color_continuous_scale='Teal', title='Correlation Heatmap')
    return style_figure(figure, 'Numeric Field', 'Numeric Field')


def plot_time_trend(trend_df: pd.DataFrame, x_col: str, y_col: str, title: str):
    if trend_df.empty or y_col not in trend_df.columns:
        return None
    figure = px.line(trend_df, x=x_col, y=y_col, markers=True, title=title)
    return style_figure(figure, x_col.replace('_', ' ').title(), y_col.replace('_', ' ').title())


def plot_bar(data: pd.DataFrame, x_col: str, y_col: str, title: str):
    if data.empty or x_col not in data.columns or y_col not in data.columns:
        return None
    figure = px.bar(data.head(15), x=x_col, y=y_col, title=title, color=y_col, color_continuous_scale='Tealgrn')
    figure.update_layout(coloraxis_showscale=False)
    return style_figure(figure, x_col.replace('_', ' ').title(), y_col.replace('_', ' ').title())


def plot_numeric_box(data: pd.DataFrame, column: str):
    if column not in data.columns:
        return None
    series = pd.to_numeric(data[column], errors='coerce').dropna()
    if len(series) > 5000:
        series = series.sample(5000, random_state=42)
    if series.empty:
        return None
    figure = px.box(series.to_frame(name=column), y=column, title=f'Box Plot for {column}')
    figure.update_traces(marker_color='#4cc9f0')
    return style_figure(figure, None, column)
