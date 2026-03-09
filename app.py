from __future__ import annotations

import io
import time

import pandas as pd
import plotly.express as px
import streamlit as st

from src.analytics import (
    COHORT_FIELDS,
    assess_data_quality,
    build_benchmarking_summary,
    build_cohort_summary,
    build_custom_cohort,
    build_department_scorecard,
    build_executive_summary,
    build_model_evaluation,
    build_monthly_trends,
    build_risk_drilldown,
    build_stakeholder_summary,
    build_summary_text,
    compare_scenarios,
    dataframe_to_csv_bytes,
    generate_key_insights,
    generate_operational_alerts,
)
from src.analytics_assistant import SUPPORTED_PATTERNS, answer_business_question
from src.charts import (
    create_cohort_metric_chart,
    create_comparison_chart,
    create_cost_by_department_chart,
    create_cost_per_day_chart,
    create_department_metric_chart,
    create_explainability_chart,
    create_highest_cost_departments_chart,
    create_monthly_trend_chart,
    create_probability_histogram,
    create_readmission_by_diagnosis_chart,
    create_risk_distribution_chart,
    create_risk_drilldown_chart,
    create_top_diagnosis_cost_chart,
)
from src.data_loader import DATA_PATH, load_hospital_data
from src.dataset_explainer import explain_dataset
from src.generic_profile import (
    build_auto_generated_dashboard_spec,
    build_quality_insights,
    create_category_bar,
    create_correlation_heatmap,
    create_generic_trend_chart,
    create_numeric_histogram,
    create_scatter_plot,
    detect_date_columns,
    infer_column_type,
    profile_dataset,
    quick_profile_dataset,
)
from src.metrics import (
    RISK_SEGMENTS,
    add_derived_buckets,
    calculate_average_cost,
    calculate_average_length_of_stay,
    calculate_readmission_rate,
    get_age_bounds,
    get_high_risk_patients,
    get_key_metrics,
    get_risk_segment_summary,
    to_numeric,
    train_readmission_model,
)
from src.schema_detection import (
    EXPECTED_FIELDS,
    FIELD_LABELS,
    apply_detected_schema,
    build_mapping_table,
    dataset_capabilities,
    detect_schema,
    schema_coverage_percent,
)

st.set_page_config(page_title="Clinical Outcomes Explorer", layout="wide")

CARD_STYLE = """
<style>
    .stApp {
        background: linear-gradient(180deg, #08131f 0%, #0d1b2a 55%, #10263a 100%);
        color: #e7f0fb;
    }
    [data-testid="stSidebar"] { background: #0b1727; }
    .block-container { padding-top: 1.1rem; padding-bottom: 3rem; }
    div[data-testid="stMetric"] {
        background: rgba(18, 35, 56, 0.95);
        border: 1px solid rgba(86, 155, 204, 0.22);
        padding: 0.85rem 1rem;
        border-radius: 14px;
        box-shadow: 0 10px 24px rgba(0, 0, 0, 0.18);
    }
</style>
"""

HEALTHCARE_TABS = ["Executive Overview", "Cohort Analysis", "Predictive Insights", "Cost & Performance", "Data Quality Review", "Export Center", "Insights Assistant"]
UPLOADED_BASE_TABS = ["Auto-Analysis", "AI Dataset Explainer", "Auto Generated Dashboard", "Data Quality Insights", "Dataset Comparison"]
QUICK_ANALYSIS_TABS = ["Auto-Analysis", "AI Dataset Explainer", "Auto Generated Dashboard", "Data Quality Insights", "Dataset Comparison", "Export Center"]
FIELD_ENABLEMENT_HELP = {
    "diagnosis": "Enables cohort analysis and insights.",
    "department": "Enables benchmarking and cost drivers.",
    "date": "Enables trends.",
    "length_of_stay": "Enables LOS metrics.",
    "cost": "Enables cost analysis.",
    "readmission": "Enables predictive analytics.",
    "age": "Enables cohort analysis.",
    "gender": "Enables cohort analysis.",
}


def apply_theme() -> None:
    st.markdown(CARD_STYLE, unsafe_allow_html=True)


def _make_chart_key(*parts: object) -> str:
    return "_".join(str(part).strip().lower().replace(" ", "_").replace("-", "_").replace("/", "_") for part in parts if str(part).strip())


def _plot_or_message(container, figure, message: str = "No data available for the current selection.", chart_key: str | None = None) -> None:
    if container is st:
        if figure is None:
            st.info(message)
            return
        st.plotly_chart(figure, key=chart_key, width="stretch")
        return
    with container:
        if figure is None:
            st.info(message)
            return
        st.plotly_chart(figure, key=chart_key, width="stretch")


def _prettify_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    return dataframe.rename(columns={column: column.replace("_", " ").title() for column in dataframe.columns})


def _format_percent(value) -> str:
    return f"{value:.1%}" if pd.notna(value) else "N/A"


def _format_currency(value) -> str:
    return f"${value:,.2f}" if pd.notna(value) else "N/A"


def _format_number(value, decimals: int = 1) -> str:
    return f"{value:.{decimals}f}" if pd.notna(value) else "N/A"


def _has_columns(data: pd.DataFrame, columns: list[str]) -> bool:
    return all(column in data.columns for column in columns)


def _healthcare_mode_enabled(schema_info: dict[str, object], uploaded: bool) -> bool:
    if not uploaded:
        return True
    capabilities = dataset_capabilities(schema_info)
    return capabilities["kpi_analysis"] and capabilities["cost_analysis"] and capabilities["cohort_analysis"]


def _load_raw_datasets(uploaded_files) -> tuple[dict[str, pd.DataFrame], str]:
    if uploaded_files:
        datasets: dict[str, pd.DataFrame] = {}
        for uploaded_file in uploaded_files:
            datasets[uploaded_file.name] = _cached_load_csv(uploaded_file.getvalue(), uploaded_file.name)
        default_name = list(datasets.keys())[0]
        return datasets, default_name
    sample = load_hospital_data(DATA_PATH)
    return {"synthetic_hospital_data.csv": sample}, "synthetic_hospital_data.csv"



@st.cache_data(show_spinner=False)
def _cached_load_csv(file_bytes: bytes, file_name: str) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(file_bytes))


@st.cache_data(show_spinner=False)
def _cached_detect_schema(data: pd.DataFrame) -> dict[str, object]:
    return detect_schema(data)


@st.cache_data(show_spinner=False)
def _cached_profile(data: pd.DataFrame) -> dict[str, object]:
    return profile_dataset(data)


@st.cache_data(show_spinner=False)
def _cached_quick_profile(data: pd.DataFrame) -> dict[str, object]:
    return quick_profile_dataset(data)


@st.cache_data(show_spinner=False)
def _cached_explainer(data: pd.DataFrame, matched_schema_items: tuple[tuple[str, str], ...]) -> dict[str, object]:
    return explain_dataset(data, dict(matched_schema_items))


@st.cache_data(show_spinner=False)
def _cached_quality_insights(data: pd.DataFrame) -> dict[str, object]:
    return build_quality_insights(data)


@st.cache_data(show_spinner=False)
def _cached_dashboard_spec(data: pd.DataFrame) -> list[dict[str, str]]:
    return build_auto_generated_dashboard_spec(data)

def _estimate_remaining(elapsed_seconds: float, progress_value: int) -> float:
    if progress_value <= 0:
        return 0.0
    return max((elapsed_seconds / progress_value) * (100 - progress_value), 0.0)


def _update_progress(progress_bar, status_placeholder, metrics_placeholder, start_time: float, progress_value: int, step_label: str) -> None:
    elapsed = time.perf_counter() - start_time
    remaining = _estimate_remaining(elapsed, progress_value)
    progress_bar.progress(progress_value)
    status_placeholder.markdown(f"**{progress_value}%**  {step_label}")
    metrics_placeholder.caption(f"Elapsed: {elapsed:.1f}s | Estimated remaining: {remaining:.1f}s")


def _build_generic_export_text(profile: dict[str, object], schema_info: dict[str, object], source_name: str) -> bytes:
    buffer = io.StringIO()
    buffer.write("Clinical Outcomes Explorer Dataset Summary\n")
    buffer.write("=" * 44 + "\n\n")
    buffer.write(f"Source: {source_name}\n")
    buffer.write(f"Rows: {profile['row_count']:,}\n")
    buffer.write(f"Columns: {profile['column_count']:,}\n")
    buffer.write(f"Detected Date Columns: {', '.join(profile['date_columns']) if profile['date_columns'] else 'None'}\n")
    buffer.write(f"Healthcare Mapping Coverage: {schema_coverage_percent(schema_info['matched_schema']):.0%}\n")
    return buffer.getvalue().encode("utf-8")


def _short_label(label: str, max_length: int = 22) -> str:
    if len(label) <= max_length:
        return label
    return label[: max_length - 1].rstrip() + "..."


def _widget_label(label: str, max_length: int = 24) -> tuple[str, str | None]:
    short = _short_label(label, max_length=max_length)
    return short, (label if short != label else None)


def _resolve_available_sections(uploaded: bool, full_ready: bool, healthcare_enabled: bool) -> list[str]:
    if uploaded and not full_ready:
        return list(dict.fromkeys(QUICK_ANALYSIS_TABS))
    if uploaded:
        sections = list(UPLOADED_BASE_TABS)
        if "Export Center" not in sections:
            sections.append("Export Center")
        if healthcare_enabled:
            sections.extend(HEALTHCARE_TABS)
        return list(dict.fromkeys(sections))
    return list(dict.fromkeys(HEALTHCARE_TABS))


def resolve_current_section(available_sections: list[str], requested_section: str | None, fallback_default: str) -> tuple[str, bool, str | None]:
    if not available_sections:
        return fallback_default, True, fallback_default
    if requested_section in available_sections:
        return requested_section, False, None
    fallback_target = fallback_default if fallback_default in available_sections else available_sections[0]
    return fallback_target, True, fallback_target


def build_default_filter_state(data: pd.DataFrame) -> dict[str, object]:
    defaults: dict[str, object] = {"text_search": ""}
    default_date_ranges: dict[str, tuple[object, object]] = {}
    date_columns = detect_date_columns(data)
    for column in date_columns[:2]:
        parsed = pd.to_datetime(data[column], errors="coerce", format="mixed")
        valid = parsed.dropna()
        valid_ratio = float(parsed.notna().mean()) if len(parsed) else 0.0
        if valid.empty or valid_ratio < 0.5:
            continue
        default_date_ranges[f"date_{column}"] = (valid.min().date(), valid.max().date())

    detected_dates = detect_date_columns(data)
    numeric_columns = [column for column in data.columns if infer_column_type(data[column], detected_dates, column) == "numeric"]
    for column in numeric_columns[:5]:
        numeric = pd.to_numeric(data[column], errors="coerce")
        valid = numeric.dropna()
        if valid.empty:
            continue
        min_value = float(valid.min())
        max_value = float(valid.max())
        if min_value == max_value:
            continue
        defaults[f"num_{column}"] = (min_value, max_value)

    categorical_columns = [column for column in data.columns if infer_column_type(data[column], detected_dates, column) in {"categorical", "boolean"} and data[column].nunique(dropna=True) <= 20]
    for column in categorical_columns[:5]:
        defaults[f"cat_{column}"] = []

    defaults.update(default_date_ranges)
    defaults["default_date_ranges"] = default_date_ranges
    defaults["filter_widget_keys"] = [key for key in defaults.keys() if key not in {"default_date_ranges", "filter_widget_keys"}]
    return defaults


def reset_filter_state(default_filter_state: dict[str, object], reset_reason: str = "manual") -> None:
    filter_keys = set(default_filter_state.get("filter_widget_keys", []))
    for key in list(st.session_state.keys()):
        if key == "text_search" or key.startswith("date_") or key.startswith("num_") or key.startswith("cat_"):
            if key not in filter_keys:
                st.session_state.pop(key, None)
    for key in filter_keys:
        st.session_state[key] = default_filter_state[key]
    st.session_state["filter_reset_triggered_this_rerun"] = reset_reason


def render_mapping_overrides(raw_data: pd.DataFrame, schema_info: dict[str, object]) -> tuple[dict[str, str], dict[str, str]]:
    st.sidebar.markdown("**Detected Column Mapping**")
    mapping = dict(schema_info["matched_schema"])
    auto_mapping = dict(schema_info["matched_schema"])
    statuses: dict[str, str] = {}
    options = ["Auto-detect"] + list(raw_data.columns)
    for field in EXPECTED_FIELDS:
        auto_value = auto_mapping.get(field, "Auto-detect")
        default_value = auto_value if auto_value in options else "Auto-detect"
        field_label = FIELD_LABELS.get(field, field.title())
        short_label, help_text = _widget_label(field_label)
        selection = st.sidebar.selectbox(short_label, options=options, index=options.index(default_value), key=f"mapping_{field}", help=help_text)
        if selection == "Auto-detect":
            if field in auto_mapping:
                mapping[field] = auto_mapping[field]
                statuses[field] = "Mapped Automatically"
            else:
                mapping.pop(field, None)
                statuses[field] = "Not Mapped"
        else:
            mapping[field] = selection
            statuses[field] = "Mapped Automatically" if selection == auto_mapping.get(field) else "Manually Selected"
    return mapping, statuses


def apply_dynamic_filters(data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object], dict[str, object]]:
    filtered = data.copy()
    selected_filters: dict[str, object] = {}
    default_filter_state = build_default_filter_state(data)
    dataset_filter_signature = f"{len(data)}:{len(data.columns)}:{'|'.join(list(data.columns)[:20])}"
    reset_reason = False
    if st.session_state.get("filter_dataset_signature") != dataset_filter_signature:
        reset_filter_state(default_filter_state, reset_reason="dataset_switch")
        st.session_state["filter_dataset_signature"] = dataset_filter_signature
        reset_reason = True

    meta = {
        "text_search": "none",
        "text_filter_active": False,
        "date_filter_active": False,
        "numeric_filter_count": 0,
        "categorical_filter_count": 0,
        "rows_total": len(data),
        "rows_after_text": len(data),
        "rows_after_date": len(data),
        "rows_after_numeric": len(data),
        "rows_after_categorical": len(data),
        "rows_in_scope": len(data),
        "active_filter_count": 0,
        "fallback_used": False,
        "skipped_filters": [],
        "invalid_filter_warnings": [],
        "default_date_range": default_filter_state.get("default_date_ranges", {}),
        "active_date_range": None,
        "reset_triggered_this_rerun": st.session_state.get("filter_reset_triggered_this_rerun", False),
    }
    st.sidebar.markdown("**Dataset Filters**")
    reset_clicked = st.sidebar.button("Reset Filters", key="reset_filters", width="stretch")
    if reset_clicked:
        reset_filter_state(default_filter_state, reset_reason="manual")
        st.session_state["filter_dataset_signature"] = dataset_filter_signature
        meta["reset_triggered_this_rerun"] = "manual"
        reset_reason = True

    text_label, text_help = _widget_label("Text Search", max_length=18)
    text_query = st.sidebar.text_input(text_label, placeholder="Search across all columns", key="text_search", help=text_help)
    clean_text_query = text_query.strip() if isinstance(text_query, str) else ""
    if clean_text_query:
        mask = filtered.astype(str).apply(lambda column: column.str.contains(clean_text_query, case=False, na=False))
        filtered = filtered[mask.any(axis=1)]
        selected_filters["Text Search"] = clean_text_query
        meta["text_search"] = clean_text_query
        meta["text_filter_active"] = True
        meta["active_filter_count"] += 1
    meta["rows_after_text"] = len(filtered)

    date_columns = detect_date_columns(data)
    date_ranges: list[str] = []
    active_date_range: tuple[object, object] | None = None
    for column in date_columns[:2]:
        parsed = pd.to_datetime(data[column], errors="coerce", format="mixed")
        valid = parsed.dropna()
        valid_ratio = float(parsed.notna().mean()) if len(parsed) else 0.0
        if valid.empty or valid_ratio < 0.5:
            meta["skipped_filters"].append(f"Skipped date filter for {column}: unable to parse enough values reliably.")
            continue
        start, end = valid.min().date(), valid.max().date()
        label, help_text = _widget_label(f"{column.replace('_', ' ').title()} Range")
        widget_key = f"date_{column}"
        if widget_key not in st.session_state:
            st.session_state[widget_key] = default_filter_state.get(widget_key, (start, end))
        chosen = st.sidebar.date_input(label, key=widget_key, help=help_text)
        if not isinstance(chosen, (tuple, list)) or len(chosen) != 2 or chosen[0] is None or chosen[1] is None:
            meta["invalid_filter_warnings"].append(f"Incomplete date range for {column.replace('_', ' ').title()}. The filter was not applied.")
            continue
        chosen_start, chosen_end = chosen[0], chosen[1]
        active_date_range = (chosen_start, chosen_end)
        if chosen_start > chosen_end:
            meta["invalid_filter_warnings"].append(f"Invalid date range for {column.replace('_', ' ').title()}. The filter was not applied.")
            continue
        if chosen_start != start or chosen_end != end:
            current_parsed = parsed.loc[filtered.index]
            filtered = filtered[current_parsed.between(pd.Timestamp(chosen_start), pd.Timestamp(chosen_end), inclusive="both")]
            selected_filters[column] = f"{chosen_start} to {chosen_end}"
            date_ranges.append(f"{chosen_start} to {chosen_end}")
            meta["date_filter_active"] = True
            meta["active_filter_count"] += 1
    if date_ranges:
        selected_filters["Date Range"] = date_ranges[0]
    meta["active_date_range"] = active_date_range
    meta["rows_after_date"] = len(filtered)

    detected_dates = detect_date_columns(data)
    numeric_columns = [column for column in data.columns if infer_column_type(data[column], detected_dates, column) == "numeric"]
    for column in numeric_columns[:5]:
        numeric = pd.to_numeric(data[column], errors="coerce")
        valid = numeric.dropna()
        if valid.empty:
            meta["skipped_filters"].append(f"Skipped numeric filter for {column}: all values became NaN after numeric conversion.")
            continue
        min_value = float(valid.min())
        max_value = float(valid.max())
        if min_value == max_value:
            continue
        label, help_text = _widget_label(f"{column.replace('_', ' ').title()} Range")
        widget_key = f"num_{column}"
        if widget_key not in st.session_state:
            st.session_state[widget_key] = default_filter_state.get(widget_key, (min_value, max_value))
        chosen = st.sidebar.slider(label, min_value=min_value, max_value=max_value, key=widget_key, help=help_text)
        if chosen[0] != min_value or chosen[1] != max_value:
            current_numeric = numeric.loc[filtered.index]
            filtered = filtered[current_numeric.between(chosen[0], chosen[1], inclusive="both") | current_numeric.isna()]
            selected_filters[column] = f"{chosen[0]:.2f} to {chosen[1]:.2f}"
            meta["numeric_filter_count"] += 1
            meta["active_filter_count"] += 1
    meta["rows_after_numeric"] = len(filtered)

    categorical_columns = [column for column in data.columns if infer_column_type(data[column], detected_dates, column) in {"categorical", "boolean"} and data[column].nunique(dropna=True) <= 20]
    for column in categorical_columns[:5]:
        options = data[column].dropna().astype(str).sort_values().unique().tolist()
        if not options:
            continue
        label, help_text = _widget_label(column.replace("_", " ").title())
        widget_key = f"cat_{column}"
        if widget_key not in st.session_state:
            st.session_state[widget_key] = default_filter_state.get(widget_key, [])
        selected = st.sidebar.multiselect(label, options=options, key=widget_key, help=help_text)
        if selected:
            filtered = filtered[filtered[column].astype(str).isin(selected)]
            selected_filters[column] = ", ".join(selected)
            meta["categorical_filter_count"] += 1
            meta["active_filter_count"] += 1
    meta["rows_after_categorical"] = len(filtered)

    if filtered.empty and meta["active_filter_count"] == 0:
        filtered = data.copy()
        meta["fallback_used"] = True
        meta["rows_after_text"] = len(data)
        meta["rows_after_date"] = len(data)
        meta["rows_after_numeric"] = len(data)
        meta["rows_after_categorical"] = len(data)

    meta["rows_in_scope"] = len(filtered)
    meta["filter_session_keys"] = {key: st.session_state.get(key) for key in st.session_state.keys() if key == "text_search" or key.startswith("date_") or key.startswith("num_") or key.startswith("cat_")}
    if not reset_reason:
        st.session_state["filter_reset_triggered_this_rerun"] = False
    return filtered, selected_filters, meta


def render_active_filters_summary(dataset_name: str, selected_filters: dict[str, object], filter_meta: dict[str, object]) -> None:
    st.subheader("Active Filters")
    lines = [
        f"Dataset: {dataset_name}",
        f"Text Search: {'active' if filter_meta.get('text_filter_active') else 'none'}",
        f"Date Filter: {selected_filters.get('Date Range', 'full range') if filter_meta.get('date_filter_active') else 'full range'}",
        f"Numeric Filters: {filter_meta.get('numeric_filter_count', 0)} active",
        f"Categorical Filters: {filter_meta.get('categorical_filter_count', 0)} active",
        f"Rows in Scope: {filter_meta.get('rows_in_scope', 0):,} of {filter_meta.get('rows_total', 0):,}",
    ]
    for line in lines:
        st.markdown(f"- {line}")


def render_detected_mapping_section(schema_info: dict[str, object], mapping_status: dict[str, str], filtered_data: pd.DataFrame) -> None:
    st.markdown("**Detected Column Mapping**")
    mode_cols = st.columns(3)
    mode_cols[0].metric("Detected Dataset Mode", schema_info.get("dataset_mode", "generic_tabular").replace("_", " ").title())
    mode_cols[1].metric("Mode Confidence", f"{schema_info.get('mode_confidence', 0.0):.0%}")
    mode_cols[2].metric("Mapped Fields", f"{len(schema_info.get('matched_schema', {}))}/{len(build_mapping_table({}, None))}")
    st.caption(schema_info.get("mode_reason", "Mode classification details are unavailable."))

    mapping_table = build_mapping_table(schema_info["matched_schema"], schema_info.get("match_details"))
    mapping_table["status"] = mapping_table["internal_field"].map(lambda field: mapping_status.get(field, "Mapped Automatically") if field in schema_info["matched_schema"] else "Not Mapped")
    mapping_table["sample_value"] = mapping_table.apply(lambda row: filtered_data[row["mapped_column"]].dropna().astype(str).iloc[0] if row["mapped_column"] in filtered_data.columns and not filtered_data[row["mapped_column"]].dropna().empty else "-", axis=1)
    mapping_table["help_text"] = mapping_table["internal_field"].map(lambda field: FIELD_ENABLEMENT_HELP.get(field, "Supports dataset understanding or downstream analytics."))
    st.dataframe(_prettify_columns(mapping_table[["field_label", "mapped_column", "status", "sample_value", "help_text", "confidence_score", "reason"]]), width="stretch", height=420)

    unmatched = schema_info.get("unmatched_columns", [])
    missing = schema_info.get("missing_fields", [])
    detail_cols = st.columns(2)
    with detail_cols[0]:
        st.markdown("**Unmapped Source Columns**")
        st.write(unmatched if unmatched else ["None"])
    with detail_cols[1]:
        st.markdown("**Unmapped Internal Fields**")
        st.write([FIELD_LABELS.get(field, field.replace("_", " ").title()) for field in missing] if missing else ["None"])


def render_auto_analysis(filtered_data: pd.DataFrame, schema_info: dict[str, object], source_name: str, healthcare_enabled: bool, uploaded: bool, mapping_status: dict[str, str], filter_meta: dict[str, object], selected_filters: dict[str, object], profile: dict[str, object]) -> dict[str, object]:
    st.subheader("Auto-Analysis Overview")
    st.caption("A flexible first-look profile for any uploaded CSV, with automatic structure detection, completeness checks, and exploratory analytics.")
    render_active_filters_summary(source_name, selected_filters, filter_meta)
    cards = st.columns(5)
    cards[0].metric("Rows", f"{profile['row_count']:,}")
    cards[1].metric("Columns", f"{profile['column_count']:,}")
    cards[2].metric("Duplicates", f"{profile['duplicate_row_count']:,}")
    cards[3].metric("Date Columns", f"{len(profile['date_columns']):,}")
    cards[4].metric("Schema Coverage", f"{schema_coverage_percent(schema_info['matched_schema']):.0%}")
    if profile.get("is_sampled"):
        st.info("Large dataset detected. Analysis uses sampling for faster performance.")
    capabilities = dataset_capabilities(schema_info)
    if uploaded and capabilities.get("patient_level_mode") and healthcare_enabled:
        st.success("Patient-Level Healthcare Analytics mode is enabled. Clinical, operational, and predictive tabs are ready for this dataset.")
    elif uploaded and capabilities.get("hospital_measure_mode"):
        st.info("Hospital Measure Reporting mode was detected. Generic analysis remains available, while patient-level healthcare sections stay disabled because the upload looks like measure reporting data.")
    elif uploaded:
        missing_core = [FIELD_LABELS.get(field, field.replace("_", " ").title()) for field in ["age", "diagnosis", "department", "length_of_stay", "cost", "readmission"] if field not in schema_info.get("matched_schema", {})]
        st.info("Advanced healthcare analytics are not enabled for this upload because patient-level clinical coverage is not yet strong enough. Generic profiling remains fully available.")
        st.caption(f"Detected mode: {schema_info.get('dataset_mode', 'generic_tabular').replace('_', ' ').title()} | Why disabled: {schema_info.get('mode_reason', 'Insufficient patient-level signals.')} ")
        if missing_core:
            st.caption(f"Suggested mappings to unlock more patient analytics: {', '.join(missing_core[:6])}")
    render_detected_mapping_section(schema_info, mapping_status, filtered_data)
    cols = st.columns(2)
    cols[0].dataframe(_prettify_columns(profile["column_profile"]), width="stretch", height=320)
    cols[1].dataframe(_prettify_columns(profile["missing_values"]), width="stretch", height=320)
    if not profile["numeric_summary"].empty:
        st.markdown("**Numeric Summary Statistics**")
        st.dataframe(_prettify_columns(profile["numeric_summary"].round(3)), width="stretch")
    top_category_items = list(profile["top_categories"].items())[:3]
    if top_category_items:
        st.markdown("**Top Category Snapshots**")
        cols = st.columns(2)
        for index, (column_name, summary) in enumerate(top_category_items):
            _plot_or_message(cols[index % 2], create_category_bar(summary, column_name), f"No category chart is available for {column_name}.", chart_key=_make_chart_key("auto_analysis", "category", column_name, index))
    return profile

def render_dataset_explainer(explainer: dict[str, object]) -> None:
    st.subheader("AI Dataset Explainer")
    st.caption("A rule-based plain-English interpretation of the uploaded dataset, generated directly from its structure and observed patterns.")
    st.markdown("**Dataset Summary**")
    for line in explainer["summary"]:
        st.markdown(f"- {line}")
    st.markdown("**Likely Dataset Theme**")
    st.write(explainer["theme"])
    st.markdown("**Key Observations**")
    for line in explainer["observations"]:
        st.markdown(f"- {line}")
    st.markdown("**Recommended Next Steps**")
    for line in explainer["next_steps"]:
        st.markdown(f"- {line}")


def render_auto_generated_dashboard(filtered_data: pd.DataFrame, profile: dict[str, object], specs: list[dict[str, str]]) -> None:
    st.subheader("Auto Generated Dashboard")
    st.caption("Charts are selected automatically based on the available column types in the current filtered dataset.")
    if not specs:
        st.info("No auto-generated charts are available for the current dataset and filter selection.")
        return
    cols = st.columns(2)
    for index, spec in enumerate(specs[:10]):
        container = cols[index % 2]
        figure = None
        if spec["chart_type"] == "histogram":
            figure = create_numeric_histogram(filtered_data, spec["column"])
        elif spec["chart_type"] == "category":
            figure = create_category_bar(profile["top_categories"].get(spec["column"], pd.DataFrame()), spec["column"])
        elif spec["chart_type"] == "correlation":
            figure = create_correlation_heatmap(profile["correlation_matrix"])
        elif spec["chart_type"] == "trend":
            figure = create_generic_trend_chart(profile["trend_data"], spec["column"], spec["title"], "Value")
        elif spec["chart_type"] == "scatter":
            x_column, y_column = spec["column"].split("|")
            figure = create_scatter_plot(filtered_data, x_column, y_column)
        _plot_or_message(container, figure, f"Unable to generate {spec['title'].lower()} for the current selection.", chart_key=_make_chart_key("auto_dashboard", spec["chart_type"], spec["column"], index))

def render_data_quality_insights(filtered_data: pd.DataFrame, profile: dict[str, object], quality: dict[str, object] | None = None) -> None:
    st.subheader("Data Quality Insights")
    quality = quality or _cached_quality_insights(filtered_data)
    cols = st.columns(4)
    cols[0].metric("Data Quality Score", f"{quality['quality_score']}/100")
    cols[1].metric("Duplicate Rows", f"{quality['duplicate_row_count']:,}")
    cols[2].metric("Invalid Numeric Values", f"{quality['invalid_numeric_values']:,}")
    cols[3].metric("Constant Columns", f"{len(quality['constant_columns']):,}")
    tabs = st.tabs(["Missing Data", "Outliers", "Duplication"])
    with tabs[0]:
        st.dataframe(_prettify_columns(quality["missing_values"]), width="stretch")
    with tabs[1]:
        outlier_table = quality["outlier_summary"].copy()
        if not outlier_table.empty:
            outlier_table["outlier_pct"] = outlier_table["outlier_pct"].map(_format_percent)
        st.dataframe(_prettify_columns(outlier_table), width="stretch")
    with tabs[2]:
        dup = quality["duplication_summary"].copy()
        if not dup.empty:
            dup["duplication_pct"] = dup["duplication_pct"].map(_format_percent)
        st.dataframe(_prettify_columns(dup), width="stretch")


def render_dataset_comparison(raw_datasets: dict[str, pd.DataFrame], precomputed_profiles: dict[str, dict[str, object]] | None = None) -> None:
    st.subheader("Dataset Comparison")
    if len(raw_datasets) < 2:
        st.info("Upload at least two CSV files to compare dataset size, shared numeric averages, and time trends across files.")
        return
    names = list(raw_datasets.keys())
    left_name = st.selectbox("Dataset A", options=names, index=0)
    right_name = st.selectbox("Dataset B", options=[name for name in names if name != left_name], index=0)
    left = raw_datasets[left_name]
    right = raw_datasets[right_name]
    left_profile = (precomputed_profiles or {}).get(left_name) or profile_dataset(left)
    right_profile = (precomputed_profiles or {}).get(right_name) or profile_dataset(right)
    compare = pd.DataFrame([
        {"metric": "Rows", left_name: len(left), right_name: len(right)},
        {"metric": "Columns", left_name: len(left.columns), right_name: len(right.columns)},
        {"metric": "Duplicate Rows", left_name: left_profile["duplicate_row_count"], right_name: right_profile["duplicate_row_count"]},
    ])
    st.dataframe(compare, width="stretch")
    shared_numeric = [column for column in left.columns if column in right.columns and infer_column_type(left[column], detect_date_columns(left), column) == "numeric"]
    if shared_numeric:
        mean_compare = pd.DataFrame([{"column": column, left_name: pd.to_numeric(left[column], errors="coerce").mean(), right_name: pd.to_numeric(right[column], errors="coerce").mean()} for column in shared_numeric[:10]])
        st.dataframe(_prettify_columns(mean_compare.round(3)), width="stretch")
        figure = px.bar(mean_compare.melt(id_vars="column", var_name="dataset", value_name="average"), x="column", y="average", color="dataset", barmode="group", title="Shared Numeric Mean Comparison")
        _plot_or_message(st, figure, chart_key=_make_chart_key("dataset_comparison", "mean_compare"))


def render_benchmark_baselines(data: pd.DataFrame) -> None:
    st.subheader("Benchmark Comparison")
    baseline_readmission = st.number_input("Readmission Rate Benchmark", min_value=0.0, max_value=1.0, value=0.12, step=0.01, format="%.2f")
    baseline_los = st.number_input("Average LOS Benchmark (Days)", min_value=0.0, value=5.0, step=0.5)
    baseline_cost = st.number_input("Average Cost Benchmark", min_value=0.0, value=10000.0, step=500.0)
    comparison = pd.DataFrame([
        {"metric": "Readmission Rate", "current_value": calculate_readmission_rate(data), "benchmark": baseline_readmission},
        {"metric": "Average Length of Stay", "current_value": calculate_average_length_of_stay(data), "benchmark": baseline_los},
        {"metric": "Average Cost", "current_value": calculate_average_cost(data), "benchmark": baseline_cost},
    ])
    comparison["difference"] = comparison["current_value"] - comparison["benchmark"]
    comparison["status"] = comparison.apply(lambda row: "Better" if row["current_value"] < row["benchmark"] else "Needs Attention", axis=1)
    st.dataframe(_prettify_columns(comparison.round(3)), width="stretch")


def render_executive_summary(data: pd.DataFrame, scored_data: pd.DataFrame | None, summary: dict[str, object] | None = None) -> None:
    summary = summary or build_executive_summary(data, scored_data)
    cards = summary["cards"]
    st.subheader("Executive Summary")
    metric_cols = st.columns(6)
    metric_cols[0].metric("Admissions in Scope", f"{cards['total_admissions']:,}")
    metric_cols[1].metric("Estimated Readmissions", f"{cards['total_estimated_readmissions']:.0f}")
    metric_cols[2].metric("High-Risk Patients", f"{cards['high_risk_patients']:,}")
    metric_cols[3].metric("Total Cost", f"${cards['total_cost']:,.0f}")
    metric_cols[4].metric("Potentially Preventable Readmissions", f"{cards['estimated_preventable_readmissions']:.1f}")
    metric_cols[5].metric("1-Day LOS Savings Opportunity", f"${cards['estimated_savings_opportunity']:,.0f}")
    st.markdown(summary["narrative"])


def render_core_kpis(data: pd.DataFrame, metrics: dict[str, float | int] | None = None) -> dict[str, float | int]:
    st.subheader("Operational KPI Snapshot")
    metrics = metrics or get_key_metrics(data)
    cols = st.columns(4)
    cols[0].metric("Admissions in Scope", f"{metrics['total_admissions']:,}")
    cols[1].metric("Readmission Rate", f"{metrics['readmission_rate']:.1%}")
    cols[2].metric("Average Length of Stay", f"{metrics['average_length_of_stay']:.1f} days")
    cols[3].metric("Average Cost per Admission", f"${metrics['average_cost']:,.2f}")
    return metrics


def render_operational_overview(data: pd.DataFrame, figures: dict[str, object] | None = None) -> None:
    st.subheader("Operational Performance Overview")
    figures = figures or {
        "cost_by_department": create_cost_by_department_chart(data),
        "readmission_by_diagnosis": create_readmission_by_diagnosis_chart(data),
    }
    cols = st.columns(2)
    _plot_or_message(cols[0], figures.get("cost_by_department"), chart_key=_make_chart_key("operational_overview", "cost_by_department"))
    _plot_or_message(cols[1], figures.get("readmission_by_diagnosis"), chart_key=_make_chart_key("operational_overview", "readmission_by_diagnosis"))


def render_trend_analysis(data: pd.DataFrame, trend_data: pd.DataFrame | None = None, figures: dict[str, object] | None = None) -> None:
    st.subheader("Time-Based Performance Trends")
    trend_data = trend_data if trend_data is not None else build_monthly_trends(data)
    if trend_data.empty:
        st.caption("Trend analysis is not available for the current selection because no usable mapped date values remain.")
        return
    figures = figures or {
        "admissions": create_monthly_trend_chart(trend_data, "admissions", "Monthly Admissions Trend", "Admissions"),
        "average_cost": create_monthly_trend_chart(trend_data, "average_cost", "Monthly Average Cost Trend", "Average Cost"),
        "readmission_rate": create_monthly_trend_chart(trend_data, "readmission_rate", "Monthly Readmission Rate Trend", "Readmission Rate"),
        "average_length_of_stay": create_monthly_trend_chart(trend_data, "average_length_of_stay", "Monthly Average Length of Stay Trend", "Average Length of Stay"),
    }
    cols = st.columns(2)
    _plot_or_message(cols[0], figures.get("admissions"), chart_key=_make_chart_key("trend_chart", "admissions"))
    _plot_or_message(cols[1], figures.get("average_cost"), chart_key=_make_chart_key("trend_chart", "average_cost"))
    cols = st.columns(2)
    _plot_or_message(cols[0], figures.get("readmission_rate"), chart_key=_make_chart_key("trend_chart", "readmission_rate"))
    _plot_or_message(cols[1], figures.get("average_length_of_stay"), chart_key=_make_chart_key("trend_chart", "average_length_of_stay"))


def render_benchmarking(filtered_data: pd.DataFrame, overall_data: pd.DataFrame, scored_filtered: pd.DataFrame | None, scored_overall: pd.DataFrame | None, comparison: pd.DataFrame | None = None, figure=None) -> None:
    st.subheader("Internal Benchmark Comparison")
    comparison = comparison if comparison is not None else build_benchmarking_summary(filtered_data, overall_data, scored_filtered, scored_overall)
    st.dataframe(_prettify_columns(comparison.round(3)), width="stretch")
    _plot_or_message(st, figure or create_comparison_chart(comparison), chart_key=_make_chart_key("benchmark", "internal_comparison"))


def render_operational_alerts(data: pd.DataFrame, overall_data: pd.DataFrame, quality: dict[str, object], scored_filtered: pd.DataFrame | None, scored_overall: pd.DataFrame | None, alerts: list[dict[str, str]] | None = None) -> list[dict[str, str]]:
    st.subheader("Operational Alerts")
    alerts = list(alerts) if alerts is not None else generate_operational_alerts(data, overall_data, quality, scored_filtered, scored_overall)
    if _has_columns(data, ["readmission"]):
        readmission_rate = calculate_readmission_rate(data)
        if readmission_rate > 0.15:
            alerts.insert(0, {"level": "warning", "title": "High Readmission Rate Alert", "message": f"Current readmission rate is {readmission_rate:.1%}, above the 15% monitoring threshold."})
    if _has_columns(data, ["length_of_stay"]):
        avg_los = calculate_average_length_of_stay(data)
        if avg_los > 7:
            alerts.append({"level": "warning", "title": "Length of Stay Alert", "message": f"Average length of stay is {avg_los:.1f} days, above the 7-day threshold."})
    if _has_columns(data, ["cost"]):
        avg_cost = calculate_average_cost(data)
        if avg_cost > 10000:
            alerts.append({"level": "info", "title": "Cost Alert", "message": f"Average cost is ${avg_cost:,.0f}, above the default benchmark of $10,000."})
    if not alerts:
        st.success("No active operational alerts are currently triggered.")
        return []
    seen: set[str] = set()
    deduped: list[dict[str, str]] = []
    for alert in alerts:
        key = f"{alert['title']}|{alert['message']}"
        if key not in seen:
            seen.add(key)
            deduped.append(alert)
    for alert in deduped:
        getattr(st, alert["level"])(f"{alert['title']}: {alert['message']}")
    return deduped


def render_patient_cohort_explorer(data: pd.DataFrame, cohort_summaries: dict[str, pd.DataFrame] | None = None) -> None:
    st.subheader("Patient Cohort Explorer")
    enriched = add_derived_buckets(data)
    tabs = st.tabs(list(COHORT_FIELDS.values()))
    for tab, (cohort_key, label) in zip(tabs, COHORT_FIELDS.items()):
        with tab:
            summary = cohort_summaries.get(cohort_key) if cohort_summaries else build_cohort_summary(enriched, cohort_key)
            if summary.empty:
                st.caption("No cohort summary is available for this cohort view under the active filters.")
                continue
            cols = st.columns(3)
            _plot_or_message(cols[0], create_cohort_metric_chart(summary, cohort_key, "readmission_rate", f"Readmission Rate by {label}", "Readmission Rate"), chart_key=_make_chart_key("cohort_chart", cohort_key, "readmission_rate"))
            _plot_or_message(cols[1], create_cohort_metric_chart(summary, cohort_key, "average_cost", f"Average Cost by {label}", "Average Cost"), chart_key=_make_chart_key("cohort_chart", cohort_key, "average_cost"))
            _plot_or_message(cols[2], create_cohort_metric_chart(summary, cohort_key, "average_length_of_stay", f"Average Length of Stay by {label}", "Average Length of Stay"), chart_key=_make_chart_key("cohort_chart", cohort_key, "average_length_of_stay"))
            display = summary.copy()
            display["readmission_rate"] = display["readmission_rate"].map(_format_percent)
            display["average_cost"] = display["average_cost"].map(_format_currency)
            display["average_length_of_stay"] = display["average_length_of_stay"].map(lambda value: f"{_format_number(value)} days" if pd.notna(value) else "N/A")
            st.dataframe(_prettify_columns(display), width="stretch")


def render_custom_cohort_builder(data: pd.DataFrame) -> None:
    st.subheader("Custom Cohort Builder")
    rules = {}
    if "age" in data.columns:
        try:
            min_age, max_age = get_age_bounds(data)
            selected = st.slider("Age Rule", min_value=min_age, max_value=max_age, value=(min_age, max_age), key="custom_age")
            rules["min_age"], rules["max_age"] = selected
        except ValueError:
            st.caption("Age-based rules are unavailable because the mapped age field is not numeric.")
    if "gender" in data.columns:
        rules["gender"] = st.multiselect("Gender Equals", options=sorted(data["gender"].dropna().astype(str).unique().tolist()), key="custom_gender")
    if "diagnosis" in data.columns:
        rules["diagnosis"] = st.multiselect("Diagnosis Equals", options=sorted(data["diagnosis"].dropna().astype(str).unique().tolist()), key="custom_diagnosis")
    if "department" in data.columns:
        rules["department"] = st.multiselect("Department Equals", options=sorted(data["department"].dropna().astype(str).unique().tolist()), key="custom_department")
    if "comorbidity_score" in data.columns:
        numeric = to_numeric(data["comorbidity_score"]).dropna()
        rules["comorbidity_min"] = st.slider("Comorbidity Score Threshold", 0, int(numeric.max()) if not numeric.empty else 10, 0, key="custom_comorbidity")
    cohort = build_custom_cohort(data, rules)
    if cohort.empty:
        st.info("No records match the current custom cohort definition.")
        return
    cols = st.columns(4)
    cols[0].metric("Cohort Size", f"{len(cohort):,}")
    cols[1].metric("Readmission Rate", f"{calculate_readmission_rate(cohort):.1%}")
    cols[2].metric("Average Cost", f"${calculate_average_cost(cohort):,.2f}")
    cols[3].metric("Average Length of Stay", f"{calculate_average_length_of_stay(cohort):.1f} days")
    st.dataframe(_prettify_columns(cohort.head(25)), width="stretch")


def render_machine_learning_section(data: pd.DataFrame, ml_context: dict[str, object] | None, ml_error: str | None) -> tuple[pd.DataFrame, float]:
    st.subheader("Readmission Risk Modeling")
    if ml_context is None:
        st.info(ml_error or "Predictive modeling is not available for the current selection because the model could not be trained.")
        return pd.DataFrame(), 0.6
    scored_data = ml_context["scored_data"]
    threshold = st.slider("Risk Classification Threshold", min_value=0.10, max_value=0.90, value=0.60, step=0.05)
    risk_summary = get_risk_segment_summary(scored_data)
    evaluation = build_model_evaluation(scored_data, threshold)
    cols = st.columns(5)
    cols[0].metric("ROC AUC", f"{evaluation['roc_auc']:.3f}" if evaluation["roc_auc"] is not None else "N/A")
    cols[1].metric("Precision", f"{evaluation['precision']:.2f}" if evaluation["precision"] is not None else "N/A")
    cols[2].metric("Recall", f"{evaluation['recall']:.2f}" if evaluation["recall"] is not None else "N/A")
    cols[3].metric("Patients Above Threshold", f"{(scored_data['predicted_readmission_risk'] >= threshold).sum():,}")
    cols[4].metric("Critical-Risk Patients", f"{(scored_data['risk_segment'] == 'Critical Risk').sum():,}")
    chart_cols = st.columns(2)
    _plot_or_message(chart_cols[0], create_risk_distribution_chart(risk_summary), chart_key=_make_chart_key("predictive", "risk_distribution"))
    _plot_or_message(chart_cols[1], create_probability_histogram(scored_data), chart_key=_make_chart_key("predictive", "probability_histogram"))
    if not evaluation["confusion_matrix"].empty:
        st.dataframe(evaluation["confusion_matrix"], width="stretch")
    display = risk_summary.copy()
    display["percentage"] = display["percentage"].map(_format_percent)
    st.dataframe(_prettify_columns(display), width="stretch")
    high_risk_patients = get_high_risk_patients(scored_data, threshold=threshold)
    if not high_risk_patients.empty:
        table = high_risk_patients.copy()
        table["predicted_readmission_risk"] = table["predicted_readmission_risk"].map(_format_percent)
        st.dataframe(_prettify_columns(table), width="stretch")
    st.markdown("**Model Explainability**")
    cols = st.columns([1.25, 1])
    _plot_or_message(cols[0], create_explainability_chart(ml_context["explainability"]), chart_key=_make_chart_key("predictive", "explainability"))
    cols[1].dataframe(_prettify_columns(ml_context["explainability"].round(3)), width="stretch")
    return high_risk_patients, threshold


def render_risk_drilldown(scored_data: pd.DataFrame | None) -> None:
    st.subheader("Risk Segment Drill-Down")
    if scored_data is None or scored_data.empty:
        st.info("Risk segment drill-down is not available because model output is not available for the current selection.")
        return
    segment = st.selectbox("Select a Risk Segment", options=RISK_SEGMENTS)
    drilldown = build_risk_drilldown(scored_data, segment)
    if drilldown["segment_data"].empty:
        st.info("No patients are currently assigned to the selected risk segment under the active filters.")
        return
    cols = st.columns(3)
    cols[0].metric("Patient Count", f"{drilldown['patient_count']:,}")
    cols[1].metric("Average Cost", f"${drilldown['average_cost']:,.2f}")
    cols[2].metric("Average Length of Stay", f"{drilldown['average_length_of_stay']:.1f} days")
    chart_cols = st.columns(2)
    _plot_or_message(chart_cols[0], create_risk_drilldown_chart(drilldown["top_diagnoses"], drilldown["top_diagnoses"].columns[0] if not drilldown["top_diagnoses"].empty else "diagnosis", "Most Common Diagnoses in Selected Segment"), chart_key=_make_chart_key("risk_drilldown", "diagnoses", segment))
    _plot_or_message(chart_cols[1], create_risk_drilldown_chart(drilldown["top_departments"], drilldown["top_departments"].columns[0] if not drilldown["top_departments"].empty else "department", "Most Common Departments in Selected Segment"), chart_key=_make_chart_key("risk_drilldown", "departments", segment))


def render_cost_driver_analysis(data: pd.DataFrame, figures: dict[str, object] | None = None) -> None:
    st.subheader("Cost Driver Analysis")
    figures = figures or {
        "top_diagnosis_cost": create_top_diagnosis_cost_chart(data),
        "cost_per_day": create_cost_per_day_chart(data),
        "highest_cost_departments": create_highest_cost_departments_chart(data),
    }
    cols = st.columns(2)
    _plot_or_message(cols[0], figures.get("top_diagnosis_cost"), chart_key=_make_chart_key("cost_drivers", "top_diagnosis_cost"))
    _plot_or_message(cols[1], figures.get("cost_per_day"), chart_key=_make_chart_key("cost_drivers", "cost_per_day"))
    _plot_or_message(st, figures.get("highest_cost_departments"), chart_key=_make_chart_key("cost_drivers", "highest_cost_departments"))


def render_department_scorecard(data: pd.DataFrame, scored_data: pd.DataFrame | None, scorecard: pd.DataFrame | None = None) -> pd.DataFrame:
    st.subheader("Department Performance Scorecard")
    scorecard = scorecard if scorecard is not None else build_department_scorecard(data, scored_data)
    if scorecard.empty:
        st.info("A department scorecard is not available for the current selection.")
        return scorecard
    display = scorecard.copy()
    display["readmission_rate"] = display["readmission_rate"].map(_format_percent)
    display["average_predicted_readmission_risk"] = display["average_predicted_readmission_risk"].map(_format_percent)
    display["average_length_of_stay"] = display["average_length_of_stay"].map(lambda value: f"{_format_number(value)} days" if pd.notna(value) else "N/A")
    display["average_cost"] = display["average_cost"].map(_format_currency)
    display["performance_flag"] = display["performance_flag"].astype(str)
    st.dataframe(_prettify_columns(display), width="stretch")
    cols = st.columns(2)
    _plot_or_message(cols[0], create_department_metric_chart(scorecard, "readmission_rate", "Readmission Rate by Department", "Readmission Rate"), chart_key=_make_chart_key("department_scorecard", "readmission_rate"))
    _plot_or_message(cols[1], create_department_metric_chart(scorecard, "average_length_of_stay", "Average Length of Stay by Department", "Average Length of Stay"), chart_key=_make_chart_key("department_scorecard", "average_length_of_stay"))
    return scorecard


def render_scenario_comparison(data: pd.DataFrame) -> pd.DataFrame:
    st.subheader("Operational Scenario Comparison")
    cols = st.columns(2)
    scenario_a_los = cols[0].slider("Scenario A LOS Reduction", 0.0, 3.0, 1.0, 0.1)
    scenario_a_readm = cols[0].slider("Scenario A Readmission Reduction", 0.0, 0.10, 0.03, 0.01, format="%.2f")
    scenario_b_los = cols[1].slider("Scenario B LOS Reduction", 0.0, 3.0, 2.0, 0.1)
    scenario_b_readm = cols[1].slider("Scenario B Readmission Reduction", 0.0, 0.10, 0.06, 0.01, format="%.2f")
    scenario_table = compare_scenarios(data, {"Current State": {"los_reduction": 0.0, "readmission_reduction": 0.0}, "Scenario A": {"los_reduction": scenario_a_los, "readmission_reduction": scenario_a_readm}, "Scenario B": {"los_reduction": scenario_b_los, "readmission_reduction": scenario_b_readm}})
    st.dataframe(_prettify_columns(scenario_table), width="stretch")
    return scenario_table


def render_key_insights(data: pd.DataFrame, scored_data: pd.DataFrame | None, insights: list[str] | None = None) -> list[str]:
    st.subheader("Key Business Insights")
    if insights is None:
        try:
            insights = generate_key_insights(data, scored_data)
        except (KeyError, ValueError) as error:
            st.info(f"Automated business insights are unavailable for the current selection: {error}")
            st.text_area("Analyst Notes", placeholder="Add your business interpretation here.")
            return []
    for insight in insights:
        st.markdown(f"- {insight}")
    st.text_area("Analyst Notes", placeholder="Add your business interpretation here.")
    return insights


def render_healthcare_data_quality(data: pd.DataFrame, quality: dict[str, object] | None = None) -> None:
    st.subheader("Healthcare Data Quality Review")
    quality = quality or assess_data_quality(data)
    cols = st.columns(4)
    cols[0].metric("Data Quality Score", f"{quality['quality_score']}/100")
    cols[1].metric("Quality Assessment", quality["interpretation"])
    cols[2].metric("Duplicate Admissions", f"{quality['duplicate_admission_count']:,}")
    cols[3].metric("Invalid LOS / Cost Rows", f"{quality['invalid_los_count']:,} / {quality['invalid_cost_count']:,}")
    st.dataframe(_prettify_columns(quality["missing_values"]), width="stretch")


def render_exports(filtered_data: pd.DataFrame, metrics: dict[str, float | int], scorecard: pd.DataFrame, high_risk_patients: pd.DataFrame, insights: list[str], selected_filters: dict[str, object], alerts: list[dict[str, str]], scenario_table: pd.DataFrame, generic_summary_text: bytes | None) -> None:
    st.subheader("Export Center")
    st.download_button("Download Filtered Dataset CSV", data=dataframe_to_csv_bytes(filtered_data), file_name="clinical_outcomes_filtered_dataset.csv", mime="text/csv", width="stretch")
    summary_text = build_summary_text(metrics, insights, selected_filters)
    st.download_button("Download Dashboard Summary TXT", data=summary_text if insights else (generic_summary_text or summary_text), file_name="clinical_outcomes_summary.txt", mime="text/plain", width="stretch")
    col1, col2 = st.columns(2)
    col1.download_button("Download Department Scorecard CSV", data=dataframe_to_csv_bytes(scorecard if not scorecard.empty else pd.DataFrame()), file_name="clinical_outcomes_department_scorecard.csv", mime="text/csv", width="stretch")
    col2.download_button("Download High-Risk Patient Table CSV", data=dataframe_to_csv_bytes(high_risk_patients if not high_risk_patients.empty else pd.DataFrame()), file_name="clinical_outcomes_high_risk_patients.csv", mime="text/csv", width="stretch")
    stakeholder = build_stakeholder_summary(metrics, insights, alerts, scenario_table)
    st.download_button("Download Stakeholder Summary TXT", data=stakeholder, file_name="clinical_outcomes_stakeholder_summary.txt", mime="text/plain", width="stretch")


def render_insights_assistant(filtered_data: pd.DataFrame, scored_data: pd.DataFrame | None) -> None:
    st.subheader("Insights Assistant")
    st.caption("Ask a supported business question and the app will answer it using the currently filtered data and available analytics outputs.")
    st.caption(f"Supported examples: {'; '.join(SUPPORTED_PATTERNS[:6])}")
    question = st.text_input("Ask a business question", placeholder="Which department has the highest cost?")
    if not question:
        return
    result = answer_business_question(question, filtered_data, scored_data)
    st.markdown(f"**Answer:** {result['answer']}")
    if result.get("table") is not None and not result["table"].empty:
        st.dataframe(_prettify_columns(result["table"]), width="stretch")


def main() -> None:
    apply_theme()
    st.title("Clinical Outcomes Explorer")
    st.caption("An analytics workspace for flexible CSV profiling, operational monitoring, and healthcare performance analysis when schema coverage supports it.")

    uploaded_files = st.sidebar.file_uploader("Upload CSV Files", type=["csv"], accept_multiple_files=True)
    raw_datasets, default_name = _load_raw_datasets(uploaded_files)
    dataset_names = list(raw_datasets.keys())
    uploaded = uploaded_files is not None and len(uploaded_files) > 0
    primary_name = st.sidebar.selectbox("Primary Dataset", options=dataset_names, index=dataset_names.index(default_name)) if uploaded else default_name
    raw_primary = raw_datasets[primary_name]
    st.session_state["active_dataframe"] = raw_primary

    dataset_token = f"{primary_name}:{len(raw_primary)}:{len(raw_primary.columns)}"
    if st.session_state.get("analysis_dataset_token") != dataset_token:
        st.session_state["analysis_dataset_token"] = dataset_token
        st.session_state["dataset_loaded"] = raw_primary is not None
        st.session_state["quick_analysis_ready"] = False
        st.session_state["full_analysis_ready"] = not uploaded
        st.session_state["analysis_section"] = "Auto-Analysis" if uploaded else "Executive Overview"
        if uploaded:
            st.session_state.pop("full_analysis_results", None)

    st.sidebar.caption(f"Active source: {primary_name}")
    st.subheader("Dataset Preview")
    st.caption("Review the dataset preview below, then choose a lightweight quick pass or the full analytics workflow.")
    preview_cols = st.columns(3)
    preview_cols[0].metric("Source", _short_label(primary_name, 18))
    preview_cols[1].metric("Rows", f"{len(raw_primary):,}")
    preview_cols[2].metric("Columns", f"{len(raw_primary.columns):,}")
    st.caption(f"Full file name: {primary_name}")
    st.dataframe(raw_primary.head(20), width="stretch")
    st.success("Dataset loaded successfully.")

    if uploaded:
        action_cols = st.columns(2)
        quick_requested = action_cols[0].button("Quick Analysis", type="primary", width="stretch")
        full_requested = action_cols[1].button("Full Analysis", width="stretch")
        if quick_requested:
            st.session_state["quick_analysis_ready"] = True
            st.session_state["full_analysis_ready"] = False
            st.session_state["analysis_section"] = "Auto-Analysis"
        if full_requested:
            st.session_state["quick_analysis_ready"] = True
            st.session_state["full_analysis_ready"] = True
            st.session_state["analysis_section"] = "Auto-Analysis"

    quick_ready = st.session_state.get("quick_analysis_ready", False)
    full_ready = st.session_state.get("full_analysis_ready", False)
    analysis_mode = "full" if full_ready else "quick" if quick_ready else "preview"

    if uploaded and analysis_mode == "preview":
        st.info("Dataset loaded. Click Quick Analysis for a fast profile or Full Analysis for the complete dashboard.")
        return

    quick_progress_start = time.perf_counter()
    progress_bar = None
    progress_status = None
    progress_metrics = None
    if uploaded and analysis_mode == "quick":
        progress_status = st.empty()
        progress_bar = st.progress(0)
        progress_metrics = st.empty()
        _update_progress(progress_bar, progress_status, progress_metrics, quick_progress_start, 5, "Loading dataset context")

    if progress_bar is not None:
        _update_progress(progress_bar, progress_status, progress_metrics, quick_progress_start, 20, "Detecting schema")
    with st.spinner("Preparing schema and analysis context..."):
        schema_start = time.perf_counter()
        schema_info = _cached_detect_schema(raw_primary)
    mapping_status = {field: "Not Mapped" for field in EXPECTED_FIELDS}
    if uploaded:
        effective_mapping, mapping_status = render_mapping_overrides(schema_info["normalized_data"], schema_info)
    else:
        effective_mapping = dict(schema_info["matched_schema"])
        for field in mapping_status:
            if field in effective_mapping:
                mapping_status[field] = "Mapped Automatically"

    schema_info["matched_schema"] = effective_mapping
    prepared_data = apply_detected_schema(raw_primary, effective_mapping)
    healthcare_enabled = _healthcare_mode_enabled(schema_info, uploaded)

    filtered_data, selected_filters, filter_meta = apply_dynamic_filters(prepared_data) if uploaded else (prepared_data, {"Dataset": primary_name}, {"text_search": "none", "text_filter_active": False, "date_filter_active": False, "numeric_filter_count": 0, "categorical_filter_count": 0, "rows_total": len(prepared_data), "rows_after_text": len(prepared_data), "rows_after_date": len(prepared_data), "rows_after_numeric": len(prepared_data), "rows_after_categorical": len(prepared_data), "rows_in_scope": len(prepared_data), "active_filter_count": 0, "fallback_used": False, "skipped_filters": [], "invalid_filter_warnings": []})
    if uploaded:
        for warning_message in filter_meta.get("invalid_filter_warnings", []):
            st.warning(warning_message)
    if filter_meta.get("fallback_used"):
        st.warning("Default filters produced no rows, so the app reverted to the full dataset.")
    if filtered_data.empty:
        st.warning("No records match the current sidebar filters. Adjust the filters to continue exploring the dataset.")
        return

    available_sections = _resolve_available_sections(uploaded, full_ready, healthcare_enabled)
    default_section = "Auto-Analysis" if "Auto-Analysis" in available_sections else available_sections[0]
    requested_section = st.session_state.get("analysis_section")
    current_section, fallback_used, fallback_target = resolve_current_section(available_sections, requested_section, default_section)
    if fallback_used or st.session_state.get("analysis_section") != current_section:
        st.session_state["analysis_section"] = current_section
    current_section = st.radio("Analysis Section", options=available_sections, horizontal=False, key="analysis_section")


    generic_profile = None
    generic_summary_text = None
    explainer = None
    dashboard_profile = None
    dashboard_specs = None
    quick_profile = None
    if current_section in {"Auto-Analysis", "Export Center"} and analysis_mode == "quick":
        if progress_bar is not None:
            _update_progress(progress_bar, progress_status, progress_metrics, quick_progress_start, 45, "Profiling dataset")
        with st.spinner("Preparing lightweight dataset profile..."):
            profile_start = time.perf_counter()
            quick_profile = _cached_quick_profile(filtered_data)
            generic_summary_text = _build_generic_export_text(quick_profile, schema_info, primary_name)
        if progress_bar is not None:
            _update_progress(progress_bar, progress_status, progress_metrics, quick_progress_start, 70, "Building lightweight summaries")
    elif current_section in {"Auto-Analysis", "Auto Generated Dashboard", "Data Quality Insights", "Export Center"}:
        with st.spinner("Preparing dataset profile..."):
            profile_start = time.perf_counter()
            generic_profile = _cached_profile(filtered_data)
            generic_summary_text = _build_generic_export_text(generic_profile, schema_info, primary_name)

    if current_section == "AI Dataset Explainer":
        with st.spinner("Generating dataset explanation..."):
            explainer_start = time.perf_counter()
            explainer = _cached_explainer(filtered_data, tuple(sorted(schema_info["matched_schema"].items())))

    if current_section == "Auto Generated Dashboard":
        dashboard_profile = generic_profile or _cached_profile(filtered_data)
        dashboard_start = time.perf_counter()
        dashboard_specs = _cached_dashboard_spec(filtered_data)

    if progress_bar is not None:
        _update_progress(progress_bar, progress_status, progress_metrics, quick_progress_start, 90, "Rendering auto-analysis")

    if uploaded and analysis_mode == "quick":
        if progress_bar is not None:
            status_total = time.perf_counter() - quick_progress_start
            _update_progress(progress_bar, progress_status, progress_metrics, quick_progress_start, 100, "Complete")
            progress_status.success("Quick Analysis complete")
            progress_metrics.caption(f"Completed in {status_total:.1f}s")

    full_signature = (
        dataset_token,
        tuple(sorted(schema_info["matched_schema"].items())),
        tuple(sorted((str(key), str(value)) for key, value in selected_filters.items())),
        len(filtered_data),
        len(prepared_data),
    )
    full_results = st.session_state.get("full_analysis_results")
    if full_ready and (not full_results or full_results.get("signature") != full_signature):
        full_start = time.perf_counter()
        full_progress_bar = None
        full_progress_status = None
        full_progress_metrics = None
        if uploaded:
            full_progress_status = st.empty()
            full_progress_bar = st.progress(0)
            full_progress_metrics = st.empty()
            _update_progress(full_progress_bar, full_progress_status, full_progress_metrics, full_start, 5, "Loading dataset context")
            _update_progress(full_progress_bar, full_progress_status, full_progress_metrics, full_start, 15, "Detecting schema")
        full_results = {
            "signature": full_signature,
            "selected_filters": selected_filters,
            "filter_meta": filter_meta,
            "schema_info": schema_info,
        }
        if uploaded and full_progress_bar is not None:
            _update_progress(full_progress_bar, full_progress_status, full_progress_metrics, full_start, 25, "Profiling dataset")
        full_results["generic_profile"] = generic_profile or _cached_profile(filtered_data)
        full_results["generic_summary_text"] = generic_summary_text or _build_generic_export_text(full_results["generic_profile"], schema_info, primary_name)
        full_results["quality_insights"] = _cached_quality_insights(filtered_data)
        if uploaded and full_progress_bar is not None:
            _update_progress(full_progress_bar, full_progress_status, full_progress_metrics, full_start, 35, "Building explainer")
        full_results["dataset_explainer"] = _cached_explainer(filtered_data, tuple(sorted(schema_info["matched_schema"].items())))
        if uploaded and full_progress_bar is not None:
            _update_progress(full_progress_bar, full_progress_status, full_progress_metrics, full_start, 45, "Building auto dashboard")
        full_results["dashboard_specs"] = _cached_dashboard_spec(filtered_data)
        full_results["dataset_comparison_profiles"] = {name: _cached_quick_profile(dataset) for name, dataset in raw_datasets.items()}
        if healthcare_enabled:
            if uploaded and full_progress_bar is not None:
                _update_progress(full_progress_bar, full_progress_status, full_progress_metrics, full_start, 55, "Preparing healthcare summaries")
            metrics = get_key_metrics(filtered_data)
            cohort_summaries = {cohort_key: build_cohort_summary(add_derived_buckets(filtered_data), cohort_key) for cohort_key in COHORT_FIELDS}
            trend_data = build_monthly_trends(filtered_data)
            if uploaded and full_progress_bar is not None:
                _update_progress(full_progress_bar, full_progress_status, full_progress_metrics, full_start, 70, "Training predictive model")
            ml_error = None
            ml_context_filtered = None
            ml_context_overall = None
            scored_filtered = None
            scored_overall = None
            if _has_columns(filtered_data, ["age", "comorbidity_score", "prior_admissions_12m", "length_of_stay", "readmission"]):
                try:
                    ml_context_filtered = train_readmission_model(filtered_data)
                except (ImportError, KeyError, ValueError) as error:
                    ml_error = str(error)
            else:
                ml_error = "Readmission modeling requires age, comorbidity score, prior admissions, length of stay, and readmission fields."
            if _has_columns(prepared_data, ["age", "comorbidity_score", "prior_admissions_12m", "length_of_stay", "readmission"]):
                try:
                    ml_context_overall = train_readmission_model(prepared_data)
                except (ImportError, KeyError, ValueError):
                    ml_context_overall = None
            scored_filtered = ml_context_filtered["scored_data"] if ml_context_filtered else None
            scored_overall = ml_context_overall["scored_data"] if ml_context_overall else None
            if uploaded and full_progress_bar is not None:
                _update_progress(full_progress_bar, full_progress_status, full_progress_metrics, full_start, 80, "Building scorecards and alerts")
            healthcare_quality = assess_data_quality(filtered_data)
            scorecard = build_department_scorecard(filtered_data, scored_filtered) if _has_columns(filtered_data, ["department", "readmission", "cost", "length_of_stay"]) else pd.DataFrame()
            risk_summary = get_risk_segment_summary(scored_filtered) if scored_filtered is not None and not scored_filtered.empty else pd.DataFrame()
            high_risk_patients = get_high_risk_patients(scored_filtered, threshold=0.6) if scored_filtered is not None and not scored_filtered.empty else pd.DataFrame()
            benchmarking_summary = build_benchmarking_summary(filtered_data, prepared_data, scored_filtered, scored_overall)
            alerts = generate_operational_alerts(filtered_data, prepared_data, healthcare_quality, scored_filtered, scored_overall)
            scenario_table = compare_scenarios(filtered_data, {
                "Current State": {"los_reduction": 0.0, "readmission_reduction": 0.0},
                "Scenario A": {"los_reduction": 1.0, "readmission_reduction": 0.03},
                "Scenario B": {"los_reduction": 2.0, "readmission_reduction": 0.06},
            })
            try:
                insights = generate_key_insights(filtered_data, scored_filtered) if _has_columns(filtered_data, ["diagnosis", "department", "readmission", "cost", "length_of_stay"]) else []
            except (KeyError, ValueError):
                insights = []
            if uploaded and full_progress_bar is not None:
                _update_progress(full_progress_bar, full_progress_status, full_progress_metrics, full_start, 90, "Preparing exports")
            full_results.update({
                "metrics": metrics,
                "ml_context_filtered": ml_context_filtered,
                "ml_context_overall": ml_context_overall,
                "ml_error": ml_error,
                "scored_filtered": scored_filtered,
                "scored_overall": scored_overall,
                "executive_summary": build_executive_summary(filtered_data, scored_filtered),
                "cohort_summaries": cohort_summaries,
                "trend_data": trend_data,
                "benchmarking_summary": benchmarking_summary,
                "alerts": alerts,
                "scenario_table": scenario_table,
                "insights": insights,
                "healthcare_quality": healthcare_quality,
                "scorecard": scorecard,
                "risk_summary": risk_summary,
                "high_risk_patients": high_risk_patients,
                "operational_overview_figures": {
                    "cost_by_department": create_cost_by_department_chart(filtered_data),
                    "readmission_by_diagnosis": create_readmission_by_diagnosis_chart(filtered_data),
                },
                "trend_figures": {
                    "admissions": create_monthly_trend_chart(trend_data, "admissions", "Monthly Admissions Trend", "Admissions"),
                    "average_cost": create_monthly_trend_chart(trend_data, "average_cost", "Monthly Average Cost Trend", "Average Cost"),
                    "readmission_rate": create_monthly_trend_chart(trend_data, "readmission_rate", "Monthly Readmission Rate Trend", "Readmission Rate"),
                    "average_length_of_stay": create_monthly_trend_chart(trend_data, "average_length_of_stay", "Monthly Average Length of Stay Trend", "Average Length of Stay"),
                },
                "benchmarking_figure": create_comparison_chart(benchmarking_summary),
                "cost_driver_figures": {
                    "top_diagnosis_cost": create_top_diagnosis_cost_chart(filtered_data),
                    "cost_per_day": create_cost_per_day_chart(filtered_data),
                    "highest_cost_departments": create_highest_cost_departments_chart(filtered_data),
                },
                "risk_distribution_figure": create_risk_distribution_chart(risk_summary) if not risk_summary.empty else None,
                "probability_histogram_figure": create_probability_histogram(scored_filtered) if scored_filtered is not None and not scored_filtered.empty else None,
                "explainability_figure": create_explainability_chart(ml_context_filtered["explainability"]) if ml_context_filtered else None,
                "export_summary_text": build_summary_text(metrics, insights, selected_filters),
                "stakeholder_summary": build_stakeholder_summary(metrics, insights, alerts, scenario_table),
            })
        if uploaded and full_progress_bar is not None:
            _update_progress(full_progress_bar, full_progress_status, full_progress_metrics, full_start, 100, "Full Analysis complete")
            full_progress_status.success("Full Analysis complete")
            full_progress_metrics.caption(f"Completed in {time.perf_counter() - full_start:.1f}s | All sections are now preloaded")
        st.session_state["full_analysis_results"] = full_results
    elif full_ready:
        full_results = st.session_state.get("full_analysis_results")
        if uploaded and full_results and full_results.get("signature") == full_signature:
            st.info("Using cached Full Analysis results for this dataset.")

    if full_results and full_results.get("signature") == full_signature:
        generic_profile = full_results.get("generic_profile", generic_profile)
        generic_summary_text = full_results.get("generic_summary_text", generic_summary_text)
        explainer = full_results.get("dataset_explainer", explainer)
        dashboard_profile = generic_profile
        dashboard_specs = full_results.get("dashboard_specs", dashboard_specs)

    if current_section == "Auto-Analysis":
        render_auto_analysis(filtered_data, schema_info, primary_name, healthcare_enabled, uploaded, mapping_status, filter_meta, selected_filters, quick_profile or generic_profile or _cached_quick_profile(filtered_data))
        return

    if current_section == "AI Dataset Explainer":
        render_dataset_explainer(explainer or _cached_explainer(filtered_data, tuple(sorted(schema_info["matched_schema"].items()))))
        return

    if current_section == "Auto Generated Dashboard":
        render_auto_generated_dashboard(filtered_data, dashboard_profile or generic_profile or _cached_profile(filtered_data), dashboard_specs or _cached_dashboard_spec(filtered_data))
        return
    if current_section == "Data Quality Insights":
        render_data_quality_insights(filtered_data, generic_profile or _cached_profile(filtered_data), full_results.get("quality_insights") if full_results else None)
        return

    if current_section == "Dataset Comparison":
        render_dataset_comparison(raw_datasets, full_results.get("dataset_comparison_profiles") if full_results else None)
        return

    metrics = {"total_admissions": len(filtered_data), "readmission_rate": 0.0, "readmission_count": 0, "average_length_of_stay": 0.0, "average_cost": 0.0, "total_cost": 0.0, "average_cost_per_day": 0.0}
    ml_context_filtered = None
    ml_context_overall = None
    ml_error: str | None = None
    scored_filtered = None
    scored_overall = None
    scorecard = pd.DataFrame()
    high_risk_patients = pd.DataFrame()
    alerts: list[dict[str, str]] = []
    insights: list[str] = []
    scenario_table = pd.DataFrame()

    if full_results and full_results.get("signature") == full_signature and healthcare_enabled:
        metrics = full_results.get("metrics", metrics)
        ml_context_filtered = full_results.get("ml_context_filtered")
        ml_context_overall = full_results.get("ml_context_overall")
        ml_error = full_results.get("ml_error")
        scored_filtered = full_results.get("scored_filtered")
        scored_overall = full_results.get("scored_overall")
        scorecard = full_results.get("scorecard", scorecard)
        high_risk_patients = full_results.get("high_risk_patients", high_risk_patients)
        alerts = full_results.get("alerts", alerts)
        insights = full_results.get("insights", insights)
        scenario_table = full_results.get("scenario_table", scenario_table)
    elif healthcare_enabled:
        metrics = get_key_metrics(filtered_data)
        needs_model = current_section in {"Executive Overview", "Predictive Insights", "Cost & Performance", "Insights Assistant", "Export Center"}
        if needs_model and _has_columns(filtered_data, ["age", "comorbidity_score", "prior_admissions_12m", "length_of_stay", "readmission"]):
            with st.spinner("Training readmission model..."):
                try:
                    ml_context_filtered = train_readmission_model(filtered_data)
                except (ImportError, KeyError, ValueError) as error:
                    ml_error = str(error)
        elif needs_model:
            ml_error = "Readmission modeling requires age, comorbidity score, prior admissions, length of stay, and readmission fields."
        if needs_model and _has_columns(prepared_data, ["age", "comorbidity_score", "prior_admissions_12m", "length_of_stay", "readmission"]):
            try:
                ml_context_overall = train_readmission_model(prepared_data)
            except (ImportError, KeyError, ValueError):
                ml_context_overall = None
        scored_filtered = ml_context_filtered["scored_data"] if ml_context_filtered else None
        scored_overall = ml_context_overall["scored_data"] if ml_context_overall else None
        if current_section == "Cost & Performance" and _has_columns(filtered_data, ["department", "readmission", "cost", "length_of_stay"]):
            scorecard = build_department_scorecard(filtered_data, scored_filtered)
        if current_section in {"Predictive Insights", "Export Center"} and scored_filtered is not None and not scored_filtered.empty:
            high_risk_patients = get_high_risk_patients(scored_filtered, threshold=0.6)

    if current_section == "Export Center":
        if healthcare_enabled:
            if scorecard.empty and _has_columns(filtered_data, ["department", "readmission", "cost", "length_of_stay"]):
                scorecard = build_department_scorecard(filtered_data, scored_filtered)
            if high_risk_patients.empty and scored_filtered is not None and not scored_filtered.empty:
                high_risk_patients = get_high_risk_patients(scored_filtered, threshold=0.6)
            if not insights and _has_columns(filtered_data, ["diagnosis", "department", "readmission", "cost", "length_of_stay"]):
                try:
                    insights = generate_key_insights(filtered_data, scored_filtered)
                except (KeyError, ValueError):
                    insights = []
            if not alerts:
                alerts = generate_operational_alerts(filtered_data, prepared_data, healthcare_quality, scored_filtered, scored_overall)
            if scenario_table.empty:
                scenario_table = compare_scenarios(
                    filtered_data,
                    {
                        "Current State": {"los_reduction": 0.0, "readmission_reduction": 0.0},
                        "Scenario A": {"los_reduction": 1.0, "readmission_reduction": 0.03},
                        "Scenario B": {"los_reduction": 2.0, "readmission_reduction": 0.06},
                    },
                )
        render_exports(filtered_data, metrics, scorecard, high_risk_patients, insights, selected_filters, alerts, scenario_table, generic_summary_text)
        return

    if current_section == "Executive Overview":
        render_active_filters_summary(primary_name, selected_filters, filter_meta)
        render_executive_summary(filtered_data, scored_filtered, full_results.get("executive_summary") if full_results else None)
        render_core_kpis(filtered_data, full_results.get("metrics") if full_results else None)
        render_operational_overview(filtered_data, full_results.get("operational_overview_figures") if full_results else None)
        render_trend_analysis(filtered_data, full_results.get("trend_data") if full_results else None, full_results.get("trend_figures") if full_results else None)
        alerts = render_operational_alerts(filtered_data, prepared_data, assess_data_quality(filtered_data), scored_filtered, scored_overall, full_results.get("alerts") if full_results else None)
        render_benchmarking(filtered_data, prepared_data, scored_filtered, scored_overall, full_results.get("benchmarking_summary") if full_results else None, full_results.get("benchmarking_figure") if full_results else None)
        render_benchmark_baselines(filtered_data)
        scenario_table = render_scenario_comparison(filtered_data)
        insights = render_key_insights(filtered_data, scored_filtered, full_results.get("insights") if full_results else None)
        return

    if current_section == "Cohort Analysis":
        render_patient_cohort_explorer(filtered_data, full_results.get("cohort_summaries") if full_results else None)
        render_custom_cohort_builder(filtered_data)
        return

    if current_section == "Predictive Insights":
        render_machine_learning_section(filtered_data, ml_context_filtered, ml_error)
        render_risk_drilldown(scored_filtered)
        return

    if current_section == "Cost & Performance":
        render_cost_driver_analysis(filtered_data, full_results.get("cost_driver_figures") if full_results else None)
        render_department_scorecard(filtered_data, scored_filtered, full_results.get("scorecard") if full_results else None)
        return

    if current_section == "Data Quality Review":
        render_healthcare_data_quality(filtered_data, full_results.get("healthcare_quality") if full_results else None)
        render_data_quality_insights(filtered_data, generic_profile or _cached_profile(filtered_data), full_results.get("quality_insights") if full_results else None)
        return

    if current_section == "Insights Assistant":
        render_insights_assistant(filtered_data, scored_filtered)
        return

if __name__ == "__main__":
    main()

































