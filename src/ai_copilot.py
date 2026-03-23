from __future__ import annotations

import json
import os
import re
from typing import Any

import src.logger as logger_module
from src.healthcare_analysis import benchmarking_analysis, build_cohort_summary, readmission_risk_analytics

import pandas as pd
import plotly.express as px
import streamlit as st
from src.ui_components import style_figure

try:
    from src.analytics import build_department_scorecard, get_top_diagnosis_groups_by_total_cost
    from src.charts import create_department_metric_chart, create_top_diagnosis_cost_chart
    from src.metrics import (
        calculate_average_cost,
        calculate_average_length_of_stay,
        calculate_readmission_rate,
        get_key_metrics,
    )
    LEGACY_COPILOT_IMPORT_ERROR = None
except Exception as legacy_import_error:
    build_department_scorecard = None
    get_top_diagnosis_groups_by_total_cost = None
    create_department_metric_chart = None
    create_top_diagnosis_cost_chart = None
    calculate_average_cost = None
    calculate_average_length_of_stay = None
    calculate_readmission_rate = None
    get_key_metrics = None
    LEGACY_COPILOT_IMPORT_ERROR = legacy_import_error

COPILOT_MEMORY_KEY = 'copilot_messages'
COPILOT_CACHE_KEY = 'copilot_response_cache'
OPENAI_MODEL = 'gpt-4.1-mini'
COPILOT_PROVIDER_ENV = 'SMART_DATASET_ANALYZER_LLM_PROVIDER'
COPILOT_MODEL_ENV = 'SMART_DATASET_ANALYZER_COPILOT_MODEL'
ANTHROPIC_MODEL_ENV = 'SMART_DATASET_ANALYZER_ANTHROPIC_MODEL'
OLLAMA_MODEL_ENV = 'SMART_DATASET_ANALYZER_OLLAMA_MODEL'
OLLAMA_BASE_URL_ENV = 'SMART_DATASET_ANALYZER_OLLAMA_BASE_URL'
DEFAULT_COPILOT_PROMPTS = [
    'Summarize the dataset',
    'Which patients are high risk?',
    'Show key healthcare insights',
]


def _readmission_unavailable_payload(readmission: dict[str, Any], fallback_reason: str) -> dict[str, Any]:
    readiness = readmission.get('readiness', {}) if isinstance(readmission, dict) else {}
    missing_fields = readiness.get('missing_fields', readmission.get('missing_fields', []))
    available = readiness.get('available_analysis', readmission.get('available_analysis', []))
    unlock = readiness.get('additional_fields_to_unlock_full_analysis', missing_fields)
    answer = readmission.get('reason', fallback_reason) if isinstance(readmission, dict) else fallback_reason
    if available:
        answer += ' What still works now: ' + ', '.join(str(item) for item in available) + '.'
    if unlock:
        answer += ' Add or map these fields next: ' + ', '.join(str(item).replace('_', ' ') for item in unlock) + '.'
    return _response_payload(
        'readmission_unavailable',
        answer,
        structured_result={'missing_fields': missing_fields, 'available_analysis': available, 'unlock_fields': unlock},
        table=None,
        chart=None,
        follow_ups=_follow_ups('Summarize the dataset', 'Explain readiness blockers'),
    )


def _numeric_mean(data: pd.DataFrame, column: str) -> float:
    values = pd.to_numeric(data[column], errors='coerce')
    return float(values.mean()) if values.notna().any() else 0.0


def _fallback_readmission_rate(data: pd.DataFrame) -> float:
    for column in ['readmission', 'readmission_flag', 'is_readmission']:
        if column in data.columns:
            values = pd.to_numeric(data[column], errors='coerce')
            if values.notna().any():
                return float(values.mean())
    return 0.0


def _fallback_key_metrics(data: pd.DataFrame) -> dict[str, float]:
    return {
        'readmission_rate': calculate_readmission_rate(data) if calculate_readmission_rate else _fallback_readmission_rate(data),
        'average_length_of_stay': calculate_average_length_of_stay(data) if calculate_average_length_of_stay else _numeric_mean(data, 'length_of_stay'),
        'average_cost': calculate_average_cost(data) if calculate_average_cost else _numeric_mean(data, 'cost'),
    }


def _response_payload(tool: str, answer: str, structured_result: dict[str, Any] | None = None, table: pd.DataFrame | None = None, chart: Any | None = None, follow_ups: list[str] | None = None) -> dict[str, Any]:
    if chart is not None and hasattr(chart, 'update_layout'):
        try:
            chart = style_figure(chart)
        except Exception:
            pass
    payload = {
        'tool': tool,
        'answer': answer,
        'summary_text': answer,
        'structured_result': structured_result or {},
        'table': table,
        'table_data': table,
        'chart': chart,
        'chart_figure': chart,
        'follow_ups': follow_ups or [],
        'follow_up_questions': follow_ups or [],
        'citations': [],
        'explanation': '',
        'llm_provider': None,
        'cached_response': False,
    }
    return payload


def initialize_copilot_memory() -> list[dict[str, Any]]:
    if COPILOT_MEMORY_KEY not in st.session_state:
        st.session_state[COPILOT_MEMORY_KEY] = []
    return st.session_state[COPILOT_MEMORY_KEY]


def append_copilot_message(role: str, content: str, payload: dict[str, Any] | None = None) -> None:
    messages = initialize_copilot_memory()
    message = {'role': role, 'content': content}
    if payload:
        message.update(payload)
    messages.append(message)
    st.session_state[COPILOT_MEMORY_KEY] = messages


def _dataset_cache_token(data: pd.DataFrame, schema_info: dict[str, Any]) -> str:
    dataset_name = str(schema_info.get('dataset_name', 'current-dataset'))
    cache_key = str(data.attrs.get('dataset_cache_key', '')).strip()
    if cache_key:
        return f'{dataset_name}::{cache_key}'
    return f"{dataset_name}::{len(data)}::{len(data.columns)}::{'|'.join(map(str, data.columns[:25]))}"


def _copilot_response_cache() -> dict[str, dict[str, Any]]:
    cache = st.session_state.get(COPILOT_CACHE_KEY)
    if not isinstance(cache, dict):
        cache = {}
        st.session_state[COPILOT_CACHE_KEY] = cache
    return cache


def _cache_lookup(dataset_token: str, question: str) -> dict[str, Any] | None:
    key = f'{dataset_token}::{question.strip().lower()}'
    cached = _copilot_response_cache().get(key)
    if not isinstance(cached, dict):
        return None
    cached_payload = dict(cached)
    cached_payload['cached_response'] = True
    return cached_payload


def _cache_store(dataset_token: str, question: str, payload: dict[str, Any]) -> dict[str, Any]:
    key = f'{dataset_token}::{question.strip().lower()}'
    cached = dict(payload)
    cached['cached_response'] = False
    _copilot_response_cache()[key] = cached
    return cached


def build_copilot_panel_config(data: pd.DataFrame, schema_info: dict[str, Any] | None = None) -> dict[str, Any]:
    schema_info = schema_info or {}
    matched_schema = schema_info.get('matched_schema', {}) if isinstance(schema_info, dict) else {}
    prompt_suggestions = list(DEFAULT_COPILOT_PROMPTS)
    readmission_supported = bool(
        matched_schema.get('readmission')
        or 'readmission' in data.columns
        or matched_schema.get('department')
        or 'department' in data.columns
    )
    if readmission_supported:
        prompt_suggestions.extend(
            [
                'Show readmission by department',
                'What drives readmission?',
                'Summarize readmission risk',
                'Generate readmission report',
                'Show highest-risk readmission patients',
                'What interventions could reduce readmissions?',
                'Explain readmission blockers',
            ]
        )
    if matched_schema.get('department') or 'department' in data.columns:
        prompt_suggestions.append('Show readmission by department')
    if matched_schema.get('diagnosis') or matched_schema.get('diagnosis_code') or 'diagnosis' in data.columns:
        prompt_suggestions.append('What is the top diagnosis by cost?')
    if matched_schema.get('length_of_stay') or 'length_of_stay' in data.columns:
        prompt_suggestions.append('What is the average length of stay?')
    if matched_schema.get('readmission') or 'readmission' in data.columns:
        prompt_suggestions.append('Show highest-risk readmission patients')
    if matched_schema.get('cost') or 'cost' in data.columns or 'estimated_cost' in data.columns:
        prompt_suggestions.append('What is the average cost?')

    deduped_prompts: list[str] = []
    for prompt in prompt_suggestions:
        if prompt not in deduped_prompts:
            deduped_prompts.append(prompt)

    backend = _get_llm_backend()
    ai_backend_available = backend is not None
    backend_label = str(backend.get('label', 'AI')).strip() if ai_backend_available else ''
    return {
        'title': 'AI Copilot Panel',
        'mode_label': f'AI-assisted ({backend_label})' if ai_backend_available else 'Rule-based fallback',
        'mode_note': (
            f"External AI enhancement is available through {backend.get('label', 'the configured provider')} for polished healthcare explanations."
            if ai_backend_available
            else 'No external AI backend is configured, so the copilot is using deterministic healthcare-aware summaries from the current dataset.'
        ),
        'suggested_prompts': deduped_prompts,
    }


def _follow_ups(*items: str) -> list[str]:
    return [item for item in items if item]


def get_dataset_summary(data: pd.DataFrame, schema_info: dict) -> dict[str, Any]:
    coverage = len(schema_info.get('matched_schema', {}))
    structured = {
        'row_count': len(data),
        'column_count': len(data.columns),
        'mapped_fields': coverage,
    }
    table = pd.DataFrame([
        {
            'row_count': len(data),
            'column_count': len(data.columns),
            'mapped_fields': coverage,
        }
    ])
    if {'readmission', 'length_of_stay', 'cost'}.issubset(data.columns):
        metrics = get_key_metrics(data) if get_key_metrics else _fallback_key_metrics(data)
        structured.update({
            'readmission_rate': metrics['readmission_rate'],
            'average_length_of_stay': metrics['average_length_of_stay'],
            'average_cost': metrics['average_cost'],
        })
        table = pd.DataFrame([
            {
                'row_count': len(data),
                'column_count': len(data.columns),
                'mapped_fields': coverage,
                'readmission_rate': metrics['readmission_rate'],
                'average_length_of_stay': metrics['average_length_of_stay'],
                'average_cost': metrics['average_cost'],
            }
        ])
        answer = (
            f"The current filtered dataset includes {len(data):,} rows, {len(data.columns):,} columns, and {coverage} mapped healthcare fields. "
            f"Readmission rate is {(calculate_readmission_rate(data) if calculate_readmission_rate else metrics['readmission_rate']):.1%}, "
            f"average length of stay is {metrics['average_length_of_stay']:.1f} days, "
            f"and average cost is ${metrics['average_cost']:,.2f}."
        )
    else:
        answer = f"The current filtered dataset includes {len(data):,} rows, {len(data.columns):,} columns, and {coverage} mapped healthcare fields."
    return _response_payload(
        'get_dataset_summary',
        answer,
        structured_result=structured,
        table=table,
        chart=None,
        follow_ups=_follow_ups('What is the average cost?', 'Show readmission by department'),
    )


def get_department_readmission(data: pd.DataFrame, schema_info: dict) -> dict[str, Any]:
    readmission = readmission_risk_analytics(data, schema_info.get('matched_schema', {}))
    if not readmission.get('available'):
        payload = _readmission_unavailable_payload(readmission, 'Department readmission analysis is not available because the current dataset is missing one or more supporting fields.')
        payload['tool'] = 'get_department_readmission'
        payload['follow_ups'] = _follow_ups('What is the average cost?', 'Summarize this dataset')
        payload['follow_up_questions'] = payload['follow_ups']
        return payload
    scorecard = readmission.get('by_department', pd.DataFrame())
    department_col = readmission.get('department_column') or (scorecard.columns[0] if isinstance(scorecard, pd.DataFrame) and not scorecard.empty else 'department')
    if not isinstance(scorecard, pd.DataFrame) or scorecard.empty:
        return _response_payload(
            'get_department_readmission',
            'No department-level readmission summary is available for the current selection.',
            structured_result={},
            table=None,
            chart=None,
            follow_ups=_follow_ups('Summarize this dataset'),
        )
    ranked = scorecard.sort_values('readmission_rate', ascending=False).reset_index(drop=True)
    top_row = ranked.iloc[0]
    department = top_row[department_col]
    structured = {
        'department': department,
        'readmission_rate': float(top_row['readmission_rate']),
    }
    answer = f"{department} has the highest readmission rate at {top_row['readmission_rate']:.1%} in the current filtered dataset."
    return _response_payload(
        'get_department_readmission',
        answer,
        structured_result=structured,
        table=ranked.head(10),
        chart=px.bar(ranked.head(10), x=department_col, y='readmission_rate', title='Readmission Rate by Department'),
        follow_ups=_follow_ups('What is the average cost?', 'What is the average length of stay?'),
    )


def get_highest_readmission_risk(data: pd.DataFrame, schema_info: dict) -> dict[str, Any]:
    readmission = readmission_risk_analytics(data, schema_info.get('matched_schema', {}))
    if not readmission.get('available'):
        payload = _readmission_unavailable_payload(readmission, 'High readmission risk identification is not available for the current dataset.')
        payload['tool'] = 'get_highest_readmission_risk'
        payload['follow_ups'] = _follow_ups('What factors drive readmission?', 'Generate a readmission summary report')
        payload['follow_up_questions'] = payload['follow_ups']
        return payload
    segments = readmission.get('high_risk_segments', pd.DataFrame())
    patients = readmission.get('high_risk_patients', pd.DataFrame())
    answer = 'The current dataset highlights the top readmission-risk segments and the highest-risk patient or encounter rows for follow-up review.'
    if isinstance(segments, pd.DataFrame) and not segments.empty:
        top = segments.iloc[0]
        answer = (
            f"The highest readmission-risk segment is {top['segment_type']} = {top['segment_value']}, "
            f"with a readmission rate of {float(top['readmission_rate']):.1%}. "
            'Use the row-level list to focus case management, discharge follow-up, or service-line review.'
        )
    return _response_payload(
        'get_highest_readmission_risk',
        answer,
        structured_result={'overall_readmission_rate': readmission.get('overview', {}).get('overall_readmission_rate')},
        table=patients.head(15) if isinstance(patients, pd.DataFrame) else None,
        chart=px.bar(segments.head(8), x='segment_value', y='readmission_rate', color='segment_type', title='Top Readmission-Risk Segments') if isinstance(segments, pd.DataFrame) and not segments.empty else None,
        follow_ups=_follow_ups('What factors drive readmission?', 'Show readmission by department'),
    )


def get_readmission_drivers(data: pd.DataFrame, schema_info: dict) -> dict[str, Any]:
    readmission = readmission_risk_analytics(data, schema_info.get('matched_schema', {}))
    if not readmission.get('available'):
        payload = _readmission_unavailable_payload(readmission, 'Readmission driver analysis is not available for the current dataset.')
        payload['tool'] = 'get_readmission_drivers'
        payload['follow_ups'] = _follow_ups('Which patients are highest readmission risk?', 'Generate a readmission summary report')
        payload['follow_up_questions'] = payload['follow_ups']
        return payload
    drivers = readmission.get('driver_table', pd.DataFrame())
    return _response_payload(
        'get_readmission_drivers',
        readmission.get('driver_interpretation', 'Readmission driver analysis is available for the current dataset.')
        + ' Use this driver ranking to decide which diagnosis, department, age band, or LOS cohort to target first.',
        structured_result={'overall_readmission_rate': readmission.get('overview', {}).get('overall_readmission_rate')},
        table=drivers,
        chart=px.bar(drivers, x='factor', y='gap_vs_overall', title='Readmission Driver Gaps') if isinstance(drivers, pd.DataFrame) and not drivers.empty else None,
        follow_ups=_follow_ups('Show readmission by department', 'Which patients are highest readmission risk?'),
    )


def get_readmission_summary(data: pd.DataFrame, schema_info: dict) -> dict[str, Any]:
    readmission = readmission_risk_analytics(data, schema_info.get('matched_schema', {}))
    if not readmission.get('available'):
        payload = _readmission_unavailable_payload(readmission, 'A readmission summary is not available for the current dataset.')
        payload['tool'] = 'get_readmission_summary'
        payload['follow_ups'] = _follow_ups('Show readmission by department', 'What factors drive readmission?')
        payload['follow_up_questions'] = payload['follow_ups']
        return payload
    overview = readmission.get('overview', {})
    segments = readmission.get('high_risk_segments', pd.DataFrame())
    table = pd.DataFrame([{
        'overall_readmission_rate': overview.get('overall_readmission_rate'),
        'readmission_count': overview.get('readmission_count'),
        'records_in_scope': overview.get('records_in_scope'),
    }])
    if isinstance(segments, pd.DataFrame) and not segments.empty:
        table = pd.concat([table, segments.head(3)], ignore_index=True, sort=False)
    return _response_payload(
        'get_readmission_summary',
        f"Overall readmission rate is {float(overview.get('overall_readmission_rate', 0.0)):.1%}, "
        f"with {int(overview.get('readmission_count', 0))} readmissions in scope. "
        f"{readmission.get('top_high_risk_summary', '')}",
        structured_result=overview,
        table=table,
        chart=px.line(readmission['trend'], x='month', y='readmission_rate', title='Readmission Trend') if isinstance(readmission.get('trend'), pd.DataFrame) and not readmission['trend'].empty else None,
        follow_ups=_follow_ups('Which patients are highest readmission risk?', 'What factors drive readmission?'),
    )


def get_average_cost(data: pd.DataFrame) -> dict[str, Any]:
    if 'cost' not in data.columns:
        return _response_payload(
            'get_average_cost',
            'Average cost is not available because the current dataset does not include a mapped cost field.',
            structured_result={'missing_fields': ['cost']},
            table=None,
            chart=None,
            follow_ups=_follow_ups('Summarize this dataset'),
        )
    average_cost = calculate_average_cost(data) if calculate_average_cost else _numeric_mean(data, 'cost')
    table = pd.DataFrame([{'average_cost': average_cost, 'records_in_scope': len(data)}])
    return _response_payload(
        'get_average_cost',
        f"The current filtered dataset has an average cost of ${average_cost:,.2f} per record.",
        structured_result={'average_cost': average_cost},
        table=table,
        chart=None,
        follow_ups=_follow_ups('What is the top diagnosis by cost?', 'Summarize this dataset'),
    )


def get_average_length_of_stay(data: pd.DataFrame) -> dict[str, Any]:
    if 'length_of_stay' not in data.columns:
        return _response_payload(
            'get_average_length_of_stay',
            'Average length of stay is not available because the current dataset does not include a mapped length-of-stay field.',
            structured_result={'missing_fields': ['length_of_stay']},
            table=None,
            chart=None,
            follow_ups=_follow_ups('Summarize this dataset'),
        )
    average_los = calculate_average_length_of_stay(data) if calculate_average_length_of_stay else _numeric_mean(data, 'length_of_stay')
    table = pd.DataFrame([{'average_length_of_stay': average_los, 'records_in_scope': len(data)}])
    return _response_payload(
        'get_average_length_of_stay',
        f"The current filtered dataset has an average length of stay of {average_los:.1f} days.",
        structured_result={'average_length_of_stay': average_los},
        table=table,
        chart=None,
        follow_ups=_follow_ups('Show readmission by department', 'Summarize this dataset'),
    )


def get_top_diagnosis_by_cost(data: pd.DataFrame) -> dict[str, Any]:
    required = {'diagnosis', 'cost'}
    if not required.issubset(data.columns):
        return _response_payload(
            'get_top_diagnosis_by_cost',
            'Top diagnosis by cost is not available because the current dataset is missing diagnosis or cost.',
            structured_result={'missing_fields': sorted(required - set(data.columns))},
            table=None,
            chart=None,
            follow_ups=_follow_ups('What is the average cost?', 'Summarize this dataset'),
        )
    if get_top_diagnosis_groups_by_total_cost is not None:
        summary = get_top_diagnosis_groups_by_total_cost(data)
    else:
        summary = (
            data.assign(_cost=pd.to_numeric(data['cost'], errors='coerce'))
            .dropna(subset=['_cost'])
            .groupby('diagnosis', dropna=False)
            .agg(total_cost=('_cost', 'sum'))
            .reset_index()
            .sort_values('total_cost', ascending=False)
            .reset_index(drop=True)
        )
    if summary.empty:
        return _response_payload(
            'get_top_diagnosis_by_cost',
            'No diagnosis-level cost summary is available for the current selection.',
            structured_result={},
            table=None,
            chart=None,
            follow_ups=_follow_ups('What is the average cost?'),
        )
    top_row = summary.iloc[0]
    diagnosis = top_row.iloc[0]
    total_cost = float(top_row['total_cost'])
    return _response_payload(
        'get_top_diagnosis_by_cost',
        f"{diagnosis} is the top diagnosis group by total cost at ${total_cost:,.2f}.",
        structured_result={'diagnosis': diagnosis, 'total_cost': total_cost},
        table=summary.head(10),
        chart=create_top_diagnosis_cost_chart(data) if create_top_diagnosis_cost_chart else None,
        follow_ups=_follow_ups('What is the average cost?', 'Show readmission by department'),
    )


def get_key_healthcare_insights(data: pd.DataFrame, schema_info: dict[str, Any]) -> dict[str, Any]:
    matched_schema = schema_info.get('matched_schema', {}) if isinstance(schema_info, dict) else {}
    rows: list[dict[str, str]] = []

    los_col = matched_schema.get('length_of_stay')
    if los_col and los_col in data.columns:
        los = pd.to_numeric(data[los_col], errors='coerce')
        if los.notna().any():
            rows.append({'insight': 'Average length of stay', 'value': f'{los.mean():.1f} days'})

    readmit_col = matched_schema.get('readmission')
    if readmit_col and readmit_col in data.columns:
        readmit = pd.to_numeric(data[readmit_col], errors='coerce')
        if readmit.notna().any():
            rows.append({'insight': 'Readmission rate', 'value': f'{readmit.mean():.1%}'})

    cost_col = matched_schema.get('cost')
    if not cost_col or cost_col not in data.columns:
        cost_col = 'estimated_cost' if 'estimated_cost' in data.columns else None
    if cost_col and cost_col in data.columns:
        cost = pd.to_numeric(data[cost_col], errors='coerce')
        if cost.notna().any():
            rows.append({'insight': 'Average cost', 'value': f"${cost.mean():,.2f}"})

    diagnosis_col = matched_schema.get('diagnosis') or matched_schema.get('diagnosis_code')
    if diagnosis_col and diagnosis_col in data.columns:
        top_counts = data[diagnosis_col].astype(str).replace('', pd.NA).dropna().value_counts()
        if not top_counts.empty:
            rows.append({'insight': 'Most common diagnosis', 'value': str(top_counts.index[0])})

    if not rows:
        rows.append({'insight': 'Dataset coverage', 'value': f'{len(data):,} rows ready for guided review'})

    insights = pd.DataFrame(rows)
    return _response_payload(
        'get_key_healthcare_insights',
        'Here are the most relevant healthcare insights available from the current dataset, pulled directly from the current fields and filters.',
        structured_result={'insight_count': len(insights)},
        table=insights,
        chart=None,
        follow_ups=_follow_ups('Summarize the dataset', 'Which patients are high risk?'),
    )


def detect_intent(question: str) -> str | None:
    question_text = (question or '').strip().lower()
    if not question_text:
        return None
    if 'key healthcare insights' in question_text or 'healthcare insights' in question_text:
        return 'healthcare_insights'
    if 'department' in question_text and 'readmission' in question_text:
        return 'department_readmission'
    if 'blocker' in question_text and 'readmission' in question_text:
        return 'readmission_summary'
    if ('highest' in question_text or 'high-risk' in question_text or 'high risk' in question_text) and ('readmission' in question_text or 'patient' in question_text or 'patients' in question_text):
        return 'readmission_high_risk'
    if 'drive readmission' in question_text or 'drivers of readmission' in question_text or 'factors drive readmission' in question_text:
        return 'readmission_drivers'
    if 'intervention' in question_text and 'readmission' in question_text:
        return 'readmission_report'
    if (
        'readmission summary' in question_text
        or 'summarize readmission risk' in question_text
        or 'readmission risk summary' in question_text
    ):
        return 'readmission_summary'
    if 'generate readmission report' in question_text or ('readmission' in question_text and 'report' in question_text):
        return 'readmission_report'
    if 'top diagnosis' in question_text and 'cost' in question_text:
        return 'top_diagnosis_by_cost'
    if 'average length of stay' in question_text or 'average los' in question_text or ('los' in question_text and 'average' in question_text):
        return 'average_length_of_stay'
    if 'average cost' in question_text or ('cost' in question_text and 'average' in question_text):
        return 'average_cost'
    if 'dataset summary' in question_text or 'summary' in question_text or 'summarize' in question_text:
        return 'dataset_summary'
    return None


def _configured_provider() -> str:
    return str(os.getenv(COPILOT_PROVIDER_ENV, 'auto')).strip().lower() or 'auto'


def _get_openai_backend() -> dict[str, Any] | None:
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        return None
    try:
        from openai import OpenAI
    except ImportError:
        return None
    return {
        'provider': 'openai',
        'label': 'OpenAI',
        'client': OpenAI(api_key=api_key),
        'model': str(os.getenv(COPILOT_MODEL_ENV, OPENAI_MODEL)).strip() or OPENAI_MODEL,
    }


def _get_anthropic_backend() -> dict[str, Any] | None:
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        return None
    try:
        from anthropic import Anthropic
    except ImportError:
        return None
    default_model = 'claude-3-5-sonnet-20241022'
    return {
        'provider': 'anthropic',
        'label': 'Claude',
        'client': Anthropic(api_key=api_key),
        'model': str(os.getenv(ANTHROPIC_MODEL_ENV, default_model)).strip() or default_model,
    }


def _get_ollama_backend() -> dict[str, Any] | None:
    model_name = str(os.getenv(OLLAMA_MODEL_ENV, '')).strip()
    if not model_name:
        return None
    base_url = str(os.getenv(OLLAMA_BASE_URL_ENV, 'http://localhost:11434')).strip() or 'http://localhost:11434'
    return {
        'provider': 'ollama',
        'label': 'Local LLM',
        'model': model_name,
        'base_url': base_url.rstrip('/'),
    }


def _get_llm_backend() -> dict[str, Any] | None:
    configured = _configured_provider()
    backends = {
        'openai': _get_openai_backend,
        'anthropic': _get_anthropic_backend,
        'claude': _get_anthropic_backend,
        'ollama': _get_ollama_backend,
        'local': _get_ollama_backend,
    }
    if configured != 'auto':
        loader = backends.get(configured)
        return loader() if loader is not None else None
    for loader in (_get_openai_backend, _get_anthropic_backend, _get_ollama_backend):
        backend = loader()
        if backend is not None:
            return backend
    return None


def _build_citations(tool_result: dict[str, Any], schema_info: dict[str, Any]) -> list[dict[str, str]]:
    structured_result = tool_result.get('structured_result', {}) or {}
    matched_schema = schema_info.get('matched_schema', {}) or {}
    citations = [
        {
            'title': 'Structured analytics result',
            'detail': f"Tool `{tool_result.get('tool', 'unknown')}` returned dataset-grounded values from the current filtered selection.",
        }
    ]
    if matched_schema:
        citations.append(
            {
                'title': 'Mapped source fields',
                'detail': 'Matched schema fields: ' + ', '.join(sorted(map(str, matched_schema.keys()))[:10]),
            }
        )
    if structured_result:
        top_fields = ', '.join(sorted(map(str, structured_result.keys()))[:8])
        citations.append(
            {
                'title': 'Structured metrics used',
                'detail': f'Grounding fields: {top_fields}',
            }
        )
    return citations


def _build_explanation(tool_result: dict[str, Any], schema_info: dict[str, Any]) -> str:
    tool_name = str(tool_result.get('tool', 'unknown'))
    dataset_name = str(schema_info.get('dataset_name', 'current dataset'))
    matched_schema = schema_info.get('matched_schema', {}) or {}
    return (
        f"This answer was generated from `{tool_name}` using the current filtered view of {dataset_name}. "
        f"It is grounded in {len(matched_schema):,} mapped field(s) and the structured analytics payload returned by the platform."
    )


def _healthcare_llm_prompt(question: str, tool_result: dict[str, Any], schema_info: dict[str, Any]) -> str:
    matched_schema = schema_info.get('matched_schema', {})
    return (
        'You are a healthcare analytics SaaS copilot. '
        'Write a concise, clinically aware answer that stays faithful to the provided structured analytics. '
        'Do not invent metrics or unsupported clinical conclusions. '
        'Use a professional healthcare operations tone. '
        'If data support is weak, say so clearly. '
        'End with one short actionable next step if appropriate.\n\n'
        f'User question: {question}\n'
        f"Dataset name: {schema_info.get('dataset_name', 'Current dataset')}\n"
        f"Matched schema fields: {', '.join(sorted(matched_schema.keys())) if matched_schema else 'none'}\n"
        f"Tool used: {tool_result.get('tool', 'unknown')}\n"
        f"Structured result JSON: {json.dumps(tool_result.get('structured_result', {}), default=str)}\n"
        f"Base deterministic answer: {tool_result.get('answer', '')}\n"
        f"Citations to preserve context: {json.dumps(_build_citations(tool_result, schema_info), default=str)}"
    )


def _complete_with_backend(backend: dict[str, Any], prompt: str) -> str | None:
    try:
        if backend.get('provider') == 'openai':
            response = backend['client'].responses.create(
                model=backend['model'],
                input=prompt,
            )
            output_text = getattr(response, 'output_text', None)
            return output_text.strip() if output_text and output_text.strip() else None
        if backend.get('provider') == 'anthropic':
            response = backend['client'].messages.create(
                model=backend['model'],
                max_tokens=350,
                messages=[{'role': 'user', 'content': prompt}],
            )
            parts = getattr(response, 'content', [])
            text = ''.join(getattr(part, 'text', '') for part in parts)
            return text.strip() if text.strip() else None
        if backend.get('provider') == 'ollama':
            import urllib.request

            body = json.dumps(
                {
                    'model': backend['model'],
                    'prompt': prompt,
                    'stream': False,
                }
            ).encode('utf-8')
            request = urllib.request.Request(
                f"{backend['base_url']}/api/generate",
                data=body,
                headers={'Content-Type': 'application/json'},
                method='POST',
            )
            with urllib.request.urlopen(request, timeout=20) as response:
                payload = json.loads(response.read().decode('utf-8'))
            output_text = str(payload.get('response', '')).strip()
            return output_text or None
    except Exception:
        return None
    return None


def _format_llm_response(question: str, tool_result: dict[str, Any], schema_info: dict[str, Any]) -> tuple[str | None, str | None]:
    backend = _get_llm_backend()
    if backend is None:
        return None, None
    prompt = _healthcare_llm_prompt(question, tool_result, schema_info)
    response = _complete_with_backend(backend, prompt)
    if not response:
        return None, None
    return response, str(backend.get('label', backend.get('provider', 'LLM')))


def _normalise_follow_ups(question: str, follow_ups: list[str] | None) -> list[str]:
    question_text = (question or '').strip().lower()
    deduped: list[str] = []
    for item in follow_ups or []:
        candidate = str(item).strip()
        if not candidate or candidate.lower() == question_text or candidate in deduped:
            continue
        deduped.append(candidate)
    return deduped[:4]


def _augment_copilot_payload(
    question: str,
    tool_result: dict[str, Any],
    schema_info: dict[str, Any],
    *,
    llm_answer: str | None = None,
    llm_provider: str | None = None,
    cached_response: bool = False,
) -> dict[str, Any]:
    answer_text = str(llm_answer or tool_result.get('answer', '') or '').strip()
    follow_ups = _normalise_follow_ups(question, tool_result.get('follow_ups', []))
    citations = _build_citations(tool_result, schema_info)
    explanation = _build_explanation(tool_result, schema_info)
    return {
        **tool_result,
        'answer': answer_text,
        'summary_text': answer_text,
        'content': answer_text,
        'follow_ups': follow_ups,
        'follow_up_questions': follow_ups,
        'citations': citations,
        'explanation': explanation,
        'llm_provider': llm_provider,
        'cached_response': cached_response,
    }


def route_question(question: str, data: pd.DataFrame, schema_info: dict) -> dict[str, Any]:
    if data.empty:
        return _response_payload(
            'no_data',
            'Load a dataset or broaden the current filters to unlock AI Copilot answers for this workflow.',
            structured_result={},
            table=None,
            chart=None,
            follow_ups=[],
        )

    intent = detect_intent(question)
    tools = {
        'dataset_summary': lambda: get_dataset_summary(data, schema_info),
        'healthcare_insights': lambda: get_key_healthcare_insights(data, schema_info),
        'department_readmission': lambda: get_department_readmission(data, schema_info),
        'readmission_high_risk': lambda: get_highest_readmission_risk(data, schema_info),
        'readmission_drivers': lambda: get_readmission_drivers(data, schema_info),
        'readmission_summary': lambda: get_readmission_summary(data, schema_info),
        'readmission_report': lambda: get_readmission_report(data, schema_info),
        'average_cost': lambda: get_average_cost(data),
        'average_length_of_stay': lambda: get_average_length_of_stay(data),
        'top_diagnosis_by_cost': lambda: get_top_diagnosis_by_cost(data),
    }
    if intent is None or intent not in tools:
        return _response_payload(
            'unsupported',
            'I can currently answer questions about dataset summary, key healthcare insights, high-risk patients, readmission by department, readmission drivers, readmission blockers, readmission summary, readmission reporting, average cost, average length of stay, and top diagnosis by cost.',
            structured_result={'supported_tools': sorted(tools.keys())},
            table=None,
            chart=None,
        follow_ups=_follow_ups('Show readmission by department', 'Which patients are highest readmission risk?'),
    )
    return tools[intent]()


def get_readmission_report(data: pd.DataFrame, schema_info: dict) -> dict[str, Any]:
    readmission = readmission_risk_analytics(data, schema_info.get('matched_schema', {}))
    if not readmission.get('available'):
        payload = _readmission_unavailable_payload(
            readmission,
            'A readmission report is not available yet because the current dataset is missing one or more supporting fields.',
        )
        payload['tool'] = 'get_readmission_report'
        payload['table'] = pd.DataFrame(
            [
                {
                    'status': 'Unavailable',
                    'what_still_works': ', '.join(payload['structured_result'].get('available_analysis', [])) or 'No readmission-specific views are available yet.',
                    'next_unlock': ', '.join(str(item).replace('_', ' ') for item in payload['structured_result'].get('unlock_fields', [])) or 'No additional fields specified',
                }
            ]
        )
        payload['table_data'] = payload['table']
        payload['follow_ups'] = _follow_ups('Show readiness gaps', 'What factors drive readmission?')
        payload['follow_up_questions'] = payload['follow_ups']
        return payload

    overview = readmission.get('overview', {})
    segments = readmission.get('high_risk_segments', pd.DataFrame())
    drivers = readmission.get('driver_table', pd.DataFrame())
    report_rows = [
        {
            'section': 'Overview',
            'headline': f"Overall readmission rate {float(overview.get('overall_readmission_rate', 0.0)):.1%}",
            'detail': f"{int(overview.get('readmission_count', 0))} readmissions across {int(overview.get('records_in_scope', 0))} reviewed records.",
        }
    ]
    if isinstance(segments, pd.DataFrame) and not segments.empty:
        top_segment = segments.iloc[0]
        report_rows.append(
            {
                'section': 'Risk segment',
                'headline': f"{top_segment['segment_type']} = {top_segment['segment_value']}",
                'detail': f"Readmission rate {float(top_segment['readmission_rate']):.1%} across {int(top_segment['record_count'])} records.",
            }
        )
    if isinstance(drivers, pd.DataFrame) and not drivers.empty:
        top_driver = drivers.iloc[0]
        report_rows.append(
            {
                'section': 'Top driver',
                'headline': f"{top_driver['factor']} / {top_driver['driver_group']}",
                'detail': f"{float(top_driver['gap_vs_overall']):.1%} above the overall readmission rate.",
            }
        )
    recommendations = readmission.get('intervention_recommendations', pd.DataFrame())
    if isinstance(recommendations, pd.DataFrame) and not recommendations.empty:
        top_recommendation = recommendations.iloc[0]
        report_rows.append(
            {
                'section': 'Intervention',
                'headline': str(top_recommendation.get('intervention', 'Intervention recommendation')),
                'detail': str(top_recommendation.get('why_it_matters', 'Start with the highest-gap readmission cohort.')),
            }
        )
    report_rows.append(
        {
            'section': 'Next step',
            'headline': 'Generate the readmission summary export',
            'detail': 'Open Export Center to download the readmission report with rates, drivers, interventions, and recommended next steps.',
        }
    )
    return _response_payload(
        'get_readmission_report',
        'The readmission workflow is ready for reporting. Export Center can generate a readmission summary with the current overview, high-risk segments, drivers, intervention recommendations, and recommended next steps.',
        structured_result=overview,
        table=pd.DataFrame(report_rows),
        chart=px.line(readmission['trend'], x='month', y='readmission_rate', title='Readmission Trend') if isinstance(readmission.get('trend'), pd.DataFrame) and not readmission['trend'].empty else None,
        follow_ups=_follow_ups('What factors drive readmission?', 'Show readmission by department'),
    )


def run_copilot_question(question: str, data: pd.DataFrame, schema_info: dict) -> dict[str, Any]:
    question_text = (question or '').strip()
    dataset_token = _dataset_cache_token(data, schema_info or {})
    logger_module.log_platform_event(
        'copilot_question_received',
        logger_name='copilot',
        question=question_text,
        row_count=int(len(data)),
        column_count=int(len(data.columns)),
    )
    if not question_text:
        return _response_payload(
            'empty',
            'Ask about dataset summary, high-risk patients, key healthcare insights, readmission by department, readmission drivers, readmission blockers, readmission interventions, average cost, average length of stay, or top diagnosis by cost.',
            structured_result={},
            table=None,
            chart=None,
            follow_ups=[],
        )

    cached_result = _cache_lookup(dataset_token, question_text)
    if cached_result is not None:
        logger_module.log_platform_event(
            'copilot_question_completed',
            logger_name='copilot',
            question=question_text,
            tool=cached_result.get('tool', ''),
            has_table=bool(cached_result.get('table') is not None),
            has_chart=bool(cached_result.get('chart') is not None),
            cached_response=True,
            llm_provider=cached_result.get('llm_provider'),
        )
        return cached_result

    tool_result = route_question(question_text, data, schema_info)
    llm_answer, llm_provider = _format_llm_response(question_text, tool_result, schema_info)
    tool_result = _augment_copilot_payload(
        question_text,
        tool_result,
        schema_info,
        llm_answer=llm_answer,
        llm_provider=llm_provider,
        cached_response=False,
    )
    _cache_store(dataset_token, question_text, tool_result)
    logger_module.log_platform_event(
        'copilot_question_completed',
        logger_name='copilot',
        question=question_text,
        tool=tool_result.get('tool', ''),
        has_table=bool(tool_result.get('table') is not None),
        has_chart=bool(tool_result.get('chart') is not None),
        cached_response=False,
        llm_provider=tool_result.get('llm_provider'),
    )
    return tool_result


def answer_question(question: str, data: pd.DataFrame, schema_info: dict) -> str:
    return run_copilot_question(question, data, schema_info)['answer']


def _simple_group_summary(data: pd.DataFrame, group_column: str, outcome_column: str | None = None, duration_column: str | None = None) -> pd.DataFrame:
    frame = data.copy()
    frame[group_column] = frame[group_column].astype(str)
    summary = frame.groupby(group_column).size().reset_index(name='record_count')
    if outcome_column and outcome_column in frame.columns:
        outcome = pd.to_numeric(frame[outcome_column], errors='coerce')
        frame = frame.assign(_copilot_outcome=outcome)
        outcome_summary = frame.groupby(group_column)['_copilot_outcome'].mean().reset_index(name='average_outcome')
        summary = summary.merge(outcome_summary, on=group_column, how='left')
    if duration_column and duration_column in frame.columns:
        duration = pd.to_numeric(frame[duration_column], errors='coerce')
        frame = frame.assign(_copilot_duration=duration)
        duration_summary = frame.groupby(group_column)['_copilot_duration'].mean().reset_index(name='average_duration')
        summary = summary.merge(duration_summary, on=group_column, how='left')
    return summary.sort_values('record_count', ascending=False).reset_index(drop=True)


def _segment_chart(table: pd.DataFrame, group_column: str, value_column: str, title: str):
    if table.empty or group_column not in table.columns or value_column not in table.columns:
        return None
    return px.bar(table.head(10), x=group_column, y=value_column, title=title)


def _extract_question_value(question_text: str, options: list[str]) -> str | None:
    lower_options = {str(option).lower(): str(option) for option in options}
    for lower, original in lower_options.items():
        if not lower:
            continue
        pattern = r'(?<![a-z0-9])' + re.escape(lower) + r'(?![a-z0-9])'
        if re.search(pattern, question_text):
            return original
    return None


def _cohort_action_from_question(question_text: str, data: pd.DataFrame, canonical_map: dict[str, str]) -> dict[str, Any] | None:
    gender_col = canonical_map.get('gender')
    stage_col = canonical_map.get('cancer_stage')
    treatment_col = canonical_map.get('treatment_type')
    risk_col = None
    risk_frame = data.copy()
    if 'risk_segment' in risk_frame.columns:
        risk_col = 'risk_segment'
    filters = {}
    widget_updates = {}
    if gender_col and gender_col in data.columns:
        gender = _extract_question_value(question_text, sorted(data[gender_col].dropna().astype(str).unique().tolist()))
        if gender:
            filters['genders'] = [gender]
            widget_updates['cohort_gender'] = [gender]
    if stage_col and stage_col in data.columns:
        stage = _extract_question_value(question_text, sorted(data[stage_col].dropna().astype(str).unique().tolist()))
        if stage:
            filters['cancer_stages'] = [stage]
            widget_updates['cohort_stage'] = [stage]
    if treatment_col and treatment_col in data.columns:
        treatment = _extract_question_value(question_text, sorted(data[treatment_col].dropna().astype(str).unique().tolist()))
        if treatment:
            filters['treatments'] = [treatment]
            widget_updates['cohort_treatment'] = [treatment]
    risk_options = ['High Risk', 'Medium Risk', 'Low Risk']
    for option in risk_options:
        if option.lower() in question_text:
            filters['risk_segments'] = [option]
            widget_updates['cohort_risk_segment'] = [option]
            break
    if not filters:
        return None
    cohort = build_cohort_summary(data, canonical_map, **filters)
    summary = cohort.get('summary', {}) if isinstance(cohort, dict) else {}
    table = pd.DataFrame([summary]) if summary else pd.DataFrame()
    return {
        'available': cohort.get('available', False),
        'action_type': 'build_cohort',
        'planned_action': 'Create a cohort from the detected question filters and prepare the cohort builder controls.',
        'message': 'The copilot found cohort criteria in your question and prepared a cohort preview.',
        'recommended_section': 'Healthcare Analytics - Cohort Analysis',
        'widget_updates': widget_updates,
        'preview_table': table,
        'preview_chart': None,
        'suggested_prompts': ['Compare treatment groups', 'Generate executive summary'],
    }


def _compare_treatments_action(question_text: str, data: pd.DataFrame, canonical_map: dict[str, str]) -> dict[str, Any] | None:
    treatment_col = canonical_map.get('treatment_type')
    outcome_col = canonical_map.get('survived')
    duration_col = 'treatment_duration_days' if 'treatment_duration_days' in data.columns else None
    if not treatment_col or treatment_col not in data.columns:
        return None
    options = sorted(data[treatment_col].dropna().astype(str).unique().tolist())
    mentioned = [option for option in options if option.lower() in question_text]
    if 'treatment' not in question_text and len(mentioned) < 2:
        return None
    if len(mentioned) >= 2:
        benchmark = benchmarking_analysis(
            data,
            canonical_map,
            'Treatment Type A vs Treatment Type B',
            treatment_a=mentioned[0],
            treatment_b=mentioned[1],
        )
        if benchmark.get('available'):
            table = benchmark.get('summary_table', pd.DataFrame())
        else:
            comparison = data[data[treatment_col].astype(str).isin(mentioned)].copy()
            table = _simple_group_summary(comparison, treatment_col, outcome_col, duration_col)
    else:
        comparison = data
        table = _simple_group_summary(comparison, treatment_col, outcome_col, duration_col)
    return {
        'available': not table.empty,
        'action_type': 'compare_treatments',
        'planned_action': 'Compare treatment groups using the current filtered dataset and surface a simple outcome summary.',
        'message': 'The copilot prepared a treatment-group comparison using the current dataset.',
        'recommended_section': 'Healthcare Analytics - Healthcare Intelligence',
        'widget_updates': {},
        'preview_table': table,
        'preview_chart': _segment_chart(table, treatment_col, 'record_count', 'Treatment group comparison'),
        'suggested_prompts': ['Build cohort for high risk patients', 'Generate clinical report'],
    }


def _compare_cohorts_action(question_text: str, data: pd.DataFrame, canonical_map: dict[str, str]) -> dict[str, Any] | None:
    if 'compare' not in question_text or 'cohort' not in question_text:
        return None
    cohort_action = _cohort_action_from_question(question_text, data, canonical_map)
    if cohort_action is None or not cohort_action.get('available'):
        return None

    widget_updates = cohort_action.get('widget_updates', {})
    filters = {
        'genders': widget_updates.get('cohort_gender'),
        'cancer_stages': widget_updates.get('cohort_stage'),
        'treatments': widget_updates.get('cohort_treatment'),
        'risk_segments': widget_updates.get('cohort_risk_segment'),
    }
    cohort = build_cohort_summary(
        data,
        canonical_map,
        age_range=widget_updates.get('cohort_age_range'),
        genders=filters.get('genders'),
        cancer_stages=filters.get('cancer_stages'),
        treatments=filters.get('treatments'),
        risk_segments=filters.get('risk_segments'),
    )
    if not cohort.get('available'):
        return None

    benchmark = benchmarking_analysis(
        data,
        canonical_map,
        'Current Cohort vs Full Dataset',
        cohort_frame=cohort.get('cohort_frame'),
    )
    table = benchmark.get('summary_table', pd.DataFrame()) if benchmark.get('available') else pd.DataFrame()
    metric_column = benchmark.get('metric_columns', [None])[0] if benchmark.get('available') else None
    chart = _segment_chart(table, benchmark.get('group_column', 'benchmark_group'), metric_column, 'Cohort vs Full Dataset') if metric_column else None
    return {
        'available': benchmark.get('available', False),
        'action_type': 'compare_cohorts',
        'planned_action': 'Build the requested cohort, compare it against the full dataset, and prepare the cohort-builder controls for review.',
        'message': 'The copilot prepared a cohort-versus-population comparison using the detected question filters.',
        'recommended_section': 'Healthcare Analytics - Cohort Analysis',
        'widget_updates': widget_updates,
        'preview_table': table,
        'preview_chart': chart,
        'suggested_prompts': ['Generate executive summary', 'What factors drive readmission?'],
    }


def _compare_segments_action(question_text: str, data: pd.DataFrame, canonical_map: dict[str, str], healthcare: dict[str, object]) -> dict[str, Any] | None:
    smoking_col = canonical_map.get('smoking_status')
    stage_col = canonical_map.get('cancer_stage')
    gender_col = canonical_map.get('gender')
    outcome_col = canonical_map.get('survived')
    duration_col = 'treatment_duration_days' if 'treatment_duration_days' in data.columns else None
    if 'segment' not in question_text and 'cohort' not in question_text and 'compare' not in question_text:
        return None
    group_column = None
    title = 'Segment comparison'
    if 'smoking' in question_text and smoking_col and smoking_col in data.columns:
        group_column = smoking_col
        title = 'Smoking segment comparison'
    elif 'stage' in question_text and stage_col and stage_col in data.columns:
        group_column = stage_col
        title = 'Cancer stage comparison'
    elif 'gender' in question_text and gender_col and gender_col in data.columns:
        group_column = gender_col
        title = 'Gender comparison'
    elif healthcare.get('risk_segmentation', {}).get('available') and 'risk_segment' in data.columns:
        group_column = 'risk_segment'
        title = 'Risk segment comparison'
    if not group_column:
        return None
    benchmark_type = None
    if group_column == smoking_col:
        benchmark_type = 'Smoking vs Non-Smoking Cohorts'
    elif group_column == stage_col:
        benchmark_type = 'Cancer Stage Groups'
    if benchmark_type:
        benchmark = benchmarking_analysis(data, canonical_map, benchmark_type)
        table = benchmark.get('summary_table', pd.DataFrame()) if benchmark.get('available') else pd.DataFrame()
    else:
        table = _simple_group_summary(data, group_column, outcome_col, duration_col)
    return {
        'available': not table.empty,
        'action_type': 'compare_segments',
        'planned_action': 'Compare the current population across the most relevant detected segment dimension from your question.',
        'message': f'The copilot prepared a {title.lower()} for the current dataset.',
        'recommended_section': 'Healthcare Analytics - Healthcare Intelligence',
        'widget_updates': {},
        'preview_table': table,
        'preview_chart': _segment_chart(table, group_column, 'record_count', title),
        'suggested_prompts': ['Show remediation suggestions', 'Generate operational report'],
    }


def plan_workflow_action(question: str, data: pd.DataFrame, canonical_map: dict[str, str], readiness: dict[str, object], healthcare: dict[str, object], remediation: pd.DataFrame | None = None) -> dict[str, Any]:
    question_text = (question or '').strip().lower()
    logger_module.log_platform_event(
        'copilot_workflow_action_requested',
        logger_name='copilot',
        question=question_text,
        row_count=int(len(data)),
        available_modules=int(readiness.get('available_count', 0)),
    )
    available_modules = set(
        readiness.get('readiness_table', pd.DataFrame())
        .loc[lambda frame: frame['status'] == 'Available', 'analysis_module']
        .tolist()
    ) if isinstance(readiness.get('readiness_table'), pd.DataFrame) and not readiness.get('readiness_table').empty else set()

    if not question_text:
        return {
            'available': False,
            'planned_action': 'No workflow action has been planned yet.',
            'message': 'Ask the copilot to build a cohort, compare segments or treatments, prepare a report, review readiness gaps, or focus on risk and root-cause analysis.',
            'widget_updates': {},
            'preview_table': pd.DataFrame(),
            'preview_chart': None,
            'suggested_prompts': [
                'Build a cohort for female stage iii patients',
                'Compare smoking vs non-smoking cohorts',
                'Generate executive summary',
            ],
        }

    compare_cohorts_action = _compare_cohorts_action(question_text, data, canonical_map)
    if compare_cohorts_action is not None:
        return compare_cohorts_action

    cohort_action = _cohort_action_from_question(question_text, data, canonical_map)
    if cohort_action is not None:
        return cohort_action

    treatment_action = _compare_treatments_action(question_text, data, canonical_map)
    if treatment_action is not None:
        return treatment_action

    segment_action = _compare_segments_action(question_text, data, canonical_map, healthcare)
    if segment_action is not None:
        return segment_action

    if 'readmission' in question_text and ('report' in question_text or 'summary' in question_text):
        return {
            'available': True,
            'action_type': 'readmission_report',
            'planned_action': 'Prepare a readmission-focused summary in Export Center and keep the final download under user control.',
            'message': 'Prepared Export Center for a readmission-focused operational handoff. Use the readmission summary export to share key rates, drivers, and intervention ideas.',
            'recommended_section': 'Insights & Export - Export Center',
            'widget_updates': {'report_mode': 'Operational Report'},
            'preview_table': pd.DataFrame([{'report_mode': 'Operational Report', 'focus': 'Readmission'}]),
            'preview_chart': None,
            'suggested_prompts': ['What factors drive readmission?', 'Which patients are highest readmission risk?'],
        }

    if 'readmission' in question_text and ('blocker' in question_text or 'missing' in question_text):
        return {
            'available': True,
            'action_type': 'readmission_readiness',
            'planned_action': 'Guide the user to the readmission readiness view and clarify what still works versus which fields unlock the full workflow.',
            'message': 'Open Healthcare Intelligence to review readmission readiness, current coverage, and the next fields to map for a fuller workflow.',
            'recommended_section': 'Healthcare Analytics - Healthcare Intelligence',
            'widget_updates': {},
            'preview_table': pd.DataFrame(),
            'preview_chart': None,
            'suggested_prompts': ['Summarize readmission risk', 'Generate a readmission summary report'],
        }

    if 'readmission' in question_text and ('driver' in question_text or 'factor' in question_text):
        return {
            'available': True,
            'action_type': 'readmission_drivers',
            'planned_action': 'Guide the user to the readmission driver analysis already prepared from the current dataset.',
            'message': 'Open Healthcare Intelligence to review the readmission driver table, top risk segments, and row-level high-risk list.',
            'recommended_section': 'Healthcare Analytics - Healthcare Intelligence',
            'widget_updates': {},
            'preview_table': pd.DataFrame(),
            'preview_chart': None,
            'suggested_prompts': ['Show readmission by department', 'Generate a readmission summary report'],
        }

    if 'report' in question_text or 'summary' in question_text:
        report_mode = 'Executive Summary'
        if 'analyst' in question_text:
            report_mode = 'Analyst Report'
        elif 'manager' in question_text or 'operations' in question_text or 'operational' in question_text:
            report_mode = 'Operational Report'
        elif 'clinical' in question_text:
            report_mode = 'Clinical Report'
        preview = pd.DataFrame([{'setting': 'report_mode', 'value': report_mode}])
        return {
            'available': True,
            'action_type': 'generate_report',
            'planned_action': f'Prepare the Export Center for a {report_mode.lower()} and leave the final download decision with the user.',
            'message': f"Prepared {report_mode} in Export Center. Open the export tab to preview or download it.",
            'recommended_section': 'Insights & Export - Export Center',
            'widget_updates': {'report_mode': report_mode},
            'preview_table': preview,
            'preview_chart': None,
            'suggested_prompts': ['Preview the executive summary', 'Show action recommendations'],
        }

    if 'presentation' in question_text or 'board' in question_text or 'demo view' in question_text:
        return {
            'available': True,
            'action_type': 'presentation_mode',
            'planned_action': 'Point the user to the most concise executive-style summary view without changing the current analysis state.',
            'message': 'Presentation Mode is the best next step for a concise board-style summary. Open Dataset Overview to review the executive snapshot and lead chart.',
            'recommended_section': 'Dataset Profile - Overview',
            'widget_updates': {},
            'preview_table': pd.DataFrame(),
            'preview_chart': None,
            'suggested_prompts': ['Prepare an executive summary', 'Show top recommendations'],
        }

    if 'monitor' in question_text or ('trend' in question_text and 'cohort' in question_text):
        if 'Cohort Monitoring Over Time' not in available_modules and not healthcare.get('risk_segmentation', {}).get('available'):
            return {
                'available': False,
                'planned_action': 'Check whether a usable date field and cohort-tracking context are available before trying to open cohort monitoring.',
                'message': 'Cohort monitoring needs a usable date field and a cohort that can be tracked over time. Review Analysis Readiness to see what is still missing.',
                'widget_updates': {},
                'preview_table': pd.DataFrame(),
                'preview_chart': None,
                'suggested_prompts': ['Show readiness gaps', 'Open data remediation assistant'],
            }
        return {
            'available': True,
            'action_type': 'cohort_monitoring',
            'planned_action': 'Guide the user to cohort monitoring and preserve the current cohort scope for time-based review.',
            'message': 'Prepared the app for cohort monitoring. Open Healthcare Intelligence to compare the current cohort against the full dataset over time.',
            'recommended_section': 'Healthcare Analytics - Cohort Analysis',
            'widget_updates': {'cohort_monitor_metric': 'record_count'},
            'preview_table': pd.DataFrame([{'monitoring_metric': 'record_count'}]),
            'preview_chart': None,
            'suggested_prompts': ['Compare smoking vs non-smoking cohorts', 'Show operational alerts'],
        }

    if 'root cause' in question_text or 'why' in question_text:
        target = 'Low Survival'
        if 'risk' in question_text:
            target = 'High Risk Share'
        elif 'duration' in question_text:
            target = 'Long Treatment Duration'
        return {
            'available': True,
            'action_type': 'root_cause',
            'planned_action': f'Prepare the root-cause review around {target.lower()} without changing the underlying dataset or filters.',
            'message': f"Prepared Root Cause Explorer for {target.lower()}. Open Healthcare Intelligence to review the drill-down.",
            'recommended_section': 'Healthcare Analytics - Healthcare Intelligence',
            'widget_updates': {'root_cause_target': target},
            'preview_table': pd.DataFrame([{'root_cause_target': target}]),
            'preview_chart': None,
            'suggested_prompts': ['Show driver analysis', 'Compare cancer stage groups'],
        }

    if 'alert' in question_text or 'operational' in question_text:
        alerts = healthcare.get('operational_alerts', {})
        preview = alerts.get('alerts_table', pd.DataFrame()) if isinstance(alerts, dict) else pd.DataFrame()
        return {
            'available': True,
            'action_type': 'operational_alerts',
            'planned_action': 'Open the operational monitoring path and surface the current alert summary for review.',
            'message': 'Operational Alerts is the right next step for threshold-based monitoring. Open Healthcare Intelligence to review active alerts and supporting rationale.',
            'recommended_section': 'Healthcare Analytics - Healthcare Intelligence',
            'widget_updates': {},
            'preview_table': preview.head(8) if isinstance(preview, pd.DataFrame) else pd.DataFrame(),
            'preview_chart': None,
            'suggested_prompts': ['Compare cohorts', 'Open root cause review for survival'],
        }

    if 'intervention' in question_text or 'scenario' in question_text or 'simulate' in question_text:
        scenario = healthcare.get('scenario', {})
        preview = pd.DataFrame([scenario]) if isinstance(scenario, dict) and scenario.get('available') else pd.DataFrame()
        return {
            'available': True,
            'action_type': 'intervention_planning',
            'planned_action': 'Guide the user to intervention planning and surface the current scenario assumptions if the dataset supports them.',
            'message': 'Prepared the app for intervention planning. Open Healthcare Intelligence to compare scenario assumptions and intervention impact.',
            'recommended_section': 'Healthcare Analytics - Healthcare Intelligence',
            'widget_updates': {},
            'preview_table': preview,
            'preview_chart': None,
            'suggested_prompts': ['Prepare an executive summary', 'Show top recommendations'],
        }

    if 'readiness' in question_text or 'remediation' in question_text or 'unlock' in question_text or 'missing fields' in question_text:
        preview = remediation.head(6) if isinstance(remediation, pd.DataFrame) else pd.DataFrame()
        return {
            'available': True,
            'action_type': 'readiness_review',
            'planned_action': 'Open Analysis Readiness and surface the highest-impact remediation items first.',
            'message': 'The next best step is Analysis Readiness, where the app can show missing fields, remediation suggestions, and which modules would unlock next.',
            'recommended_section': 'Data Quality - Analysis Readiness',
            'widget_updates': {},
            'preview_table': preview,
            'preview_chart': None,
            'suggested_prompts': ['Open data quality review', 'Prepare an analyst report'],
        }

    if 'benchmark' in question_text:
        risk = healthcare.get('risk_segmentation', {})
        preview = risk.get('segment_table', pd.DataFrame()) if isinstance(risk, dict) else pd.DataFrame()
        return {
            'available': True,
            'action_type': 'benchmarking',
            'planned_action': 'Guide the user to a benchmark-style comparison using the current segmentation outputs.',
            'message': 'Use Healthcare Intelligence and Cohort Analysis to benchmark the current cohort against the broader dataset context.',
            'recommended_section': 'Healthcare Analytics - Cohort Analysis',
            'widget_updates': {},
            'preview_table': preview.head(6) if isinstance(preview, pd.DataFrame) else pd.DataFrame(),
            'preview_chart': _segment_chart(preview, 'risk_segment', 'patient_count', 'Current segment mix') if isinstance(preview, pd.DataFrame) else None,
            'suggested_prompts': ['Compare treatment groups', 'Show remediation suggestions'],
        }

    if 'risk' in question_text or 'high risk' in question_text:
        if healthcare.get('risk_segmentation', {}).get('available'):
            preview = healthcare.get('risk_segmentation', {}).get('segment_table', pd.DataFrame())
            return {
                'available': True,
                'action_type': 'risk_review',
                'planned_action': 'Open the risk review path and surface the current segment mix for context.',
                'message': 'The dataset supports patient risk review. Open Healthcare Intelligence to inspect Patient Risk Segmentation and Explainability & Fairness.',
                'recommended_section': 'Healthcare Analytics - Healthcare Intelligence',
                'widget_updates': {},
                'preview_table': preview.head(6) if isinstance(preview, pd.DataFrame) else pd.DataFrame(),
                'preview_chart': _segment_chart(preview, 'risk_segment', 'patient_count', 'Risk segment mix') if isinstance(preview, pd.DataFrame) else None,
                'suggested_prompts': ['Show operational alerts', 'Open root cause review for risk'],
            }
        return {
            'available': False,
            'planned_action': 'Check whether enough risk-driving fields exist before attempting a risk review.',
            'message': 'Risk review is not available yet because the current dataset does not expose enough risk-driving fields. Analysis Readiness can show what would unlock it.',
            'widget_updates': {},
            'preview_table': pd.DataFrame(),
            'preview_chart': None,
            'suggested_prompts': ['Show readiness gaps', 'Open data remediation assistant'],
        }

    return {
        'available': False,
        'planned_action': 'No supported deterministic workflow action matched the current question.',
        'message': 'Supported workflow actions include building cohorts, comparing segments or treatments, preparing reports, reviewing readiness gaps, and focusing on alerts, risk, and root-cause analysis.',
        'widget_updates': {},
        'preview_table': pd.DataFrame(),
        'preview_chart': None,
        'suggested_prompts': [
            'Build a cohort for female stage iii patients',
            'Compare smoking vs non-smoking cohorts',
            'Generate executive summary',
        ],
    }
