from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

import src.auth as auth_module
import src.logger as logger_module
from src.data_loader import DEMO_DATASETS
from src.modules.audit import log_audit_event

TABLE_BADGE_STYLES = {
    'critical': 'background-color: #FEECEC; color: #991B1B; font-weight: 700; border-radius: 999px; padding: 0.14rem 0.5rem; border: 1px solid rgba(220, 38, 38, 0.12);',
    'error': 'background-color: #FEECEC; color: #991B1B; font-weight: 700; border-radius: 999px; padding: 0.14rem 0.5rem; border: 1px solid rgba(220, 38, 38, 0.12);',
    'high': 'background-color: #FFF5E6; color: #9A6700; font-weight: 700; border-radius: 999px; padding: 0.14rem 0.5rem; border: 1px solid rgba(245, 158, 11, 0.14);',
    'warning': 'background-color: #FFF5E6; color: #9A6700; font-weight: 700; border-radius: 999px; padding: 0.14rem 0.5rem; border: 1px solid rgba(245, 158, 11, 0.14);',
    'medium': 'background-color: #FFF7ED; color: #B45309; font-weight: 700; border-radius: 999px; padding: 0.14rem 0.5rem; border: 1px solid rgba(245, 158, 11, 0.12);',
    'moderate': 'background-color: #FFF7ED; color: #B45309; font-weight: 700; border-radius: 999px; padding: 0.14rem 0.5rem; border: 1px solid rgba(245, 158, 11, 0.12);',
    'low': 'background-color: #F1F5F9; color: #334155; font-weight: 700; border-radius: 999px; padding: 0.14rem 0.5rem; border: 1px solid rgba(71, 85, 105, 0.12);',
    'resolved': 'background-color: #EAF8EF; color: #166534; font-weight: 700; border-radius: 999px; padding: 0.14rem 0.5rem; border: 1px solid rgba(22, 163, 74, 0.12);',
    'success': 'background-color: #EAF8EF; color: #166534; font-weight: 700; border-radius: 999px; padding: 0.14rem 0.5rem; border: 1px solid rgba(22, 163, 74, 0.12);',
    'available': 'background-color: #EAF8EF; color: #166534; font-weight: 700; border-radius: 999px; padding: 0.14rem 0.5rem; border: 1px solid rgba(22, 163, 74, 0.12);',
    'ready': 'background-color: #EAF8EF; color: #166534; font-weight: 700; border-radius: 999px; padding: 0.14rem 0.5rem; border: 1px solid rgba(22, 163, 74, 0.12);',
    'partial': 'background-color: #FFF7ED; color: #B45309; font-weight: 700; border-radius: 999px; padding: 0.14rem 0.5rem; border: 1px solid rgba(245, 158, 11, 0.12);',
    'limited': 'background-color: #FFF7ED; color: #B45309; font-weight: 700; border-radius: 999px; padding: 0.14rem 0.5rem; border: 1px solid rgba(245, 158, 11, 0.12);',
    'inferred': 'background-color: #E7FBF8; color: #0F766E; font-weight: 700; border-radius: 999px; padding: 0.14rem 0.5rem; border: 1px solid rgba(20, 184, 166, 0.12);',
    'preserved': 'background-color: #E7FBF8; color: #0F766E; font-weight: 700; border-radius: 999px; padding: 0.14rem 0.5rem; border: 1px solid rgba(20, 184, 166, 0.12);',
    'remapped': 'background-color: #E7FBF8; color: #0F766E; font-weight: 700; border-radius: 999px; padding: 0.14rem 0.5rem; border: 1px solid rgba(20, 184, 166, 0.12);',
    'defaulted': 'background-color: #F1F5F9; color: #334155; font-weight: 700; border-radius: 999px; padding: 0.14rem 0.5rem; border: 1px solid rgba(71, 85, 105, 0.12);',
    'review': 'background-color: #FEECEC; color: #991B1B; font-weight: 700; border-radius: 999px; padding: 0.14rem 0.5rem; border: 1px solid rgba(220, 38, 38, 0.10);',
    'manual': 'background-color: #FEECEC; color: #991B1B; font-weight: 700; border-radius: 999px; padding: 0.14rem 0.5rem; border: 1px solid rgba(220, 38, 38, 0.10);',
    'high confidence': 'background-color: #EAF8EF; color: #166534; font-weight: 700; border-radius: 999px; padding: 0.14rem 0.5rem; border: 1px solid rgba(22, 163, 74, 0.12);',
    'medium confidence': 'background-color: #FFF7ED; color: #B45309; font-weight: 700; border-radius: 999px; padding: 0.14rem 0.5rem; border: 1px solid rgba(245, 158, 11, 0.12);',
    'low confidence': 'background-color: #F1F5F9; color: #334155; font-weight: 700; border-radius: 999px; padding: 0.14rem 0.5rem; border: 1px solid rgba(71, 85, 105, 0.12);',
}


def _table_badge_style(value: Any) -> str:
    normalized = str(value).strip().lower()
    if not normalized:
        return ''
    for token, style in TABLE_BADGE_STYLES.items():
        if token in normalized:
            return style
    return ''


def _styled_table(table: pd.DataFrame):
    styler = table.style
    badge_columns = [
        column for column in table.columns
        if any(
            token in str(column).strip().lower()
            for token in (
                'confidence',
                'severity',
                'status',
                'review',
                'method',
                'resolution',
                'outcome',
                'rule',
                'flag',
                'risk',
                'support level',
            )
        )
    ]
    for column in badge_columns:
        styler = styler.applymap(_table_badge_style, subset=pd.IndexSlice[:, [column]])
    styler = styler.set_properties(**{
        'border-bottom': '1px solid #E2E8F0',
        'color': '#0F172A',
        'font-size': '0.92rem',
        'background-color': '#F4F8FC',
    })
    styler = styler.set_table_styles(
        [
            {
                'selector': 'table',
                'props': [
                    ('background-color', '#F4F8FC'),
                    ('color', '#0F172A'),
                    ('border-collapse', 'separate'),
                    ('border-spacing', '0'),
                ],
            },
            {
                'selector': 'thead th',
                'props': [
                    ('background-color', '#EAF2F8'),
                    ('color', '#0F172A'),
                    ('font-weight', '700'),
                    ('border-bottom', '1px solid #E2E8F0'),
                ],
            },
            {
                'selector': 'tbody td',
                'props': [
                    ('background-color', '#F4F8FC'),
                    ('color', '#0F172A'),
                    ('padding', '0.42rem 0.55rem'),
                    ('border-bottom', '1px solid #E2E8F0'),
                ],
            },
            {
                'selector': 'tbody tr:nth-child(even) td',
                'props': [('background-color', '#EDF5FB')],
            },
            {
                'selector': 'tbody tr:hover td',
                'props': [('background-color', '#E8F1F8')],
            },
        ]
    )
    return styler

def log_event(
    event_type: str,
    details: str,
    user_interaction: str = 'User action',
    analysis_step: str = 'Session activity',
    *,
    resource_type: str = '',
    resource_name: str = '',
    action_outcome: str = 'success',
) -> None:
    identity = st.session_state.get('workspace_identity') or {'workspace_id': 'guest-demo-workspace', 'user_id': 'guest-user'}
    log_context = logger_module.build_log_context(
        st.session_state,
        event_type=event_type,
        user_interaction=user_interaction,
        analysis_step=analysis_step,
        resource_type=resource_type,
        resource_name=resource_name,
        action_outcome=action_outcome,
    )
    logger_module.log_platform_event(
        'ui_event',
        logger_name='ui',
        details=details,
        **log_context,
    )
    application_service = st.session_state.get('application_service')
    if application_service is not None:
        application_service.record_usage_event(
            identity,
            {
                'event_type': event_type,
                'details': details,
                'user_interaction': user_interaction,
                'analysis_step': analysis_step,
            },
        )
    st.session_state['analysis_log'] = log_audit_event(
        st.session_state.get('analysis_log', []),
        event_type,
        details,
        user_interaction=user_interaction,
        analysis_step=analysis_step,
        actor_context={
            **identity,
            'session_id': st.session_state.get('platform_session_id', ''),
            'request_id': st.session_state.get('platform_request_id', ''),
        },
        resource_context={
            'resource_type': resource_type,
            'resource_name': resource_name,
            'action_outcome': action_outcome,
        },
    )
    if application_service is not None:
        application_service.persist_workspace_state(st.session_state)

def safe_df(table: Any, columns: list[str] | None = None) -> pd.DataFrame:
    return table if isinstance(table, pd.DataFrame) else pd.DataFrame(columns=columns or [])

def info_or_table(table: pd.DataFrame, message: str) -> None:
    if table.empty:
        st.info(message)
    else:
        st.dataframe(_styled_table(table), width='stretch')

def info_or_chart(fig, message: str) -> None:
    if fig is None:
        st.info(message)
    else:
        st.plotly_chart(fig, width='stretch')

def tracked_download_button(
    label: str,
    *,
    data,
    file_name: str,
    mime: str,
    event_detail: str,
    disabled: bool = False,
    resource_type: str = 'export_artifact',
    resource_name: str | None = None,
) -> None:
    st.download_button(
        label,
        data=data,
        file_name=file_name,
        mime=mime,
        disabled=disabled,
        on_click=log_event,
        args=('Export Downloaded', event_detail, 'Export download', 'Export center'),
        kwargs={
            'resource_type': resource_type,
            'resource_name': resource_name or file_name,
        },
    )


def workspace_allows_permission(permission: str) -> bool:
    identity = st.session_state.get('workspace_identity') or {}
    return auth_module.workspace_identity_can_access(identity, permission)

def fmt(value: Any, kind: str = 'int') -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return 'Not available'
    if kind == 'pct':
        return f'{float(value):.1%}'
    if kind == 'score':
        return f'{float(value) * 100:.0f}/100'
    if kind == 'float':
        return f'{float(value):,.2f}'
    if kind == 'money':
        return f'${float(value):,.2f}'
    return f'{int(value):,}' if isinstance(value, (int, float)) else str(value)

def build_demo_dataset_cards() -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for dataset_name, details in DEMO_DATASETS.items():
        rows.append(
            {
                'demo_dataset': dataset_name,
                'best_for': str(details.get('best_for', 'General walkthroughs and readiness review.')),
                'what_it_demonstrates': str(details.get('description', 'Built-in dataset for onboarding and feature walkthroughs.')),
            }
        )
    return pd.DataFrame(rows)

def build_recommended_workflow_component(pipeline: dict[str, Any]) -> dict[str, object]:
    intelligence = pipeline.get('dataset_intelligence', {})
    demo_guidance = pipeline.get('demo_guidance', {})
    use_case = pipeline.get('use_case_detection', {})
    solution_packages = pipeline.get('solution_packages', {})

    steps = [
        '1. Dataset Overview',
        '2. Data Quality Review',
        '3. Healthcare Analytics',
        '4. Insights & Reporting',
    ]
    return {
        'dataset_type': str(intelligence.get('dataset_type_label', demo_guidance.get('detected_dataset_type', 'Dataset type not classified'))),
        'recommended_workflow': str(demo_guidance.get('recommended_workflow', use_case.get('recommended_workflow', 'Healthcare Data Readiness'))),
        'recommended_package': str(demo_guidance.get('recommended_package', solution_packages.get('recommended_package', 'Healthcare Data Readiness'))),
        'rationale': str(demo_guidance.get('narrative', use_case.get('narrative', 'Use this guided flow to move from dataset understanding to stakeholder-ready outputs.'))),
        'steps': steps,
    }

def build_cohort_placeholder_frame(
    data: pd.DataFrame,
    *,
    diagnosis_col: str | None,
    treatment_col: str | None,
    risk_table: pd.DataFrame,
) -> tuple[pd.DataFrame, str]:
    if diagnosis_col and diagnosis_col in data.columns:
        diagnosis_counts = (
            data[diagnosis_col]
            .dropna()
            .astype(str)
            .value_counts()
            .head(8)
            .rename_axis('diagnosis')
            .reset_index(name='record_count')
        )
        if not diagnosis_counts.empty:
            return diagnosis_counts, 'Top diagnosis groups available for cohort exploration'
    if treatment_col and treatment_col in data.columns:
        treatment_counts = (
            data[treatment_col]
            .dropna()
            .astype(str)
            .value_counts()
            .head(8)
            .rename_axis('treatment')
            .reset_index(name='record_count')
        )
        if not treatment_counts.empty:
            return treatment_counts, 'Top treatment groups available for cohort exploration'
    if isinstance(risk_table, pd.DataFrame) and not risk_table.empty and {'risk_segment', 'patient_count'}.issubset(risk_table.columns):
        return risk_table[['risk_segment', 'patient_count']].head(8), 'Current risk segment mix'
    return pd.DataFrame(columns=['group_name', 'record_count']), 'Add diagnosis, treatment, or risk-supporting fields to unlock guided cohort exploration'

