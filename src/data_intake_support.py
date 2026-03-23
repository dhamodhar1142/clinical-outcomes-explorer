from __future__ import annotations

from typing import Any

import pandas as pd

from src.profiler import build_numeric_summary
from src.semantic_mapper import CANONICAL_FIELDS

try:
    import plotly.graph_objects as go
except Exception:  # pragma: no cover - optional chart dependency
    go = None


ROLE_OPTIONS = ['Admin', 'Analyst', 'Executive', 'Clinician', 'Researcher', 'Data Steward', 'Viewer']
REPORT_MODES = ['Analyst Report', 'Operational Report', 'Executive Summary', 'Clinical Report', 'Data Readiness Review', 'Population Health Summary']
PLAN_OPTIONS = ['Free', 'Pro', 'Team', 'Enterprise']
PLAN_ENFORCEMENT_OPTIONS = ['Demo-safe', 'Strict']
TAB_LABELS = ['Data Intake', 'Dataset Profile', 'Data Quality', 'Healthcare Analytics', 'Key Insights & Export']


def safe_df(value: Any) -> pd.DataFrame:
    return value if isinstance(value, pd.DataFrame) else pd.DataFrame()


def fmt(value: Any) -> str:
    if isinstance(value, int):
        return f'{value:,}'
    if isinstance(value, float):
        return f'{value:,.2f}'
    return str(value)


def build_deployment_support_notes() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                'support_area': 'Environment profile',
                'guidance': 'Keep local/demo installs fallback-safe, and use staging/production profiles to tighten persistence, worker, and storage expectations.',
            },
            {
                'support_area': 'Large dataset handling',
                'guidance': 'Use summary-first profiling and capped previews when sampling is active so the UI stays responsive on very large files.',
            },
            {
                'support_area': 'Export and storage',
                'guidance': 'Prefer ZIP bundle and object-storage persistence for large stakeholder handoffs instead of very large inline downloads.',
            },
            {
                'support_area': 'Worker backend',
                'guidance': 'Move heavy profiling, modeling, and export operations onto the configured job backend when a worker is available.',
            },
        ]
    )


def build_lineage_sankey(lineage: dict[str, Any]):
    if go is None:
        return None
    derived_fields = safe_df(lineage.get('derived_fields_table'))
    if derived_fields.empty:
        return None
    labels = ['Source Dataset', 'Standardized Columns', *derived_fields['derived_role'].astype(str).tolist()]
    source = [0, *([1] * len(derived_fields))]
    target = [1, *range(2, len(labels))]
    values = [max(len(derived_fields), 1), *([1] * len(derived_fields))]
    return go.Figure(
        go.Sankey(
            arrangement='snap',
            node={'label': labels, 'pad': 14, 'thickness': 16},
            link={'source': source, 'target': target, 'value': values},
        )
    )


def build_mapping_confidence_table(pipeline: dict[str, Any]) -> pd.DataFrame:
    mapping_table = safe_df(pipeline.get('semantic', {}).get('mapping_table'))
    suggestion_table = safe_df(pipeline.get('semantic', {}).get('suggestion_table'))
    field_profile = safe_df(pipeline.get('field_profile'))
    if mapping_table.empty:
        if suggestion_table.empty:
            return pd.DataFrame(columns=['source_column', 'mapped_field', 'confidence_score', 'confidence_label', 'inferred_type', 'used_downstream', 'top_suggestions'])
        fallback_rows = []
        for source_column, group in suggestion_table.groupby('source_column'):
            top = group.sort_values(['suggestion_rank', 'confidence_score'], ascending=[True, False]).head(3)
            fallback_rows.append(
                {
                    'source_column': source_column,
                    'mapped_field': top.iloc[0]['suggested_field'],
                    'confidence_score': float(top.iloc[0]['confidence_score']),
                    'confidence_label': top.iloc[0]['confidence_label'],
                    'inferred_type': 'unknown',
                    'used_downstream': bool(top.iloc[0].get('auto_apply', False)),
                    'top_suggestions': ', '.join(
                        f"{row['suggested_field']} ({float(row['confidence_score']):.2f})"
                        for _, row in top.iterrows()
                    ),
                }
            )
        return pd.DataFrame(fallback_rows).sort_values(['confidence_score', 'source_column'], ascending=[False, True]).reset_index(drop=True)
    profile_lookup = field_profile.set_index('column_name').to_dict('index') if not field_profile.empty else {}
    suggestion_lookup = {
        str(source_column): group.sort_values(['suggestion_rank', 'confidence_score'], ascending=[True, False]).head(3)
        for source_column, group in suggestion_table.groupby('source_column')
    } if not suggestion_table.empty else {}
    rows = []
    for row in mapping_table.to_dict(orient='records'):
        profile = profile_lookup.get(str(row.get('original_column', '')), {})
        source_column = str(row.get('original_column', ''))
        top_suggestions = suggestion_lookup.get(source_column, pd.DataFrame())
        rows.append(
            {
                'source_column': source_column,
                'mapped_field': row.get('semantic_label', ''),
                'confidence_score': float(row.get('confidence_score', 0.0) or 0.0),
                'confidence_label': row.get('confidence_label', 'Low'),
                'inferred_type': profile.get('inferred_type', 'unknown'),
                'used_downstream': bool(row.get('used_downstream', False)),
                'top_suggestions': ', '.join(
                    f"{suggestion_row['suggested_field']} ({float(suggestion_row['confidence_score']):.2f})"
                    for _, suggestion_row in top_suggestions.iterrows()
                ) or str(row.get('semantic_label', '')),
            }
        )
    return pd.DataFrame(rows).sort_values(['confidence_score', 'source_column'], ascending=[False, True]).reset_index(drop=True)


def build_remap_board(pipeline: dict[str, Any]) -> pd.DataFrame:
    field_profile = safe_df(pipeline.get('field_profile'))
    mapping_table = safe_df(pipeline.get('semantic', {}).get('mapping_table'))
    suggestion_table = safe_df(pipeline.get('semantic', {}).get('suggestion_table'))
    mapped_lookup = mapping_table.set_index('original_column').to_dict('index') if not mapping_table.empty else {}
    suggestion_lookup = {
        str(source_column): group.sort_values(['suggestion_rank', 'confidence_score'], ascending=[True, False]).head(1).iloc[0].to_dict()
        for source_column, group in suggestion_table.groupby('source_column')
    } if not suggestion_table.empty else {}
    rows: list[dict[str, Any]] = []
    for order, column_name in enumerate(field_profile.get('column_name', pd.Series(dtype=str)).astype(str).tolist(), start=1):
        mapping = mapped_lookup.get(column_name, {})
        suggestion = suggestion_lookup.get(column_name, {})
        rows.append(
            {
                'display_order': order,
                'source_column': column_name,
                'mapped_field': mapping.get('semantic_label', suggestion.get('suggested_field', 'Not mapped')),
                'confidence_score': float(mapping.get('confidence_score', suggestion.get('confidence_score', 0.0)) or 0.0),
                'confidence_label': mapping.get('confidence_label', suggestion.get('confidence_label', 'Low')),
                'inferred_type': field_profile.loc[field_profile['column_name'] == column_name, 'inferred_type'].iloc[0] if not field_profile.empty else 'unknown',
                'suggested_field': suggestion.get('suggested_field', 'Not mapped'),
                'suggested_confidence': float(suggestion.get('confidence_score', 0.0) or 0.0),
            }
        )
    return pd.DataFrame(rows)


def build_mapping_template(board: pd.DataFrame, *, template_name: str, dataset_type: str = '') -> dict[str, Any]:
    safe_board = safe_df(board)
    return {
        'template_name': template_name,
        'dataset_type': dataset_type,
        'mappings': safe_board[['source_column', 'mapped_field', 'display_order']].to_dict(orient='records')
        if not safe_board.empty else [],
    }


def apply_mapping_template(board: pd.DataFrame, template: dict[str, Any]) -> pd.DataFrame:
    safe_board = safe_df(board).copy()
    template_rows = pd.DataFrame(template.get('mappings', []))
    if safe_board.empty or template_rows.empty:
        return safe_board
    merged = safe_board.merge(
        template_rows,
        on='source_column',
        how='left',
        suffixes=('', '_template'),
    )
    merged['mapped_field'] = merged['mapped_field_template'].fillna(merged['mapped_field'])
    merged['display_order'] = merged['display_order_template'].fillna(merged['display_order'])
    return merged[[column for column in safe_board.columns if column in merged.columns]].sort_values(['display_order', 'source_column']).reset_index(drop=True)


def apply_auto_mapping_suggestions(board: pd.DataFrame) -> pd.DataFrame:
    safe_board = safe_df(board).copy()
    if safe_board.empty or 'suggested_field' not in safe_board.columns:
        return safe_board
    replacement_mask = (
        safe_board['mapped_field'].astype(str).eq('Not mapped')
        | (
            safe_board['suggested_confidence'].astype(float) > safe_board['confidence_score'].astype(float)
        )
    )
    safe_board.loc[replacement_mask, 'mapped_field'] = safe_board.loc[replacement_mask, 'suggested_field']
    safe_board.loc[replacement_mask, 'confidence_score'] = safe_board.loc[replacement_mask, 'suggested_confidence']
    safe_board.loc[replacement_mask, 'confidence_label'] = safe_board.loc[replacement_mask, 'suggested_confidence'].map(
        lambda value: 'High' if float(value) >= 0.82 else 'Medium' if float(value) >= 0.62 else 'Low'
    )
    return safe_board.sort_values(['display_order', 'source_column']).reset_index(drop=True)


def build_data_profiling_summary(pipeline: dict[str, Any]) -> dict[str, Any]:
    field_profile = safe_df(pipeline.get('field_profile'))
    sample_info = dict(pipeline.get('sample_info', {}))
    summary_cards = [
        {'label': 'Rows in scope', 'value': fmt(int(pipeline.get('overview', {}).get('rows', 0) or 0))},
        {'label': 'Profile sample rows', 'value': fmt(int(sample_info.get('profile_sample_rows', 0) or 0))},
        {'label': 'Columns profiled', 'value': fmt(len(field_profile))},
        {'label': 'Sampling Applied', 'value': 'Yes' if bool(sample_info.get('sampling_applied')) else 'No'},
    ]
    high_risk = field_profile[
        (field_profile.get('null_percentage', pd.Series(dtype=float)) >= 0.40)
        | (field_profile.get('is_high_cardinality_identifier', pd.Series(dtype=bool)) == True)
        | (field_profile.get('has_numeric_outlier_signal', pd.Series(dtype=bool)) == True)
    ].copy()
    if not high_risk.empty:
        keep_columns = [column for column in ['column_name', 'inferred_type', 'null_percentage', 'unique_count', 'outlier_count', 'sample_values'] if column in high_risk.columns]
        high_risk = high_risk[keep_columns].head(20)
    return {
        'summary_cards': summary_cards,
        'high_risk_table': high_risk,
        'numeric_summary': build_numeric_summary(field_profile).head(20),
        'field_profile_preview': field_profile.head(25),
    }


__all__ = [
    'CANONICAL_FIELDS',
    'PLAN_ENFORCEMENT_OPTIONS',
    'PLAN_OPTIONS',
    'REPORT_MODES',
    'ROLE_OPTIONS',
    'TAB_LABELS',
    'build_data_profiling_summary',
    'build_deployment_support_notes',
    'build_lineage_sankey',
    'build_mapping_confidence_table',
    'apply_auto_mapping_suggestions',
    'apply_mapping_template',
    'build_mapping_template',
    'build_remap_board',
]
