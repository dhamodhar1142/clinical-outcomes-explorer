from __future__ import annotations

from io import StringIO

import pandas as pd

from src.reports.common import _combine_report_tables, _safe_df, dataframe_to_csv_bytes


def build_claims_export_tables(
    dataset_name: str,
    overview: dict[str, object],
    claims_workflow: dict[str, object],
) -> dict[str, pd.DataFrame]:
    summary_rows = [
        {'metric': 'Dataset', 'value': dataset_name},
        {'metric': 'Rows in scope', 'value': int(overview.get('rows', 0) or 0)},
        {'metric': 'Analyzed columns', 'value': int(overview.get('analyzed_columns', overview.get('columns', 0)) or 0)},
        {'metric': 'Workflow status', 'value': str(claims_workflow.get('readiness_label', 'Not available'))},
    ]
    for card in claims_workflow.get('summary_cards', []):
        summary_rows.append({'metric': str(card.get('label', 'Summary')), 'value': str(card.get('value', ''))})
    qc_summary = pd.DataFrame(summary_rows)

    issue_log = _safe_df(claims_workflow.get('validation_table')).copy()
    flagged_rows = _safe_df(claims_workflow.get('flagged_rows')).copy()
    if not flagged_rows.empty:
        flagged_rows = flagged_rows.copy()
        flagged_rows.insert(0, 'issue_source', 'Flagged claims review')
    if not issue_log.empty:
        issue_log = issue_log.copy()
        issue_log.insert(0, 'issue_source', 'Validation checks')
    claims_validation_issue_log = _combine_report_tables(
        {
            'Validation checks': issue_log,
            'Flagged claims review': flagged_rows,
        }
    )

    utilization_metrics = _combine_report_tables(
        {
            'Financial summary': _safe_df(claims_workflow.get('financial_summary')),
            'Payer utilization': _safe_df(claims_workflow.get('payer_utilization')),
            'Provider utilization': _safe_df(claims_workflow.get('provider_utilization')),
            'Diagnosis utilization': _safe_df(claims_workflow.get('diagnosis_utilization')),
            'Monthly claims trend': _safe_df(claims_workflow.get('monthly_utilization')),
        }
    )
    return {
        'qc_summary': qc_summary,
        'claims_validation_issue_log': claims_validation_issue_log,
        'utilization_metrics': utilization_metrics,
    }


def build_claims_validation_report_markdown(
    dataset_name: str,
    overview: dict[str, object],
    claims_workflow: dict[str, object],
) -> bytes:
    buffer = StringIO()
    rows = int(overview.get('rows', 0) or 0)
    columns = int(overview.get('analyzed_columns', overview.get('columns', 0)) or 0)
    summary_cards = claims_workflow.get('summary_cards', [])
    validation_table = _safe_df(claims_workflow.get('validation_table'))
    financial_summary = _safe_df(claims_workflow.get('financial_summary'))
    payer_table = _safe_df(claims_workflow.get('payer_utilization'))
    provider_table = _safe_df(claims_workflow.get('provider_utilization'))
    diagnosis_table = _safe_df(claims_workflow.get('diagnosis_utilization'))
    monthly_table = _safe_df(claims_workflow.get('monthly_utilization'))
    flagged_rows = _safe_df(claims_workflow.get('flagged_rows'))

    buffer.write('# Claims Validation & Utilization Engine\n\n')
    buffer.write(f'## Dataset Scope\n\n')
    buffer.write(f'- Dataset: `{dataset_name}`\n')
    buffer.write(f'- Rows in scope: {rows:,}\n')
    buffer.write(f'- Analyzed columns: {columns:,}\n')
    buffer.write(f"- Workflow status: {claims_workflow.get('readiness_label', 'Not available')}\n")
    for card in summary_cards:
        buffer.write(f"- {card.get('label', 'Summary')}: {card.get('value', '')}\n")
    buffer.write('\n')
    if claims_workflow.get('narrative'):
        buffer.write(f"{claims_workflow.get('narrative')}\n\n")

    buffer.write('## Key Validation Findings\n\n')
    if validation_table.empty:
        buffer.write('- No claims validation findings were generated for the current dataset.\n\n')
    else:
        for _, row in validation_table.head(5).iterrows():
            buffer.write(
                f"- {row.get('check', 'Validation check')}: {int(row.get('failed_rows', 0)):,} rows "
                f"({float(row.get('failure_rate', 0.0)):.1%}) | Severity: {row.get('severity', 'Unknown')}\n"
            )
        buffer.write('\n')

    buffer.write('## Major Financial Integrity Findings\n\n')
    if financial_summary.empty:
        buffer.write('- Financial integrity metrics are not available for the current dataset.\n\n')
    else:
        for _, row in financial_summary.head(8).iterrows():
            value = row.get('value')
            if isinstance(value, (int, float)):
                buffer.write(f"- {row.get('metric', 'Metric')}: {float(value):,.2f}\n")
            else:
                buffer.write(f"- {row.get('metric', 'Metric')}: {value}\n")
        buffer.write('\n')

    buffer.write('## Utilization Highlights\n\n')
    if not payer_table.empty:
        top_payer = payer_table.iloc[0]
        buffer.write(f"- Top payer concentration: {top_payer.iloc[0]} with {int(top_payer.get('claim_rows', 0)):,} claim rows.\n")
    if not provider_table.empty:
        top_provider = provider_table.iloc[0]
        buffer.write(f"- Highest observed provider volume: {top_provider.iloc[0]} with {int(top_provider.get('claim_rows', 0)):,} claim rows.\n")
    if not diagnosis_table.empty:
        top_dx = diagnosis_table.iloc[0]
        buffer.write(f"- Most common diagnosis group: {top_dx.iloc[0]} with {int(top_dx.get('claim_rows', 0)):,} claim rows.\n")
    if not monthly_table.empty:
        latest_month = monthly_table.iloc[-1]
        month_label = latest_month.get('service_month', 'Latest month')
        buffer.write(f"- Latest monthly utilization snapshot: {month_label} with {int(latest_month.get('claim_rows', 0)):,} claim rows.\n")
    if payer_table.empty and provider_table.empty and diagnosis_table.empty and monthly_table.empty:
        buffer.write('- Utilization highlights are limited for the current dataset.\n')
    buffer.write('\n')

    buffer.write('## Plain-English Interpretation\n\n')
    if not validation_table.empty:
        top_issue = validation_table.sort_values('failed_rows', ascending=False).iloc[0]
        buffer.write(
            f'This claims file is usable for payer and utilization review, but the highest-priority issue is '
            f'"{top_issue.get("check", "validation review")}" affecting {int(top_issue.get("failed_rows", 0)):,} rows. '
        )
    else:
        buffer.write('This claims file is usable for exploratory payer and utilization review. ')
    if not flagged_rows.empty:
        buffer.write(
            f'The engine also isolated {len(flagged_rows):,} flagged claim rows for follow-up so reviewers can quickly inspect duplicates, amount mismatches, or reversals. '
        )
    if not payer_table.empty:
        buffer.write('The payer and provider summaries make it easier to explain where volume and financial exposure are concentrated.')
    buffer.write('\n')
    return buffer.getvalue().encode('utf-8')


def build_claims_export_csv_bundle(
    dataset_name: str,
    overview: dict[str, object],
    claims_workflow: dict[str, object],
) -> dict[str, bytes]:
    tables = build_claims_export_tables(dataset_name, overview, claims_workflow)
    return {
        'qc_summary.csv': dataframe_to_csv_bytes(tables['qc_summary']),
        'claims_validation_issue_log.csv': dataframe_to_csv_bytes(tables['claims_validation_issue_log']),
        'utilization_metrics.csv': dataframe_to_csv_bytes(tables['utilization_metrics']),
        'claims_validation_report.md': build_claims_validation_report_markdown(dataset_name, overview, claims_workflow),
    }

