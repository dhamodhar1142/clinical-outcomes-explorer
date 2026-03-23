from __future__ import annotations

import json

import pandas as pd


REPORT_MODE_ALIASES = {
    'Analyst Summary': 'Analyst Report',
    'Manager Summary': 'Operational Report',
    'Operations Summary': 'Operational Report',
    'Operational Review': 'Operational Report',
    'Data Readiness Summary': 'Data Readiness Review',
    'Clinical Summary': 'Clinical Report',
    'Clinical Review': 'Clinical Report',
    'Population Health Review': 'Population Health Summary',
}


def _safe_df(value) -> pd.DataFrame:
    return value if isinstance(value, pd.DataFrame) else pd.DataFrame()


def _analyzed_columns(overview: dict[str, object]) -> int:
    return int(overview.get('analyzed_columns', overview.get('columns', 0)))


def _source_columns(overview: dict[str, object]) -> int:
    return int(overview.get('source_columns', _analyzed_columns(overview)))


def dataframe_to_csv_bytes(data: pd.DataFrame) -> bytes:
    return data.to_csv(index=False).encode('utf-8')


def json_bytes(payload: dict[str, object]) -> bytes:
    return json.dumps(payload, indent=2, default=str).encode('utf-8')


def normalize_report_mode(report_mode: str) -> str:
    return REPORT_MODE_ALIASES.get(report_mode, report_mode)


def _combine_report_tables(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for section, table in tables.items():
        if isinstance(table, pd.DataFrame) and not table.empty:
            frame = table.copy()
            frame.insert(0, 'report_section', section)
            frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=['report_section'])
    return pd.concat(frames, ignore_index=True, sort=False)


def _privacy_redaction_note(role: str, privacy_review: dict[str, object]) -> str:
    sensitive_fields = privacy_review.get('sensitive_fields', pd.DataFrame()) if isinstance(privacy_review, dict) else pd.DataFrame()
    if sensitive_fields.empty:
        return ''
    if role in {'Viewer', 'Researcher'}:
        return 'Privacy note: sensitive column names and detailed identifier guidance are partially redacted in this export for the current role.\n\n'
    return 'Privacy note: this export may reference sensitive-field findings. Review the Privacy & Security Review panel before broader sharing.\n\n'


def apply_role_based_redaction(text_bytes: bytes, role: str, privacy_review: dict[str, object]) -> bytes:
    text = text_bytes.decode('utf-8') if isinstance(text_bytes, (bytes, bytearray)) else str(text_bytes)
    sensitive_fields = privacy_review.get('sensitive_fields', pd.DataFrame()) if isinstance(privacy_review, dict) else pd.DataFrame()
    if isinstance(sensitive_fields, pd.DataFrame) and not sensitive_fields.empty and role in {'Viewer', 'Researcher'}:
        for column_name in sensitive_fields['column_name'].astype(str).tolist():
            text = text.replace(column_name, '[sensitive field]')
    note = _privacy_redaction_note(role, privacy_review)
    return (note + text).encode('utf-8')


def apply_export_policy(text_bytes: bytes, policy_name: str, privacy_review: dict[str, object]) -> bytes:
    text = text_bytes.decode('utf-8') if isinstance(text_bytes, (bytes, bytearray)) else str(text_bytes)
    sensitive_fields = privacy_review.get('sensitive_fields', pd.DataFrame()) if isinstance(privacy_review, dict) else pd.DataFrame()
    if not isinstance(sensitive_fields, pd.DataFrame) or sensitive_fields.empty:
        return text.encode('utf-8')

    policy = str(policy_name or 'Internal Review')
    direct_identifier_labels = {'Name', 'First Name', 'Last Name', 'Ssn', 'Email', 'Phone', 'Address', 'Medical Record Number'}
    quasi_identifier_labels = {'Date Of Birth', 'Zip', 'Member Id', 'Insurance Id'}

    fields_to_redact = sensitive_fields.copy()
    if policy == 'HIPAA-style Limited Dataset':
        fields_to_redact = sensitive_fields[sensitive_fields['sensitive_type'].isin(direct_identifier_labels)]
    elif policy == 'Research-safe Extract':
        fields_to_redact = sensitive_fields[sensitive_fields['sensitive_type'].isin(direct_identifier_labels.union(quasi_identifier_labels))]

    if not fields_to_redact.empty and 'column_name' in fields_to_redact.columns:
        for column_name in fields_to_redact['column_name'].astype(str).tolist():
            text = text.replace(column_name, '[protected field]')

    if policy == 'HIPAA-style Limited Dataset':
        prefix = 'Export policy: HIPAA-style Limited Dataset. Direct identifiers are masked in this export.\n\n'
    elif policy == 'Research-safe Extract':
        prefix = 'Export policy: Research-safe Extract. Direct identifiers and major quasi-identifiers are masked in this export.\n\n'
    else:
        prefix = 'Export policy: Internal Review. Role-aware protections remain active for this export.\n\n'
    return (prefix + text).encode('utf-8')
