from __future__ import annotations

import re
from typing import Any

import pandas as pd

SENSITIVE_FIELD_RULES = {
    'name': ['name', 'full_name', 'patient_name'],
    'first_name': ['first_name', 'given_name'],
    'last_name': ['last_name', 'family_name', 'surname'],
    'date_of_birth': ['dob', 'birth_date', 'date_of_birth'],
    'ssn': ['ssn', 'social_security'],
    'email': ['email', 'email_address'],
    'phone': ['phone', 'mobile', 'telephone'],
    'address': ['address', 'street', 'addr'],
    'zip': ['zip', 'postal', 'postcode'],
    'medical_record_number': ['mrn', 'medical_record_number', 'chart_number'],
    'member_id': ['member_id', 'subscriber_id'],
    'insurance_id': ['insurance_id', 'policy_id', 'plan_member_id'],
}

EMAIL_PATTERN = re.compile(r'^[^@\s]+@[^@\s]+\.[^@\s]+$')
PHONE_PATTERN = re.compile(r'^(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}$')
SSN_PATTERN = re.compile(r'^\d{3}-?\d{2}-?\d{4}$')
ZIP_PATTERN = re.compile(r'^\d{5}(?:-\d{4})?$')

REDACTION_LEVEL_OPTIONS = ['Low', 'Medium', 'High']
WORKSPACE_EXPORT_ACCESS_OPTIONS = ['Owner only', 'Editors and owners', 'All workspace roles']
WORKSPACE_EXPORT_ROLE_OPTIONS = ['Owner', 'Editor', 'Viewer']

EXPORT_POLICY_PRESETS = {
    'Internal Review': {
        'description': 'Best for internal analyst or operations review when the team is working inside a controlled environment.',
        'redaction_level': 'Low',
        'recommended_roles': 'Admin, Analyst',
        'workspace_export_access': 'Editors and owners',
        'watermark_sensitive_exports': False,
    },
    'HIPAA-style Limited Dataset': {
        'description': 'Best when direct identifiers should be removed before sharing with a broader operational audience.',
        'redaction_level': 'Medium',
        'recommended_roles': 'Analyst, Researcher',
        'workspace_export_access': 'Editors and owners',
        'watermark_sensitive_exports': True,
    },
    'Research-safe Extract': {
        'description': 'Best when the goal is external research review with stronger masking of identifiers and quasi-identifiers.',
        'redaction_level': 'High',
        'recommended_roles': 'Researcher, Viewer',
        'workspace_export_access': 'Owner only',
        'watermark_sensitive_exports': True,
    },
}

DIRECT_IDENTIFIER_LABELS = {'Name', 'First Name', 'Last Name', 'Ssn', 'Email', 'Phone', 'Address'}
QUASI_IDENTIFIER_LABELS = {'Date Of Birth', 'Zip', 'Member Id', 'Insurance Id'}
PHI_LABELS = {'Medical Record Number', 'Diagnosis Text With Possible Identifiers'}


def _normalize(text: str) -> str:
    return ''.join(ch.lower() for ch in str(text) if ch.isalnum())


def _safe_df(value: Any) -> pd.DataFrame:
    return value if isinstance(value, pd.DataFrame) else pd.DataFrame()


def _normalize_redaction_level(value: str | None) -> str:
    level = str(value or '').strip().lower()
    if level == 'moderate':
        level = 'medium'
    if level == 'high':
        return 'High'
    if level == 'medium':
        return 'Medium'
    return 'Low'


def _normalize_workspace_export_access(value: str | None) -> str:
    access = str(value or '').strip().lower()
    if access in {'owner only', 'owner'}:
        return 'Owner only'
    if access in {'all workspace roles', 'all'}:
        return 'All workspace roles'
    return 'Editors and owners'


def _workspace_export_role(workspace_identity: dict[str, Any] | None) -> str:
    identity = workspace_identity or {}
    role = str(identity.get('role', 'viewer')).strip().lower()
    if role == 'owner':
        return 'Owner'
    if role in {'admin', 'analyst'}:
        return 'Editor'
    return 'Viewer'


def _workspace_export_allowed(
    workspace_identity: dict[str, Any] | None,
    access_level: str,
    *,
    allow_guest_demo: bool = True,
) -> bool:
    identity = workspace_identity or {}
    if allow_guest_demo and str(identity.get('auth_mode', 'guest')) == 'guest':
        return True
    workspace_role = _workspace_export_role(identity)
    normalized_access = _normalize_workspace_export_access(access_level)
    if normalized_access == 'All workspace roles':
        return True
    if normalized_access == 'Editors and owners':
        return workspace_role in {'Owner', 'Editor'}
    return workspace_role == 'Owner'


def _classification_for_sensitive_type(sensitive_type: str) -> str:
    normalized = str(sensitive_type or '').strip().title()
    if normalized in DIRECT_IDENTIFIER_LABELS:
        return 'PII'
    if normalized in QUASI_IDENTIFIER_LABELS or normalized in PHI_LABELS:
        return 'PHI'
    if 'Diagnosis Text' in normalized:
        return 'PHI'
    return 'Sensitive'


def _mask_value(value: object, field_type: str) -> str:
    text = str(value)
    if not text or text.lower() == 'nan':
        return ''
    if field_type in {'name', 'first_name', 'last_name'}:
        parts = text.split()
        return ' '.join(part[:1] + '*' * max(len(part) - 1, 1) for part in parts)
    if field_type == 'email':
        if '@' not in text:
            return text[:1] + '***'
        local, domain = text.split('@', 1)
        return local[:1] + '***@' + domain
    if field_type == 'phone':
        digits = ''.join(ch for ch in text if ch.isdigit())
        return '***-***-' + digits[-4:] if len(digits) >= 4 else '***'
    if field_type == 'ssn':
        digits = ''.join(ch for ch in text if ch.isdigit())
        return '***-**-' + digits[-4:] if len(digits) >= 4 else '***-**-****'
    if field_type in {'medical_record_number', 'member_id', 'insurance_id'}:
        return '*' * max(len(text) - 4, 0) + text[-4:]
    if field_type == 'address':
        return text[:3] + '***' if len(text) > 3 else '***'
    if field_type == 'zip':
        return text[:2] + '***' if len(text) >= 2 else '***'
    if field_type == 'date_of_birth':
        return '****-**-' + text[-2:] if len(text) >= 2 else '****-**-**'
    return text[:1] + '***'


def detect_sensitive_fields(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for column in df.columns:
        column_name = str(column)
        normalized = _normalize(column_name)
        series = df[column].dropna().astype(str).head(100)
        for field_type, aliases in SENSITIVE_FIELD_RULES.items():
            name_hit = any(alias in normalized for alias in aliases)
            pattern_hit = False
            if not series.empty:
                if field_type == 'email':
                    pattern_hit = float(series.str.match(EMAIL_PATTERN).mean()) >= 0.4
                elif field_type == 'phone':
                    pattern_hit = float(series.str.match(PHONE_PATTERN).mean()) >= 0.4
                elif field_type == 'ssn':
                    pattern_hit = float(series.str.match(SSN_PATTERN).mean()) >= 0.4
                elif field_type == 'zip':
                    pattern_hit = float(series.str.match(ZIP_PATTERN).mean()) >= 0.4
            if name_hit or pattern_hit:
                sensitive_type = field_type.replace('_', ' ').title()
                rows.append({
                    'column_name': column_name,
                    'sensitive_type': sensitive_type,
                    'classification': _classification_for_sensitive_type(sensitive_type),
                    'detection_reason': 'Column naming pattern' if name_hit and not pattern_hit else 'Value pattern' if pattern_hit and not name_hit else 'Column naming pattern and value pattern',
                    'recommended_action': 'Mask or remove before broader sharing',
                })
                break
        if 'diagnosis' in normalized and any(token in ' '.join(series.astype(str).head(5).tolist()).lower() for token in [' mrn ', ' dob ', ' name:']):
            rows.append({
                'column_name': column_name,
                'sensitive_type': 'Diagnosis Text With Possible Identifiers',
                'classification': 'PHI',
                'detection_reason': 'Free-text diagnosis values may contain identifiers',
                'recommended_action': 'Review and redact before external sharing',
            })
    return pd.DataFrame(rows).drop_duplicates().reset_index(drop=True) if rows else pd.DataFrame(columns=['column_name', 'sensitive_type', 'classification', 'detection_reason', 'recommended_action'])


def build_data_classification_table(df: pd.DataFrame, sensitive_fields: pd.DataFrame) -> pd.DataFrame:
    sensitive = _safe_df(sensitive_fields)
    sensitive_lookup: dict[str, dict[str, Any]] = {}
    if not sensitive.empty and {'column_name', 'classification', 'sensitive_type'}.issubset(sensitive.columns):
        sensitive_lookup = sensitive.set_index('column_name')[['classification', 'sensitive_type', 'recommended_action']].to_dict('index')

    rows: list[dict[str, object]] = []
    for column in df.columns:
        column_name = str(column)
        detected = sensitive_lookup.get(column_name)
        if detected:
            classification = str(detected.get('classification', 'Sensitive'))
            sensitive_type = str(detected.get('sensitive_type', 'Sensitive field'))
            recommended_action = str(detected.get('recommended_action', 'Review before sharing'))
        else:
            classification = 'Public'
            sensitive_type = 'No sensitive signal detected'
            recommended_action = 'No extra redaction is required for standard internal review.'
        rows.append({
            'column_name': column_name,
            'classification': classification,
            'sensitive_type': sensitive_type,
            'recommended_action': recommended_action,
        })
    return pd.DataFrame(rows)


def build_data_classification_summary(classification_table: pd.DataFrame) -> dict[str, Any]:
    safe_table = _safe_df(classification_table)
    if safe_table.empty or 'classification' not in safe_table.columns:
        counts = {'Public': 0, 'PII': 0, 'PHI': 0}
    else:
        counts = safe_table['classification'].astype(str).value_counts().to_dict()
    return {
        'public_count': int(counts.get('Public', 0)),
        'pii_count': int(counts.get('PII', 0)),
        'phi_count': int(counts.get('PHI', 0)),
        'sensitive_data_present': int(counts.get('PII', 0)) + int(counts.get('PHI', 0)) > 0,
    }


def generate_deid_preview(df: pd.DataFrame, sensitive_fields: pd.DataFrame, max_rows: int = 8) -> pd.DataFrame:
    if sensitive_fields.empty:
        return pd.DataFrame()
    rows = []
    for _, field in sensitive_fields.head(6).iterrows():
        column = field['column_name']
        if column not in df.columns:
            continue
        original_values = df[column].dropna().astype(str).head(max_rows)
        for value in original_values[:3]:
            rows.append({
                'column_name': column,
                'sensitive_type': field['sensitive_type'],
                'classification': field.get('classification', _classification_for_sensitive_type(str(field['sensitive_type']))),
                'original_sample': value,
                'masked_preview': _mask_value(value, str(field['sensitive_type']).lower().replace(' ', '_')),
            })
    return pd.DataFrame(rows)


def compute_hipaa_risk(df: pd.DataFrame, sensitive_fields: pd.DataFrame) -> dict[str, object]:
    direct_identifiers = sensitive_fields[sensitive_fields['sensitive_type'].isin(DIRECT_IDENTIFIER_LABELS)]
    risk_score = min(len(direct_identifiers) * 12 + len(sensitive_fields) * 4, 100)
    if risk_score >= 60:
        risk_level = 'High'
    elif risk_score >= 30:
        risk_level = 'Moderate'
    else:
        risk_level = 'Low'
    return {
        'risk_score': risk_score,
        'risk_level': risk_level,
        'direct_identifier_count': int(len(direct_identifiers)),
        'safe_harbor_ready': len(direct_identifiers) == 0,
        'summary': pd.DataFrame([
            {'metric': 'Sensitive columns detected', 'value': str(int(len(sensitive_fields)))},
            {'metric': 'Direct identifiers detected', 'value': str(int(len(direct_identifiers)))},
            {'metric': 'Safe Harbor indicator', 'value': 'Ready' if len(direct_identifiers) == 0 else 'Needs de-identification'},
        ]),
    }


def compute_gdpr_impact(df: pd.DataFrame, sensitive_fields: pd.DataFrame) -> pd.DataFrame:
    if sensitive_fields.empty:
        return pd.DataFrame(columns=['column_name', 'gdpr_action', 'impact_note'])
    rows = []
    for _, row in sensitive_fields.iterrows():
        sensitive_type = str(row['sensitive_type']).lower()
        if any(token in sensitive_type for token in ['name', 'email', 'phone', 'address', 'ssn']):
            action = 'Delete or strongly mask'
            note = 'Likely direct identifier subject to erasure or redaction workflows.'
        elif any(token in sensitive_type for token in ['medical', 'member', 'insurance', 'date of birth']):
            action = 'Mask and review linkage'
            note = 'Likely identifier or quasi-identifier that may require linkage review.'
        else:
            action = 'Review before sharing'
            note = 'Potentially sensitive health context that may require localized policy review.'
        rows.append({'column_name': row['column_name'], 'gdpr_action': action, 'impact_note': note})
    return pd.DataFrame(rows)


def export_encryption_option() -> dict[str, object]:
    return {
        'available': False,
        'status': 'Configuration-ready placeholder',
        'note': 'Export encryption is not enforced in this single-user demo runtime yet. Use this hook to connect managed key storage or protected exports in a deployed environment.',
    }


def run_privacy_security_review(df: pd.DataFrame) -> dict[str, object]:
    sensitive_fields = detect_sensitive_fields(df)
    classification_table = build_data_classification_table(df, sensitive_fields)
    classification_summary = build_data_classification_summary(classification_table)
    hipaa = compute_hipaa_risk(df, sensitive_fields)
    return {
        'available': True,
        'sensitive_fields': sensitive_fields,
        'classification_table': classification_table,
        'classification_summary': classification_summary,
        'deidentification_preview': generate_deid_preview(df, sensitive_fields),
        'hipaa': hipaa,
        'gdpr_impact': compute_gdpr_impact(df, sensitive_fields),
        'export_protection': export_encryption_option(),
        'privacy_rule_pack': build_privacy_rule_pack(sensitive_fields),
        'note': 'Privacy and security checks are onboarding-oriented readiness indicators. Review them before broader sharing or regulated use.',
    }


def build_privacy_rule_pack(sensitive_fields: pd.DataFrame) -> pd.DataFrame:
    if sensitive_fields.empty:
        return pd.DataFrame(columns=['rule_name', 'scope', 'status', 'impacted_columns', 'recommended_action'])

    rows: list[dict[str, object]] = []

    direct_rows = sensitive_fields[sensitive_fields['sensitive_type'].isin(DIRECT_IDENTIFIER_LABELS)]
    rows.append({
        'rule_name': 'Direct identifier suppression',
        'scope': 'HIPAA Safe Harbor',
        'status': 'Needs action' if not direct_rows.empty else 'Ready',
        'impacted_columns': ', '.join(direct_rows['column_name'].astype(str).tolist()) if not direct_rows.empty else 'None',
        'recommended_action': 'Mask or remove direct identifiers before external reporting or wider sharing.' if not direct_rows.empty else 'No direct identifiers detected in the current sample.',
    })

    quasi_rows = sensitive_fields[sensitive_fields['sensitive_type'].isin(QUASI_IDENTIFIER_LABELS)]
    rows.append({
        'rule_name': 'Quasi-identifier linkage review',
        'scope': 'GDPR / linkage review',
        'status': 'Review' if not quasi_rows.empty else 'Ready',
        'impacted_columns': ', '.join(quasi_rows['column_name'].astype(str).tolist()) if not quasi_rows.empty else 'None',
        'recommended_action': 'Review whether these fields should be masked, generalized, or retained only with controlled access.' if not quasi_rows.empty else 'No major quasi-identifiers were flagged in the current sample.',
    })

    free_text_rows = sensitive_fields[sensitive_fields['sensitive_type'].astype(str).str.contains('Diagnosis Text', case=False, na=False)]
    rows.append({
        'rule_name': 'Clinical free-text review',
        'scope': 'Narrative field protection',
        'status': 'Review' if not free_text_rows.empty else 'Ready',
        'impacted_columns': ', '.join(free_text_rows['column_name'].astype(str).tolist()) if not free_text_rows.empty else 'None',
        'recommended_action': 'Review free-text fields for embedded identifiers before export or downstream sharing.' if not free_text_rows.empty else 'No free-text identifier risk was detected in the diagnosis-style fields reviewed.',
    })

    return pd.DataFrame(rows)


def get_export_policy_presets() -> pd.DataFrame:
    return pd.DataFrame([
        {
            'policy_name': name,
            'description': config['description'],
            'redaction_level': config['redaction_level'],
            'recommended_roles': config['recommended_roles'],
            'workspace_export_access': config['workspace_export_access'],
            'watermark_sensitive_exports': 'Yes' if bool(config.get('watermark_sensitive_exports')) else 'No',
        }
        for name, config in EXPORT_POLICY_PRESETS.items()
    ])


def evaluate_export_policy(
    policy_name: str,
    privacy_review: dict[str, object],
    workspace_identity: dict[str, object] | None = None,
    governance_config: dict[str, object] | None = None,
) -> dict[str, object]:
    config = EXPORT_POLICY_PRESETS.get(policy_name, EXPORT_POLICY_PRESETS['Internal Review'])
    overrides = governance_config or {}
    sensitive_fields = privacy_review.get('sensitive_fields', pd.DataFrame()) if isinstance(privacy_review, dict) else pd.DataFrame()
    classification_summary = dict(privacy_review.get('classification_summary', {})) if isinstance(privacy_review, dict) else {}
    direct_count = 0
    quasi_count = 0
    if isinstance(sensitive_fields, pd.DataFrame) and not sensitive_fields.empty and 'sensitive_type' in sensitive_fields.columns:
        direct_count = int(sensitive_fields['sensitive_type'].isin(DIRECT_IDENTIFIER_LABELS).sum())
        quasi_count = int(sensitive_fields['sensitive_type'].isin(QUASI_IDENTIFIER_LABELS).sum())

    effective_redaction_level = _normalize_redaction_level(
        overrides.get('redaction_level', config.get('redaction_level', 'Low'))
    )
    workspace_export_access = _normalize_workspace_export_access(
        overrides.get('workspace_export_access', config.get('workspace_export_access', 'Editors and owners'))
    )
    watermark_sensitive_exports = bool(
        overrides.get('watermark_sensitive_exports', config.get('watermark_sensitive_exports', False))
    )
    workspace_export_role = _workspace_export_role(workspace_identity)
    workspace_export_allowed = _workspace_export_allowed(workspace_identity, workspace_export_access)

    if effective_redaction_level == 'Low':
        sharing_readiness = 'Internal use only' if direct_count else 'Ready for controlled internal sharing'
        guidance = 'Use this preset for internal team review when standard RBAC and role-aware report handling are sufficient.'
    elif effective_redaction_level == 'Medium':
        sharing_readiness = 'Suitable after direct-identifier removal' if direct_count else 'Ready for limited-dataset style sharing'
        guidance = 'Use this preset when direct identifiers should be masked before broader operational review.'
    else:
        sharing_readiness = 'Best for de-identified research sharing' if (direct_count or quasi_count) else 'Ready for research-safe export'
        guidance = 'Use this preset when you want stronger masking for external review or research-style collaboration.'

    if not workspace_export_allowed:
        guidance += f' The current workspace export role ({workspace_export_role}) does not satisfy the active workspace export policy ({workspace_export_access}).'

    watermark_label = build_export_watermark_label(
        dataset_name='Current dataset',
        policy_name=policy_name,
        redaction_level=effective_redaction_level,
        classification_summary=classification_summary,
        workspace_identity=workspace_identity,
        watermark_sensitive_exports=watermark_sensitive_exports,
    )
    return {
        'policy_name': policy_name,
        'description': config['description'],
        'redaction_level': effective_redaction_level,
        'recommended_roles': config['recommended_roles'],
        'direct_identifier_count': direct_count,
        'quasi_identifier_count': quasi_count,
        'sharing_readiness': sharing_readiness,
        'guidance': guidance,
        'workspace_export_access': workspace_export_access,
        'workspace_export_role': workspace_export_role,
        'workspace_export_allowed': workspace_export_allowed,
        'watermark_sensitive_exports': watermark_sensitive_exports,
        'watermark_label': watermark_label,
        'classification_summary': classification_summary,
    }


def build_export_watermark_label(
    *,
    dataset_name: str,
    policy_name: str,
    redaction_level: str,
    classification_summary: dict[str, Any] | None,
    workspace_identity: dict[str, Any] | None,
    watermark_sensitive_exports: bool,
) -> str:
    summary = classification_summary or {}
    sensitive_present = bool(summary.get('pii_count', 0) or summary.get('phi_count', 0))
    workspace_name = str((workspace_identity or {}).get('workspace_name', 'Guest Demo Workspace'))
    export_role = _workspace_export_role(workspace_identity)
    if not watermark_sensitive_exports or not sensitive_present:
        return f'Internal export | {dataset_name} | {policy_name} | {workspace_name} | {export_role}'
    return (
        f'SENSITIVE EXPORT | {dataset_name} | Policy {policy_name} | Redaction {redaction_level} | '
        f'Workspace {workspace_name} | Access {export_role}'
    )


def apply_export_watermark(text_bytes: bytes, watermark_label: str) -> bytes:
    text = text_bytes.decode('utf-8') if isinstance(text_bytes, (bytes, bytearray)) else str(text_bytes)
    return f'Watermark: {watermark_label}\n\n{text}'.encode('utf-8')


def apply_dataframe_redaction(
    table: pd.DataFrame,
    *,
    privacy_review: dict[str, Any],
    redaction_level: str,
    role: str,
) -> pd.DataFrame:
    safe_table = _safe_df(table).copy()
    if safe_table.empty:
        return safe_table
    sensitive_fields = _safe_df(privacy_review.get('sensitive_fields'))
    if sensitive_fields.empty or 'column_name' not in sensitive_fields.columns:
        return safe_table

    direct_fields = sensitive_fields[sensitive_fields['sensitive_type'].isin(DIRECT_IDENTIFIER_LABELS)]['column_name'].astype(str).tolist()
    quasi_fields = sensitive_fields[sensitive_fields['sensitive_type'].isin(QUASI_IDENTIFIER_LABELS)]['column_name'].astype(str).tolist()
    level = _normalize_redaction_level(redaction_level)
    fields_to_redact = list(direct_fields)
    if level == 'High':
        fields_to_redact.extend(quasi_fields)

    renamed_columns = []
    for column in safe_table.columns:
        column_name = str(column)
        if column_name in fields_to_redact or (role in {'Viewer', 'Researcher'} and column_name in sensitive_fields['column_name'].astype(str).tolist()):
            renamed_columns.append('[protected field]')
        else:
            renamed_columns.append(column_name)
    safe_table.columns = renamed_columns

    if not fields_to_redact and role not in {'Viewer', 'Researcher'}:
        return safe_table

    all_sensitive_names = sensitive_fields['column_name'].astype(str).tolist()

    def _clean_value(value: Any) -> Any:
        text = str(value)
        if role in {'Viewer', 'Researcher'}:
            for name in all_sensitive_names:
                text = text.replace(name, '[sensitive field]')
        for name in fields_to_redact:
            text = text.replace(name, '[protected field]')
        return text

    object_columns = safe_table.select_dtypes(include=['object']).columns
    for column in object_columns:
        safe_table[column] = safe_table[column].map(_clean_value)
    return safe_table


def build_export_governance_summary(
    policy_name: str,
    privacy_review: dict[str, object],
    workspace_identity: dict[str, object] | None = None,
    governance_config: dict[str, object] | None = None,
) -> dict[str, object]:
    workspace_identity = workspace_identity or {}
    privacy_review = privacy_review or {}
    policy_eval = evaluate_export_policy(policy_name, privacy_review, workspace_identity, governance_config)
    classification_table = _safe_df(privacy_review.get('classification_table'))
    classification_summary = policy_eval.get('classification_summary', {})
    watermark_label = str(policy_eval.get('watermark_label', ''))

    summary_cards = [
        {'label': 'Export Policy', 'value': str(policy_eval.get('policy_name', policy_name))},
        {'label': 'Sharing Readiness', 'value': str(policy_eval.get('sharing_readiness', 'Internal only'))},
        {'label': 'Redaction Level', 'value': str(policy_eval.get('redaction_level', 'Low'))},
        {'label': 'Workspace Export Access', 'value': str(policy_eval.get('workspace_export_access', 'Editors and owners'))},
        {'label': 'Workspace Export Role', 'value': str(policy_eval.get('workspace_export_role', 'Viewer'))},
        {'label': 'Watermarking', 'value': 'Sensitive exports watermarked' if bool(policy_eval.get('watermark_sensitive_exports')) else 'Standard watermark'},
    ]
    controls_table = pd.DataFrame(
        [
            {
                'control_area': 'Policy guidance',
                'status': str(policy_eval.get('policy_name', policy_name)),
                'detail': str(policy_eval.get('guidance', 'Export policy guidance is not available yet.')),
            },
            {
                'control_area': 'Direct identifier check',
                'status': str(policy_eval.get('direct_identifier_count', 0)),
                'detail': 'Direct identifiers should be reviewed before broader sharing.',
            },
            {
                'control_area': 'Workspace-level access control',
                'status': 'Allowed' if bool(policy_eval.get('workspace_export_allowed')) else 'Restricted',
                'detail': (
                    f"Active export access is '{policy_eval.get('workspace_export_access', 'Editors and owners')}' "
                    f"and the current workspace role resolves to {policy_eval.get('workspace_export_role', 'Viewer')}."
                ),
            },
            {
                'control_area': 'Data classification',
                'status': f"PII {int(classification_summary.get('pii_count', 0))} | PHI {int(classification_summary.get('phi_count', 0))}",
                'detail': 'The export path now classifies fields as PII, PHI, or Public so policy posture and watermarking can respond to real dataset sensitivity.',
            },
            {
                'control_area': 'Export watermark',
                'status': 'Enabled' if bool(policy_eval.get('watermark_sensitive_exports')) else 'Standard',
                'detail': watermark_label or 'Watermark details will appear here when export governance is active.',
            },
            {
                'control_area': 'Workspace ownership boundary',
                'status': str(workspace_identity.get('owner_label', 'Guest session')),
                'detail': 'Export actions are being reviewed in the context of the active workspace owner and role.',
            },
        ]
    )
    notes = [
        'Export governance is a product trust layer that combines privacy review, role-aware sharing, workspace ownership context, and data classification.',
        'All export and sharing actions continue to flow through the audit trail so pilot reviews can inspect what was generated or downloaded.',
    ]
    if not classification_table.empty:
        notes.append(
            f"Detected classifications: {int(classification_summary.get('public_count', 0))} public, "
            f"{int(classification_summary.get('pii_count', 0))} PII, and {int(classification_summary.get('phi_count', 0))} PHI fields."
        )
    return {
        'summary_cards': summary_cards,
        'controls_table': controls_table,
        'classification_table': classification_table,
        'notes': notes,
        'policy_evaluation': policy_eval,
    }
