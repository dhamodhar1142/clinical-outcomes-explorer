from __future__ import annotations

import re

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


EXPORT_POLICY_PRESETS = {
    'Internal Review': {
        'description': 'Best for internal analyst or operations review when the team is working inside a controlled environment.',
        'redaction_level': 'Low',
        'recommended_roles': 'Admin, Analyst',
    },
    'HIPAA-style Limited Dataset': {
        'description': 'Best when direct identifiers should be removed before sharing with a broader operational audience.',
        'redaction_level': 'Moderate',
        'recommended_roles': 'Analyst, Researcher',
    },
    'Research-safe Extract': {
        'description': 'Best when the goal is external research review with stronger masking of identifiers and quasi-identifiers.',
        'redaction_level': 'High',
        'recommended_roles': 'Researcher, Viewer',
    },
}


def _normalize(text: str) -> str:
    return ''.join(ch.lower() for ch in str(text) if ch.isalnum())


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
                rows.append({
                    'column_name': column_name,
                    'sensitive_type': field_type.replace('_', ' ').title(),
                    'detection_reason': 'Column naming pattern' if name_hit and not pattern_hit else 'Value pattern' if pattern_hit and not name_hit else 'Column naming pattern and value pattern',
                    'recommended_action': 'Mask or remove before broader sharing',
                })
                break
        if 'diagnosis' in normalized and any(token in ' '.join(series.astype(str).head(5).tolist()).lower() for token in [' mrn ', ' dob ', ' name:']):
            rows.append({
                'column_name': column_name,
                'sensitive_type': 'Diagnosis text with possible identifiers',
                'detection_reason': 'Free-text diagnosis values may contain identifiers',
                'recommended_action': 'Review and redact before external sharing',
            })
    return pd.DataFrame(rows).drop_duplicates().reset_index(drop=True) if rows else pd.DataFrame(columns=['column_name', 'sensitive_type', 'detection_reason', 'recommended_action'])


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
                'original_sample': value,
                'masked_preview': _mask_value(value, str(field['sensitive_type']).lower().replace(' ', '_')),
            })
    return pd.DataFrame(rows)


def compute_hipaa_risk(df: pd.DataFrame, sensitive_fields: pd.DataFrame) -> dict[str, object]:
    direct_identifiers = sensitive_fields[sensitive_fields['sensitive_type'].isin(['Name', 'First Name', 'Last Name', 'Ssn', 'Email', 'Phone', 'Address', 'Medical Record Number'])]
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
    hipaa = compute_hipaa_risk(df, sensitive_fields)
    return {
        'available': True,
        'sensitive_fields': sensitive_fields,
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

    direct_identifier_labels = {'Name', 'First Name', 'Last Name', 'Ssn', 'Email', 'Phone', 'Address', 'Medical Record Number'}
    quasi_identifier_labels = {'Date Of Birth', 'Zip', 'Member Id', 'Insurance Id'}

    rows: list[dict[str, object]] = []

    direct_rows = sensitive_fields[sensitive_fields['sensitive_type'].isin(direct_identifier_labels)]
    rows.append({
        'rule_name': 'Direct identifier suppression',
        'scope': 'HIPAA Safe Harbor',
        'status': 'Needs action' if not direct_rows.empty else 'Ready',
        'impacted_columns': ', '.join(direct_rows['column_name'].astype(str).tolist()) if not direct_rows.empty else 'None',
        'recommended_action': 'Mask or remove direct identifiers before external reporting or wider sharing.' if not direct_rows.empty else 'No direct identifiers detected in the current sample.',
    })

    quasi_rows = sensitive_fields[sensitive_fields['sensitive_type'].isin(quasi_identifier_labels)]
    rows.append({
        'rule_name': 'Quasi-identifier linkage review',
        'scope': 'GDPR / linkage review',
        'status': 'Review' if not quasi_rows.empty else 'Ready',
        'impacted_columns': ', '.join(quasi_rows['column_name'].astype(str).tolist()) if not quasi_rows.empty else 'None',
        'recommended_action': 'Review whether these fields should be masked, generalized, or retained only with controlled access.' if not quasi_rows.empty else 'No major quasi-identifiers were flagged in the current sample.',
    })

    free_text_rows = sensitive_fields[sensitive_fields['sensitive_type'].astype(str).str.contains('Diagnosis text', case=False, na=False)]
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
        }
        for name, config in EXPORT_POLICY_PRESETS.items()
    ])


def evaluate_export_policy(policy_name: str, privacy_review: dict[str, object]) -> dict[str, object]:
    config = EXPORT_POLICY_PRESETS.get(policy_name, EXPORT_POLICY_PRESETS['Internal Review'])
    sensitive_fields = privacy_review.get('sensitive_fields', pd.DataFrame()) if isinstance(privacy_review, dict) else pd.DataFrame()
    direct_identifier_labels = {'Name', 'First Name', 'Last Name', 'Ssn', 'Email', 'Phone', 'Address', 'Medical Record Number'}
    quasi_identifier_labels = {'Date Of Birth', 'Zip', 'Member Id', 'Insurance Id'}
    direct_count = 0
    quasi_count = 0
    if isinstance(sensitive_fields, pd.DataFrame) and not sensitive_fields.empty and 'sensitive_type' in sensitive_fields.columns:
        direct_count = int(sensitive_fields['sensitive_type'].isin(direct_identifier_labels).sum())
        quasi_count = int(sensitive_fields['sensitive_type'].isin(quasi_identifier_labels).sum())
    if config['redaction_level'] == 'Low':
        sharing_readiness = 'Internal use only' if direct_count else 'Ready for controlled internal sharing'
        guidance = 'Use this preset for internal team review when standard RBAC and role-aware report handling are sufficient.'
    elif config['redaction_level'] == 'Moderate':
        sharing_readiness = 'Suitable after direct-identifier removal' if direct_count else 'Ready for limited-dataset style sharing'
        guidance = 'Use this preset when direct identifiers should be masked before broader operational review.'
    else:
        sharing_readiness = 'Best for de-identified research sharing' if (direct_count or quasi_count) else 'Ready for research-safe export'
        guidance = 'Use this preset when you want stronger masking for external review or research-style collaboration.'
    return {
        'policy_name': policy_name,
        'description': config['description'],
        'redaction_level': config['redaction_level'],
        'recommended_roles': config['recommended_roles'],
        'direct_identifier_count': direct_count,
        'quasi_identifier_count': quasi_count,
        'sharing_readiness': sharing_readiness,
        'guidance': guidance,
    }
