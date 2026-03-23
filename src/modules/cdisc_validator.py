from __future__ import annotations

from difflib import SequenceMatcher
import re

import pandas as pd

SDTM_REQUIRED_FIELDS = ['STUDYID', 'USUBJID', 'SUBJID', 'VISIT', 'VISITNUM', 'DOMAIN']
SDTM_DOMAIN_TEMPLATES = {
    'DM': ['STUDYID', 'USUBJID', 'SUBJID', 'DOMAIN', 'SEX', 'AGE'],
    'AE': ['STUDYID', 'USUBJID', 'SUBJID', 'DOMAIN', 'AEDECOD', 'AESTDTC'],
    'LB': ['STUDYID', 'USUBJID', 'SUBJID', 'DOMAIN', 'LBTEST', 'LBORRES'],
    'VS': ['STUDYID', 'USUBJID', 'SUBJID', 'DOMAIN', 'VSTEST', 'VSORRES'],
}
ADAM_SIGNALS = ['PARAM', 'PARAMCD', 'AVAL', 'AVISIT', 'ADT', 'ADY', 'CHG', 'BASE', 'TRTA', 'TRTP']

CDISC_SYNONYMS = {
    'STUDYID': ['studyid', 'study_id', 'study', 'protocol', 'protocol_id'],
    'USUBJID': ['usubjid', 'subject_id', 'subjectid', 'patient_id', 'patientid', 'participant_id'],
    'SUBJID': ['subjid', 'subject', 'subject_number', 'screening_id'],
    'VISIT': ['visit', 'visit_name', 'visit_label', 'visit_date'],
    'VISITNUM': ['visitnum', 'visit_number', 'visit_no', 'visit_order'],
    'DOMAIN': ['domain', 'dataset_domain', 'domain_code'],
}


def _normalize(text: str) -> str:
    return ''.join(ch.lower() for ch in str(text) if ch.isalnum())


def _best_match(columns: list[str], targets: list[str], threshold: float = 0.62) -> tuple[str | None, float]:
    best_column = None
    best_score = 0.0
    for column in columns:
        normalized_column = _normalize(column)
        score = max(SequenceMatcher(None, normalized_column, _normalize(target)).ratio() for target in targets)
        if score > best_score:
            best_column = column
            best_score = score
    if best_score >= threshold:
        return best_column, best_score
    return None, best_score


def suggest_cdisc_mappings(df: pd.DataFrame) -> pd.DataFrame:
    columns = [str(column) for column in df.columns]
    rows: list[dict[str, object]] = []
    for cdisc_field, aliases in CDISC_SYNONYMS.items():
        match, score = _best_match(columns, aliases)
        if match:
            rows.append({
                'cdisc_field': cdisc_field,
                'suggested_source_column': match,
                'confidence_score': round(score, 2),
                'reason': 'Column name similarity to CDISC variable naming',
            })
    return pd.DataFrame(rows).sort_values('confidence_score', ascending=False).reset_index(drop=True) if rows else pd.DataFrame(columns=['cdisc_field', 'suggested_source_column', 'confidence_score', 'reason'])


def detect_sdtm_dataset(df: pd.DataFrame) -> dict[str, object]:
    normalized_columns = {_normalize(column): str(column) for column in df.columns}
    required_hits = []
    for field in ['STUDYID', 'DOMAIN']:
        if _normalize(field) in normalized_columns:
            required_hits.append(field)
    subject_present = any(_normalize(field) in normalized_columns for field in ['USUBJID', 'SUBJID'])
    visit_present = any(_normalize(field) in normalized_columns for field in ['VISIT', 'VISITNUM'])
    domain_values = set(df[normalized_columns[_normalize('DOMAIN')]].dropna().astype(str).str.upper().unique().tolist()) if _normalize('DOMAIN') in normalized_columns else set()
    known_domains = sorted(domain_values.intersection(SDTM_DOMAIN_TEMPLATES.keys()))
    naming_hits = sum(1 for field in SDTM_REQUIRED_FIELDS if _normalize(field) in normalized_columns)
    confidence = min((len(required_hits) / 2) * 0.35 + (0.2 if subject_present else 0) + (0.15 if visit_present else 0) + min(naming_hits / len(SDTM_REQUIRED_FIELDS), 1.0) * 0.15 + (0.15 if known_domains else 0), 0.99)
    return {
        'detected': confidence >= 0.35,
        'confidence_score': round(confidence, 2),
        'known_domains': known_domains,
        'subject_present': subject_present,
        'visit_present': visit_present,
    }


def validate_sdtm_structure(df: pd.DataFrame) -> dict[str, object]:
    normalized_columns = {_normalize(column): str(column) for column in df.columns}
    passed_checks: list[str] = []
    missing_fields: list[str] = []
    required_identifier_coverage = 0

    if _normalize('STUDYID') in normalized_columns:
        passed_checks.append('STUDYID is present.')
        required_identifier_coverage += 1
    else:
        missing_fields.append('STUDYID')

    if any(_normalize(field) in normalized_columns for field in ['USUBJID', 'SUBJID']):
        passed_checks.append('Subject-level identifier is present (USUBJID or SUBJID).')
        required_identifier_coverage += 1
    else:
        missing_fields.extend(['USUBJID / SUBJID'])

    if any(_normalize(field) in normalized_columns for field in ['VISIT', 'VISITNUM']):
        passed_checks.append('Visit structure is present (VISIT or VISITNUM).')
    else:
        missing_fields.append('VISIT / VISITNUM')

    if _normalize('DOMAIN') in normalized_columns:
        passed_checks.append('DOMAIN is present.')
    else:
        missing_fields.append('DOMAIN')

    domain_template_rows = []
    domain_column = normalized_columns.get(_normalize('DOMAIN'))
    detected_domains: list[str] = []
    if domain_column and domain_column in df.columns:
        detected_domains = sorted(df[domain_column].dropna().astype(str).str.upper().unique().tolist())
        for domain in detected_domains[:10]:
            template = SDTM_DOMAIN_TEMPLATES.get(domain)
            if not template:
                continue
            missing_template_fields = [field for field in template if _normalize(field) not in normalized_columns]
            domain_template_rows.append({
                'domain': domain,
                'template_fields_expected': ', '.join(template),
                'missing_fields': ', '.join(missing_template_fields) if missing_template_fields else 'None',
            })

    return {
        'passed_checks': passed_checks,
        'missing_required_fields': missing_fields,
        'domain_templates': pd.DataFrame(domain_template_rows),
        'required_identifier_coverage': required_identifier_coverage / 2 if 2 else 0,
        'visit_structure_readiness': 1.0 if any(_normalize(field) in normalized_columns for field in ['VISIT', 'VISITNUM']) else 0.0,
        'domain_consistency': 1.0 if detected_domains else 0.0,
        'naming_similarity': min(sum(1 for field in SDTM_REQUIRED_FIELDS if _normalize(field) in normalized_columns) / len(SDTM_REQUIRED_FIELDS), 1.0),
        'detected_domains': detected_domains,
    }


def detect_adam_structure(df: pd.DataFrame) -> dict[str, object]:
    normalized_columns = {_normalize(column): str(column) for column in df.columns}
    matched_signals = [signal for signal in ADAM_SIGNALS if _normalize(signal) in normalized_columns]
    confidence = min(len(matched_signals) / max(len(ADAM_SIGNALS), 1) * 1.6, 0.99)
    likely_adsl = all(_normalize(field) in normalized_columns for field in ['STUDYID', 'USUBJID']) and any(_normalize(field) in normalized_columns for field in ['TRTA', 'TRTP', 'SEX', 'AGE'])
    return {
        'detected': confidence >= 0.30 or likely_adsl,
        'confidence_score': round(max(confidence, 0.55 if likely_adsl else confidence), 2),
        'matched_signals': matched_signals,
        'likely_structure': 'ADSL-like' if likely_adsl else 'ADaM-like analysis dataset',
    }


def compute_cdisc_readiness(df: pd.DataFrame) -> dict[str, object]:
    sdtm_detection = detect_sdtm_dataset(df)
    sdtm_validation = validate_sdtm_structure(df)
    adam_detection = detect_adam_structure(df)
    subcomponents = {
        'required_identifier_coverage': round(sdtm_validation['required_identifier_coverage'] * 100, 1),
        'visit_structure_readiness': round(sdtm_validation['visit_structure_readiness'] * 100, 1),
        'domain_consistency': round(sdtm_validation['domain_consistency'] * 100, 1),
        'naming_similarity': round(sdtm_validation['naming_similarity'] * 100, 1),
    }
    score = round(
        subcomponents['required_identifier_coverage'] * 0.35
        + subcomponents['visit_structure_readiness'] * 0.20
        + subcomponents['domain_consistency'] * 0.20
        + subcomponents['naming_similarity'] * 0.15
        + adam_detection['confidence_score'] * 10 * 0.10,
        1,
    )
    if score >= 75:
        badge = 'Strong CDISC Readiness'
    elif score >= 45:
        badge = 'Moderate CDISC Readiness'
    else:
        badge = 'Early CDISC Readiness'
    likely_dataset_type = 'CDISC SDTM-like' if sdtm_detection['confidence_score'] >= adam_detection['confidence_score'] else 'CDISC ADaM-like'
    return {
        'readiness_score': score,
        'badge_text': badge,
        'subcomponents': subcomponents,
        'likely_dataset_type': likely_dataset_type,
        'sdtm_detection': sdtm_detection,
        'sdtm_validation': sdtm_validation,
        'adam_detection': adam_detection,
    }


def generate_cdisc_report(df: pd.DataFrame) -> dict[str, object]:
    readiness = compute_cdisc_readiness(df)
    mappings = suggest_cdisc_mappings(df)
    sdtm_validation = readiness['sdtm_validation']
    report_rows = [
        {'check': 'CDISC readiness badge', 'result': readiness['badge_text']},
        {'check': 'Likely dataset type', 'result': readiness['likely_dataset_type']},
        {'check': 'SDTM required fields missing', 'result': ', '.join(sdtm_validation['missing_required_fields']) or 'None'},
        {'check': 'Detected SDTM domains', 'result': ', '.join(sdtm_validation['detected_domains']) or 'None detected'},
        {'check': 'ADaM structure signals', 'result': ', '.join(readiness['adam_detection']['matched_signals']) or 'None detected'},
    ]
    return {
        'available': readiness['readiness_score'] >= 15 or not mappings.empty,
        'readiness_score': readiness['readiness_score'],
        'badge_text': readiness['badge_text'],
        'likely_dataset_type': readiness['likely_dataset_type'],
        'subcomponents': pd.DataFrame([{'component': key.replace('_', ' ').title(), 'score': value} for key, value in readiness['subcomponents'].items()]),
        'sdtm_detection': readiness['sdtm_detection'],
        'adam_detection': readiness['adam_detection'],
        'validation_report': pd.DataFrame(report_rows).assign(result=lambda frame: frame['result'].astype(str)),
        'mapping_suggestions': mappings,
        'domain_templates': sdtm_validation['domain_templates'],
        'passed_checks': sdtm_validation['passed_checks'],
        'missing_required_fields': sdtm_validation['missing_required_fields'],
        'note': 'CDISC support is heuristic and demo-oriented. Use it as a readiness and mapping aid rather than a formal submission validator.',
    }
