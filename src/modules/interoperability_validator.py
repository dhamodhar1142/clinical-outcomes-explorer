from __future__ import annotations

from difflib import SequenceMatcher
import re

import pandas as pd

FHIR_RESOURCE_RULES = {
    'Patient': ['patient_id', 'mrn', 'gender', 'birth_date', 'name'],
    'Encounter': ['encounter_id', 'patient_id', 'admission_date', 'discharge_date', 'visit'],
    'Condition': ['diagnosis_code', 'condition_code', 'patient_id', 'onset_date'],
    'Procedure': ['procedure_code', 'procedure_date', 'encounter_id', 'patient_id'],
    'Observation': ['lab_code', 'observation_code', 'result_value', 'result_date'],
    'Medication': ['medication', 'medication_code', 'dose', 'start_date'],
}

FHIR_MAPPING_TARGETS = {
    'patient_id': 'Patient.id',
    'encounter_id': 'Encounter.id',
    'diagnosis_code': 'Condition.code',
    'procedure_code': 'Procedure.code',
    'lab_code': 'Observation.code',
    'observation_code': 'Observation.code',
    'result_value': 'Observation.value',
    'medication': 'MedicationRequest.medicationCodeableConcept',
}

HL7_SEGMENTS = {
    'PID': ['patient', 'mrn', 'dob', 'sex'],
    'PV1': ['visit', 'encounter', 'admit', 'discharge'],
    'OBR': ['order', 'placer', 'filler'],
    'OBX': ['result', 'value', 'observation', 'lab'],
    'MSH': ['message', 'sending', 'receiving'],
}

ICD10_PATTERN = re.compile(r'^[A-TV-Z][0-9][0-9A-Z](?:\.[0-9A-Z]{1,4})?$')
LOINC_PATTERN = re.compile(r'^\d{1,5}-\d$')
SNOMED_PATTERN = re.compile(r'^\d{6,18}$')

EHR_EXPORT_RULES = {
    'Epic-like export': {
        'tokens': ['pat_id', 'patient_id', 'enc_id', 'encounter_id', 'department', 'provider', 'result_flag', 'visit_date'],
        'description': 'Looks like an operational export shaped by common Epic-style encounter, patient, and result fields.',
    },
    'Cerner-like export': {
        'tokens': ['person_id', 'encntr_id', 'encounter_id', 'catalog_cd', 'event_cd', 'result_val', 'loc_nurse_unit'],
        'description': 'Looks like an operational export shaped by common Cerner-style person, encounter, catalog, and event fields.',
    },
}


def _normalize(text: str) -> str:
    return ''.join(ch.lower() for ch in str(text) if ch.isalnum())


def _best_match(columns: list[str], targets: list[str], threshold: float = 0.58) -> tuple[str | None, float]:
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


def detect_fhir_resources(df: pd.DataFrame) -> pd.DataFrame:
    columns = [str(column) for column in df.columns]
    rows: list[dict[str, object]] = []
    for resource, signals in FHIR_RESOURCE_RULES.items():
        matches = []
        for signal in signals:
            match, score = _best_match(columns, [signal], threshold=0.52)
            if match:
                matches.append((match, score))
        confidence = min(sum(score for _, score in matches) / max(len(signals), 1), 0.99)
        if matches:
            rows.append({
                'resource_type': resource,
                'confidence_score': round(confidence, 2),
                'matched_fields': ', '.join(sorted({column for column, _ in matches})),
                'status': 'Detected' if confidence >= 0.40 else 'Possible',
            })
    return pd.DataFrame(rows).sort_values('confidence_score', ascending=False).reset_index(drop=True) if rows else pd.DataFrame(columns=['resource_type', 'confidence_score', 'matched_fields', 'status'])


def detect_hl7_patterns(df: pd.DataFrame) -> pd.DataFrame:
    columns = [str(column) for column in df.columns]
    normalized_columns = {_normalize(column): column for column in columns}
    rows: list[dict[str, object]] = []
    for segment, signals in HL7_SEGMENTS.items():
        explicit_segment = any(_normalize(column).startswith(segment.lower()) for column in columns)
        signal_hits = [signal for signal in signals if any(signal in _normalize(column) for column in columns)]
        confidence = 0.5 if explicit_segment else 0.0
        confidence += min(len(signal_hits) / max(len(signals), 1), 1.0) * 0.45
        if confidence > 0:
            rows.append({
                'segment': segment,
                'confidence_score': round(min(confidence, 0.99), 2),
                'matched_signals': ', '.join(signal_hits) if signal_hits else 'Column naming pattern only',
                'status': 'Detected' if confidence >= 0.45 else 'Possible',
            })
    return pd.DataFrame(rows).sort_values('confidence_score', ascending=False).reset_index(drop=True) if rows else pd.DataFrame(columns=['segment', 'confidence_score', 'matched_signals', 'status'])


def validate_terminology_fields(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for column in df.columns:
        series = df[column].dropna().astype(str).str.strip()
        if series.empty:
            continue
        sample = series.head(100)
        ratios = {
            'ICD-10': float(sample.str.match(ICD10_PATTERN).mean()),
            'LOINC': float(sample.str.match(LOINC_PATTERN).mean()),
            'SNOMED CT': float(sample.str.match(SNOMED_PATTERN).mean()),
        }
        best_label = max(ratios, key=ratios.get)
        best_ratio = ratios[best_label]
        if best_ratio >= 0.35:
            rows.append({
                'column_name': str(column),
                'terminology_type': best_label,
                'match_ratio': round(best_ratio, 2),
                'status': 'Likely coded field' if best_ratio >= 0.65 else 'Possible coded field',
            })
    return pd.DataFrame(rows).sort_values('match_ratio', ascending=False).reset_index(drop=True) if rows else pd.DataFrame(columns=['column_name', 'terminology_type', 'match_ratio', 'status'])


def suggest_fhir_mappings(df: pd.DataFrame) -> pd.DataFrame:
    columns = [str(column) for column in df.columns]
    rows: list[dict[str, object]] = []
    for local_field, target in FHIR_MAPPING_TARGETS.items():
        match, score = _best_match(columns, [local_field], threshold=0.55)
        if match:
            rows.append({
                'local_field_signal': local_field,
                'suggested_source_column': match,
                'reference_model_target': target,
                'confidence_score': round(score, 2),
            })
    return pd.DataFrame(rows).sort_values('confidence_score', ascending=False).reset_index(drop=True) if rows else pd.DataFrame(columns=['local_field_signal', 'suggested_source_column', 'reference_model_target', 'confidence_score'])


def detect_ehr_export_patterns(df: pd.DataFrame) -> pd.DataFrame:
    columns = [str(column).lower() for column in df.columns]
    rows: list[dict[str, object]] = []
    for label, rule in EHR_EXPORT_RULES.items():
        token_hits = [token for token in rule['tokens'] if any(token in column for column in columns)]
        score = round(min(len(token_hits) / max(len(rule['tokens']), 1), 1.0), 2)
        if token_hits:
            rows.append({
                'export_pattern': label,
                'confidence_score': score,
                'matched_signals': ', '.join(token_hits),
                'status': 'Likely' if score >= 0.45 else 'Possible',
                'description': rule['description'],
            })
    return pd.DataFrame(rows).sort_values('confidence_score', ascending=False).reset_index(drop=True) if rows else pd.DataFrame(columns=['export_pattern', 'confidence_score', 'matched_signals', 'status', 'description'])


def compute_interoperability_readiness(df: pd.DataFrame) -> dict[str, object]:
    fhir = detect_fhir_resources(df)
    hl7 = detect_hl7_patterns(df)
    terminology = validate_terminology_fields(df)
    ehr_patterns = detect_ehr_export_patterns(df)
    epic_score = 0.0
    cerner_score = 0.0
    if not ehr_patterns.empty:
        epic_row = ehr_patterns[ehr_patterns['export_pattern'] == 'Epic-like export']
        cerner_row = ehr_patterns[ehr_patterns['export_pattern'] == 'Cerner-like export']
        if not epic_row.empty:
            epic_score = float(epic_row.iloc[0]['confidence_score'])
        if not cerner_row.empty:
            cerner_score = float(cerner_row.iloc[0]['confidence_score'])
    score = round(min((float(fhir['confidence_score'].mean()) if not fhir.empty else 0) * 45 + (float(hl7['confidence_score'].mean()) if not hl7.empty else 0) * 30 + (min(len(terminology), 3) / 3) * 15 + max(epic_score, cerner_score) * 10, 99.0), 1)
    if score >= 70:
        badge = 'Strong Interoperability Readiness'
    elif score >= 40:
        badge = 'Moderate Interoperability Readiness'
    else:
        badge = 'Early Interoperability Readiness'
    return {
        'readiness_score': score,
        'badge_text': badge,
        'fhir_resources': fhir,
        'hl7_patterns': hl7,
        'terminology_validation': terminology,
        'ehr_export_patterns': ehr_patterns,
        'epic_like_score': round(epic_score, 2),
        'cerner_like_score': round(cerner_score, 2),
    }


def generate_interoperability_report(df: pd.DataFrame) -> dict[str, object]:
    readiness = compute_interoperability_readiness(df)
    mapping = suggest_fhir_mappings(df)
    top_resource = readiness['fhir_resources'].iloc[0]['resource_type'] if not readiness['fhir_resources'].empty else 'None detected'
    top_segment = readiness['hl7_patterns'].iloc[0]['segment'] if not readiness['hl7_patterns'].empty else 'None detected'
    summary = pd.DataFrame([
        {'check': 'FHIR resource detection', 'result': str(top_resource)},
        {'check': 'HL7 v2 pattern detection', 'result': str(top_segment)},
        {'check': 'Terminology-aware fields', 'result': str(int(len(readiness['terminology_validation'])))},
        {'check': 'Epic-like export confidence', 'result': str(readiness['epic_like_score'])},
        {'check': 'Cerner-like export confidence', 'result': str(readiness['cerner_like_score'])},
        {'check': 'Likely EHR export patterns', 'result': str(int(len(readiness['ehr_export_patterns'])))},
    ])
    return {
        'available': readiness['readiness_score'] >= 15 or not mapping.empty,
        'readiness_score': readiness['readiness_score'],
        'badge_text': readiness['badge_text'],
        'fhir_resources': readiness['fhir_resources'],
        'hl7_patterns': readiness['hl7_patterns'],
        'terminology_validation': readiness['terminology_validation'],
        'ehr_export_patterns': readiness['ehr_export_patterns'],
        'mapping_suggestions': mapping,
        'validation_report': summary,
        'note': 'FHIR and HL7 support is heuristic and intended for onboarding, export review, and mapping preparation rather than formal interface certification.',
    }
