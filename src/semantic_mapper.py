from __future__ import annotations

from difflib import SequenceMatcher
import re

import pandas as pd

from src.mapping_profiles import infer_dataset_family
from src.schema_detection import StructureSummary


CANONICAL_FIELDS = {
    'entity_id': ['entity_id', 'member_id', 'person_id', 'record_id', 'entity_identifier'],
    'patient_id': ['patient_id', 'pat_id', 'patient_identifier', 'patient_number', 'medical_record', 'mrn'],
    'member_id': ['member_id', 'subscriber_id'],
    'claim_id': ['claim_id', 'claim_number'],
    'encounter_id': ['encounter_id', 'visit_id', 'admission_id', 'encsntr_id', 'visit_number', 'medt_id', 'enc_id', 'visit_identifier'],
    'encounter_status_code': ['encounter_status_code', 'visit_status_code', 'vstat_cd', 'encounter_status_cd', 'status_code'],
    'encounter_status': ['encounter_status', 'visit_status', 'vstat_des', 'encounter_status_description', 'status_description'],
    'encounter_type_code': ['encounter_type_code', 'visit_type_code', 'vtype_cd', 'encounter_type_cd'],
    'encounter_type': ['encounter_type', 'visit_type', 'vtype_des', 'encounter_type_description', 'visit_class'],
    'provider_id': ['provider_id', 'rendering_provider_id', 'npi', 'provider_number', 'attending_provider_id'],
    'provider_name': ['provider_name', 'physician', 'doctor_name'],
    'facility': ['facility', 'hospital', 'site', 'location_name'],
    'room_id': ['room_id', 'rom_id', 'room_identifier', 'bed_id'],
    'diagnosis_code': ['diagnosis', 'diagnosis_code', 'dx', 'dx_code', 'icd', 'icd_code', 'icd10', 'icd_10', 'snomed', 'snomed_code'],
    'procedure_code': ['procedure', 'procedure_code', 'px', 'cpt', 'cpt_code', 'cpt4', 'hcpcs', 'procedure terminology'],
    'admission_date': ['admission_date', 'admit_date', 'adm_dt', 'admission_dt', 'admit_datetime', 'vis_en', 'visit_start', 'encounter_start'],
    'discharge_date': ['discharge_date', 'disch_dt', 'discharge_dt', 'discharge_datetime', 'vis_ex', 'visit_end', 'encounter_end'],
    'service_date': ['service_date', 'date_of_service', 'dos', 'svc_date', 'event_date'],
    'birth_date': ['birth_date', 'date_of_birth', 'dob'],
    'diagnosis_date': ['diagnosis_date', 'diagnosed_on', 'initial_diagnosis_date'],
    'end_treatment_date': ['end_treatment_date', 'treatment_end_date', 'therapy_end_date'],
    'age': ['age', 'age_years'],
    'gender': ['gender', 'sex'],
    'payer': ['payer', 'insurance', 'payor'],
    'plan': ['plan', 'plan_name'],
    'cost_amount': ['cost', 'cost_amount', 'total_cost', 'charge', 'charges', 'hospital_cost'],
    'paid_amount': ['paid', 'paid_amount', 'net_paid'],
    'allowed_amount': ['allowed', 'allowed_amount'],
    'billed_amount': ['billed', 'billed_amount', 'gross_billed'],
    'quantity': ['quantity', 'qty', 'units'],
    'utilization_count': ['count', 'utilization_count', 'visits'],
    'length_of_stay': ['length_of_stay', 'los', 'stay_days', 'days_in_hospital'],
    'specialty': ['specialty'],
    'department': ['department', 'dept', 'service_line', 'unit'],
    'city': ['city'],
    'state': ['state'],
    'zip': ['zip', 'zipcode', 'postal_code'],
    'event_date': ['event_date', 'service_date', 'admission_date'],
    'category': ['category', 'segment', 'product_line', 'diagnosis', 'procedure'],
    'provider': ['provider_name', 'provider_id', 'facility'],
    'location': ['facility', 'city', 'state', 'zip'],
    'smoking_status': ['smoking_status', 'smoker', 'smoking', 'tobacco_use'],
    'cancer_stage': ['cancer_stage', 'stage', 'tumor_stage', 'oncology_stage'],
    'treatment_type': ['treatment_type', 'therapy', 'treatment', 'regimen'],
    'survived': ['survived', 'survival_status', 'alive', 'outcome'],
    'readmission': ['readmission', 'readmitted', 'readmit_flag', 'is_readmission', 'readmission_flag'],
    'bmi': ['bmi', 'body_mass_index', 'bmi_remediated_value', 'bmi_value', 'body_mass_index_value'],
    'cholesterol_level': ['cholesterol_level', 'cholesterol', 'ldl', 'hdl'],
    'comorbidities': ['comorbidities', 'comorbidity', 'comorbidity_score', 'chronic_conditions'],
}

HEALTHCARE_TERMINOLOGY_DICTIONARY = {
    'ICD-10': ['icd10', 'icd_10', 'diagnosis_code', 'dx_code', 'principal_diagnosis', 'secondary_diagnosis'],
    'CPT': ['cpt', 'cpt_code', 'procedure_code', 'hcpcs', 'procedure terminology'],
    'SNOMED': ['snomed', 'snomed_code', 'clinical_concept', 'problem_code'],
}

HEALTHCARE_CANONICALS = {
    'patient_id', 'member_id', 'claim_id', 'encounter_id', 'provider_id', 'provider_name', 'facility',
    'diagnosis_code', 'procedure_code', 'admission_date', 'discharge_date', 'service_date', 'payer',
    'plan', 'cost_amount', 'paid_amount', 'allowed_amount', 'billed_amount', 'length_of_stay', 'specialty',
    'department', 'encounter_status_code', 'encounter_status', 'encounter_type_code', 'encounter_type', 'room_id',
    'smoking_status', 'cancer_stage', 'treatment_type', 'survived', 'bmi',
    'readmission',
    'cholesterol_level', 'comorbidities', 'age', 'gender', 'diagnosis_date', 'end_treatment_date'
}

FIELD_TYPE_HINTS = {
    'entity_id': {'identifier'},
    'patient_id': {'identifier'},
    'member_id': {'identifier'},
    'claim_id': {'identifier'},
    'encounter_id': {'identifier'},
    'encounter_status_code': {'categorical', 'text', 'identifier'},
    'encounter_status': {'categorical', 'text'},
    'encounter_type_code': {'categorical', 'text', 'identifier'},
    'encounter_type': {'categorical', 'text'},
    'provider_id': {'identifier'},
    'provider_name': {'categorical', 'text'},
    'facility': {'categorical', 'text'},
    'room_id': {'identifier', 'categorical', 'text'},
    'diagnosis_code': {'categorical', 'text'},
    'procedure_code': {'categorical', 'text'},
    'admission_date': {'datetime'},
    'discharge_date': {'datetime'},
    'service_date': {'datetime'},
    'birth_date': {'datetime'},
    'diagnosis_date': {'datetime'},
    'end_treatment_date': {'datetime'},
    'age': {'numeric'},
    'gender': {'categorical', 'text'},
    'payer': {'categorical', 'text'},
    'plan': {'categorical', 'text'},
    'cost_amount': {'numeric'},
    'paid_amount': {'numeric'},
    'allowed_amount': {'numeric'},
    'billed_amount': {'numeric'},
    'quantity': {'numeric'},
    'utilization_count': {'numeric'},
    'length_of_stay': {'numeric'},
    'specialty': {'categorical', 'text'},
    'department': {'categorical', 'text'},
    'city': {'categorical', 'text'},
    'state': {'categorical', 'text'},
    'zip': {'categorical', 'text', 'identifier'},
    'event_date': {'datetime'},
    'category': {'categorical', 'text'},
    'provider': {'categorical', 'text', 'identifier'},
    'location': {'categorical', 'text'},
    'smoking_status': {'categorical', 'text'},
    'cancer_stage': {'categorical', 'text'},
    'treatment_type': {'categorical', 'text'},
    'survived': {'boolean', 'categorical', 'text', 'numeric'},
    'readmission': {'boolean', 'categorical', 'text', 'numeric'},
    'bmi': {'numeric'},
    'cholesterol_level': {'numeric'},
    'comorbidities': {'categorical', 'text', 'numeric'},
}

FIELD_DESCRIPTIONS = {
    'entity_id': 'General entity or record identifier used for event-level frequency analysis.',
    'patient_id': 'Patient-level identifier used for cohort, utilization, and outcome analysis.',
    'member_id': 'Member or subscriber identifier used in claims-style datasets.',
    'claim_id': 'Claim-level identifier useful for financial and utilization analysis.',
    'encounter_id': 'Encounter or visit identifier used for event-level healthcare logic.',
    'encounter_status_code': 'Encounter or visit status code used for visit-state monitoring and operational review.',
    'encounter_status': 'Encounter or visit status description used for workflow monitoring and operational segmentation.',
    'encounter_type_code': 'Encounter or visit type code used for utilization and setting segmentation.',
    'encounter_type': 'Encounter or visit type description used for service-class and utilization review.',
    'provider_id': 'Provider identifier used for provider benchmarking and variation review.',
    'provider_name': 'Provider or clinician name used for operational comparisons.',
    'facility': 'Facility or site field used for location-based comparison.',
    'room_id': 'Room, ward, or bed-style identifier used for encounter-location detail.',
    'diagnosis_code': 'Diagnosis field used for clinical segmentation and outcome comparison.',
    'procedure_code': 'Procedure field used for treatment and utilization segmentation.',
    'admission_date': 'Admission-style date used for encounter sequencing and readmission logic.',
    'discharge_date': 'Discharge-style date used for length-of-stay and readmission logic.',
    'service_date': 'Event or service date used for time-based utilization and trend analysis.',
    'birth_date': 'Date of birth field that can support age derivation and demographic review.',
    'diagnosis_date': 'Diagnosis date used for oncology-style time-to-treatment and outcome review.',
    'end_treatment_date': 'Treatment end date used for duration and outcome analysis.',
    'age': 'Age field used for demographic review and risk segmentation.',
    'gender': 'Gender field used for cohort filtering and demographic analysis.',
    'payer': 'Payer field used for segmentation of utilization or spend.',
    'plan': 'Plan or product line field used for enrollment-style segmentation.',
    'cost_amount': 'Primary cost or charge field used for financial benchmarking.',
    'paid_amount': 'Paid amount field used for reimbursement and payment analysis.',
    'allowed_amount': 'Allowed amount field used for payment benchmarking.',
    'billed_amount': 'Billed charge field used for revenue or charge analysis.',
    'quantity': 'Quantity or unit count used for operational volume review.',
    'utilization_count': 'Pre-aggregated utilization count used for activity summaries.',
    'length_of_stay': 'Length-of-stay field used for operational and outcome analysis.',
    'specialty': 'Specialty field used for provider or service-line comparisons.',
    'department': 'Department or service-line field used for operational segmentation.',
    'city': 'City field used for geographic grouping.',
    'state': 'State field used for geographic grouping.',
    'zip': 'ZIP or postal field used for geographic grouping.',
    'event_date': 'Canonical event date used for trends and frequency analysis.',
    'category': 'General category field used for high-level segmentation.',
    'provider': 'Canonical provider grouping used for comparisons and benchmarking.',
    'location': 'Canonical location grouping used for geographic segmentation.',
    'smoking_status': 'Smoking or tobacco-use field used for clinical risk review.',
    'cancer_stage': 'Cancer stage field used for oncology outcome and cohort analysis.',
    'treatment_type': 'Treatment or therapy field used for outcome and cohort comparison.',
    'survived': 'Outcome field used for survival-style analysis.',
    'readmission': 'Readmission indicator used for 30-day readmission review and operational risk analysis.',
    'bmi': 'Body mass index field used for anomaly detection and risk review.',
    'cholesterol_level': 'Cholesterol field used for anomaly detection and risk review.',
    'comorbidities': 'Comorbidity indicator or score used for patient risk segmentation.',
}


def _normalize_token(value: str) -> str:
    return re.sub(r'[^a-z0-9]+', '_', str(value).strip().lower()).strip('_')


def _token_overlap_score(left: str, right: str) -> float:
    left_tokens = {token for token in left.split('_') if token}
    right_tokens = {token for token in right.split('_') if token}
    if not left_tokens or not right_tokens:
        return 0.0
    overlap = len(left_tokens & right_tokens)
    if overlap == 0:
        return 0.0
    return overlap / max(len(left_tokens), len(right_tokens))


def _name_score(column: str, aliases: list[str]) -> tuple[float, str]:
    best = 0.0
    reason = 'weak name match'
    normalized_column = _normalize_token(column)
    for alias in aliases:
        normalized_alias = _normalize_token(alias)
        ratio = SequenceMatcher(None, normalized_column, normalized_alias).ratio()
        if normalized_column == normalized_alias:
            return 1.0, f'exact column-name match: {alias}'
        if normalized_alias in normalized_column or normalized_column in normalized_alias:
            ratio = max(ratio, 0.82)
        ratio = max(ratio, _token_overlap_score(normalized_column, normalized_alias))
        if ratio > best:
            best = ratio
            reason = f'column-name similarity to {alias}'
    return best, reason


def _healthcare_terminology_bonus(column: str, canonical_field: str) -> tuple[float, str]:
    normalized_column = _normalize_token(column)
    for terminology_name, aliases in HEALTHCARE_TERMINOLOGY_DICTIONARY.items():
        alias_score = max((_name_score(normalized_column, [alias])[0] for alias in aliases), default=0.0)
        if alias_score < 0.6:
            continue
        if terminology_name == 'ICD-10' and canonical_field == 'diagnosis_code':
            return 0.16, 'healthcare terminology dictionary suggests an ICD-10 style diagnosis field'
        if terminology_name == 'CPT' and canonical_field == 'procedure_code':
            return 0.16, 'healthcare terminology dictionary suggests a CPT / HCPCS procedure field'
        if terminology_name == 'SNOMED' and canonical_field == 'diagnosis_code':
            return 0.14, 'healthcare terminology dictionary suggests a SNOMED-style diagnosis field'
    if canonical_field == 'provider_id' and any(token in normalized_column for token in ['npi', 'provider_id', 'rendering_provider']):
        return 0.14, 'healthcare dictionary suggests a provider identifier field'
    if canonical_field == 'patient_id' and any(token in normalized_column for token in ['patient_id', 'pat_id', 'mrn', 'medical_record']):
        return 0.14, 'healthcare dictionary suggests a patient identifier field'
    if canonical_field == 'encounter_id' and any(token in normalized_column for token in ['encounter', 'visit_id', 'medt_id', 'enc_id', 'visit_number']):
        return 0.15, 'healthcare dictionary suggests an encounter identifier field'
    if canonical_field == 'admission_date' and any(token in normalized_column for token in ['admission', 'admit']):
        return 0.14, 'healthcare dictionary suggests an admission date field'
    if canonical_field == 'discharge_date' and any(token in normalized_column for token in ['discharge', 'disch']):
        return 0.14, 'healthcare dictionary suggests a discharge date field'
    if canonical_field == 'admission_date' and normalized_column in {'vis_en', 'visit_entry', 'visit_start'}:
        return 0.18, 'healthcare dictionary suggests a visit-start style encounter date'
    if canonical_field == 'discharge_date' and normalized_column in {'vis_ex', 'visit_exit', 'visit_end'}:
        return 0.18, 'healthcare dictionary suggests a visit-end style encounter date'
    if canonical_field == 'encounter_status_code' and any(token in normalized_column for token in ['vstat_cd', 'visit_status_code', 'encounter_status_code']):
        return 0.18, 'healthcare dictionary suggests a visit status code'
    if canonical_field == 'encounter_status' and any(token in normalized_column for token in ['vstat_des', 'visit_status', 'encounter_status']):
        return 0.18, 'healthcare dictionary suggests a visit status description'
    if canonical_field == 'encounter_type_code' and any(token in normalized_column for token in ['vtype_cd', 'visit_type_code', 'encounter_type_code']):
        return 0.18, 'healthcare dictionary suggests a visit type code'
    if canonical_field == 'encounter_type' and any(token in normalized_column for token in ['vtype_des', 'visit_type', 'encounter_type']):
        return 0.18, 'healthcare dictionary suggests a visit type description'
    if canonical_field == 'room_id' and any(token in normalized_column for token in ['rom_id', 'room_id', 'bed_id']):
        return 0.14, 'healthcare dictionary suggests a room or bed identifier'
    return 0.0, 'healthcare terminology dictionary did not add extra support'


def _value_bonus(series: pd.Series, inferred_type: str, canonical_field: str) -> tuple[float, str]:
    values = series.dropna().astype(str).head(300)
    text = values.str.lower()
    if values.empty:
        return 0.0, 'no sample values available'
    if canonical_field in {'admission_date', 'discharge_date', 'service_date', 'birth_date', 'event_date', 'diagnosis_date', 'end_treatment_date'} and inferred_type == 'datetime':
        return 0.18, 'date-like values support a time field'
    if canonical_field == 'gender' and text.isin({'m', 'f', 'male', 'female', 'unknown'}).mean() >= 0.7:
        return 0.2, 'values resemble gender categories'
    if canonical_field == 'smoking_status' and text.isin({'smoker', 'non-smoker', 'never', 'former', 'current', 'yes', 'no'}).mean() >= 0.5:
        return 0.18, 'values resemble smoking-status categories'
    if canonical_field == 'cancer_stage' and text.str.contains(r'^(?:i|ii|iii|iv|stage)', regex=True).mean() >= 0.4:
        return 0.18, 'values resemble stage groupings'
    if canonical_field == 'survived' and text.isin({'1', '0', 'yes', 'no', 'true', 'false', 'alive', 'deceased', 'survived'}).mean() >= 0.6:
        return 0.18, 'values resemble a binary outcome'
    if canonical_field == 'age' and inferred_type == 'numeric':
        numeric = pd.to_numeric(series, errors='coerce').dropna()
        if not numeric.empty and ((numeric >= 0) & (numeric <= 120)).mean() >= 0.9:
            return 0.18, 'numeric values look like ages'
    if canonical_field in {'cost_amount', 'paid_amount', 'allowed_amount', 'billed_amount'} and inferred_type == 'numeric':
        return 0.16, 'distribution looks like a currency or payment field'
    if canonical_field in {'bmi', 'cholesterol_level'} and inferred_type == 'numeric':
        return 0.14, 'numeric distribution fits a clinical measurement field'
    if canonical_field == 'diagnosis_code' and text.str.contains(r'^[a-z]\d', regex=True).mean() >= 0.2:
        return 0.18, 'values resemble ICD-like diagnosis codes'
    if canonical_field == 'procedure_code' and text.str.contains(r'^[a-z0-9]{4,5}$', regex=True).mean() >= 0.2:
        return 0.14, 'values resemble procedure-style codes'
    if canonical_field in {'encounter_status_code', 'encounter_type_code'} and inferred_type in {'categorical', 'identifier', 'text'}:
        if text.str.fullmatch(r'[a-z0-9_\-]{1,12}').mean() >= 0.75:
            return 0.12, 'values resemble short encounter code values'
    if canonical_field in {'encounter_status', 'encounter_type'} and inferred_type in {'categorical', 'text'}:
        if text.str.len().between(3, 40).mean() >= 0.8:
            return 0.1, 'values resemble encounter category labels'
    if canonical_field == 'zip' and text.str.contains(r'^\d{5}$', regex=True).mean() >= 0.6:
        return 0.2, 'values resemble ZIP codes'
    if canonical_field == 'state' and text.str.fullmatch(r'[a-z]{2}').mean() >= 0.6:
        return 0.16, 'values resemble state abbreviations'
    return 0.0, 'value patterns do not strongly support this mapping'


def _type_adjustment(inferred_type: str, canonical_field: str) -> tuple[float, str]:
    expected = FIELD_TYPE_HINTS.get(canonical_field, set())
    if not expected:
        return 0.0, 'no explicit type expectation'
    if inferred_type in expected:
        return 0.12, 'inferred type aligns with the semantic field'
    if inferred_type == 'unknown':
        return 0.0, 'inferred type is unknown'
    return -0.18, 'inferred type conflicts with the semantic field'


def _sanitize_manual_overrides(manual_overrides: dict[str, str] | None, data: pd.DataFrame) -> dict[str, str]:
    if not isinstance(manual_overrides, dict):
        return {}
    valid_fields = set(CANONICAL_FIELDS.keys())
    valid_columns = {str(column) for column in data.columns}
    sanitized: dict[str, str] = {}
    used_columns: set[str] = set()
    for field, column in manual_overrides.items():
        field_name = str(field).strip()
        column_name = str(column).strip()
        if not field_name or not column_name:
            continue
        if field_name not in valid_fields or column_name not in valid_columns or column_name in used_columns:
            continue
        sanitized[field_name] = column_name
        used_columns.add(column_name)
    return sanitized


def _apply_manual_overrides(
    mapping_table: pd.DataFrame,
    canonical_map: dict[str, str],
    manual_overrides: dict[str, str],
) -> tuple[pd.DataFrame, dict[str, str]]:
    if not manual_overrides:
        return mapping_table, canonical_map

    override_columns = set(manual_overrides.values())
    adjusted_table = mapping_table.copy()
    if not adjusted_table.empty:
        adjusted_table = adjusted_table[
            ~adjusted_table['semantic_label'].astype(str).isin(manual_overrides.keys())
        ]
        adjusted_table = adjusted_table[
            ~adjusted_table['original_column'].astype(str).isin(override_columns)
        ]

    adjusted_map = {
        field: column
        for field, column in canonical_map.items()
        if field not in manual_overrides and column not in override_columns
    }
    manual_rows = []
    for field, column in manual_overrides.items():
        adjusted_map[field] = column
        manual_rows.append(
            {
                'original_column': column,
                'semantic_label': field,
                'confidence_label': 'High',
                'confidence_score': 1.0,
                'evidence': 'manual mapping override confirmed by the current review session',
                'used_downstream': True,
                'mapping_source': 'Manual override',
            }
        )
    if manual_rows:
        adjusted_table = pd.concat([adjusted_table, pd.DataFrame(manual_rows)], ignore_index=True, sort=False)
    return adjusted_table, adjusted_map


def infer_semantic_mapping(
    data: pd.DataFrame,
    structure: StructureSummary,
    manual_overrides: dict[str, str] | None = None,
) -> dict[str, object]:
    detection_lookup = structure.detection_table.set_index('column_name').to_dict('index') if not structure.detection_table.empty else {}
    candidates: list[dict[str, object]] = []

    for column in data.columns:
        inferred_type = str(detection_lookup.get(column, {}).get('inferred_type', 'unknown'))
        for canonical_field, aliases in CANONICAL_FIELDS.items():
            name_score, name_reason = _name_score(column, aliases)
            value_bonus, value_reason = _value_bonus(data[column], inferred_type, canonical_field)
            type_adjustment, type_reason = _type_adjustment(inferred_type, canonical_field)
            terminology_bonus, terminology_reason = _healthcare_terminology_bonus(column, canonical_field)
            score = min(max((name_score * 0.68) + value_bonus + type_adjustment + terminology_bonus, 0.0), 0.99)
            candidates.append({
                'raw_column': column,
                'canonical_field': canonical_field,
                'score': score,
                'name_reason': name_reason,
                'value_reason': value_reason,
                'type_reason': type_reason,
                'terminology_reason': terminology_reason,
                'inferred_type': inferred_type,
            })

    candidate_df = pd.DataFrame(candidates).sort_values(['score'], ascending=False)
    chosen_fields: set[str] = set()
    chosen_columns: set[str] = set()
    rows: list[dict[str, object]] = []
    canonical_map: dict[str, str] = {}

    for _, row in candidate_df.iterrows():
        if row['score'] < 0.46:
            break
        field = str(row['canonical_field'])
        column = str(row['raw_column'])
        if field in chosen_fields or column in chosen_columns:
            continue
        chosen_fields.add(field)
        chosen_columns.add(column)
        confidence_label = 'High' if row['score'] >= 0.82 else 'Medium' if row['score'] >= 0.62 else 'Low'
        used_downstream = confidence_label in {'High', 'Medium'}
        if used_downstream:
            canonical_map[field] = column
        rows.append({
            'original_column': column,
            'semantic_label': field,
            'confidence_label': confidence_label,
            'confidence_score': round(float(row['score']), 3),
            'evidence': f"{row['name_reason']}; {row['value_reason']}; {row['type_reason']}; {row['terminology_reason']}",
            'used_downstream': used_downstream,
            'mapping_source': 'Auto-detected',
        })

    mapping_table = pd.DataFrame(rows)
    sanitized_overrides = _sanitize_manual_overrides(manual_overrides, data)
    mapping_table, canonical_map = _apply_manual_overrides(mapping_table, canonical_map, sanitized_overrides)
    mapped_columns = set(mapping_table['original_column'].astype(str).tolist()) if not mapping_table.empty else set()
    suggestion_rows: list[dict[str, object]] = []
    for column in data.columns:
        column_candidates = candidate_df[candidate_df['raw_column'] == column].head(3)
        for rank, (_, candidate) in enumerate(column_candidates.iterrows(), start=1):
            score = round(float(candidate['score']), 3)
            suggestion_rows.append({
                'source_column': str(candidate['raw_column']),
                'suggested_field': str(candidate['canonical_field']),
                'confidence_score': score,
                'confidence_label': 'High' if score >= 0.82 else 'Medium' if score >= 0.62 else 'Low',
                'suggestion_rank': rank,
                'evidence': f"{candidate['name_reason']}; {candidate['value_reason']}; {candidate['type_reason']}; {candidate['terminology_reason']}",
                'auto_apply': rank == 1 and score >= 0.62,
            })
    suggestion_table = pd.DataFrame(suggestion_rows)
    suggestion_only_columns = [str(column) for column in data.columns if str(column) not in mapped_columns]
    strong_mapping_count = len(mapped_columns)
    total_columns = len(data.columns)
    field_coverage_score = strong_mapping_count / max(total_columns, 1)
    healthcare_hits = [field for field in canonical_map if field in HEALTHCARE_CANONICALS]
    encounter_native_fields = {
        'patient_id', 'member_id', 'encounter_id', 'admission_date', 'discharge_date', 'service_date',
        'encounter_status_code', 'encounter_status', 'encounter_type_code', 'encounter_type', 'room_id',
    }
    clinical_native_fields = {
        'diagnosis_code', 'procedure_code', 'length_of_stay', 'department', 'provider_id', 'provider_name', 'facility',
    }
    encounter_score = min(len([field for field in canonical_map if field in encounter_native_fields]) / 5, 1.0)
    clinical_score = min(len([field for field in canonical_map if field in clinical_native_fields]) / 4, 1.0)
    healthcare_readiness = min((encounter_score * 0.65) + (clinical_score * 0.35), 1.0)
    semantic_confidence = float(mapping_table['confidence_score'].mean()) if not mapping_table.empty else 0.0
    dataset_family = infer_dataset_family(data.columns)
    return {
        'canonical_map': canonical_map,
        'mapping_table': mapping_table,
        'suggestion_table': suggestion_table,
        'manual_overrides_applied': sanitized_overrides,
        'dataset_family': dataset_family,
        'column_accounting_summary': {
            'total_source_columns': total_columns,
            'strongly_mapped_columns': strong_mapping_count,
            'suggestion_only_columns': len(suggestion_only_columns),
            'field_coverage_score': round(field_coverage_score, 3),
            'unresolved_columns': suggestion_only_columns[:20],
        },
        'semantic_confidence_score': semantic_confidence,
        'healthcare_field_hits': healthcare_hits,
        'healthcare_readiness_score': healthcare_readiness,
    }


def build_data_dictionary(structure: StructureSummary, semantic: dict[str, object]) -> pd.DataFrame:
    columns = ['source_column_name', 'inferred_internal_role', 'confidence_score', 'confidence_label', 'inferred_data_type', 'mapping_source', 'business_meaning', 'warning']
    detection = structure.detection_table.copy()
    if detection.empty:
        return pd.DataFrame(columns=columns)

    mapping_table = semantic.get('mapping_table', pd.DataFrame()).copy()
    detection = detection.rename(columns={
        'column_name': 'source_column_name',
        'inferred_type': 'inferred_data_type',
        'confidence_score': 'type_confidence_score',
    })

    if mapping_table.empty:
        detection['inferred_internal_role'] = 'Not mapped'
        detection['confidence_score'] = detection['type_confidence_score']
        detection['confidence_label'] = 'Low'
        detection['mapping_source'] = 'Auto-detected'
        detection['business_meaning'] = 'No reliable semantic role was detected yet.'
        detection['warning'] = 'Map or rename this field if you want it used in advanced analysis.'
        return detection[columns]

    dictionary = detection.merge(mapping_table, left_on='source_column_name', right_on='original_column', how='left')
    dictionary = dictionary.rename(columns={'semantic_label': 'inferred_internal_role'})
    dictionary['inferred_internal_role'] = dictionary['inferred_internal_role'].fillna('Not mapped')
    dictionary['confidence_score'] = dictionary['confidence_score'].fillna(dictionary['type_confidence_score'])
    dictionary['confidence_label'] = dictionary['confidence_label'].fillna('Low')
    dictionary['mapping_source'] = dictionary['mapping_source'].fillna('Auto-detected')
    dictionary['business_meaning'] = dictionary['inferred_internal_role'].map(FIELD_DESCRIPTIONS).fillna('No business meaning has been inferred for this field yet.')
    dictionary['warning'] = dictionary['used_downstream'].map({True: '', False: 'Confidence is still weak, so this mapping is not used downstream.'})
    dictionary.loc[dictionary['inferred_internal_role'] == 'Not mapped', 'warning'] = 'This field is currently unmapped and used only in generic profiling.'
    return dictionary[columns]


def _closest_existing_columns(existing_columns: list[str], aliases: list[str], threshold: float = 0.35) -> str:
    scored_columns: list[tuple[str, float]] = []
    for column in existing_columns:
        score = max(SequenceMatcher(None, column, alias).ratio() for alias in aliases)
        if score >= threshold:
            scored_columns.append((column, score))
    return ', '.join(column for column, _ in sorted(scored_columns, key=lambda item: item[1], reverse=True)[:3]) or 'No close match detected'


def _suggest_remediation_action(field: str, canonical_map: dict[str, str], existing_columns: list[str], weak_mapping_column: str | None = None) -> str:
    if weak_mapping_column:
        return f"Review or manually confirm the current '{weak_mapping_column}' mapping so the app can use it downstream with higher confidence."
    if field == 'age' and ('birth_date' in canonical_map or any('birth' in column for column in existing_columns)):
        return 'Derive age from an existing birth date field.'
    if field == 'length_of_stay' and all(candidate in canonical_map for candidate in ['admission_date', 'discharge_date']):
        return 'Derive length of stay from admission and discharge dates.'
    if field == 'event_date' and any(candidate in canonical_map for candidate in ['service_date', 'admission_date', 'diagnosis_date']):
        return 'Use an existing service, admission, or diagnosis date as the primary event date.'
    if field == 'cost_amount' and any(candidate in canonical_map for candidate in ['paid_amount', 'allowed_amount', 'billed_amount']):
        return 'Promote an existing payment or charge field into a canonical cost field for financial analysis.'
    if field == 'survived' and any('outcome' in column or 'status' in column for column in existing_columns):
        return 'Standardize the outcome field into a binary survived / not survived flag.'
    if field == 'end_treatment_date' and 'diagnosis_date' in canonical_map:
        return 'Map the treatment end date so treatment duration can be derived from diagnosis and end-treatment dates.'
    if field == 'diagnosis_date' and 'end_treatment_date' in canonical_map:
        return 'Map the diagnosis date so treatment duration and longitudinal outcome review can be derived cleanly.'
    return 'Add or rename a source field so the app can map this concept directly.'


def _remediation_issue(field: str, weak_mapping_column: str | None = None) -> str:
    field_name = str(field).replace('_', ' ').title()
    if weak_mapping_column:
        return f"{field_name} appears to exist in the dataset, but the current match is still too weak to use in downstream analysis."
    return f"{field_name} is still missing or not mapped strongly enough for advanced analytics."


def _impact_summary(modules_unlocked: str) -> str:
    modules = [item.strip() for item in str(modules_unlocked).split(',') if item.strip()]
    if not modules:
        return 'Improves overall dataset usability for downstream review.'
    if len(modules) == 1:
        return f"Unlocks {modules[0].lower()} immediately."
    if len(modules) == 2:
        return f"Unlocks {modules[0].lower()} and {modules[1].lower()}."
    return f"Unlocks {len(modules)} advanced modules, including {modules[0].lower()} and {modules[1].lower()}."


def _estimate_readiness_gain(unlock_count: int, readiness: dict[str, object]) -> str:
    total_modules = max(len(readiness.get('readiness_table', pd.DataFrame())), 1)
    current_score = float(readiness.get('readiness_score', 0.0) or 0.0)
    estimated_gain = min((unlock_count / total_modules) * 100, max(5.0, 100.0 - (current_score * 100.0)))
    return f'+{estimated_gain:.0f} readiness points (estimated)'


def _projected_readiness_score(unlock_count: int, readiness: dict[str, object]) -> float:
    total_modules = max(len(readiness.get('readiness_table', pd.DataFrame())), 1)
    current_score = float(readiness.get('readiness_score', 0.0) or 0.0)
    estimated_gain = min(unlock_count / total_modules, max(0.05, 1.0 - current_score))
    return round(min(current_score + estimated_gain, 1.0), 2)


def _current_readiness_label(readiness: dict[str, object]) -> str:
    score = float(readiness.get('readiness_score', 0.0) or 0.0)
    if score >= 0.75:
        return f'Strong ({score:.0%})'
    if score >= 0.45:
        return f'Partial ({score:.0%})'
    return f'Early ({score:.0%})'


def _blocker_summary(modules_unlocked: str) -> str:
    modules = [item.strip() for item in str(modules_unlocked).split(',') if item.strip()]
    if not modules:
        return 'General healthcare analytics coverage is still limited.'
    if len(modules) == 1:
        return f"{modules[0]} is the main blocker right now."
    if len(modules) == 2:
        return f"{modules[0]} and {modules[1]} are currently blocked by this gap."
    return f"{len(modules)} modules are waiting on this fix, led by {modules[0]} and {modules[1]}."


def _build_derivation_opportunities(canonical_map: dict[str, str]) -> list[dict[str, object]]:
    opportunities: list[dict[str, object]] = []
    if all(field in canonical_map for field in ['diagnosis_date', 'end_treatment_date']):
        opportunities.append({
            'priority_field': 'treatment_duration_days',
            'issue': 'Treatment duration is not stored explicitly, even though the required dates are present.',
            'impact_on_analytics': 'Strengthens survival, cohort, and intervention review with a clearer treatment-duration measure.',
            'why_it_matters': 'A derived treatment-duration field makes clinical pathways and treatment comparisons easier to interpret and export.',
            'closest_existing_columns': f"{canonical_map['diagnosis_date']}, {canonical_map['end_treatment_date']}",
            'suggested_remediation': 'Derive treatment duration from diagnosis date and end-treatment date.',
            'recommended_fix': 'Create a treatment_duration_days field from diagnosis_date and end_treatment_date for cleaner downstream reporting.',
            'modules_unlocked': 'Survival & Outcome Analysis, Cohort Analysis, Intervention Planner',
            'estimated_readiness_improvement': '+8 readiness points (estimated)',
        })
    if all(field in canonical_map for field in ['admission_date', 'discharge_date']) and 'length_of_stay' not in canonical_map:
        opportunities.append({
            'priority_field': 'length_of_stay',
            'issue': 'Length of stay is not mapped explicitly, even though admission and discharge dates are available.',
            'impact_on_analytics': 'Improves operational benchmarking and cohort review with a direct stay-duration measure.',
            'why_it_matters': 'A direct LOS field makes quality review and export-ready operational summaries more straightforward.',
            'closest_existing_columns': f"{canonical_map['admission_date']}, {canonical_map['discharge_date']}",
            'suggested_remediation': 'Derive length of stay from admission and discharge dates.',
            'recommended_fix': 'Create a length_of_stay field from admission_date and discharge_date for operational analysis.',
            'modules_unlocked': 'Trend Analysis, Cohort Analysis, Operational Review',
            'estimated_readiness_improvement': '+6 readiness points (estimated)',
        })
    return opportunities


def build_data_remediation_assistant(structure: StructureSummary, semantic: dict[str, object], readiness: dict[str, object]) -> pd.DataFrame:
    detection = structure.detection_table.copy()
    base_columns = [
        'priority_field',
        'current_readiness',
        'projected_readiness',
        'blockers',
        'issue',
        'impact',
        'impact_on_analytics',
        'why_it_matters',
        'closest_existing_columns',
        'suggested_remediation',
        'recommended_fix',
        'estimated_readiness_improvement',
        'modules_unlocked_after_remediation',
        'modules_unlocked',
    ]
    if detection.empty:
        return pd.DataFrame(columns=base_columns)

    unavailable = readiness['readiness_table'][readiness['readiness_table']['status'] != 'Available'].copy()
    field_unlocks: dict[str, set[str]] = {}
    for _, row in unavailable.iterrows():
        missing = [item.strip() for item in str(row['missing_prerequisites']).split(',') if item.strip() and item.strip() != '-']
        for field in missing:
            field_unlocks.setdefault(field, set()).add(str(row['analysis_module']))

    suggestions: list[dict[str, object]] = []
    existing_columns = detection['column_name'].astype(str).tolist()
    mapping_table = semantic.get('mapping_table', pd.DataFrame()).copy()
    canonical_map = semantic.get('canonical_map', {})
    for field, modules in field_unlocks.items():
        aliases = CANONICAL_FIELDS.get(field, [field])
        suggested_remediation = _suggest_remediation_action(field, canonical_map, existing_columns)
        modules_text = ', '.join(sorted(modules))
        unlock_count = len(modules)
        suggestions.append({
            'priority_field': field,
            'current_readiness': _current_readiness_label(readiness),
            'projected_readiness': f"{_projected_readiness_score(unlock_count, readiness):.0%}",
            'blockers': _blocker_summary(modules_text),
            'issue': _remediation_issue(field),
            'impact': _impact_summary(modules_text),
            'impact_on_analytics': _impact_summary(modules_text),
            'why_it_matters': FIELD_DESCRIPTIONS.get(field, 'This field would unlock stronger analytics coverage.'),
            'closest_existing_columns': _closest_existing_columns(existing_columns, aliases),
            'suggested_remediation': suggested_remediation,
            'recommended_fix': suggested_remediation,
            'estimated_readiness_improvement': _estimate_readiness_gain(unlock_count, readiness),
            'modules_unlocked_after_remediation': modules_text,
            'modules_unlocked': modules_text,
            'unlock_count': unlock_count,
        })

    if not mapping_table.empty:
        weak_mappings = mapping_table[mapping_table['used_downstream'] == False].copy()
        for _, row in weak_mappings.iterrows():
            field = str(row.get('semantic_label', '')).strip()
            if not field or field in field_unlocks:
                continue
            modules = set()
            for _, readiness_row in unavailable.iterrows():
                missing = [item.strip() for item in str(readiness_row['missing_prerequisites']).split(',') if item.strip() and item.strip() != '-']
                if field in missing:
                    modules.add(str(readiness_row['analysis_module']))
            if not modules:
                modules = {'Improve downstream semantic mapping confidence'}
            weak_column = str(row.get('original_column', '')).strip() or None
            suggested_remediation = _suggest_remediation_action(field, canonical_map, existing_columns, weak_mapping_column=weak_column)
            modules_text = ', '.join(sorted(modules))
            unlock_count = len(modules)
            suggestions.append({
                'priority_field': field,
                'current_readiness': _current_readiness_label(readiness),
                'projected_readiness': f"{_projected_readiness_score(unlock_count, readiness):.0%}",
                'blockers': _blocker_summary(modules_text),
                'issue': _remediation_issue(field, weak_mapping_column=weak_column),
                'impact': _impact_summary(modules_text),
                'impact_on_analytics': _impact_summary(modules_text),
                'why_it_matters': FIELD_DESCRIPTIONS.get(field, 'This field may already exist, but the app needs a clearer mapping before using it downstream.'),
                'closest_existing_columns': weak_column or 'No close match detected',
                'suggested_remediation': suggested_remediation,
                'recommended_fix': suggested_remediation,
                'estimated_readiness_improvement': _estimate_readiness_gain(unlock_count, readiness),
                'modules_unlocked_after_remediation': modules_text,
                'modules_unlocked': modules_text,
                'unlock_count': unlock_count,
            })

    suggestions.extend(_build_derivation_opportunities(canonical_map))

    if not suggestions:
        return pd.DataFrame(columns=base_columns)

    remediation_table = pd.DataFrame(suggestions).sort_values(['unlock_count', 'priority_field'], ascending=[False, True], na_position='last')
    if 'unlock_count' in remediation_table.columns:
        remediation_table = remediation_table.drop(columns=['unlock_count'])
    return remediation_table[base_columns].head(12).reset_index(drop=True)



def build_dataset_improvement_plan(structure: StructureSummary, semantic: dict[str, object], readiness: dict[str, object]) -> pd.DataFrame:
    remediation = build_data_remediation_assistant(structure, semantic, readiness)
    columns = ['step', 'issue', 'impact', 'recommended_fix', 'modules_unlocked']
    if remediation.empty:
        return pd.DataFrame(columns=columns)

    plan = remediation.copy()
    plan.insert(0, 'step', [f"Step {index}" for index in range(1, len(plan) + 1)])
    plan['issue'] = plan['priority_field'].map(lambda value: f"{str(value).replace('_', ' ').title()} is missing or still too weak to use confidently.")
    plan['impact'] = plan['modules_unlocked'].map(_impact_summary)
    plan['recommended_fix'] = plan.apply(
        lambda row: f"{row['suggested_remediation']} Closest current columns: {row['closest_existing_columns']}.",
        axis=1,
    )
    return plan[columns]
