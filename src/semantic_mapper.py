from __future__ import annotations

from difflib import SequenceMatcher

import pandas as pd

from src.schema_detection import StructureSummary


CANONICAL_FIELDS = {
    'entity_id': ['entity_id', 'member_id', 'person_id'],
    'patient_id': ['patient_id', 'pat_id', 'patient_identifier'],
    'member_id': ['member_id', 'subscriber_id'],
    'claim_id': ['claim_id', 'claim_number'],
    'encounter_id': ['encounter_id', 'visit_id', 'admission_id'],
    'provider_id': ['provider_id', 'rendering_provider_id'],
    'provider_name': ['provider_name', 'physician', 'doctor_name'],
    'facility': ['facility', 'hospital', 'site', 'location_name'],
    'diagnosis_code': ['diagnosis', 'dx', 'dx_code', 'icd', 'icd_code'],
    'procedure_code': ['procedure', 'px', 'cpt', 'hcpcs', 'procedure_code'],
    'admission_date': ['admission_date', 'admit_date', 'adm_dt'],
    'discharge_date': ['discharge_date', 'disch_dt'],
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
    'bmi': ['bmi', 'body_mass_index'],
    'cholesterol_level': ['cholesterol_level', 'cholesterol', 'ldl', 'hdl'],
    'comorbidities': ['comorbidities', 'comorbidity', 'comorbidity_score', 'chronic_conditions'],
}

HEALTHCARE_CANONICALS = {
    'patient_id', 'member_id', 'claim_id', 'encounter_id', 'provider_id', 'provider_name', 'facility',
    'diagnosis_code', 'procedure_code', 'admission_date', 'discharge_date', 'service_date', 'payer',
    'plan', 'cost_amount', 'paid_amount', 'allowed_amount', 'billed_amount', 'length_of_stay', 'specialty',
    'department', 'smoking_status', 'cancer_stage', 'treatment_type', 'survived', 'bmi',
    'readmission',
    'cholesterol_level', 'comorbidities', 'age', 'gender', 'diagnosis_date', 'end_treatment_date'
}

FIELD_TYPE_HINTS = {
    'entity_id': {'identifier'},
    'patient_id': {'identifier'},
    'member_id': {'identifier'},
    'claim_id': {'identifier'},
    'encounter_id': {'identifier'},
    'provider_id': {'identifier'},
    'provider_name': {'categorical', 'text'},
    'facility': {'categorical', 'text'},
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
    'provider_id': 'Provider identifier used for provider benchmarking and variation review.',
    'provider_name': 'Provider or clinician name used for operational comparisons.',
    'facility': 'Facility or site field used for location-based comparison.',
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


def _name_score(column: str, aliases: list[str]) -> tuple[float, str]:
    best = 0.0
    reason = 'weak name match'
    for alias in aliases:
        ratio = SequenceMatcher(None, column, alias).ratio()
        if column == alias:
            return 1.0, f'exact column-name match: {alias}'
        if alias in column or column in alias:
            ratio = max(ratio, 0.82)
        if ratio > best:
            best = ratio
            reason = f'column-name similarity to {alias}'
    return best, reason


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


def infer_semantic_mapping(data: pd.DataFrame, structure: StructureSummary) -> dict[str, object]:
    detection_lookup = structure.detection_table.set_index('column_name').to_dict('index') if not structure.detection_table.empty else {}
    candidates: list[dict[str, object]] = []

    for column in data.columns:
        inferred_type = str(detection_lookup.get(column, {}).get('inferred_type', 'unknown'))
        for canonical_field, aliases in CANONICAL_FIELDS.items():
            name_score, name_reason = _name_score(column, aliases)
            value_bonus, value_reason = _value_bonus(data[column], inferred_type, canonical_field)
            type_adjustment, type_reason = _type_adjustment(inferred_type, canonical_field)
            score = min(max((name_score * 0.72) + value_bonus + type_adjustment, 0.0), 0.99)
            candidates.append({
                'raw_column': column,
                'canonical_field': canonical_field,
                'score': score,
                'name_reason': name_reason,
                'value_reason': value_reason,
                'type_reason': type_reason,
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
            'evidence': f"{row['name_reason']}; {row['value_reason']}; {row['type_reason']}",
            'used_downstream': used_downstream,
        })

    mapping_table = pd.DataFrame(rows)
    healthcare_hits = [field for field in canonical_map if field in HEALTHCARE_CANONICALS]
    healthcare_readiness = min(len(healthcare_hits) / 8, 1.0)
    semantic_confidence = float(mapping_table['confidence_score'].mean()) if not mapping_table.empty else 0.0
    return {
        'canonical_map': canonical_map,
        'mapping_table': mapping_table,
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
    dictionary['mapping_source'] = 'Auto-detected'
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
    base_columns = ['priority_field', 'issue', 'impact_on_analytics', 'why_it_matters', 'closest_existing_columns', 'suggested_remediation', 'recommended_fix', 'estimated_readiness_improvement', 'modules_unlocked']
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
            'issue': _remediation_issue(field),
            'impact_on_analytics': _impact_summary(modules_text),
            'why_it_matters': FIELD_DESCRIPTIONS.get(field, 'This field would unlock stronger analytics coverage.'),
            'closest_existing_columns': _closest_existing_columns(existing_columns, aliases),
            'suggested_remediation': suggested_remediation,
            'recommended_fix': suggested_remediation,
            'estimated_readiness_improvement': _estimate_readiness_gain(unlock_count, readiness),
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
                'issue': _remediation_issue(field, weak_mapping_column=weak_column),
                'impact_on_analytics': _impact_summary(modules_text),
                'why_it_matters': FIELD_DESCRIPTIONS.get(field, 'This field may already exist, but the app needs a clearer mapping before using it downstream.'),
                'closest_existing_columns': weak_column or 'No close match detected',
                'suggested_remediation': suggested_remediation,
                'recommended_fix': suggested_remediation,
                'estimated_readiness_improvement': _estimate_readiness_gain(unlock_count, readiness),
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
