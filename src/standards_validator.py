from __future__ import annotations

from difflib import SequenceMatcher

import pandas as pd

from src.modules.cdisc_validator import generate_cdisc_report
from src.modules.interoperability_validator import generate_interoperability_report
from src.schema_detection import StructureSummary


STANDARDS = {
    'FHIR-style Resources': {
        'description': 'Looks for resource-style healthcare fields that resemble flattened FHIR extracts.',
        'required_raw': ['resourceType', 'id', 'subject', 'code', 'status'],
        'required_canonical': ['patient_id', 'service_date', 'diagnosis_code'],
        'optional_raw': ['encounter', 'performer', 'effectiveDateTime', 'issued', 'category'],
        'optional_canonical': ['provider_id', 'provider_name', 'facility', 'procedure_code'],
    },
    'HL7-style Messages': {
        'description': 'Looks for tabular extracts built from HL7 message segments and operational message fields.',
        'required_raw': ['msh', 'pid', 'pv1'],
        'required_canonical': ['patient_id', 'admission_date'],
        'optional_raw': ['obr', 'obx', 'orc', 'dg1', 'evn'],
        'optional_canonical': ['diagnosis_code', 'procedure_code', 'provider_id'],
    },
    'CDISC SDTM': {
        'description': 'Looks for study-oriented SDTM structures with subject, domain, and collected observation fields.',
        'required_raw': ['STUDYID', 'USUBJID', 'DOMAIN'],
        'required_canonical': ['patient_id', 'service_date'],
        'optional_raw': ['VISIT', 'VISITNUM', 'AESTDTC', 'CMTRT', 'AEDECOD', 'SEX', 'AGE'],
        'optional_canonical': ['gender', 'age', 'diagnosis_date', 'treatment_type'],
    },
    'CDISC ADaM': {
        'description': 'Looks for analysis-ready ADaM structures with parameterized outcomes and subject-level analysis variables.',
        'required_raw': ['STUDYID', 'USUBJID', 'PARAM', 'PARAMCD', 'AVAL'],
        'required_canonical': ['patient_id', 'survived'],
        'optional_raw': ['ADT', 'ADY', 'AVISIT', 'TRTA', 'CHG', 'BASE'],
        'optional_canonical': ['service_date', 'treatment_type', 'age', 'gender'],
    },
}


TERMINOLOGY_OVERRIDE_FIELDS = {
    'ICD-10': 'Confirm the field that best represents ICD-10 style diagnosis coding.',
    'LOINC': 'Confirm the field that best represents LOINC-style laboratory or observation coding.',
    'SNOMED CT': 'Confirm the field that best represents SNOMED CT style clinical concept coding.',
}


STANDARDS_OVERRIDE_FIELDS = {
    'CDISC': {
        'STUDYID': 'Study identifier used across SDTM and ADaM domains.',
        'USUBJID': 'Unique subject identifier used for subject-level trial analysis.',
        'SUBJID': 'Alternative subject identifier when USUBJID is not available.',
        'VISIT': 'Visit label used for clinical trial visit structure.',
        'VISITNUM': 'Visit number used for ordered visit structure.',
        'DOMAIN': 'Clinical trial domain code such as DM, AE, LB, or VS.',
    },
    'Interoperability': {
        'Patient.id': 'FHIR Patient identifier target.',
        'Encounter.id': 'FHIR Encounter identifier target.',
        'Condition.code': 'FHIR Condition code target for diagnosis fields.',
        'Procedure.code': 'FHIR Procedure code target for procedure fields.',
        'Observation.code': 'FHIR Observation code target for labs or measurements.',
        'Observation.value': 'FHIR Observation value target for result values.',
    },
}


def _normalize_column_name(name: str) -> str:
    return ''.join(char.lower() for char in str(name) if char.isalnum())



def _closest_columns(columns: list[str], target_names: list[str], limit: int = 3) -> str:
    scored: list[tuple[str, float]] = []
    for column in columns:
        normalized = _normalize_column_name(column)
        score = max(SequenceMatcher(None, normalized, _normalize_column_name(target)).ratio() for target in target_names)
        if score >= 0.35:
            scored.append((column, score))
    ranked = [column for column, _ in sorted(scored, key=lambda item: item[1], reverse=True)[:limit]]
    return ', '.join(ranked) if ranked else 'No close match detected'



def _match_raw_fields(raw_columns: list[str], required_names: list[str]) -> tuple[list[str], list[str]]:
    normalized_columns = {_normalize_column_name(column): column for column in raw_columns}
    matched: list[str] = []
    missing: list[str] = []
    for name in required_names:
        normalized = _normalize_column_name(name)
        if normalized in normalized_columns:
            matched.append(name)
        else:
            missing.append(name)
    return matched, missing



def _match_canonical_fields(canonical_map: dict[str, str], field_names: list[str]) -> tuple[list[str], list[str]]:
    matched = [field for field in field_names if field in canonical_map]
    missing = [field for field in field_names if field not in canonical_map]
    return matched, missing



def _recommendation_for_standard(standard_name: str, missing_raw: list[str], missing_canonical: list[str], all_columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for field in missing_raw:
        rows.append({
            'missing_field': field,
            'field_type': 'Raw standard field',
            'closest_existing_columns': _closest_columns(all_columns, [field]),
            'recommended_mapping': f'Add or rename a source field so the dataset more clearly resembles {standard_name}.',
        })
    for field in missing_canonical:
        rows.append({
            'missing_field': field,
            'field_type': 'Canonical healthcare role',
            'closest_existing_columns': _closest_columns(all_columns, [field]),
            'recommended_mapping': f'Map or derive {field.replace("_", " ")} so this dataset can better support {standard_name}.',
        })
    return pd.DataFrame(rows)



def _base_standards_summary(data: pd.DataFrame, semantic: dict[str, object]) -> pd.DataFrame:
    raw_columns = [str(column) for column in data.columns]
    canonical_map = semantic.get('canonical_map', {})
    results: list[dict[str, object]] = []
    for standard_name, rule in STANDARDS.items():
        matched_raw_required, missing_raw_required = _match_raw_fields(raw_columns, rule['required_raw'])
        matched_canonical_required, missing_canonical_required = _match_canonical_fields(canonical_map, rule['required_canonical'])
        matched_raw_optional, _ = _match_raw_fields(raw_columns, rule['optional_raw'])
        matched_canonical_optional, _ = _match_canonical_fields(canonical_map, rule['optional_canonical'])
        required_total = max(len(rule['required_raw']) + len(rule['required_canonical']), 1)
        optional_total = max(len(rule['optional_raw']) + len(rule['optional_canonical']), 1)
        required_hits = len(matched_raw_required) + len(matched_canonical_required)
        optional_hits = len(matched_raw_optional) + len(matched_canonical_optional)
        confidence = min((required_hits / required_total) * 0.75 + (optional_hits / optional_total) * 0.25, 0.99)
        if required_hits == 0 and optional_hits == 0:
            confidence = 0.0
        results.append({
            'standard': standard_name,
            'confidence_label': 'High' if confidence >= 0.75 else 'Medium' if confidence >= 0.45 else 'Low',
            'compliance_confidence': float(confidence),
            'matched_required_fields': required_hits,
            'missing_required_fields': len(missing_raw_required) + len(missing_canonical_required),
            'matched_optional_fields': optional_hits,
            'description': rule['description'],
            'missing_raw_fields': missing_raw_required,
            'missing_canonical_fields': missing_canonical_required,
        })
    return pd.DataFrame(results).sort_values('compliance_confidence', ascending=False).reset_index(drop=True)



def validate_healthcare_standards(data: pd.DataFrame, structure: StructureSummary, semantic: dict[str, object]) -> dict[str, object]:
    raw_columns = [str(column) for column in data.columns]
    summary_table = _base_standards_summary(data, semantic)
    cdisc_report = generate_cdisc_report(data)
    interoperability_report = generate_interoperability_report(data)

    top_standard = summary_table.iloc[0] if not summary_table.empty else None
    top_standard_name = str(top_standard['standard']) if top_standard is not None else 'No dominant standard detected'
    top_standard_confidence = float(top_standard['compliance_confidence']) if top_standard is not None else 0.0
    top_standard_label = str(top_standard['confidence_label']) if top_standard is not None else 'Low'
    missing_required = int(top_standard['missing_required_fields']) if top_standard is not None else 0

    recommendations = pd.DataFrame()
    if top_standard is not None and top_standard_confidence >= 0.20:
        recommendations = _recommendation_for_standard(
            top_standard_name,
            list(top_standard['missing_raw_fields']),
            list(top_standard['missing_canonical_fields']),
            raw_columns,
        )

    combined_readiness = round(max(
        top_standard_confidence * 100,
        float(cdisc_report.get('readiness_score', 0.0)),
        float(interoperability_report.get('readiness_score', 0.0)),
    ), 1)

    available = bool((top_standard is not None and top_standard_confidence >= 0.20) or cdisc_report.get('available') or interoperability_report.get('available'))
    if not available:
        return {
            'available': False,
            'reason': 'The dataset does not currently resemble a supported healthcare exchange, trial, or interoperability structure strongly enough for a useful standards review.',
            'summary_table': summary_table[['standard', 'confidence_label', 'compliance_confidence', 'matched_required_fields', 'missing_required_fields', 'matched_optional_fields', 'description']] if not summary_table.empty else pd.DataFrame(),
            'recommendations': pd.DataFrame(),
            'cdisc_report': cdisc_report,
            'interoperability_report': interoperability_report,
        }

    if combined_readiness >= 75:
        badge = 'Strong Healthcare Standards Readiness'
    elif combined_readiness >= 45:
        badge = 'Moderate Healthcare Standards Readiness'
    else:
        badge = 'Early Healthcare Standards Readiness'

    return {
        'available': True,
        'detected_standard': top_standard_name,
        'compliance_confidence': top_standard_confidence,
        'confidence_label': top_standard_label,
        'missing_required_fields': missing_required,
        'summary_table': summary_table[['standard', 'confidence_label', 'compliance_confidence', 'matched_required_fields', 'missing_required_fields', 'matched_optional_fields', 'description']],
        'recommendations': recommendations,
        'combined_readiness_score': combined_readiness,
        'badge_text': badge,
        'cdisc_report': cdisc_report,
        'interoperability_report': interoperability_report,
        'note': 'The validator is heuristic and intended for onboarding, mapping preparation, and healthcare standards readiness review rather than formal certification.',
    }


def _build_effective_mapping_table(standards_validation: dict[str, object], overrides: dict[str, str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    cdisc_report = standards_validation.get('cdisc_report', {})
    interoperability_report = standards_validation.get('interoperability_report', {})

    cdisc_suggestions = cdisc_report.get('mapping_suggestions', pd.DataFrame())
    if isinstance(cdisc_suggestions, pd.DataFrame) and not cdisc_suggestions.empty:
        for _, row in cdisc_suggestions.iterrows():
            target_field = str(row.get('cdisc_field', ''))
            rows.append({
                'mapping_group': 'CDISC',
                'target_field': target_field,
                'source_column': overrides.get(target_field, row.get('suggested_source_column', '')),
                'mapping_source': 'Manual override' if overrides.get(target_field) else 'Auto-detected',
                'help_text': STANDARDS_OVERRIDE_FIELDS['CDISC'].get(target_field, 'CDISC-aligned field.'),
            })

    interop_suggestions = interoperability_report.get('mapping_suggestions', pd.DataFrame())
    if isinstance(interop_suggestions, pd.DataFrame) and not interop_suggestions.empty:
        for _, row in interop_suggestions.iterrows():
            target_field = str(row.get('reference_model_target', ''))
            rows.append({
                'mapping_group': 'Interoperability',
                'target_field': target_field,
                'source_column': overrides.get(target_field, row.get('suggested_source_column', '')),
                'mapping_source': 'Manual override' if overrides.get(target_field) else 'Auto-detected',
                'help_text': STANDARDS_OVERRIDE_FIELDS['Interoperability'].get(target_field, 'FHIR or HL7-aligned reference target.'),
            })

    for group_name, fields in STANDARDS_OVERRIDE_FIELDS.items():
        for target_field, help_text in fields.items():
            if target_field in overrides and not any(row['target_field'] == target_field for row in rows):
                rows.append({
                    'mapping_group': group_name,
                    'target_field': target_field,
                    'source_column': overrides[target_field],
                    'mapping_source': 'Manual override',
                    'help_text': help_text,
                })

    if not rows:
        return pd.DataFrame(columns=['mapping_group', 'target_field', 'source_column', 'mapping_source', 'help_text'])
    return pd.DataFrame(rows).sort_values(['mapping_group', 'target_field']).reset_index(drop=True)


def build_standards_override_catalog(standards_validation: dict[str, object]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    effective = _build_effective_mapping_table(standards_validation, {})
    existing_targets = set(effective['target_field']) if not effective.empty else set()
    for group_name, fields in STANDARDS_OVERRIDE_FIELDS.items():
        for target_field, help_text in fields.items():
            if target_field in existing_targets:
                default_source = str(effective.loc[effective['target_field'] == target_field, 'source_column'].iloc[0])
            else:
                default_source = ''
            rows.append({
                'mapping_group': group_name,
                'target_field': target_field,
                'default_source': default_source,
                'help_text': help_text,
            })
    return pd.DataFrame(rows)


def build_terminology_override_catalog(standards_validation: dict[str, object]) -> pd.DataFrame:
    interop = standards_validation.get('interoperability_report', {})
    terminology = interop.get('terminology_validation', pd.DataFrame()) if isinstance(interop, dict) else pd.DataFrame()
    rows: list[dict[str, object]] = []
    for terminology_type, help_text in TERMINOLOGY_OVERRIDE_FIELDS.items():
        default_source = ''
        default_status = 'Not detected'
        if isinstance(terminology, pd.DataFrame) and not terminology.empty and 'terminology_type' in terminology.columns:
            matched = terminology[terminology['terminology_type'] == terminology_type]
            if not matched.empty:
                default_source = str(matched.iloc[0].get('column_name', ''))
                default_status = str(matched.iloc[0].get('status', 'Detected'))
        rows.append({
            'terminology_type': terminology_type,
            'default_source': default_source,
            'status': default_status,
            'help_text': help_text,
        })
    return pd.DataFrame(rows)


def apply_standards_mapping_overrides(standards_validation: dict[str, object], overrides: dict[str, str], terminology_overrides: dict[str, str] | None = None) -> dict[str, object]:
    if not standards_validation.get('available'):
        return standards_validation
    cleaned_overrides = {key: value for key, value in overrides.items() if value}
    cleaned_terminology = {key: value for key, value in (terminology_overrides or {}).items() if value}
    updated = dict(standards_validation)
    updated['override_catalog'] = build_standards_override_catalog(standards_validation)
    updated['terminology_override_catalog'] = build_terminology_override_catalog(standards_validation)
    updated['manual_overrides'] = cleaned_overrides
    updated['terminology_manual_overrides'] = cleaned_terminology
    updated['effective_mappings'] = _build_effective_mapping_table(standards_validation, cleaned_overrides)

    cdisc_report = dict(updated.get('cdisc_report', {}))
    interoperability_report = dict(updated.get('interoperability_report', {}))
    if cdisc_report:
        cdisc_effective = updated['effective_mappings']
        if not cdisc_effective.empty:
            cdisc_report['effective_mappings'] = cdisc_effective[cdisc_effective['mapping_group'] == 'CDISC'].drop(columns=['mapping_group'])
    if interoperability_report:
        interop_effective = updated['effective_mappings']
        if not interop_effective.empty:
            interoperability_report['effective_mappings'] = interop_effective[interop_effective['mapping_group'] == 'Interoperability'].drop(columns=['mapping_group'])
        terminology = interoperability_report.get('terminology_validation', pd.DataFrame())
        effective_rows: list[dict[str, object]] = []
        if isinstance(terminology, pd.DataFrame) and not terminology.empty:
            for _, row in terminology.iterrows():
                term_type = str(row.get('terminology_type', ''))
                source_column = cleaned_terminology.get(term_type, str(row.get('column_name', '')))
                effective_rows.append({
                    'terminology_type': term_type,
                    'source_column': source_column,
                    'match_ratio': row.get('match_ratio', None),
                    'status': 'Manually confirmed' if term_type in cleaned_terminology else row.get('status', 'Likely coded field'),
                })
        for term_type, source_column in cleaned_terminology.items():
            if not any(str(row.get('terminology_type')) == term_type for row in effective_rows):
                effective_rows.append({
                    'terminology_type': term_type,
                    'source_column': source_column,
                    'match_ratio': None,
                    'status': 'Manually confirmed',
                })
        interoperability_report['effective_terminology'] = pd.DataFrame(effective_rows) if effective_rows else pd.DataFrame(columns=['terminology_type', 'source_column', 'match_ratio', 'status'])
    updated['cdisc_report'] = cdisc_report
    updated['interoperability_report'] = interoperability_report
    return updated
