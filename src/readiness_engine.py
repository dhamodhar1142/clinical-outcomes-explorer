from __future__ import annotations

import pandas as pd


ANALYSIS_RULES = {
    'Encounter Flow Analysis': {
        'required_all_groups': [['patient_id', 'member_id', 'entity_id'], ['encounter_id', 'claim_id', 'service_date', 'admission_date'], ['encounter_status', 'encounter_status_code', 'encounter_type', 'encounter_type_code']],
        'description': 'Review visit or encounter flow using patient, timing, and visit-state fields.',
        'weight': 1.15,
    },
    'Trend Analysis': {
        'required_any': ['event_date', 'service_date', 'admission_date', 'discharge_date', 'diagnosis_date'],
        'description': 'Track counts or numeric metrics over time.',
        'weight': 1.0,
    },
    'Utilization Analysis': {
        'required_all_groups': [['entity_id', 'patient_id', 'member_id'], ['event_date', 'service_date', 'admission_date']],
        'description': 'Measure events per person and repeat activity over time.',
        'weight': 1.0,
    },
    'Cost Analysis': {
        'required_any': ['cost_amount', 'paid_amount', 'allowed_amount', 'billed_amount'],
        'description': 'Compare spend, payments, and high-cost segments.',
        'weight': 0.9,
    },
    'Provider / Facility Analysis': {
        'required_any': ['provider_id', 'provider_name', 'facility', 'room_id'],
        'description': 'Compare volume and cost across providers or facilities.',
        'weight': 0.95,
    },
    'Diagnosis / Procedure Segmentation': {
        'required_any': ['diagnosis_code', 'procedure_code', 'department', 'specialty', 'encounter_type', 'encounter_type_code', 'encounter_status', 'encounter_status_code'],
        'description': 'Break down activity and spend by clinical groupings.',
        'weight': 1.0,
    },
    'Readmission-Style Analysis': {
        'required_all_groups': [['entity_id', 'patient_id', 'member_id'], ['admission_date', 'service_date', 'event_date']],
        'description': 'Estimate approximate 30-day readmission-style patterns using encounter dates.',
        'weight': 1.05,
    },
    'Survival & Outcome Analysis': {
        'required_all_groups': [['survived'], ['cancer_stage', 'treatment_type', 'diagnosis_date']],
        'description': 'Compare outcomes by stage, treatment, and timing when clinical outcome fields are present.',
        'weight': 0.95,
    },
}


def _present_fields(canonical_map: dict[str, str], fields: list[str]) -> list[str]:
    return [field for field in fields if field in canonical_map]


def evaluate_analysis_readiness(canonical_map: dict[str, str], synthetic_fields: set[str] | None = None) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    available_count = 0
    partial_count = 0
    synthetic_fields = synthetic_fields or set()
    synthetic_supported_modules = 0
    weighted_score = 0.0
    total_weight = 0.0

    for analysis_name, rule in ANALYSIS_RULES.items():
        missing: list[str] = []
        status = 'Unavailable'
        support_type = 'Native'
        weight = float(rule.get('weight', 1.0) or 1.0)
        total_weight += weight
        if 'required_any' in rule:
            present = _present_fields(canonical_map, rule['required_any'])
            if present:
                status = 'Available'
                available_count += 1
                weighted_score += weight
                native_present = [field for field in present if field not in synthetic_fields]
                if not native_present and any(field in synthetic_fields for field in present):
                    support_type = 'Synthetic-assisted'
                    synthetic_supported_modules += 1
                    weighted_score -= weight * 0.15
            else:
                missing = rule['required_any']
        else:
            group_hits = []
            for group in rule['required_all_groups']:
                present = _present_fields(canonical_map, group)
                group_hits.append(present)
                if not present:
                    missing.extend(group)
            if all(group_hits):
                status = 'Available'
                available_count += 1
                weighted_score += weight
                if any(not [field for field in group if field not in synthetic_fields] for group in group_hits):
                    support_type = 'Synthetic-assisted'
                    synthetic_supported_modules += 1
                    weighted_score -= weight * 0.15
            elif any(group_hits):
                status = 'Partial'
                partial_count += 1
                weighted_score += weight * 0.55
                supported_groups = [group for group in group_hits if group]
                if supported_groups and all(not [field for field in group if field not in synthetic_fields] for group in supported_groups):
                    support_type = 'Synthetic-assisted'
                    weighted_score -= weight * 0.1
            else:
                status = 'Unavailable'

        rows.append({
            'analysis_module': analysis_name,
            'status': status,
            'support_type': support_type,
            'description': rule['description'],
            'module_weight': round(weight, 2),
            'missing_prerequisites': ', '.join(sorted(set(missing))) if missing else '-',
        })

    readiness_score = min(weighted_score / max(total_weight, 1.0), 1.0)
    readiness_score = max(readiness_score, 0.0)
    readiness_table = pd.DataFrame(rows)
    return {
        'readiness_table': readiness_table,
        'readiness_score': readiness_score,
        'available_count': available_count,
        'partial_count': partial_count,
        'synthetic_supported_modules': synthetic_supported_modules,
    }
