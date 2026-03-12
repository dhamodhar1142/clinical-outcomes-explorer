from __future__ import annotations

import pandas as pd


ANALYSIS_RULES = {
    'Trend Analysis': {
        'required_any': ['event_date', 'service_date', 'admission_date', 'discharge_date', 'diagnosis_date'],
        'description': 'Track counts or numeric metrics over time.',
    },
    'Utilization Analysis': {
        'required_all_groups': [['entity_id', 'patient_id', 'member_id'], ['event_date', 'service_date', 'admission_date']],
        'description': 'Measure events per person and repeat activity over time.',
    },
    'Cost Analysis': {
        'required_any': ['cost_amount', 'paid_amount', 'allowed_amount', 'billed_amount'],
        'description': 'Compare spend, payments, and high-cost segments.',
    },
    'Provider / Facility Analysis': {
        'required_any': ['provider_id', 'provider_name', 'facility'],
        'description': 'Compare volume and cost across providers or facilities.',
    },
    'Diagnosis / Procedure Segmentation': {
        'required_any': ['diagnosis_code', 'procedure_code', 'department', 'specialty'],
        'description': 'Break down activity and spend by clinical groupings.',
    },
    'Readmission-Style Analysis': {
        'required_all_groups': [['entity_id', 'patient_id', 'member_id'], ['admission_date', 'service_date', 'event_date']],
        'description': 'Estimate approximate 30-day readmission-style patterns using encounter dates.',
    },
    'Survival & Outcome Analysis': {
        'required_all_groups': [['survived'], ['cancer_stage', 'treatment_type', 'diagnosis_date']],
        'description': 'Compare outcomes by stage, treatment, and timing when clinical outcome fields are present.',
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

    for analysis_name, rule in ANALYSIS_RULES.items():
        missing: list[str] = []
        status = 'Unavailable'
        support_type = 'Native'
        if 'required_any' in rule:
            present = _present_fields(canonical_map, rule['required_any'])
            if present:
                status = 'Available'
                available_count += 1
                if any(field in synthetic_fields for field in present):
                    support_type = 'Synthetic-assisted'
                    synthetic_supported_modules += 1
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
                present_fields = {field for group in group_hits for field in group}
                if any(field in synthetic_fields for field in present_fields):
                    support_type = 'Synthetic-assisted'
                    synthetic_supported_modules += 1
            elif any(group_hits):
                status = 'Partial'
                partial_count += 1
                present_fields = {field for group in group_hits for field in group}
                if any(field in synthetic_fields for field in present_fields):
                    support_type = 'Synthetic-assisted'
            else:
                status = 'Unavailable'

        rows.append({
            'analysis_module': analysis_name,
            'status': status,
            'support_type': support_type,
            'description': rule['description'],
            'missing_prerequisites': ', '.join(sorted(set(missing))) if missing else '-',
        })

    readiness_score = min(((available_count * 1.0) + (partial_count * 0.5) - (synthetic_supported_modules * 0.15)) / max(len(ANALYSIS_RULES), 1), 1.0)
    readiness_score = max(readiness_score, 0.0)
    readiness_table = pd.DataFrame(rows)
    return {
        'readiness_table': readiness_table,
        'readiness_score': readiness_score,
        'available_count': available_count,
        'partial_count': partial_count,
        'synthetic_supported_modules': synthetic_supported_modules,
    }
