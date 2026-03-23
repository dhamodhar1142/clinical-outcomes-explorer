from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.data_loader import DEMO_DATASETS
from src.readiness_engine import ANALYSIS_RULES
from src.semantic_mapper import FIELD_DESCRIPTIONS


SECTION_REQUIREMENTS = {
    'Healthcare Intelligence': {
        'required_any': ['patient_id', 'member_id', 'encounter_id', 'provider_id', 'provider_name', 'facility', 'diagnosis_code', 'procedure_code'],
        'example_fields': ['patient_id', 'admission_date', 'discharge_date', 'diagnosis_code', 'department', 'cost_amount'],
        'best_docs': ['demo_notes.md'],
        'module_name': 'Diagnosis / Procedure Segmentation',
    },
    'Trend Analysis': {
        'module_name': 'Trend Analysis',
        'example_fields': ['event_date', 'record_count', 'department'],
        'best_docs': ['demo_notes.md'],
    },
    'Cohort Analysis': {
        'required_any': ['patient_id', 'member_id', 'entity_id', 'diagnosis_code', 'treatment_type', 'cancer_stage', 'gender', 'age'],
        'example_fields': ['patient_id', 'diagnosis_code', 'treatment_type', 'event_date', 'readmission', 'cost_amount'],
        'best_docs': ['demo_notes.md'],
        'module_name': 'Cohort Analysis',
    },
    'Export Center': {
        'required_any': ['entity_id', 'patient_id', 'member_id', 'event_date', 'service_date', 'cost_amount', 'diagnosis_code'],
        'example_fields': ['entity_id', 'event_date', 'category', 'cost_amount', 'status'],
        'best_docs': ['demo_notes.md', 'deployment.md'],
        'module_name': 'Data Profiling',
    },
}


FIELD_EXAMPLES = {
    'entity_id': 'ORD-10021',
    'patient_id': 'PAT-0001',
    'member_id': 'MBR-9001',
    'encounter_id': 'ENC-44019',
    'provider_id': 'NPI-1881761201',
    'provider_name': 'Dr. Avery Chen',
    'facility': 'North Campus Hospital',
    'diagnosis_code': 'I50.9',
    'procedure_code': '99233',
    'admission_date': '2025-01-08',
    'discharge_date': '2025-01-10',
    'service_date': '2025-01-08',
    'event_date': '2025-01-08',
    'age': '67',
    'gender': 'F',
    'department': 'Cardiology',
    'treatment_type': 'Inpatient follow-up',
    'cancer_stage': 'Stage II',
    'readmission': '0',
    'cost_amount': '18250.00',
    'record_count': '42',
    'category': 'Regional operations',
    'status': 'Complete',
}

DOC_TITLES = {
    'demo_notes.md': 'Demo Dataset Guide',
    'deployment.md': 'Deployment and Runtime Guide',
    'storage_setup.md': 'Storage Setup Guide',
    'persistence_setup.md': 'Persistence Setup Guide',
    'auth_setup.md': 'Authentication Setup Guide',
}


@dataclass(frozen=True)
class ActionableRemediationMessage:
    severity: str
    module: str
    timestamp: str
    title: str
    issue: str
    cause: str
    remediations: list[str]
    doc_links: list[dict[str, str]]
    details_table: pd.DataFrame
    suggestions_table: pd.DataFrame
    example_structure: pd.DataFrame


def _safe_df(value: Any) -> pd.DataFrame:
    return value if isinstance(value, pd.DataFrame) else pd.DataFrame()


def _section_requirement(section_name: str) -> dict[str, Any]:
    for key, value in SECTION_REQUIREMENTS.items():
        if key in section_name:
            return value
    return {
        'required_any': ['entity_id', 'event_date', 'category'],
        'example_fields': ['entity_id', 'event_date', 'category'],
        'best_docs': ['demo_notes.md'],
        'module_name': '',
    }


def _required_fields_for_section(section_name: str) -> list[str]:
    config = _section_requirement(section_name)
    module_name = str(config.get('module_name', '')).strip()
    if module_name and module_name in ANALYSIS_RULES:
        rule = ANALYSIS_RULES[module_name]
        if 'required_any' in rule:
            return list(rule['required_any'])
        fields: list[str] = []
        for group in rule.get('required_all_groups', []):
            fields.extend(group)
        return list(dict.fromkeys(fields))
    return list(config.get('required_any', []))


def _example_structure_table(fields: list[str]) -> pd.DataFrame:
    rows = []
    for field in fields:
        rows.append(
            {
                'required_field': field,
                'example_value': FIELD_EXAMPLES.get(field, 'example_value'),
                'why_needed': FIELD_DESCRIPTIONS.get(field, 'Supports this workflow when mapped.'),
            }
        )
    return pd.DataFrame(rows)


def _mapping_suggestions_for_fields(pipeline: dict[str, Any], required_fields: list[str]) -> pd.DataFrame:
    suggestion_table = _safe_df(pipeline.get('semantic', {}).get('suggestion_table'))
    mapping_table = _safe_df(pipeline.get('semantic', {}).get('mapping_table'))
    mapped_fields = set(mapping_table.get('semantic_label', pd.Series(dtype=str)).astype(str).tolist()) if not mapping_table.empty else set()
    rows: list[dict[str, Any]] = []
    for field in required_fields:
        if field in mapped_fields:
            continue
        field_suggestions = suggestion_table[suggestion_table.get('suggested_field', pd.Series(dtype=str)).astype(str) == field] if not suggestion_table.empty else pd.DataFrame()
        if field_suggestions.empty:
            rows.append(
                {
                    'required_field': field,
                    'suggested_source_column': 'No strong suggestion yet',
                    'confidence_score': 0.0,
                    'why': FIELD_DESCRIPTIONS.get(field, 'Map a source column to unlock this feature.'),
                }
            )
            continue
        best = field_suggestions.sort_values(['confidence_score', 'suggestion_rank'], ascending=[False, True]).iloc[0]
        rows.append(
            {
                'required_field': field,
                'suggested_source_column': str(best.get('source_column', '')),
                'confidence_score': float(best.get('confidence_score', 0.0) or 0.0),
                'why': str(best.get('reason', FIELD_DESCRIPTIONS.get(field, 'Map a source column to unlock this feature.'))),
            }
        )
    return pd.DataFrame(rows)


def _missing_fields_table(pipeline: dict[str, Any], required_fields: list[str]) -> pd.DataFrame:
    canonical_map = pipeline.get('semantic', {}).get('canonical_map', {})
    rows = []
    for field in required_fields:
        rows.append(
            {
                'required_field': field,
                'status': 'Mapped' if field in canonical_map else 'Missing',
                'mapped_source_column': str(canonical_map.get(field, '')),
                'why_needed': FIELD_DESCRIPTIONS.get(field, 'Supports this workflow when mapped.'),
            }
        )
    return pd.DataFrame(rows)


def _docs_table(doc_names: list[str]) -> pd.DataFrame:
    docs_root = Path.cwd() / 'docs'
    rows = []
    for doc_name in doc_names:
        doc_path = docs_root / doc_name
        rows.append(
            {
                'resource': doc_name,
                'path': str(doc_path),
                'sample_datasets': ', '.join(DEMO_DATASETS.keys()),
            }
        )
    return pd.DataFrame(rows)


def _docs_links(doc_names: list[str]) -> list[dict[str, str]]:
    docs_root = Path.cwd() / 'docs'
    links: list[dict[str, str]] = []
    for doc_name in doc_names:
        links.append(
            {
                'title': DOC_TITLES.get(doc_name, doc_name),
                'path': str(docs_root / doc_name),
            }
        )
    return links


def _structured_fallback_message(section_name: str, pipeline: dict[str, Any], missing_fields: list[str], docs: list[str]) -> ActionableRemediationMessage:
    readiness_score = float(pipeline.get('readiness', {}).get('readiness_score', 0.0) or 0.0)
    mapping_suggestions = _mapping_suggestions_for_fields(pipeline, _required_fields_for_section(section_name))
    canonical_map = pipeline.get('semantic', {}).get('canonical_map', {}) or {}
    entity_fields = {'entity_id', 'patient_id', 'member_id'}
    missing_entity_fields = [field for field in missing_fields if field in entity_fields]
    weak_entity_identifier = bool('Healthcare' in section_name and not any(field in canonical_map for field in entity_fields))
    if missing_entity_fields or weak_entity_identifier:
        issue = (
            f"Entity ID field is missing or too weakly mapped for {section_name.lower()} "
            f"(readiness: {readiness_score:.0%})."
        )
        cause = (
            'The dataset does not yet have a strongly detected patient, member, or entity identifier. '
            'Without that anchor, advanced cohorting and healthcare workflows cannot activate safely.'
        )
        remediations = [
            'Check the source data for a unique identifier column such as patient_id, entity_id, member_id, or subject_id.',
            'Use Data Intake > Field Remapping Studio to map the strongest identifier-like field to the missing canonical role.',
            'If the source field name is ambiguous, rename it before upload or apply the Auto-fix option below when the confidence is strong enough.',
        ]
    else:
        issue = f"{section_name} is blocked because required fields are missing or weakly mapped."
        cause = (
            'This view depends on a minimum set of identifiers, dates, outcomes, or grouping columns. '
            'The current dataset does not satisfy enough of those requirements yet.'
        )
        remediations = [
            'Review the missing field list below and compare it with your source schema.',
            'Use the mapping suggestions to connect likely source columns to the required canonical fields.',
            'If the necessary fields are not present, re-upload a wider extract or a dataset prepared for this workflow.',
        ]
    return ActionableRemediationMessage(
        severity='Error',
        module=section_name,
        timestamp=datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC'),
        title=f'ERROR in {section_name}',
        issue=issue,
        cause=cause,
        remediations=remediations,
        doc_links=_docs_links(docs),
        details_table=_missing_fields_table(pipeline, _required_fields_for_section(section_name)),
        suggestions_table=mapping_suggestions,
        example_structure=_example_structure_table(_section_requirement(section_name).get('example_fields', _required_fields_for_section(section_name))),
    )


def build_large_dataset_actionable_message(source_meta: dict[str, Any], sample_info: dict[str, Any] | None = None) -> ActionableRemediationMessage:
    sample_info = sample_info or {}
    file_size_mb = float(source_meta.get('file_size_mb', 0.0) or 0.0)
    total_rows = int(sample_info.get('total_rows', source_meta.get('row_count', 0)) or 0)
    analyzed_rows = int(sample_info.get('profile_sample_rows', sample_info.get('analyzed_rows', total_rows)) or 0)
    issue = (
        f'Large dataset detected ({file_size_mb:.1f} MB, {total_rows:,} rows). '
        'Performance mode is using streaming or sampling to keep the app stable.'
    )
    cause = (
        'Files above 50 MB switch into streaming mode, and larger files can use sampled profiling so interactive analysis avoids memory spikes and timeouts.'
    )
    remediations = [
        'The current summary remains suitable for iteration and should stay within about +/-5% for key statistics in sampled mode.',
        'Choose Analyze full dataset in the upload flow if you need full-fidelity profiling and can accept a slower run.',
        'Reduce the file to relevant rows or columns if you want faster interactive turnaround.',
    ]
    details = pd.DataFrame(
        [
            {
                'file_size_mb': round(file_size_mb, 1),
                'total_rows': total_rows,
                'analyzed_rows': analyzed_rows,
                'sampling_mode': str(sample_info.get('sampling_mode', source_meta.get('sampling_mode', 'full'))),
                'ingestion_strategy': str(source_meta.get('ingestion_strategy', sample_info.get('ingestion_strategy', 'standard'))),
            }
        ]
    )
    return ActionableRemediationMessage(
        severity='Warning',
        module='Large Dataset Handling',
        timestamp=datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC'),
        title='WARNING in Large Dataset Handling',
        issue=issue,
        cause=cause,
        remediations=remediations,
        doc_links=_docs_links(['demo_notes.md', 'deployment.md']),
        details_table=details,
        suggestions_table=pd.DataFrame(),
        example_structure=pd.DataFrame(),
    )


def build_cancellation_timeout_message(*, file_size_mb: float, row_count: int, column_count: int) -> ActionableRemediationMessage:
    issue = 'Cancellation took longer than 30 seconds, so the analysis job was force-terminated.'
    cause = (
        f'The running analysis ({file_size_mb:.1f} MB, {row_count:,} rows, {column_count:,} columns) exceeded the cancellation timeout. '
        'This usually happens when a heavy stage is waiting on disk or in-flight processing.'
    )
    remediations = [
        'The job has already been terminated and the UI is reset for a new analysis.',
        'Use sampling or a narrower extract for faster iteration if you plan to cancel and retry often.',
        'If this repeats on similar files, capture the file size and row count and share them with support for tuning.',
    ]
    details = pd.DataFrame(
        [
            {
                'file_size_mb': round(file_size_mb, 1),
                'row_count': row_count,
                'column_count': column_count,
                'timeout_seconds': 30,
                'completion_status': 'Force-killed and cleaned up',
            }
        ]
    )
    return ActionableRemediationMessage(
        severity='Critical',
        module='Cancellation Timeout',
        timestamp=datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC'),
        title='CRITICAL in Cancellation Timeout',
        issue=issue,
        cause=cause,
        remediations=remediations,
        doc_links=_docs_links(['deployment.md', 'demo_notes.md']),
        details_table=details,
        suggestions_table=pd.DataFrame(),
        example_structure=pd.DataFrame(),
    )


def build_actionable_fallback_context(section_name: str, pipeline: dict[str, Any]) -> dict[str, Any]:
    required_fields = _required_fields_for_section(section_name)
    missing_fields = _missing_fields_table(pipeline, required_fields)
    suggestions = _mapping_suggestions_for_fields(pipeline, required_fields)
    doc_names = _section_requirement(section_name).get('best_docs', ['demo_notes.md'])
    docs = _docs_table(doc_names)
    auto_fix_available = bool(
        not suggestions.empty
        and (suggestions.get('confidence_score', pd.Series(dtype=float)) >= 0.62).any()
    )
    message = _structured_fallback_message(
        section_name,
        pipeline,
        missing_fields[missing_fields['status'] == 'Missing']['required_field'].astype(str).tolist() if not missing_fields.empty else [],
        doc_names,
    )
    return {
        'message': message,
        'required_fields_table': missing_fields,
        'missing_fields': missing_fields[missing_fields['status'] == 'Missing']['required_field'].astype(str).tolist() if not missing_fields.empty else [],
        'mapping_suggestions': suggestions,
        'example_structure': _example_structure_table(_section_requirement(section_name).get('example_fields', required_fields)),
        'docs_table': docs,
        'auto_fix_available': auto_fix_available,
    }


__all__ = [
    'ActionableRemediationMessage',
    'build_actionable_fallback_context',
    'build_cancellation_timeout_message',
    'build_large_dataset_actionable_message',
]
