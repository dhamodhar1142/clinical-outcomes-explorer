from __future__ import annotations

from collections.abc import Iterable
from difflib import SequenceMatcher
import re
from typing import Any

import pandas as pd


def _normalize_token(value: str) -> str:
    return re.sub(r'[^a-z0-9]+', '_', str(value).strip().lower()).strip('_')


def _normalized_columns(columns: Iterable[str]) -> list[str]:
    return [_normalize_token(column) for column in columns if _normalize_token(column)]


def dataset_family_signature(columns: Iterable[str]) -> str:
    normalized = sorted(set(_normalized_columns(columns)))
    return '|'.join(normalized[:60])


def infer_dataset_family(columns: Iterable[str], dataset_name: str = '') -> dict[str, Any]:
    normalized = set(_normalized_columns(columns))
    dataset_hint = _normalize_token(dataset_name)
    families = [
        {
            'family_key': 'hospital-encounters',
            'family_label': 'Hospital Encounter Feed',
            'template_name': 'Hospital Encounter Template',
            'benchmark_profile': 'Hospital Encounters',
            'dataset_type_hint': 'Claims / encounter-level healthcare dataset',
            'signature_columns': {'pat_id', 'patient_id', 'medt_id', 'encounter_id', 'vis_en', 'vis_ex', 'vstat_cd', 'vtype_cd'},
        },
        {
            'family_key': 'payer-claims',
            'family_label': 'Payer Claims Feed',
            'template_name': 'Payer Claims Template',
            'benchmark_profile': 'Payer Claims',
            'dataset_type_hint': 'Claims / encounter-level healthcare dataset',
            'signature_columns': {'claim_id', 'member_id', 'paid_amount', 'allowed_amount', 'billed_amount', 'payer', 'plan'},
        },
        {
            'family_key': 'clinical-registry',
            'family_label': 'Clinical Registry Feed',
            'template_name': 'Clinical Registry Template',
            'benchmark_profile': 'Clinical Registry',
            'dataset_type_hint': 'Clinical registry dataset',
            'signature_columns': {'patient_id', 'diagnosis_date', 'treatment_type', 'cancer_stage', 'survived'},
        },
        {
            'family_key': 'generic-healthcare',
            'family_label': 'Generic Healthcare Feed',
            'template_name': 'Generic Healthcare Template',
            'benchmark_profile': 'Generic Healthcare',
            'dataset_type_hint': 'Healthcare dataset',
            'signature_columns': {'patient_id', 'service_date', 'department', 'diagnosis_code'},
        },
    ]
    best = families[-1]
    best_score = 0.0
    for family in families:
        signature = family['signature_columns']
        overlap_score = len(normalized & signature) / max(len(signature), 1)
        if family['family_key'] in dataset_hint:
            overlap_score += 0.15
        if overlap_score > best_score:
            best = family
            best_score = overlap_score
    return {
        **best,
        'match_score': round(best_score, 3),
        'signature': dataset_family_signature(columns),
        'normalized_columns': sorted(normalized),
    }


BUILTIN_MAPPING_PROFILES: list[dict[str, Any]] = [
    {
        'profile_id': 'builtin-hospital-encounters',
        'profile_name': 'Hospital Encounter Template',
        'profile_scope': 'Built-in',
        'family_key': 'hospital-encounters',
        'family_label': 'Hospital Encounter Feed',
        'benchmark_profile': 'Hospital Encounters',
        'dataset_type': 'Claims / encounter-level healthcare dataset',
        'mappings': {
            'patient_id': 'PAT_ID',
            'encounter_id': 'MEDT_ID',
            'admission_date': 'VIS_EN',
            'discharge_date': 'VIS_EX',
            'encounter_status_code': 'VSTAT_CD',
            'encounter_status': 'VSTAT_DES',
            'encounter_type_code': 'VTYPE_CD',
            'encounter_type': 'VTYPE_DES',
            'room_id': 'ROM_ID',
        },
    },
    {
        'profile_id': 'builtin-payer-claims',
        'profile_name': 'Payer Claims Template',
        'profile_scope': 'Built-in',
        'family_key': 'payer-claims',
        'family_label': 'Payer Claims Feed',
        'benchmark_profile': 'Payer Claims',
        'dataset_type': 'Claims / encounter-level healthcare dataset',
        'mappings': {
            'member_id': 'MEMBER_ID',
            'claim_id': 'CLAIM_ID',
            'service_date': 'SERVICE_DATE',
            'diagnosis_code': 'DIAGNOSIS_CODE',
            'procedure_code': 'PROCEDURE_CODE',
            'payer': 'PAYER',
            'plan': 'PLAN',
            'paid_amount': 'PAID_AMOUNT',
            'allowed_amount': 'ALLOWED_AMOUNT',
            'billed_amount': 'BILLED_AMOUNT',
        },
    },
    {
        'profile_id': 'builtin-clinical-registry',
        'profile_name': 'Clinical Registry Template',
        'profile_scope': 'Built-in',
        'family_key': 'clinical-registry',
        'family_label': 'Clinical Registry Feed',
        'benchmark_profile': 'Clinical Registry',
        'dataset_type': 'Clinical registry dataset',
        'mappings': {
            'patient_id': 'PATIENT_ID',
            'diagnosis_date': 'DIAGNOSIS_DATE',
            'treatment_type': 'TREATMENT_TYPE',
            'cancer_stage': 'CANCER_STAGE',
            'survived': 'SURVIVED',
            'provider_id': 'PROVIDER_ID',
        },
    },
    {
        'profile_id': 'builtin-ambiguous-encounter-raw',
        'profile_name': 'Encounter Raw Feed Template',
        'profile_scope': 'Built-in',
        'family_key': 'hospital-encounters',
        'family_label': 'Hospital Encounter Feed',
        'benchmark_profile': 'Hospital Encounters',
        'dataset_type': 'Claims / encounter-level healthcare dataset',
        'mappings': {
            'patient_id': 'MEMBER_KEY',
            'admission_date': 'VISIT_START_RAW',
            'discharge_date': 'VISIT_END_RAW',
            'encounter_type': 'VISIT_TYPE_LABEL',
            'encounter_status': 'VISIT_STATUS_LABEL',
            'room_id': 'ROOM_CODE',
        },
    },
]


def build_mapping_profile(
    board: pd.DataFrame,
    *,
    profile_name: str,
    dataset_type: str = '',
    family_key: str = '',
    family_label: str = '',
    benchmark_profile: str = 'Auto',
    profile_scope: str = 'Workspace',
) -> dict[str, Any]:
    safe_board = board if isinstance(board, pd.DataFrame) else pd.DataFrame()
    mappings = {
        str(row.get('mapped_field', '')).strip(): str(row.get('source_column', '')).strip()
        for row in safe_board.to_dict(orient='records')
        if str(row.get('mapped_field', '')).strip() and str(row.get('mapped_field', '')).strip() != 'Not mapped'
        and str(row.get('source_column', '')).strip()
    }
    signature = dataset_family_signature(mappings.values())
    return {
        'profile_id': f"user-{_normalize_token(profile_name)}",
        'profile_name': profile_name,
        'profile_scope': profile_scope,
        'dataset_type': dataset_type,
        'family_key': family_key or infer_dataset_family(mappings.values(), profile_name).get('family_key', 'generic-healthcare'),
        'family_label': family_label or infer_dataset_family(mappings.values(), profile_name).get('family_label', 'Generic Healthcare Feed'),
        'benchmark_profile': benchmark_profile,
        'signature': signature,
        'mappings': mappings,
    }


def available_mapping_profiles(user_profiles: Any = None) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for profile in BUILTIN_MAPPING_PROFILES:
        normalized.append(dict(profile))
    if isinstance(user_profiles, list):
        candidates = user_profiles
    elif isinstance(user_profiles, dict):
        candidates = list(user_profiles.values())
    else:
        candidates = []
    for profile in candidates:
        if not isinstance(profile, dict):
            continue
        mappings = profile.get('mappings', {})
        if not isinstance(mappings, dict) or not mappings:
            continue
        normalized.append(dict(profile))
    return normalized


def profile_mapping_for_columns(profile: dict[str, Any], columns: Iterable[str]) -> dict[str, str]:
    available_lookup = {_normalize_token(column): str(column) for column in columns}
    resolved: dict[str, str] = {}
    for canonical_field, raw_column in dict(profile.get('mappings', {})).items():
        matched = available_lookup.get(_normalize_token(str(raw_column)))
        if matched:
            resolved[str(canonical_field)] = matched
    return resolved


def _profile_match_score(profile: dict[str, Any], normalized_columns: set[str], family: dict[str, Any]) -> float:
    mappings = dict(profile.get('mappings', {}))
    mapped_columns = {_normalize_token(value) for value in mappings.values()}
    direct_overlap = len(mapped_columns & normalized_columns) / max(len(mapped_columns), 1)
    signature_similarity = 0.0
    profile_signature = str(profile.get('signature', '')).strip()
    if profile_signature:
        signature_similarity = SequenceMatcher(None, profile_signature, family.get('signature', '')).ratio()
    family_bonus = 0.12 if str(profile.get('family_key', '')) == str(family.get('family_key', '')) else 0.0
    scope_bonus = 0.06 if str(profile.get('profile_scope', '')).lower() != 'built-in' else 0.0
    return round((direct_overlap * 0.72) + (signature_similarity * 0.16) + family_bonus + scope_bonus, 3)


def suggest_mapping_profile(
    columns: Iterable[str],
    *,
    dataset_name: str = '',
    user_profiles: Any = None,
) -> dict[str, Any] | None:
    normalized_columns = set(_normalized_columns(columns))
    if not normalized_columns:
        return None
    family = infer_dataset_family(columns, dataset_name)
    profiles = available_mapping_profiles(user_profiles)
    ranked: list[dict[str, Any]] = []
    for profile in profiles:
        score = _profile_match_score(profile, normalized_columns, family)
        if score < 0.35:
            continue
        resolved_mappings = profile_mapping_for_columns(profile, columns)
        if not resolved_mappings:
            continue
        ranked.append(
            {
                **dict(profile),
                'suggestion_score': score,
                'resolved_mappings': resolved_mappings,
                'suggested_family': family,
            }
        )
    if not ranked:
        return None
    ranked.sort(key=lambda item: (float(item.get('suggestion_score', 0.0)), len(item.get('resolved_mappings', {}))), reverse=True)
    return ranked[0]


def build_profile_suggestion_table(columns: Iterable[str], *, dataset_name: str = '', user_profiles: Any = None) -> pd.DataFrame:
    family = infer_dataset_family(columns, dataset_name)
    profiles = available_mapping_profiles(user_profiles)
    rows: list[dict[str, Any]] = []
    normalized_columns = set(_normalized_columns(columns))
    for profile in profiles:
        score = _profile_match_score(profile, normalized_columns, family)
        if score < 0.2:
            continue
        resolved = profile_mapping_for_columns(profile, columns)
        rows.append(
            {
                'profile_name': str(profile.get('profile_name', 'Mapping profile')),
                'profile_scope': str(profile.get('profile_scope', 'Workspace')),
                'family_label': str(profile.get('family_label', family.get('family_label', 'Generic Healthcare Feed'))),
                'benchmark_profile': str(profile.get('benchmark_profile', 'Auto')),
                'suggestion_score': score,
                'resolved_mapping_count': len(resolved),
            }
        )
    return pd.DataFrame(rows).sort_values(['suggestion_score', 'resolved_mapping_count'], ascending=[False, False]).reset_index(drop=True) if rows else pd.DataFrame()


__all__ = [
    'BUILTIN_MAPPING_PROFILES',
    'available_mapping_profiles',
    'build_mapping_profile',
    'build_profile_suggestion_table',
    'dataset_family_signature',
    'infer_dataset_family',
    'profile_mapping_for_columns',
    'suggest_mapping_profile',
]
