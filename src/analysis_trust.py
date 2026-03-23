from __future__ import annotations

from typing import Any

import pandas as pd


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def build_analysis_trust_summary(
    *,
    dataset_name: str,
    source_meta: dict[str, Any],
    overview: dict[str, Any],
    readiness: dict[str, Any],
    healthcare: dict[str, Any],
    remediation_context: dict[str, Any],
    sample_info: dict[str, Any],
) -> dict[str, Any]:
    authoritative_row_count = int(source_meta.get('source_row_count', 0) or overview.get('rows', 0) or 0)
    analyzed_row_count = int(overview.get('rows', 0) or 0)
    profiled_row_count = int(sample_info.get('profile_sample_rows', analyzed_row_count) or analyzed_row_count)
    quality_row_count = int(sample_info.get('quality_sample_rows', analyzed_row_count) or analyzed_row_count)
    sampled_row_count = int(sample_info.get('total_rows', analyzed_row_count) or analyzed_row_count)
    sampling_active = bool(sample_info.get('sampling_applied', False))
    synthetic_helper_fields = int(remediation_context.get('synthetic_field_count', 0) or 0)
    derived_helper_fields = int(remediation_context.get('derived_field_count', 0) or 0)
    synthetic_supported_modules = int(readiness.get('synthetic_supported_modules', 0) or 0)
    readiness_score = float(readiness.get('readiness_score', 0.0) or 0.0)
    healthcare_readiness_score = float(healthcare.get('healthcare_readiness_score', 0.0) or 0.0)
    timeout_fallback = bool(healthcare.get('timeout_fallback', False))
    likely_dataset_type = str(healthcare.get('likely_dataset_type', 'Unclassified dataset'))

    trust_score = 1.0
    if sampling_active:
        trust_score -= 0.18
    if synthetic_helper_fields:
        trust_score -= min(0.16, synthetic_helper_fields * 0.02)
    if synthetic_supported_modules:
        trust_score -= min(0.18, synthetic_supported_modules * 0.05)
    if timeout_fallback:
        trust_score -= 0.25
    if healthcare_readiness_score < 0.35:
        trust_score -= 0.18
    elif healthcare_readiness_score < 0.6:
        trust_score -= 0.08
    if readiness_score < 0.45:
        trust_score -= 0.15
    elif readiness_score < 0.7:
        trust_score -= 0.07
    trust_score = _clamp(trust_score)

    if trust_score >= 0.85:
        trust_level = 'High'
        interpretation_mode = 'Authoritative for internal decision support'
    elif trust_score >= 0.68:
        trust_level = 'Moderate'
        interpretation_mode = 'Operationally useful with caveats'
    elif trust_score >= 0.5:
        trust_level = 'Directional'
        interpretation_mode = 'Directional only; validate before broad use'
    else:
        trust_level = 'Low'
        interpretation_mode = 'Not decision-ready without remediation'

    notes: list[str] = []
    recommended_uses: list[str] = []
    restricted_uses: list[str] = []

    if sampling_active:
        notes.append(
            f"Large-file safeguards are active. The source dataset contains {authoritative_row_count:,} rows while interactive analysis is using {analyzed_row_count:,} sampled rows."
        )
        restricted_uses.append('Use sampled outputs for prioritization and trend direction, not for externally reported precise prevalence without citing source-row context.')
    else:
        notes.append(f'Interactive analysis is using the full in-memory dataset slice of {analyzed_row_count:,} rows.')

    if synthetic_helper_fields:
        notes.append(
            f"{synthetic_helper_fields} synthetic helper fields and {derived_helper_fields} derived helper fields are participating in the current workflow."
        )
        restricted_uses.append('Do not treat synthetic helper-backed outputs as source-grade clinical or financial truth without native field confirmation.')
    else:
        notes.append('Current outputs do not require synthetic helper fields.')

    if synthetic_supported_modules:
        notes.append(f'{synthetic_supported_modules} modules are currently synthetic-assisted rather than fully native.')
    if timeout_fallback:
        notes.append('Healthcare analytics timeout fallback was used for at least one stage, so affected healthcare modules are lighter-weight than the normal path.')
        restricted_uses.append('Avoid making strong module-specific claims from timeout-fallback outputs until the full healthcare workflow completes normally.')

    if healthcare_readiness_score < 0.35:
        notes.append(f"Healthcare dataset fit is limited ({healthcare_readiness_score:.0%}) for this file type: {likely_dataset_type}.")
        restricted_uses.append('Use healthcare outputs mainly for exploratory triage and remediation planning, not for definitive clinical interpretation.')
    elif healthcare_readiness_score < 0.6:
        notes.append(f"Healthcare dataset fit is partial ({healthcare_readiness_score:.0%}); some domain outputs remain best treated as directional.")

    if trust_level == 'High':
        recommended_uses.extend([
            'Internal operational prioritization',
            'Stakeholder-ready reporting with normal caveat disclosure',
            'Trend and cohort review',
        ])
    elif trust_level == 'Moderate':
        recommended_uses.extend([
            'Internal prioritization and planning',
            'Exploratory cohort and trend review',
            'Decision support with explicit disclosure notes',
        ])
    elif trust_level == 'Directional':
        recommended_uses.extend([
            'Hypothesis generation',
            'Remediation planning',
            'Schema and workflow triage',
        ])
    else:
        recommended_uses.extend([
            'Dataset triage',
            'Remediation targeting',
            'Field mapping review',
        ])

    if not restricted_uses:
        restricted_uses.append('No additional trust restrictions were triggered beyond standard governance review.')

    disclosure_table = pd.DataFrame(
        [
            {'signal': 'Trust level', 'status': trust_level, 'detail': interpretation_mode},
            {'signal': 'Authoritative row count', 'status': f'{authoritative_row_count:,}', 'detail': 'Source-row count from the uploaded dataset context.'},
            {'signal': 'Analyzed row count', 'status': f'{analyzed_row_count:,}', 'detail': 'Rows currently loaded into the interactive analysis pipeline.'},
            {'signal': 'Profile sample rows', 'status': f'{profiled_row_count:,}', 'detail': 'Rows used for profile-level detail surfaces.'},
            {'signal': 'Quality sample rows', 'status': f'{quality_row_count:,}', 'detail': 'Rows used for quality-review sampling when large-file safeguards apply.'},
            {'signal': 'Sampling mode', 'status': 'Sampled' if sampling_active else 'Full', 'detail': str(source_meta.get('sampling_mode', 'full')).replace('_', ' ').title()},
            {'signal': 'Synthetic helper fields', 'status': f'{synthetic_helper_fields:,}', 'detail': 'Synthetic helper columns currently participating in the workflow.'},
            {'signal': 'Synthetic-assisted modules', 'status': f'{synthetic_supported_modules:,}', 'detail': 'Modules with synthetic-assisted readiness support.'},
            {'signal': 'Healthcare fit', 'status': f'{healthcare_readiness_score:.0%}', 'detail': likely_dataset_type},
            {'signal': 'Readiness score', 'status': f'{readiness_score:.0%}', 'detail': 'Overall module readiness across the mapped workflow.'},
        ]
    )

    summary_text = (
        f"{dataset_name} is currently rated {trust_level.lower()} trust ({trust_score:.0%}) "
        f"for internal interpretation. {interpretation_mode}."
    )

    return {
        'available': True,
        'dataset_name': dataset_name,
        'trust_score': trust_score,
        'trust_level': trust_level,
        'interpretation_mode': interpretation_mode,
        'authoritative_row_count': authoritative_row_count,
        'analyzed_row_count': analyzed_row_count,
        'profiled_row_count': profiled_row_count,
        'quality_row_count': quality_row_count,
        'sampled_row_count': sampled_row_count,
        'sampling_active': sampling_active,
        'sampling_mode': str(source_meta.get('sampling_mode', 'full')),
        'synthetic_helper_fields': synthetic_helper_fields,
        'derived_helper_fields': derived_helper_fields,
        'synthetic_supported_modules': synthetic_supported_modules,
        'timeout_fallback': timeout_fallback,
        'healthcare_readiness_score': healthcare_readiness_score,
        'readiness_score': readiness_score,
        'likely_dataset_type': likely_dataset_type,
        'notes': notes,
        'recommended_uses': recommended_uses,
        'restricted_uses': restricted_uses,
        'summary_text': summary_text,
        'disclosure_table': disclosure_table,
    }

