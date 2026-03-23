from __future__ import annotations

import math
from typing import Any

import pandas as pd


BENCHMARK_PROFILES: dict[str, dict[str, Any]] = {
    'Generic Healthcare': {
        'profile_family': 'generic-healthcare',
        'rate_bands': {
            'Readmission rate': (0.05, 0.35),
            'High-risk share': (0.05, 0.50),
        },
        'numeric_bands': {
            'Average length of stay': (1.0, 12.0),
            'Average cost': (500.0, 100000.0),
        },
        'detail_note': 'Generic healthcare reference band. Validate against local source-system norms when available.',
    },
    'Hospital Encounters': {
        'profile_family': 'hospital-encounters',
        'rate_bands': {
            'Readmission rate': (0.08, 0.25),
            'High-risk share': (0.08, 0.45),
        },
        'numeric_bands': {
            'Average length of stay': (2.0, 9.0),
            'Average cost': (1200.0, 60000.0),
        },
        'detail_note': 'Encounter-oriented benchmark profile tuned for inpatient and facility operational extracts.',
    },
    'Payer Claims': {
        'profile_family': 'payer-claims',
        'rate_bands': {
            'Readmission rate': (0.06, 0.22),
            'High-risk share': (0.10, 0.55),
        },
        'numeric_bands': {
            'Average length of stay': (1.0, 8.0),
            'Average cost': (800.0, 85000.0),
        },
        'detail_note': 'Claims-oriented benchmark profile tuned for payer and adjudication-style datasets.',
    },
    'Clinical Registry': {
        'profile_family': 'clinical-registry',
        'rate_bands': {
            'Readmission rate': (0.03, 0.18),
            'High-risk share': (0.03, 0.35),
        },
        'numeric_bands': {
            'Average length of stay': (1.0, 10.0),
            'Average cost': (500.0, 45000.0),
        },
        'detail_note': 'Registry-oriented benchmark profile tuned for disease- or program-specific cohorts.',
    },
}

REPORTING_THRESHOLD_PRESETS: dict[str, dict[str, Any]] = {
    'Conservative': {
        'minimum_trust_score': 0.84,
        'allow_directional_external_reporting': False,
        'label': 'Conservative',
    },
    'Standard': {
        'minimum_trust_score': 0.76,
        'allow_directional_external_reporting': False,
        'label': 'Standard',
    },
    'Permissive': {
        'minimum_trust_score': 0.68,
        'allow_directional_external_reporting': True,
        'label': 'Permissive',
    },
}

MODULE_METRIC_LINEAGE = {
    'Analysis readiness': {
        'analytics_module': 'Readiness Review',
        'driving_fields': ['canonical_map', 'module_prerequisites'],
        'lineage_type': 'derived',
    },
    'Healthcare readiness': {
        'analytics_module': 'Healthcare Intelligence',
        'driving_fields': ['matched_healthcare_fields', 'canonical_map'],
        'lineage_type': 'derived',
    },
    'Overall readmission rate': {
        'analytics_module': 'Readmission Analytics',
        'driving_fields': ['readmission', 'patient_id', 'admission_date', 'discharge_date'],
        'lineage_type': 'native',
    },
    'High-risk population share': {
        'analytics_module': 'Risk Segmentation',
        'driving_fields': ['diagnosis_code', 'cancer_stage', 'smoking_status', 'comorbidities'],
        'lineage_type': 'inferred',
    },
    'Average cost': {
        'analytics_module': 'Cost Analysis',
        'driving_fields': ['cost_amount', 'paid_amount', 'allowed_amount', 'billed_amount'],
        'lineage_type': 'native',
    },
}

ROLE_REPORTING_DEFAULTS: dict[str, str] = {
    'Admin': 'Standard',
    'Data Steward': 'Conservative',
    'Executive': 'Standard',
    'Analyst': 'Standard',
    'Clinician': 'Conservative',
    'Researcher': 'Permissive',
    'Viewer': 'Conservative',
}


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _safe_df(table: Any) -> pd.DataFrame:
    return table if isinstance(table, pd.DataFrame) else pd.DataFrame()


def _rate_confidence_interval(rate: float, count: int, z: float = 1.96) -> tuple[float | None, float | None]:
    if count <= 0 or pd.isna(rate):
        return None, None
    rate = _clamp(rate)
    denominator = 1.0 + (z * z / count)
    center = (rate + (z * z) / (2 * count)) / denominator
    margin = (
        z
        * math.sqrt(((rate * (1.0 - rate)) / count) + ((z * z) / (4 * count * count)))
        / denominator
    )
    return _clamp(center - margin), _clamp(center + margin)


def _two_sided_normal_p_value(z_score: float) -> float:
    return float(math.erfc(abs(float(z_score)) / math.sqrt(2.0)))


def _numeric_confidence_interval(
    mean: float,
    stddev: float | None,
    count: int,
    z: float = 1.96,
) -> tuple[float | None, float | None]:
    if count <= 1 or stddev is None or pd.isna(mean) or pd.isna(stddev):
        return None, None
    stddev = float(stddev)
    if stddev <= 0:
        return None, None
    margin = z * (stddev / math.sqrt(count))
    return float(mean - margin), float(mean + margin)


def _welch_numeric_test(
    selected_mean: float,
    selected_stddev: float | None,
    selected_count: int,
    population_mean: float,
    population_stddev: float | None,
    population_count: int,
) -> tuple[float | None, str, str]:
    if (
        selected_count <= 1
        or population_count <= 1
        or selected_stddev is None
        or population_stddev is None
        or pd.isna(selected_mean)
        or pd.isna(population_mean)
        or pd.isna(selected_stddev)
        or pd.isna(population_stddev)
    ):
        return None, 'Not tested', 'Not tested'
    selected_variance = (float(selected_stddev) ** 2) / max(selected_count, 1)
    population_variance = (float(population_stddev) ** 2) / max(population_count, 1)
    denominator = math.sqrt(selected_variance + population_variance)
    if denominator <= 0:
        return None, 'Not tested', 'Not tested'
    z_score = (float(selected_mean) - float(population_mean)) / denominator
    p_value = _two_sided_normal_p_value(z_score)
    significance = (
        'Highly significant'
        if p_value < 0.01
        else 'Statistically significant'
        if p_value < 0.05
        else 'Not statistically significant'
    )
    effect_size = abs(float(selected_mean) - float(population_mean))
    effect_label = (
        'Large effect'
        if effect_size >= 1.0
        else 'Moderate effect'
        if effect_size >= 0.5
        else 'Small effect'
        if effect_size >= 0.2
        else 'Minimal effect'
    )
    return p_value, significance, effect_label


def _two_proportion_test(
    selected_rate: float,
    selected_count: int,
    population_rate: float,
    population_count: int,
) -> tuple[float | None, str]:
    if selected_count <= 0 or population_count <= 0:
        return None, 'Not tested'
    pooled_events = (selected_rate * selected_count) + (population_rate * population_count)
    pooled_rate = pooled_events / max(selected_count + population_count, 1)
    variance = pooled_rate * (1.0 - pooled_rate) * ((1.0 / selected_count) + (1.0 / population_count))
    if variance <= 0:
        return None, 'Not tested'
    z_score = (selected_rate - population_rate) / math.sqrt(variance)
    p_value = _two_sided_normal_p_value(z_score)
    if p_value < 0.01:
        label = 'Highly significant'
    elif p_value < 0.05:
        label = 'Statistically significant'
    else:
        label = 'Not statistically significant'
    return p_value, label


def _effect_size_label(delta: float) -> str:
    magnitude = abs(float(delta))
    if magnitude >= 0.20:
        return 'Large effect'
    if magnitude >= 0.10:
        return 'Moderate effect'
    if magnitude >= 0.03:
        return 'Small effect'
    return 'Minimal effect'


def resolve_benchmark_profile(
    *,
    dataset_name: str,
    source_meta: dict[str, Any],
    healthcare: dict[str, Any],
    active_control_values: dict[str, Any],
) -> dict[str, Any]:
    custom_packs = active_control_values.get('organization_benchmark_packs', {})
    custom_packs = custom_packs if isinstance(custom_packs, dict) else {}
    available_profiles = {**BENCHMARK_PROFILES, **custom_packs}
    active_pack_name = str(active_control_values.get('active_benchmark_pack_name', 'None')).strip()
    requested = str(active_control_values.get('accuracy_benchmark_profile', 'Auto')).strip()
    if active_pack_name and active_pack_name != 'None' and active_pack_name in available_profiles:
        profile_name = active_pack_name
        reason = 'Organization-specific benchmark pack selected in the current workspace.'
    elif requested and requested != 'Auto' and requested in available_profiles:
        profile_name = requested
        reason = 'User-selected benchmark profile.'
    else:
        dataset_hint = f"{dataset_name} {source_meta.get('description', '')} {healthcare.get('likely_dataset_type', '')}".lower()
        if any(token in dataset_hint for token in ['claim', 'payer', 'adjudic']):
            profile_name = 'Payer Claims'
        elif any(token in dataset_hint for token in ['registry', 'oncology registry', 'clinical registry']):
            profile_name = 'Clinical Registry'
        elif any(token in dataset_hint for token in ['encounter', 'visit', 'hospital', 'inpatient', 'ed']):
            profile_name = 'Hospital Encounters'
        else:
            profile_name = 'Generic Healthcare'
        reason = 'Auto-selected benchmark profile from dataset and workflow hints.'
    profile = dict(available_profiles.get(profile_name, BENCHMARK_PROFILES['Generic Healthcare']))
    profile['profile_name'] = profile_name
    profile['selection_reason'] = reason
    profile['profile_scope'] = 'Organization pack' if profile_name in custom_packs else 'Built-in'
    return profile


def resolve_reporting_threshold_policy(active_control_values: dict[str, Any]) -> dict[str, Any]:
    role = str(active_control_values.get('active_role', 'Analyst')).strip() or 'Analyst'
    requested_profile = str(active_control_values.get('accuracy_reporting_threshold_profile', 'Role default')).strip() or 'Role default'
    if requested_profile == 'Role default':
        profile_name = ROLE_REPORTING_DEFAULTS.get(role, 'Standard')
        selection_reason = f'Role-based default for {role}.'
    else:
        profile_name = requested_profile if requested_profile in REPORTING_THRESHOLD_PRESETS else 'Standard'
        selection_reason = 'User-selected reporting threshold profile.'
    policy = dict(REPORTING_THRESHOLD_PRESETS.get(profile_name, REPORTING_THRESHOLD_PRESETS['Standard']))
    if 'accuracy_reporting_min_trust_score' in active_control_values:
        policy['minimum_trust_score'] = _clamp(
            float(active_control_values.get('accuracy_reporting_min_trust_score', policy['minimum_trust_score'])),
            0.50,
            0.95,
        )
        selection_reason += ' Minimum trust score overridden in the current workspace.'
    if 'accuracy_allow_directional_external_reporting' in active_control_values:
        policy['allow_directional_external_reporting'] = bool(
            active_control_values.get('accuracy_allow_directional_external_reporting')
        )
        selection_reason += ' Directional external reporting preference overridden in the current workspace.'
    policy['profile_name'] = profile_name
    policy['role'] = role
    policy['selection_reason'] = selection_reason
    return policy


def _reporting_safe(
    *,
    support_type: str,
    trust_score: float,
    sampling_active: bool,
    policy: dict[str, Any],
) -> bool:
    if support_type == 'native' and trust_score >= float(policy.get('minimum_trust_score', 0.76)) and not sampling_active:
        return True
    if support_type != 'native' and bool(policy.get('allow_directional_external_reporting')) and trust_score >= max(
        0.60,
        float(policy.get('minimum_trust_score', 0.76)) - 0.08,
    ):
        return True
    return False


def add_rate_stability_columns(
    table: pd.DataFrame,
    *,
    rate_col: str,
    count_col: str = 'record_count',
    lower_col: str = 'confidence_lower',
    upper_col: str = 'confidence_upper',
    width_col: str = 'confidence_width',
    band_col: str = 'stability_band',
) -> pd.DataFrame:
    frame = _safe_df(table).copy()
    if frame.empty or rate_col not in frame.columns or count_col not in frame.columns:
        return frame
    lower_values: list[float | None] = []
    upper_values: list[float | None] = []
    widths: list[float | None] = []
    bands: list[str] = []
    for _, row in frame.iterrows():
        count = int(pd.to_numeric(pd.Series([row.get(count_col)]), errors='coerce').fillna(0).iloc[0])
        rate = float(pd.to_numeric(pd.Series([row.get(rate_col)]), errors='coerce').fillna(0.0).iloc[0])
        lower, upper = _rate_confidence_interval(rate, count)
        lower_values.append(lower)
        upper_values.append(upper)
        width = (upper - lower) if lower is not None and upper is not None else None
        widths.append(width)
        if count < 10:
            bands.append('Very low stability')
        elif count < 25:
            bands.append('Low stability')
        elif width is not None and width > 0.25:
            bands.append('Directional only')
        elif width is not None and width > 0.12:
            bands.append('Moderate stability')
        else:
            bands.append('High stability')
    frame[lower_col] = lower_values
    frame[upper_col] = upper_values
    frame[width_col] = widths
    frame[band_col] = bands
    return frame


def add_comparison_stability_columns(
    table: pd.DataFrame,
    *,
    cohort_size: int,
    population_size: int,
    selected_col: str = 'selected_cohort',
    overall_col: str = 'overall_population',
) -> pd.DataFrame:
    frame = _safe_df(table).copy()
    if frame.empty:
        return frame
    if selected_col not in frame.columns or overall_col not in frame.columns:
        return frame
    selected_lower: list[float | None] = []
    selected_upper: list[float | None] = []
    overall_lower: list[float | None] = []
    overall_upper: list[float | None] = []
    bands: list[str] = []
    p_values: list[float | None] = []
    significance_labels: list[str] = []
    effect_labels: list[str] = []
    test_names: list[str] = []
    selected_numeric_lower: list[float | None] = []
    selected_numeric_upper: list[float | None] = []
    overall_numeric_lower: list[float | None] = []
    overall_numeric_upper: list[float | None] = []
    rate_metrics = {'Readmission Rate', 'Survival Rate', 'High-Risk Share', 'Population Share'}
    for _, row in frame.iterrows():
        metric = str(row.get('metric', ''))
        selected = pd.to_numeric(pd.Series([row.get(selected_col)]), errors='coerce').iloc[0]
        overall = pd.to_numeric(pd.Series([row.get(overall_col)]), errors='coerce').iloc[0]
        if metric in rate_metrics and pd.notna(selected) and pd.notna(overall):
            lower, upper = _rate_confidence_interval(float(selected), cohort_size)
            pop_lower, pop_upper = _rate_confidence_interval(float(overall), population_size)
            selected_lower.append(lower)
            selected_upper.append(upper)
            overall_lower.append(pop_lower)
            overall_upper.append(pop_upper)
            p_value, significance = _two_proportion_test(float(selected), cohort_size, float(overall), population_size)
            p_values.append(p_value)
            significance_labels.append(significance)
            effect_labels.append(_effect_size_label(float(selected) - float(overall)))
            test_names.append('Two-proportion z-test')
            selected_numeric_lower.append(None)
            selected_numeric_upper.append(None)
            overall_numeric_lower.append(None)
            overall_numeric_upper.append(None)
            width = (upper - lower) if lower is not None and upper is not None else None
            if cohort_size < 15:
                bands.append('Low stability')
            elif width is not None and width > 0.20:
                bands.append('Directional only')
            elif width is not None and width > 0.10:
                bands.append('Moderate stability')
            else:
                bands.append('High stability')
        elif pd.notna(selected) and pd.notna(overall):
            selected_stddev = pd.to_numeric(pd.Series([row.get('selected_stddev')]), errors='coerce').iloc[0]
            overall_stddev = pd.to_numeric(pd.Series([row.get('overall_stddev')]), errors='coerce').iloc[0]
            lower, upper = _numeric_confidence_interval(float(selected), float(selected_stddev) if pd.notna(selected_stddev) else None, cohort_size)
            pop_lower, pop_upper = _numeric_confidence_interval(float(overall), float(overall_stddev) if pd.notna(overall_stddev) else None, population_size)
            selected_lower.append(None)
            selected_upper.append(None)
            overall_lower.append(None)
            overall_upper.append(None)
            selected_numeric_lower.append(lower)
            selected_numeric_upper.append(upper)
            overall_numeric_lower.append(pop_lower)
            overall_numeric_upper.append(pop_upper)
            p_value, significance, effect_label = _welch_numeric_test(
                float(selected),
                float(selected_stddev) if pd.notna(selected_stddev) else None,
                cohort_size,
                float(overall),
                float(overall_stddev) if pd.notna(overall_stddev) else None,
                population_size,
            )
            p_values.append(p_value)
            significance_labels.append(significance)
            effect_labels.append(effect_label)
            test_names.append('Approximate Welch test')
            width = (upper - lower) if lower is not None and upper is not None else None
            if cohort_size < 15:
                bands.append('Low stability')
            elif width is not None and width > 8.0:
                bands.append('Directional only')
            elif width is not None and width > 3.0:
                bands.append('Moderate stability')
            else:
                bands.append('High stability')
        else:
            selected_lower.append(None)
            selected_upper.append(None)
            overall_lower.append(None)
            overall_upper.append(None)
            selected_numeric_lower.append(None)
            selected_numeric_upper.append(None)
            overall_numeric_lower.append(None)
            overall_numeric_upper.append(None)
            p_values.append(None)
            significance_labels.append('Not tested')
            effect_labels.append('Not tested')
            test_names.append('Not tested')
            bands.append('Not interval-scored')
    frame['selected_confidence_lower'] = selected_lower
    frame['selected_confidence_upper'] = selected_upper
    frame['overall_confidence_lower'] = overall_lower
    frame['overall_confidence_upper'] = overall_upper
    frame['selected_numeric_confidence_lower'] = selected_numeric_lower
    frame['selected_numeric_confidence_upper'] = selected_numeric_upper
    frame['overall_numeric_confidence_lower'] = overall_numeric_lower
    frame['overall_numeric_confidence_upper'] = overall_numeric_upper
    frame['stability_band'] = bands
    frame['delta_p_value'] = p_values
    frame['delta_significance'] = significance_labels
    frame['delta_effect_size'] = effect_labels
    frame['statistical_test'] = test_names
    frame['cohort_record_count'] = int(cohort_size)
    frame['population_record_count'] = int(population_size)
    return frame


def build_field_uncertainty_summary(
    semantic: dict[str, Any],
    remediation_context: dict[str, Any],
) -> pd.DataFrame:
    canonical_map = semantic.get('canonical_map', {})
    helper_fields = _safe_df(remediation_context.get('helper_fields'))
    helper_type_lookup: dict[str, str] = {}
    if not helper_fields.empty and 'helper_field' in helper_fields.columns:
        helper_type_lookup = {
            str(row['helper_field']): str(row.get('helper_type', 'derived'))
            for _, row in helper_fields.iterrows()
        }

    rows: list[dict[str, object]] = []
    for canonical_field, source_column in canonical_map.items():
        helper_type = helper_type_lookup.get(str(source_column), '')
        support_type = 'native'
        confidence = 1.0
        if helper_type == 'inferred':
            support_type = 'inferred'
            confidence = 0.78
        elif helper_type == 'synthetic':
            support_type = 'synthetic'
            confidence = 0.58
        elif helper_type == 'derived':
            support_type = 'derived'
            confidence = 0.72
        rows.append(
            {
                'canonical_field': str(canonical_field),
                'source_column': str(source_column),
                'support_type': support_type,
                'field_confidence_score': confidence,
                'confidence_note': (
                    'Mapped directly from the uploaded source.'
                    if support_type == 'native'
                    else 'Mapped from an inferred field pattern.'
                    if support_type == 'inferred'
                    else 'Produced by a synthetic helper.'
                    if support_type == 'synthetic'
                    else 'Derived from existing source fields.'
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ['field_confidence_score', 'canonical_field'],
        ascending=[False, True],
    ).reset_index(drop=True)


def _support_multiplier(label: str) -> float:
    normalized = str(label).strip().lower()
    if normalized == 'native':
        return 1.0
    if normalized == 'derived':
        return 0.82
    if normalized == 'inferred':
        return 0.74
    if normalized == 'synthetic-assisted':
        return 0.62
    if normalized == 'synthetic':
        return 0.58
    return 0.5


def build_metric_confidence_table(
    *,
    overview: dict[str, Any],
    readiness: dict[str, Any],
    healthcare: dict[str, Any],
    remediation_context: dict[str, Any],
    trust_summary: dict[str, Any],
    reporting_policy: dict[str, Any],
) -> pd.DataFrame:
    trust_score = float(trust_summary.get('trust_score', 0.0) or 0.0)
    sampling_active = bool(trust_summary.get('sampling_active', False))
    rows: list[dict[str, object]] = [
        {
            'metric': 'Analysis readiness',
            'value': float(readiness.get('readiness_score', 0.0) or 0.0),
            'support_type': 'native',
            'metric_confidence_score': round(_clamp(trust_score * 1.0), 2),
            'external_reporting_safe': trust_score >= float(reporting_policy.get('minimum_trust_score', 0.76)),
            'confidence_note': 'Pipeline readiness is computed directly from detected field coverage and module prerequisites.',
        },
        {
            'metric': 'Healthcare readiness',
            'value': float(healthcare.get('healthcare_readiness_score', 0.0) or 0.0),
            'support_type': 'native',
            'metric_confidence_score': round(_clamp(trust_score * 0.95), 2),
            'external_reporting_safe': trust_score >= float(reporting_policy.get('minimum_trust_score', 0.76)),
            'confidence_note': 'Healthcare readiness reflects detected domain coverage, not clinical outcome truth.',
        },
    ]

    readmission = healthcare.get('readmission', {})
    if readmission.get('available'):
        source = str(readmission.get('source', 'native')).lower()
        support_type = 'derived' if source == 'derived' else 'synthetic-assisted' if source == 'synthetic' else 'native'
        rows.append(
            {
                'metric': 'Overall readmission rate',
                'value': float(readmission.get('overview', {}).get('overall_readmission_rate', 0.0) or 0.0),
                'support_type': support_type,
                'metric_confidence_score': round(_clamp(trust_score * _support_multiplier(support_type)), 2),
                'external_reporting_safe': _reporting_safe(
                    support_type=support_type,
                    trust_score=trust_score,
                    sampling_active=sampling_active,
                    policy=reporting_policy,
                ),
                'confidence_note': 'Readmission confidence is reduced when the workflow is derived, synthetic-assisted, or sampled.',
            }
        )

    risk = healthcare.get('risk_segmentation', {})
    risk_table = _safe_df(risk.get('segment_table'))
    if risk.get('available') and not risk_table.empty:
        high_risk = risk_table[risk_table.get('risk_segment', pd.Series(dtype=str)).astype(str) == 'High Risk']
        if not high_risk.empty:
            support_type = 'synthetic-assisted' if remediation_context.get('synthetic_clinical', {}).get('available') else 'native'
            rows.append(
                {
                    'metric': 'High-risk population share',
                    'value': float(high_risk.iloc[0].get('percentage', 0.0) or 0.0),
                    'support_type': support_type,
                    'metric_confidence_score': round(_clamp(trust_score * _support_multiplier(support_type)), 2),
                    'external_reporting_safe': _reporting_safe(
                        support_type=support_type,
                        trust_score=trust_score,
                        sampling_active=sampling_active,
                        policy=reporting_policy,
                    ),
                    'confidence_note': 'Risk segmentation confidence depends on native clinical support versus derived helper coverage.',
                }
            )

    cost = healthcare.get('cost', {})
    if cost.get('available'):
        support_type = 'synthetic-assisted' if remediation_context.get('synthetic_cost', {}).get('available') else 'native'
        rows.append(
            {
                'metric': 'Average cost',
                'value': float(cost.get('summary', {}).get('average_cost', 0.0) or 0.0),
                'support_type': support_type,
                'metric_confidence_score': round(_clamp(trust_score * _support_multiplier(support_type)), 2),
                'external_reporting_safe': _reporting_safe(
                    support_type=support_type,
                    trust_score=trust_score,
                    sampling_active=sampling_active,
                    policy=reporting_policy,
                ),
                'confidence_note': 'Cost confidence is reduced when financial fields are synthetic or estimated.',
            }
        )
    return pd.DataFrame(rows)


def build_benchmark_calibration_summary(
    *,
    dataset_name: str,
    source_meta: dict[str, Any],
    overview: dict[str, Any],
    healthcare: dict[str, Any],
    remediation_context: dict[str, Any],
    active_control_values: dict[str, Any],
) -> pd.DataFrame:
    profile = resolve_benchmark_profile(
        dataset_name=dataset_name,
        source_meta=source_meta,
        healthcare=healthcare,
        active_control_values=active_control_values,
    )
    rate_bands = profile.get('rate_bands', {})
    numeric_bands = profile.get('numeric_bands', {})
    rows: list[dict[str, object]] = []
    readmission = healthcare.get('readmission', {})
    if readmission.get('available'):
        rate = float(readmission.get('overview', {}).get('overall_readmission_rate', 0.0) or 0.0)
        lower, upper = rate_bands.get('Readmission rate', (0.05, 0.35))
        rows.append(
            {
                'metric': 'Readmission rate',
                'observed_value': rate,
                'reference_band': f'{lower:.0%} to {upper:.0%}',
                'calibration_status': 'Outside expected range' if rate < lower or rate > upper else 'Within expected range',
                'detail': str(profile.get('detail_note', '')),
                'benchmark_profile': profile.get('profile_name', 'Generic Healthcare'),
            }
        )
    los = healthcare.get('length_of_stay_prediction', {})
    if los.get('available'):
        avg_los = float(los.get('summary', {}).get('average_actual_length_of_stay', 0.0) or 0.0)
        lower, upper = numeric_bands.get('Average length of stay', (1.0, 12.0))
        rows.append(
            {
                'metric': 'Average length of stay',
                'observed_value': avg_los,
                'reference_band': f'{lower:.1f} to {upper:.1f} days',
                'calibration_status': 'Outside expected range' if avg_los < lower or avg_los > upper else 'Within expected range',
                'detail': 'Longer average stays can be valid but should be checked against facility and service-line context.',
                'benchmark_profile': profile.get('profile_name', 'Generic Healthcare'),
            }
        )
    cost = healthcare.get('cost', {})
    if cost.get('available'):
        avg_cost = float(cost.get('summary', {}).get('average_cost', 0.0) or 0.0)
        lower, upper = numeric_bands.get('Average cost', (500.0, 100000.0))
        rows.append(
            {
                'metric': 'Average cost',
                'observed_value': avg_cost,
                'reference_band': f'${lower:,.0f} to ${upper:,.0f}',
                'calibration_status': 'Outside expected range' if avg_cost < lower or avg_cost > upper else 'Within expected range',
                'detail': (
                    'Synthetic-assisted cost support should be treated as directional only.'
                    if remediation_context.get('synthetic_cost', {}).get('available')
                    else 'Validate against source-system financial extracts for external reporting.'
                ),
                'benchmark_profile': profile.get('profile_name', 'Generic Healthcare'),
            }
        )
    risk = healthcare.get('risk_segmentation', {})
    risk_table = _safe_df(risk.get('segment_table'))
    if risk.get('available') and not risk_table.empty:
        high_risk = risk_table[risk_table.get('risk_segment', pd.Series(dtype=str)).astype(str) == 'High Risk']
        if not high_risk.empty:
            share = float(high_risk.iloc[0].get('percentage', 0.0) or 0.0)
            lower, upper = rate_bands.get('High-risk share', (0.05, 0.50))
            rows.append(
                {
                    'metric': 'High-risk share',
                    'observed_value': share,
                    'reference_band': f'{lower:.0%} to {upper:.0%}',
                    'calibration_status': 'Outside expected range' if share < lower or share > upper else 'Within expected range',
                    'detail': 'High-risk prevalence should be validated against internal cohort expectations before broad reporting.',
                    'benchmark_profile': profile.get('profile_name', 'Generic Healthcare'),
                }
            )
    if not rows:
        rows.append(
            {
                'metric': 'Calibration status',
                'observed_value': float(overview.get('rows', 0) or 0),
                'reference_band': 'No benchmark-calibrated metrics available',
                'calibration_status': 'Limited',
                'detail': 'Add native outcome, utilization, or cost fields to unlock richer benchmark calibration.',
                'benchmark_profile': profile.get('profile_name', 'Generic Healthcare'),
            }
        )
    return pd.DataFrame(rows)


def build_module_reporting_gates(
    capability_matrix: pd.DataFrame,
    *,
    trust_summary: dict[str, Any],
    reporting_policy: dict[str, Any],
) -> pd.DataFrame:
    matrix = _safe_df(capability_matrix).copy()
    if matrix.empty:
        return pd.DataFrame(columns=['analytics_module', 'reporting_gate', 'detail'])
    trust_score = float(trust_summary.get('trust_score', 0.0) or 0.0)
    sampling_active = bool(trust_summary.get('sampling_active', False))
    timeout_fallback = bool(trust_summary.get('timeout_fallback', False))
    minimum_trust = float(reporting_policy.get('minimum_trust_score', 0.76))
    allow_directional = bool(reporting_policy.get('allow_directional_external_reporting', False))
    gate_rows: list[dict[str, object]] = []
    for _, row in matrix.iterrows():
        module = str(row.get('analytics_module', 'Unknown module'))
        status = str(row.get('status', 'blocked')).lower()
        support = str(row.get('support', 'unavailable')).lower()
        if status == 'blocked':
            gate = 'Blocked'
            detail = 'This module is not currently supportable with the uploaded dataset.'
        elif timeout_fallback and module in {'Readmission Analytics', 'Trend Analysis', 'Cohort Monitoring Over Time'}:
            gate = 'Restricted'
            detail = 'Timeout fallback was used, so this module should not be used for external reporting.'
        elif support in {'synthetic-assisted', 'synthetic', 'inferred'}:
            if allow_directional and trust_score >= max(0.60, minimum_trust - 0.08):
                gate = 'Safe with disclosure'
                detail = 'This module uses helper-assisted support, but the active reporting policy allows directional external use with explicit disclosure.'
            else:
                gate = 'Directional only'
                detail = 'This module depends on helper-assisted support and should be treated as directional for external use.'
        elif sampling_active and module in {'Trend Analysis', 'Cohort Analysis', 'Readmission Analytics', 'Clinical Outcome Benchmarks'}:
            if allow_directional and trust_score >= max(0.60, minimum_trust - 0.08):
                gate = 'Safe with disclosure'
                detail = 'Sampling is active, and the current reporting policy allows disclosed directional external use for this module.'
            else:
                gate = 'Directional only'
                detail = 'Sampling is active, so external reporting should cite source-row context and caveats.'
        elif trust_score < minimum_trust:
            gate = 'Internal only'
            detail = 'Overall trust is still moderate-to-low, so external reporting is not recommended yet.'
        else:
            gate = 'Safe with disclosure'
            detail = 'This module is suitable for external handoff if normal governance and disclosure notes are included.'
        gate_rows.append(
            {
                'analytics_module': module,
                'reporting_gate': gate,
                'detail': detail,
                'support_type': str(row.get('support', 'unavailable')),
                'module_status': str(row.get('status', 'blocked')),
                'minimum_trust_score': minimum_trust,
                'directional_external_allowed': allow_directional,
            }
        )
    return pd.DataFrame(gate_rows)


def build_metric_lineage_table(
    metric_confidence_table: pd.DataFrame,
    module_reporting_gates: pd.DataFrame,
) -> pd.DataFrame:
    metrics = _safe_df(metric_confidence_table)
    gates = _safe_df(module_reporting_gates)
    if metrics.empty:
        return pd.DataFrame(columns=['metric', 'analytics_module', 'lineage_type', 'support_type', 'driving_fields', 'reporting_gate'])
    gate_lookup = {
        str(row.get('analytics_module', '')): str(row.get('reporting_gate', ''))
        for _, row in gates.iterrows()
    } if not gates.empty else {}
    rows: list[dict[str, Any]] = []
    for _, row in metrics.iterrows():
        metric = str(row.get('metric', ''))
        lineage = MODULE_METRIC_LINEAGE.get(
            metric,
            {
                'analytics_module': 'General Analytics',
                'driving_fields': [],
                'lineage_type': str(row.get('support_type', 'derived') or 'derived').replace('-assisted', ''),
            },
        )
        module_name = str(lineage.get('analytics_module', 'General Analytics'))
        rows.append(
            {
                'metric': metric,
                'analytics_module': module_name,
                'lineage_type': str(lineage.get('lineage_type', 'derived')),
                'support_type': str(row.get('support_type', 'derived')),
                'driving_fields': ', '.join(str(item) for item in lineage.get('driving_fields', [])),
                'reporting_gate': gate_lookup.get(module_name, 'Not scored'),
                'metric_confidence_score': float(row.get('metric_confidence_score', 0.0) or 0.0),
                'external_reporting_safe': bool(row.get('external_reporting_safe')),
            }
        )
    return pd.DataFrame(rows)


def build_approval_workflow_summary(
    *,
    dataset_identifier: str,
    active_control_values: dict[str, Any],
    reporting_policy: dict[str, Any],
) -> dict[str, Any]:
    approval_store = active_control_values.get('dataset_review_approvals', {})
    approval_store = approval_store if isinstance(approval_store, dict) else {}
    review_state = approval_store.get(dataset_identifier, {})
    review_state = review_state if isinstance(review_state, dict) else {}
    mapping_status = str(review_state.get('mapping_status', 'Pending'))
    trust_status = str(review_state.get('trust_gate_status', 'Pending'))
    export_status = str(review_state.get('export_eligibility_status', 'Pending'))
    approved = mapping_status == 'Approved' and trust_status == 'Approved' and export_status == 'Approved'
    return {
        'dataset_identifier': dataset_identifier,
        'mapping_status': mapping_status,
        'trust_gate_status': trust_status,
        'export_eligibility_status': export_status,
        'approved_for_release': approved,
        'review_notes': str(review_state.get('review_notes', '')),
        'reviewed_by_role': str(review_state.get('reviewed_by_role', active_control_values.get('active_role', 'Analyst'))),
        'reporting_threshold_profile': str(reporting_policy.get('profile_name', 'Standard')),
    }


def _keyword_weight(title: str, rationale: str, *, healthcare: dict[str, Any], remediation_context: dict[str, Any]) -> tuple[float, str]:
    text = f'{title} {rationale}'.lower()
    if 'readmission' in text:
        source = str(healthcare.get('readmission', {}).get('source', 'native')).lower()
        support = 'derived' if source == 'derived' else 'synthetic-assisted' if source == 'synthetic' else 'native'
        return _support_multiplier(support), support
    if 'cost' in text or 'financial' in text or 'spend' in text:
        support = 'synthetic-assisted' if remediation_context.get('synthetic_cost', {}).get('available') else 'native'
        return _support_multiplier(support), support
    if 'diagnosis' in text or 'clinical' in text or 'risk' in text or 'cohort' in text:
        support = 'synthetic-assisted' if remediation_context.get('synthetic_clinical', {}).get('available') else 'native'
        return _support_multiplier(support), support
    return 1.0, 'native'


def weight_recommendation_frame(
    frame: pd.DataFrame,
    *,
    title_col: str,
    rationale_col: str,
    base_priority_col: str,
    healthcare: dict[str, Any],
    remediation_context: dict[str, Any],
) -> pd.DataFrame:
    weighted = _safe_df(frame).copy()
    if weighted.empty:
        return weighted
    weights: list[float] = []
    support_labels: list[str] = []
    weighted_scores: list[float] = []
    external_safe: list[bool] = []
    for _, row in weighted.iterrows():
        weight, support = _keyword_weight(
            str(row.get(title_col, '')),
            str(row.get(rationale_col, '')),
            healthcare=healthcare,
            remediation_context=remediation_context,
        )
        base = float(pd.to_numeric(pd.Series([row.get(base_priority_col)]), errors='coerce').fillna(0.0).iloc[0])
        weights.append(weight)
        support_labels.append(support)
        weighted_scores.append(round(base * weight, 2))
        external_safe.append(support == 'native')
    weighted['native_coverage_weight'] = weights
    weighted['support_basis'] = support_labels
    weighted['weighted_priority_score'] = weighted_scores
    weighted['external_reporting_safe'] = external_safe
    return weighted.sort_values(
        ['weighted_priority_score', base_priority_col],
        ascending=[False, False],
    ).reset_index(drop=True)


def build_result_accuracy_summary(
    *,
    dataset_name: str,
    source_meta: dict[str, Any],
    overview: dict[str, Any],
    semantic: dict[str, Any],
    readiness: dict[str, Any],
    healthcare: dict[str, Any],
    remediation_context: dict[str, Any],
    dataset_intelligence: dict[str, Any],
    trust_summary: dict[str, Any],
    active_control_values: dict[str, Any],
) -> dict[str, Any]:
    dataset_identifier = str(source_meta.get('dataset_cache_key', '') or source_meta.get('dataset_identifier', '') or dataset_name)
    benchmark_profile = resolve_benchmark_profile(
        dataset_name=dataset_name,
        source_meta=source_meta,
        healthcare=healthcare,
        active_control_values=active_control_values,
    )
    reporting_policy = resolve_reporting_threshold_policy(active_control_values)
    field_uncertainty_table = build_field_uncertainty_summary(semantic, remediation_context)
    metric_confidence_table = build_metric_confidence_table(
        overview=overview,
        readiness=readiness,
        healthcare=healthcare,
        remediation_context=remediation_context,
        trust_summary=trust_summary,
        reporting_policy=reporting_policy,
    )
    benchmark_calibration_table = build_benchmark_calibration_summary(
        dataset_name=dataset_name,
        source_meta=source_meta,
        overview=overview,
        healthcare=healthcare,
        remediation_context=remediation_context,
        active_control_values=active_control_values,
    )
    module_reporting_gates = build_module_reporting_gates(
        _safe_df(dataset_intelligence.get('analytics_capability_matrix')),
        trust_summary=trust_summary,
        reporting_policy=reporting_policy,
    )
    metric_lineage_table = build_metric_lineage_table(metric_confidence_table, module_reporting_gates)
    approval_workflow = build_approval_workflow_summary(
        dataset_identifier=dataset_identifier,
        active_control_values=active_control_values,
        reporting_policy=reporting_policy,
    )
    safe_modules = module_reporting_gates[
        module_reporting_gates.get('reporting_gate', pd.Series(dtype=str)).astype(str) == 'Safe with disclosure'
    ]['analytics_module'].astype(str).tolist() if not module_reporting_gates.empty else []
    restricted_modules = module_reporting_gates[
        module_reporting_gates.get('reporting_gate', pd.Series(dtype=str)).astype(str).isin(['Directional only', 'Restricted', 'Blocked', 'Internal only'])
    ]['analytics_module'].astype(str).tolist() if not module_reporting_gates.empty else []
    narrative = build_uncertainty_narrative(
        trust_summary=trust_summary,
        metric_confidence_table=metric_confidence_table,
        module_reporting_gates=module_reporting_gates,
        benchmark_profile=benchmark_profile,
        reporting_policy=reporting_policy,
    )
    return {
        'available': True,
        'field_uncertainty_table': field_uncertainty_table,
        'metric_confidence_table': metric_confidence_table,
        'benchmark_calibration_table': benchmark_calibration_table,
        'module_reporting_gates': module_reporting_gates,
        'metric_lineage_table': metric_lineage_table,
        'benchmark_profile': benchmark_profile,
        'reporting_policy': reporting_policy,
        'approval_workflow': approval_workflow,
        'uncertainty_narrative': narrative,
        'safe_for_external_reporting_modules': safe_modules,
        'restricted_reporting_modules': restricted_modules,
    }


def build_uncertainty_narrative(
    *,
    trust_summary: dict[str, Any],
    metric_confidence_table: pd.DataFrame,
    module_reporting_gates: pd.DataFrame,
    benchmark_profile: dict[str, Any],
    reporting_policy: dict[str, Any],
) -> dict[str, Any]:
    trust_level = str(trust_summary.get('trust_level', 'Unknown'))
    interpretation_mode = str(trust_summary.get('interpretation_mode', 'Unknown'))
    sampled = bool(trust_summary.get('sampling_active', False))
    helper_sensitive = _safe_df(metric_confidence_table)
    helper_metrics = helper_sensitive[
        helper_sensitive.get('support_type', pd.Series(dtype=str)).astype(str).isin(['derived', 'inferred', 'synthetic-assisted', 'synthetic'])
    ]['metric'].astype(str).tolist() if not helper_sensitive.empty else []
    restricted = _safe_df(module_reporting_gates)
    restricted_modules = restricted[
        restricted.get('reporting_gate', pd.Series(dtype=str)).astype(str).isin(['Directional only', 'Restricted', 'Blocked', 'Internal only'])
    ]['analytics_module'].astype(str).tolist() if not restricted.empty else []
    notes: list[str] = [
        f"Result trust is currently {trust_level.lower()} and the workflow is operating in {interpretation_mode.lower()} mode.",
        f"Benchmark calibration is using the {benchmark_profile.get('profile_name', 'Generic Healthcare')} profile.",
        (
            f"External reporting is governed by the {reporting_policy.get('profile_name', 'Standard')} threshold profile "
            f"with a minimum trust score of {float(reporting_policy.get('minimum_trust_score', 0.76)):.0%}."
        ),
    ]
    if sampled:
        notes.append('Sampling is active, so longitudinal and subgroup outputs should be treated as directional unless source-grade counts are confirmed.')
    if helper_metrics:
        notes.append(f"Metrics with non-native support include: {', '.join(helper_metrics[:4])}.")
    if restricted_modules:
        notes.append(f"Modules still restricted for broader sharing include: {', '.join(restricted_modules[:4])}.")
    return {
        'headline': notes[0],
        'notes': notes,
        'narrative_prefix': 'Directional signal:' if sampled or restricted_modules else 'Decision-ready signal:',
    }
