from __future__ import annotations

from typing import Any

import pandas as pd


BMI_MIN = 10.0
BMI_MAX = 80.0
BMI_PRACTICAL_MIN = 18.5
BMI_PRACTICAL_MAX = 49.9
BMI_OUTLIER_MIN = 12.0
BMI_OUTLIER_MAX = 60.0

BMI_HIGH_CONFIDENCE = 0.95
BMI_MEDIUM_CONFIDENCE = 0.85
SECONDARY_DX_HIGH_CONFIDENCE = 0.95
SECONDARY_DX_MEDIUM_CONFIDENCE = 0.85

OBESITY_DIAGNOSIS_PREFIXES = ('E66',)
OBESITY_KEYWORDS = ('obesity', 'morbid obesity', 'overweight')


def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors='coerce')


def _distribution_summary(series: pd.Series) -> dict[str, float | int | None]:
    numeric = _safe_numeric(series).dropna()
    if numeric.empty:
        return {
            'count': 0,
            'min': None,
            'max': None,
            'mean': None,
            'median': None,
        }
    return {
        'count': int(len(numeric)),
        'min': float(numeric.min()),
        'max': float(numeric.max()),
        'mean': float(numeric.mean()),
        'median': float(numeric.median()),
    }


def _find_column(frame: pd.DataFrame, candidates: list[str]) -> str | None:
    normalized_lookup = {str(column).strip().lower(): str(column) for column in frame.columns}
    for candidate in candidates:
        if candidate in normalized_lookup:
            return normalized_lookup[candidate]
    return None


def _normalize_height_meters(frame: pd.DataFrame) -> tuple[pd.Series, str | None]:
    height_m_col = _find_column(frame, ['height_m', 'height_meters', 'height_meter'])
    if height_m_col:
        return _safe_numeric(frame[height_m_col]), height_m_col
    height_cm_col = _find_column(frame, ['height_cm', 'height_centimeters', 'height'])
    if height_cm_col:
        return _safe_numeric(frame[height_cm_col]) / 100.0, height_cm_col
    return pd.Series(float('nan'), index=frame.index, dtype='float64'), None


def _normalize_weight_kg(frame: pd.DataFrame) -> tuple[pd.Series, str | None]:
    weight_kg_col = _find_column(frame, ['weight_kg', 'weight_kilograms', 'weight'])
    if weight_kg_col:
        return _safe_numeric(frame[weight_kg_col]), weight_kg_col
    weight_lb_col = _find_column(frame, ['weight_lb', 'weight_lbs', 'weight_pounds'])
    if weight_lb_col:
        return _safe_numeric(frame[weight_lb_col]) * 0.45359237, weight_lb_col
    return pd.Series(float('nan'), index=frame.index, dtype='float64'), None


def _derive_bmi_from_vitals(frame: pd.DataFrame) -> tuple[pd.Series, list[str]]:
    height_m, height_source = _normalize_height_meters(frame)
    weight_kg, weight_source = _normalize_weight_kg(frame)
    derived = pd.Series(float('nan'), index=frame.index, dtype='float64')
    valid = height_m.gt(0) & weight_kg.gt(0)
    derived.loc[valid] = weight_kg.loc[valid] / (height_m.loc[valid] ** 2)
    sources = [source for source in [height_source, weight_source] if source]
    return derived, sources


def _obesity_signal(frame: pd.DataFrame) -> pd.Series:
    diagnosis_code_col = _find_column(frame, ['diagnosis_code', 'dx_code', 'icd10', 'icd_10'])
    diagnosis_label_col = _find_column(frame, ['diagnosis_label', 'primary_diagnosis', 'secondary_diagnosis_label'])
    code_signal = pd.Series(False, index=frame.index)
    label_signal = pd.Series(False, index=frame.index)
    if diagnosis_code_col:
        codes = frame[diagnosis_code_col].astype(str).str.upper().str.strip()
        code_signal = codes.str.startswith(OBESITY_DIAGNOSIS_PREFIXES, na=False)
    if diagnosis_label_col:
        labels = frame[diagnosis_label_col].astype(str).str.lower().str.strip()
        label_signal = labels.str.contains('|'.join(OBESITY_KEYWORDS), regex=True, na=False)
    return code_signal | label_signal


def _confidence_label(score: float) -> str:
    if score >= BMI_HIGH_CONFIDENCE:
        return 'High'
    if score >= BMI_MEDIUM_CONFIDENCE:
        return 'Medium'
    return 'Low'


def _clinical_average_bmi(frame: pd.DataFrame) -> pd.Series:
    age_col = _find_column(frame, ['age', 'age_years'])
    gender_col = _find_column(frame, ['gender', 'sex'])
    bmi_col = _find_column(frame, ['bmi', 'bmi_remediated_value', 'body_mass_index'])
    if not bmi_col:
        return pd.Series(27.5, index=frame.index, dtype='float64')

    numeric_bmi = _safe_numeric(frame[bmi_col])
    valid_bmi = numeric_bmi.where(numeric_bmi.between(BMI_PRACTICAL_MIN, BMI_PRACTICAL_MAX, inclusive='both'))
    if age_col and gender_col:
        grouped = pd.DataFrame(
            {
                'age': _safe_numeric(frame[age_col]),
                'gender': frame[gender_col].astype(str).fillna('Unknown'),
                'bmi': valid_bmi,
            }
        )
        grouped['age_band'] = pd.cut(grouped['age'], bins=[-1, 17, 34, 49, 64, 200], labels=['0-17', '18-34', '35-49', '50-64', '65+'])
        lookup = grouped.groupby(['age_band', 'gender'], observed=False)['bmi'].median().reset_index()
        merged = grouped[['age_band', 'gender']].merge(lookup, on=['age_band', 'gender'], how='left')
        fallback = merged['bmi'].fillna(valid_bmi.median())
        return pd.Series(fallback, index=frame.index, dtype='float64').fillna(27.5)
    fallback_value = float(valid_bmi.median()) if valid_bmi.notna().any() else 27.5
    return pd.Series(fallback_value, index=frame.index, dtype='float64')


def _temporal_validation_flags(frame: pd.DataFrame, remediated_bmi: pd.Series) -> pd.Series:
    patient_col = _find_column(frame, ['patient_id', 'member_id', 'entity_id'])
    date_col = _find_column(frame, ['admission_date', 'service_date', 'event_date', 'diagnosis_date', 'discharge_date'])
    flags = pd.Series(False, index=frame.index)
    if not patient_col or not date_col:
        return flags
    working = pd.DataFrame(
        {
            'patient_id': frame[patient_col],
            'event_date': pd.to_datetime(frame[date_col], errors='coerce'),
            'bmi_value': remediated_bmi,
        },
        index=frame.index,
    ).dropna(subset=['patient_id', 'event_date', 'bmi_value'])
    if working.empty:
        return flags
    working = working.sort_values(['patient_id', 'event_date'])
    working['prior_bmi'] = working.groupby('patient_id')['bmi_value'].shift(1)
    working['temporal_jump_flag'] = (working['bmi_value'] - working['prior_bmi']).abs() > 5.0
    flags.loc[working.index] = working['temporal_jump_flag'].fillna(False)
    return flags


def _build_synthetic_cost_estimate(frame: pd.DataFrame, bmi_values: pd.Series) -> pd.Series:
    age = _safe_numeric(frame['age']) if 'age' in frame.columns else pd.Series(0.0, index=frame.index)
    smoking = frame['smoking_history'] if 'smoking_history' in frame.columns else frame.get('smoking_status', pd.Series('', index=frame.index))
    smoking_text = smoking.astype(str).str.strip().str.lower()
    hypertension = _safe_numeric(frame['hypertension']) if 'hypertension' in frame.columns else pd.Series(0.0, index=frame.index)
    heart_disease = _safe_numeric(frame['heart_disease']) if 'heart_disease' in frame.columns else pd.Series(0.0, index=frame.index)
    diabetes_signal = _safe_numeric(frame['diabetes']) if 'diabetes' in frame.columns else pd.Series(0.0, index=frame.index)
    if 'readmission_risk_proxy' in frame.columns:
        diabetes_signal = diabetes_signal.combine_first((_safe_numeric(frame['readmission_risk_proxy']) > 0.6).astype(float))

    bmi_component = pd.Series(0.0, index=frame.index)
    bmi_component.loc[bmi_values >= 25] = 150.0
    bmi_component.loc[bmi_values >= 30] = 350.0
    bmi_component.loc[bmi_values >= 40] = 550.0

    smoking_component = pd.Series(0.0, index=frame.index)
    smoking_component.loc[smoking_text.str.contains('former', na=False)] = 250.0
    smoking_component.loc[smoking_text.str.contains('current|smoker|yes', na=False)] = 500.0

    return (
        500.0
        + age.fillna(age.median()) * 8.0
        + bmi_component
        + smoking_component
        + (hypertension.fillna(0) > 0).astype(float) * 400.0
        + (heart_disease.fillna(0) > 0).astype(float) * 700.0
        + (diabetes_signal.fillna(0) > 0).astype(float) * 300.0
    ).clip(lower=500.0, upper=7500.0).round(2)


def _build_synthetic_readmission_proxy(frame: pd.DataFrame, bmi_values: pd.Series) -> pd.Series:
    age = _safe_numeric(frame['age']) if 'age' in frame.columns else pd.Series(0.0, index=frame.index)
    glucose = _safe_numeric(frame['blood_glucose_level']) if 'blood_glucose_level' in frame.columns else pd.Series(float('nan'), index=frame.index)
    smoking = frame['smoking_history'] if 'smoking_history' in frame.columns else frame.get('smoking_status', pd.Series('', index=frame.index))
    smoking_text = smoking.astype(str).str.strip().str.lower()
    hypertension = _safe_numeric(frame['hypertension']) if 'hypertension' in frame.columns else pd.Series(0.0, index=frame.index)
    heart_disease = _safe_numeric(frame['heart_disease']) if 'heart_disease' in frame.columns else pd.Series(0.0, index=frame.index)

    proxy = pd.Series(0.08, index=frame.index, dtype='float64')
    proxy += (age.fillna(age.median()) >= 65).astype(float) * 0.12
    proxy += (bmi_values >= 30).astype(float) * 0.08
    proxy += smoking_text.str.contains('current|smoker|yes', na=False).astype(float) * 0.10
    proxy += (hypertension.fillna(0) > 0).astype(float) * 0.08
    proxy += (heart_disease.fillna(0) > 0).astype(float) * 0.14
    if glucose.notna().any():
        proxy += (glucose.fillna(glucose.median()) >= 180).astype(float) * 0.10
    if 'risk_score' in frame.columns:
        proxy += (_safe_numeric(frame['risk_score']).fillna(0) / 10.0).clip(upper=0.15)
    return proxy.clip(lower=0.03, upper=0.65).round(4)


def _safe_string(series: pd.Series) -> pd.Series:
    text = series.astype('string').str.strip()
    return text.mask(text.eq('')).replace({'nan': pd.NA, 'None': pd.NA, '<NA>': pd.NA})


def _mode_lookup(frame: pd.DataFrame, key_column: str, value_column: str) -> pd.Series:
    working = frame[[key_column, value_column]].copy()
    working[key_column] = _safe_string(working[key_column])
    working[value_column] = _safe_string(working[value_column])
    working = working.dropna(subset=[key_column, value_column])
    if working.empty:
        return pd.Series(dtype='object')
    return (
        working.groupby(key_column, observed=False)[value_column]
        .agg(lambda values: values.mode().iloc[0] if not values.mode().empty else values.iloc[0])
    )


def remediate_secondary_diagnosis(data: pd.DataFrame, mode: str = 'default') -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = data.copy()
    secondary_column = _find_column(frame, ['secondary_diagnosis_label', 'secondary_diagnosis', 'secondary_dx_label'])
    if not secondary_column:
        return frame, {
            'available': False,
            'reason': 'Secondary diagnosis field is not available for remediation.',
            'remediation_mode': mode,
        }

    original = _safe_string(frame[secondary_column])
    remediated = original.copy()
    patient_col = _find_column(frame, ['patient_id', 'member_id', 'entity_id'])
    date_col = _find_column(frame, ['admission_date', 'service_date', 'event_date', 'diagnosis_date', 'discharge_date'])
    diagnosis_code_col = _find_column(frame, ['diagnosis_code', 'dx_code', 'icd10', 'icd_10'])
    primary_diagnosis_col = _find_column(frame, ['diagnosis_label', 'primary_diagnosis', 'diagnosis'])

    method = pd.Series('original', index=frame.index, dtype='object')
    confidence_score = pd.Series(SECONDARY_DX_HIGH_CONFIDENCE, index=frame.index, dtype='float64')
    source_fields_used = pd.Series(secondary_column, index=frame.index, dtype='object')
    validation_rule = pd.Series('Original secondary diagnosis retained.', index=frame.index, dtype='object')
    flag_reason = pd.Series('', index=frame.index, dtype='object')

    missing_mask = remediated.isna()
    if patient_col:
        working = pd.DataFrame(
            {
                'patient_key': _safe_string(frame[patient_col]),
                'secondary_value': remediated,
                'event_date': pd.to_datetime(frame[date_col], errors='coerce') if date_col else pd.Series(pd.NaT, index=frame.index),
            },
            index=frame.index,
        )
        sort_columns = ['patient_key']
        if date_col:
            sort_columns.append('event_date')
        working = working.sort_values(sort_columns)
        working['forward_fill_value'] = working.groupby('patient_key', dropna=False)['secondary_value'].ffill()
        forward_fill_mask = missing_mask & working['forward_fill_value'].notna()
        remediated.loc[forward_fill_mask] = working.loc[forward_fill_mask, 'forward_fill_value']
        method.loc[forward_fill_mask] = 'forward_fill'
        confidence_score.loc[forward_fill_mask] = 0.90
        source_fields_used.loc[forward_fill_mask] = patient_col if not date_col else f'{patient_col} + {date_col}'
        validation_rule.loc[forward_fill_mask] = 'Forward fill within the same patient timeline.'
        missing_mask = remediated.isna()

    pattern_sources: list[str] = []
    if diagnosis_code_col:
        diagnosis_lookup = _mode_lookup(frame.assign(_secondary_value=original), diagnosis_code_col, '_secondary_value')
        if not diagnosis_lookup.empty:
            code_series = _safe_string(frame[diagnosis_code_col])
            code_fill_mask = missing_mask & code_series.notna() & code_series.map(diagnosis_lookup).notna()
            remediated.loc[code_fill_mask] = code_series.loc[code_fill_mask].map(diagnosis_lookup)
            method.loc[code_fill_mask] = 'diagnosis_code_pattern'
            confidence_score.loc[code_fill_mask] = 0.87
            source_fields_used.loc[code_fill_mask] = diagnosis_code_col
            validation_rule.loc[code_fill_mask] = 'Remapped from the most common secondary diagnosis for the same diagnosis code.'
            pattern_sources.append(diagnosis_code_col)
            missing_mask = remediated.isna()
    if primary_diagnosis_col:
        primary_lookup = _mode_lookup(frame.assign(_secondary_value=original), primary_diagnosis_col, '_secondary_value')
        if not primary_lookup.empty:
            primary_series = _safe_string(frame[primary_diagnosis_col])
            primary_fill_mask = missing_mask & primary_series.notna() & primary_series.map(primary_lookup).notna()
            remediated.loc[primary_fill_mask] = primary_series.loc[primary_fill_mask].map(primary_lookup)
            method.loc[primary_fill_mask] = 'primary_diagnosis_pattern'
            confidence_score.loc[primary_fill_mask] = 0.86
            source_fields_used.loc[primary_fill_mask] = primary_diagnosis_col
            validation_rule.loc[primary_fill_mask] = 'Remapped from the most common secondary diagnosis for the same primary diagnosis.'
            pattern_sources.append(primary_diagnosis_col)
            missing_mask = remediated.isna()

    default_mask = remediated.isna()
    remediated.loc[default_mask] = 'No Secondary Diagnosis'
    method.loc[default_mask] = 'default_no_secondary'
    confidence_score.loc[default_mask] = 0.76
    source_fields_used.loc[default_mask] = 'default'
    validation_rule.loc[default_mask] = 'No secondary diagnosis evidence was available, so a default label was applied.'

    if primary_diagnosis_col:
        primary_values = _safe_string(frame[primary_diagnosis_col])
        same_as_primary = remediated.notna() & primary_values.notna() & remediated.eq(primary_values)
        flag_reason.loc[same_as_primary] = 'Secondary diagnosis matches the primary diagnosis and should be reviewed.'
        confidence_score.loc[same_as_primary] = confidence_score.loc[same_as_primary].clip(upper=0.82)
        validation_rule.loc[same_as_primary] = 'Validation flagged a duplicate primary/secondary diagnosis pairing.'

    confidence_level = confidence_score.apply(_confidence_label)
    has_secondary_diagnosis = remediated.ne('No Secondary Diagnosis')
    manual_review_flag = flag_reason.ne('') | confidence_score.lt(SECONDARY_DX_MEDIUM_CONFIDENCE)

    frame[f'{secondary_column}_original_value'] = original
    frame[secondary_column] = remediated
    frame['has_secondary_diagnosis'] = has_secondary_diagnosis
    frame['secondary_diagnosis_imputation_method'] = method
    frame['secondary_diagnosis_confidence_score'] = confidence_score.round(3)
    frame['secondary_diagnosis_confidence_level'] = confidence_level
    frame['secondary_diagnosis_source_fields_used'] = source_fields_used
    frame['secondary_diagnosis_validation_rule_applied'] = validation_rule
    frame['secondary_diagnosis_flag_reason'] = flag_reason.mask(flag_reason.eq(''), pd.NA)
    frame['secondary_diagnosis_manual_review_flag'] = manual_review_flag

    patient_values = (
        _safe_string(frame[patient_col]).fillna(pd.Series(frame.index.astype(str), index=frame.index))
        if patient_col
        else pd.Series(frame.index.astype(str), index=frame.index)
    )
    audit_table = pd.DataFrame(
        {
            'patient_id': patient_values,
            'original_value': original,
            'imputed_value': remediated,
            'method_used': method,
            'confidence_score': confidence_score.round(3),
            'confidence_level': confidence_level,
            'source_fields_used': source_fields_used,
            'validation_rule_applied': validation_rule,
            'flag_reason': flag_reason.mask(flag_reason.eq(''), pd.NA),
        }
    )

    original_distribution = (
        original.fillna('Missing')
        .value_counts(dropna=False)
        .head(15)
        .rename_axis('secondary_diagnosis_label')
        .reset_index(name='original_count')
    )
    cleaned_distribution = (
        remediated.fillna('Missing')
        .value_counts(dropna=False)
        .head(15)
        .rename_axis('secondary_diagnosis_label')
        .reset_index(name='cleaned_count')
    )
    method_breakdown = (
        audit_table.groupby('method_used', observed=False)
        .size()
        .rename('record_count')
        .reset_index()
        .sort_values('record_count', ascending=False)
        .reset_index(drop=True)
    )
    confidence_distribution = (
        audit_table.groupby('confidence_level', observed=False)
        .size()
        .rename('record_count')
        .reset_index()
        .sort_values('confidence_level')
        .reset_index(drop=True)
    )
    confidence_distribution['record_pct'] = confidence_distribution['record_count'] / max(len(audit_table), 1)
    validation_report = pd.DataFrame(
        [
            {
                'validation_issue': 'Original missing secondary diagnosis',
                'row_count': int(original.isna().sum()),
            },
            {
                'validation_issue': 'Rows defaulted to No Secondary Diagnosis',
                'row_count': int(default_mask.sum()),
            },
            {
                'validation_issue': 'Duplicate primary / secondary diagnosis pairs',
                'row_count': int(flag_reason.ne('').sum()),
            },
            {
                'validation_issue': 'Manual review flagged',
                'row_count': int(manual_review_flag.sum()),
            },
        ]
    )

    summary = {
        'available': True,
        'secondary_diagnosis_column': secondary_column,
        'total_rows_checked': int(len(frame)),
        'original_missing_count': int(original.isna().sum()),
        'original_missing_pct': float(original.isna().mean()),
        'post_remediation_missing_count': int(frame[secondary_column].isna().sum()),
        'remediation_mode': mode,
        'pattern_sources_used': pattern_sources,
        'method_breakdown': method_breakdown,
        'confidence_distribution': confidence_distribution,
        'validation_error_report': validation_report,
        'mapping_audit_table': audit_table,
        'original_distribution': original_distribution,
        'cleaned_distribution': cleaned_distribution,
        'manual_review_count': int(manual_review_flag.sum()),
        'lineage_note': (
            'Secondary diagnosis labels were preserved when present, forward-filled within patient timelines when possible, '
            'pattern-mapped from diagnosis context when enough signal existed, and defaulted to No Secondary Diagnosis otherwise.'
        ),
    }
    return frame, summary


def remediate_bmi(data: pd.DataFrame, mode: str = 'median') -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = data.copy()
    bmi_column = _find_column(frame, ['bmi', 'bmi_remediated_value', 'body_mass_index'])
    if not bmi_column:
        return frame, {
            'available': False,
            'reason': 'BMI field is not available for remediation.',
            'remediation_mode': mode,
        }

    original = frame[bmi_column].copy()
    numeric_original = _safe_numeric(frame[bmi_column])
    derived_bmi, vital_sources = _derive_bmi_from_vitals(frame)
    obesity_signal = _obesity_signal(frame)
    clinical_average = _clinical_average_bmi(frame)
    practical_valid_mask = numeric_original.between(BMI_PRACTICAL_MIN, BMI_PRACTICAL_MAX, inclusive='both')
    broad_valid_mask = numeric_original.between(BMI_MIN, BMI_MAX, inclusive='both')
    outlier_mask = numeric_original.isna() | ~broad_valid_mask
    severe_outlier_mask = numeric_original.notna() & ~numeric_original.between(BMI_OUTLIER_MIN, BMI_OUTLIER_MAX, inclusive='both')

    derived_valid_mask = derived_bmi.between(BMI_PRACTICAL_MIN, BMI_PRACTICAL_MAX, inclusive='both')
    remediated = numeric_original.astype('float64').copy()
    source_field_used = pd.Series(bmi_column, index=frame.index, dtype='object')
    validation_rule = pd.Series('Original BMI retained after practical-range validation', index=frame.index, dtype='object')
    flag_reason = pd.Series('', index=frame.index, dtype='object')
    confidence_score = pd.Series(0.88, index=frame.index, dtype='float64')
    method = pd.Series('validated_original', index=frame.index, dtype='object')

    if vital_sources:
        direct_calc_mask = derived_valid_mask & (
            outlier_mask
            | derived_bmi.sub(numeric_original).abs().gt(1.5)
            | ~practical_valid_mask
        )
        remediated.loc[direct_calc_mask] = derived_bmi.loc[direct_calc_mask]
        source_field_used.loc[direct_calc_mask] = ' + '.join(vital_sources)
        validation_rule.loc[direct_calc_mask] = 'Direct BMI recalculation from validated height and weight'
        confidence_score.loc[direct_calc_mask] = 0.98
        method.loc[direct_calc_mask] = 'direct_height_weight_calc'

    obesity_mismatch = obesity_signal & remediated.lt(30.0)
    flag_reason.loc[obesity_mismatch] = 'BMI conflicts with obesity-coded diagnosis'
    confidence_score.loc[obesity_mismatch] = confidence_score.loc[obesity_mismatch].clip(upper=0.84)
    validation_rule.loc[obesity_mismatch] = 'Diagnosis cross-check flagged BMI/obesity mismatch'
    method.loc[obesity_mismatch & method.eq('validated_original')] = 'validated_mismatch'

    fallback_mask = remediated.isna() | severe_outlier_mask
    if mode == 'null':
        remediated.loc[fallback_mask] = float('nan')
        validation_rule.loc[fallback_mask] = 'Invalid BMI set to null for manual review'
        confidence_score.loc[fallback_mask] = 0.20
        method.loc[fallback_mask] = 'null_for_review'
    else:
        remediated.loc[fallback_mask] = clinical_average.loc[fallback_mask].astype('float64')
        source_field_used.loc[fallback_mask] = source_field_used.loc[fallback_mask].where(source_field_used.loc[fallback_mask].ne(bmi_column), 'clinical_average_by_age_gender')
        validation_rule.loc[fallback_mask] = 'Clinical average fallback by age and gender'
        confidence_score.loc[fallback_mask] = 0.72
        method.loc[fallback_mask] = 'clinical_average_fallback'
        flag_reason.loc[fallback_mask & flag_reason.eq('')] = 'Original BMI outside clinically plausible range'

    temporal_jump_flag = _temporal_validation_flags(frame, remediated)
    flag_reason.loc[temporal_jump_flag & flag_reason.eq('')] = 'BMI changed more than 5 points between visits'
    confidence_score.loc[temporal_jump_flag] = confidence_score.loc[temporal_jump_flag].clip(upper=0.82)
    validation_rule.loc[temporal_jump_flag] = 'Temporal validation flagged an implausible BMI trajectory'

    remediated = remediated.clip(lower=BMI_OUTLIER_MIN, upper=BMI_OUTLIER_MAX)
    outlier_flag = remediated.lt(BMI_PRACTICAL_MIN) | remediated.gt(BMI_PRACTICAL_MAX) | flag_reason.ne('')
    confidence_score = confidence_score.clip(lower=0.0, upper=0.99)
    confidence_level = confidence_score.apply(_confidence_label)

    frame['bmi_original_value'] = original
    frame['bmi_outlier_flag'] = outlier_flag.fillna(True)
    frame['bmi_remediated_value'] = remediated
    frame['bmi_remediation_action'] = method
    frame['bmi_confidence_score'] = confidence_score.round(3)
    frame['bmi_confidence_level'] = confidence_level
    frame['bmi_source_fields_used'] = source_field_used
    frame['bmi_validation_rule_applied'] = validation_rule
    frame['bmi_flag_reason'] = flag_reason.mask(flag_reason.eq(''), pd.NA)
    frame['bmi_manual_review_flag'] = outlier_flag | confidence_score.lt(BMI_MEDIUM_CONFIDENCE)
    if mode != 'none':
        frame[bmi_column] = remediated

    patient_col = _find_column(frame, ['patient_id', 'member_id', 'entity_id']) or 'row_index'
    if patient_col == 'row_index':
        patient_values = pd.Series(frame.index.astype(str), index=frame.index)
    else:
        patient_values = frame[patient_col].astype(str)

    audit_table = pd.DataFrame(
        {
            'patient_id': patient_values,
            'original_value': original,
            'remapped_value': remediated,
            'confidence_score': confidence_score.round(3),
            'confidence_level': confidence_level,
            'source_fields_used': source_field_used,
            'validation_rule_applied': validation_rule,
            'flag_reason': flag_reason.replace('', pd.NA),
            'method_used': method,
        }
    )

    confidence_distribution = (
        audit_table.groupby('confidence_level')
        .size()
        .rename('record_count')
        .reset_index()
        .sort_values('confidence_level')
        .reset_index(drop=True)
    )
    confidence_distribution['record_pct'] = confidence_distribution['record_count'] / max(len(audit_table), 1)

    validation_rows = [
        {'validation_issue': 'BMI outside practical range', 'row_count': int((~remediated.between(BMI_PRACTICAL_MIN, BMI_PRACTICAL_MAX, inclusive='both')).sum())},
        {'validation_issue': 'Diagnosis/BMI obesity mismatch', 'row_count': int(obesity_mismatch.sum())},
        {'validation_issue': 'Temporal BMI jump > 5 points', 'row_count': int(temporal_jump_flag.sum())},
        {'validation_issue': 'Manual review flagged', 'row_count': int(frame['bmi_manual_review_flag'].sum())},
    ]
    validation_errors = pd.DataFrame(validation_rows)

    before_cost = _build_synthetic_cost_estimate(frame.assign(bmi=numeric_original.fillna(clinical_average)), numeric_original.fillna(clinical_average))
    after_cost = _build_synthetic_cost_estimate(frame.assign(bmi=remediated), remediated)
    before_readmission = _build_synthetic_readmission_proxy(frame.assign(bmi=numeric_original.fillna(clinical_average)), numeric_original.fillna(clinical_average))
    after_readmission = _build_synthetic_readmission_proxy(frame.assign(bmi=remediated), remediated)
    before_high_risk = (before_readmission >= float(before_readmission.quantile(0.80))).mean() if not before_readmission.empty else 0.0
    after_high_risk = (after_readmission >= float(after_readmission.quantile(0.80))).mean() if not after_readmission.empty else 0.0

    downstream_impact = pd.DataFrame(
        [
            {
                'analysis_area': 'Synthetic cost estimate',
                'before_value': float(before_cost.mean()) if not before_cost.empty else 0.0,
                'after_value': float(after_cost.mean()) if not after_cost.empty else 0.0,
                'delta': (float(after_cost.mean()) - float(before_cost.mean())) if not before_cost.empty else 0.0,
                'interpretation': 'Average estimated cost impact after BMI remapping.',
            },
            {
                'analysis_area': 'Readmission risk proxy',
                'before_value': float(before_readmission.mean()) if not before_readmission.empty else 0.0,
                'after_value': float(after_readmission.mean()) if not after_readmission.empty else 0.0,
                'delta': (float(after_readmission.mean()) - float(before_readmission.mean())) if not before_readmission.empty else 0.0,
                'interpretation': 'Average readmission-risk proxy change after BMI remapping.',
            },
            {
                'analysis_area': 'High-risk share',
                'before_value': float(before_high_risk),
                'after_value': float(after_high_risk),
                'delta': float(after_high_risk - before_high_risk),
                'interpretation': 'Share of rows in the top 20% readmission-risk band before and after BMI cleanup.',
            },
        ]
    )

    total_outliers = int(frame['bmi_outlier_flag'].sum())
    high_confidence_share = float((confidence_score >= BMI_HIGH_CONFIDENCE).mean())
    summary = {
        'available': True,
        'bmi_column': bmi_column,
        'total_rows_checked': int(len(frame)),
        'total_bmi_outliers': total_outliers,
        'outlier_pct': float(total_outliers / max(len(frame), 1)),
        'remediation_mode': mode,
        'replacement_value_if_used': None,
        'pre_remediation_distribution_summary': _distribution_summary(original),
        'post_remediation_distribution_summary': _distribution_summary(frame[bmi_column]),
        'current_mapping_assessment': {
            'practical_range_rule': f'BMI practical range {BMI_PRACTICAL_MIN:.1f}-{BMI_PRACTICAL_MAX:.1f}',
            'height_weight_validation_available': bool(vital_sources),
            'obesity_cross_check_available': True,
            'temporal_validation_available': bool(_find_column(frame, ['patient_id', 'member_id', 'entity_id']) and _find_column(frame, ['admission_date', 'service_date', 'event_date', 'diagnosis_date', 'discharge_date'])),
        },
        'confidence_distribution': confidence_distribution,
        'mapping_audit_table': audit_table,
        'validation_error_report': validation_errors,
        'downstream_impact_report': downstream_impact,
        'high_confidence_share': high_confidence_share,
        'target_high_confidence_share': 0.90,
        'target_met': high_confidence_share >= 0.90,
        'calculation_method_breakdown': (
            audit_table.groupby('method_used')
            .size()
            .rename('record_count')
            .reset_index()
            .sort_values('record_count', ascending=False)
            .reset_index(drop=True)
        ),
        'lineage_note': (
            'BMI values were validated against practical clinical ranges, recalculated from height/weight when possible, '
            'cross-checked against obesity diagnosis signals, and reviewed for implausible temporal jumps.'
        ),
    }
    return frame, summary


def add_synthetic_cost_fields(data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = data.copy()
    native_candidates = ['cost_amount', 'paid_amount', 'allowed_amount', 'billed_amount', 'estimated_cost']
    native_present = next((column for column in native_candidates if column in frame.columns), None)
    if native_present and native_present != 'estimated_cost':
        return frame, {
            'available': False,
            'reason': 'A native financial field already exists.',
            'native_cost_column': native_present,
        }

    bmi_values = _safe_numeric(frame['bmi']) if 'bmi' in frame.columns else pd.Series(27.5, index=frame.index)
    estimated = _build_synthetic_cost_estimate(frame, bmi_values)

    frame['estimated_cost'] = estimated
    frame['estimated_cost_source'] = 'synthetic'
    if 'cost_amount' not in frame.columns:
        frame['cost_amount'] = frame['estimated_cost']

    summary = {
        'available': True,
        'cost_column': 'cost_amount',
        'estimated_cost_source': 'synthetic',
        'synthetic_cost_summary': {
            'min': float(frame['estimated_cost'].min()),
            'max': float(frame['estimated_cost'].max()),
            'median': float(frame['estimated_cost'].median()),
            'mean': float(frame['estimated_cost'].mean()),
            'synthetic_row_pct': 1.0,
        },
        'top_synthetic_cost_drivers': pd.DataFrame([
            {'driver': 'Age', 'impact_note': 'Older patients raise the synthetic baseline cost.'},
            {'driver': 'Smoking history', 'impact_note': 'Current or former smoking increases the deterministic cost estimate.'},
            {'driver': 'Cardiometabolic burden', 'impact_note': 'Hypertension, heart disease, diabetes, and BMI burden increase cost.'},
        ]),
        'lineage_note': 'Estimated cost was generated with a deterministic demo-only formula because no native cost field was available.',
    }
    return frame, summary


def add_synthetic_clinical_labels(data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = data.copy()
    if 'diagnosis_code' in frame.columns and 'diagnosis_label' in frame.columns:
        return frame, {'available': False, 'reason': 'Native diagnosis fields already exist.'}

    age = _safe_numeric(frame['age']) if 'age' in frame.columns else pd.Series(float('nan'), index=frame.index)
    bmi = _safe_numeric(frame['bmi']) if 'bmi' in frame.columns else pd.Series(float('nan'), index=frame.index)
    smoking = frame['smoking_history'] if 'smoking_history' in frame.columns else frame.get('smoking_status', pd.Series('', index=frame.index))
    smoking_text = smoking.astype(str).str.strip().str.lower()
    hypertension = _safe_numeric(frame['hypertension']) if 'hypertension' in frame.columns else pd.Series(0.0, index=frame.index)
    heart_disease = _safe_numeric(frame['heart_disease']) if 'heart_disease' in frame.columns else pd.Series(0.0, index=frame.index)

    primary_label = pd.Series('General Metabolic Review', index=frame.index, dtype='object')
    primary_code = pd.Series('SYN-GEN', index=frame.index, dtype='object')
    secondary_label = pd.Series('', index=frame.index, dtype='object')
    risk_label = pd.Series('Baseline Adult Risk', index=frame.index, dtype='object')

    obesity = bmi >= 30
    smoking_risk = smoking_text.str.contains('current|smoker|yes', na=False)
    htn = hypertension.fillna(0) > 0
    heart = heart_disease.fillna(0) > 0
    older = age >= 65

    primary_label.loc[obesity] = 'Obesity'
    primary_code.loc[obesity] = 'SYN-OBE'

    primary_label.loc[~obesity & htn] = 'Hypertension'
    primary_code.loc[~obesity & htn] = 'SYN-HTN'

    primary_label.loc[~obesity & ~htn & heart] = 'Heart Disease Risk'
    primary_code.loc[~obesity & ~htn & heart] = 'SYN-HDR'

    primary_label.loc[~obesity & ~htn & ~heart & smoking_risk] = 'Nicotine Dependence'
    primary_code.loc[~obesity & ~htn & ~heart & smoking_risk] = 'SYN-NIC'

    secondary_label.loc[obesity & htn] = 'Hypertension'
    secondary_label.loc[obesity & smoking_risk] = secondary_label.loc[obesity & smoking_risk].replace('', 'Nicotine Dependence')
    secondary_label.loc[heart] = secondary_label.loc[heart].replace('', 'Heart Disease Risk')

    risk_label.loc[older] = 'Older Adult Risk'
    risk_label.loc[older & (obesity | htn | heart | smoking_risk)] = 'Elevated Older Adult Risk'
    risk_label.loc[~older & (obesity | htn | heart | smoking_risk)] = 'Elevated Chronic Risk'

    frame['diagnosis_label'] = primary_label
    if 'diagnosis_code' not in frame.columns:
        frame['diagnosis_code'] = primary_code
    generated_secondary = secondary_label.replace('', pd.NA)
    if 'secondary_diagnosis_label' in frame.columns:
        existing_secondary = _safe_string(frame['secondary_diagnosis_label'])
        frame['secondary_diagnosis_label'] = existing_secondary.fillna(generated_secondary)
    else:
        frame['secondary_diagnosis_label'] = generated_secondary
    frame['clinical_risk_label'] = risk_label

    summary = {
        'available': True,
        'diagnosis_column': 'diagnosis_code',
        'label_column': 'diagnosis_label',
        'lineage_note': 'Diagnosis and clinical risk labels were derived from BMI, smoking, age, and cardiometabolic risk signals.',
        'summary_table': pd.DataFrame([
            {'derived_field': 'diagnosis_label', 'source': 'synthetic', 'note': 'Approximate clinical grouping, not a billing diagnosis.'},
            {'derived_field': 'diagnosis_code', 'source': 'synthetic', 'note': 'Internal synthetic grouping code such as SYN-OBE or SYN-HTN.'},
            {'derived_field': 'clinical_risk_label', 'source': 'synthetic', 'note': 'Rule-based clinical risk descriptor for segmentation.'},
        ]),
    }
    return frame, summary


def add_synthetic_readmission_fields(data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = data.copy()
    native_present = 'readmission' in frame.columns or 'readmission_flag' in frame.columns
    if native_present:
        return frame, {'available': False, 'reason': 'A native or already-derived readmission field exists.'}

    bmi_values = _safe_numeric(frame['bmi']) if 'bmi' in frame.columns else pd.Series(27.5, index=frame.index)
    proxy = _build_synthetic_readmission_proxy(frame, bmi_values)
    threshold = float(proxy.quantile(0.80))
    flag = (proxy >= threshold).astype(int)

    frame['readmission_risk_proxy'] = proxy
    frame['readmission_flag'] = flag
    frame['readmission_source'] = 'synthetic'
    if 'readmission' not in frame.columns:
        frame['readmission'] = flag

    summary = {
        'available': True,
        'lineage_note': 'Readmission support was enabled with a synthetic, deterministic proxy because no native readmission field was available.',
        'prevalence': float(flag.mean()),
        'threshold': threshold,
        'summary_table': pd.DataFrame([
            {'metric': 'Synthetic readmission prevalence', 'value': float(flag.mean())},
            {'metric': 'Readmission proxy threshold', 'value': float(threshold)},
        ]),
    }
    return frame, summary


def apply_remediation_augmentations(
    data: pd.DataFrame,
    bmi_mode: str = 'median',
    helper_mode: str = 'Auto',
    synthetic_cost_mode: str = 'Auto',
    synthetic_readmission_mode: str = 'Auto',
) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = data.copy()
    frame.attrs.update(getattr(data, 'attrs', {}))
    lineage_rows: list[dict[str, str]] = []
    helper_mode = str(helper_mode).lower()
    synthetic_cost_mode = str(synthetic_cost_mode).lower()
    synthetic_readmission_mode = str(synthetic_readmission_mode).lower()

    frame, bmi_summary = remediate_bmi(frame, mode=bmi_mode)
    if bmi_summary.get('available'):
        for field in [
            'bmi',
            'bmi_original_value',
            'bmi_outlier_flag',
            'bmi_remediated_value',
            'bmi_remediation_action',
            'bmi_confidence_score',
            'bmi_confidence_level',
            'bmi_source_fields_used',
            'bmi_validation_rule_applied',
            'bmi_flag_reason',
            'bmi_manual_review_flag',
        ]:
            helper_type = 'audit' if field in {
                'bmi_original_value',
                'bmi_remediation_action',
                'bmi_source_fields_used',
                'bmi_validation_rule_applied',
                'bmi_flag_reason',
            } else 'remediated'
            lineage_rows.append({'helper_field': field, 'helper_type': helper_type, 'note': bmi_summary['lineage_note']})

    frame, secondary_summary = remediate_secondary_diagnosis(frame)
    if secondary_summary.get('available'):
        for field in [
            secondary_summary.get('secondary_diagnosis_column', 'secondary_diagnosis_label'),
            f"{secondary_summary.get('secondary_diagnosis_column', 'secondary_diagnosis_label')}_original_value",
            'has_secondary_diagnosis',
            'secondary_diagnosis_imputation_method',
            'secondary_diagnosis_confidence_score',
            'secondary_diagnosis_confidence_level',
            'secondary_diagnosis_source_fields_used',
            'secondary_diagnosis_validation_rule_applied',
            'secondary_diagnosis_flag_reason',
            'secondary_diagnosis_manual_review_flag',
        ]:
            helper_type = 'audit' if field in {
                f"{secondary_summary.get('secondary_diagnosis_column', 'secondary_diagnosis_label')}_original_value",
                'secondary_diagnosis_imputation_method',
                'secondary_diagnosis_source_fields_used',
                'secondary_diagnosis_validation_rule_applied',
                'secondary_diagnosis_flag_reason',
            } else 'remediated'
            lineage_rows.append({'helper_field': str(field), 'helper_type': helper_type, 'note': secondary_summary['lineage_note']})

    cost_summary: dict[str, Any] = {'available': False, 'reason': 'Synthetic cost support is disabled for the current run.'}
    if helper_mode != 'off' and synthetic_cost_mode != 'off':
        frame, cost_summary = add_synthetic_cost_fields(frame)
    if cost_summary.get('available'):
        lineage_rows.append({'helper_field': 'cost_amount', 'helper_type': 'synthetic', 'note': cost_summary['lineage_note']})
        lineage_rows.append({'helper_field': 'estimated_cost', 'helper_type': 'synthetic', 'note': 'Deterministic demo-only cost estimate.'})
        lineage_rows.append({'helper_field': 'estimated_cost_source', 'helper_type': 'audit', 'note': 'Tracks whether cost support is native or synthetic.'})

    clinical_summary: dict[str, Any] = {'available': False, 'reason': 'Synthetic clinical support is disabled for the current run.'}
    if helper_mode != 'off':
        frame, clinical_summary = add_synthetic_clinical_labels(frame)
    if clinical_summary.get('available'):
        for field in ['diagnosis_label', 'diagnosis_code', 'clinical_risk_label']:
            lineage_rows.append({'helper_field': field, 'helper_type': 'derived', 'note': clinical_summary['lineage_note']})

    readmission_summary: dict[str, Any] = {'available': False, 'reason': 'Synthetic readmission support is disabled for the current run.'}
    if helper_mode != 'off' and synthetic_readmission_mode != 'off':
        frame, readmission_summary = add_synthetic_readmission_fields(frame)
    if readmission_summary.get('available'):
        for field in ['readmission', 'readmission_flag', 'readmission_risk_proxy']:
            lineage_rows.append({'helper_field': field, 'helper_type': 'synthetic', 'note': readmission_summary['lineage_note']})
        lineage_rows.append({'helper_field': 'readmission_source', 'helper_type': 'audit', 'note': 'Tracks whether readmission support is native or synthetic.'})

    helper_fields = pd.DataFrame(lineage_rows)
    native_fields = [column for column in data.columns if column in frame.columns]
    synthetic_fields = [row['helper_field'] for row in lineage_rows if row['helper_type'] == 'synthetic']
    derived_fields = [row['helper_field'] for row in lineage_rows if row['helper_type'] == 'derived']
    remediated_fields = [row['helper_field'] for row in lineage_rows if row['helper_type'] == 'remediated']

    frame.attrs.update(getattr(data, 'attrs', {}))
    return frame, {
        'bmi_remediation': bmi_summary,
        'secondary_diagnosis_remediation': secondary_summary,
        'synthetic_cost': cost_summary,
        'synthetic_clinical': clinical_summary,
        'synthetic_readmission': readmission_summary,
        'helper_fields': helper_fields,
        'native_field_count': len(native_fields),
        'synthetic_field_count': len(set(synthetic_fields)),
        'derived_field_count': len(set(derived_fields)),
        'remediated_field_count': len(set(remediated_fields)),
        'demo_config_applied': {
            'bmi_mode': bmi_mode,
            'helper_mode': helper_mode,
            'synthetic_cost_mode': synthetic_cost_mode,
            'synthetic_readmission_mode': synthetic_readmission_mode,
        },
    }
