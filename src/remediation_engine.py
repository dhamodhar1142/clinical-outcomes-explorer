from __future__ import annotations

from typing import Any

import pandas as pd


BMI_MIN = 10.0
BMI_MAX = 80.0


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


def remediate_bmi(data: pd.DataFrame, mode: str = 'median') -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = data.copy()
    bmi_column = 'bmi' if 'bmi' in frame.columns else None
    if not bmi_column:
        return frame, {
            'available': False,
            'reason': 'BMI field is not available for remediation.',
            'remediation_mode': mode,
        }

    original = frame[bmi_column].copy()
    numeric = _safe_numeric(frame[bmi_column])
    valid_mask = numeric.between(BMI_MIN, BMI_MAX, inclusive='both')
    outlier_mask = numeric.isna() | ~valid_mask
    valid_values = numeric[valid_mask]
    replacement_value = float(valid_values.median()) if not valid_values.empty else 27.5

    if mode == 'median':
        remediated = numeric.mask(outlier_mask, replacement_value)
        action = pd.Series('Retained valid value', index=frame.index)
        action.loc[outlier_mask] = 'Replaced with median BMI'
    elif mode == 'clip':
        remediated = numeric.clip(lower=BMI_MIN, upper=BMI_MAX)
        action = pd.Series('Retained valid value', index=frame.index)
        action.loc[numeric < BMI_MIN] = 'Clipped to BMI minimum'
        action.loc[numeric > BMI_MAX] = 'Clipped to BMI maximum'
        action.loc[numeric.isna()] = 'Unable to clip non-numeric BMI'
    elif mode == 'null':
        remediated = numeric.mask(outlier_mask)
        action = pd.Series('Retained valid value', index=frame.index)
        action.loc[outlier_mask] = 'Set invalid BMI to null'
    else:
        remediated = numeric
        action = pd.Series('Retained original BMI', index=frame.index)
        action.loc[outlier_mask] = 'Flagged suspicious BMI only'

    frame['bmi_original_value'] = original
    frame['bmi_outlier_flag'] = outlier_mask.fillna(True)
    frame['bmi_remediated_value'] = remediated
    frame['bmi_remediation_action'] = action
    if mode != 'none':
        frame[bmi_column] = remediated

    total_outliers = int(frame['bmi_outlier_flag'].sum())
    summary = {
        'available': True,
        'bmi_column': bmi_column,
        'total_rows_checked': int(len(frame)),
        'total_bmi_outliers': total_outliers,
        'outlier_pct': float(total_outliers / max(len(frame), 1)),
        'remediation_mode': mode,
        'replacement_value_if_used': float(replacement_value) if mode == 'median' else None,
        'pre_remediation_distribution_summary': _distribution_summary(original),
        'post_remediation_distribution_summary': _distribution_summary(frame[bmi_column]),
        'lineage_note': f"BMI values outside {BMI_MIN:.0f}-{BMI_MAX:.0f} were handled with {mode} remediation.",
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

    age = _safe_numeric(frame['age']) if 'age' in frame.columns else pd.Series(0.0, index=frame.index)
    bmi = _safe_numeric(frame['bmi']) if 'bmi' in frame.columns else pd.Series(27.5, index=frame.index)
    smoking = frame['smoking_history'] if 'smoking_history' in frame.columns else frame.get('smoking_status', pd.Series('', index=frame.index))
    smoking_text = smoking.astype(str).str.strip().str.lower()
    hypertension = _safe_numeric(frame['hypertension']) if 'hypertension' in frame.columns else pd.Series(0.0, index=frame.index)
    heart_disease = _safe_numeric(frame['heart_disease']) if 'heart_disease' in frame.columns else pd.Series(0.0, index=frame.index)
    diabetes_signal = _safe_numeric(frame['diabetes']) if 'diabetes' in frame.columns else pd.Series(0.0, index=frame.index)
    if 'readmission_risk_proxy' in frame.columns:
        diabetes_signal = diabetes_signal.combine_first((_safe_numeric(frame['readmission_risk_proxy']) > 0.6).astype(float))

    bmi_component = pd.Series(0.0, index=frame.index)
    bmi_component.loc[bmi >= 25] = 150.0
    bmi_component.loc[bmi >= 30] = 350.0
    bmi_component.loc[bmi >= 40] = 550.0

    smoking_component = pd.Series(0.0, index=frame.index)
    smoking_component.loc[smoking_text.str.contains('former', na=False)] = 250.0
    smoking_component.loc[smoking_text.str.contains('current|smoker|yes', na=False)] = 500.0

    estimated = (
        500.0
        + age.fillna(age.median()) * 8.0
        + bmi_component
        + smoking_component
        + (hypertension.fillna(0) > 0).astype(float) * 400.0
        + (heart_disease.fillna(0) > 0).astype(float) * 700.0
        + (diabetes_signal.fillna(0) > 0).astype(float) * 300.0
    ).clip(lower=500.0, upper=7500.0)

    frame['estimated_cost'] = estimated.round(2)
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
            {'driver': 'Cardiometabolic burden', 'impact_note': 'Hypertension, heart disease, and diabetes-style signals increase cost.'},
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
    frame['secondary_diagnosis_label'] = secondary_label.replace('', pd.NA)
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

    age = _safe_numeric(frame['age']) if 'age' in frame.columns else pd.Series(0.0, index=frame.index)
    bmi = _safe_numeric(frame['bmi']) if 'bmi' in frame.columns else pd.Series(27.5, index=frame.index)
    glucose = _safe_numeric(frame['blood_glucose_level']) if 'blood_glucose_level' in frame.columns else pd.Series(float('nan'), index=frame.index)
    smoking = frame['smoking_history'] if 'smoking_history' in frame.columns else frame.get('smoking_status', pd.Series('', index=frame.index))
    smoking_text = smoking.astype(str).str.strip().str.lower()
    hypertension = _safe_numeric(frame['hypertension']) if 'hypertension' in frame.columns else pd.Series(0.0, index=frame.index)
    heart_disease = _safe_numeric(frame['heart_disease']) if 'heart_disease' in frame.columns else pd.Series(0.0, index=frame.index)

    proxy = pd.Series(0.08, index=frame.index, dtype='float64')
    proxy += (age.fillna(age.median()) >= 65).astype(float) * 0.12
    proxy += (bmi >= 30).astype(float) * 0.08
    proxy += smoking_text.str.contains('current|smoker|yes', na=False).astype(float) * 0.10
    proxy += (hypertension.fillna(0) > 0).astype(float) * 0.08
    proxy += (heart_disease.fillna(0) > 0).astype(float) * 0.14
    if glucose.notna().any():
        proxy += (glucose.fillna(glucose.median()) >= 180).astype(float) * 0.10
    if 'risk_score' in frame.columns:
        proxy += (_safe_numeric(frame['risk_score']).fillna(0) / 10.0).clip(upper=0.15)
    proxy = proxy.clip(lower=0.03, upper=0.65)
    threshold = float(proxy.quantile(0.80))
    flag = (proxy >= threshold).astype(int)

    frame['readmission_risk_proxy'] = proxy.round(4)
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
    lineage_rows: list[dict[str, str]] = []
    helper_mode = str(helper_mode).lower()
    synthetic_cost_mode = str(synthetic_cost_mode).lower()
    synthetic_readmission_mode = str(synthetic_readmission_mode).lower()

    frame, bmi_summary = remediate_bmi(frame, mode=bmi_mode)
    if bmi_summary.get('available'):
        lineage_rows.append({'helper_field': 'bmi', 'helper_type': 'remediated', 'note': bmi_summary['lineage_note']})

    cost_summary: dict[str, Any] = {'available': False, 'reason': 'Synthetic cost support is disabled for the current run.'}
    if helper_mode != 'off' and synthetic_cost_mode != 'off':
        frame, cost_summary = add_synthetic_cost_fields(frame)
    if cost_summary.get('available'):
        lineage_rows.append({'helper_field': 'cost_amount', 'helper_type': 'synthetic', 'note': cost_summary['lineage_note']})
        lineage_rows.append({'helper_field': 'estimated_cost', 'helper_type': 'synthetic', 'note': 'Deterministic demo-only cost estimate.'})

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

    helper_fields = pd.DataFrame(lineage_rows)
    native_fields = [column for column in data.columns if column in frame.columns]
    synthetic_fields = [row['helper_field'] for row in lineage_rows if row['helper_type'] == 'synthetic']
    derived_fields = [row['helper_field'] for row in lineage_rows if row['helper_type'] == 'derived']
    remediated_fields = [row['helper_field'] for row in lineage_rows if row['helper_type'] == 'remediated']

    return frame, {
        'bmi_remediation': bmi_summary,
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
