from __future__ import annotations

import re

import pandas as pd


HEALTHCARE_KEY_FIELDS = [
    'patient_id', 'member_id', 'encounter_id', 'claim_id', 'admission_date', 'discharge_date', 'service_date',
    'diagnosis_code', 'procedure_code', 'provider_id', 'provider_name', 'facility', 'payer', 'plan',
    'cost_amount', 'paid_amount', 'allowed_amount', 'billed_amount', 'length_of_stay', 'department',
    'smoking_status', 'cancer_stage', 'treatment_type', 'survived', 'bmi', 'cholesterol_level', 'comorbidities',
    'diagnosis_date', 'end_treatment_date',
]


def _first_available(canonical_map: dict[str, str], candidates: list[str]) -> str | None:
    for candidate in candidates:
        if candidate in canonical_map:
            return canonical_map[candidate]
    return None


def _resolve_column(data: pd.DataFrame, canonical_map: dict[str, str], canonical_candidates: list[str], raw_candidates: list[str]) -> str | None:
    mapped = _first_available(canonical_map, canonical_candidates)
    if mapped and mapped in data.columns:
        return mapped
    for candidate in raw_candidates:
        if candidate in data.columns:
            return candidate
    return None


def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors='coerce')


def _clean_group_field(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    cleaned = frame.copy()
    series = cleaned[column]
    if isinstance(series.dtype, pd.CategoricalDtype):
        if 'Unknown' not in series.cat.categories:
            series = series.cat.add_categories(['Unknown'])
        cleaned[column] = series.fillna('Unknown')
    else:
        cleaned[column] = series.fillna('Unknown')
    cleaned[column] = cleaned[column].astype(str)
    cleaned = cleaned[cleaned[column].str.strip() != '']
    return cleaned


def _binary_outcome(series: pd.Series) -> pd.Series:
    mapping = {
        '1': 1,
        '0': 0,
        'true': 1,
        'false': 0,
        'yes': 1,
        'no': 0,
        'y': 1,
        'n': 0,
        'alive': 1,
        'survived': 1,
        'deceased': 0,
        'dead': 0,
    }
    text = series.astype(str).str.strip().str.lower()
    mapped = text.map(mapping)
    numeric = pd.to_numeric(series, errors='coerce')
    return mapped.fillna(numeric)


def _comorbidity_present(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors='coerce')
    if numeric.notna().mean() >= 0.5:
        return numeric.fillna(0) > 0
    text = series.astype(str).str.strip().str.lower()
    return ~text.isin({'', '0', 'none', 'no', 'false', 'nan'})


def _stage_high_risk(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.strip().str.lower()
    return text.str.contains('iii|iv|stage iii|stage iv|3|4', regex=True, na=False)


def _readmission_flag(frame: pd.DataFrame, canonical_map: dict[str, str]) -> tuple[pd.Series | None, str | None]:
    readmission_col = _resolve_column(
        frame,
        canonical_map,
        ['readmission'],
        ['readmission', 'readmitted', 'readmit_flag', 'readmission_flag', 'is_readmission'],
    )
    if not readmission_col:
        return None, None
    values = _binary_outcome(frame[readmission_col])
    if values.notna().any():
        return values, readmission_col
    return None, readmission_col


def _band_numeric(series: pd.Series, bins: list[float], labels: list[str]) -> pd.Series:
    numeric = pd.to_numeric(series, errors='coerce')
    return pd.cut(numeric, bins=bins, labels=labels, include_lowest=True)


def _readmission_group_rate(frame: pd.DataFrame, group_col: str) -> pd.DataFrame:
    working = _clean_group_field(frame[[group_col, 'readmission_flag']].copy(), group_col)
    if working.empty:
        return pd.DataFrame()
    grouped = working.groupby(group_col).agg(
        record_count=('readmission_flag', 'size'),
        readmission_rate=('readmission_flag', 'mean'),
    ).reset_index()
    return grouped.sort_values(['readmission_rate', 'record_count'], ascending=[False, False]).reset_index(drop=True)



def _risk_detail_frame(data: pd.DataFrame, canonical_map: dict[str, str]) -> pd.DataFrame:
    age_col = _resolve_column(data, canonical_map, ['age'], ['age'])
    smoking_col = _resolve_column(data, canonical_map, ['smoking_status'], ['smoking_status', 'smoker'])
    stage_col = _resolve_column(data, canonical_map, ['cancer_stage'], ['cancer_stage', 'stage'])
    comorbidity_col = _resolve_column(data, canonical_map, ['comorbidities'], ['comorbidities', 'comorbidity_score'])
    outcome_col = _resolve_column(data, canonical_map, ['survived'], ['survived', 'alive', 'outcome'])

    available_inputs = [col for col in [age_col, smoking_col, stage_col, comorbidity_col] if col]
    if not available_inputs:
        return pd.DataFrame()

    frame = data.copy()
    risk_score = pd.Series(0, index=frame.index, dtype='float64')
    if age_col:
        risk_score += (_safe_numeric(frame[age_col]) > 60).fillna(False).astype(int)
    if smoking_col:
        smoking_text = frame[smoking_col].astype(str).str.strip().str.lower()
        risk_score += smoking_text.str.contains('smoker|current|yes', regex=True, na=False).astype(int)
    if stage_col:
        risk_score += _stage_high_risk(frame[stage_col]).astype(int)
    if comorbidity_col:
        risk_score += _comorbidity_present(frame[comorbidity_col]).astype(int)

    frame['risk_score'] = risk_score
    frame['risk_segment'] = pd.cut(frame['risk_score'], bins=[-1, 1, 2, 10], labels=['Low Risk', 'Medium Risk', 'High Risk']).astype(str)
    if outcome_col:
        frame['survived_binary'] = _binary_outcome(frame[outcome_col])
    return frame


def _build_group_metrics(frame: pd.DataFrame, group_col: str, canonical_map: dict[str, str]) -> pd.DataFrame:
    working = _clean_group_field(frame.copy(), group_col)
    if working.empty:
        return pd.DataFrame()

    metrics = working.groupby(group_col).size().reset_index(name='record_count')
    age_col = _resolve_column(working, canonical_map, ['age'], ['age'])
    bmi_col = _resolve_column(working, canonical_map, ['bmi'], ['bmi'])
    if age_col:
        metrics = metrics.merge(working.groupby(group_col)[age_col].apply(lambda s: pd.to_numeric(s, errors='coerce').mean()).reset_index(name='average_age'), on=group_col, how='left')
    if bmi_col:
        metrics = metrics.merge(working.groupby(group_col)[bmi_col].apply(lambda s: pd.to_numeric(s, errors='coerce').mean()).reset_index(name='average_bmi'), on=group_col, how='left')
    if 'survived_binary' in working.columns:
        metrics = metrics.merge(working.groupby(group_col)['survived_binary'].mean().reset_index(name='survival_rate'), on=group_col, how='left')
    if 'risk_score' in working.columns:
        metrics = metrics.merge(working.groupby(group_col)['risk_score'].mean().reset_index(name='average_risk_score'), on=group_col, how='left')
        metrics = metrics.merge(working.groupby(group_col)['risk_segment'].apply(lambda s: (s == 'High Risk').mean()).reset_index(name='high_risk_share'), on=group_col, how='left')
    if 'treatment_duration_days' in working.columns:
        metrics = metrics.merge(working.groupby(group_col)['treatment_duration_days'].mean().reset_index(name='average_treatment_duration_days'), on=group_col, how='left')
    return metrics.sort_values('record_count', ascending=False).reset_index(drop=True)


def _segment_priority_band(score: float) -> str:
    if score >= 0.35:
        return 'High Priority'
    if score >= 0.18:
        return 'Moderate Priority'
    return 'Watchlist'


def _segment_follow_up(metric_focus: str, dimensions: list[str]) -> str:
    dimension_text = ', '.join(dimensions)
    if metric_focus == 'Poor Survival':
        return f'Review subgroup outcomes across {dimension_text} and compare treatment patterns to identify where survival support can improve.'
    if metric_focus == 'High Risk':
        return f'Use Benchmarking or the cohort builder to compare {dimension_text} against lower-risk groups and target preventive follow-up.'
    if metric_focus == 'Long Treatment Duration':
        return f'Use Root Cause Explorer or Care Pathway View to review why {dimension_text} is associated with longer treatment duration.'
    return f'Use Benchmarking or Root Cause Explorer to compare {dimension_text} against the overall population and identify the next operational action.'

def assess_healthcare_dataset(canonical_map: dict[str, str], synthetic_fields: set[str] | None = None) -> dict[str, object]:
    synthetic_fields = synthetic_fields or set()
    hits = [field for field in HEALTHCARE_KEY_FIELDS if field in canonical_map]
    synthetic_hits = [field for field in hits if field in synthetic_fields]
    score = min((len(hits) - (0.4 * len(synthetic_hits))) / 8, 1.0)
    score = max(score, 0.0)
    if score >= 0.75:
        dataset_type = 'Claims or encounter-level healthcare dataset'
    elif score >= 0.45:
        dataset_type = 'Partially healthcare-related dataset'
    else:
        dataset_type = 'General tabular dataset with limited healthcare context'
    return {
        'healthcare_readiness_score': score,
        'likely_dataset_type': dataset_type,
        'matched_healthcare_fields': hits,
        'synthetic_supported_healthcare_fields': synthetic_hits,
    }


def utilization_analysis(data: pd.DataFrame, canonical_map: dict[str, str]) -> dict[str, object]:
    entity_col = _resolve_column(data, canonical_map, ['entity_id', 'patient_id', 'member_id'], ['entity_id', 'patient_id', 'member_id'])
    date_col = _resolve_column(data, canonical_map, ['event_date', 'service_date', 'admission_date'], ['event_date', 'service_date', 'admission_date'])
    category_col = _resolve_column(data, canonical_map, ['diagnosis_code', 'procedure_code', 'department', 'category'], ['diagnosis_code', 'procedure_code', 'department', 'category'])
    if not entity_col or not date_col:
        return {'available': False, 'reason': 'Needs an entity field and an event-style date field.'}

    frame = data[[entity_col, date_col] + ([category_col] if category_col else [])].copy()
    frame[date_col] = pd.to_datetime(frame[date_col], errors='coerce')
    frame = frame.dropna(subset=[entity_col, date_col])
    if frame.empty:
        return {'available': False, 'reason': 'No usable entity/date records remain after parsing.'}

    monthly = frame.assign(month=frame[date_col].dt.to_period('M').dt.to_timestamp()).groupby('month').size().reset_index(name='event_count')
    per_entity = frame.groupby(entity_col).size().reset_index(name='event_count').sort_values('event_count', ascending=False)
    top_category = pd.DataFrame()
    if category_col:
        category_frame = _clean_group_field(frame, category_col)
        if not category_frame.empty:
            top_category = category_frame.groupby(category_col).size().reset_index(name='event_count').sort_values('event_count', ascending=False).head(10)
    return {
        'available': True,
        'entity_column': entity_col,
        'date_column': date_col,
        'monthly_utilization': monthly,
        'events_per_entity': per_entity.head(20),
        'average_events_per_entity': float(per_entity['event_count'].mean()),
        'top_category_utilization': top_category,
    }


def cost_analysis(data: pd.DataFrame, canonical_map: dict[str, str]) -> dict[str, object]:
    cost_col = _resolve_column(data, canonical_map, ['cost_amount', 'paid_amount', 'allowed_amount', 'billed_amount'], ['cost_amount', 'paid_amount', 'allowed_amount', 'billed_amount', 'cost'])
    segment_col = _resolve_column(data, canonical_map, ['provider_name', 'facility', 'department', 'payer', 'diagnosis_code', 'procedure_code'], ['provider_name', 'facility', 'department', 'payer', 'diagnosis_code', 'procedure_code'])
    date_col = _resolve_column(data, canonical_map, ['event_date', 'service_date', 'admission_date'], ['event_date', 'service_date', 'admission_date'])
    if not cost_col:
        return {'available': False, 'reason': 'Needs a cost, payment, or billed amount field.'}

    frame = data.copy()
    frame[cost_col] = _safe_numeric(frame[cost_col])
    frame = frame.dropna(subset=[cost_col])
    if frame.empty:
        return {'available': False, 'reason': 'No usable numeric cost values remain after parsing.'}

    summary = {
        'total_cost': float(frame[cost_col].sum()),
        'average_cost': float(frame[cost_col].mean()),
        'median_cost': float(frame[cost_col].median()),
    }
    by_segment = pd.DataFrame()
    if segment_col:
        segment_frame = _clean_group_field(frame[[segment_col, cost_col]].copy(), segment_col)
        if not segment_frame.empty:
            by_segment = segment_frame.groupby(segment_col)[cost_col].agg(total_cost='sum', average_cost='mean', records='count').reset_index().sort_values('total_cost', ascending=False).head(15)
    outliers = frame[frame[cost_col] >= frame[cost_col].quantile(0.95)].copy().sort_values(cost_col, ascending=False).head(25)
    trend = pd.DataFrame()
    if date_col:
        trend_frame = frame[[date_col, cost_col]].copy()
        trend_frame[date_col] = pd.to_datetime(trend_frame[date_col], errors='coerce')
        trend_frame = trend_frame.dropna(subset=[date_col])
        if not trend_frame.empty:
            trend = trend_frame.assign(month=trend_frame[date_col].dt.to_period('M').dt.to_timestamp()).groupby('month')[cost_col].sum().reset_index(name='total_cost')
    return {'available': True, 'cost_column': cost_col, 'summary': summary, 'by_segment': by_segment, 'outliers': outliers, 'trend': trend}


def claims_cost_analyzer(data: pd.DataFrame, canonical_map: dict[str, str]) -> dict[str, object]:
    cost = cost_analysis(data, canonical_map)
    utilization = utilization_analysis(data, canonical_map)
    provider = provider_analysis(data, canonical_map)
    diagnosis = diagnosis_procedure_analysis(data, canonical_map)

    payer_col = _resolve_column(data, canonical_map, ['payer', 'plan'], ['payer', 'payor', 'insurance', 'plan'])
    cost_col = cost.get('cost_column') if cost.get('available') else _resolve_column(
        data,
        canonical_map,
        ['cost_amount', 'paid_amount', 'allowed_amount', 'billed_amount'],
        ['cost', 'cost_amount', 'paid_amount', 'allowed_amount', 'billed_amount'],
    )

    if not cost.get('available') and not utilization.get('available'):
        return {'available': False, 'reason': 'Needs a cost field and/or an encounter-style utilization structure to run the healthcare claims cost workflow.'}

    frame = data.copy()
    if cost_col and cost_col in frame.columns:
        frame[cost_col] = _safe_numeric(frame[cost_col])
        frame = frame.dropna(subset=[cost_col]).copy()

    payer_comparison = pd.DataFrame()
    if payer_col and cost_col and payer_col in frame.columns and cost_col in frame.columns and not frame.empty:
        payer_frame = _clean_group_field(frame[[payer_col, cost_col]].copy(), payer_col)
        if not payer_frame.empty:
            payer_comparison = payer_frame.groupby(payer_col)[cost_col].agg(
                total_cost='sum',
                average_cost='mean',
                median_cost='median',
                records='count',
            ).reset_index().sort_values('total_cost', ascending=False).head(15)

    high_cost_rows = pd.DataFrame()
    cost_anomaly_summary = pd.DataFrame()
    if cost_col and cost_col in frame.columns and not frame.empty:
        threshold = float(frame[cost_col].quantile(0.95))
        high_cost_rows = frame[frame[cost_col] >= threshold].copy().sort_values(cost_col, ascending=False).head(30)
        if not high_cost_rows.empty:
            cost_anomaly_summary = pd.DataFrame([
                {
                    'signal': 'High-cost rows',
                    'affected_rows': int(len(high_cost_rows)),
                    'threshold': threshold,
                    'recommended_investigation': 'Review the top-cost records for payer, provider, diagnosis, and utilization concentration.',
                }
            ])

    utilization_summary = pd.DataFrame()
    if utilization.get('available'):
        utilization_summary = pd.DataFrame([
            {
                'metric': 'Average events per entity',
                'value': float(utilization.get('average_events_per_entity', 0.0)),
            },
            {
                'metric': 'Tracked event records',
                'value': int(utilization.get('events_per_entity', pd.DataFrame()).get('event_count', pd.Series(dtype=float)).sum()) if isinstance(utilization.get('events_per_entity'), pd.DataFrame) else 0,
            },
        ])

    summary_rows: list[dict[str, object]] = []
    if cost.get('available'):
        summary_rows.extend([
            {'metric': 'Total cost', 'value': float(cost['summary']['total_cost'])},
            {'metric': 'Average cost', 'value': float(cost['summary']['average_cost'])},
            {'metric': 'Median cost', 'value': float(cost['summary']['median_cost'])},
        ])
    if utilization.get('available'):
        summary_rows.append({'metric': 'Average events per entity', 'value': float(utilization.get('average_events_per_entity', 0.0))})
    summary_table = pd.DataFrame(summary_rows)

    return {
        'available': True,
        'summary_table': summary_table,
        'cost_summary': cost.get('summary', {}),
        'cost_by_segment': cost.get('by_segment', pd.DataFrame()),
        'cost_trend': cost.get('trend', pd.DataFrame()),
        'high_cost_rows': high_cost_rows,
        'payer_comparison': payer_comparison,
        'provider_comparison': provider.get('table', pd.DataFrame()) if provider.get('available') else pd.DataFrame(),
        'diagnosis_cost_drivers': diagnosis.get('table', pd.DataFrame()) if diagnosis.get('available') else pd.DataFrame(),
        'utilization_summary': utilization_summary,
        'utilization_trend': utilization.get('monthly_utilization', pd.DataFrame()) if utilization.get('available') else pd.DataFrame(),
        'utilization_by_category': utilization.get('top_category_utilization', pd.DataFrame()) if utilization.get('available') else pd.DataFrame(),
        'cost_anomaly_summary': cost_anomaly_summary,
    }


def provider_analysis(data: pd.DataFrame, canonical_map: dict[str, str]) -> dict[str, object]:
    provider_col = _resolve_column(data, canonical_map, ['provider_name', 'facility', 'provider_id'], ['provider_name', 'facility', 'provider_id'])
    cost_col = _resolve_column(data, canonical_map, ['cost_amount', 'paid_amount', 'allowed_amount', 'billed_amount'], ['cost_amount', 'paid_amount', 'allowed_amount', 'billed_amount', 'cost'])
    if not provider_col:
        return {'available': False, 'reason': 'Needs a provider or facility field.'}
    frame = _clean_group_field(data.copy(), provider_col)
    if frame.empty:
        return {'available': False, 'reason': 'The detected provider or facility field has no usable values.'}
    metrics = frame.groupby(provider_col).size().reset_index(name='volume').sort_values('volume', ascending=False).head(20)
    if cost_col:
        frame[cost_col] = _safe_numeric(frame[cost_col])
        cost_summary = frame.groupby(provider_col)[cost_col].agg(total_cost='sum', average_cost='mean').reset_index()
        metrics = metrics.merge(cost_summary, on=provider_col, how='left').sort_values(['volume', 'total_cost'], ascending=[False, False])
    return {'available': True, 'provider_column': provider_col, 'table': metrics.head(20)}


def diagnosis_procedure_analysis(data: pd.DataFrame, canonical_map: dict[str, str]) -> dict[str, object]:
    clinical_col = _resolve_column(data, canonical_map, ['diagnosis_code', 'procedure_code', 'department'], ['diagnosis_code', 'procedure_code', 'department'])
    cost_col = _resolve_column(data, canonical_map, ['cost_amount', 'paid_amount', 'allowed_amount', 'billed_amount'], ['cost_amount', 'paid_amount', 'allowed_amount', 'billed_amount', 'cost'])
    if not clinical_col:
        return {'available': False, 'reason': 'Needs a diagnosis, procedure, or department field.'}
    frame = _clean_group_field(data.copy(), clinical_col)
    if frame.empty:
        return {'available': False, 'reason': 'The detected clinical grouping field has no usable values.'}
    grouped = frame.groupby(clinical_col).size().reset_index(name='volume').sort_values('volume', ascending=False).head(15)
    if cost_col:
        frame[cost_col] = _safe_numeric(frame[cost_col])
        cost_summary = frame.groupby(clinical_col)[cost_col].sum().reset_index(name='total_cost')
        grouped = grouped.merge(cost_summary, on=clinical_col, how='left').sort_values(['total_cost', 'volume'], ascending=[False, False])
    return {'available': True, 'clinical_column': clinical_col, 'table': grouped.head(15)}


def readmission_demo_analysis(data: pd.DataFrame, canonical_map: dict[str, str]) -> dict[str, object]:
    entity_col = _resolve_column(data, canonical_map, ['patient_id', 'entity_id', 'member_id'], ['patient_id', 'entity_id', 'member_id'])
    admit_col = _resolve_column(data, canonical_map, ['admission_date', 'service_date', 'event_date'], ['admission_date', 'service_date', 'event_date'])
    discharge_col = _resolve_column(data, canonical_map, ['discharge_date'], ['discharge_date'])
    if not entity_col or not admit_col:
        return {'available': False, 'reason': 'Needs a patient/entity field and an admission or service date field.'}

    columns = [entity_col, admit_col]
    if discharge_col:
        columns.append(discharge_col)
    frame = data[columns].copy()
    frame[admit_col] = pd.to_datetime(frame[admit_col], errors='coerce')
    if discharge_col:
        frame[discharge_col] = pd.to_datetime(frame[discharge_col], errors='coerce')
    frame = frame.dropna(subset=[entity_col, admit_col]).sort_values([entity_col, admit_col])
    if frame.empty:
        return {'available': False, 'reason': 'No usable encounter-style records remain after date parsing.'}

    base_date_col = discharge_col or admit_col
    frame['next_event_date'] = frame.groupby(entity_col)[admit_col].shift(-1)
    frame['index_date'] = frame[base_date_col] if discharge_col else frame[admit_col]
    frame['days_to_next_event'] = (frame['next_event_date'] - frame['index_date']).dt.days
    frame['approx_readmission_30d'] = frame['days_to_next_event'].between(0, 30, inclusive='both')
    rate = float(frame['approx_readmission_30d'].mean()) if len(frame) else 0.0
    return {
        'available': True,
        'approximate_readmission_rate': rate,
        'table': frame[[entity_col, admit_col] + ([discharge_col] if discharge_col else []) + ['days_to_next_event', 'approx_readmission_30d']].head(30),
        'note': 'This is generalized demo logic based on event timing. It is not production clinical readmission logic and should be interpreted cautiously.',
    }


def readmission_risk_analytics(data: pd.DataFrame, canonical_map: dict[str, str]) -> dict[str, object]:
    entity_col = _resolve_column(data, canonical_map, ['patient_id', 'entity_id', 'member_id'], ['patient_id', 'entity_id', 'member_id'])
    admit_col = _resolve_column(data, canonical_map, ['admission_date', 'service_date', 'event_date'], ['admission_date', 'service_date', 'event_date'])
    discharge_col = _resolve_column(data, canonical_map, ['discharge_date'], ['discharge_date'])
    age_col = _resolve_column(data, canonical_map, ['age'], ['age'])
    diagnosis_col = _resolve_column(data, canonical_map, ['diagnosis_code'], ['diagnosis', 'diagnosis_code', 'dx', 'dx_code'])
    department_col = _resolve_column(data, canonical_map, ['department'], ['department', 'dept', 'service_line', 'unit'])
    los_col = _resolve_column(data, canonical_map, ['length_of_stay'], ['length_of_stay', 'los', 'stay_days'])
    cost_col = _resolve_column(data, canonical_map, ['cost_amount', 'paid_amount', 'allowed_amount', 'billed_amount'], ['cost', 'cost_amount', 'paid_amount', 'allowed_amount', 'billed_amount'])
    smoking_col = _resolve_column(data, canonical_map, ['smoking_status'], ['smoking_status', 'smoker'])
    comorbidity_col = _resolve_column(data, canonical_map, ['comorbidities'], ['comorbidities', 'comorbidity_score'])
    treatment_col = _resolve_column(data, canonical_map, ['treatment_type'], ['treatment_type', 'treatment', 'therapy'])

    present_fields = {
        'patient_id': entity_col,
        'event_date': admit_col,
        'discharge_date': discharge_col,
        'readmission_flag': None,
        'age': age_col,
        'diagnosis': diagnosis_col,
        'department': department_col,
        'length_of_stay': los_col,
        'cost': cost_col,
    }
    readmission_values, readmission_col = _readmission_flag(data, canonical_map)
    present_fields['readmission_flag'] = readmission_col

    missing_for_full = [label for label, column in present_fields.items() if label in {'patient_id', 'event_date', 'readmission_flag', 'age', 'diagnosis', 'department', 'length_of_stay'} and not column]
    analyzable = ['Overall readmission rate']
    if department_col:
        analyzable.append('Readmission by department')
    if diagnosis_col:
        analyzable.append('Readmission by diagnosis')
    if age_col:
        analyzable.append('Readmission by age band')
    if los_col or (admit_col and discharge_col):
        analyzable.append('Readmission by length-of-stay band')
    if admit_col:
        analyzable.append('Readmission trend over time')

    if readmission_values is None and not (entity_col and admit_col):
        return {
            'available': False,
            'reason': 'Needs either a readmission flag or an encounter-style patient/date structure to run readmission analytics.',
            'missing_fields': ['readmission flag or patient/date structure'],
            'available_analysis': analyzable,
            'additional_fields_to_unlock_full_analysis': missing_for_full,
        }

    frame_cols = [col for col in [entity_col, admit_col, discharge_col, age_col, diagnosis_col, department_col, los_col, cost_col, smoking_col, comorbidity_col, treatment_col] if col]
    frame = data[frame_cols].copy()
    if admit_col:
        frame[admit_col] = pd.to_datetime(frame[admit_col], errors='coerce')
    if discharge_col:
        frame[discharge_col] = pd.to_datetime(frame[discharge_col], errors='coerce')

    derived_flag_note = None
    source_label = 'native'
    if readmission_values is not None:
        frame['readmission_flag'] = readmission_values
        frame = frame[frame['readmission_flag'].notna()].copy()
        derived_flag_note = f'Using {readmission_col} as the readmission indicator.'
        if 'readmission_source' in data.columns and data['readmission_source'].astype(str).str.contains('synthetic', case=False, na=False).any():
            derived_flag_note = 'Using a synthetic, deterministic readmission flag to support demo workflow analysis.'
            source_label = 'synthetic'
    else:
        frame = frame.dropna(subset=[entity_col, admit_col]).sort_values([entity_col, admit_col]).copy()
        base_date_col = discharge_col or admit_col
        frame['next_event_date'] = frame.groupby(entity_col)[admit_col].shift(-1)
        frame['index_date'] = frame[base_date_col] if discharge_col else frame[admit_col]
        frame['days_to_next_event'] = (frame['next_event_date'] - frame['index_date']).dt.days
        frame['readmission_flag'] = frame['days_to_next_event'].between(0, 30, inclusive='both').astype(float)
        derived_flag_note = 'Derived a generalized 30-day readmission flag from repeat encounters within 30 days.'
        source_label = 'derived'

    if frame.empty or not frame['readmission_flag'].notna().any():
        return {
            'available': False,
            'reason': 'No usable readmission records remain after parsing the current fields.',
            'missing_fields': missing_for_full,
            'available_analysis': analyzable,
            'additional_fields_to_unlock_full_analysis': missing_for_full,
        }

    if los_col:
        frame[los_col] = _safe_numeric(frame[los_col])
    elif admit_col and discharge_col:
        frame['derived_length_of_stay'] = (frame[discharge_col] - frame[admit_col]).dt.days
        los_col = 'derived_length_of_stay'
    if age_col:
        frame[age_col] = _safe_numeric(frame[age_col])
        frame['age_band'] = _band_numeric(frame[age_col], [0, 44, 64, 79, 200], ['18-44', '45-64', '65-79', '80+'])
    if los_col and los_col in frame.columns:
        frame['los_band'] = _band_numeric(frame[los_col], [-1, 2, 5, 8, 15, 10_000], ['0-2 days', '3-5 days', '6-8 days', '9-15 days', '15+ days'])

    overall_rate = float(frame['readmission_flag'].mean())
    overview = {
        'overall_readmission_rate': overall_rate,
        'readmission_count': int(frame['readmission_flag'].sum()),
        'records_in_scope': int(len(frame)),
    }

    by_department = _readmission_group_rate(frame, department_col) if department_col else pd.DataFrame()
    by_diagnosis = _readmission_group_rate(frame, diagnosis_col) if diagnosis_col else pd.DataFrame()
    by_age_band = _readmission_group_rate(frame, 'age_band') if 'age_band' in frame.columns else pd.DataFrame()
    by_los_band = _readmission_group_rate(frame, 'los_band') if 'los_band' in frame.columns else pd.DataFrame()

    trend = pd.DataFrame()
    if admit_col:
        trend_frame = frame[[admit_col, 'readmission_flag']].dropna(subset=[admit_col]).copy()
        if not trend_frame.empty:
            trend = trend_frame.assign(month=trend_frame[admit_col].dt.to_period('M').dt.to_timestamp()).groupby('month').agg(
                record_count=('readmission_flag', 'size'),
                readmission_rate=('readmission_flag', 'mean'),
            ).reset_index()

    segment_rows: list[dict[str, object]] = []
    for label, group_col in [('Department', department_col), ('Diagnosis', diagnosis_col), ('Age Band', 'age_band' if 'age_band' in frame.columns else None), ('LOS Band', 'los_band' if 'los_band' in frame.columns else None)]:
        if not group_col or group_col not in frame.columns:
            continue
        grouped = _readmission_group_rate(frame, group_col)
        if grouped.empty:
            continue
        standout = grouped[grouped['record_count'] >= 5].head(5)
        for _, row in standout.iterrows():
            gap = float(row['readmission_rate'] - overall_rate)
            if gap <= 0:
                continue
            segment_rows.append({
                'segment_type': label,
                'segment_value': row[group_col],
                'record_count': int(row['record_count']),
                'readmission_rate': float(row['readmission_rate']),
                'gap_vs_overall': gap,
                'suggested_next_action': f"Review {label.lower()} follow-up planning for {row[group_col]}.",
            })
    high_risk_segments = pd.DataFrame(segment_rows).sort_values(['gap_vs_overall', 'record_count'], ascending=[False, False]).head(12).reset_index(drop=True) if segment_rows else pd.DataFrame()

    if smoking_col and smoking_col in frame.columns:
        smoking_text = frame[smoking_col].astype(str).str.strip().str.lower()
    else:
        smoking_text = pd.Series('', index=frame.index)
    if comorbidity_col and comorbidity_col in frame.columns:
        comorbidity_present = _comorbidity_present(frame[comorbidity_col])
    else:
        comorbidity_present = pd.Series(False, index=frame.index)

    patient_score = pd.Series(0.0, index=frame.index)
    if age_col and age_col in frame.columns:
        patient_score += (frame[age_col] >= 65).fillna(False).astype(int)
    if los_col and los_col in frame.columns:
        patient_score += (pd.to_numeric(frame[los_col], errors='coerce') >= 6).fillna(False).astype(int)
    patient_score += smoking_text.str.contains('smoker|current|yes', regex=True, na=False).astype(int)
    patient_score += comorbidity_present.astype(int)
    if department_col and not by_department.empty:
        dept_rates = by_department.set_index(department_col)['readmission_rate']
        patient_score += frame[department_col].map(dept_rates).fillna(0).ge(overall_rate + 0.05).astype(int)
    if diagnosis_col and not by_diagnosis.empty:
        diag_rates = by_diagnosis.set_index(diagnosis_col)['readmission_rate']
        patient_score += frame[diagnosis_col].map(diag_rates).fillna(0).ge(overall_rate + 0.05).astype(int)

    frame['readmission_risk_score'] = patient_score
    frame['readmission_risk_segment'] = pd.cut(
        frame['readmission_risk_score'],
        bins=[-1, 1, 3, 10],
        labels=['Low Risk', 'Medium Risk', 'High Risk'],
    ).astype(str)
    patient_columns = [col for col in [entity_col, admit_col, discharge_col, age_col, diagnosis_col, department_col, los_col, treatment_col] if col and col in frame.columns]
    high_risk_patients = frame[patient_columns + ['readmission_flag', 'readmission_risk_score', 'readmission_risk_segment']].sort_values(
        ['readmission_risk_score', 'readmission_flag'],
        ascending=[False, False],
    ).head(50)

    driver_rows: list[dict[str, object]] = []
    for label, table, group_col in [
        ('Department', by_department, department_col),
        ('Diagnosis', by_diagnosis, diagnosis_col),
        ('Age Band', by_age_band, 'age_band'),
        ('LOS Band', by_los_band, 'los_band'),
    ]:
        if table.empty or not group_col:
            continue
        top = table.iloc[0]
        driver_rows.append({
            'factor': label,
            'driver_group': top[group_col],
            'readmission_rate': float(top['readmission_rate']),
            'overall_rate': overall_rate,
            'gap_vs_overall': float(top['readmission_rate'] - overall_rate),
            'record_count': int(top['record_count']),
        })
    if smoking_col and smoking_col in frame.columns:
        smoking_summary = frame.assign(
            smoking_group=smoking_text.apply(lambda value: 'Smoker' if re.search('smoker|current|yes', value) else 'Other')
        ).groupby('smoking_group').agg(record_count=('readmission_flag', 'size'), readmission_rate=('readmission_flag', 'mean')).reset_index()
        if not smoking_summary.empty:
            top = smoking_summary.sort_values('readmission_rate', ascending=False).iloc[0]
            driver_rows.append({
                'factor': 'Smoking Status',
                'driver_group': top['smoking_group'],
                'readmission_rate': float(top['readmission_rate']),
                'overall_rate': overall_rate,
                'gap_vs_overall': float(top['readmission_rate'] - overall_rate),
                'record_count': int(top['record_count']),
            })
    readmission_driver_table = pd.DataFrame(driver_rows).sort_values('gap_vs_overall', ascending=False).reset_index(drop=True) if driver_rows else pd.DataFrame()
    driver_interpretation = (
        f"{readmission_driver_table.iloc[0]['factor']} is the strongest visible readmission driver in the current selection, "
        f"with {readmission_driver_table.iloc[0]['driver_group']} showing a readmission rate of {readmission_driver_table.iloc[0]['readmission_rate']:.1%}."
        if not readmission_driver_table.empty else
        'Readmission drivers are limited because the current dataset does not contain enough segmentation detail.'
    )

    return {
        'available': True,
        'readiness': {
            'present_fields': {key: value for key, value in present_fields.items() if value},
            'missing_fields': missing_for_full,
            'available_analysis': analyzable,
            'additional_fields_to_unlock_full_analysis': [field for field in ['readmission_flag', 'age', 'diagnosis', 'department', 'length_of_stay', 'cost'] if field in missing_for_full],
            'badge_text': 'Full readmission workflow available' if not missing_for_full else 'Partial readmission workflow available',
        },
        'overview': overview,
        'by_department': by_department,
        'by_diagnosis': by_diagnosis,
        'by_age_band': by_age_band,
        'by_los_band': by_los_band,
        'trend': trend,
        'high_risk_segments': high_risk_segments,
        'high_risk_patients': high_risk_patients,
        'driver_table': readmission_driver_table,
        'driver_interpretation': driver_interpretation,
        'working_frame': frame,
        'entity_column': entity_col,
        'date_column': admit_col,
        'diagnosis_column': diagnosis_col,
        'department_column': department_col,
        'age_column': age_col,
        'los_column': los_col,
        'note': derived_flag_note,
        'source': source_label,
    }


def build_readmission_cohort_review(readmission: dict[str, object], cohort_name: str) -> dict[str, object]:
    if not readmission.get('available'):
        return {'available': False, 'reason': 'Readmission analytics are not available for the current dataset.'}
    frame = readmission.get('working_frame', pd.DataFrame()).copy()
    if frame.empty:
        return {'available': False, 'reason': 'No row-level readmission detail is available for cohort review.'}

    age_col = readmission.get('age_column')
    diagnosis_col = readmission.get('diagnosis_column')
    los_col = readmission.get('los_column')
    overall_rate = float(readmission.get('overview', {}).get('overall_readmission_rate', 0.0))

    if cohort_name == 'Older Adults (65+)':
        if not age_col or age_col not in frame.columns:
            return {'available': False, 'reason': 'An age field is needed for the older-adult readmission cohort.'}
        cohort = frame[pd.to_numeric(frame[age_col], errors='coerce') >= 65].copy()
        suggested = 'Prioritize post-discharge outreach and medication reconciliation for older adults with elevated readmission exposure.'
    elif cohort_name == 'High LOS Patients (6+ days)':
        if not los_col or los_col not in frame.columns:
            return {'available': False, 'reason': 'A length-of-stay field is needed for the high-LOS readmission cohort.'}
        cohort = frame[pd.to_numeric(frame[los_col], errors='coerce') >= 6].copy()
        suggested = 'Review inpatient discharge planning and step-down follow-up for longer-stay patients.'
    elif cohort_name == 'Top Diagnosis Group':
        by_diagnosis = readmission.get('by_diagnosis', pd.DataFrame())
        if by_diagnosis.empty or not diagnosis_col or diagnosis_col not in frame.columns:
            return {'available': False, 'reason': 'A diagnosis field is needed for diagnosis-focused readmission cohorts.'}
        top_value = str(by_diagnosis.iloc[0][diagnosis_col])
        cohort = frame[frame[diagnosis_col].astype(str) == top_value].copy()
        suggested = f'Review discharge and follow-up pathways for the top diagnosis group: {top_value}.'
    elif cohort_name == 'Highest Readmission Segment':
        segments = readmission.get('high_risk_segments', pd.DataFrame())
        if segments.empty:
            return {'available': False, 'reason': 'No standout readmission segments were detected for this dataset.'}
        top = segments.iloc[0]
        segment_type = str(top['segment_type'])
        segment_value = str(top['segment_value'])
        column_lookup = {
            'Department': readmission.get('department_column'),
            'Diagnosis': readmission.get('diagnosis_column'),
            'Age Band': 'age_band',
            'LOS Band': 'los_band',
        }
        target_col = column_lookup.get(segment_type)
        if not target_col or target_col not in frame.columns:
            return {'available': False, 'reason': 'The standout segment could not be mapped back to the current row-level dataset.'}
        cohort = frame[frame[target_col].astype(str) == segment_value].copy()
        suggested = f'Use targeted case management and follow-up for the standout {segment_type.lower()} segment: {segment_value}.'
    else:
        cohort = frame.copy()
        suggested = 'Use the overall readmission summary to identify the next service line or diagnosis cohort for follow-up review.'

    if cohort.empty:
        return {'available': False, 'reason': 'The selected readmission cohort does not contain any matching records.'}

    readmission_rate = float(cohort['readmission_flag'].mean())
    summary = {
        'cohort_name': cohort_name,
        'cohort_size': int(len(cohort)),
        'readmission_rate': readmission_rate,
        'overall_population_rate': overall_rate,
        'gap_vs_overall': readmission_rate - overall_rate,
    }
    preview_cols = [col for col in [readmission.get('entity_column'), readmission.get('date_column'), age_col, diagnosis_col, readmission.get('department_column'), los_col] if col and col in cohort.columns]
    preview = cohort[preview_cols + ['readmission_flag', 'readmission_risk_score', 'readmission_risk_segment']].head(40)
    return {
        'available': True,
        'summary': summary,
        'preview': preview,
        'suggested_next_action': suggested,
    }


def plan_readmission_intervention(
    readmission: dict[str, object],
    cohort_name: str,
    enhanced_follow_up: float,
    targeted_case_management: float,
    los_reduction_days: float,
    early_follow_up_improvement: float,
) -> dict[str, object]:
    cohort = build_readmission_cohort_review(readmission, cohort_name)
    if not cohort.get('available'):
        return {'available': False, 'reason': cohort.get('reason', 'Readmission intervention planning is not available for the current dataset.')}

    summary = cohort['summary']
    baseline_rate = float(summary['readmission_rate'])
    cohort_size = int(summary['cohort_size'])
    overall_rate = float(summary['overall_population_rate'])
    reduction = 0.0
    assumptions: list[str] = []

    if enhanced_follow_up > 0:
        reduction += min((enhanced_follow_up / 100.0) * 0.10, 0.10)
        assumptions.append(f'Enhanced discharge follow-up is assumed to reduce readmissions by up to 10% of the targeted cohort effect size.')
    if targeted_case_management > 0:
        reduction += min((targeted_case_management / 100.0) * 0.12, 0.12)
        assumptions.append('Targeted case management is modeled as a moderate reduction lever for high-risk discharges.')
    if los_reduction_days > 0:
        reduction += min(los_reduction_days * 0.01, 0.05)
        assumptions.append('Reducing average LOS is treated as a small proxy for smoother discharge readiness and handoff quality.')
    if early_follow_up_improvement > 0:
        reduction += min((early_follow_up_improvement / 100.0) * 0.08, 0.08)
        assumptions.append('Improved early follow-up completion is modeled as a smaller but still meaningful readmission reducer.')

    reduction = min(reduction, 0.35)
    simulated_rate = max(baseline_rate * (1.0 - reduction), 0.0)
    avoided = max((baseline_rate - simulated_rate) * cohort_size, 0.0)
    overall_after = max(overall_rate - (((baseline_rate - simulated_rate) * cohort_size) / max(int(readmission.get('overview', {}).get('records_in_scope', cohort_size)), 1)), 0.0)

    summary_table = pd.DataFrame([
        {'metric': 'Baseline readmission rate', 'value': baseline_rate},
        {'metric': 'Projected readmission rate', 'value': simulated_rate},
        {'metric': 'Estimated readmissions avoided', 'value': avoided},
        {'metric': 'Projected overall readmission rate', 'value': overall_after},
    ])
    return {
        'available': True,
        'summary_table': summary_table,
        'baseline_readmission_rate': baseline_rate,
        'projected_readmission_rate': simulated_rate,
        'projected_overall_readmission_rate': overall_after,
        'estimated_readmissions_avoided': avoided,
        'assumptions': assumptions or ['No intervention levers were applied, so the projected rate remains at baseline.'],
        'target_cohort': cohort_name,
    }


def patient_risk_segmentation(data: pd.DataFrame, canonical_map: dict[str, str]) -> dict[str, object]:
    frame = _risk_detail_frame(data, canonical_map)
    if frame.empty:
        return {'available': False, 'reason': 'Needs at least one patient risk factor such as age, smoking status, cancer stage, or comorbidities.'}

    age_col = _resolve_column(frame, canonical_map, ['age'], ['age'])
    outcome_col = _resolve_column(frame, canonical_map, ['survived'], ['survived', 'alive', 'outcome'])
    summary = frame.groupby('risk_segment').size().reset_index(name='patient_count')
    summary['percentage'] = summary['patient_count'] / max(len(frame), 1)
    if age_col:
        summary = summary.merge(frame.groupby('risk_segment')[age_col].apply(lambda s: pd.to_numeric(s, errors='coerce').mean()).reset_index(name='average_age'), on='risk_segment', how='left')
    survival_rate = None
    if outcome_col and 'survived_binary' in frame.columns:
        summary = summary.merge(frame.groupby('risk_segment')['survived_binary'].mean().reset_index(name='survival_rate'), on='risk_segment', how='left')
        survival_rate = float(frame['survived_binary'].mean()) if frame['survived_binary'].notna().any() else None
    detail_cols = [col for col in [_resolve_column(frame, canonical_map, ['age'], ['age']), _resolve_column(frame, canonical_map, ['smoking_status'], ['smoking_status', 'smoker']), _resolve_column(frame, canonical_map, ['cancer_stage'], ['cancer_stage', 'stage']), _resolve_column(frame, canonical_map, ['comorbidities'], ['comorbidities', 'comorbidity_score']), outcome_col] if col]
    return {
        'available': True,
        'segment_table': summary.sort_values('patient_count', ascending=False),
        'detail_table': frame[detail_cols + ['risk_score', 'risk_segment']].head(100),
        'survival_rate': survival_rate,
        'detail_frame': frame,
    }


def _append_anomaly_summary(summary_rows: list[dict[str, object]], field_label: str, source_column: str, anomaly_type: str, affected_rows: int, total_rows: int, anomaly_score: float, recommended_investigation: str) -> None:
    summary_rows.append({
        'field': field_label,
        'source_column': source_column,
        'anomaly_type': anomaly_type,
        'anomaly_label': f'{field_label} · {anomaly_type}',
        'affected_rows': int(affected_rows),
        'anomaly_count': int(affected_rows),
        'anomaly_rate': float(affected_rows / max(total_rows, 1)),
        'anomaly_score': float(anomaly_score),
        'recommended_investigation': recommended_investigation,
    })


def anomaly_detection(data: pd.DataFrame, canonical_map: dict[str, str]) -> dict[str, object]:
    anomaly_fields = []
    for canonical, raw_names in [('age', ['age']), ('bmi', ['bmi']), ('cholesterol_level', ['cholesterol_level', 'cholesterol'])]:
        column = _resolve_column(data, canonical_map, [canonical], raw_names)
        if column:
            anomaly_fields.append((canonical, column))

    categorical_candidates = []
    for canonical, raw_names in [('smoking_status', ['smoking_status', 'smoker']), ('cancer_stage', ['cancer_stage', 'stage']), ('treatment_type', ['treatment_type', 'treatment', 'therapy']), ('gender', ['gender', 'sex'])]:
        column = _resolve_column(data, canonical_map, [canonical], raw_names)
        if column:
            categorical_candidates.append((canonical, column))

    date_col = _resolve_column(data, canonical_map, ['event_date', 'service_date', 'admission_date', 'diagnosis_date'], ['event_date', 'service_date', 'admission_date', 'diagnosis_date'])

    if not anomaly_fields and not categorical_candidates and not date_col:
        return {'available': False, 'reason': 'Needs usable numeric, categorical, or date fields to run anomaly detection.'}

    summary_rows: list[dict[str, object]] = []
    detail_rows: list[pd.DataFrame] = []

    for canonical, column in anomaly_fields:
        numeric = _safe_numeric(data[column])
        valid = numeric.dropna()
        if len(valid) < 8 or float(valid.std()) == 0.0:
            continue
        z_scores = ((valid - valid.mean()) / valid.std()).abs()
        q1 = valid.quantile(0.25)
        q3 = valid.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        anomalies = valid[(z_scores > 3) | (valid < lower) | (valid > upper)]
        if anomalies.empty:
            continue
        _append_anomaly_summary(summary_rows, canonical, column, 'Numeric Outlier', len(anomalies), len(valid), float(z_scores.loc[anomalies.index].max()), 'Validate the extreme values and confirm whether they represent valid clinical edge cases or data entry issues.')
        detail = pd.DataFrame({
            'field': canonical,
            'source_column': column,
            'anomaly_type': 'Numeric Outlier',
            'row_index': anomalies.index.astype(int),
            'value': anomalies.values,
            'anomaly_score': z_scores.loc[anomalies.index].values,
            'recommended_investigation': 'Review the flagged numeric values for data entry, unit, or measurement issues.'
        })
        detail_rows.append(detail)

    for canonical, column in categorical_candidates:
        series = data[column].dropna().astype(str).str.strip()
        series = series[series != '']
        if len(series) < 20:
            continue
        counts = series.value_counts()
        shares = counts / max(len(series), 1)
        rare_values = shares[(shares <= 0.02) & (counts <= max(3, int(len(series) * 0.02)))]
        if not rare_values.empty:
            affected_rows = int(counts.loc[rare_values.index].sum())
            _append_anomaly_summary(summary_rows, canonical, column, 'Rare Value Pattern', affected_rows, len(series), float(min(rare_values.shape[0] / 3, 1.0)), 'Review rare categories to confirm whether they are valid subgroup labels, spelling variants, or mapping issues.')
            rare_detail = pd.DataFrame({
                'field': canonical,
                'source_column': column,
                'anomaly_type': 'Rare Value Pattern',
                'row_index': [-1] * len(rare_values),
                'value': rare_values.index.tolist(),
                'anomaly_score': rare_values.values.tolist(),
                'recommended_investigation': ['Review these rare values for coding consistency.'] * len(rare_values),
            })
            detail_rows.append(rare_detail)

        if len(counts) >= 2:
            top_share = float(shares.iloc[0])
            second_share = float(shares.iloc[1]) if len(shares) > 1 else 0.0
            median_share = float(shares.median()) if not shares.empty else 0.0
            if top_share >= 0.60 and (second_share == 0.0 or top_share >= max(second_share * 2.5, median_share * 3)):
                _append_anomaly_summary(summary_rows, canonical, column, 'Category Frequency Spike', int(counts.iloc[0]), len(series), top_share, 'Confirm whether one category is genuinely dominant or whether upstream coding collapsed multiple groups into a single value.')
                spike_detail = pd.DataFrame({
                    'field': [canonical],
                    'source_column': [column],
                    'anomaly_type': ['Category Frequency Spike'],
                    'row_index': [-1],
                    'value': [counts.index[0]],
                    'anomaly_score': [top_share],
                    'recommended_investigation': ['Review the dominant category for coding drift or one-value collapse.'],
                })
                detail_rows.append(spike_detail)

    if date_col:
        parsed_dates = pd.to_datetime(data[date_col], errors='coerce')
        valid_dates = parsed_dates.dropna()
        if len(valid_dates) >= 20:
            monthly_counts = parsed_dates.dropna().dt.to_period('M').dt.to_timestamp().value_counts().sort_index()
            if len(monthly_counts) >= 4 and float(monthly_counts.std()) > 0.0:
                count_z = ((monthly_counts - monthly_counts.mean()) / monthly_counts.std()).abs()
                spiky_periods = monthly_counts[count_z > 2.5]
                if not spiky_periods.empty:
                    _append_anomaly_summary(summary_rows, 'record volume', date_col, 'Time-Series Spike', int(spiky_periods.sum()), int(monthly_counts.sum()), float(count_z.loc[spiky_periods.index].max()), 'Review time periods with unusual record volume to confirm whether they reflect operational events, backfills, or ingestion issues.')
                    spike_detail = pd.DataFrame({
                        'field': 'record volume',
                        'source_column': date_col,
                        'anomaly_type': 'Time-Series Spike',
                        'row_index': -1,
                        'value': spiky_periods.index.astype(str),
                        'anomaly_score': count_z.loc[spiky_periods.index].values,
                        'recommended_investigation': 'Review months with unusual record volume for backlog loads, seasonal spikes, or extract issues.',
                    })
                    detail_rows.append(spike_detail)

            for canonical, column in anomaly_fields:
                frame = pd.DataFrame({'period': parsed_dates, 'metric': _safe_numeric(data[column])}).dropna()
                if len(frame) < 20:
                    continue
                monthly_metric = frame.assign(month=frame['period'].dt.to_period('M').dt.to_timestamp()).groupby('month')['metric'].mean()
                if len(monthly_metric) < 4:
                    continue
                baseline = monthly_metric.iloc[:-1]
                latest = float(monthly_metric.iloc[-1])
                if baseline.empty or float(baseline.std()) == 0.0:
                    continue
                shift_score = abs(latest - float(baseline.mean())) / float(baseline.std())
                if shift_score > 2.5:
                    _append_anomaly_summary(summary_rows, canonical, column, 'Distribution Shift', 1, len(monthly_metric), float(shift_score), 'Review recent shifts in the average value to confirm whether they reflect real population change, measurement drift, or extraction changes.')
                    shift_detail = pd.DataFrame({
                        'field': [canonical],
                        'source_column': [column],
                        'anomaly_type': ['Distribution Shift'],
                        'row_index': [-1],
                        'value': [str(monthly_metric.index[-1])],
                        'anomaly_score': [float(shift_score)],
                        'recommended_investigation': ['Compare the latest period with the earlier baseline to understand whether the shift is operational, clinical, or data-quality related.'],
                    })
                    detail_rows.append(shift_detail)

    if not summary_rows:
        return {'available': False, 'reason': 'No unusual numeric, categorical, or time-based patterns were flagged by the anomaly rules.'}

    summary_table = pd.DataFrame(summary_rows).sort_values(['anomaly_score', 'affected_rows'], ascending=[False, False]).reset_index(drop=True)
    detail_table = pd.concat(detail_rows, ignore_index=True) if detail_rows else pd.DataFrame()
    if not detail_table.empty:
        detail_table = detail_table.sort_values('anomaly_score', ascending=False).head(150)
    return {
        'available': True,
        'summary_table': summary_table,
        'detail_table': detail_table,
    }

def ai_insight_summary(data: pd.DataFrame, canonical_map: dict[str, str], risk_summary: dict[str, object]) -> list[str]:
    insights: list[str] = []
    age_col = _resolve_column(data, canonical_map, ['age'], ['age'])
    stage_col = _resolve_column(data, canonical_map, ['cancer_stage'], ['cancer_stage', 'stage'])
    smoking_col = _resolve_column(data, canonical_map, ['smoking_status'], ['smoking_status', 'smoker'])
    treatment_col = _resolve_column(data, canonical_map, ['treatment_type'], ['treatment_type', 'treatment', 'therapy'])
    outcome_col = _resolve_column(data, canonical_map, ['survived'], ['survived', 'alive', 'outcome'])

    if outcome_col:
        outcome = _binary_outcome(data[outcome_col])
        if outcome.notna().any():
            insights.append(f"Overall survival rate is {float(outcome.mean()):.1%} across the current dataset.")
    if age_col and outcome_col:
        age_numeric = _safe_numeric(data[age_col])
        older = outcome[age_numeric > 60]
        younger = outcome[age_numeric <= 60]
        if older.notna().any() and younger.notna().any():
            insights.append(f"Patients older than 60 show a survival rate of {float(older.mean()):.1%} versus {float(younger.mean()):.1%} for younger patients.")
    if smoking_col and outcome_col:
        smoking_text = data[smoking_col].astype(str).str.strip().str.lower()
        smoker_rate = outcome[smoking_text.str.contains('smoker|current|yes', regex=True, na=False)]
        non_rate = outcome[~smoking_text.str.contains('smoker|current|yes', regex=True, na=False)]
        if smoker_rate.notna().any() and non_rate.notna().any():
            insights.append(f"Smoking status is a visible differentiator: likely smokers show {float(smoker_rate.mean()):.1%} survival versus {float(non_rate.mean()):.1%} for other patients.")
    if stage_col and outcome_col:
        stage_frame = pd.DataFrame({'stage': data[stage_col].astype(str), 'outcome': outcome}).dropna()
        if not stage_frame.empty:
            grouped = stage_frame.groupby('stage')['outcome'].mean().sort_values()
            if not grouped.empty:
                insights.append(f"The weakest observed survival appears in {grouped.index[0]}, at {float(grouped.iloc[0]):.1%}.")
    if treatment_col and outcome_col:
        treatment_frame = pd.DataFrame({'treatment': data[treatment_col].astype(str), 'outcome': outcome}).dropna()
        if not treatment_frame.empty:
            grouped = treatment_frame.groupby('treatment')['outcome'].mean().sort_values(ascending=False)
            if not grouped.empty:
                insights.append(f"{grouped.index[0]} currently shows the strongest observed survival rate at {float(grouped.iloc[0]):.1%}.")
    if risk_summary.get('available') and not risk_summary['segment_table'].empty:
        top_risk = risk_summary['segment_table'].sort_values('patient_count', ascending=False).iloc[0]
        insights.append(f"The largest patient segment is {top_risk['risk_segment']}, representing {float(top_risk['percentage']):.1%} of reviewed records.")
    return insights[:5]


def build_cohort_summary(
    data: pd.DataFrame,
    canonical_map: dict[str, str],
    age_range: tuple[int, int] | None = None,
    age_bands: list[str] | None = None,
    genders: list[str] | None = None,
    diagnoses: list[str] | None = None,
    treatments: list[str] | None = None,
    smoking_statuses: list[str] | None = None,
    cancer_stages: list[str] | None = None,
    risk_segments: list[str] | None = None,
    comorbidity_filters: list[str] | None = None,
) -> dict[str, object]:
    prepared = _prepare_clinical_frame(data, canonical_map)
    frame = prepared['frame'].copy()
    age_col = prepared['age_col']
    gender_col = prepared['gender_col']
    smoking_col = prepared['smoking_col']
    stage_col = prepared['stage_col']
    treatment_col = prepared['treatment_col']
    outcome_col = prepared['outcome_col']
    diagnosis_col = _resolve_column(frame, canonical_map, ['diagnosis_code'], ['diagnosis_code', 'diagnosis', 'dx'])
    comorbidity_col = _resolve_column(frame, canonical_map, ['comorbidities'], ['comorbidities', 'comorbidity_score'])

    if age_range and age_col:
        age_numeric = _safe_numeric(frame[age_col])
        frame = frame[(age_numeric >= age_range[0]) & (age_numeric <= age_range[1])]
    if age_bands and 'age_band' in frame.columns:
        frame = frame[frame['age_band'].astype(str).isin(age_bands)]
    if genders and gender_col:
        frame = frame[frame[gender_col].astype(str).isin(genders)]
    if diagnoses and diagnosis_col:
        frame = frame[frame[diagnosis_col].astype(str).isin(diagnoses)]
    if treatments and treatment_col:
        frame = frame[frame[treatment_col].astype(str).isin(treatments)]
    if smoking_statuses and smoking_col:
        frame = frame[frame[smoking_col].astype(str).isin(smoking_statuses)]
    if cancer_stages and stage_col:
        frame = frame[frame[stage_col].astype(str).isin(cancer_stages)]
    if risk_segments and 'risk_segment' in frame.columns:
        frame = frame[frame['risk_segment'].astype(str).isin(risk_segments)]
    if comorbidity_filters and comorbidity_col:
        presence = _comorbidity_present(frame[comorbidity_col])
        labels = presence.map({True: 'Present', False: 'Absent'})
        frame = frame[labels.isin(comorbidity_filters)]

    if frame.empty:
        return {'available': False, 'reason': 'The current cohort filters produce no matching records.'}

    summary = {
        'cohort_size': int(len(frame)),
        'average_age': float(_safe_numeric(frame[age_col]).mean()) if age_col else None,
        'average_risk_score': float(pd.to_numeric(frame['risk_score'], errors='coerce').mean()) if 'risk_score' in frame.columns else None,
        'high_risk_share': float((frame['risk_segment'].astype(str) == 'High Risk').mean()) if 'risk_segment' in frame.columns else None,
        'average_treatment_duration_days': float(pd.to_numeric(frame['treatment_duration_days'], errors='coerce').mean()) if 'treatment_duration_days' in frame.columns and frame['treatment_duration_days'].notna().any() else None,
    }
    if outcome_col:
        outcome = frame['survived_binary'] if 'survived_binary' in frame.columns else _binary_outcome(frame[outcome_col])
        summary['survival_rate'] = float(outcome.mean()) if outcome.notna().any() else None

    risk_distribution = pd.DataFrame()
    if 'risk_segment' in frame.columns:
        risk_distribution = frame.groupby('risk_segment').size().reset_index(name='patient_count')
        risk_distribution['percentage'] = risk_distribution['patient_count'] / max(len(frame), 1)
        risk_distribution = risk_distribution.sort_values('patient_count', ascending=False).reset_index(drop=True)

    outcome_metric_rows: list[dict[str, object]] = []
    if stage_col and stage_col in frame.columns:
        stage_working = frame[[stage_col]].copy()
        stage_working[stage_col] = stage_working[stage_col].astype(str).str.strip()
        stage_working = stage_working[(stage_working[stage_col] != '') & (stage_working[stage_col].str.lower() != 'nan')]
        if not stage_working.empty:
            stage_counts = stage_working[stage_col].value_counts().head(8)
            for stage_value, count in stage_counts.items():
                row = {'segment_type': 'Cancer Stage', 'segment_value': stage_value, 'cohort_size': int(count)}
                if 'survived_binary' in frame.columns:
                    stage_outcomes = frame.loc[stage_working[stage_working[stage_col] == stage_value].index, 'survived_binary']
                    row['survival_rate'] = float(stage_outcomes.mean()) if stage_outcomes.notna().any() else None
                outcome_metric_rows.append(row)
    if treatment_col and treatment_col in frame.columns:
        treatment_working = frame[[treatment_col]].copy()
        treatment_working[treatment_col] = treatment_working[treatment_col].astype(str).str.strip()
        treatment_working = treatment_working[(treatment_working[treatment_col] != '') & (treatment_working[treatment_col].str.lower() != 'nan')]
        if not treatment_working.empty:
            treatment_counts = treatment_working[treatment_col].value_counts().head(8)
            for treatment_value, count in treatment_counts.items():
                row = {'segment_type': 'Treatment Type', 'segment_value': treatment_value, 'cohort_size': int(count)}
                cohort_slice = frame.loc[treatment_working[treatment_working[treatment_col] == treatment_value].index]
                if 'survived_binary' in cohort_slice.columns:
                    treatment_outcomes = cohort_slice['survived_binary']
                    row['survival_rate'] = float(treatment_outcomes.mean()) if treatment_outcomes.notna().any() else None
                if 'treatment_duration_days' in cohort_slice.columns:
                    durations = pd.to_numeric(cohort_slice['treatment_duration_days'], errors='coerce').dropna()
                    row['average_treatment_duration_days'] = float(durations.mean()) if not durations.empty else None
                outcome_metric_rows.append(row)
    outcome_metrics_table = pd.DataFrame(outcome_metric_rows)

    cohort_trend_table = pd.DataFrame()
    trend_date_col = next((candidate for candidate in [prepared.get('diagnosis_date_col'), prepared.get('service_date_col'), prepared.get('admission_date_col')] if candidate and candidate in frame.columns), None)
    if trend_date_col:
        trend_frame = frame[[trend_date_col]].copy()
        trend_frame[trend_date_col] = pd.to_datetime(trend_frame[trend_date_col], errors='coerce')
        trend_frame = trend_frame.dropna(subset=[trend_date_col])
        if not trend_frame.empty:
            trend_frame['month'] = trend_frame[trend_date_col].dt.to_period('M').dt.to_timestamp()
            cohort_trend_table = trend_frame.groupby('month').size().reset_index(name='record_count')
            if 'survived_binary' in frame.columns:
                survival_working = frame[[trend_date_col, 'survived_binary']].copy()
                survival_working[trend_date_col] = pd.to_datetime(survival_working[trend_date_col], errors='coerce')
                survival_working = survival_working.dropna(subset=[trend_date_col])
                if not survival_working.empty and survival_working['survived_binary'].notna().any():
                    survival_table = survival_working.assign(month=survival_working[trend_date_col].dt.to_period('M').dt.to_timestamp()).groupby('month')['survived_binary'].mean().reset_index(name='survival_rate')
                    cohort_trend_table = cohort_trend_table.merge(survival_table, on='month', how='left')
            if 'risk_segment' in frame.columns:
                risk_working = frame[[trend_date_col, 'risk_segment']].copy()
                risk_working[trend_date_col] = pd.to_datetime(risk_working[trend_date_col], errors='coerce')
                risk_working = risk_working.dropna(subset=[trend_date_col])
                if not risk_working.empty:
                    risk_table = risk_working.assign(month=risk_working[trend_date_col].dt.to_period('M').dt.to_timestamp()).groupby('month')['risk_segment'].apply(lambda s: (s.astype(str) == 'High Risk').mean()).reset_index(name='high_risk_share')
                    cohort_trend_table = cohort_trend_table.merge(risk_table, on='month', how='left')

    preview_columns = [column for column in [age_col, 'age_band', gender_col, diagnosis_col, treatment_col, smoking_col, stage_col, comorbidity_col, 'risk_score', 'risk_segment', 'treatment_duration_days'] if column and column in frame.columns]
    if outcome_col and outcome_col in frame.columns and outcome_col not in preview_columns:
        preview_columns.append(outcome_col)
    preview = frame[preview_columns].head(50) if preview_columns else frame.head(50)

    return {
        'available': True,
        'summary': summary,
        'preview': preview,
        'cohort_frame': frame,
        'risk_distribution_table': risk_distribution,
        'outcome_metrics_table': outcome_metrics_table,
        'cohort_trend_table': cohort_trend_table,
        'cohort_trend_date_column': trend_date_col,
        'filter_columns': {
            'age': age_col,
            'gender': gender_col,
            'diagnosis': diagnosis_col,
            'treatment': treatment_col,
            'smoking': smoking_col,
            'stage': stage_col,
            'comorbidity': comorbidity_col,
        },
    }


def scenario_simulation(data: pd.DataFrame, canonical_map: dict[str, str], smoking_prevalence: float, treatment_type: str | None, treatment_share: float) -> dict[str, object]:
    smoking_col = _resolve_column(data, canonical_map, ['smoking_status'], ['smoking_status', 'smoker'])
    treatment_col = _resolve_column(data, canonical_map, ['treatment_type'], ['treatment_type', 'treatment', 'therapy'])
    outcome_col = _resolve_column(data, canonical_map, ['survived'], ['survived', 'alive', 'outcome'])
    if not outcome_col or (not smoking_col and not treatment_col):
        return {'available': False, 'reason': 'Needs an outcome field plus smoking status and/or treatment type to run a scenario.'}

    outcome = _binary_outcome(data[outcome_col])
    if not outcome.notna().any():
        return {'available': False, 'reason': 'The outcome field does not contain usable survival-style values.'}

    baseline = float(outcome.mean())
    simulated = baseline

    if smoking_col:
        smoking_text = data[smoking_col].astype(str).str.strip().str.lower()
        smoker_mask = smoking_text.str.contains('smoker|current|yes', regex=True, na=False)
        smoker_rate = float(outcome[smoker_mask].mean()) if outcome[smoker_mask].notna().any() else baseline
        non_smoker_rate = float(outcome[~smoker_mask].mean()) if outcome[~smoker_mask].notna().any() else baseline
        current_smoker_rate = float(smoker_mask.mean())
        scenario_smoker_rate = smoking_prevalence / 100.0
        current_blend = (current_smoker_rate * smoker_rate) + ((1 - current_smoker_rate) * non_smoker_rate)
        scenario_blend = (scenario_smoker_rate * smoker_rate) + ((1 - scenario_smoker_rate) * non_smoker_rate)
        simulated += scenario_blend - current_blend

    if treatment_col and treatment_type:
        treatment_series = data[treatment_col].astype(str)
        treatment_mask = treatment_series == treatment_type
        selected_rate = float(outcome[treatment_mask].mean()) if outcome[treatment_mask].notna().any() else baseline
        other_rate = float(outcome[~treatment_mask].mean()) if outcome[~treatment_mask].notna().any() else baseline
        current_share = float(treatment_mask.mean())
        target_share = treatment_share / 100.0
        current_blend = (current_share * selected_rate) + ((1 - current_share) * other_rate)
        scenario_blend = (target_share * selected_rate) + ((1 - target_share) * other_rate)
        simulated += scenario_blend - current_blend

    simulated = min(max(simulated, 0.0), 1.0)
    return {
        'available': True,
        'baseline_survival_rate': baseline,
        'simulated_survival_rate': simulated,
        'improvement': simulated - baseline,
    }



def survival_outcome_analysis(data: pd.DataFrame, canonical_map: dict[str, str]) -> dict[str, object]:
    diagnosis_date_col = _resolve_column(data, canonical_map, ['diagnosis_date', 'admission_date', 'service_date'], ['diagnosis_date', 'admission_date', 'service_date'])
    treatment_end_col = _resolve_column(data, canonical_map, ['end_treatment_date', 'discharge_date'], ['end_treatment_date', 'discharge_date'])
    outcome_col = _resolve_column(data, canonical_map, ['survived'], ['survived', 'alive', 'outcome'])
    stage_col = _resolve_column(data, canonical_map, ['cancer_stage'], ['cancer_stage', 'stage'])
    treatment_col = _resolve_column(data, canonical_map, ['treatment_type'], ['treatment_type', 'treatment', 'therapy'])

    if not outcome_col or not (stage_col or treatment_col or diagnosis_date_col):
        return {'available': False, 'reason': 'Needs a survival-style outcome plus at least one stage, treatment, or diagnosis date field.'}

    frame = data.copy()
    frame['survived_binary'] = _binary_outcome(frame[outcome_col])
    frame = frame[frame['survived_binary'].notna()].copy()
    if frame.empty:
        return {'available': False, 'reason': 'The outcome field does not contain usable survival-style values.'}

    duration_summary = None
    duration_distribution = pd.DataFrame()
    treatment_duration_trend = pd.DataFrame()
    progression_timeline = pd.DataFrame()
    outcome_trend = pd.DataFrame()
    if diagnosis_date_col:
        frame[diagnosis_date_col] = pd.to_datetime(frame[diagnosis_date_col], errors='coerce')
    if treatment_end_col:
        frame[treatment_end_col] = pd.to_datetime(frame[treatment_end_col], errors='coerce')
    if diagnosis_date_col and treatment_end_col:
        frame['treatment_duration_days'] = (frame[treatment_end_col] - frame[diagnosis_date_col]).dt.days
        valid_duration = frame['treatment_duration_days'].dropna()
        valid_duration = valid_duration[valid_duration >= 0]
        if not valid_duration.empty:
            duration_summary = {
                'average_duration_days': float(valid_duration.mean()),
                'median_duration_days': float(valid_duration.median()),
            }
            duration_bins = pd.cut(valid_duration, bins=[-1, 30, 90, 180, 365, float('inf')], labels=['0-30 days', '31-90 days', '91-180 days', '181-365 days', '365+ days'])
            duration_distribution = duration_bins.value_counts(sort=False).reset_index()
            duration_distribution.columns = ['duration_band', 'record_count']
            duration_distribution['percentage'] = duration_distribution['record_count'] / max(int(duration_distribution['record_count'].sum()), 1)
        if diagnosis_date_col and 'treatment_duration_days' in frame.columns:
            duration_trend_frame = frame[[diagnosis_date_col, 'treatment_duration_days']].dropna().copy()
            duration_trend_frame = duration_trend_frame[duration_trend_frame['treatment_duration_days'] >= 0]
            if not duration_trend_frame.empty:
                treatment_duration_trend = duration_trend_frame.assign(month=duration_trend_frame[diagnosis_date_col].dt.to_period('M').dt.to_timestamp()).groupby('month').agg(average_treatment_duration_days=('treatment_duration_days', 'mean'), record_count=('treatment_duration_days', 'size')).reset_index()

    stage_table = pd.DataFrame()
    if stage_col:
        stage_frame = _clean_group_field(frame[[stage_col, 'survived_binary'] + (['treatment_duration_days'] if 'treatment_duration_days' in frame.columns else [])].copy(), stage_col)
        if not stage_frame.empty:
            stage_table = stage_frame.groupby(stage_col).agg(record_count=('survived_binary', 'size'), survival_rate=('survived_binary', 'mean')).reset_index().sort_values('survival_rate')
            if 'treatment_duration_days' in stage_frame.columns:
                duration = stage_frame.groupby(stage_col)['treatment_duration_days'].mean().reset_index(name='average_treatment_duration_days')
                stage_table = stage_table.merge(duration, on=stage_col, how='left')

    treatment_table = pd.DataFrame()
    if treatment_col:
        treatment_frame = _clean_group_field(frame[[treatment_col, 'survived_binary'] + (['treatment_duration_days'] if 'treatment_duration_days' in frame.columns else [])].copy(), treatment_col)
        if not treatment_frame.empty:
            treatment_table = treatment_frame.groupby(treatment_col).agg(record_count=('survived_binary', 'size'), survival_rate=('survived_binary', 'mean')).reset_index().sort_values('survival_rate', ascending=False)
            if 'treatment_duration_days' in treatment_frame.columns:
                duration = treatment_frame.groupby(treatment_col)['treatment_duration_days'].mean().reset_index(name='average_treatment_duration_days')
                treatment_table = treatment_table.merge(duration, on=treatment_col, how='left')

    trend = pd.DataFrame()
    if diagnosis_date_col:
        trend_frame = frame[[diagnosis_date_col, 'survived_binary']].copy()
        trend_frame = trend_frame.dropna(subset=[diagnosis_date_col])
        if not trend_frame.empty:
            trend_frame = trend_frame.assign(month=trend_frame[diagnosis_date_col].dt.to_period('M').dt.to_timestamp())
            trend = trend_frame.groupby('month').agg(survival_rate=('survived_binary', 'mean'), record_count=('survived_binary', 'size')).reset_index()
            outcome_trend = trend_frame.assign(outcome_group=trend_frame['survived_binary'].map({1.0: 'Survived', 0.0: 'Did Not Survive'}).fillna('Outcome Unclear')).groupby(['month', 'outcome_group']).size().reset_index(name='record_count')
        if stage_col:
            progression_frame = frame[[diagnosis_date_col, stage_col]].dropna().copy()
            if not progression_frame.empty:
                progression_frame = _clean_group_field(progression_frame, stage_col)
                top_stages = progression_frame[stage_col].value_counts().head(5).index.tolist()
                progression_frame = progression_frame[progression_frame[stage_col].isin(top_stages)]
                if not progression_frame.empty:
                    progression_timeline = progression_frame.assign(month=progression_frame[diagnosis_date_col].dt.to_period('M').dt.to_timestamp()).groupby(['month', stage_col]).size().reset_index(name='record_count')
                    progression_timeline = progression_timeline.rename(columns={stage_col: 'stage_group'})

    if stage_table.empty and treatment_table.empty and trend.empty and duration_summary is None and progression_timeline.empty and duration_distribution.empty and outcome_trend.empty:
        return {'available': False, 'reason': 'The required outcome fields exist, but the dataset does not have enough usable stage, treatment, or date detail for survival analysis.'}

    return {
        'available': True,
        'stage_column': stage_col,
        'treatment_column': treatment_col,
        'date_column': diagnosis_date_col,
        'duration_summary': duration_summary,
        'duration_distribution': duration_distribution,
        'treatment_duration_trend': treatment_duration_trend,
        'stage_table': stage_table,
        'treatment_table': treatment_table,
        'trend': trend,
        'outcome_trend': outcome_trend,
        'progression_timeline': progression_timeline,
        'note': 'This is a practical demo-oriented survival view based on observed outcomes. It is not a substitute for formal survival modeling or clinical validation.',
    }


def benchmarking_analysis(data: pd.DataFrame, canonical_map: dict[str, str], benchmark_type: str, cohort_frame: pd.DataFrame | None = None, treatment_a: str | None = None, treatment_b: str | None = None) -> dict[str, object]:
    risk_frame = _risk_detail_frame(data, canonical_map)
    base_frame = risk_frame if not risk_frame.empty else data.copy()

    diagnosis_date_col = _resolve_column(base_frame, canonical_map, ['diagnosis_date', 'admission_date', 'service_date'], ['diagnosis_date', 'admission_date', 'service_date'])
    treatment_end_col = _resolve_column(base_frame, canonical_map, ['end_treatment_date', 'discharge_date'], ['end_treatment_date', 'discharge_date'])
    if diagnosis_date_col and treatment_end_col:
        start = pd.to_datetime(base_frame[diagnosis_date_col], errors='coerce')
        end = pd.to_datetime(base_frame[treatment_end_col], errors='coerce')
        base_frame['treatment_duration_days'] = (end - start).dt.days

    treatment_col = _resolve_column(base_frame, canonical_map, ['treatment_type'], ['treatment_type', 'treatment', 'therapy'])
    smoking_col = _resolve_column(base_frame, canonical_map, ['smoking_status'], ['smoking_status', 'smoker'])
    stage_col = _resolve_column(base_frame, canonical_map, ['cancer_stage'], ['cancer_stage', 'stage'])

    if benchmark_type == 'Current Cohort vs Full Dataset':
        if cohort_frame is None or cohort_frame.empty or len(cohort_frame) == len(data):
            return {'available': False, 'reason': 'Create a narrower cohort in Cohort Builder to benchmark it against the full dataset.'}
        cohort_metrics = _build_group_metrics(cohort_frame.copy().assign(benchmark_group='Current Cohort'), 'benchmark_group', canonical_map)
        full_metrics = _build_group_metrics(base_frame.copy().assign(benchmark_group='Full Dataset'), 'benchmark_group', canonical_map)
        summary = pd.concat([cohort_metrics, full_metrics], ignore_index=True)
        group_col = 'benchmark_group'
    elif benchmark_type == 'Treatment Type A vs Treatment Type B':
        if not treatment_col or not treatment_a or not treatment_b:
            return {'available': False, 'reason': 'Select two treatment groups to compare.'}
        comparison = base_frame[base_frame[treatment_col].astype(str).isin([treatment_a, treatment_b])].copy()
        if comparison.empty:
            return {'available': False, 'reason': 'No records matched the selected treatment groups.'}
        summary = _build_group_metrics(comparison, treatment_col, canonical_map)
        group_col = treatment_col
    elif benchmark_type == 'Smoking vs Non-Smoking Cohorts':
        if not smoking_col:
            return {'available': False, 'reason': 'A smoking-status field is needed for this comparison.'}
        comparison = base_frame.copy()
        smoking_text = comparison[smoking_col].astype(str).str.strip().str.lower()
        comparison['smoking_group'] = smoking_text.apply(lambda value: 'Smoking / Former Smoking' if any(token in value for token in ['smoker', 'current', 'former', 'yes']) else 'Other / Non-Smoking')
        summary = _build_group_metrics(comparison, 'smoking_group', canonical_map)
        group_col = 'smoking_group'
    else:
        if not stage_col:
            return {'available': False, 'reason': 'A cancer-stage field is needed for stage benchmarking.'}
        comparison = base_frame.copy()
        summary = _build_group_metrics(comparison, stage_col, canonical_map)
        group_col = stage_col

    if summary.empty:
        return {'available': False, 'reason': 'The selected benchmark groups do not contain enough usable records for comparison.'}

    metric_columns = [column for column in ['survival_rate', 'average_age', 'average_bmi', 'average_treatment_duration_days', 'average_risk_score', 'high_risk_share'] if column in summary.columns]
    return {
        'available': True,
        'group_column': group_col,
        'summary_table': summary,
        'metric_columns': metric_columns,
    }


def _prepare_clinical_frame(data: pd.DataFrame, canonical_map: dict[str, str]) -> dict[str, object]:
    frame = _risk_detail_frame(data, canonical_map)
    if frame.empty:
        frame = data.copy()

    age_col = _resolve_column(frame, canonical_map, ['age'], ['age'])
    smoking_col = _resolve_column(frame, canonical_map, ['smoking_status'], ['smoking_status', 'smoker'])
    stage_col = _resolve_column(frame, canonical_map, ['cancer_stage'], ['cancer_stage', 'stage'])
    treatment_col = _resolve_column(frame, canonical_map, ['treatment_type'], ['treatment_type', 'treatment', 'therapy'])
    gender_col = _resolve_column(frame, canonical_map, ['gender'], ['gender', 'sex'])
    bmi_col = _resolve_column(frame, canonical_map, ['bmi'], ['bmi'])
    cholesterol_col = _resolve_column(frame, canonical_map, ['cholesterol_level'], ['cholesterol_level', 'cholesterol'])
    outcome_col = _resolve_column(frame, canonical_map, ['survived'], ['survived', 'alive', 'outcome'])
    diagnosis_date_col = _resolve_column(frame, canonical_map, ['diagnosis_date', 'admission_date', 'service_date'], ['diagnosis_date', 'admission_date', 'service_date'])
    treatment_end_col = _resolve_column(frame, canonical_map, ['end_treatment_date', 'discharge_date'], ['end_treatment_date', 'discharge_date'])

    if outcome_col and 'survived_binary' not in frame.columns:
        frame['survived_binary'] = _binary_outcome(frame[outcome_col])
    if age_col:
        age_numeric = _safe_numeric(frame[age_col])
        frame['age_band'] = pd.cut(age_numeric, bins=[-1, 39, 59, 74, 200], labels=['18-39', '40-59', '60-74', '75+']).astype(str)
    if diagnosis_date_col and treatment_end_col:
        start = pd.to_datetime(frame[diagnosis_date_col], errors='coerce')
        end = pd.to_datetime(frame[treatment_end_col], errors='coerce')
        frame['treatment_duration_days'] = (end - start).dt.days

    return {
        'frame': frame,
        'age_col': age_col,
        'smoking_col': smoking_col,
        'stage_col': stage_col,
        'treatment_col': treatment_col,
        'gender_col': gender_col,
        'bmi_col': bmi_col,
        'cholesterol_col': cholesterol_col,
        'outcome_col': outcome_col,
        'diagnosis_date_col': diagnosis_date_col,
        'treatment_end_col': treatment_end_col,
    }


def intervention_planner(
    data: pd.DataFrame,
    canonical_map: dict[str, str],
    smoking_reduction_points: float,
    treatment_type: str | None,
    treatment_share_increase_points: float,
    high_risk_follow_up_share: float,
    early_stage_increase_points: float,
) -> dict[str, object]:
    prepared = _prepare_clinical_frame(data, canonical_map)
    frame = prepared['frame']
    if 'survived_binary' not in frame.columns or not frame['survived_binary'].notna().any():
        return {'available': False, 'reason': 'An outcome field such as survived is needed to simulate intervention impact.'}

    baseline_survival = float(frame['survived_binary'].mean())
    high_risk_share = float((frame.get('risk_segment', pd.Series(index=frame.index, dtype='object')) == 'High Risk').mean()) if 'risk_segment' in frame.columns else None
    baseline_duration = float(frame['treatment_duration_days'].dropna().mean()) if 'treatment_duration_days' in frame.columns and frame['treatment_duration_days'].notna().any() else None
    simulated_survival = baseline_survival
    simulated_high_risk_share = high_risk_share
    simulated_duration = baseline_duration
    assumptions: list[str] = []

    smoking_col = prepared['smoking_col']
    if smoking_col and smoking_reduction_points > 0:
        smoking_text = frame[smoking_col].astype(str).str.strip().str.lower()
        smoker_mask = smoking_text.str.contains('smoker|current|yes', regex=True, na=False)
        if smoker_mask.any() and (~smoker_mask).any():
            smoker_rate = float(frame.loc[smoker_mask, 'survived_binary'].mean())
            non_smoker_rate = float(frame.loc[~smoker_mask, 'survived_binary'].mean())
            current_share = float(smoker_mask.mean())
            target_share = max(current_share - (smoking_reduction_points / 100.0), 0.0)
            simulated_survival += (target_share * smoker_rate + (1 - target_share) * non_smoker_rate) - (current_share * smoker_rate + (1 - current_share) * non_smoker_rate)
            assumptions.append('Smoking reduction shifts the population mix toward the observed non-smoker survival pattern.')
            if simulated_high_risk_share is not None:
                simulated_high_risk_share = max(simulated_high_risk_share - (smoking_reduction_points / 100.0) * 0.35, 0.0)

    treatment_col = prepared['treatment_col']
    if treatment_col and treatment_type and treatment_share_increase_points > 0:
        treatment_mask = frame[treatment_col].astype(str) == treatment_type
        if treatment_mask.any() and (~treatment_mask).any():
            selected_rate = float(frame.loc[treatment_mask, 'survived_binary'].mean())
            other_rate = float(frame.loc[~treatment_mask, 'survived_binary'].mean())
            current_share = float(treatment_mask.mean())
            target_share = min(current_share + (treatment_share_increase_points / 100.0), 1.0)
            simulated_survival += (target_share * selected_rate + (1 - target_share) * other_rate) - (current_share * selected_rate + (1 - current_share) * other_rate)
            assumptions.append('Treatment expansion assumes the selected therapy performs in the future as it does in the current dataset.')
            if simulated_duration is not None and 'treatment_duration_days' in frame.columns:
                selected_duration = float(frame.loc[treatment_mask, 'treatment_duration_days'].dropna().mean()) if frame.loc[treatment_mask, 'treatment_duration_days'].notna().any() else simulated_duration
                other_duration = float(frame.loc[~treatment_mask, 'treatment_duration_days'].dropna().mean()) if frame.loc[~treatment_mask, 'treatment_duration_days'].notna().any() else simulated_duration
                simulated_duration += (target_share * selected_duration + (1 - target_share) * other_duration) - (current_share * selected_duration + (1 - current_share) * other_duration)

    if 'risk_segment' in frame.columns and high_risk_follow_up_share > 0:
        followup_fraction = high_risk_follow_up_share / 100.0
        high_risk_mask = frame['risk_segment'] == 'High Risk'
        if high_risk_mask.any():
            high_risk_rate = float(frame.loc[high_risk_mask, 'survived_binary'].mean()) if frame.loc[high_risk_mask, 'survived_binary'].notna().any() else baseline_survival
            other_rate = float(frame.loc[~high_risk_mask, 'survived_binary'].mean()) if frame.loc[~high_risk_mask, 'survived_binary'].notna().any() else baseline_survival
            recoverable_gap = max(other_rate - high_risk_rate, 0.0)
            simulated_survival += recoverable_gap * followup_fraction * float(high_risk_mask.mean())
            simulated_high_risk_share = max((simulated_high_risk_share or float(high_risk_mask.mean())) - (float(high_risk_mask.mean()) * followup_fraction * 0.15), 0.0)
            assumptions.append('Focused follow-up assumes a modest closing of the current survival gap for high-risk patients.')
            if simulated_duration is not None:
                simulated_duration = max(simulated_duration * (1 - (0.05 * followup_fraction)), 0.0)

    stage_col = prepared['stage_col']
    if stage_col and early_stage_increase_points > 0:
        stage_text = frame[stage_col].astype(str).str.strip().str.lower()
        early_mask = stage_text.str.contains('i|ii|stage i|stage ii', regex=True, na=False) & ~stage_text.str.contains('iii|iv|stage iii|stage iv', regex=True, na=False)
        late_mask = stage_text.str.contains('iii|iv|stage iii|stage iv|3|4', regex=True, na=False)
        if early_mask.any() and late_mask.any():
            early_rate = float(frame.loc[early_mask, 'survived_binary'].mean())
            late_rate = float(frame.loc[late_mask, 'survived_binary'].mean())
            current_early_share = float(early_mask.mean())
            target_early_share = min(current_early_share + (early_stage_increase_points / 100.0), 1.0)
            simulated_survival += (target_early_share * early_rate + (1 - target_early_share) * late_rate) - (current_early_share * early_rate + (1 - current_early_share) * late_rate)
            assumptions.append('Earlier-stage diagnosis assumes more patients benefit from the observed early-stage survival pattern.')
            if simulated_high_risk_share is not None:
                simulated_high_risk_share = max(simulated_high_risk_share - (early_stage_increase_points / 100.0) * 0.4, 0.0)
            if simulated_duration is not None and 'treatment_duration_days' in frame.columns:
                early_duration = float(frame.loc[early_mask, 'treatment_duration_days'].dropna().mean()) if frame.loc[early_mask, 'treatment_duration_days'].notna().any() else simulated_duration
                late_duration = float(frame.loc[late_mask, 'treatment_duration_days'].dropna().mean()) if frame.loc[late_mask, 'treatment_duration_days'].notna().any() else simulated_duration
                simulated_duration += (target_early_share * early_duration + (1 - target_early_share) * late_duration) - (current_early_share * early_duration + (1 - current_early_share) * late_duration)

    simulated_survival = min(max(simulated_survival, 0.0), 1.0)
    result_table = pd.DataFrame([
        {'metric': 'Survival Rate', 'before': baseline_survival, 'after': simulated_survival, 'delta': simulated_survival - baseline_survival},
        {'metric': 'High-Risk Share', 'before': high_risk_share, 'after': simulated_high_risk_share, 'delta': (simulated_high_risk_share - high_risk_share) if high_risk_share is not None and simulated_high_risk_share is not None else None},
        {'metric': 'Average Treatment Duration (Days)', 'before': baseline_duration, 'after': simulated_duration, 'delta': (simulated_duration - baseline_duration) if baseline_duration is not None and simulated_duration is not None else None},
    ])
    return {
        'available': True,
        'summary_table': result_table,
        'assumptions': assumptions or ['No intervention lever had enough supporting data to change the baseline metrics.'],
        'baseline_survival_rate': baseline_survival,
        'simulated_survival_rate': simulated_survival,
        'baseline_high_risk_share': high_risk_share,
        'simulated_high_risk_share': simulated_high_risk_share,
        'baseline_duration_days': baseline_duration,
        'simulated_duration_days': simulated_duration,
    }


def driver_analysis(data: pd.DataFrame, canonical_map: dict[str, str]) -> dict[str, object]:
    prepared = _prepare_clinical_frame(data, canonical_map)
    frame = prepared['frame']
    rows: list[dict[str, object]] = []
    min_required = 4 if len(frame) < 12 else 8
    factors = [
        ('age', prepared['age_col'], 'numeric'),
        ('smoking_status', prepared['smoking_col'], 'categorical'),
        ('cancer_stage', prepared['stage_col'], 'categorical'),
        ('treatment_type', prepared['treatment_col'], 'categorical'),
        ('bmi', prepared['bmi_col'], 'numeric'),
        ('cholesterol_level', prepared['cholesterol_col'], 'numeric'),
        ('comorbidities', _resolve_column(frame, canonical_map, ['comorbidities'], ['comorbidities', 'comorbidity_score']), 'numeric_or_flag'),
    ]
    targets = [('survived_binary', 'Survival'), ('risk_score', 'Risk Score'), ('treatment_duration_days', 'Treatment Duration')]

    for factor_name, column, factor_type in factors:
        if not column or column not in frame.columns:
            continue
        for target_col, target_label in targets:
            if target_col not in frame.columns:
                continue
            target_series = pd.to_numeric(frame[target_col], errors='coerce')
            if target_series.notna().sum() < min_required:
                continue
            if factor_type == 'categorical':
                grouped = _clean_group_field(frame[[column, target_col]].copy(), column)
                grouped[target_col] = pd.to_numeric(grouped[target_col], errors='coerce')
                summary = grouped.groupby(column)[target_col].mean().dropna()
                if len(summary) < 2:
                    continue
                top = summary.max()
                bottom = summary.min()
                score = abs(float(top - bottom))
                reason = f'Largest observed gap is between {summary.idxmax()} and {summary.idxmin()}.'
            else:
                numeric_factor = pd.to_numeric(frame[column], errors='coerce')
                valid = pd.DataFrame({'factor': numeric_factor, 'target': target_series}).dropna()
                if len(valid) < min_required or valid['factor'].nunique() < 2:
                    continue
                threshold = valid['factor'].median()
                high_group = valid.loc[valid['factor'] >= threshold, 'target']
                low_group = valid.loc[valid['factor'] < threshold, 'target']
                if high_group.empty or low_group.empty:
                    continue
                score = abs(float(high_group.mean() - low_group.mean()))
                reason = f'Upper-half versus lower-half comparison around {threshold:.1f} shows the strongest difference.'
            rows.append({'factor': factor_name, 'target_metric': target_label, 'influence_score': score, 'interpretation': reason})

    if not rows:
        return {'available': False, 'reason': 'Not enough aligned clinical fields are available to estimate drivers.'}

    ranked = pd.DataFrame(rows).sort_values('influence_score', ascending=False).reset_index(drop=True)
    interpretation = f"{ranked.iloc[0]['factor'].replace('_', ' ').title()} appears to be the strongest overall driver in the current dataset, especially for {ranked.iloc[0]['target_metric'].lower()}."
    return {'available': True, 'ranked_table': ranked, 'interpretation': interpretation}


def operational_alerts(data: pd.DataFrame, canonical_map: dict[str, str], healthcare: dict[str, object], survival_threshold: float, high_risk_threshold: float, anomaly_threshold: int, duration_threshold: float, cohort_gap_threshold: float) -> dict[str, object]:
    alerts: list[dict[str, object]] = []
    risk = healthcare.get('risk_segmentation', {})
    anomaly = healthcare.get('anomaly_detection', {})
    survival = healthcare.get('survival_outcomes', {})

    if risk.get('available') and risk.get('survival_rate') is not None and risk['survival_rate'] < survival_threshold:
        alerts.append({'severity': 'High', 'alert_title': 'Overall survival below target', 'rationale': f"Observed survival is {risk['survival_rate']:.1%}, below the threshold of {survival_threshold:.1%}.", 'recommended_next_action': 'Review treatment pathways and high-risk cohorts to understand where the outcome gap is concentrated.'})
    if risk.get('available') and not risk['segment_table'].empty:
        high_row = risk['segment_table'][risk['segment_table']['risk_segment'] == 'High Risk']
        if not high_row.empty and float(high_row.iloc[0]['percentage']) > high_risk_threshold:
            alerts.append({'severity': 'High', 'alert_title': 'High-risk cohort share is elevated', 'rationale': f"High Risk patients represent {float(high_row.iloc[0]['percentage']):.1%}, above the threshold of {high_risk_threshold:.1%}.", 'recommended_next_action': 'Focus outreach and follow-up planning on the highest-risk segments first.'})
    if anomaly.get('available') and not anomaly['summary_table'].empty:
        top_anomaly = anomaly['summary_table'].iloc[0]
        if int(top_anomaly['anomaly_count']) >= anomaly_threshold:
            alerts.append({'severity': 'Medium', 'alert_title': 'Clinical measurement anomalies require validation', 'rationale': f"{top_anomaly['field']} has {int(top_anomaly['anomaly_count'])} flagged records.", 'recommended_next_action': 'Validate outliers before using the affected field for operational or clinical comparison.'})
    if survival.get('available') and survival.get('duration_summary') and survival['duration_summary']['average_duration_days'] > duration_threshold:
        alerts.append({'severity': 'Medium', 'alert_title': 'Treatment duration is longer than expected', 'rationale': f"Average treatment duration is {survival['duration_summary']['average_duration_days']:.1f} days, above the threshold of {duration_threshold:.1f} days.", 'recommended_next_action': 'Review stages or treatment pathways with the longest duration to identify operational bottlenecks.'})
    if survival.get('available') and not survival.get('stage_table', pd.DataFrame()).empty and risk.get('survival_rate') is not None:
        weakest = survival['stage_table'].iloc[0]
        if (risk['survival_rate'] - float(weakest['survival_rate'])) > cohort_gap_threshold:
            alerts.append({'severity': 'Medium', 'alert_title': 'A poor-outcome cohort is underperforming the population average', 'rationale': f"{weakest[survival['stage_column']]} is {risk['survival_rate'] - float(weakest['survival_rate']):.1%} below overall survival.", 'recommended_next_action': 'Compare treatment use and follow-up intensity for this cohort against better-performing groups.'})

    if not alerts:
        return {'available': False, 'reason': 'No alerts crossed the selected thresholds.'}
    severity_order = {'High': 0, 'Medium': 1, 'Low': 2}
    table = pd.DataFrame(alerts).sort_values('severity', key=lambda series: series.map(severity_order)).reset_index(drop=True)
    return {'available': True, 'alerts_table': table}


def segment_discovery(data: pd.DataFrame, canonical_map: dict[str, str]) -> dict[str, object]:
    prepared = _prepare_clinical_frame(data, canonical_map)
    frame = prepared['frame']
    dimensions = []
    for column in ['age_band', prepared['gender_col'], prepared['smoking_col'], prepared['stage_col'], prepared['treatment_col']]:
        if column and column not in dimensions and column in frame.columns:
            dimensions.append(column)
    if len(dimensions) < 1:
        return {'available': False, 'reason': 'Not enough patient or treatment dimensions are available for segment discovery.'}

    if 'survived_binary' not in frame.columns and 'risk_score' not in frame.columns and 'treatment_duration_days' not in frame.columns:
        return {'available': False, 'reason': 'Outcome, risk, or duration metrics are needed to discover standout segments.'}

    overall_survival = float(frame['survived_binary'].mean()) if 'survived_binary' in frame.columns and frame['survived_binary'].notna().any() else None
    overall_risk = float(frame['risk_score'].mean()) if 'risk_score' in frame.columns else None
    overall_duration = float(frame['treatment_duration_days'].mean()) if 'treatment_duration_days' in frame.columns and frame['treatment_duration_days'].notna().any() else None
    discovered: list[dict[str, object]] = []

    combinations = [(dim,) for dim in dimensions]
    if 'age_band' in frame.columns:
        combinations.extend([('age_band', dim) for dim in dimensions if dim != 'age_band'])

    for combo in combinations:
        combo_cols = list(combo)
        working = frame.copy()
        for column in combo_cols:
            working = _clean_group_field(working, column)
        grouped = working.groupby(combo_cols).size().reset_index(name='record_count')
        grouped = grouped[grouped['record_count'] >= max(8, int(len(frame) * 0.03))]
        if grouped.empty:
            continue
        if 'survived_binary' in working.columns:
            grouped = grouped.merge(working.groupby(combo_cols)['survived_binary'].mean().reset_index(name='survival_rate'), on=combo_cols, how='left')
        if 'risk_score' in working.columns:
            grouped = grouped.merge(working.groupby(combo_cols)['risk_score'].mean().reset_index(name='average_risk_score'), on=combo_cols, how='left')
        if 'treatment_duration_days' in working.columns:
            grouped = grouped.merge(working.groupby(combo_cols)['treatment_duration_days'].mean().reset_index(name='average_treatment_duration_days'), on=combo_cols, how='left')
        if prepared['bmi_col'] and prepared['bmi_col'] in working.columns:
            grouped = grouped.merge(working.groupby(combo_cols)[prepared['bmi_col']].apply(lambda s: pd.to_numeric(s, errors='coerce').mean()).reset_index(name='average_bmi'), on=combo_cols, how='left')

        for _, row in grouped.iterrows():
            standout = 0.0
            reasons = []
            dominant_signal = None
            dominant_gap = 0.0
            if overall_survival is not None and pd.notna(row.get('survival_rate')):
                gap = overall_survival - float(row['survival_rate'])
                if gap > 0:
                    weighted_gap = gap
                    standout += weighted_gap
                    reasons.append(f"survival is {gap:.1%} below overall")
                    if weighted_gap > dominant_gap:
                        dominant_signal = 'Poor Survival'
                        dominant_gap = weighted_gap
            if overall_risk is not None and pd.notna(row.get('average_risk_score')):
                gap = float(row['average_risk_score']) - overall_risk
                if gap > 0:
                    weighted_gap = gap * 0.08
                    standout += weighted_gap
                    reasons.append(f"risk score is {gap:.2f} above average")
                    if weighted_gap > dominant_gap:
                        dominant_signal = 'High Risk'
                        dominant_gap = weighted_gap
            if overall_duration is not None and pd.notna(row.get('average_treatment_duration_days')):
                gap = float(row['average_treatment_duration_days']) - overall_duration
                if gap > 0:
                    weighted_gap = gap / max(overall_duration, 1.0) * 0.1
                    standout += weighted_gap
                    reasons.append(f"treatment duration is {gap:.1f} days longer than average")
                    if weighted_gap > dominant_gap:
                        dominant_signal = 'Long Treatment Duration'
                        dominant_gap = weighted_gap
            if standout <= 0:
                continue
            segment_label = ' | '.join(f"{column}={row[column]}" for column in combo_cols)
            dominant_signal = dominant_signal or 'Mixed Signal'
            discovered.append({
                'discovered_segment': segment_label,
                'why_it_stands_out': '; '.join(reasons),
                'segment_dimensions': ', '.join(combo_cols),
                'dimension_count': len(combo_cols),
                'dominant_signal': dominant_signal,
                'benchmark_gap': dominant_gap,
                'priority_band': _segment_priority_band(standout),
                'record_count': int(row['record_count']),
                'survival_rate': row.get('survival_rate'),
                'average_risk_score': row.get('average_risk_score'),
                'average_treatment_duration_days': row.get('average_treatment_duration_days'),
                'suggested_follow_up_analysis': _segment_follow_up(dominant_signal, combo_cols),
                'review_question': f"Why does {segment_label} show {dominant_signal.lower()} compared with the overall population?",
                'standout_score': standout,
            })

    if not discovered:
        return {'available': False, 'reason': 'No segment stood out strongly enough using the current discovery rules.'}
    table = pd.DataFrame(discovered).sort_values(['standout_score', 'record_count'], ascending=[False, False]).head(12).reset_index(drop=True)
    top_segment = table.iloc[0]
    metric_leaders = table.groupby('dominant_signal').agg(
        segment_count=('discovered_segment', 'size'),
        average_standout_score=('standout_score', 'mean'),
        highest_priority=('priority_band', 'first'),
    ).reset_index().sort_values(['segment_count', 'average_standout_score'], ascending=[False, False])
    summary = (
        f"Segment Discovery identified {len(table)} notable segments. "
        f"The strongest current signal is {top_segment['dominant_signal'].lower()} for {top_segment['discovered_segment']} "
        f"with a standout score of {top_segment['standout_score']:.2f}."
    )
    return {
        'available': True,
        'segment_table': table,
        'metric_leaders': metric_leaders,
        'summary': summary,
        'top_segment': top_segment.to_dict(),
    }


def root_cause_explorer(data: pd.DataFrame, canonical_map: dict[str, str], target_metric: str) -> dict[str, object]:
    prepared = _prepare_clinical_frame(data, canonical_map)
    frame = prepared['frame']
    dimensions = [('Cancer Stage', prepared['stage_col']), ('Smoking Status', prepared['smoking_col']), ('Treatment Type', prepared['treatment_col']), ('Gender', prepared['gender_col']), ('Age Band', 'age_band')]
    dimension_tables: list[pd.DataFrame] = []

    for dimension_label, column in dimensions:
        if not column or column not in frame.columns:
            continue
        grouped = _build_group_metrics(frame, column, canonical_map)
        if grouped.empty:
            continue
        grouped = grouped.copy()
        grouped['dimension'] = dimension_label
        grouped['segment'] = grouped[column].astype(str)
        dimension_tables.append(grouped)

    if not dimension_tables:
        return {'available': False, 'reason': 'No drill-down dimensions are available for root cause exploration.'}

    combined = pd.concat(dimension_tables, ignore_index=True)
    if target_metric == 'Low Survival' and 'survival_rate' in combined.columns:
        ranked = combined.sort_values(['survival_rate', 'record_count'])
        explanation = 'These segments have the lowest observed survival and are the best starting points for follow-up review.'
        chart_metric = 'survival_rate'
    elif target_metric == 'High Risk Share' and 'high_risk_share' in combined.columns:
        ranked = combined.sort_values(['high_risk_share', 'record_count'], ascending=[False, False])
        explanation = 'These segments contain the highest concentration of high-risk patients.'
        chart_metric = 'high_risk_share'
    elif 'average_treatment_duration_days' in combined.columns:
        ranked = combined.sort_values(['average_treatment_duration_days', 'record_count'], ascending=[False, False])
        explanation = 'These segments are associated with the longest treatment duration and may indicate process friction.'
        chart_metric = 'average_treatment_duration_days'
    else:
        return {'available': False, 'reason': 'The selected target metric is not supported by the current dataset.'}

    ranked = ranked.head(12).reset_index(drop=True)
    summary = f"Root Cause Explorer is currently prioritizing {ranked.iloc[0]['dimension']} = {ranked.iloc[0]['segment']} as the strongest contributor for {target_metric.lower()}."
    return {'available': True, 'ranked_table': ranked[['dimension', 'segment', 'record_count'] + [column for column in ['survival_rate', 'high_risk_share', 'average_treatment_duration_days', 'average_risk_score'] if column in ranked.columns]], 'explanation': explanation, 'summary': summary, 'chart_metric': chart_metric}


def care_pathway_view(data: pd.DataFrame, canonical_map: dict[str, str]) -> dict[str, object]:
    prepared = _prepare_clinical_frame(data, canonical_map)
    frame = prepared['frame']
    stage_col = prepared['stage_col']
    treatment_col = prepared['treatment_col']
    outcome_col = prepared['outcome_col']
    if ('treatment_duration_days' not in frame.columns or frame['treatment_duration_days'].dropna().empty) and not outcome_col:
        return {'available': False, 'reason': 'Diagnosis dates, treatment dates, and outcome fields are needed for pathway analysis.'}

    by_stage = pd.DataFrame()
    if stage_col and 'treatment_duration_days' in frame.columns:
        stage_frame = _clean_group_field(frame[[stage_col, 'treatment_duration_days'] + (['survived_binary'] if 'survived_binary' in frame.columns else [])].copy(), stage_col)
        if not stage_frame.empty:
            by_stage = stage_frame.groupby(stage_col).agg(record_count=('treatment_duration_days', 'size'), average_treatment_duration_days=('treatment_duration_days', 'mean')).reset_index()
            if 'survived_binary' in stage_frame.columns:
                by_stage = by_stage.merge(stage_frame.groupby(stage_col)['survived_binary'].mean().reset_index(name='survival_rate'), on=stage_col, how='left')

    by_treatment = pd.DataFrame()
    if treatment_col and 'treatment_duration_days' in frame.columns:
        treatment_frame = _clean_group_field(frame[[treatment_col, 'treatment_duration_days'] + (['survived_binary'] if 'survived_binary' in frame.columns else [])].copy(), treatment_col)
        if not treatment_frame.empty:
            by_treatment = treatment_frame.groupby(treatment_col).agg(record_count=('treatment_duration_days', 'size'), average_treatment_duration_days=('treatment_duration_days', 'mean')).reset_index()
            if 'survived_binary' in treatment_frame.columns:
                by_treatment = by_treatment.merge(treatment_frame.groupby(treatment_col)['survived_binary'].mean().reset_index(name='survival_rate'), on=treatment_col, how='left')

    pathway = pd.DataFrame()
    if stage_col and treatment_col:
        pathway_frame = _clean_group_field(frame[[stage_col, treatment_col] + (['treatment_duration_days'] if 'treatment_duration_days' in frame.columns else []) + (['survived_binary'] if 'survived_binary' in frame.columns else [])].copy(), stage_col)
        pathway_frame = _clean_group_field(pathway_frame, treatment_col)
        if not pathway_frame.empty:
            agg_dict = {'record_count': (treatment_col, 'size')}
            if 'treatment_duration_days' in pathway_frame.columns:
                agg_dict['average_treatment_duration_days'] = ('treatment_duration_days', 'mean')
            if 'survived_binary' in pathway_frame.columns:
                agg_dict['survival_rate'] = ('survived_binary', 'mean')
            pathway = pathway_frame.groupby([stage_col, treatment_col]).agg(**agg_dict).reset_index().sort_values('record_count', ascending=False)

    if by_stage.empty and by_treatment.empty and pathway.empty:
        return {'available': False, 'reason': 'The dataset does not contain enough pathway fields for a usable care pathway view.'}

    bottleneck_summary = pd.DataFrame()
    poor_outcome_pathways = pd.DataFrame()
    summary_text = 'Care pathway intelligence is using observed treatment duration and outcome patterns to highlight the most review-worthy pathways.'

    if not pathway.empty:
        pathway_working = pathway.copy()
        if 'average_treatment_duration_days' in pathway_working.columns:
            pathway_working['duration_gap_vs_pathway_average'] = (
                pathway_working['average_treatment_duration_days'] - pathway_working['average_treatment_duration_days'].mean()
            )
        else:
            pathway_working['duration_gap_vs_pathway_average'] = pd.NA

        if 'survival_rate' in pathway_working.columns and pathway_working['survival_rate'].notna().any():
            overall_survival = float(pathway_working['survival_rate'].mean())
            pathway_working['survival_gap_vs_pathway_average'] = pathway_working['survival_rate'] - overall_survival
        else:
            overall_survival = None
            pathway_working['survival_gap_vs_pathway_average'] = pd.NA

        bottleneck_rows: list[dict[str, object]] = []
        for _, row in pathway_working.iterrows():
            bottleneck_score = 0.0
            reasons: list[str] = []
            if pd.notna(row.get('duration_gap_vs_pathway_average')) and float(row['duration_gap_vs_pathway_average']) > 0:
                bottleneck_score += float(row['duration_gap_vs_pathway_average']) / max(float(pathway_working['average_treatment_duration_days'].mean() or 1.0), 1.0)
                reasons.append(f"duration is {float(row['duration_gap_vs_pathway_average']):.1f} days above the pathway average")
            if pd.notna(row.get('survival_gap_vs_pathway_average')) and float(row['survival_gap_vs_pathway_average']) < 0:
                bottleneck_score += abs(float(row['survival_gap_vs_pathway_average'])) * 2.0
                reasons.append(f"survival is {abs(float(row['survival_gap_vs_pathway_average'])):.1%} below the pathway average")
            if bottleneck_score <= 0:
                continue
            bottleneck_rows.append({
                'pathway': f"{row[stage_col]} -> {row[treatment_col]}",
                'record_count': int(row.get('record_count', 0)),
                'average_treatment_duration_days': row.get('average_treatment_duration_days'),
                'survival_rate': row.get('survival_rate'),
                'bottleneck_score': bottleneck_score,
                'why_it_stands_out': '; '.join(reasons),
                'suggested_next_action': 'Review this pathway against the Care Pathway View and Root Cause Explorer to identify delays, treatment variation, or subgroup-specific outcome gaps.',
            })

        if bottleneck_rows:
            bottleneck_summary = pd.DataFrame(bottleneck_rows).sort_values(['bottleneck_score', 'record_count'], ascending=[False, False]).head(10).reset_index(drop=True)

        if 'survival_rate' in pathway_working.columns and pathway_working['survival_rate'].notna().any():
            poor_outcome_pathways = pathway_working.sort_values(['survival_rate', 'record_count'], ascending=[True, False]).head(10).reset_index(drop=True)

        if not bottleneck_summary.empty:
            top_pathway = bottleneck_summary.iloc[0]
            summary_text = (
                f"The highest-priority pathway review is currently {top_pathway['pathway']}, "
                f"which combines longer duration or weaker outcomes than peer pathways."
            )
        elif not poor_outcome_pathways.empty:
            summary_text = (
                f"The pathway with the weakest observed outcome is {poor_outcome_pathways.iloc[0][stage_col]} -> "
                f"{poor_outcome_pathways.iloc[0][treatment_col]}."
            )

    return {
        'available': True,
        'stage_table': by_stage,
        'treatment_table': by_treatment,
        'pathway_table': pathway,
        'stage_column': stage_col,
        'treatment_column': treatment_col,
        'bottleneck_summary': bottleneck_summary,
        'poor_outcome_pathways': poor_outcome_pathways,
        'summary': summary_text,
    }


def explainability_and_fairness(data: pd.DataFrame, canonical_map: dict[str, str]) -> dict[str, object]:
    prepared = _prepare_clinical_frame(data, canonical_map)
    frame = prepared['frame']
    if 'risk_score' not in frame.columns and 'survived_binary' not in frame.columns:
        return {'available': False, 'reason': 'Risk or outcome fields are needed for explainability and subgroup review.'}

    high_risk_reasons: list[dict[str, object]] = []
    if 'risk_segment' in frame.columns:
        high_risk = frame[frame['risk_segment'] == 'High Risk']
        if not high_risk.empty:
            if prepared['age_col']:
                high_risk_reasons.append({'factor': 'Age > 60', 'share_in_high_risk_group': float((_safe_numeric(high_risk[prepared['age_col']]) > 60).mean())})
            if prepared['smoking_col']:
                smoking_text = high_risk[prepared['smoking_col']].astype(str).str.strip().str.lower()
                high_risk_reasons.append({'factor': 'Smoking status', 'share_in_high_risk_group': float(smoking_text.str.contains('smoker|current|yes', regex=True, na=False).mean())})
            if prepared['stage_col']:
                high_risk_reasons.append({'factor': 'Stage III/IV', 'share_in_high_risk_group': float(_stage_high_risk(high_risk[prepared['stage_col']]).mean())})
            comorbidity_col = _resolve_column(frame, canonical_map, ['comorbidities'], ['comorbidities', 'comorbidity_score'])
            if comorbidity_col:
                high_risk_reasons.append({'factor': 'Comorbidities present', 'share_in_high_risk_group': float(_comorbidity_present(high_risk[comorbidity_col]).mean())})
    high_risk_reason_table = pd.DataFrame(high_risk_reasons).sort_values('share_in_high_risk_group', ascending=False) if high_risk_reasons else pd.DataFrame()

    segment_explanation = ''
    if 'risk_segment' in frame.columns:
        explain_dims = [('Age Band', 'age_band'), ('Gender', prepared['gender_col']), ('Smoking Status', prepared['smoking_col']), ('Cancer Stage', prepared['stage_col']), ('Treatment Type', prepared['treatment_col'])]
        segment_candidates: list[dict[str, object]] = []
        for dimension_label, column in explain_dims:
            if not column or column not in frame.columns:
                continue
            grouped = _build_group_metrics(frame, column, canonical_map)
            if grouped.empty or 'high_risk_share' not in grouped.columns:
                continue
            strongest = grouped.sort_values(['high_risk_share', 'record_count'], ascending=[False, False]).iloc[0]
            if pd.isna(strongest.get('high_risk_share')):
                continue
            segment_candidates.append({
                'dimension': dimension_label,
                'segment': str(strongest[column]),
                'high_risk_share': float(strongest['high_risk_share']),
                'record_count': int(strongest.get('record_count', 0)),
            })
        if segment_candidates:
            top_segment = pd.DataFrame(segment_candidates).sort_values(['high_risk_share', 'record_count'], ascending=[False, False]).iloc[0]
            segment_explanation = (
                f"{top_segment['dimension']} = {top_segment['segment']} currently shows the highest concentration of high-risk patients at {top_segment['high_risk_share']:.1%}. This is a practical starting point for deeper review and targeted follow-up planning."
            )

    subgroup_tables: list[pd.DataFrame] = []
    fairness_flags: list[dict[str, object]] = []
    subgroup_dims = [('Gender', prepared['gender_col']), ('Age Band', 'age_band')]
    for label, column in subgroup_dims:
        if not column or column not in frame.columns:
            continue
        grouped = _build_group_metrics(frame, column, canonical_map)
        if grouped.empty:
            continue
        grouped['group_dimension'] = label
        subgroup_tables.append(grouped)
        if 'survival_rate' in grouped.columns and len(grouped) >= 2:
            gap = float(grouped['survival_rate'].max() - grouped['survival_rate'].min())
            if gap >= 0.10:
                fairness_flags.append({
                    'severity': 'High' if gap >= 0.20 else 'Moderate',
                    'group_dimension': label,
                    'flag_type': 'Survival gap',
                    'gap_size': gap,
                    'detail': f"Observed survival differs by {gap:.1%} across {label.lower()} groups.",
                })
        if 'high_risk_share' in grouped.columns and len(grouped) >= 2:
            gap = float(grouped['high_risk_share'].max() - grouped['high_risk_share'].min())
            if gap >= 0.10:
                fairness_flags.append({
                    'severity': 'High' if gap >= 0.20 else 'Moderate',
                    'group_dimension': label,
                    'flag_type': 'Risk concentration gap',
                    'gap_size': gap,
                    'detail': f"High-risk share differs by {gap:.1%} across {label.lower()} groups.",
                })

    subgroup_comparison = pd.concat(subgroup_tables, ignore_index=True) if subgroup_tables else pd.DataFrame()
    if high_risk_reason_table.empty and subgroup_comparison.empty:
        return {'available': False, 'reason': 'Not enough subgroup detail is available for explainability and fairness review.'}

    narrative = 'The explainability view highlights which factors are most common in the high-risk cohort, where subgroup outcome gaps deserve review, and which segment is the clearest candidate for focused follow-up.'
    return {
        'available': True,
        'high_risk_reason_table': high_risk_reason_table,
        'subgroup_comparison': subgroup_comparison,
        'fairness_flags': pd.DataFrame(fairness_flags),
        'high_risk_segment_explanation': segment_explanation,
        'narrative': narrative,
    }
def run_healthcare_analysis(data: pd.DataFrame, canonical_map: dict[str, str], synthetic_fields: set[str] | None = None) -> dict[str, object]:
    readiness = assess_healthcare_dataset(canonical_map, synthetic_fields=synthetic_fields)
    risk = patient_risk_segmentation(data, canonical_map)
    ai_summary = ai_insight_summary(data, canonical_map, risk)
    anomaly = anomaly_detection(data, canonical_map)
    default_cohort = build_cohort_summary(data, canonical_map)
    survival = survival_outcome_analysis(data, canonical_map)
    driver = driver_analysis(data, canonical_map)
    segments = segment_discovery(data, canonical_map)
    pathway = care_pathway_view(data, canonical_map)
    explainability = explainability_and_fairness(data, canonical_map)
    default_treatment_col = _resolve_column(data, canonical_map, ['treatment_type'], ['treatment_type', 'treatment', 'therapy'])
    default_treatment = None
    if default_treatment_col:
        values = data[default_treatment_col].dropna().astype(str)
        if not values.empty:
            default_treatment = values.value_counts().index[0]
    scenario = scenario_simulation(data, canonical_map, smoking_prevalence=25.0, treatment_type=default_treatment, treatment_share=50.0)
    return {
        **readiness,
        'utilization': utilization_analysis(data, canonical_map),
        'cost': cost_analysis(data, canonical_map),
        'provider': provider_analysis(data, canonical_map),
        'diagnosis': diagnosis_procedure_analysis(data, canonical_map),
        'readmission': readmission_risk_analytics(data, canonical_map),
        'risk_segmentation': risk,
        'ai_insight_summary': ai_summary,
        'anomaly_detection': anomaly,
        'default_cohort_summary': default_cohort,
        'scenario': scenario,
        'survival_outcomes': survival,
        'driver_analysis': driver,
        'segment_discovery': segments,
        'care_pathway': pathway,
        'explainability_fairness': explainability,
    }








