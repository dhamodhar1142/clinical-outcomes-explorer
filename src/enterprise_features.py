from __future__ import annotations

from difflib import SequenceMatcher

import pandas as pd

from src.schema_detection import StructureSummary


RULE_LIBRARY = {
    'age': {'min': 0, 'max': 120, 'label': 'Age should be between 0 and 120.', 'severity': 'HIGH', 'category': 'Numeric range'},
    'bmi': {'min': 10, 'max': 80, 'label': 'BMI should be between 10 and 80.', 'severity': 'HIGH', 'category': 'Numeric range'},
    'cholesterol_level': {'min': 50, 'max': 400, 'label': 'Cholesterol should be between 50 and 400.', 'severity': 'MEDIUM', 'category': 'Numeric range'},
    'cost_amount': {'min': 0, 'label': 'Cost should not be negative.', 'severity': 'HIGH', 'category': 'Numeric range'},
    'paid_amount': {'min': 0, 'label': 'Paid amount should not be negative.', 'severity': 'HIGH', 'category': 'Numeric range'},
    'allowed_amount': {'min': 0, 'label': 'Allowed amount should not be negative.', 'severity': 'HIGH', 'category': 'Numeric range'},
    'billed_amount': {'min': 0, 'label': 'Billed amount should not be negative.', 'severity': 'HIGH', 'category': 'Numeric range'},
}


def build_quality_rule_catalog() -> pd.DataFrame:
    rows = []
    for field, rule in RULE_LIBRARY.items():
        rows.append({
            'rule_name': rule['label'],
            'field': field,
            'severity': rule.get('severity', 'MEDIUM'),
            'category': rule.get('category', 'Field validation'),
        })
    rows.extend([
        {
            'rule_name': 'Survived should contain boolean-style outcome values.',
            'field': 'survived',
            'severity': 'HIGH',
            'category': 'Field validation',
        },
        {
            'rule_name': 'End treatment date should not be earlier than diagnosis date.',
            'field': 'end_treatment_date',
            'severity': 'HIGH',
            'category': 'Temporal consistency',
        },
        {
            'rule_name': 'Discharge date should not be earlier than admission date.',
            'field': 'discharge_date',
            'severity': 'HIGH',
            'category': 'Temporal consistency',
        },
    ])
    return pd.DataFrame(rows)


def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors='coerce')


def _best_overlap_score(left: pd.Series, right: pd.Series) -> float:
    left_values = set(left.dropna().astype(str).head(250).str.lower())
    right_values = set(right.dropna().astype(str).head(250).str.lower())
    if not left_values or not right_values:
        return 0.0
    overlap = left_values.intersection(right_values)
    return len(overlap) / max(min(len(left_values), len(right_values)), 1)



def infer_linked_dataset_role(semantic: dict[str, object]) -> str:
    canonical_map = semantic.get('canonical_map', {}) if isinstance(semantic, dict) else {}
    fields = set(canonical_map.keys())
    if {'patient_id', 'member_id'} & fields:
        if {'treatment_type', 'diagnosis_date', 'end_treatment_date', 'survived'} & fields:
            return 'Treatment or outcomes dataset'
        if {'cost_amount', 'paid_amount', 'allowed_amount', 'billed_amount'} & fields:
            return 'Patient-level cost dataset'
        return 'Patient-level dataset'
    if {'provider_id', 'provider_name', 'facility', 'department'} & fields:
        if {'cost_amount', 'paid_amount', 'allowed_amount', 'billed_amount'} & fields:
            return 'Facility or provider cost dataset'
        return 'Facility or provider dataset'
    if {'cost_amount', 'paid_amount', 'allowed_amount', 'billed_amount'} & fields:
        return 'Financial or cost dataset'
    return 'General related dataset'


def build_join_recommendation(join_candidates: pd.DataFrame, left_role: str, right_role: str) -> str:
    if join_candidates.empty:
        return (
            f"No strong join recommendation was detected automatically between the current {left_role.lower()} and "
            f"the selected {right_role.lower()}. You can still choose keys manually if you know the relationship."
        )
    top = join_candidates.iloc[0]
    return (
        f"Best automatic match: join the current {left_role.lower()} to the selected {right_role.lower()} using "
        f"{top['left_key']} -> {top['right_key']} (confidence {top['confidence_score']:.2f}) because {top['reason']}."
    )
def detect_join_candidates(
    left_data: pd.DataFrame,
    left_structure: StructureSummary,
    right_data: pd.DataFrame,
    right_structure: StructureSummary,
) -> pd.DataFrame:
    candidate_rows: list[dict[str, object]] = []
    left_candidates = list(dict.fromkeys(left_structure.identifier_columns + left_structure.categorical_columns[:6]))
    right_candidates = list(dict.fromkeys(right_structure.identifier_columns + right_structure.categorical_columns[:6]))

    for left_col in left_candidates:
        for right_col in right_candidates:
            name_score = SequenceMatcher(None, left_col, right_col).ratio()
            overlap_score = _best_overlap_score(left_data[left_col], right_data[right_col])
            left_unique = float(left_data[left_col].nunique(dropna=True) / max(len(left_data), 1))
            right_unique = float(right_data[right_col].nunique(dropna=True) / max(len(right_data), 1))
            uniqueness_alignment = 1.0 - abs(left_unique - right_unique)
            score = (name_score * 0.45) + (overlap_score * 0.4) + (uniqueness_alignment * 0.15)
            if score < 0.28:
                continue
            reasons = []
            if name_score >= 0.75:
                reasons.append('column names are closely aligned')
            if overlap_score >= 0.15:
                reasons.append('sample values overlap across files')
            if left_col in left_structure.identifier_columns or right_col in right_structure.identifier_columns:
                reasons.append('one or both fields look identifier-like')
            candidate_rows.append(
                {
                    'left_key': left_col,
                    'right_key': right_col,
                    'confidence_score': round(score, 3),
                    'reason': '; '.join(reasons) or 'moderate structural similarity',
                    'left_uniqueness': round(left_unique, 3),
                    'right_uniqueness': round(right_unique, 3),
                }
            )

    if not candidate_rows:
        return pd.DataFrame(columns=['left_key', 'right_key', 'confidence_score', 'reason', 'left_uniqueness', 'right_uniqueness'])
    return pd.DataFrame(candidate_rows).sort_values('confidence_score', ascending=False).drop_duplicates(['left_key', 'right_key']).reset_index(drop=True)


def preview_linked_merge(
    left_data: pd.DataFrame,
    right_data: pd.DataFrame,
    left_key: str,
    right_key: str,
    how: str = 'left',
) -> dict[str, object]:
    if left_key not in left_data.columns or right_key not in right_data.columns:
        return {'available': False, 'reason': 'Choose a valid key from both datasets before previewing the merge.'}

    right_subset = right_data.copy()
    duplicate_cols = [column for column in right_subset.columns if column in left_data.columns and column != right_key]
    rename_map = {column: f"{column}_linked" for column in duplicate_cols}
    right_subset = right_subset.rename(columns=rename_map)

    left_frame = left_data.copy()
    right_frame = right_subset.copy()
    left_frame[left_key] = left_frame[left_key].astype(str)
    right_frame[right_key] = right_frame[right_key].astype(str)

    merged = left_frame.merge(right_frame, left_on=left_key, right_on=right_key, how=how, indicator=True)
    preview = merged.head(50)
    matched_share = float((merged['_merge'] != 'left_only').mean()) if how == 'left' else float((merged['_merge'] != 'right_only').mean())
    unmatched_rows = int((merged['_merge'] == 'left_only').sum()) if how == 'left' else int((merged['_merge'] == 'right_only').sum())
    duplicate_key_share = float(right_frame[right_key].duplicated().mean()) if len(right_frame) else 0.0
    return {
        'available': True,
        'preview': preview,
        'merged_data': merged.drop(columns=['_merge']),
        'summary': {
            'rows_before': len(left_data),
            'rows_after': len(merged),
            'matched_share': matched_share,
            'unmatched_rows': unmatched_rows,
            'duplicate_key_share': duplicate_key_share,
            'new_columns_added': max(len(merged.columns) - len(left_data.columns), 0),
        },
    }


def build_quality_rule_engine(data: pd.DataFrame, canonical_map: dict[str, str]) -> dict[str, object]:
    summary_rows: list[dict[str, object]] = []
    detail_frames: list[pd.DataFrame] = []
    checked_fields: list[str] = []
    skipped_fields: list[str] = []
    affected_columns: set[str] = set()
    boolean_like_values = {'1', '0', 'true', 'false', 'yes', 'no', 'y', 'n', 'alive', 'dead', 'survived', 'deceased'}

    for canonical_field, rule in RULE_LIBRARY.items():
        column = canonical_map.get(canonical_field)
        if not column or column not in data.columns:
            skipped_fields.append(canonical_field)
            continue
        checked_fields.append(canonical_field)
        numeric = _safe_numeric(data[column])
        valid = numeric.dropna()
        if valid.empty:
            continue
        violations = pd.Series(False, index=valid.index)
        if 'min' in rule:
            violations = violations | (valid < rule['min'])
        if 'max' in rule:
            violations = violations | (valid > rule['max'])
        if not violations.any():
            continue
        failed = valid[violations]
        summary_rows.append(
            {
                'rule_name': rule['label'],
                'field': canonical_field,
                'source_column': column,
                'category': rule.get('category', 'Field validation'),
                'severity': rule.get('severity', 'MEDIUM'),
                'rule': rule['label'],
                'rows_violating': int(len(failed)),
                'failure_count': int(len(failed)),
                'failure_rate': float(len(failed) / max(len(valid), 1)),
            }
        )
        affected_columns.add(column)
        detail_frames.append(
            pd.DataFrame(
                {
                    'field': canonical_field,
                    'source_column': column,
                    'severity': rule.get('severity', 'MEDIUM'),
                    'row_index': failed.index.astype(int),
                    'invalid_value': failed.values,
                    'rule': rule['label'],
                }
            ).head(50)
        )

    survived_col = canonical_map.get('survived')
    if survived_col and survived_col in data.columns:
        if 'survived' not in checked_fields:
            checked_fields.append('survived')
        survived_text = data[survived_col].dropna().astype(str).str.strip().str.lower()
        invalid_survival = survived_text[~survived_text.isin(boolean_like_values)]
        if not invalid_survival.empty:
            summary_rows.append(
                {
                    'rule_name': 'Survived should contain boolean-style outcome values.',
                    'field': 'survived',
                    'source_column': survived_col,
                    'category': 'Field validation',
                    'severity': 'HIGH',
                    'rule': 'Survived should contain boolean-style outcome values.',
                    'rows_violating': int(len(invalid_survival)),
                    'failure_count': int(len(invalid_survival)),
                    'failure_rate': float(len(invalid_survival) / max(len(survived_text), 1)),
                }
            )
            affected_columns.add(survived_col)
            detail_frames.append(
                pd.DataFrame(
                    {
                        'rule_name': 'Survived should contain boolean-style outcome values.',
                        'field': 'survived',
                        'source_column': survived_col,
                        'severity': 'HIGH',
                        'row_index': invalid_survival.index.astype(int),
                        'invalid_value': invalid_survival.values,
                        'rule': 'Survived should contain boolean-style outcome values.',
                    }
                ).head(50)
            )

    diagnosis_date = canonical_map.get('diagnosis_date')
    treatment_end = canonical_map.get('end_treatment_date')
    if diagnosis_date and treatment_end and diagnosis_date in data.columns and treatment_end in data.columns:
        checked_fields.extend([field for field in ['diagnosis_date', 'end_treatment_date'] if field not in checked_fields])
        frame = data[[diagnosis_date, treatment_end]].copy()
        frame[diagnosis_date] = pd.to_datetime(frame[diagnosis_date], errors='coerce')
        frame[treatment_end] = pd.to_datetime(frame[treatment_end], errors='coerce')
        frame = frame.dropna()
        if not frame.empty:
            violations = frame[frame[treatment_end] < frame[diagnosis_date]]
            if not violations.empty:
                summary_rows.append(
                    {
                        'rule_name': 'End treatment date should not be earlier than diagnosis date.',
                        'field': 'end_treatment_date',
                        'source_column': treatment_end,
                        'category': 'Temporal consistency',
                    'severity': 'HIGH',
                        'rule': 'End treatment date should not be earlier than diagnosis date.',
                        'rows_violating': int(len(violations)),
                        'failure_count': int(len(violations)),
                        'failure_rate': float(len(violations) / max(len(frame), 1)),
                    }
                )
                affected_columns.add(treatment_end)
                detail_frames.append(
                    violations.reset_index().rename(columns={'index': 'row_index', diagnosis_date: 'diagnosis_value', treatment_end: 'end_treatment_value'}).assign(rule_name='End treatment date should not be earlier than diagnosis date.', field='end_treatment_date', severity='High', rule='End treatment date should not be earlier than diagnosis date.')
                )

    admission = canonical_map.get('admission_date')
    discharge = canonical_map.get('discharge_date')
    if admission and discharge and admission in data.columns and discharge in data.columns:
        checked_fields.extend([field for field in ['admission_date', 'discharge_date'] if field not in checked_fields])
        frame = data[[admission, discharge]].copy()
        frame[admission] = pd.to_datetime(frame[admission], errors='coerce')
        frame[discharge] = pd.to_datetime(frame[discharge], errors='coerce')
        frame = frame.dropna()
        if not frame.empty:
            violations = frame[frame[discharge] < frame[admission]]
            if not violations.empty:
                summary_rows.append(
                    {
                        'rule_name': 'Discharge date should not be earlier than admission date.',
                        'field': 'discharge_date',
                        'source_column': discharge,
                        'category': 'Temporal consistency',
                    'severity': 'HIGH',
                        'rule': 'Discharge date should not be earlier than admission date.',
                        'rows_violating': int(len(violations)),
                        'failure_count': int(len(violations)),
                        'failure_rate': float(len(violations) / max(len(frame), 1)),
                    }
                )
                affected_columns.add(discharge)
                detail_frames.append(
                    violations.reset_index().rename(columns={'index': 'row_index', admission: 'admission_value', discharge: 'discharge_value'}).assign(rule_name='Discharge date should not be earlier than admission date.', field='discharge_date', severity='High', rule='Discharge date should not be earlier than admission date.')
                )

    duplicate_rows = data.duplicated()
    if duplicate_rows.any():
        failed = data[duplicate_rows].head(50).copy()
        summary_rows.append(
            {
                'rule_name': 'Duplicate rows detected in the current dataset.',
                'field': 'row_level',
                'source_column': 'multiple columns',
                'category': 'Duplicate detection',
                'severity': 'MEDIUM',
                'rule': 'Duplicate rows detected in the current dataset.',
                'rows_violating': int(duplicate_rows.sum()),
                'failure_count': int(duplicate_rows.sum()),
                'failure_rate': float(duplicate_rows.mean()),
            }
        )
        detail_frames.append(failed.reset_index().rename(columns={'index': 'row_index'}).assign(field='row_level', source_column='multiple columns', severity='MEDIUM', rule='Duplicate rows detected in the current dataset.'))

    required_fields = ['patient_id', 'event_date', 'diagnosis_code', 'cost_amount', 'readmission']
    for canonical_field in required_fields:
        column = canonical_map.get(canonical_field)
        if not column or column not in data.columns:
            skipped_fields.append(f'{canonical_field}:missing')
            continue
        checked_fields.append(canonical_field) if canonical_field not in checked_fields else None
        null_mask = data[column].isna()
        if not null_mask.any():
            continue
        severity = 'CRITICAL' if null_mask.mean() >= 0.95 else 'HIGH'
        summary_rows.append(
            {
                'rule_name': f'{canonical_field} should not be entirely or mostly missing when used by advanced modules.',
                'field': canonical_field,
                'source_column': column,
                'category': 'Null expectation',
                'severity': severity,
                'rule': f'{canonical_field} should not be entirely or mostly missing when used by advanced modules.',
                'rows_violating': int(null_mask.sum()),
                'failure_count': int(null_mask.sum()),
                'failure_rate': float(null_mask.mean()),
            }
        )
        affected_columns.add(column)
        detail_frames.append(
            data.loc[null_mask, [column]].head(50).reset_index().rename(columns={'index': 'row_index', column: 'invalid_value'}).assign(
                field=canonical_field,
                source_column=column,
                severity=severity,
                rule=f'{canonical_field} should not be entirely or mostly missing when used by advanced modules.',
            )
        )

    categorical_rules = {
        'smoking_status': {'allowed': {'current', 'former', 'never', 'no info', 'unknown', 'yes', 'no'}, 'severity': 'LOW'},
        'gender': {'allowed': {'male', 'female', 'm', 'f', 'unknown', 'other'}, 'severity': 'LOW'},
        'readmission': {'allowed': {'0', '1', 'true', 'false', 'yes', 'no'}, 'severity': 'MEDIUM'},
    }
    for canonical_field, rule in categorical_rules.items():
        column = canonical_map.get(canonical_field)
        if not column or column not in data.columns:
            continue
        checked_fields.append(canonical_field) if canonical_field not in checked_fields else None
        values = data[column].dropna().astype(str).str.strip().str.lower()
        if values.empty:
            continue
        invalid = values[~values.isin(rule['allowed'])]
        if invalid.empty:
            continue
        summary_rows.append(
            {
                'rule_name': f'{canonical_field} contains values outside the expected review set.',
                'field': canonical_field,
                'source_column': column,
                'category': 'Categorical allowed values',
                'severity': rule['severity'],
                'rule': f'{canonical_field} contains values outside the expected review set.',
                'rows_violating': int(len(invalid)),
                'failure_count': int(len(invalid)),
                'failure_rate': float(len(invalid) / max(len(values), 1)),
            }
        )
        affected_columns.add(column)
        detail_frames.append(
            invalid.head(50).reset_index().rename(columns={'index': 'row_index', column: 'invalid_value'}).assign(
                field=canonical_field,
                source_column=column,
                severity=rule['severity'],
                rule=f'{canonical_field} contains values outside the expected review set.',
            )
        )

    for canonical_field, column in canonical_map.items():
        if column not in data.columns:
            continue
        lowered = canonical_field.lower()
        if 'id' not in lowered:
            continue
        values = data[column].dropna().astype(str)
        if values.empty:
            continue
        uniqueness_ratio = values.nunique() / max(len(values), 1)
        duplicate_ratio = 1.0 - uniqueness_ratio
        if uniqueness_ratio < 0.5 and len(values) > 20:
            summary_rows.append(
                {
                    'rule_name': 'Identifier-like field shows suspiciously low cardinality.',
                    'field': canonical_field,
                    'source_column': column,
                    'category': 'Identifier cardinality',
                    'severity': 'MEDIUM',
                    'rule': 'Identifier-like field shows suspiciously low cardinality.',
                    'rows_violating': int(values.duplicated().sum()),
                    'failure_count': int(values.duplicated().sum()),
                    'failure_rate': float(duplicate_ratio),
                }
            )
            affected_columns.add(column)

    overview = {
        'rules_checked': len(checked_fields),
        'fields_with_failures': len({row['field'] for row in summary_rows}),
        'failed_rules': len(summary_rows),
        'skipped_fields': skipped_fields,
        'total_rules_checked': len(checked_fields),
        'total_failed_rules': len(summary_rows),
        'failure_rate': float(len(summary_rows) / max(len(checked_fields), 1)) if checked_fields else 0.0,
        'affected_columns': sorted(affected_columns),
        'rows_violating_by_rule': {row['rule_name']: int(row['rows_violating']) for row in summary_rows},
        'skipped_rules': skipped_fields,
    }

    if not checked_fields:
        return {
            'available': False,
            'passed': False,
            'reason': 'No mapped fields currently align to the built-in quality rules.',
            'summary_table': pd.DataFrame(),
            'detail_table': pd.DataFrame(),
            'rule_catalog': build_quality_rule_catalog(),
            'overview': overview,
            'severity_summary': pd.DataFrame(),
            'rule_details': pd.DataFrame(),
        }

    if not summary_rows:
        return {
            'available': True,
            'passed': True,
            'reason': 'No rule-based quality violations were detected for the currently mapped fields.',
            'summary_table': pd.DataFrame(columns=['rule_name', 'field', 'source_column', 'category', 'severity', 'rule', 'rows_violating', 'failure_count', 'failure_rate']),
            'detail_table': pd.DataFrame(),
            'rule_catalog': build_quality_rule_catalog(),
            'overview': overview,
            'severity_summary': pd.DataFrame(columns=['severity', 'failed_rules', 'rows_violating']),
            'rule_details': pd.DataFrame(),
        }

    severity_order = {'CRITICAL': 0, 'HIGH': 1, 'MEDIUM': 2, 'LOW': 3, 'High': 1, 'Medium': 2, 'Low': 3}
    summary = pd.DataFrame(summary_rows).sort_values(['severity', 'failure_count', 'failure_rate'], ascending=[True, False, False], key=lambda series: series.map(severity_order) if series.name == 'severity' else series).reset_index(drop=True)
    details = pd.concat(detail_frames, ignore_index=True) if detail_frames else pd.DataFrame()
    severity_summary = summary.groupby('severity').agg(failed_rules=('rule_name', 'count'), rows_violating=('rows_violating', 'sum')).reset_index()
    return {
        'available': True,
        'passed': False,
        'summary_table': summary,
        'detail_table': details.head(100),
        'rule_catalog': build_quality_rule_catalog(),
        'overview': overview,
        'severity_summary': severity_summary,
        'rule_details': summary,
    }
def cohort_monitoring_over_time(
    data: pd.DataFrame,
    canonical_map: dict[str, str],
    cohort_frame: pd.DataFrame | None = None,
) -> dict[str, object]:
    date_col = next((canonical_map.get(field) for field in ['diagnosis_date', 'service_date', 'admission_date', 'event_date'] if canonical_map.get(field) in data.columns), None)
    if not date_col:
        return {'available': False, 'reason': 'A diagnosis, service, admission, or event date is required for cohort monitoring over time.'}

    frame = data.copy()
    frame[date_col] = pd.to_datetime(frame[date_col], errors='coerce')
    frame = frame.dropna(subset=[date_col])
    if frame.empty:
        return {'available': False, 'reason': 'No usable date records remain after parsing the cohort-monitoring date field.'}

    risk_col_present = 'risk_segment' in frame.columns
    outcome_col = canonical_map.get('survived')
    stage_col = canonical_map.get('cancer_stage') if canonical_map.get('cancer_stage') in frame.columns else None
    treatment_col = canonical_map.get('treatment_type') if canonical_map.get('treatment_type') in frame.columns else None
    if outcome_col and outcome_col in frame.columns:
        mapping = {'1': 1, '0': 0, 'true': 1, 'false': 0, 'yes': 1, 'no': 0, 'alive': 1, 'survived': 1, 'deceased': 0, 'dead': 0}
        text = frame[outcome_col].astype(str).str.strip().str.lower()
        frame['survived_binary'] = text.map(mapping).fillna(pd.to_numeric(frame[outcome_col], errors='coerce'))

    cohort_base = cohort_frame.copy() if cohort_frame is not None and not cohort_frame.empty else frame.copy()
    if date_col not in cohort_base.columns:
        cohort_base = cohort_base.merge(frame[[date_col]], left_index=True, right_index=True, how='left')
    cohort_base[date_col] = pd.to_datetime(cohort_base[date_col], errors='coerce')
    cohort_base = cohort_base.dropna(subset=[date_col])
    if cohort_base.empty:
        return {'available': False, 'reason': 'The current cohort filters do not leave enough date-aligned rows for monitoring.'}

    def _trend_table(source: pd.DataFrame, label: str) -> pd.DataFrame:
        working = source.copy()
        working['month'] = working[date_col].dt.to_period('M').dt.to_timestamp()
        grouped = working.groupby('month').size().reset_index(name='record_count')
        grouped['cohort'] = label
        if 'survived_binary' in working.columns and working['survived_binary'].notna().any():
            grouped = grouped.merge(working.groupby('month')['survived_binary'].mean().reset_index(name='survival_rate'), on='month', how='left')
        if risk_col_present and 'risk_segment' in working.columns:
            grouped = grouped.merge(working.groupby('month')['risk_segment'].apply(lambda s: (s.astype(str) == 'High Risk').mean()).reset_index(name='high_risk_share'), on='month', how='left')
        return grouped

    def _mix_table(source: pd.DataFrame, label: str, column: str, dimension_label: str) -> pd.DataFrame:
        working = source[[date_col, column]].copy()
        working[column] = working[column].astype(str).str.strip()
        working = working[(working[column] != '') & (working[column].str.lower() != 'nan')]
        if working.empty:
            return pd.DataFrame()
        top_groups = working[column].value_counts().head(4).index.tolist()
        working = working[working[column].isin(top_groups)]
        if working.empty:
            return pd.DataFrame()
        working['month'] = working[date_col].dt.to_period('M').dt.to_timestamp()
        counts = working.groupby(['month', column]).size().reset_index(name='record_count')
        counts['share'] = counts.groupby('month')['record_count'].transform(lambda s: s / max(s.sum(), 1))
        counts['cohort'] = label
        counts['dimension'] = dimension_label
        counts = counts.rename(columns={column: 'group_value'})
        return counts

    overall_trend = _trend_table(frame, 'Full Dataset')
    cohort_trend = _trend_table(cohort_base, 'Current Cohort')
    trend = pd.concat([overall_trend, cohort_trend], ignore_index=True)

    stage_mix_frames = []
    if stage_col:
        stage_mix_frames.append(_mix_table(frame, 'Full Dataset', stage_col, 'Cancer Stage'))
        stage_mix_frames.append(_mix_table(cohort_base, 'Current Cohort', stage_col, 'Cancer Stage'))
    stage_mix_table = pd.concat([item for item in stage_mix_frames if not item.empty], ignore_index=True) if stage_mix_frames else pd.DataFrame()

    treatment_mix_frames = []
    if treatment_col:
        treatment_mix_frames.append(_mix_table(frame, 'Full Dataset', treatment_col, 'Treatment Type'))
        treatment_mix_frames.append(_mix_table(cohort_base, 'Current Cohort', treatment_col, 'Treatment Type'))
    treatment_mix_table = pd.concat([item for item in treatment_mix_frames if not item.empty], ignore_index=True) if treatment_mix_frames else pd.DataFrame()

    latest_month = trend['month'].max()
    latest = trend[trend['month'] == latest_month]
    available_metrics = ['record_count'] + [metric for metric in ['survival_rate', 'high_risk_share'] if metric in trend.columns and trend[metric].notna().any()]
    summary_lines = [f"Cohort monitoring is using {date_col} to track the current cohort against the full dataset over time."]
    if not stage_mix_table.empty:
        summary_lines.append('Cancer stage mix can be monitored over time for both the current cohort and the full dataset.')
    if not treatment_mix_table.empty:
        summary_lines.append('Treatment mix can be monitored over time to show how pathway composition shifts month by month.')
    latest_summary: dict[str, object] = {}
    if not latest.empty:
        overall_latest = latest[latest['cohort'] == 'Full Dataset']
        cohort_latest = latest[latest['cohort'] == 'Current Cohort']
        if not overall_latest.empty and not cohort_latest.empty:
            overall_row = overall_latest.iloc[0]
            cohort_row = cohort_latest.iloc[0]
            latest_summary = {
                'month': latest_month,
                'cohort_record_count': int(cohort_row['record_count']),
                'overall_record_count': int(overall_row['record_count']),
            }
            summary_lines.append(
                f"In {latest_month:%b %Y}, the current cohort recorded {int(cohort_row['record_count'])} rows versus {int(overall_row['record_count'])} across the full dataset."
            )
            if 'survival_rate' in available_metrics and pd.notna(cohort_row.get('survival_rate')) and pd.notna(overall_row.get('survival_rate')):
                latest_summary['cohort_survival_rate'] = float(cohort_row['survival_rate'])
                latest_summary['overall_survival_rate'] = float(overall_row['survival_rate'])
                summary_lines.append(
                    f"Observed survival for the current cohort is {float(cohort_row['survival_rate']):.1%} compared with {float(overall_row['survival_rate']):.1%} across the full dataset."
                )
            if 'high_risk_share' in available_metrics and pd.notna(cohort_row.get('high_risk_share')) and pd.notna(overall_row.get('high_risk_share')):
                latest_summary['cohort_high_risk_share'] = float(cohort_row['high_risk_share'])
                latest_summary['overall_high_risk_share'] = float(overall_row['high_risk_share'])
                summary_lines.append(
                    f"The current cohort's high-risk share is {float(cohort_row['high_risk_share']):.1%} versus {float(overall_row['high_risk_share']):.1%} for the full dataset."
                )

    return {
        'available': True,
        'trend_table': trend,
        'summary_lines': summary_lines,
        'date_column': date_col,
        'available_metrics': available_metrics,
        'latest_summary': latest_summary,
        'stage_mix_table': stage_mix_table,
        'treatment_mix_table': treatment_mix_table,
    }
def build_workflow_pack_summary(widget_state: dict[str, object], template: str) -> str:
    populated = [key for key, value in widget_state.items() if value not in (None, '', [], (), {})]
    return f"{template} with {len(populated)} saved controls"

def build_workflow_pack_details(widget_state: dict[str, object], template: str, dataset_context: dict[str, object] | None = None, copilot_prompts: list[str] | None = None) -> dict[str, object]:
    populated = {key: value for key, value in widget_state.items() if value not in (None, '', [], (), {})}
    highlighted_controls = []
    labels = {
        'analysis_template': 'analysis template',
        'benchmark_type': 'benchmark comparison',
        'report_mode': 'report mode',
        'root_cause_target': 'root-cause target',
        'cohort_monitor_metric': 'monitoring metric',
        'scenario_treatment': 'scenario treatment',
        'planner_treatment_choice': 'intervention treatment focus',
        'workflow_action_prompt': 'copilot workflow prompt',
    }
    for key in ['benchmark_type', 'report_mode', 'root_cause_target', 'cohort_monitor_metric', 'scenario_treatment', 'planner_treatment_choice', 'workflow_action_prompt']:
        value = populated.get(key)
        if value not in (None, '', [], (), {}):
            highlighted_controls.append(f"{labels.get(key, key)} = {value}")
    if dataset_context:
        source_mode = dataset_context.get('dataset_source_mode')
        dataset_name = dataset_context.get('demo_dataset_name')
        if source_mode == 'Demo dataset' and dataset_name:
            highlighted_controls.insert(0, f"dataset source = {dataset_name} (demo dataset)")
        elif source_mode:
            highlighted_controls.insert(0, f"dataset source = {source_mode}")
    if copilot_prompts:
        for prompt in copilot_prompts[:2]:
            highlighted_controls.append(f"saved copilot prompt = {prompt}")
    if not highlighted_controls:
        highlighted_controls.append('general reusable analysis controls only')
    summary = f"{template} with {len(populated)} saved controls"
    return {
        'summary': summary,
        'control_count': len(populated),
        'highlighted_controls': highlighted_controls,
    }



def _format_lineage_value(value: object) -> str:
    if value is None:
        return 'Not set'
    if isinstance(value, (list, tuple, set)):
        values = [str(item) for item in value if str(item).strip()]
        return ', '.join(values) if values else 'Not set'
    if isinstance(value, dict):
        return ', '.join(f'{key}={item}' for key, item in value.items()) if value else 'Not set'
    text = str(value).strip()
    return text if text else 'Not set'


def build_data_lineage_view(
    dataset_name: str,
    source_meta: dict[str, str],
    semantic: dict[str, object],
    readiness: dict[str, object],
    widget_state: dict[str, object],
    accuracy_summary: dict[str, object] | None = None,
) -> dict[str, pd.DataFrame | list[str] | str]:
    source_rows = pd.DataFrame(
        [
            {'lineage_stage': 'Source dataset', 'detail': dataset_name},
            {'lineage_stage': 'Source mode', 'detail': source_meta.get('source_mode', 'Unknown')},
            {'lineage_stage': 'Dataset description', 'detail': source_meta.get('description', 'Not provided')},
            {'lineage_stage': 'Best-fit use case', 'detail': source_meta.get('best_for', 'General profiling and conditional analysis')},
        ]
    )

    canonical_map = semantic.get('canonical_map', {})
    derived_rows = pd.DataFrame(
        [
            {
                'source_column': source_column,
                'derived_role': canonical_field,
                'business_meaning': canonical_field.replace('_', ' ').title(),
            }
            for canonical_field, source_column in canonical_map.items()
        ]
    )

    transformations = [
        'Column names are standardized internally for safer downstream analysis.',
        'Structure detection classifies numeric, date, categorical, identifier, boolean, and text fields.',
        'Semantic mapping translates source fields into canonical healthcare or business roles when confidence is sufficient.',
        'Analysis readiness determines which modules can run safely without forcing unsupported workflows.',
    ]
    if readiness.get('available_count', 0) > 0:
        transformations.append(f"{readiness['available_count']} analysis modules are currently ready to use.")

    active_control_rows = []
    for key, value in widget_state.items():
        formatted = _format_lineage_value(value)
        if formatted == 'Not set':
            continue
        active_control_rows.append(
            {
                'analysis_control': key.replace('_', ' ').title(),
                'current_value': formatted,
            }
        )
    active_controls = pd.DataFrame(active_control_rows)
    if not active_controls.empty:
        active_controls = active_controls.sort_values('analysis_control').reset_index(drop=True)

    metric_lineage_table = pd.DataFrame()
    approval_table = pd.DataFrame()
    if isinstance(accuracy_summary, dict):
        metric_lineage_table = accuracy_summary.get('metric_lineage_table')
        metric_lineage_table = metric_lineage_table if isinstance(metric_lineage_table, pd.DataFrame) else pd.DataFrame()
        approval_workflow = accuracy_summary.get('approval_workflow', {})
        if isinstance(approval_workflow, dict) and approval_workflow:
            approval_table = pd.DataFrame(
                [
                    {
                        'review_stage': 'Mapping approval',
                        'status': approval_workflow.get('mapping_status', 'Pending'),
                        'detail': approval_workflow.get('review_notes', '') or 'Awaiting reviewer confirmation.',
                    },
                    {
                        'review_stage': 'Trust gate',
                        'status': approval_workflow.get('trust_gate_status', 'Pending'),
                        'detail': f"Threshold profile: {approval_workflow.get('reporting_threshold_profile', 'Standard')}",
                    },
                    {
                        'review_stage': 'Export eligibility',
                        'status': approval_workflow.get('export_eligibility_status', 'Pending'),
                        'detail': 'Controls whether governed external handoff is approved for the current dataset context.',
                    },
                    {
                        'review_stage': 'Release signoff',
                        'status': approval_workflow.get('release_signoff_status', 'Pending'),
                        'detail': 'Tracks whether the current dataset review is approved for outward release or remains internal-only.',
                    },
                ]
            )

    return {
        'source_table': source_rows,
        'derived_fields_table': derived_rows,
        'transformation_steps': transformations,
        'active_controls_table': active_controls,
        'metric_lineage_table': metric_lineage_table,
        'approval_table': approval_table,
    }


def build_audit_log_view(log_entries: list[dict[str, object]]) -> pd.DataFrame:
    if not log_entries:
        return pd.DataFrame(columns=['sequence', 'timestamp', 'action', 'user_interaction', 'analysis_step', 'detail'])
    audit = pd.DataFrame(log_entries).copy()
    rename_map = {
        'action_type': 'action',
    }
    audit = audit.rename(columns=rename_map)
    for column in ['user_interaction', 'analysis_step']:
        if column not in audit.columns:
            audit[column] = 'Session activity'
    ordered = ['sequence', 'timestamp', 'action', 'user_interaction', 'analysis_step', 'detail']
    return audit[[column for column in ordered if column in audit.columns]]


def build_dataset_comparison_dashboard(comparison_rows: list[dict[str, object]]) -> dict[str, object]:
    columns = [
        'dataset_name',
        'source_mode',
        'rows',
        'columns',
        'missing_values',
        'missing_rate',
        'schema_coverage',
        'readiness_score',
        'quality_score',
        'ready_modules',
        'structure_confidence',
        'semantic_confidence',
    ]
    if len(comparison_rows) < 2:
        return {
            'available': False,
            'reason': 'Add at least one additional dataset to compare it side by side with the active dataset.',
            'summary_table': pd.DataFrame(columns=columns),
            'metric_table': pd.DataFrame(columns=['dataset_name', 'metric', 'value']),
        }

    summary = pd.DataFrame(comparison_rows)
    for column in columns:
        if column not in summary.columns:
            summary[column] = pd.NA
    summary = summary[columns].sort_values(['source_mode', 'dataset_name']).reset_index(drop=True)
    summary['missing_rate'] = pd.to_numeric(summary['missing_rate'], errors='coerce')
    summary['schema_coverage'] = pd.to_numeric(summary['schema_coverage'], errors='coerce')
    summary['readiness_score'] = pd.to_numeric(summary['readiness_score'], errors='coerce')
    summary['quality_score'] = pd.to_numeric(summary['quality_score'], errors='coerce')
    summary['structure_confidence'] = pd.to_numeric(summary['structure_confidence'], errors='coerce')
    summary['semantic_confidence'] = pd.to_numeric(summary['semantic_confidence'], errors='coerce')

    metric_table = summary.melt(
        id_vars=['dataset_name'],
        value_vars=['rows', 'missing_values', 'schema_coverage', 'readiness_score', 'quality_score', 'ready_modules'],
        var_name='metric',
        value_name='value',
    )
    metric_labels = {
        'rows': 'Rows',
        'missing_values': 'Missing Values',
        'schema_coverage': 'Schema Coverage',
        'readiness_score': 'Readiness Score',
        'quality_score': 'Data Quality Score',
        'ready_modules': 'Ready Modules',
    }
    metric_table['metric'] = metric_table['metric'].map(metric_labels).fillna(metric_table['metric'])

    readiness_min = summary['readiness_score'].dropna().min() if summary['readiness_score'].notna().any() else None
    readiness_max = summary['readiness_score'].dropna().max() if summary['readiness_score'].notna().any() else None
    quality_best_row = summary.loc[summary['quality_score'].astype(float).idxmax()] if summary['quality_score'].notna().any() else None
    overview = {
        'dataset_count': int(len(summary)),
        'readiness_gap': float(readiness_max - readiness_min) if readiness_min is not None and readiness_max is not None else None,
        'best_quality_dataset': quality_best_row['dataset_name'] if quality_best_row is not None else None,
        'best_quality_score': float(quality_best_row['quality_score']) if quality_best_row is not None and pd.notna(quality_best_row['quality_score']) else None,
    }

    return {
        'available': True,
        'summary_table': summary,
        'metric_table': metric_table,
        'overview': overview,
    }
