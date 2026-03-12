from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


try:
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    SKLEARN_AVAILABLE = True
    SKLEARN_IMPORT_ERROR = None
except Exception as import_error:  # pragma: no cover - fallback path
    ColumnTransformer = None
    RandomForestClassifier = None
    SimpleImputer = None
    LogisticRegression = None
    confusion_matrix = None
    accuracy_score = None
    precision_score = None
    recall_score = None
    f1_score = None
    roc_auc_score = None
    GradientBoostingClassifier = None
    train_test_split = None
    Pipeline = None
    OneHotEncoder = None
    StandardScaler = None
    SKLEARN_AVAILABLE = False
    SKLEARN_IMPORT_ERROR = import_error

try:
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True
except Exception:
    XGBClassifier = None
    XGBOOST_AVAILABLE = False


PREFERRED_TARGETS = [
    'readmission',
    'survived',
]

PREFERRED_FEATURES = [
    'age',
    'gender',
    'diagnosis_code',
    'department',
    'length_of_stay',
    'cost_amount',
    'smoking_status',
    'cancer_stage',
    'treatment_type',
    'comorbidities',
    'bmi',
    'cholesterol_level',
]

WEAK_TARGET_FIELDS = {'gender', 'race', 'race_africanamerican', 'race_asian', 'race_caucasian', 'race_hispanic', 'race_other'}


@dataclass
class TargetCandidate:
    canonical_field: str
    source_column: str
    target_type: str
    class_count: int
    reason: str


def _is_interval_categorical(series: pd.Series) -> bool:
    return (
        isinstance(series.dtype, pd.CategoricalDtype)
        and hasattr(series, 'cat')
        and len(series.cat.categories) > 0
        and isinstance(series.cat.categories.dtype, pd.IntervalDtype)
    )


def _normalize_feature_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    normalized = frame.copy()
    transformations: list[dict[str, object]] = []

    for column in normalized.columns:
        series = normalized[column]
        if isinstance(series.dtype, pd.IntervalDtype) or _is_interval_categorical(series):
            categorical = pd.Categorical(series)
            normalized[column] = (
                pd.Series(categorical.codes, index=series.index)
                .replace(-1, pd.NA)
                .astype('Float64')
            )
            transformations.append(
                {
                    'feature': column,
                    'transformation': 'interval_to_bin_index',
                    'detail': 'Converted interval-style bins into numeric bin indices for model compatibility.',
                }
            )
        elif pd.api.types.is_datetime64_any_dtype(series):
            normalized[column] = (
                pd.Series(series.astype('int64') // 10**9, index=series.index)
                .where(series.notna(), pd.NA)
                .astype('Float64')
            )
            transformations.append(
                {
                    'feature': column,
                    'transformation': 'datetime_to_unix_timestamp',
                    'detail': 'Converted datetime values into Unix timestamps for model compatibility.',
                }
            )

    return normalized, pd.DataFrame(transformations)


def _build_probability_distribution(probabilities: pd.Series | Any) -> pd.DataFrame:
    bin_edges = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    labels = ['0-20%', '20-40%', '40-60%', '60-80%', '80-100%']
    band_index = pd.cut(probabilities, bins=bin_edges, labels=False, include_lowest=True)
    counts = (
        pd.Series(band_index, dtype='Int64')
        .value_counts(sort=False)
        .reindex(range(len(labels)), fill_value=0)
        .reset_index()
    )
    counts.columns = ['probability_band_index', 'record_count']
    counts['probability_band_index'] = counts['probability_band_index'].astype(int)
    counts['probability_band_label'] = counts['probability_band_index'].map(dict(enumerate(labels)))
    counts['band_start'] = counts['probability_band_index'].map(dict(enumerate(bin_edges[:-1])))
    counts['band_end'] = counts['probability_band_index'].map(dict(enumerate(bin_edges[1:])))
    return counts[['probability_band_index', 'probability_band_label', 'band_start', 'band_end', 'record_count']]


def _binary_target(series: pd.Series) -> pd.Series:
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
        'readmitted': 1,
        'not readmitted': 0,
    }
    text = series.astype(str).str.strip().str.lower()
    mapped = text.map(mapping)
    numeric = pd.to_numeric(series, errors='coerce')
    if mapped.notna().sum() >= max(int(len(series.dropna()) * 0.5), 1):
        return mapped
    return numeric


def _candidate_target(data: pd.DataFrame, source_column: str) -> TargetCandidate | None:
    series = data[source_column]
    binary = _binary_target(series)
    valid_binary = binary.dropna()
    if valid_binary.nunique() == 2 and len(valid_binary) >= 30:
        return TargetCandidate(
            canonical_field=source_column,
            source_column=source_column,
            target_type='binary',
            class_count=2,
            reason='Looks like a binary healthcare outcome or flag.',
        )
    text = series.dropna().astype(str)
    if 2 <= text.nunique() <= 6 and len(text) >= 30:
        return TargetCandidate(
            canonical_field=source_column,
            source_column=source_column,
            target_type='categorical',
            class_count=int(text.nunique()),
            reason='Looks like a small multi-class field that could support classification review.',
        )
    return None


def get_modeling_candidates(data: pd.DataFrame, canonical_map: dict[str, str]) -> dict[str, object]:
    if data.empty:
        return {'available': False, 'reason': 'No data is available for predictive modeling.'}

    rows: list[dict[str, object]] = []
    candidates: list[TargetCandidate] = []
    seen: set[str] = set()
    for canonical in PREFERRED_TARGETS:
        column = canonical_map.get(canonical)
        if column and column in data.columns and column not in seen:
            candidate = _candidate_target(data, column)
            if candidate:
                candidate.canonical_field = canonical
                candidates.append(candidate)
                seen.add(column)
    for column in data.columns:
        if column in seen:
            continue
        candidate = _candidate_target(data, column)
        if candidate:
            candidates.append(candidate)
            seen.add(column)

    for candidate in candidates:
        rows.append({
            'target_field': candidate.canonical_field,
            'source_column': candidate.source_column,
            'target_type': candidate.target_type,
            'class_count': candidate.class_count,
            'reason': candidate.reason,
        })

    if not rows:
        return {
            'available': False,
            'reason': 'No suitable target variable was detected. Add or map a binary or low-cardinality outcome field such as readmission or survived.',
        }

    return {
        'available': True,
        'candidate_table': pd.DataFrame(rows),
    }


def _default_features(data: pd.DataFrame, canonical_map: dict[str, str], target_column: str) -> list[str]:
    features: list[str] = []
    for canonical in PREFERRED_FEATURES:
        column = canonical_map.get(canonical)
        if column and column in data.columns and column != target_column and column not in features:
            features.append(column)
    for column in data.columns:
        if column != target_column and column not in features:
            if pd.api.types.is_numeric_dtype(data[column]) or data[column].dropna().astype(str).nunique() <= 20:
                features.append(column)
    return features[:12]


def _prepare_target(data: pd.DataFrame, target_column: str) -> tuple[pd.Series, str]:
    series = data[target_column]
    binary = _binary_target(series)
    valid = binary.dropna()
    if valid.nunique() == 2:
        return binary, 'binary'
    return series.astype(str), 'categorical'


def _build_age_band(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors='coerce')
    return pd.cut(numeric, bins=[-1, 39, 59, 74, 200], labels=['18-39', '40-59', '60-74', '75+'])


def _estimator_for_model(model_type: str):
    if model_type == 'Random Forest':
        return RandomForestClassifier(n_estimators=200, random_state=42, min_samples_leaf=2)
    if model_type == 'Gradient Boosting':
        return GradientBoostingClassifier(random_state=42)
    if model_type == 'XGBoost' and XGBOOST_AVAILABLE:
        return XGBClassifier(
            n_estimators=160,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            objective='binary:logistic',
            eval_metric='logloss',
            random_state=42,
        )
    return LogisticRegression(max_iter=1000, class_weight='balanced')


def build_predictive_model(
    data: pd.DataFrame,
    canonical_map: dict[str, str],
    target_column: str,
    feature_columns: list[str],
    model_type: str,
) -> dict[str, object]:
    if not SKLEARN_AVAILABLE:
        return {
            'available': False,
            'reason': f'Predictive modeling needs scikit-learn in the runtime environment. Import error: {SKLEARN_IMPORT_ERROR}',
        }
    if data.empty:
        return {'available': False, 'reason': 'No data is available for predictive modeling.'}
    if not target_column or target_column not in data.columns:
        return {'available': False, 'reason': 'Select a valid target field to continue.'}
    if model_type == 'XGBoost' and not XGBOOST_AVAILABLE:
        return {'available': False, 'reason': 'XGBoost is not installed in the current runtime, so this comparison path is skipped safely.'}
    feature_columns = [column for column in feature_columns if column in data.columns and column != target_column]
    if len(feature_columns) < 2:
        return {'available': False, 'reason': 'Select at least two usable feature fields to build a guided model.'}

    target, target_type = _prepare_target(data, target_column)
    modeling_features, feature_transformations = _normalize_feature_frame(data[feature_columns].copy())
    modeling = modeling_features.copy()
    modeling['_target'] = target
    modeling = modeling.dropna(subset=['_target']).copy()
    if len(modeling) < 60:
        return {'available': False, 'reason': 'The current selection does not contain enough usable rows to support a stable train/test split.'}

    class_counts = modeling['_target'].value_counts(dropna=True)
    if len(class_counts) < 2:
        return {'available': False, 'reason': 'The selected target field contains only one usable class after cleaning.'}
    if class_counts.min() < 10:
        return {'available': False, 'reason': 'One target class is too small for a reliable demo-grade model. Narrow the target or use a larger cohort.'}

    numeric_features = [column for column in feature_columns if pd.api.types.is_numeric_dtype(modeling[column])]
    categorical_features = [column for column in feature_columns if column not in numeric_features]

    preprocess = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler()),
            ]), numeric_features),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore')),
            ]), categorical_features),
        ],
        remainder='drop',
    )

    estimator = _estimator_for_model(model_type)

    model = Pipeline([
        ('preprocess', preprocess),
        ('model', estimator),
    ])

    X = modeling[feature_columns]
    y = modeling['_target']
    stratify = y if len(y.value_counts()) > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=stratify)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    probabilities = None
    roc_auc = None
    if hasattr(model.named_steps['model'], 'predict_proba'):
        try:
            probabilities = model.predict_proba(X_test)
            if target_type == 'binary' and probabilities is not None and probabilities.shape[1] >= 2:
                roc_auc = float(roc_auc_score(y_test, probabilities[:, 1]))
        except Exception:
            probabilities = None

    matrix_labels = sorted(pd.Series(y).dropna().astype(str).unique().tolist())
    confusion = confusion_matrix(y_test.astype(str), pd.Series(y_pred).astype(str), labels=matrix_labels)
    confusion_table = pd.DataFrame(confusion, index=[f'actual_{label}' for label in matrix_labels], columns=[f'predicted_{label}' for label in matrix_labels])

    feature_names = model.named_steps['preprocess'].get_feature_names_out()
    estimator_step = model.named_steps['model']
    if model_type == 'Random Forest':
        importance_values = estimator_step.feature_importances_
    elif model_type == 'Gradient Boosting' and getattr(estimator_step, 'feature_importances_', None) is not None:
        importance_values = estimator_step.feature_importances_
    elif model_type == 'XGBoost' and getattr(estimator_step, 'feature_importances_', None) is not None:
        importance_values = estimator_step.feature_importances_
    else:
        importance_values = abs(estimator_step.coef_[0]) if getattr(estimator_step, 'coef_', None) is not None else [0.0] * len(feature_names)
    importance_table = pd.DataFrame({
        'feature': feature_names,
        'importance': importance_values,
    }).sort_values('importance', ascending=False).head(15).reset_index(drop=True)

    prediction_distribution = pd.DataFrame()
    high_risk_rows = pd.DataFrame()
    prediction_table = pd.DataFrame()
    if probabilities is not None and probabilities.shape[1] >= 2:
        positive_probability = probabilities[:, 1]
        prediction_distribution = _build_probability_distribution(positive_probability)
        test_preview = X_test.copy()
        test_preview['actual_target'] = y_test.values
        test_preview['predicted_probability'] = positive_probability
        test_preview['predicted_target'] = y_pred
        gender_col = canonical_map.get('gender')
        age_col = canonical_map.get('age')
        if age_col and age_col in test_preview.columns and 'age_band' not in test_preview.columns:
            test_preview['age_band'] = _build_age_band(test_preview[age_col]).astype(str)
        preview_cols = [column for column in [gender_col, 'age_band'] if column and column in test_preview.columns]
        candidate_ids = [column for column in [canonical_map.get('patient_id'), canonical_map.get('member_id'), canonical_map.get('entity_id'), canonical_map.get('encounter_id')] if column and column in test_preview.columns]
        preview_cols = candidate_ids + preview_cols + [column for column in feature_columns[:5] if column in test_preview.columns and column not in preview_cols]
        high_risk_rows = test_preview.sort_values('predicted_probability', ascending=False)[preview_cols + ['actual_target', 'predicted_probability']].head(20)
        prediction_table = test_preview.reset_index(drop=True)

    average = 'binary' if target_type == 'binary' else 'weighted'
    metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred, average=average, zero_division=0)),
        'recall': float(recall_score(y_test, y_pred, average=average, zero_division=0)),
        'f1': float(f1_score(y_test, y_pred, average=average, zero_division=0)),
    }
    weak_target_note = None
    if target_column.lower() in WEAK_TARGET_FIELDS:
        weak_target_note = 'The selected target is demographic rather than clinical or operational, so use this model as a technical demonstration rather than a decision-support model.'
    synthetic_features_used = [feature for feature in feature_columns if str(feature).startswith(('estimated_', 'diagnosis_', 'clinical_risk_', 'readmission_'))]

    return {
        'available': True,
        'target_column': target_column,
        'feature_columns': feature_columns,
        'model_type': model_type,
        'target_type': target_type,
        'feature_transformations': feature_transformations,
        'train_test_summary': pd.DataFrame([{
            'train_rows': int(len(X_train)),
            'test_rows': int(len(X_test)),
            'feature_count': int(len(feature_columns)),
            'target_classes': int(len(class_counts)),
        }]),
        **metrics,
        'roc_auc': roc_auc,
        'confusion_matrix': confusion_table,
        'feature_importance': importance_table,
        'prediction_distribution': prediction_distribution,
        'high_risk_rows': high_risk_rows,
        'prediction_table': prediction_table,
        'synthetic_features_used': synthetic_features_used,
        'weak_target_note': weak_target_note,
    }


def default_modeling_selection(data: pd.DataFrame, canonical_map: dict[str, str]) -> dict[str, object]:
    candidates = get_modeling_candidates(data, canonical_map)
    if not candidates.get('available'):
        return candidates
    candidate_table = candidates['candidate_table']
    target_column = candidate_table.iloc[0]['source_column']
    feature_columns = _default_features(data, canonical_map, target_column)
    return {
        **candidates,
        'default_target': target_column,
        'default_features': feature_columns,
    }


def _strip_feature_name(feature_name: str) -> str:
    if '__' in feature_name:
        feature_name = feature_name.split('__', 1)[1]
    if '_' in feature_name and not feature_name.startswith('num_'):
        parts = feature_name.split('_')
        if len(parts) > 1 and parts[0] in {'cat', 'num'}:
            return '_'.join(parts[1:])
    return feature_name


def build_prediction_explainability(model_result: dict[str, object]) -> dict[str, object]:
    if not model_result.get('available'):
        return {'available': False, 'reason': 'Model output is not available for explainability review.'}

    importance = model_result.get('feature_importance', pd.DataFrame())
    high_risk_rows = model_result.get('high_risk_rows', pd.DataFrame())
    if not isinstance(importance, pd.DataFrame) or importance.empty:
        return {'available': False, 'reason': 'Feature importance is not available for the current model output.'}

    top_drivers = importance.copy()
    top_drivers['display_feature'] = top_drivers['feature'].astype(str).map(_strip_feature_name)
    top_drivers = top_drivers[['display_feature', 'importance']].rename(columns={'display_feature': 'feature'}).drop_duplicates(subset=['feature']).head(10).reset_index(drop=True)

    narratives: list[str] = []
    for row in top_drivers.head(3).itertuples(index=False):
        narratives.append(f"{row.feature} is one of the strongest contributors in the current model, based on its relative importance score of {float(row.importance):.3f}.")

    row_explanations = pd.DataFrame()
    segment_explanations = pd.DataFrame()
    if isinstance(high_risk_rows, pd.DataFrame) and not high_risk_rows.empty:
        candidate_features = [feature for feature in top_drivers['feature'].tolist() if feature in high_risk_rows.columns]
        row_records: list[dict[str, object]] = []
        segment_records: list[dict[str, object]] = []
        for _, row in high_risk_rows.head(10).iterrows():
            top_values: list[str] = []
            for feature in candidate_features[:3]:
                value = row.get(feature)
                if pd.notna(value):
                    top_values.append(f'{feature}={value}')
            identifier_candidates = [column for column in high_risk_rows.columns if column not in candidate_features and column not in {'actual_target', 'predicted_probability'}]
            identifier = str(row.get(identifier_candidates[0])) if identifier_candidates else f'row_{_}'
            row_records.append({
                'row_identifier': identifier,
                'predicted_probability': float(row.get('predicted_probability', 0.0)),
                'explanation': '; '.join(top_values) if top_values else 'High predicted probability with limited row-level explanatory fields.',
            })
        row_explanations = pd.DataFrame(row_records)

        if candidate_features:
            first_feature = candidate_features[0]
            grouped = high_risk_rows.groupby(high_risk_rows[first_feature].astype(str)).agg(
                row_count=('predicted_probability', 'size'),
                average_predicted_probability=('predicted_probability', 'mean'),
            ).reset_index().rename(columns={first_feature: 'segment_value'})
            grouped.insert(0, 'segment_feature', first_feature)
            grouped['explanation'] = grouped.apply(
                lambda row: f"{row['segment_feature']} = {row['segment_value']} averages a predicted probability of {float(row['average_predicted_probability']):.1%}.",
                axis=1,
            )
            segment_explanations = grouped.sort_values(['average_predicted_probability', 'row_count'], ascending=[False, False]).head(8).reset_index(drop=True)
            if not segment_explanations.empty:
                top_segment = segment_explanations.iloc[0]
                narratives.append(
                    f"The strongest predicted high-risk segment is {top_segment['segment_feature']} = {top_segment['segment_value']}, averaging {float(top_segment['average_predicted_probability']):.1%} predicted risk."
                )

    return {
        'available': True,
        'driver_table': top_drivers,
        'row_explanations': row_explanations,
        'segment_explanations': segment_explanations,
        'narrative': narratives,
    }


def build_model_fairness_review(
    model_result: dict[str, object],
    data: pd.DataFrame,
    canonical_map: dict[str, str],
) -> dict[str, object]:
    if not model_result.get('available'):
        return {'available': False, 'reason': 'Model output is not available for fairness review.'}
    prediction_table = model_result.get('prediction_table', pd.DataFrame())
    if not isinstance(prediction_table, pd.DataFrame) or prediction_table.empty:
        return {
            'available': False,
            'review_level': 'unavailable',
            'reason': 'Prediction-level outputs are not available for fairness review.',
            'comparison_table': pd.DataFrame(),
            'flags': pd.DataFrame(),
            'fairness_summary': pd.DataFrame(),
            'fairness_limitations': ['Prediction-level outputs were not available for subgroup review.'],
        }

    fairness_rows: list[dict[str, object]] = []
    flags: list[dict[str, object]] = []
    group_specs: list[tuple[str, str]] = []
    gender_col = canonical_map.get('gender')
    if gender_col and gender_col in prediction_table.columns:
        group_specs.append(('Gender', gender_col))
    if 'age_band' in prediction_table.columns:
        group_specs.append(('Age Band', 'age_band'))
    elif canonical_map.get('age') and canonical_map['age'] in prediction_table.columns:
        prediction_table = prediction_table.copy()
        prediction_table['age_band'] = _build_age_band(prediction_table[canonical_map['age']]).astype(str)
        group_specs.append(('Age Band', 'age_band'))

    if not group_specs:
        return {
            'available': True,
            'review_level': 'limited',
            'reason': 'No supported demographic grouping fields were available, so fairness review is limited.',
            'comparison_table': pd.DataFrame(),
            'flags': pd.DataFrame(),
            'fairness_summary': pd.DataFrame([{'review_level': 'limited', 'note': 'Add gender or age to enable subgroup fairness metrics.'}]),
            'fairness_limitations': ['No supported demographic grouping fields were available.'],
        }

    for label, column in group_specs:
        grouped = prediction_table.groupby(prediction_table[column].astype(str))
        for group_value, group in grouped:
            support = int(len(group))
            if support == 0:
                continue
            actual = pd.to_numeric(group['actual_target'], errors='coerce')
            predicted = pd.to_numeric(group['predicted_target'], errors='coerce')
            probability = pd.to_numeric(group['predicted_probability'], errors='coerce')
            valid = actual.notna() & predicted.notna()
            if valid.sum() < 5:
                continue
            actual = actual[valid]
            predicted = predicted[valid]
            tp = int(((predicted == 1) & (actual == 1)).sum())
            fp = int(((predicted == 1) & (actual == 0)).sum())
            fn = int(((predicted == 0) & (actual == 1)).sum())
            tn = int(((predicted == 0) & (actual == 0)).sum())
            positive_rate = float((predicted == 1).mean())
            precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
            recall = float(tp / (tp + fn)) if (tp + fn) else 0.0
            false_positive_rate = float(fp / (fp + tn)) if (fp + tn) else 0.0
            fairness_rows.append({
                'group_dimension': label,
                'group_value': str(group_value),
                'support_count': support,
                'positive_prediction_rate': positive_rate,
                'precision': precision,
                'recall': recall,
                'false_positive_rate': false_positive_rate,
                'average_predicted_probability': float(probability.mean()),
            })
        metric_frame = pd.DataFrame([row for row in fairness_rows if row['group_dimension'] == label])
        if len(metric_frame) >= 2:
            for metric in ['positive_prediction_rate', 'precision', 'recall', 'false_positive_rate']:
                gap = float(metric_frame[metric].max() - metric_frame[metric].min())
                if gap >= 0.15:
                    flags.append({
                        'severity': 'High' if gap >= 0.25 else 'Moderate',
                        'group_dimension': label,
                        'flag_type': metric.replace('_', ' ').title(),
                        'gap_value': gap,
                        'detail': f'{metric.replace("_", " ").title()} differs by {gap:.1%} across {label.lower()} groups.',
                    })

    comparison_table = pd.DataFrame(fairness_rows)
    if comparison_table.empty:
        return {
            'available': True,
            'review_level': 'limited',
            'reason': 'Demographic fields were available, but subgroup support was too low for stable fairness metrics.',
            'comparison_table': pd.DataFrame(),
            'flags': pd.DataFrame(),
            'fairness_summary': pd.DataFrame([{'review_level': 'limited', 'note': 'Subgroup support was too low for stable fairness metrics.'}]),
            'fairness_limitations': ['Subgroup support was too low for stable fairness metrics.'],
        }

    fairness_summary = comparison_table.groupby('group_dimension').agg(
        groups_reviewed=('group_value', 'nunique'),
        max_gap=('positive_prediction_rate', lambda s: float(s.max() - s.min()) if len(s) >= 2 else 0.0),
        total_support=('support_count', 'sum'),
    ).reset_index()
    return {
        'available': True,
        'review_level': 'full',
        'comparison_table': comparison_table,
        'flags': pd.DataFrame(flags),
        'fairness_summary': fairness_summary,
        'fairness_warnings': [flag['detail'] for flag in flags[:5]],
        'fairness_limitations': ['This is a transparent screening view, not a formal fairness certification or causal assessment.'],
        'narrative': 'This fairness review compares predicted-risk performance across available demographic groups. Treat large gaps as a review signal rather than proof of bias.',
    }


def build_model_comparison_studio(
    data: pd.DataFrame,
    canonical_map: dict[str, str],
    target_column: str,
    feature_columns: list[str],
) -> dict[str, object]:
    if not SKLEARN_AVAILABLE:
        return {'available': False, 'reason': f'Model comparison needs scikit-learn. Import error: {SKLEARN_IMPORT_ERROR}'}
    comparison_frame = data.copy()
    if len(comparison_frame) > 25000:
        comparison_frame = comparison_frame.sample(25000, random_state=42)
    model_names = ['Logistic Regression', 'Random Forest', 'Gradient Boosting']
    if XGBOOST_AVAILABLE:
        model_names.append('XGBoost')
    rows: list[dict[str, object]] = []
    results: dict[str, dict[str, object]] = {}
    for model_name in model_names:
        result = build_predictive_model(comparison_frame, canonical_map, target_column, feature_columns, model_name)
        if not result.get('available'):
            rows.append({
                'model_name': model_name,
                'status': 'Unavailable',
                'accuracy': None,
                'precision': None,
                'recall': None,
                'f1': None,
                'roc_auc': None,
                'train_rows': None,
                'test_rows': None,
                'feature_count': len(feature_columns),
                'notes': result.get('reason', 'Model could not be evaluated.'),
            })
            continue
        summary = result.get('train_test_summary', pd.DataFrame())
        train_rows = int(summary.iloc[0]['train_rows']) if isinstance(summary, pd.DataFrame) and not summary.empty else None
        test_rows = int(summary.iloc[0]['test_rows']) if isinstance(summary, pd.DataFrame) and not summary.empty else None
        rows.append({
            'model_name': model_name,
            'status': 'Available',
            'accuracy': result.get('accuracy'),
            'precision': result.get('precision'),
            'recall': result.get('recall'),
            'f1': result.get('f1'),
            'roc_auc': result.get('roc_auc'),
            'train_rows': train_rows,
            'test_rows': test_rows,
            'feature_count': len(feature_columns),
            'notes': result.get('weak_target_note', ''),
        })
        results[model_name] = result
    table = pd.DataFrame(rows)
    available = table[table['status'] == 'Available'].copy()
    if available.empty:
        return {'available': False, 'reason': 'No comparison model could run successfully on the current target and feature set.', 'model_comparison_table': table}
    available['roc_auc_rank'] = available['roc_auc'].fillna(-1.0)
    available['f1_rank'] = available['f1'].fillna(-1.0)
    available = available.sort_values(['roc_auc_rank', 'f1_rank', 'accuracy'], ascending=[False, False, False]).reset_index(drop=True)
    best_row = available.iloc[0]
    best_model_name = str(best_row['model_name'])
    best_model = results[best_model_name]
    notes = []
    if best_model.get('weak_target_note'):
        notes.append(best_model['weak_target_note'])
    synthetic_used = list(best_model.get('synthetic_features_used', []))
    if synthetic_used:
        notes.append('Synthetic helper fields participated in the winning model and are tracked separately for transparency.')
    notes.append('Models are ranked primarily by ROC AUC and then by F1 for binary classification.')
    return {
        'available': True,
        'model_comparison_table': table,
        'best_model_summary': {
            'model_name': best_model_name,
            'target_variable': target_column,
            'why_it_won': f"{best_model_name} led the comparison on ROC AUC and F1 for the selected target.",
            'roc_auc': best_row.get('roc_auc'),
            'f1': best_row.get('f1'),
            'synthetic_features_used': synthetic_used,
        },
        'best_model_name': best_model_name,
        'model_rankings': available[['model_name', 'roc_auc', 'f1', 'accuracy']].reset_index(drop=True),
        'comparison_notes': notes,
        'best_model_result': best_model,
    }
