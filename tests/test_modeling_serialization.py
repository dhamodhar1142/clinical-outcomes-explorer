from __future__ import annotations

import unittest

import pandas as pd

from src.modeling_studio import build_predictive_model


def _sample_modeling_frame() -> tuple[pd.DataFrame, dict[str, str], list[str]]:
    rows = 120
    data = pd.DataFrame(
        {
            'patient_id': [f'P{i:03d}' for i in range(rows)],
            'age': [35 + (i % 40) for i in range(rows)],
            'gender': ['F' if i % 2 == 0 else 'M' for i in range(rows)],
            'length_of_stay': [2 + (i % 8) for i in range(rows)],
            'visit_date': pd.date_range('2024-01-01', periods=rows, freq='D'),
        }
    )
    data['age_interval'] = pd.cut(data['age'], bins=[0, 40, 55, 70, 120])
    data['readmission'] = (
        (data['age'] >= 60).astype(int)
        | ((data['length_of_stay'] >= 6) & (data['gender'] == 'F')).astype(int)
    ).astype(int)

    canonical_map = {
        'patient_id': 'patient_id',
        'age': 'age',
        'gender': 'gender',
        'readmission': 'readmission',
    }
    feature_columns = ['age_interval', 'visit_date', 'gender', 'length_of_stay']
    return data, canonical_map, feature_columns


class ModelingSerializationTests(unittest.TestCase):
    def test_predictive_model_normalizes_interval_and_datetime_features(self) -> None:
        data, canonical_map, feature_columns = _sample_modeling_frame()
        result = build_predictive_model(
            data,
            canonical_map,
            target_column='readmission',
            feature_columns=feature_columns,
            model_type='Logistic Regression',
        )

        self.assertTrue(result.get('available'))

        transformations = result.get('feature_transformations', pd.DataFrame())
        self.assertFalse(transformations.empty)
        self.assertIn('interval_to_bin_index', transformations['transformation'].tolist())
        self.assertIn('datetime_to_unix_timestamp', transformations['transformation'].tolist())

        distribution = result.get('prediction_distribution', pd.DataFrame())
        self.assertIn('probability_band_index', distribution.columns)
        self.assertIn('probability_band_label', distribution.columns)
        self.assertTrue(pd.api.types.is_integer_dtype(distribution['probability_band_index']))
        self.assertFalse(distribution['probability_band_index'].map(lambda value: isinstance(value, pd.Interval)).any())

        high_risk_rows = result.get('high_risk_rows', pd.DataFrame())
        if not high_risk_rows.empty:
            flattened = high_risk_rows.astype('object').stack()
            self.assertFalse(flattened.map(lambda value: isinstance(value, pd.Interval)).any())


if __name__ == '__main__':
    unittest.main()
