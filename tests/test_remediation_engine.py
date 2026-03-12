from __future__ import annotations

import unittest

import pandas as pd

from app import run_pipeline
from src.enterprise_features import build_quality_rule_engine
from src.export_utils import build_readmission_summary_text
from src.remediation_engine import (
    add_synthetic_clinical_labels,
    add_synthetic_cost_fields,
    add_synthetic_readmission_fields,
    remediate_bmi,
)


class RemediationEngineTests(unittest.TestCase):
    def test_bmi_median_remediation_flags_and_replaces(self) -> None:
        frame = pd.DataFrame({'bmi': [22.0, 31.0, 200.0, 'bad']})
        remediated, summary = remediate_bmi(frame, mode='median')
        self.assertEqual(int(summary['total_bmi_outliers']), 2)
        self.assertTrue(bool(remediated.loc[2, 'bmi_outlier_flag']))
        self.assertTrue(bool(remediated.loc[3, 'bmi_outlier_flag']))
        self.assertAlmostEqual(float(remediated.loc[2, 'bmi']), 26.5, places=2)
        self.assertEqual(remediated.loc[2, 'bmi_remediation_action'], 'Replaced with median BMI')

    def test_bmi_clip_mode(self) -> None:
        frame = pd.DataFrame({'bmi': [8.0, 85.0, 30.0]})
        remediated, _ = remediate_bmi(frame, mode='clip')
        self.assertEqual(float(remediated.loc[0, 'bmi']), 10.0)
        self.assertEqual(float(remediated.loc[1, 'bmi']), 80.0)
        self.assertEqual(float(remediated.loc[2, 'bmi']), 30.0)

    def test_synthetic_cost_is_deterministic_and_non_negative(self) -> None:
        frame = pd.DataFrame(
            {
                'age': [60, 45],
                'bmi': [32, 24],
                'smoking_history': ['current', 'never'],
                'hypertension': [1, 0],
                'heart_disease': [0, 1],
            }
        )
        first, _ = add_synthetic_cost_fields(frame)
        second, _ = add_synthetic_cost_fields(frame)
        self.assertIn('estimated_cost', first.columns)
        self.assertTrue((first['estimated_cost'] >= 0).all())
        self.assertListEqual(first['estimated_cost'].tolist(), second['estimated_cost'].tolist())

    def test_synthetic_diagnosis_rules(self) -> None:
        frame = pd.DataFrame(
            {
                'age': [70, 45],
                'bmi': [34, 22],
                'smoking_history': ['current', 'never'],
                'hypertension': [1, 0],
                'heart_disease': [0, 1],
            }
        )
        derived, _ = add_synthetic_clinical_labels(frame)
        self.assertEqual(derived.loc[0, 'diagnosis_code'], 'SYN-OBE')
        self.assertEqual(derived.loc[0, 'clinical_risk_label'], 'Elevated Older Adult Risk')
        self.assertIn(derived.loc[1, 'diagnosis_code'], {'SYN-HDR', 'SYN-GEN'})

    def test_synthetic_readmission_prevalence_sensible(self) -> None:
        frame = pd.DataFrame(
            {
                'age': [30, 55, 70, 80, 65, 40],
                'bmi': [22, 29, 31, 40, 27, 35],
                'smoking_history': ['never', 'former', 'current', 'current', 'never', 'current'],
                'hypertension': [0, 1, 1, 1, 0, 1],
                'heart_disease': [0, 0, 1, 1, 0, 0],
                'blood_glucose_level': [100, 130, 210, 240, 115, 180],
            }
        )
        derived, summary = add_synthetic_readmission_fields(frame)
        self.assertIn('readmission_flag', derived.columns)
        self.assertIn('readmission_risk_proxy', derived.columns)
        self.assertGreater(summary['prevalence'], 0.05)
        self.assertLess(summary['prevalence'], 0.5)

    def test_rule_engine_expansion_adds_duplicate_and_severity_outputs(self) -> None:
        frame = pd.DataFrame(
            {
                'age': [25, 250, 250],
                'bmi': [22, 200, 200],
                'patient_id': ['a', 'b', 'b'],
            }
        )
        rules = build_quality_rule_engine(frame, {'age': 'age', 'bmi': 'bmi', 'patient_id': 'patient_id'})
        self.assertTrue(rules['available'])
        self.assertIn('severity_summary', rules)
        self.assertIn('rule_details', rules)
        self.assertGreaterEqual(rules['overview']['total_failed_rules'], 2)
        self.assertIn('Duplicate rows detected in the current dataset.', set(rules['summary_table']['rule_name']))

    def test_pipeline_tracks_synthetic_fields_and_improves_readiness(self) -> None:
        frame = pd.DataFrame(
            {
                'patient_id': ['p1', 'p2', 'p3', 'p4'],
                'year': [2020, 2020, 2021, 2021],
                'age': [67, 54, 72, 61],
                'bmi': [42, 180, 29, 15],
                'smoking_history': ['current', 'never', 'former', 'current'],
                'hypertension': [1, 0, 1, 1],
                'heart_disease': [0, 1, 1, 0],
            }
        )
        pipeline = run_pipeline(frame, 'demo.csv', {'source_mode': 'Uploaded CSV', 'description': '', 'best_for': '', 'file_size_mb': 0.1})
        remediation = pipeline['remediation_context']
        self.assertGreater(remediation['synthetic_field_count'], 0)
        self.assertIn('support_type', pipeline['readiness']['readiness_table'].columns)
        self.assertGreaterEqual(pipeline['readiness']['synthetic_supported_modules'], 1)
        derived_fields = pipeline['lineage']['derived_fields_table']
        self.assertTrue((derived_fields['source_column'].astype(str) == 'cost_amount').any())
        self.assertTrue((derived_fields['source_column'].astype(str) == 'readmission').any())

    def test_readmission_export_handles_synthetic_mode(self) -> None:
        payload = {
            'available': True,
            'source': 'synthetic',
            'overview': {'overall_readmission_rate': 0.14, 'readmission_count': 14, 'records_in_scope': 100},
            'high_risk_segments': pd.DataFrame([{'segment_type': 'Department', 'segment_value': 'Cardiology', 'readmission_rate': 0.24, 'record_count': 12}]),
            'driver_table': pd.DataFrame([{'factor': 'Age Band', 'driver_group': '65+', 'gap_vs_overall': 0.08, 'readmission_rate': 0.22}]),
        }
        text = build_readmission_summary_text('demo.csv', payload, pd.DataFrame()).decode('utf-8')
        self.assertIn('synthetic or demo-derived support', text)


if __name__ == '__main__':
    unittest.main()
