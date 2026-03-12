from __future__ import annotations

import unittest

import pandas as pd

from src.healthcare_analysis import care_pathway_view, readmission_risk_analytics, segment_discovery
from src.insights_engine import build_automated_insight_board


def _sample_clinical_frame() -> tuple[pd.DataFrame, dict[str, str]]:
    data = pd.DataFrame(
        {
            'patient_id': [f'P{i}' for i in range(1, 21)],
            'age': [72, 69, 45, 48, 63, 67, 38, 41, 75, 77, 59, 61, 64, 68, 70, 52, 66, 71, 58, 62],
            'gender': ['F', 'F', 'M', 'M', 'F', 'F', 'M', 'M', 'F', 'F', 'M', 'F', 'F', 'M', 'F', 'M', 'F', 'F', 'M', 'F'],
            'smoking_status': ['Smoker', 'Smoker', 'Non-smoker', 'Smoker', 'Smoker', 'Smoker', 'Non-smoker', 'Non-smoker', 'Smoker', 'Smoker', 'Non-smoker', 'Smoker', 'Smoker', 'Non-smoker', 'Smoker', 'Non-smoker', 'Smoker', 'Smoker', 'Non-smoker', 'Smoker'],
            'cancer_stage': ['III', 'IV', 'II', 'II', 'III', 'IV', 'I', 'II', 'IV', 'III', 'II', 'III', 'IV', 'I', 'III', 'II', 'III', 'IV', 'II', 'III'],
            'treatment_type': ['Chemo', 'Chemo', 'Radiation', 'Radiation', 'Chemo', 'Immunotherapy', 'Radiation', 'Radiation', 'Chemo', 'Chemo', 'Radiation', 'Immunotherapy', 'Chemo', 'Radiation', 'Chemo', 'Radiation', 'Chemo', 'Immunotherapy', 'Radiation', 'Chemo'],
            'survived': ['No', 'No', 'Yes', 'Yes', 'No', 'No', 'Yes', 'Yes', 'No', 'No', 'Yes', 'No', 'No', 'Yes', 'No', 'Yes', 'No', 'No', 'Yes', 'No'],
            'diagnosis_date': pd.date_range('2024-01-01', periods=20, freq='10D'),
            'end_treatment_date': pd.date_range('2024-02-15', periods=20, freq='14D'),
            'department': ['Oncology', 'Oncology', 'Medicine', 'Medicine', 'Oncology', 'Oncology', 'Medicine', 'Medicine', 'Oncology', 'Oncology', 'Medicine', 'Oncology', 'Oncology', 'Medicine', 'Oncology', 'Medicine', 'Oncology', 'Oncology', 'Medicine', 'Oncology'],
            'diagnosis_code': ['C34', 'C50', 'I10', 'E11', 'C34', 'C18', 'I10', 'E11', 'C50', 'C34', 'E11', 'C18', 'C34', 'I10', 'C50', 'E11', 'C34', 'C18', 'I10', 'C50'],
            'length_of_stay': [7, 8, 3, 4, 6, 9, 2, 3, 8, 7, 4, 6, 9, 2, 7, 3, 8, 10, 4, 6],
            'readmission': [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1],
            'bmi': [34, 35, 24, 26, 33, 36, 23, 25, 37, 34, 27, 31, 35, 24, 33, 26, 34, 36, 25, 32],
            'comorbidities': [2, 3, 0, 1, 2, 3, 0, 0, 4, 3, 1, 2, 3, 0, 2, 1, 2, 3, 1, 2],
        }
    )
    canonical = {
        'patient_id': 'patient_id',
        'age': 'age',
        'gender': 'gender',
        'smoking_status': 'smoking_status',
        'cancer_stage': 'cancer_stage',
        'treatment_type': 'treatment_type',
        'survived': 'survived',
        'diagnosis_date': 'diagnosis_date',
        'end_treatment_date': 'end_treatment_date',
        'department': 'department',
        'diagnosis_code': 'diagnosis_code',
        'length_of_stay': 'length_of_stay',
        'readmission': 'readmission',
        'bmi': 'bmi',
        'comorbidities': 'comorbidities',
    }
    return data, canonical


class HealthcareRegressionTests(unittest.TestCase):
    def test_readmission_handles_categorical_age_band(self) -> None:
        data, canonical = _sample_clinical_frame()
        result = readmission_risk_analytics(data, canonical)
        self.assertTrue(result.get('available'))
        self.assertIn('by_age_band', result)
        self.assertFalse(result['by_age_band'].empty)

    def test_segment_discovery_returns_enhanced_outputs(self) -> None:
        data, canonical = _sample_clinical_frame()
        result = segment_discovery(data, canonical)
        self.assertTrue(result.get('available'))
        self.assertIn('metric_leaders', result)
        self.assertIn('summary', result)
        self.assertIn('priority_band', result['segment_table'].columns)
        self.assertIn('dominant_signal', result['segment_table'].columns)

    def test_care_pathway_returns_bottleneck_outputs(self) -> None:
        data, canonical = _sample_clinical_frame()
        result = care_pathway_view(data, canonical)
        self.assertTrue(result.get('available'))
        self.assertIn('bottleneck_summary', result)
        self.assertIn('poor_outcome_pathways', result)
        self.assertTrue(bool(result.get('summary')))

    def test_automated_insight_board_tolerates_partial_survival_payload(self) -> None:
        data, canonical = _sample_clinical_frame()
        readmission = readmission_risk_analytics(data, canonical)
        result = build_automated_insight_board(
            {
                'rows': len(data),
                'columns': len(data.columns),
                'duplicate_rows': int(data.duplicated().sum()),
                'missing_values': int(data.isna().sum().sum()),
                'memory_usage_bytes': int(data.memory_usage(deep=True).sum()),
            },
            {
                'readiness_score': 0.82,
                'available_count': 4,
                'ready_modules': [{'module': 'Readmission-Style Analysis'}],
                'missing_modules': [],
            },
            {
                'risk_segmentation': {
                    'available': True,
                    'segment_table': pd.DataFrame(
                        {
                            'risk_segment': ['High Risk', 'Medium Risk'],
                            'patient_count': [10, 5],
                            'percentage': [0.5, 0.25],
                        }
                    ),
                },
                'readmission': readmission,
                'survival_outcomes': {
                    'available': True,
                    'stage_table': pd.DataFrame({'cancer_stage': ['III'], 'survival_rate': [0.2]}),
                    'duration_summary': {'average_duration_days': 80.0},
                },
                'anomaly_detection': {'available': False},
                'scenario': {'available': False},
                'cost': {'available': False},
                'explainability_fairness': {'available': False},
            },
            {'summary_lines': ['Readmission risk is concentrated in oncology cohorts.']},
            pd.DataFrame(
                {
                    'priority': ['High'],
                    'recommendation_title': ['Review high-risk oncology discharges'],
                    'rationale': ['High readmission segments cluster in oncology cohorts.'],
                }
            ),
        )
        self.assertTrue(result.get('available'))
        self.assertTrue(len(result.get('top_findings', [])) >= 1)


if __name__ == '__main__':
    unittest.main()
