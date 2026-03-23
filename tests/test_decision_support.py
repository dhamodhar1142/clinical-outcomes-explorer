from __future__ import annotations

import unittest

import pandas as pd

from src.decision_support import (
    build_executive_summary,
    build_intervention_recommendations,
    build_kpi_benchmarking_layer,
    build_prioritized_insights,
    build_scenario_simulation_studio,
)
from src.export_utils import build_executive_summary_text
from src.insights_engine import build_automated_insight_board
from src.modeling_studio import build_model_comparison_studio, build_model_fairness_review, build_predictive_model


def _sample_healthcare_frame() -> tuple[pd.DataFrame, dict[str, str]]:
    rows = 240
    data = pd.DataFrame(
        {
            'patient_id': [f'P{i:04d}' for i in range(rows)],
            'age': [30 + (i % 45) for i in range(rows)],
            'gender': ['F' if i % 2 == 0 else 'M' for i in range(rows)],
            'smoking_status': ['Current' if i % 5 == 0 else 'Never' for i in range(rows)],
            'bmi': [22 + (i % 18) for i in range(rows)],
            'hypertension': [1 if i % 4 == 0 else 0 for i in range(rows)],
            'heart_disease': [1 if i % 7 == 0 else 0 for i in range(rows)],
            'length_of_stay': [2 + (i % 7) for i in range(rows)],
        }
    )
    data['readmission'] = (
        (data['age'] >= 60).astype(int)
        | ((data['smoking_status'] == 'Current') & (data['length_of_stay'] >= 5)).astype(int)
        | (data['heart_disease'] == 1).astype(int)
    ).astype(int)
    data['age_band'] = pd.cut(data['age'], bins=[-1, 39, 59, 74, 200], labels=['18-39', '40-59', '60-74', '75+']).astype(str)
    canonical_map = {
        'patient_id': 'patient_id',
        'age': 'age',
        'gender': 'gender',
        'smoking_status': 'smoking_status',
        'bmi': 'bmi',
        'length_of_stay': 'length_of_stay',
        'readmission': 'readmission',
    }
    return data, canonical_map


class DecisionSupportTests(unittest.TestCase):
    def test_model_comparison_selects_best_model_and_handles_optional_xgboost(self) -> None:
        data, canonical_map = _sample_healthcare_frame()
        result = build_model_comparison_studio(
            data,
            canonical_map,
            target_column='readmission',
            feature_columns=['age', 'gender', 'smoking_status', 'bmi', 'hypertension', 'heart_disease', 'length_of_stay'],
        )
        if not result.get('available'):
            self.skipTest(result.get('message', 'Model comparison unavailable in this environment'))
        self.assertTrue(result.get('available'))
        comparison_table = result['model_comparison_table']
        self.assertGreaterEqual(len(comparison_table), 3)
        self.assertIn('best_model_name', result)
        self.assertIn(result['best_model_name'], comparison_table['model_name'].tolist())

    def test_fairness_dashboard_computes_group_metrics_and_limited_fallback(self) -> None:
        data, canonical_map = _sample_healthcare_frame()
        model_result = build_predictive_model(
            data,
            canonical_map,
            target_column='readmission',
            feature_columns=['age', 'gender', 'smoking_status', 'bmi', 'hypertension', 'heart_disease', 'length_of_stay'],
            model_type='Logistic Regression',
        )
        if not model_result.get('available'):
            self.skipTest(model_result.get('message', 'Predictive modeling unavailable in this environment'))
        fairness = build_model_fairness_review(model_result, data, canonical_map)
        self.assertTrue(fairness.get('available'))
        self.assertIn(fairness.get('review_level'), {'full', 'limited'})

        limited_map = canonical_map.copy()
        limited_map.pop('gender', None)
        limited_map.pop('age', None)
        limited = build_model_fairness_review({**model_result, 'prediction_table': model_result['prediction_table'].drop(columns=['gender', 'age_band', 'age'], errors='ignore')}, data, limited_map)
        self.assertTrue(limited.get('available'))
        self.assertEqual(limited.get('review_level'), 'limited')

    def test_intervention_recommendations_and_priority_buckets(self) -> None:
        healthcare = {
            'risk_segmentation': {'available': True, 'segment_table': pd.DataFrame([{'risk_segment': 'High Risk', 'patient_count': 40, 'percentage': 0.2}])},
            'readmission': {'available': True, 'overview': {'overall_readmission_rate': 0.14}, 'high_risk_segments': pd.DataFrame([{'segment_type': 'department', 'segment_value': 'Cardiology', 'readmission_rate': 0.2}]), 'source': 'synthetic'},
            'anomaly_detection': {'available': True, 'summary_table': pd.DataFrame([{'field': 'bmi', 'anomaly_count': 12}])},
        }
        quality = {'high_missing': pd.DataFrame([{'column_name': 'cost_amount', 'null_percentage': 0.22}])}
        readiness = {'synthetic_supported_modules': ['Cost Analysis']}
        remediation_context = {'bmi_remediation': {'available': True, 'total_bmi_outliers': 10, 'outlier_pct': 0.04}, 'synthetic_cost': {'available': True}}
        recs = build_intervention_recommendations(healthcare, quality, readiness, remediation_context)
        self.assertFalse(recs.empty)
        self.assertIn('priority_bucket', recs.columns)
        self.assertIn('Immediate', recs['priority_bucket'].tolist())

    def test_executive_summary_and_synthetic_disclosure(self) -> None:
        summary = build_executive_summary(
            'demo.csv',
            {'rows': 1000, 'columns': 10, 'analyzed_columns': 14, 'source_columns': 10, 'helper_columns_added': 4, 'duplicate_rows': 2, 'missing_values': 20},
            {'readiness_score': 0.7, 'available_count': 5, 'readiness_table': pd.DataFrame([{'status': 'Unavailable'}])},
            {'readmission': {'high_risk_segments': pd.DataFrame()}, 'risk_segmentation': {}, 'available': False},
            pd.DataFrame([{'recommendation_title': 'Fix BMI'}]),
            pd.DataFrame([{'recommendation_title': 'Target high-risk cohort', 'why_it_matters': 'Important'}]),
            {'helper_fields': pd.DataFrame([{'helper_field': 'estimated_cost'}])},
        )
        self.assertIn('Synthetic Support Disclosure', summary['executive_summary_sections'])
        self.assertIn('analyzed columns', summary['executive_summary_sections']['Dataset Overview'])
        report = build_executive_summary_text('demo.csv', {'rows': 1000, 'columns': 10, 'analyzed_columns': 14, 'source_columns': 10, 'duplicate_rows': 2, 'missing_values': 20}, {'risk_segmentation': {}, 'ai_insight_summary': [], 'scenario': {}, 'default_cohort_summary': {}}, {'recommendations': []})
        self.assertTrue(report)
        self.assertIn(b'Analyzed columns reviewed', report)

    def test_automated_insight_board_tolerates_group_dimension_fairness_flags(self) -> None:
        result = build_automated_insight_board(
            {'rows': 1000},
            {'readiness_score': 0.7, 'available_count': 4},
            {
                'risk_segmentation': {'available': False},
                'readmission': {'available': False},
                'survival_outcomes': {'available': False},
                'anomaly_detection': {'available': False},
                'explainability_fairness': {
                    'available': True,
                    'fairness_flags': pd.DataFrame([{'group_dimension': 'Gender', 'gap_size': 0.18, 'flag_type': 'Survival gap'}]),
                },
                'scenario': {'available': False},
                'cost': {'available': False},
            },
            {'summary_lines': []},
            pd.DataFrame(),
        )
        self.assertTrue(result.get('available'))
        self.assertTrue(result.get('benchmark_gaps'))

    def test_kpi_benchmarking_and_scenario_outputs(self) -> None:
        healthcare = {
            'risk_segmentation': {'available': True, 'segment_table': pd.DataFrame([{'risk_segment': 'High Risk', 'patient_count': 20, 'percentage': 0.1}, {'risk_segment': 'Low Risk', 'patient_count': 100, 'percentage': 0.5}])},
            'cost': {'available': True, 'summary': {'average_cost': 2400}},
            'readmission': {'available': True, 'overview': {'overall_readmission_rate': 0.12}, 'source': 'synthetic'},
        }
        quality = {'high_missing': pd.DataFrame([{'column_name': 'service_date'}])}
        remediation_context = {'bmi_remediation': {'available': True, 'outlier_pct': 0.03}}
        benchmark = build_kpi_benchmarking_layer(healthcare, quality, remediation_context)
        self.assertTrue(benchmark.get('available'))
        self.assertFalse(benchmark['benchmark_table'].empty)
        scenarios = build_scenario_simulation_studio(healthcare, quality, remediation_context)
        self.assertTrue(scenarios.get('available'))
        self.assertIn('limitation_note', scenarios['scenario_table'].columns)

    def test_prioritized_insights_ordering(self) -> None:
        result = build_prioritized_insights(
            {'rows': 1000},
            {'high_missing': pd.DataFrame([{'column_name': 'cost_amount', 'null_percentage': 0.4}])},
            {'synthetic_supported_modules': ['Cost Analysis']},
            {
                'anomaly_detection': {'available': True, 'summary_table': pd.DataFrame([{'field': 'bmi', 'anomaly_count': 20}])},
                'readmission': {'available': True, 'overview': {'overall_readmission_rate': 0.14}, 'source': 'native'},
                'risk_segmentation': {'available': True, 'segment_table': pd.DataFrame([{'risk_segment': 'High Risk', 'percentage': 0.2}])},
            },
            pd.DataFrame([{'recommendation_title': 'Fix completeness'}]),
            pd.DataFrame([{'recommendation_title': 'Prioritize outreach', 'why_it_matters': 'Large high-risk cohort'}]),
            None,
        )
        self.assertTrue(result.get('available'))
        self.assertFalse(result['critical_findings'].empty)
        self.assertGreaterEqual(result['critical_findings'].iloc[0]['priority_score'], result['watchlist_findings'].iloc[0]['priority_score'] if not result['watchlist_findings'].empty else 0)


if __name__ == '__main__':
    unittest.main()
