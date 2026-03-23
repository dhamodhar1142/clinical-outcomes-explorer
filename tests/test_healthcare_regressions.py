from __future__ import annotations

import unittest

import pandas as pd

from src.healthcare_analysis import (
    anomaly_detection,
    assess_healthcare_dataset,
    build_cohort_summary,
    care_pathway_view,
    clinical_outcome_benchmarks,
    length_of_stay_prediction,
    mortality_adverse_event_indicators,
    population_health_analytics,
    readmission_risk_analytics,
    run_healthcare_analysis,
    segment_discovery,
    survival_outcome_analysis,
)
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
    def test_assess_healthcare_dataset_recognizes_encounter_workflow_schema(self) -> None:
        canonical = {
            'patient_id': 'pat_id',
            'encounter_id': 'medt_id',
            'admission_date': 'vis_en',
            'discharge_date': 'vis_ex',
            'encounter_status_code': 'vstat_cd',
            'encounter_status': 'vstat_des',
            'encounter_type_code': 'vtype_cd',
            'encounter_type': 'vtype_des',
            'room_id': 'rom_id',
        }
        result = assess_healthcare_dataset(canonical)
        self.assertGreaterEqual(float(result.get('healthcare_readiness_score', 0.0)), 0.6)
        self.assertEqual(result.get('likely_dataset_type'), 'Encounter-oriented healthcare dataset')
        self.assertEqual(result.get('synthetic_supported_healthcare_fields'), [])

    def test_expanded_clinical_analytics_are_available_for_healthcare_dataset(self) -> None:
        data, canonical = _sample_clinical_frame()
        result = run_healthcare_analysis(data, canonical)
        self.assertTrue(result.get('length_of_stay_prediction', {}).get('available'))
        self.assertTrue(result.get('mortality_adverse_events', {}).get('available'))
        self.assertTrue(result.get('population_health', {}).get('available'))
        self.assertTrue(result.get('clinical_outcome_benchmarks', {}).get('available'))

    def test_readmission_handles_categorical_age_band(self) -> None:
        data, canonical = _sample_clinical_frame()
        result = readmission_risk_analytics(data, canonical)
        self.assertTrue(result.get('available'))
        self.assertIn('by_age_band', result)
        self.assertFalse(result['by_age_band'].empty)
        self.assertEqual(result['readiness']['support_level'], 'Full')
        self.assertIn('support_table', result['readiness'])
        self.assertFalse(result['readiness']['support_table'].empty)
        self.assertIn('solution_story', result)
        self.assertIn('top_high_risk_summary', result)
        self.assertIn('intervention_recommendations', result)
        self.assertFalse(result['intervention_recommendations'].empty)
        self.assertEqual(result['readiness']['workflow_title'], 'Hospital Readmission Risk Analytics')

    def test_readmission_builds_trend_when_event_dates_exist(self) -> None:
        data, canonical = _sample_clinical_frame()
        data = data.copy()
        data['admission_date'] = pd.date_range('2024-01-01', periods=len(data), freq='7D')
        canonical = {**canonical, 'admission_date': 'admission_date'}
        result = readmission_risk_analytics(data, canonical)
        self.assertTrue(result.get('available'))
        self.assertIn('trend', result)
        self.assertFalse(result['trend'].empty)
        self.assertIn('readmission_rate', result['trend'].columns)
        self.assertIn('confidence_lower', result['trend'].columns)
        self.assertIn('stability_band', result['trend'].columns)

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
        self.assertIn('pathway_summary_table', result)
        self.assertIn('average_duration_by_pathway', result)
        self.assertIn('action_recommendations', result)
        self.assertFalse(result.get('pathway_summary_table', pd.DataFrame()).empty)
        self.assertFalse(result.get('average_duration_by_pathway', pd.DataFrame()).empty)
        self.assertTrue(len(result.get('action_recommendations', [])) >= 1)
        self.assertTrue(bool(result.get('summary')))

    def test_survival_outcome_analysis_returns_stronger_interpretation(self) -> None:
        data, canonical = _sample_clinical_frame()
        result = survival_outcome_analysis(data, canonical)
        self.assertTrue(result.get('available'))
        self.assertEqual(result.get('support_level'), 'Full')
        self.assertIn('overall_survival_rate', result)
        self.assertIsNotNone(result.get('overall_survival_rate'))
        self.assertTrue(bool(result.get('interpretation')))
        self.assertFalse(result.get('stage_table', pd.DataFrame()).empty)
        self.assertFalse(result.get('treatment_table', pd.DataFrame()).empty)
        self.assertIn('duration_by_stage', result)
        self.assertIn('duration_by_treatment', result)
        self.assertIn('duration_by_cohort', result)
        self.assertFalse(result.get('duration_by_stage', pd.DataFrame()).empty)
        self.assertFalse(result.get('duration_by_treatment', pd.DataFrame()).empty)

    def test_survival_outcome_analysis_handles_partial_support(self) -> None:
        partial = pd.DataFrame(
            {
                'patient_id': ['P1', 'P2', 'P3', 'P4'],
                'survived': ['Yes', 'No', 'Yes', 'Yes'],
                'department': ['Oncology', 'Oncology', 'Medicine', 'Medicine'],
            }
        )
        canonical = {
            'patient_id': 'patient_id',
            'survived': 'survived',
            'department': 'department',
        }
        result = survival_outcome_analysis(partial, canonical)
        self.assertTrue(result.get('available'))
        self.assertEqual(result.get('support_level'), 'Early')
        self.assertIn('what_unlocks_next', result)
        self.assertIn('overall survival', result.get('summary', '').lower())

    def test_survival_duration_intelligence_exposes_distribution_and_trend(self) -> None:
        data, canonical = _sample_clinical_frame()
        result = survival_outcome_analysis(data, canonical)
        self.assertTrue(result.get('available'))
        self.assertIsNotNone(result.get('duration_summary'))
        self.assertFalse(result.get('duration_distribution', pd.DataFrame()).empty)
        self.assertFalse(result.get('treatment_duration_trend', pd.DataFrame()).empty)

    def test_length_of_stay_prediction_returns_prioritized_rows(self) -> None:
        data, canonical = _sample_clinical_frame()
        result = length_of_stay_prediction(data, canonical)
        self.assertTrue(result.get('available'))
        self.assertIn('high_los_rows', result)
        self.assertFalse(result.get('high_los_rows', pd.DataFrame()).empty)
        self.assertIn('predicted_length_of_stay', result['high_los_rows'].columns)

    def test_mortality_adverse_event_indicators_return_grouped_summary(self) -> None:
        data, canonical = _sample_clinical_frame()
        result = mortality_adverse_event_indicators(data, canonical)
        self.assertTrue(result.get('available'))
        self.assertFalse(result.get('indicator_table', pd.DataFrame()).empty)
        self.assertFalse(result.get('group_table', pd.DataFrame()).empty)
        self.assertFalse(result.get('flagged_rows', pd.DataFrame()).empty)

    def test_population_health_analytics_returns_prevalence_rollups(self) -> None:
        data, canonical = _sample_clinical_frame()
        result = population_health_analytics(data, canonical)
        self.assertTrue(result.get('available'))
        self.assertFalse(result.get('prevalence_table', pd.DataFrame()).empty)
        self.assertTrue(len(result.get('summary_cards', [])) >= 1)

    def test_anomaly_detection_builds_classification_and_review_outputs(self) -> None:
        data, canonical = _sample_clinical_frame()
        data = data.copy()
        data.loc[0, 'bmi'] = 999.0
        data.loc[1, 'diagnosis_date'] = pd.Timestamp('2024-03-01')
        data.loc[1, 'end_treatment_date'] = pd.Timestamp('2024-02-01')
        data.loc[2, 'smoking_status'] = 'Occasional'
        canonical = {**canonical, 'admission_date': 'diagnosis_date', 'discharge_date': 'end_treatment_date'}
        result = anomaly_detection(data, canonical)
        self.assertTrue(result.get('available'))
        self.assertIn('classification_table', result)
        self.assertIn('remediation_action_log', result)
        self.assertIn('review_export', result)
        self.assertIn('validation_report', result)
        self.assertFalse(result.get('classification_table', pd.DataFrame()).empty)
        self.assertFalse(result.get('review_export', pd.DataFrame()).empty)
        self.assertIn('anomaly_resolution', result['review_export'].columns)
        self.assertIn('Original anomaly count', set(result['validation_report']['metric'].astype(str)))
        self.assertLessEqual(len(result.get('detail_table', pd.DataFrame())), 250)
        self.assertLessEqual(len(result.get('review_export', pd.DataFrame())), 120)

    def test_clinical_outcome_benchmarks_return_comparison_views(self) -> None:
        data, canonical = _sample_clinical_frame()
        result = clinical_outcome_benchmarks(data, canonical)
        self.assertTrue(result.get('available'))
        self.assertFalse(result.get('summary_table', pd.DataFrame()).empty)
        self.assertTrue(len(result.get('summary_cards', [])) >= 1)

    def test_cohort_summary_returns_comparison_and_trend_guidance(self) -> None:
        data, canonical = _sample_clinical_frame()
        result = build_cohort_summary(data, canonical, diagnoses=['C34'])
        self.assertTrue(result.get('available'))
        self.assertIn('comparison_table', result)
        self.assertIn('demographic_breakdown_table', result)
        self.assertIn('outcome_difference_table', result)
        self.assertIn('cohort_definition', result)
        self.assertIn('summary_narrative', result)
        self.assertIn('trend_summary', result)
        self.assertIn('suggested_next_steps', result)
        self.assertFalse(result.get('comparison_table', pd.DataFrame()).empty)
        self.assertFalse(result.get('demographic_breakdown_table', pd.DataFrame()).empty)
        self.assertFalse(result.get('outcome_difference_table', pd.DataFrame()).empty)
        self.assertIn('selected_confidence_lower', result['comparison_table'].columns)
        self.assertIn('selected_numeric_confidence_lower', result['comparison_table'].columns)
        self.assertIn('delta_p_value', result['comparison_table'].columns)
        self.assertIn('delta_significance', result['comparison_table'].columns)
        self.assertIn('statistical_test', result['comparison_table'].columns)
        self.assertIn('Approximate Welch test', set(result['comparison_table']['statistical_test'].astype(str)))
        if not result.get('cohort_trend_table', pd.DataFrame()).empty and 'survival_rate' in result['cohort_trend_table'].columns:
            self.assertIn('survival_confidence_lower', result['cohort_trend_table'].columns)
        self.assertFalse(result.get('cohort_definition', {}).get('filter_table', pd.DataFrame()).empty)
        self.assertTrue(bool(result.get('summary_narrative')))
        self.assertTrue(len(result.get('trend_summary', [])) >= 1)

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

    def test_readmission_readiness_falls_back_cleanly_for_weak_dataset(self) -> None:
        weak = pd.DataFrame(
            {
                'department': ['Oncology', 'Medicine', 'Oncology'],
                'age': [72, 51, 67],
                'length_of_stay': [6, 3, 8],
            }
        )
        canonical = {
            'department': 'department',
            'age': 'age',
            'length_of_stay': 'length_of_stay',
        }
        result = readmission_risk_analytics(weak, canonical)
        self.assertFalse(result.get('available'))
        self.assertIn('readiness', result)
        self.assertEqual(result['readiness']['support_level'], 'Unavailable')
        self.assertIn('patient_id', result['readiness']['additional_fields_to_unlock_full_analysis'])
        self.assertIn('event_date', result['readiness']['additional_fields_to_unlock_full_analysis'])
        self.assertTrue(result['readiness']['guidance_notes'])
        self.assertIn('what_still_works', result['readiness'])
        self.assertIn('solution_story', result)
        self.assertIn('next_best_action', result)

    def test_expanded_clinical_analytics_fall_back_cleanly_for_generic_dataset(self) -> None:
        generic = pd.DataFrame(
            {
                'customer_id': ['C1', 'C2', 'C3'],
                'region': ['North', 'South', 'North'],
                'revenue': [100.0, 200.0, 150.0],
                'event_date': pd.date_range('2025-01-01', periods=3, freq='D'),
            }
        )
        canonical = {
            'entity_id': 'customer_id',
            'event_date': 'event_date',
        }
        result = run_healthcare_analysis(generic, canonical)
        self.assertIn('length_of_stay_prediction', result)
        self.assertFalse(result['length_of_stay_prediction'].get('available'))
        self.assertIn('mortality_adverse_events', result)
        self.assertFalse(result['mortality_adverse_events'].get('available'))


if __name__ == '__main__':
    unittest.main()
