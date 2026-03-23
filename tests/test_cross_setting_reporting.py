import unittest

import pandas as pd

from src.export_utils import (
    build_audience_report_text,
    build_cross_setting_reporting_profile,
    build_generated_report_text,
    build_report_support_tables,
    normalize_report_mode,
)


class CrossSettingReportingTests(unittest.TestCase):
    def setUp(self):
        class StructureStub:
            numeric_columns = ['age', 'bmi']
            date_columns = ['event_date']
            categorical_columns = ['gender']

        self.structure = StructureStub()
        self.overview = {'rows': 1000, 'columns': 12, 'duplicate_rows': 5, 'missing_values': 20}
        self.quality = {
            'high_missing': pd.DataFrame([{'column_name': 'payer', 'null_percentage': 0.32}]),
            'mixed_type_suspects': pd.DataFrame(),
        }
        self.semantic = {'canonical_map': {'age': 'age', 'gender': 'gender'}}
        self.readiness = {
            'readiness_score': 0.58,
            'readiness_table': pd.DataFrame([
                {'analysis_module': 'Trend Analysis', 'status': 'Available'},
                {'analysis_module': 'Readmission Analytics', 'status': 'Partial'},
            ]),
        }
        self.healthcare = {
            'healthcare_readiness_score': 0.61,
            'risk_segmentation': {
                'available': True,
                'segment_table': pd.DataFrame([{'risk_segment': 'High Risk', 'patient_count': 120, 'percentage': 0.12}]),
            },
            'anomaly_detection': {'available': False, 'summary_table': pd.DataFrame()},
            'explainability_fairness': {
                'available': True,
                'fairness_flags': pd.DataFrame([{'group': 'Older Adults', 'gap': 0.12}]),
            },
            'ai_insight_summary': ['Smoking prevalence is concentrated in higher-risk segments.'],
        }
        self.insights = {'summary_lines': ['Risk is concentrated in a smaller high-risk cohort.'], 'recommendations': ['Validate payer completeness.']}
        self.actions = pd.DataFrame([{'priority': 'Immediate', 'recommendation_title': 'Validate payer completeness', 'rationale': 'This blocks stronger reporting.'}])

    def test_normalizes_new_report_alias(self):
        self.assertEqual(normalize_report_mode('Population Health Review'), 'Population Health Summary')
        self.assertEqual(normalize_report_mode('Data Readiness Summary'), 'Data Readiness Review')

    def test_builds_data_readiness_support_tables(self):
        tables = build_report_support_tables('Data Readiness Review', self.overview, self.quality, self.readiness, self.healthcare, self.actions)
        self.assertIn('Analysis Readiness', tables)
        self.assertIn('High Missingness Fields', tables)
        self.assertIn('Recommended Actions', tables)

    def test_builds_population_health_report_text(self):
        report = build_audience_report_text(
            'Population Health Summary',
            'demo.csv',
            self.overview,
            self.structure,
            self.quality,
            self.semantic,
            self.readiness,
            self.healthcare,
            self.insights,
            self.actions,
        ).decode('utf-8')
        self.assertIn('Population Health Focus', report)
        self.assertIn('Suggested Next Review Step', report)

    def test_cross_setting_reporting_profile_reflects_workflow_fit(self):
        profile = build_cross_setting_reporting_profile(
            {
                'analytics_capability_matrix': pd.DataFrame([
                    {'analytics_module': 'Provider / Facility Volume', 'status': 'enabled', 'support': 'native', 'rationale': 'available'},
                    {'analytics_module': 'Trend Analysis', 'status': 'partial', 'support': 'synthetic-assisted', 'rationale': 'Synthetic event_date is in use.'},
                    {'analytics_module': 'Readmission Analytics', 'status': 'partial', 'support': 'synthetic-assisted', 'rationale': 'Synthetic readmission flag is in use.'},
                    {'analytics_module': 'Risk Segmentation', 'status': 'enabled', 'support': 'native', 'rationale': 'available'},
                    {'analytics_module': 'Cohort Analysis', 'status': 'enabled', 'support': 'native', 'rationale': 'available'},
                    {'analytics_module': 'Care Pathway Intelligence', 'status': 'blocked', 'support': 'unavailable', 'rationale': 'Missing pathway timestamps.'},
                ])
            },
            {'recommended_workflow': 'Hospital Operations Analytics'},
            {'recommended_package': 'Hospital Operations Review'},
            'Operational Report',
        )
        self.assertFalse(profile.empty)
        op_row = profile[profile['review_type'] == 'Operational review'].iloc[0]
        self.assertEqual(op_row['workflow_alignment'], 'Recommended now')
        self.assertEqual(op_row['selected_report'], 'Current selection')
        self.assertIn(op_row['status'], {'Ready now', 'Partially supported'})

    def test_cross_setting_reporting_profile_falls_back_cleanly(self):
        profile = build_cross_setting_reporting_profile({}, {}, {}, 'Executive Summary')
        self.assertFalse(profile.empty)
        self.assertTrue((profile['status'] == 'Needs stronger source support').any())

    def test_generated_report_text_includes_required_sections(self):
        self.healthcare['default_cohort_summary'] = {
            'available': True,
            'summary': {
                'cohort_size': 125,
                'survival_rate': 0.82,
                'average_treatment_duration_days': 46.0,
            },
        }
        self.healthcare['utilization'] = {
            'monthly_utilization': pd.DataFrame([{'month': '2025-01-01', 'event_count': 220}]),
        }
        report = build_generated_report_text(
            'Executive Report',
            'demo.csv',
            self.overview,
            self.quality,
            self.readiness,
            self.healthcare,
            self.insights,
            self.actions,
        ).decode('utf-8')
        self.assertIn('Dataset Summary', report)
        self.assertIn('Quality Metrics', report)
        self.assertIn('Readiness Score', report)
        self.assertIn('Key Insights', report)
        self.assertIn('Cohort Findings', report)
        self.assertIn('Trend Summaries', report)
        self.assertIn('Recommended Actions', report)


if __name__ == '__main__':
    unittest.main()
