from __future__ import annotations

import unittest

import pandas as pd

from src.presentation_support import (
    build_audit_summary,
    build_compliance_governance_summary,
    build_executive_report_pack,
    build_landing_summary,
    build_printable_reports,
    build_run_history_entry,
    build_stakeholder_export_bundle,
    update_run_history,
)


class PresentationSupportTests(unittest.TestCase):
    def _sample_pipeline_bits(self):
        overview = {'rows': 1000, 'columns': 20, 'duplicate_rows': 10, 'missing_values': 55, 'memory_mb': 12.5}
        quality = {'quality_score': 82.0, 'high_missing': pd.DataFrame([{'column_name': 'cost_amount'}])}
        readiness = {
            'readiness_score': 0.68,
            'available_count': 6,
            'readiness_table': pd.DataFrame([
                {'analysis_module': 'Trend Analysis', 'status': 'Available'},
                {'analysis_module': 'Cost Analysis', 'status': 'Unavailable'},
            ]),
        }
        healthcare = {
            'healthcare_readiness_score': 0.71,
            'likely_dataset_type': 'Partially healthcare-related dataset',
            'explainability_fairness': {'available': True, 'comparison_table': pd.DataFrame([{'group_dimension': 'Gender'}])},
            'readmission': {'available': True, 'high_risk_segments': pd.DataFrame([{'segment_type': 'department'}])},
        }
        executive_summary = {'stakeholder_summary_bullets': ['Rows reviewed.', 'Readiness is moderate.', 'Next steps are available.']}
        action_recs = pd.DataFrame([{'recommendation_title': 'Fix completeness'}])
        intervention_recs = pd.DataFrame([{'recommendation_title': 'Target high-risk cohort'}])
        kpi = {'available': True, 'kpi_cards': [{'label': 'High-Risk Share', 'value': '12.0%'}], 'benchmark_table': pd.DataFrame([{'metric': 'High-Risk Share'}])}
        scenarios = {'available': True, 'summary': '3 directional scenarios are available.', 'scenario_table': pd.DataFrame([{'scenario_name': 'Reduce smoking'}])}
        prioritized = {'available': True}
        remediation = {
            'helper_fields': pd.DataFrame([{'helper_field': 'estimated_cost'}]),
            'synthetic_cost': {'available': True},
            'synthetic_readmission': {'available': True},
            'synthetic_clinical': {'available': True},
            'synthetic_field_count': 3,
            'bmi_remediation': {'available': True, 'total_bmi_outliers': 25, 'remediation_mode': 'median'},
            'native_field_count': 12,
        }
        demo_config = {
            'synthetic_helper_mode': 'Auto',
            'bmi_remediation_mode': 'median',
            'executive_summary_verbosity': 'Detailed',
            'scenario_simulation_mode': 'Expanded',
        }
        dataset_intelligence = {
            'dataset_type_label': 'Clinical / patient-level dataset',
            'enabled_analytics': ['Data Profiling', 'Risk Segmentation', 'Export / Executive Reporting'],
            'blocked_analytics': ['Cost Driver Analysis'],
            'next_best_actions': ['Add a native cost field to unlock stronger financial analytics.'],
            'analytics_capability_matrix': pd.DataFrame([{'analytics_module': 'Data Profiling', 'status': 'enabled', 'support': 'native'}]),
        }
        return overview, quality, readiness, healthcare, dataset_intelligence, executive_summary, action_recs, intervention_recs, kpi, scenarios, prioritized, remediation, demo_config

    def test_executive_report_pack_generation(self):
        overview, quality, readiness, healthcare, dataset_intelligence, executive_summary, action_recs, intervention_recs, kpi, scenarios, prioritized, remediation, demo_config = self._sample_pipeline_bits()
        report = build_executive_report_pack(
            'demo.csv',
            overview,
            quality,
            readiness,
            healthcare,
            dataset_intelligence,
            executive_summary,
            action_recs,
            intervention_recs,
            kpi,
            scenarios,
            prioritized,
            remediation,
            demo_config,
        )
        self.assertIn('executive_report_markdown', report)
        self.assertIn('Synthetic Support Disclosures', report['executive_report_sections'])
        self.assertIn('Dataset Intelligence Summary', report['executive_report_sections'])

    def test_printable_report_outputs(self):
        overview, quality, readiness, healthcare, dataset_intelligence, executive_summary, action_recs, intervention_recs, kpi, scenarios, prioritized, remediation, demo_config = self._sample_pipeline_bits()
        report = build_executive_report_pack('demo.csv', overview, quality, readiness, healthcare, dataset_intelligence, executive_summary, action_recs, intervention_recs, kpi, scenarios, prioritized, remediation, demo_config)
        compliance = build_compliance_governance_summary(
            {'combined_readiness_score': 55.0, 'badge_text': 'Moderate'},
            {'hipaa': {'risk_level': 'Moderate', 'direct_identifier_count': 2}},
            {'transformation_steps': ['synthetic event_date', 'BMI remediation']},
            remediation,
            readiness,
        )
        printable = build_printable_reports(report, compliance)
        self.assertIn('printable_executive_report', printable)
        self.assertIn('printable_compliance_summary', printable)

    def test_stakeholder_export_bundle_manifest(self):
        overview, quality, readiness, healthcare, dataset_intelligence, executive_summary, action_recs, intervention_recs, kpi, scenarios, prioritized, remediation, demo_config = self._sample_pipeline_bits()
        report = build_executive_report_pack('demo.csv', overview, quality, readiness, healthcare, dataset_intelligence, executive_summary, action_recs, intervention_recs, kpi, scenarios, prioritized, remediation, demo_config)
        compliance = build_compliance_governance_summary(
            {'combined_readiness_score': 55.0, 'badge_text': 'Moderate'},
            {'hipaa': {'risk_level': 'Moderate', 'direct_identifier_count': 2}},
            {'transformation_steps': ['synthetic event_date']},
            remediation,
            readiness,
        )
        bundle = build_stakeholder_export_bundle(report, dataset_intelligence, kpi, intervention_recs, healthcare['explainability_fairness'], healthcare['readmission'], quality, compliance)
        self.assertFalse(bundle['export_bundle_manifest'].empty)
        self.assertIn('Dataset Intelligence Summary', bundle['export_bundle_manifest']['bundle_item'].tolist())

    def test_run_history_and_audit_summary(self):
        overview, quality, readiness, healthcare, dataset_intelligence, executive_summary, action_recs, intervention_recs, kpi, scenarios, prioritized, remediation, demo_config = self._sample_pipeline_bits()
        pipeline = {'overview': overview, 'remediation_context': remediation}
        entry = build_run_history_entry('demo.csv', pipeline, demo_config, {'best_model_name': 'Random Forest', 'fairness_review_mode': 'full', 'model_comparison_table': pd.DataFrame([{'model_name': 'RF'}])})
        history = update_run_history([], entry)
        summary = build_audit_summary(history, [{'event_type': 'Dataset Selected'}])
        self.assertEqual(len(history), 1)
        self.assertFalse(summary['audit_summary'].empty)

    def test_compliance_summary_and_landing_stability(self):
        overview, quality, readiness, healthcare, dataset_intelligence, executive_summary, action_recs, intervention_recs, kpi, scenarios, prioritized, remediation, demo_config = self._sample_pipeline_bits()
        compliance = build_compliance_governance_summary(
            {'combined_readiness_score': 55.0, 'badge_text': 'Moderate'},
            {'hipaa': {'risk_level': 'Moderate', 'direct_identifier_count': 2}},
            {'transformation_steps': ['synthetic event_date']},
            remediation,
            readiness,
        )
        self.assertFalse(compliance['summary_table'].empty)
        pipeline = {
            'readiness': readiness,
            'healthcare': healthcare,
            'standards': {'badge_text': 'Moderate'},
            'remediation_context': remediation,
        }
        landing = build_landing_summary(pipeline, demo_config, 'demo.csv')
        self.assertIn('headline', landing)
        self.assertTrue(landing['capability_badges'])


if __name__ == '__main__':
    unittest.main()
