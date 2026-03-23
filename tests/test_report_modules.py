from __future__ import annotations

import unittest

import pandas as pd

from src.export_utils import build_generated_report_text
from src.reports.analyst_reports import build_generated_report_text as build_generated_report_text_module
from src.reports.bundles import build_role_export_bundle_manifest
from src.reports.governance_reports import build_policy_note_text


class ReportModuleSplitTests(unittest.TestCase):
    def test_export_utils_facade_matches_report_module_output(self):
        overview = {'rows': 25, 'columns': 4, 'duplicate_rows': 1, 'missing_values': 2}
        quality = {'quality_score': 91.5, 'high_missing': pd.DataFrame()}
        readiness = {
            'readiness_score': 0.75,
            'readiness_table': pd.DataFrame([
                {'analysis_module': 'Risk Segmentation', 'status': 'Available'},
                {'analysis_module': 'Readmission Analytics', 'status': 'Partial'},
            ]),
        }
        healthcare = {'default_cohort_summary': {'available': False}, 'readmission': {'available': False}}
        insights = {'summary_lines': ['Quality and readiness are strong.']}
        actions = pd.DataFrame([{'recommendation_title': 'Review blockers'}])

        facade = build_generated_report_text('Demo Report', 'demo.csv', overview, quality, readiness, healthcare, insights, actions)
        module = build_generated_report_text_module('Demo Report', 'demo.csv', overview, quality, readiness, healthcare, insights, actions)

        self.assertEqual(facade, module)

    def test_generated_report_can_include_accuracy_guardrails(self):
        overview = {'rows': 25, 'columns': 4, 'duplicate_rows': 1, 'missing_values': 2}
        quality = {'quality_score': 91.5, 'high_missing': pd.DataFrame()}
        readiness = {'readiness_score': 0.75, 'readiness_table': pd.DataFrame()}
        healthcare = {'default_cohort_summary': {'available': False}, 'readmission': {'available': False}}
        insights = {'summary_lines': ['Quality and readiness are strong.']}
        actions = pd.DataFrame([{'recommendation_title': 'Review blockers'}])
        report = build_generated_report_text(
            'Demo Report',
            'demo.csv',
            overview,
            quality,
            readiness,
            healthcare,
            insights,
            actions,
            {'trust_level': 'Moderate', 'interpretation_mode': 'Directional'},
            {
                'benchmark_profile': {'profile_name': 'Hospital Encounters'},
                'reporting_policy': {'profile_name': 'Conservative'},
                'uncertainty_narrative': {'headline': 'Directional signal: sampled operational view only.'},
            },
        ).decode('utf-8')
        self.assertIn('Accuracy and Reporting Guardrails', report)
        self.assertIn('Hospital Encounters', report)
        self.assertIn('Directional signal', report)

    def test_bundle_and_governance_modules_still_build_outputs(self):
        manifest = build_role_export_bundle_manifest('Analyst', 'Internal Review', True, 'Analyst Report', {'sensitive_fields': pd.DataFrame()})
        self.assertFalse(manifest.empty)
        note = build_policy_note_text('Internal Review', 'Analyst', {'hipaa': {}, 'sensitive_fields': pd.DataFrame()}).decode('utf-8')
        self.assertIn('Privacy & Sharing Note', note)


if __name__ == '__main__':
    unittest.main()
