from __future__ import annotations

import unittest

import pandas as pd

from src.export_utils import build_generated_report_text
from src.reports.analyst_reports import build_generated_report_text as build_generated_report_text_module
from src.reports.bundles import build_role_export_bundle_manifest
from src.reports.claims_reports import (
    build_claims_export_csv_bundle,
    build_claims_export_tables,
    build_claims_validation_report_markdown,
)
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

    def test_claims_report_module_builds_markdown_and_csv_outputs(self):
        overview = {'rows': 24, 'columns': 17, 'analyzed_columns': 17}
        claims = {
            'available': True,
            'readiness_label': 'Review needed',
            'summary_cards': [
                {'label': 'Claims in scope', 'value': '24'},
                {'label': 'Members in scope', 'value': '23'},
            ],
            'narrative': 'The claims engine reviewed 24 rows and found duplicate claims plus financial mismatches.',
            'validation_table': pd.DataFrame([
                {'check': 'Duplicate claim rows', 'failed_rows': 2, 'failure_rate': 2 / 24, 'severity': 'High', 'guidance': 'Review duplicates.'},
                {'check': 'Paid exceeds allowed', 'failed_rows': 1, 'failure_rate': 1 / 24, 'severity': 'High', 'guidance': 'Review mismatches.'},
            ]),
            'financial_summary': pd.DataFrame([
                {'metric': 'Total paid amount', 'value': 12345.0},
                {'metric': 'Claims per member', 'value': 1.04},
            ]),
            'payer_utilization': pd.DataFrame([{'payer': 'Aetna', 'claim_rows': 10, 'total_paid_amount': 5000.0}]),
            'provider_utilization': pd.DataFrame([{'provider_name': 'North Clinic', 'claim_rows': 6}]),
            'diagnosis_utilization': pd.DataFrame([{'diagnosis_code': 'I10', 'claim_rows': 4}]),
            'monthly_utilization': pd.DataFrame([{'service_month': '2025-01-01', 'claim_rows': 24, 'total_paid_amount': 12345.0}]),
            'flagged_rows': pd.DataFrame([{'claim_id': 'CLM0007', 'claim_validation_flags': 'duplicate_claim_id'}]),
        }

        tables = build_claims_export_tables('claims_demo.csv', overview, claims)
        self.assertFalse(tables['qc_summary'].empty)
        self.assertFalse(tables['claims_validation_issue_log'].empty)
        self.assertFalse(tables['utilization_metrics'].empty)

        markdown = build_claims_validation_report_markdown('claims_demo.csv', overview, claims).decode('utf-8')
        self.assertIn('Claims Validation & Utilization Engine', markdown)
        self.assertIn('Duplicate claim rows', markdown)
        self.assertIn('Plain-English Interpretation', markdown)

        bundle = build_claims_export_csv_bundle('claims_demo.csv', overview, claims)
        self.assertIn('qc_summary.csv', bundle)
        self.assertIn('claims_validation_issue_log.csv', bundle)
        self.assertIn('utilization_metrics.csv', bundle)
        self.assertIn('claims_validation_report.md', bundle)


if __name__ == '__main__':
    unittest.main()
