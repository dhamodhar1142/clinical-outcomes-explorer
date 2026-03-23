from __future__ import annotations

import unittest

import pandas as pd

from src.export_utils import build_shared_report_bundles, build_shared_report_bundle_text


class SharedReportBundleTests(unittest.TestCase):
    def test_bundle_manifest_contains_expected_bundle_names(self) -> None:
        dataset_intelligence = {
            'analytics_capability_matrix': pd.DataFrame(
                [
                    {'analytics_module': 'Data Profiling', 'status': 'enabled', 'support': 'native', 'rationale': 'Available'},
                    {'analytics_module': 'Data Quality Review', 'status': 'enabled', 'support': 'native', 'rationale': 'Available'},
                    {'analytics_module': 'Predictive Modeling', 'status': 'partial', 'support': 'synthetic-assisted', 'rationale': 'Needs stronger native outcomes'},
                    {'analytics_module': 'Standards / Governance Review', 'status': 'enabled', 'support': 'native', 'rationale': 'Available'},
                    {'analytics_module': 'Readmission Analytics', 'status': 'partial', 'support': 'synthetic-assisted', 'rationale': 'Synthetic readmission support is active'},
                    {'analytics_module': 'Care Pathway Intelligence', 'status': 'enabled', 'support': 'native', 'rationale': 'Available'},
                    {'analytics_module': 'Cohort Analysis', 'status': 'enabled', 'support': 'native', 'rationale': 'Available'},
                    {'analytics_module': 'Trend Analysis', 'status': 'enabled', 'support': 'native', 'rationale': 'Available'},
                    {'analytics_module': 'Provider / Facility Volume', 'status': 'blocked', 'support': 'unavailable', 'rationale': 'Missing provider identifiers'},
                    {'analytics_module': 'Export / Executive Reporting', 'status': 'enabled', 'support': 'native', 'rationale': 'Available'},
                    {'analytics_module': 'Decision Support', 'status': 'enabled', 'support': 'native', 'rationale': 'Available'},
                ]
            )
        }
        use_case = {'recommended_workflow': 'Clinical Outcome Analytics'}
        solution_packages = {
            'package_details': {
                'Healthcare Data Readiness': {'recommended_steps': ['Profile fields', 'Review quality', 'Fix blockers']},
                'Hospital Operations Review': {'recommended_steps': ['Check alerts', 'Review trends', 'Export ops report']},
                'Clinical Outcomes Review': {'recommended_steps': ['Review cohorts', 'Check outcomes', 'Export clinical report']},
            }
        }
        bundles = build_shared_report_bundles(dataset_intelligence, use_case, solution_packages, {}, {})
        manifest = bundles['bundle_manifest']
        self.assertFalse(manifest.empty)
        self.assertIn('Executive Bundle', manifest['bundle_name'].tolist())
        self.assertIn('Clinical Bundle', manifest['bundle_name'].tolist())
        self.assertIn('Operations Bundle', manifest['bundle_name'].tolist())

    def test_bundle_text_builds_without_error(self) -> None:
        manifest = pd.DataFrame(
            [
                {
                    'bundle_name': 'Executive Bundle',
                    'status': 'Ready now',
                    'primary_report_mode': 'Executive Summary',
                    'best_for': 'Leadership review',
                    'included_outputs': 'Executive Summary; KPI Summary',
                    'current_fit': 'Recommended now',
                    'support_note': 'Supported natively',
                }
            ]
        )
        output = build_shared_report_bundle_text(manifest).decode('utf-8')
        self.assertIn('Shared Report Bundles', output)
        self.assertIn('Executive Bundle', output)


if __name__ == '__main__':
    unittest.main()
