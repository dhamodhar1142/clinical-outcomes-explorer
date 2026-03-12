from __future__ import annotations

import unittest

import pandas as pd

from src.portfolio_support import (
    build_app_metadata,
    build_dataset_onboarding_summary,
    build_demo_mode_content,
    build_documentation_support,
    build_screenshot_support,
)


class PortfolioSupportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pipeline = {
            'readiness': {
                'readiness_score': 0.72,
                'available_count': 6,
                'readiness_table': pd.DataFrame(
                    [
                        {'analysis_module': 'Trend Analysis', 'status': 'Available', 'missing_prerequisites': ''},
                        {'analysis_module': 'Readmission Analytics', 'status': 'Unavailable', 'missing_prerequisites': 'patient_id, admission_date'},
                    ]
                ),
            },
            'healthcare': {'healthcare_readiness_score': 0.66},
            'remediation_context': {
                'synthetic_field_count': 3,
                'helper_fields': pd.DataFrame(
                    [
                        {'helper_field': 'event_date', 'helper_type': 'synthetic'},
                        {'helper_field': 'estimated_cost', 'helper_type': 'synthetic'},
                    ]
                ),
            },
            'remediation': pd.DataFrame(
                [
                    {
                        'issue': 'Missing admission date',
                        'recommended_fix': 'Add or map admission_date',
                        'modules_unlocked': 'Readmission Analytics; Trend Analysis',
                    }
                ]
            ),
        }
        self.source_meta = {
            'source_mode': 'Demo dataset',
            'best_for': 'Healthcare analytics walkthroughs and governance demos.',
        }
        self.demo_config = {
            'scenario_simulation_mode': 'Basic',
        }

    def test_demo_mode_content_generation(self) -> None:
        result = build_demo_mode_content('Demo Dataset', self.source_meta, self.pipeline, self.demo_config)
        self.assertIn('recommended_flow', result)
        self.assertGreaterEqual(len(result['recommended_flow']), 3)
        self.assertIn('synthetic', result['synthetic_support_note'].lower())

    def test_dataset_onboarding_guidance(self) -> None:
        result = build_dataset_onboarding_summary('Demo Dataset', self.source_meta, self.pipeline)
        self.assertIn('dataset_onboarding_summary', result)
        self.assertFalse(result['module_unlock_guide'].empty)
        self.assertFalse(result['data_upgrade_suggestions'].empty)

    def test_readme_documentation_content(self) -> None:
        result = build_documentation_support('Demo Dataset', self.pipeline)
        self.assertIn('readme_sections', result)
        self.assertIn('portfolio_project_summary', result)
        self.assertIn('demo_walkthrough_text', result)

    def test_screenshot_callout_plan_generation(self) -> None:
        result = build_screenshot_support('Demo Dataset', self.pipeline)
        self.assertFalse(result['screenshot_plan'].empty)
        self.assertGreaterEqual(len(result['recruiter_callouts']), 2)

    def test_about_metadata_stability(self) -> None:
        result = build_app_metadata(self.pipeline)
        self.assertEqual(result['product_name'], 'Smart Dataset Analyzer')
        self.assertIn('version', result)
        self.assertIn('synthetic', result['synthetic_support'].lower())


if __name__ == '__main__':
    unittest.main()
