import unittest

import pandas as pd

from src.audience_modes import build_audience_mode_guidance
from src.export_utils import build_role_export_bundle_manifest, recommended_report_mode_for_role


class AudienceModeGuidanceTests(unittest.TestCase):
    def test_executive_mode_prefers_executive_outputs(self):
        capability = pd.DataFrame(
            [
                {'analytics_module': 'Data Profiling', 'status': 'enabled', 'rationale': 'available'},
                {'analytics_module': 'Export / Executive Reporting', 'status': 'enabled', 'rationale': 'available'},
                {'analytics_module': 'Risk Segmentation', 'status': 'enabled', 'rationale': 'available'},
            ]
        )
        result = build_audience_mode_guidance(
            'Executive',
            {'detected_use_case': 'Hospital reporting dataset', 'recommended_workflow': 'Hospital Operations Analytics'},
            {
                'recommended_package': 'Hospital Operations Review',
                'package_details': {
                    'Hospital Operations Review': {
                        'relevant_modules': ['Export / Executive Reporting', 'Risk Segmentation'],
                        'suggested_prompts': ['Generate executive summary'],
                    }
                },
            },
            {'analytics_capability_matrix': capability},
            {'readiness_score': 0.72},
            {'healthcare_readiness_score': 0.60},
        )
        self.assertEqual(result['recommended_report'], 'Executive Summary')
        self.assertEqual(result['recommended_package'], 'Hospital Operations Review')
        self.assertIn('stakeholder walkthroughs', result['help_text'].lower())
        self.assertFalse(result['recommended_outputs'].empty)
        self.assertFalse(result['recommended_sections'].empty)
        self.assertFalse(result['summary_emphasis'].empty)

    def test_data_steward_mode_calls_for_governance(self):
        result = build_audience_mode_guidance(
            'Data Steward',
            {'detected_use_case': 'Generic healthcare dataset', 'recommended_workflow': 'Healthcare Data Readiness'},
            {
                'recommended_package': 'Healthcare Data Readiness',
                'package_details': {
                    'Healthcare Data Readiness': {
                        'relevant_modules': ['Data Quality Review', 'Standards / Governance Review'],
                        'suggested_prompts': ['Show remediation suggestions'],
                    }
                },
            },
            {'analytics_capability_matrix': pd.DataFrame()},
            {'readiness_score': 0.33},
            {'healthcare_readiness_score': 0.21},
        )
        self.assertEqual(result['recommended_report'], 'Data Readiness Review')
        outputs = result['recommended_outputs']['recommended_output'].astype(str).tolist()
        self.assertIn('Governance and audit review packet', outputs)
        self.assertIn('governance', result['focus_summary'].lower())
        sections = result['recommended_sections']['recommended_section'].astype(str).tolist()
        self.assertIn('Data Quality · Quality Review', sections)


class AudienceRoleExportTests(unittest.TestCase):
    def test_new_roles_map_to_expected_reports(self):
        self.assertEqual(recommended_report_mode_for_role('Executive'), 'Executive Summary')
        self.assertEqual(recommended_report_mode_for_role('Clinician'), 'Clinical Review')
        self.assertEqual(recommended_report_mode_for_role('Data Steward'), 'Data Readiness Review')

    def test_new_roles_get_role_specific_bundle_artifacts(self):
        manifest = build_role_export_bundle_manifest(
            'Clinician',
            'Internal Review',
            True,
            'Clinical Report',
            {'sensitive_fields': pd.DataFrame()},
        )
        artifacts = manifest['artifact'].astype(str).tolist()
        self.assertIn('Clinical Review', artifacts)
        self.assertIn('Readmission summary', artifacts)

    def test_data_steward_bundle_uses_readiness_report(self):
        manifest = build_role_export_bundle_manifest(
            'Data Steward',
            'Internal Review',
            True,
            'Data Readiness Review',
            {'sensitive_fields': pd.DataFrame()},
        )
        artifacts = manifest['artifact'].astype(str).tolist()
        self.assertIn('Data Readiness Review', artifacts)


if __name__ == '__main__':
    unittest.main()
