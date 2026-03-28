from __future__ import annotations

import unittest
from unittest.mock import patch

from src.export_orchestrator import (
    build_export_execution_plan,
    build_export_runtime_profile,
    recommended_report_label,
)


class ExportOrchestratorTests(unittest.TestCase):
    def test_recommended_report_label_falls_back_when_claims_not_available(self):
        label = recommended_report_label(
            'Admin',
            {
                'healthcare': {
                    'claims_validation_utilization': {'available': False},
                }
            },
        )
        self.assertEqual(label, 'Analyst Report')

    @patch('src.export_orchestrator._is_cloud_runtime', return_value=True)
    def test_export_runtime_profile_gates_governed_packaging_in_cloud(self, _mock_runtime):
        profile = build_export_runtime_profile(
            {
                'overview': {'rows': 125000},
                'sample_info': {'sampling_applied': True},
            }
        )
        self.assertTrue(profile['cloud_runtime'])
        self.assertTrue(profile['large_dataset_mode'])
        self.assertFalse(profile['supports_governed_packaging'])

    @patch('src.export_orchestrator._is_cloud_runtime', return_value=True)
    def test_export_execution_plan_blocks_heavy_governed_actions_in_cloud(self, _mock_runtime):
        plan = build_export_execution_plan(
            {
                'overview': {'rows': 125000},
                'sample_info': {'sampling_applied': True},
                'healthcare': {'claims_validation_utilization': {'available': True}},
            },
            role='Analyst',
            export_allowed=True,
            advanced_exports_allowed=True,
            governance_exports_allowed=True,
            stakeholder_bundle_allowed=True,
        )
        report_action = next(row for row in plan if row['task'] == 'report_generation')
        bundle_action = next(row for row in plan if row['task'] == 'governed_bundle')
        governance_action = next(row for row in plan if row['task'] == 'governance_packet')
        self.assertTrue(report_action['allowed'])
        self.assertFalse(bundle_action['allowed'])
        self.assertFalse(governance_action['allowed'])


if __name__ == '__main__':
    unittest.main()
