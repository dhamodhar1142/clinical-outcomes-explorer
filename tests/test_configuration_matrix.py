from __future__ import annotations

import unittest

import pandas as pd

from src.product_settings import DEFAULT_PRODUCT_SETTINGS, LARGE_DATASET_PROFILES
from src.services.application_service import build_workspace_application_service


class ConfigurationMatrixTests(unittest.TestCase):
    def test_application_service_builds_demo_config_across_supported_combinations(self) -> None:
        application_service = build_workspace_application_service(persistence_service=None)
        report_modes = ['Executive Summary', 'Analyst Report', 'Data Readiness Report']
        export_policies = ['Internal Review', 'Limited Share', 'Stakeholder Share']

        for profile_name in LARGE_DATASET_PROFILES:
            for report_mode in report_modes:
                for export_policy in export_policies:
                    with self.subTest(profile_name=profile_name, report_mode=report_mode, export_policy=export_policy):
                        session_state = dict(DEFAULT_PRODUCT_SETTINGS)
                        session_state.update(
                            {
                                'analysis_template': 'General Review',
                                'report_mode': report_mode,
                                'export_policy_name': export_policy,
                                'active_role': 'Analyst',
                                'active_plan': 'Pro',
                                'plan_enforcement_mode': 'Demo-safe',
                                'demo_dataset_name': 'Healthcare Operations Demo',
                                'product_large_dataset_profile': profile_name,
                            }
                        )
                        controls = application_service.build_active_controls(session_state)
                        demo_config = application_service.build_demo_config(session_state)
                        self.assertEqual(controls['report_mode'], report_mode)
                        self.assertEqual(controls['export_policy_name'], export_policy)
                        self.assertEqual(demo_config['synthetic_helper_mode'], 'Auto')

    def test_application_service_executes_analysis_with_each_large_dataset_profile(self) -> None:
        application_service = build_workspace_application_service(persistence_service=None)
        data = pd.DataFrame([{'patient_id': 1, 'cost': 10.0}])

        for profile_name in LARGE_DATASET_PROFILES:
            with self.subTest(profile_name=profile_name):
                session_state = {
                    'active_plan': 'Pro',
                    'plan_enforcement_mode': 'Demo-safe',
                    'workflow_packs': {},
                    'saved_snapshots': {},
                    'analysis_log': [],
                    'run_history': [],
                    'product_demo_mode_enabled': True,
                    'demo_usage_seeded_keys': [],
                    'product_large_dataset_profile': profile_name,
                }
                result = application_service.execute_analysis_run(
                    session_state,
                    data=data,
                    dataset_name='demo.csv',
                    source_meta={'source_mode': 'Demo dataset', 'file_size_mb': 0.01},
                )
                self.assertFalse(result.blocked)
                self.assertEqual(result.large_dataset_profile['profile_name'], profile_name)


if __name__ == '__main__':
    unittest.main()
