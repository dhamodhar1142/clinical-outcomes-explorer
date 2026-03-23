from __future__ import annotations

import unittest

from src.usage_analytics import (
    build_customer_success_summary,
    build_demo_usage_seed_events,
    build_product_admin_summary,
    build_usage_analytics_view,
)


class UsageAnalyticsTests(unittest.TestCase):
    def test_usage_summary_counts_core_events(self) -> None:
        analysis_log = [
            {'event_type': 'Dataset Selected'},
            {'event_type': 'AI Copilot Question'},
            {'event_type': 'Export Downloaded'},
            {'event_type': 'Workflow Action'},
            {'event_type': 'Export Downloaded'},
        ]
        run_history = [
            {'dataset_name': 'Demo A'},
            {'dataset_name': 'Demo A'},
            {'dataset_name': 'Demo B'},
        ]
        result = build_usage_analytics_view(analysis_log, run_history, ['Data Intake', 'Export Center'])
        self.assertEqual(result['summary_cards'][0]['value'], '1')
        self.assertEqual(result['summary_cards'][1]['value'], '1')
        self.assertEqual(result['summary_cards'][2]['value'], '1')
        self.assertEqual(result['summary_cards'][3]['value'], '2')
        self.assertFalse(result['activity_table'].empty)
        self.assertFalse(result['module_table'].empty)
        self.assertFalse(result['dataset_runs'].empty)

    def test_usage_summary_handles_empty_inputs(self) -> None:
        result = build_usage_analytics_view([], [], [])
        self.assertEqual(result['summary_cards'][0]['value'], '0')
        self.assertTrue(result['activity_table'].empty)
        self.assertTrue(result['module_table'].empty)
        self.assertTrue(any('Export activity will appear here' in note for note in result['notes']))

    def test_demo_usage_seed_events_are_created_once_for_demo_datasets(self) -> None:
        result = build_demo_usage_seed_events(
            'Healthcare Operations Demo',
            {'source_mode': 'Demo dataset'},
            demo_mode_enabled=True,
            seeded_keys=[],
        )
        self.assertTrue(result['seeded'])
        self.assertEqual(len(result['events']), 3)
        self.assertEqual(result['events'][0]['event_type'], 'Workflow Action')
        self.assertEqual(result['events'][1]['event_type'], 'AI Copilot Question')
        self.assertEqual(result['events'][2]['event_type'], 'Export Downloaded')
        repeat = build_demo_usage_seed_events(
            'Healthcare Operations Demo',
            {'source_mode': 'Demo dataset'},
            demo_mode_enabled=True,
            seeded_keys=result['seeded_keys'],
        )
        self.assertFalse(repeat['seeded'])
        self.assertEqual(repeat['events'], [])

    def test_demo_usage_seed_events_do_not_run_for_uploaded_datasets(self) -> None:
        result = build_demo_usage_seed_events(
            'uploaded.csv',
            {'source_mode': 'Uploaded dataset'},
            demo_mode_enabled=True,
            seeded_keys=[],
        )
        self.assertFalse(result['seeded'])
        self.assertEqual(result['events'], [])

    def test_customer_success_summary_counts_value_signals(self) -> None:
        analysis_log = [
            {'event_type': 'Dataset Selected'},
            {'event_type': 'Export Generated'},
            {'event_type': 'Export Downloaded'},
            {'event_type': 'Workflow Pack Saved'},
            {'event_type': 'Session Bundle Restored'},
        ]
        run_history = [
            {'dataset_name': 'Demo A', 'major_blockers_detected': 'Missing date field', 'synthetic_helper_fields_added': 2},
            {'dataset_name': 'Demo B', 'major_blockers_detected': '', 'synthetic_helper_fields_added': 0},
        ]
        result = build_customer_success_summary(analysis_log, run_history)
        self.assertEqual(result['summary_cards'][0]['value'], '2')
        self.assertEqual(result['summary_cards'][1]['value'], '1')
        self.assertEqual(result['summary_cards'][2]['value'], '2')
        self.assertEqual(result['summary_cards'][3]['value'], '2')
        self.assertFalse(result['value_table'].empty)

    def test_customer_success_summary_handles_empty_inputs(self) -> None:
        result = build_customer_success_summary([], [])
        self.assertEqual(result['summary_cards'][0]['value'], '0')
        self.assertTrue(any('local to this workspace' in note for note in result['notes']))

    def test_product_admin_summary_supports_session_fallback(self) -> None:
        result = build_product_admin_summary(
            workspace_identity={
                'workspace_name': 'Guest Demo Workspace',
                'workspace_id': 'guest-demo-workspace',
                'display_name': 'Guest User',
                'owner_label': 'Guest session',
                'role_label': 'Viewer',
            },
            persistence_service=None,
            plan_awareness={
                'active_plan': 'Pro',
                'description': 'Demo-safe plan packaging preview.',
                'strict_enforcement': False,
                'workflow_pack_limit_reached': False,
                'advanced_exports_available': True,
            },
            analysis_log=[{'event_type': 'Export Downloaded'}],
            run_history=[{'dataset_name': 'Demo A'}],
            generated_report_outputs={'Executive Summary': 'ok'},
            saved_snapshots={'one': {}},
            workflow_packs={'starter': {}},
        )
        self.assertEqual(result['summary_cards'][0]['value'], 'Guest Demo Workspace')
        self.assertEqual(result['summary_cards'][2]['value'], 'Pro')
        self.assertFalse(result['workspace_table'].empty)
        self.assertFalse(result['plan_table'].empty)
        self.assertFalse(result['usage_table'].empty)
        self.assertFalse(result['reports_table'].empty)
        self.assertTrue(any('session-safe workspace signals' in note for note in result['notes']))

    def test_product_admin_summary_uses_persisted_workspace_signals_when_available(self) -> None:
        class _Status:
            enabled = True
            mode = 'sqlite'
            storage_target = 'tmp\\workspace.db'

        class _PersistenceService:
            enabled = True
            status = _Status()

            def load_workspace_summary(self, identity):
                return {
                    'workspace': {'workspace_id': identity['workspace_id']},
                    'dataset_count': 3,
                    'report_count': 2,
                    'usage_event_count': 7,
                }

            def list_usage_events(self, identity, limit=25):
                return [{'event_type': 'Dataset Selected'}]

            def list_report_metadata(self, identity, limit=25):
                return [
                    {
                        'dataset_name': 'Hospital Reporting Demo',
                        'report_type': 'Readmission Report',
                        'status': 'generated',
                        'generated_at': '2026-03-14T12:00:00+00:00',
                    }
                ]

            def list_dataset_metadata(self, identity):
                return [{'dataset_name': 'Hospital Reporting Demo', 'source_mode': 'Demo dataset'}]

        result = build_product_admin_summary(
            workspace_identity={
                'workspace_name': 'Pilot Workspace',
                'workspace_id': 'pilot-workspace',
                'display_name': 'Analyst User',
                'owner_label': 'Analyst User',
                'role_label': 'Admin',
            },
            persistence_service=_PersistenceService(),
            plan_awareness={'active_plan': 'Team', 'description': 'Team plan', 'strict_enforcement': True},
            analysis_log=[],
            run_history=[],
        )
        self.assertEqual(result['summary_cards'][3]['value'], 'SQLite-backed')
        self.assertFalse(result['dataset_ops_table'].empty)
        self.assertEqual(result['reports_table'].iloc[0]['report_type'], 'Readmission Report')


if __name__ == '__main__':
    unittest.main()
