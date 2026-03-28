from __future__ import annotations

import unittest

from src.governance_portability import (
    build_governance_release_bundle,
    build_governance_release_bundle_compatibility_gate,
    build_governance_release_bundle_drift,
    build_governance_release_bundle_gate,
    build_governance_release_bundle_promotion_readiness,
    build_governance_release_bundle_runtime_compatibility,
    build_governance_release_bundle_bytes,
    parse_governance_release_bundle,
    restore_governance_release_bundle,
)


class GovernancePortabilityTests(unittest.TestCase):
    def test_governance_release_bundle_round_trips(self):
        bundle = build_governance_release_bundle(
            dataset_name='demo.csv',
            source_meta={'source_mode': 'Uploaded dataset', 'dataset_identifier': 'dataset-123'},
            workspace_identity={'workspace_id': 'workspace-a', 'workspace_name': 'Workspace A', 'display_name': 'Analyst A'},
            policy_pack_name='Enterprise Release Controls',
            policy_pack={
                'export_policy_name': 'Internal Review',
                'governance_default_owner': 'Admin',
                'approval_routing_rules': {'release_signoff': 'Admin'},
            },
            benchmark_packs={'Pack A': {'detail_note': 'Example pack'}},
            execution_queue=[{'proposal_title': 'Run release validation', 'priority': 'High', 'status': 'Open'}],
            review_approvals={'dataset-123': {'release_signoff_status': 'Pending'}},
        )

        parsed = parse_governance_release_bundle(build_governance_release_bundle_bytes(bundle).decode('utf-8'))
        session_state: dict[str, object] = {}
        restored = restore_governance_release_bundle(parsed, session_state)

        self.assertEqual(restored['policy_pack_name'], 'Enterprise Release Controls')
        self.assertEqual(session_state['export_policy_name'], 'Internal Review')
        self.assertEqual(session_state['governance_default_owner'], 'Admin')
        self.assertEqual(session_state['approval_routing_rules']['release_signoff'], 'Admin')
        self.assertIn('Pack A', session_state['organization_benchmark_packs'])
        self.assertEqual(session_state['evolution_execution_queue'][0]['proposal_title'], 'Run release validation')

    def test_invalid_bundle_type_is_rejected(self):
        with self.assertRaises(ValueError):
            parse_governance_release_bundle('{"bundle_type":"wrong","bundle_version":1}')

    def test_governance_release_bundle_drift_detects_policy_and_routing_changes(self):
        current_bundle = build_governance_release_bundle(
            dataset_name='demo.csv',
            source_meta={'source_mode': 'Uploaded dataset', 'dataset_identifier': 'dataset-123'},
            workspace_identity={'workspace_id': 'workspace-a', 'workspace_name': 'Workspace A', 'display_name': 'Analyst A'},
            policy_pack_name='Current Controls',
            policy_pack={
                'export_policy_name': 'Internal Review',
                'accuracy_reporting_min_trust_score': 0.76,
                'approval_routing_rules': {'release_signoff': 'Admin'},
            },
            benchmark_packs={'Pack A': {'detail_note': 'Example pack'}},
            execution_queue=[{'proposal_title': 'Run release validation', 'priority': 'High', 'status': 'Open'}],
            review_approvals={},
        )
        imported_bundle = build_governance_release_bundle(
            dataset_name='demo.csv',
            source_meta={'source_mode': 'Uploaded dataset', 'dataset_identifier': 'dataset-123'},
            workspace_identity={'workspace_id': 'workspace-a', 'workspace_name': 'Workspace A', 'display_name': 'Analyst A'},
            policy_pack_name='Imported Controls',
            policy_pack={
                'export_policy_name': 'Governed External',
                'accuracy_reporting_min_trust_score': 0.84,
                'approval_routing_rules': {'release_signoff': 'Executive'},
            },
            benchmark_packs={'Pack A': {'detail_note': 'Example pack'}, 'Pack B': {'detail_note': 'Second pack'}},
            execution_queue=[],
            review_approvals={},
        )

        drift = build_governance_release_bundle_drift(current_bundle, imported_bundle)

        export_policy_row = drift.loc[drift['drift_area'] == 'Export policy'].iloc[0]
        self.assertEqual(export_policy_row['status'], 'Changed')
        self.assertEqual(export_policy_row['bundle_value'], 'Governed External')

        release_routing_row = drift.loc[drift['drift_area'] == 'Approval routing: release_signoff'].iloc[0]
        self.assertEqual(release_routing_row['status'], 'Changed')
        self.assertEqual(release_routing_row['release_impact'], 'High')

        benchmark_summary_row = drift.loc[drift['drift_area'] == 'Organization benchmark packs'].iloc[0]
        self.assertEqual(benchmark_summary_row['bundle_value'], '2')

        gate = build_governance_release_bundle_gate(drift)
        self.assertTrue(gate['requires_signoff'])
        self.assertGreaterEqual(gate['high_impact_drift_count'], 1)

    def test_governance_release_bundle_runtime_compatibility_flags_cloud_handoff(self):
        imported_bundle = build_governance_release_bundle(
            dataset_name='demo.csv',
            source_meta={'source_mode': 'Uploaded dataset', 'dataset_identifier': 'dataset-123'},
            workspace_identity={'workspace_id': 'workspace-a', 'workspace_name': 'Workspace A', 'display_name': 'Analyst A'},
            policy_pack_name='Imported Controls',
            policy_pack={
                'validation_runtime_preference': 'Prefer local/staging for heavy actions',
                'export_runtime_preference': 'Prefer local/staging for heavy actions',
            },
            benchmark_packs={},
            execution_queue=[
                {
                    'proposal_title': 'Run release validation',
                    'status': 'Queued',
                    'suggested_validation': 'Release validation',
                    'priority': 'High',
                },
                {
                    'proposal_title': 'Prepare governed export bundle',
                    'status': 'Queued',
                    'proposed_change': 'External reporting hardening for governed export.',
                    'priority': 'High',
                },
            ],
            review_approvals={},
        )

        compatibility = build_governance_release_bundle_runtime_compatibility(
            imported_bundle,
            validation_runtime_profile={
                'runtime_label': 'Streamlit Cloud',
                'supports_heavy_validation': False,
            },
            export_runtime_profile={
                'runtime_label': 'Streamlit Cloud',
                'supports_governed_packaging': False,
            },
        )
        gate = build_governance_release_bundle_compatibility_gate(compatibility)

        self.assertTrue(gate['requires_signoff'])
        self.assertGreaterEqual(gate['high_impact_mismatch_count'], 1)
        validation_row = compatibility.loc[compatibility['compatibility_area'] == 'Validation runtime posture'].iloc[0]
        self.assertEqual(validation_row['status'], 'Needs local/staging handoff')

    def test_governance_release_bundle_promotion_readiness_blocks_open_release_blockers(self):
        imported_bundle = build_governance_release_bundle(
            dataset_name='demo.csv',
            source_meta={'source_mode': 'Uploaded dataset', 'dataset_identifier': 'dataset-123'},
            workspace_identity={'workspace_id': 'workspace-a', 'workspace_name': 'Workspace A', 'display_name': 'Analyst A'},
            policy_pack_name='Imported Controls',
            policy_pack={},
            benchmark_packs={},
            execution_queue=[
                {
                    'proposal_title': 'Resolve release blocker',
                    'status': 'In progress',
                    'release_gate_status': 'Required before release',
                    'priority': 'High',
                }
            ],
            review_approvals={},
        )

        readiness = build_governance_release_bundle_promotion_readiness(
            drift_gate={'changed_items_count': 0, 'high_impact_drift_count': 0, 'requires_signoff': False},
            compatibility_gate={'mismatch_count': 0, 'high_impact_mismatch_count': 0, 'requires_signoff': False},
            imported_bundle=imported_bundle,
        )

        self.assertEqual(readiness['status'], 'Blocked')
        self.assertEqual(readiness['open_release_blockers'], 1)


if __name__ == '__main__':
    unittest.main()
