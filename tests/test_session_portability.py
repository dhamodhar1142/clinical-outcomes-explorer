from __future__ import annotations

import json
import unittest

from src.session_portability import (
    build_session_export_bundle,
    build_session_export_text,
    parse_session_import,
    restore_session_bundle,
)


VALID_OPTIONS = {
    'dataset_source_mode': ['Built-in example dataset', 'Uploaded dataset'],
    'demo_dataset_name': ['Healthcare Operations Demo', 'Hospital Reporting Demo', 'Generic Business Demo'],
    'analysis_template': ['General Review', 'Hospital Readmission Analysis'],
    'report_mode': ['Executive Summary', 'Analyst Report'],
    'export_policy_name': ['Internal Review', 'Research-safe Extract'],
    'active_role': ['Analyst', 'Executive'],
    'active_plan': ['Free', 'Pro', 'Team'],
    'plan_enforcement_mode': ['Demo-safe', 'Strict', 'Off'],
}


class SessionPortabilityTests(unittest.TestCase):
    def test_build_bundle_includes_workspace_snapshots_and_packs(self) -> None:
        session_state = {
            'dataset_source_mode': 'Built-in example dataset',
            'demo_dataset_name': 'Healthcare Operations Demo',
            'analysis_template': 'General Review',
            'report_mode': 'Executive Summary',
            'cohort_age_range': (40, 75),
            'saved_snapshots': {'Ops Snapshot': {'dataset_name': 'Healthcare Operations Demo'}},
            'workflow_packs': {'Ops Pack': {'summary': 'Ops workflow'}},
            'collaboration_notes': [{'target_type': 'Dataset', 'target_name': 'Healthcare Operations Demo', 'note_text': 'Review payer mapping.'}],
            'beta_interest_submissions': [{'name': 'Jamie Rivera', 'email': 'jamie@example.com'}],
        }
        bundle = build_session_export_bundle(
            session_state,
            'Healthcare Operations Demo',
            {'source_mode': 'Demo dataset', 'description': 'Demo file'},
            {'workspace_name': 'Care Ops Workspace', 'display_name': 'Dana Analyst'},
        )
        self.assertEqual(bundle['workspace_context']['workspace_name'], 'Care Ops Workspace')
        self.assertIn('Ops Snapshot', bundle['saved_snapshots'])
        self.assertIn('Ops Pack', bundle['workflow_packs'])
        self.assertEqual(len(bundle['collaboration_notes']), 1)
        self.assertEqual(len(bundle['beta_interest_submissions']), 1)
        self.assertEqual(bundle['portable_state']['cohort_age_range'], [40, 75])

    def test_parse_session_import_rejects_non_bundle_payload(self) -> None:
        with self.assertRaises(ValueError):
            parse_session_import(json.dumps({'hello': 'world'}))

    def test_parse_session_import_rejects_newer_bundle_version(self) -> None:
        with self.assertRaises(ValueError):
            parse_session_import(
                json.dumps(
                    {
                        'bundle_type': 'smart_dataset_analyzer_session',
                        'bundle_version': 2,
                    }
                )
            )

    def test_restore_bundle_validates_options_and_restores_selected_assets(self) -> None:
        session_state: dict[str, object] = {}
        bundle = {
            'bundle_type': 'smart_dataset_analyzer_session',
            'bundle_version': 1,
            'dataset_context': {'source_mode': 'Built-in example dataset'},
            'workspace_context': {'workspace_name': 'Research Workspace', 'display_name': 'Dana Analyst'},
            'portable_state': {
                'analysis_template': 'General Review',
                'report_mode': 'Unsupported Report',
                'active_role': 'Analyst',
                'cohort_age_range': [25, 60],
                'selected_snapshot': 'Saved One',
                'selected_workflow_pack': 'Pack One',
            },
            'saved_snapshots': {'Saved One': {'dataset_name': 'Demo'}},
            'workflow_packs': {'Pack One': {'summary': 'Workflow'}},
            'collaboration_notes': [{'target_type': 'Report', 'target_name': 'Executive Summary', 'note_text': 'Use this for handoff.'}],
            'beta_interest_submissions': [{'name': 'Jamie Rivera', 'email': 'jamie@example.com'}],
        }
        result = restore_session_bundle(bundle, session_state, VALID_OPTIONS)
        self.assertEqual(session_state['analysis_template'], 'General Review')
        self.assertNotIn('report_mode', session_state)
        self.assertEqual(session_state['cohort_age_range'], (25, 60))
        self.assertEqual(session_state['selected_snapshot'], 'Saved One')
        self.assertEqual(session_state['selected_workflow_pack'], 'Pack One')
        self.assertEqual(len(session_state['collaboration_notes']), 1)
        self.assertEqual(len(session_state['beta_interest_submissions']), 1)
        self.assertEqual(session_state['workspace_name'], 'Research Workspace')
        self.assertIn('report_mode', result['skipped_keys'])

    def test_restore_bundle_adds_note_for_uploaded_dataset_context(self) -> None:
        session_state: dict[str, object] = {}
        bundle = {
            'bundle_type': 'smart_dataset_analyzer_session',
            'bundle_version': 1,
            'dataset_context': {'source_mode': 'Uploaded dataset'},
            'workspace_context': {},
            'portable_state': {},
            'saved_snapshots': {},
            'workflow_packs': {},
        }
        result = restore_session_bundle(bundle, session_state, VALID_OPTIONS)
        self.assertTrue(any('Uploaded file contents are not embedded' in note for note in result['notes']))

    def test_restore_bundle_sanitizes_prefixed_widget_values(self) -> None:
        session_state: dict[str, object] = {}
        bundle = {
            'bundle_type': 'smart_dataset_analyzer_session',
            'bundle_version': 1,
            'dataset_context': {'source_mode': 'Built-in example dataset'},
            'workspace_context': {},
            'portable_state': {
                'cohort_age_range': ['18', '250'],
                'readmit_follow_up': '80',
                'readmit_case_mgmt': 'abc',
                'readmit_los_reduction': '9.5',
                'workflow_action_prompt': '  Prepare a pilot summary  ',
            },
            'saved_snapshots': {},
            'workflow_packs': {},
        }
        result = restore_session_bundle(bundle, session_state, VALID_OPTIONS)
        self.assertEqual(session_state['cohort_age_range'], (18, 120))
        self.assertEqual(session_state['readmit_follow_up'], 50)
        self.assertEqual(session_state['readmit_los_reduction'], 3.0)
        self.assertEqual(session_state['workflow_action_prompt'], 'Prepare a pilot summary')
        self.assertIn('readmit_case_mgmt', result['skipped_keys'])
        self.assertTrue(any('percentage' in note.lower() for note in result['notes']))

    def test_export_text_round_trips_bundle(self) -> None:
        bundle = build_session_export_bundle(
            {'report_mode': 'Executive Summary', 'saved_snapshots': {}, 'workflow_packs': {}},
            'Demo Dataset',
            {'source_mode': 'Demo dataset'},
            {'workspace_name': 'Demo Workspace', 'display_name': 'Demo User'},
        )
        restored = parse_session_import(build_session_export_text(bundle).decode('utf-8'))
        self.assertEqual(restored['bundle_type'], 'smart_dataset_analyzer_session')


if __name__ == '__main__':
    unittest.main()
