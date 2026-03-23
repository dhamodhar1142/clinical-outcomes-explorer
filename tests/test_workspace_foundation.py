from __future__ import annotations

import unittest

from src.workspace import build_workspace_identity, sync_workspace_views


class WorkspaceFoundationTests(unittest.TestCase):
    def test_signed_in_workspace_identity_builds_stable_id(self) -> None:
        identity = build_workspace_identity('Dana Analyst', 'Care Ops', True, role='owner')
        self.assertTrue(identity['signed_in'])
        self.assertEqual(identity['display_name'], 'Dana Analyst')
        self.assertEqual(identity['workspace_name'], 'Care Ops')
        self.assertEqual(identity['workspace_id'], 'care-ops::dana-analyst')
        self.assertEqual(identity['role'], 'owner')
        self.assertTrue(identity['is_workspace_owner'])

    def test_guest_workspace_identity_uses_demo_defaults(self) -> None:
        identity = build_workspace_identity(None, None, False)
        self.assertFalse(identity['signed_in'])
        self.assertEqual(identity['workspace_id'], 'guest-demo-workspace')
        self.assertEqual(identity['workspace_name'], 'Guest Demo Workspace')
        self.assertEqual(identity['role'], 'viewer')

    def test_sync_workspace_views_separates_snapshots_and_packs_by_workspace(self) -> None:
        session_state: dict[str, object] = {
            'workspace_saved_snapshots': {},
            'workspace_workflow_packs': {},
            'workspace_collaboration_notes': {},
            'workspace_beta_interest_submissions': {},
            'workspace_identity': build_workspace_identity('User One', 'Workspace A', True),
        }
        sync_workspace_views(session_state)
        session_state['saved_snapshots']['snapshot-a'] = {'dataset_name': 'demo-a'}
        session_state['workflow_packs']['pack-a'] = {'summary': 'Pack A'}
        session_state['beta_interest_submissions'].append({'name': 'Jamie', 'email': 'jamie@example.com'})

        session_state['workspace_identity'] = build_workspace_identity('User Two', 'Workspace B', True)
        sync_workspace_views(session_state)
        self.assertEqual(session_state['saved_snapshots'], {})
        self.assertEqual(session_state['workflow_packs'], {})
        self.assertEqual(session_state['beta_interest_submissions'], [])

        session_state['saved_snapshots']['snapshot-b'] = {'dataset_name': 'demo-b'}
        sync_workspace_views(session_state)

        session_state['workspace_identity'] = build_workspace_identity('User One', 'Workspace A', True)
        sync_workspace_views(session_state)
        self.assertIn('snapshot-a', session_state['saved_snapshots'])
        self.assertNotIn('snapshot-b', session_state['saved_snapshots'])
        self.assertIn('pack-a', session_state['workflow_packs'])
        self.assertEqual(len(session_state['beta_interest_submissions']), 1)


if __name__ == '__main__':
    unittest.main()
