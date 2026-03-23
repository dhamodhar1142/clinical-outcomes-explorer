from __future__ import annotations

import unittest

from src.collaboration_notes import (
    append_collaboration_note,
    build_collaboration_note_summary,
    build_collaboration_notes_view,
    build_note_target_options,
)
from src.workspace import build_workspace_identity, sync_workspace_views


class CollaborationNotesTests(unittest.TestCase):
    def test_build_note_target_options_covers_dataset_reports_and_sections(self) -> None:
        options = build_note_target_options(
            'Healthcare Operations Demo',
            ['Ops Pack'],
            ['Executive Summary'],
            ['Data Intake'],
        )
        labels = [option['label'] for option in options]
        self.assertIn('Dataset | Healthcare Operations Demo', labels)
        self.assertIn('Workflow Pack | Ops Pack', labels)
        self.assertIn('Report | Executive Summary', labels)
        self.assertIn('Analysis Section | Data Intake', labels)

    def test_append_note_and_filter_view(self) -> None:
        notes: list[dict[str, object]] = []
        append_collaboration_note(notes, 'Dataset', 'Demo Dataset', 'Review missing payer values.', 'Dana', 'Demo Workspace')
        append_collaboration_note(notes, 'Report', 'Executive Summary', 'Use this for the operations handoff.', 'Dana', 'Demo Workspace')
        report_view = build_collaboration_notes_view(notes, 'Report | Executive Summary')
        self.assertEqual(len(report_view), 1)
        self.assertEqual(report_view.iloc[0]['target_type'], 'Report')

    def test_note_summary_counts_by_target_type(self) -> None:
        notes: list[dict[str, object]] = []
        append_collaboration_note(notes, 'Dataset', 'Demo Dataset', 'Dataset note', 'Dana', 'Demo Workspace')
        append_collaboration_note(notes, 'Workflow Pack', 'Pack One', 'Pack note', 'Dana', 'Demo Workspace')
        summary = build_collaboration_note_summary(notes)
        self.assertEqual(summary['summary_cards'][0]['value'], '2')
        self.assertFalse(summary['history'].empty)
        self.assertFalse(summary['recent_notes'].empty)
        self.assertFalse(summary['top_targets'].empty)

    def test_workspace_sync_separates_notes_by_workspace(self) -> None:
        session_state: dict[str, object] = {
            'workspace_saved_snapshots': {},
            'workspace_workflow_packs': {},
            'workspace_collaboration_notes': {},
            'workspace_identity': build_workspace_identity('User One', 'Workspace A', True),
        }
        sync_workspace_views(session_state)
        append_collaboration_note(session_state['collaboration_notes'], 'Dataset', 'A', 'First note', 'User One', 'Workspace A')

        session_state['workspace_identity'] = build_workspace_identity('User Two', 'Workspace B', True)
        sync_workspace_views(session_state)
        self.assertEqual(session_state['collaboration_notes'], [])

        append_collaboration_note(session_state['collaboration_notes'], 'Dataset', 'B', 'Second note', 'User Two', 'Workspace B')
        session_state['workspace_identity'] = build_workspace_identity('User One', 'Workspace A', True)
        sync_workspace_views(session_state)
        self.assertEqual(len(session_state['collaboration_notes']), 1)
        self.assertEqual(session_state['collaboration_notes'][0]['target_name'], 'A')


if __name__ == '__main__':
    unittest.main()
