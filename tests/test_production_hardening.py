from __future__ import annotations

import unittest

import pandas as pd

from src.ops_hardening import build_column_validation_report, build_export_safety_note, build_long_task_notice, build_performance_diagnostics, build_preflight_guardrails


class ProductionHardeningTests(unittest.TestCase):
    def test_column_validation_detects_unnamed_duplicate_and_empty_columns(self) -> None:
        data = pd.DataFrame(
            {
                'unnamed_column': [1, 2, 3],
                'patient_id_2': ['P1', 'P2', 'P3'],
                'empty_field': [None, None, None],
                'single_value': ['x', 'x', 'x'],
            }
        )
        report = build_column_validation_report(data)
        self.assertFalse(report['checks_table'].empty)
        self.assertIn('warnings', report)
        self.assertTrue(report['warnings'])
        self.assertFalse(report['issue_samples'].empty)

    def test_preflight_warns_for_large_but_not_blocked_input(self) -> None:
        guardrails = build_preflight_guardrails({'file_size_mb': 30.0}, memory_mb=320.0, row_count=100_000, column_count=20)
        self.assertFalse(guardrails['blocked'])
        self.assertTrue(guardrails['warnings'])

    def test_preflight_uses_large_dataset_profile_thresholds(self) -> None:
        guardrails = build_preflight_guardrails(
            {'file_size_mb': 18.0},
            memory_mb=210.0,
            row_count=60_000,
            column_count=20,
            profile_config={'profile_name': 'Conservative', 'warn_upload_mb': 15.0, 'warn_memory_mb': 200.0, 'recommended_rows': 50_000},
        )
        self.assertFalse(guardrails['blocked'])
        self.assertTrue(guardrails['staged_processing_required'])
        self.assertEqual(guardrails['profile_name'], 'Conservative')

    def test_preflight_allows_sampled_large_uploads_without_blocking(self) -> None:
        guardrails = build_preflight_guardrails(
            {
                'file_size_mb': 180.0,
                'sampling_mode': 'sampled',
                'ingestion_strategy': 'sampled_streaming_csv',
                'source_row_count': 917_000,
                'analyzed_row_count': 120_000,
            },
            memory_mb=240.0,
            row_count=120_000,
            column_count=21,
            profile_config={'profile_name': 'Standard', 'block_upload_mb': 100.0, 'recommended_rows': 100_000},
        )
        self.assertFalse(guardrails['blocked'])
        self.assertTrue(guardrails['staged_processing_required'])
        self.assertIn('Rows analyzed interactively', guardrails['checks_table']['check'].tolist())

    def test_long_task_notice_for_large_dataset(self) -> None:
        notice = build_long_task_notice({'file_size_mb': 10.0}, row_count=75_000, memory_mb=120.0)
        self.assertIsNotNone(notice)
        self.assertIn('staged progress', notice)

    def test_long_task_notice_mentions_sampling_mode(self) -> None:
        notice = build_long_task_notice(
            {'file_size_mb': 180.0, 'sampling_mode': 'sampled', 'ingestion_strategy': 'sampled_streaming_csv'},
            row_count=120_000,
            memory_mb=140.0,
        )
        self.assertIsNotNone(notice)
        self.assertIn('sampled mode', notice)

    def test_performance_diagnostics_includes_profile_and_staged_processing(self) -> None:
        diagnostics = build_performance_diagnostics(
            {'rows': 120_000, 'analyzed_rows': 75_000, 'columns': 20, 'analyzed_columns': 22, 'source_columns': 20, 'memory_mb': 180.0},
            {'sampling_applied': True, 'profile_sample_rows': 12_000, 'quality_sample_rows': 18_000, 'sampling_mode': 'sampled'},
            {'source_mode': 'Uploaded dataset', 'file_size_mb': 28.0, 'ingestion_strategy': 'streaming_csv', 'sampling_mode': 'sampled'},
            profile_config={'profile_name': 'Standard'},
        )
        self.assertIn('Large dataset profile', diagnostics['metric'].tolist())
        self.assertIn('Staged processing', diagnostics['metric'].tolist())
        self.assertIn('Ingestion strategy', diagnostics['metric'].tolist())
        self.assertIn('Analyzed rows in memory', diagnostics['metric'].tolist())

    def test_export_safety_note_for_sampled_dataset(self) -> None:
        note = build_export_safety_note(
            80_000,
            {'sampling_applied': True},
            {'profile_name': 'Standard', 'export_guard_rows': 250_000},
        )
        self.assertIsNotNone(note)
        self.assertIn('staged sampling safeguards', note)


if __name__ == '__main__':
    unittest.main()
