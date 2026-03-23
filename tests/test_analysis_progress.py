from __future__ import annotations

import unittest
from unittest.mock import patch

from src.analysis_progress import build_analysis_progress_snapshot


class AnalysisProgressTests(unittest.TestCase):
    def test_demo_eta_baselines_match_expected_ranges(self) -> None:
        scenarios = [
            ('Healthcare Operations Demo', 5000, 20, 40.0, 50.0),
            ('Hospital Reporting Demo', 9, 9, 25.0, 35.0),
            ('Generic Business Demo', 10, 10, 18.0, 22.0),
        ]

        for dataset_name, row_count, column_count, minimum_eta, maximum_eta in scenarios:
            with self.subTest(dataset_name=dataset_name), patch('src.analysis_progress.time.monotonic', return_value=100.0):
                snapshot = build_analysis_progress_snapshot(
                    progress=0.0,
                    message='Loading data...',
                    current_operation='Loading data',
                    step_index=1,
                    total_steps=5,
                    started_at=100.0,
                    file_size_mb=0.0,
                    row_count=row_count,
                    column_count=column_count,
                    source_mode='Demo dataset',
                    dataset_name=dataset_name,
                )

            self.assertGreaterEqual(snapshot['estimated_remaining_seconds'], minimum_eta)
            self.assertLessEqual(snapshot['estimated_remaining_seconds'], maximum_eta)

    def test_eta_refresh_is_throttled_to_two_seconds(self) -> None:
        with patch('src.analysis_progress.time.monotonic', return_value=110.0):
            first = build_analysis_progress_snapshot(
                progress=0.2,
                message='Profiling columns...',
                current_operation='Profiling columns',
                step_index=2,
                total_steps=5,
                started_at=100.0,
                file_size_mb=65.8,
                row_count=917_000,
                column_count=21,
            )
        with patch('src.analysis_progress.time.monotonic', return_value=111.0):
            second = build_analysis_progress_snapshot(
                progress=0.35,
                message='Analyzing quality...',
                current_operation='Analyzing quality',
                step_index=3,
                total_steps=5,
                started_at=100.0,
                file_size_mb=65.8,
                row_count=917_000,
                column_count=21,
                previous_snapshot=first,
            )

        self.assertEqual(second['estimated_remaining_seconds'], first['estimated_remaining_seconds'])
        self.assertEqual(second['eta_updated_at'], first['eta_updated_at'])

    def test_eta_updates_after_two_seconds_using_smoothing(self) -> None:
        with patch('src.analysis_progress.time.monotonic', return_value=110.0):
            first = build_analysis_progress_snapshot(
                progress=0.2,
                message='Profiling columns...',
                current_operation='Profiling columns',
                step_index=2,
                total_steps=5,
                started_at=100.0,
                file_size_mb=65.8,
                row_count=917_000,
                column_count=21,
            )
        with patch('src.analysis_progress.time.monotonic', return_value=114.0):
            second = build_analysis_progress_snapshot(
                progress=0.55,
                message='Computing healthcare metrics...',
                current_operation='Computing healthcare metrics',
                step_index=4,
                total_steps=5,
                started_at=100.0,
                file_size_mb=65.8,
                row_count=917_000,
                column_count=21,
                previous_snapshot=first,
            )

        self.assertNotEqual(second['eta_updated_at'], first['eta_updated_at'])
        self.assertLess(second['estimated_remaining_seconds'], first['estimated_remaining_seconds'])
        self.assertGreater(second['eta_sample_count'], first['eta_sample_count'])

    def test_eta_confidence_improves_with_more_progress(self) -> None:
        with patch('src.analysis_progress.time.monotonic', return_value=106.0):
            low = build_analysis_progress_snapshot(
                progress=0.08,
                message='Loading data...',
                current_operation='Loading data',
                step_index=1,
                total_steps=5,
                started_at=100.0,
                file_size_mb=5.0,
                row_count=5_000,
                column_count=10,
            )
        with patch('src.analysis_progress.time.monotonic', return_value=132.0):
            high = build_analysis_progress_snapshot(
                progress=0.78,
                message='Finalizing insights...',
                current_operation='Finalizing insights',
                step_index=5,
                total_steps=5,
                started_at=100.0,
                file_size_mb=5.0,
                row_count=5_000,
                column_count=10,
                previous_snapshot=low,
            )

        self.assertEqual(low['eta_confidence'], 'Low')
        self.assertEqual(high['eta_confidence'], 'High')
        self.assertLess(high['eta_uncertainty_seconds'], low['eta_uncertainty_seconds'])


if __name__ == '__main__':
    unittest.main()
