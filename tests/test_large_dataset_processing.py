from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from pandas.testing import assert_frame_equal

from src.data_loader import DataLoadError, StreamingDatasetAnalyzer, read_csv_bytes_with_strategy
from src.profiler import build_field_profile, build_profile_cache_summary, build_structure_profile_bundle, clear_profile_cache, default_profile_cache_metrics
from src.schema_detection import detect_structure


class LargeDatasetProcessingTests(unittest.TestCase):
    def test_streaming_strategy_activates_for_sampled_large_csvs(self) -> None:
        sample_df = pd.DataFrame([{'patient_id': 'P1', 'cost': 1.0}])
        sample_stats = {
            'ingestion_strategy': 'hybrid_streaming_csv',
            'sampling_mode': 'sampled',
            'source_row_count': 917_000,
            'analyzed_row_count': 1,
            'chunk_count': 4,
            'estimated_total_chunks': 4,
            'original_lookup': {'patient_id': 'patient_id', 'cost': 'cost'},
        }
        with patch('src.data_loader.STREAMING_UPLOAD_MB', 0.001), \
             patch('src.data_loader.SAMPLED_ANALYSIS_UPLOAD_MB', 0.002), \
             patch('src.data_loader.MAX_SAMPLED_UPLOAD_MB', 2.0), \
             patch('src.data_loader._try_read_csv_chunks', return_value=(sample_df, sample_stats)):
            data, stats = read_csv_bytes_with_strategy(b'a' * 4096, file_name='large.csv')

        self.assertEqual(len(data), 1)
        self.assertEqual(stats['ingestion_strategy'], 'hybrid_streaming_csv')
        self.assertEqual(stats['sampling_mode'], 'sampled')
        self.assertEqual(stats['source_row_count'], 917_000)
        self.assertTrue(stats['dataset_cache_key'])

    def test_streaming_strategy_activates_for_100mb_plus_csvs(self) -> None:
        sample_df = pd.DataFrame([{'patient_id': 'P1', 'cost': 1.0}])
        sample_stats = {
            'ingestion_strategy': 'sampled_streaming_csv',
            'sampling_mode': 'sampled',
            'source_row_count': 1_500_000,
            'analyzed_row_count': 1,
            'chunk_count': 8,
            'original_lookup': {'patient_id': 'patient_id', 'cost': 'cost'},
        }
        large_payload = b'a' * (101 * 1024 * 1024)
        with patch('src.data_loader.STREAMING_UPLOAD_MB', 50.0), \
             patch('src.data_loader.SAMPLED_ANALYSIS_UPLOAD_MB', 100.0), \
             patch('src.data_loader.MAX_SAMPLED_UPLOAD_MB', 500.0), \
             patch('src.data_loader._try_read_csv_chunks', return_value=(sample_df, sample_stats)):
            data, stats = read_csv_bytes_with_strategy(large_payload, file_name='100mb-plus.csv')

        self.assertEqual(len(data), 1)
        self.assertEqual(stats['ingestion_strategy'], 'sampled_streaming_csv')
        self.assertEqual(stats['sampling_mode'], 'sampled')
        self.assertGreater(stats['file_size_mb'], 100.0)

    def test_streaming_strategy_recommends_sampling_above_500mb(self) -> None:
        sample_df = pd.DataFrame([{'patient_id': 'P1', 'cost': 1.0}])
        sample_stats = {
            'ingestion_strategy': 'sampled_streaming_csv',
            'sampling_mode': 'sampled',
            'source_row_count': 2_000_000,
            'analyzed_row_count': 1,
            'chunk_count': 12,
            'estimated_total_chunks': 52,
            'sampling_warning': 'Sampling is recommended for interactive analysis.',
            'original_lookup': {'patient_id': 'patient_id', 'cost': 'cost'},
        }
        with patch('src.data_loader.STREAMING_UPLOAD_MB', 0.001), \
             patch('src.data_loader.SAMPLED_ANALYSIS_UPLOAD_MB', 0.002), \
             patch('src.data_loader.MAX_SAMPLED_UPLOAD_MB', 0.003), \
             patch('src.data_loader._try_read_csv_chunks', return_value=(sample_df, sample_stats)):
            data, stats = read_csv_bytes_with_strategy(b'a' * 4096, file_name='stress.csv')

        self.assertEqual(len(data), 1)
        self.assertEqual(stats['sampling_mode'], 'sampled')
        self.assertIn('Sampling is recommended', stats['sampling_warning'])

    def test_streaming_strategy_rejects_full_override_above_max_supported_size(self) -> None:
        with patch('src.data_loader.STREAMING_UPLOAD_MB', 0.001), \
             patch('src.data_loader.SAMPLED_ANALYSIS_UPLOAD_MB', 0.002), \
             patch('src.data_loader.MAX_SAMPLED_UPLOAD_MB', 0.003):
            with self.assertRaises(DataLoadError):
                read_csv_bytes_with_strategy(
                    b'a' * 4096,
                    file_name='too-large.csv',
                    sampling_override='full',
                )

    def test_streaming_analyzer_reports_chunk_progress_and_confidence(self) -> None:
        rows = ['patient_id,cost']
        rows.extend(f'P{index},{index}' for index in range(30_000))
        payload = ('\n'.join(rows)).encode('utf-8')
        messages: list[tuple[float, str]] = []

        analyzer = StreamingDatasetAnalyzer(
            file_name='stream.csv',
            file_bytes=payload,
            encoding='utf-8',
            sampling_override='sampled',
            sample_target_rows=5_000,
            progress_callback=lambda value, message: messages.append((value, message)),
        )
        analyzer.chunk_size_bytes = 1024
        analyzer.total_chunks = max(1, int(len(payload) / analyzer.chunk_size_bytes))
        analyzer.rows_per_chunk = 5_000

        data, stats = analyzer.analyze_streaming()

        self.assertFalse(data.empty)
        self.assertTrue(messages)
        self.assertIn('Processing chunk', messages[0][1])
        self.assertIn('MB/', messages[0][1])
        self.assertTrue(stats['confidence_snapshots'])
        self.assertIn('detection_table', stats['confidence_snapshots'][-1])
        self.assertLessEqual(stats['peak_retained_rows'], 5_000)

    def test_streaming_analyzer_keeps_sampled_memory_bounded(self) -> None:
        rows = ['patient_id,cost,department']
        rows.extend(f'P{index},{index % 100},Cardiology' for index in range(80_000))
        payload = ('\n'.join(rows)).encode('utf-8')

        analyzer = StreamingDatasetAnalyzer(
            file_name='bounded.csv',
            file_bytes=payload,
            encoding='utf-8',
            sampling_override='sampled',
            sample_target_rows=4_000,
        )
        analyzer.chunk_size_bytes = 1024
        analyzer.total_chunks = max(1, int(len(payload) / analyzer.chunk_size_bytes))
        analyzer.rows_per_chunk = 8_000

        data, stats = analyzer.analyze_streaming()

        self.assertFalse(data.empty)
        self.assertLessEqual(len(data), 4_000)
        self.assertLessEqual(stats['peak_retained_rows'], 4_000)

    def test_streaming_analyzer_tolerates_malformed_rows_in_middle_of_file(self) -> None:
        payload = '\n'.join(
            [
                'patient_id,cost',
                'P1,10',
                'P2,20',
                'bad,"unterminated quote',
                'P3,30',
                'P4,40',
            ]
        ).encode('utf-8')

        analyzer = StreamingDatasetAnalyzer(
            file_name='malformed.csv',
            file_bytes=payload,
            encoding='utf-8',
            sampling_override='auto',
            sample_target_rows=100,
        )
        analyzer.chunk_size_bytes = 64
        analyzer.total_chunks = max(1, int(len(payload) / analyzer.chunk_size_bytes))
        analyzer.rows_per_chunk = 2

        data, stats = analyzer.analyze_streaming()

        self.assertFalse(data.empty)
        self.assertIn('patient_id', data.columns.tolist())
        self.assertGreaterEqual(stats['source_row_count'], len(data))

    def test_streaming_sample_statistics_stay_close_to_full_analysis(self) -> None:
        rows = ['patient_id,cost,length_of_stay']
        rows.extend(f'P{index},{100 + (index % 25)},{2 + (index % 5)}' for index in range(25_000))
        payload = ('\n'.join(rows)).encode('utf-8')

        sampled_analyzer = StreamingDatasetAnalyzer(
            file_name='accuracy.csv',
            file_bytes=payload,
            encoding='utf-8',
            sampling_override='sampled',
            sample_target_rows=6_000,
        )
        sampled_analyzer.chunk_size_bytes = 1024
        sampled_analyzer.total_chunks = max(1, int(len(payload) / sampled_analyzer.chunk_size_bytes))
        sampled_analyzer.rows_per_chunk = 5_000
        sampled_data, _ = sampled_analyzer.analyze_streaming()

        full_df = pd.read_csv(io.BytesIO(payload))
        cost_delta = abs(sampled_data['cost'].mean() - full_df['cost'].mean()) / full_df['cost'].mean()
        los_delta = abs(sampled_data['length_of_stay'].mean() - full_df['length_of_stay'].mean()) / full_df['length_of_stay'].mean()
        self.assertLessEqual(cost_delta, 0.05)
        self.assertLessEqual(los_delta, 0.05)


    def test_field_profile_reuses_disk_cache(self) -> None:
        data = pd.DataFrame(
            {
                'patient_id': ['P1', 'P2', 'P3'],
                'cost': [10.0, 12.5, 15.0],
            }
        )
        data.attrs['dataset_cache_key'] = 'profile-cache-key'
        structure = detect_structure(data)
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_root = Path(temp_dir)
            with patch('src.profiler.PROFILE_CACHE_ROOT', cache_root):
                first = build_field_profile(data, structure)
                self.assertFalse(first.empty)
                with patch('src.profiler._analysis_sample', side_effect=AssertionError('cache not used')):
                    second = build_field_profile(data, structure)

        assert_frame_equal(first, second, check_dtype=False)

    def test_structure_and_field_profile_cache_bundle_reuses_disk_cache(self) -> None:
        data = pd.DataFrame(
            {
                'patient_id': ['P1', 'P2', 'P3'],
                'cost': [10.0, 12.5, 15.0],
            }
        )
        data.attrs['dataset_cache_key'] = 'bundle-cache-key'
        cache_metrics = default_profile_cache_metrics()
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_root = Path(temp_dir)
            with patch('src.profiler.PROFILE_CACHE_ROOT', cache_root):
                first = build_structure_profile_bundle(data, cache_metrics=cache_metrics)
                self.assertFalse(first['field_profile'].empty)
                with patch('src.profiler.detect_structure', side_effect=AssertionError('structure cache not used')), \
                     patch('src.profiler._analysis_sample', side_effect=AssertionError('field profile cache not used')):
                    second = build_structure_profile_bundle(data, cache_metrics=cache_metrics)

        self.assertEqual(first['dataset_version_hash'], 'bundle-cache-key')
        assert_frame_equal(first['field_profile'], second['field_profile'], check_dtype=False)
        self.assertGreaterEqual(cache_metrics['hits'], 2)
        summary = build_profile_cache_summary(cache_metrics)
        self.assertGreater(summary['hit_rate'], 0.0)

    def test_profile_cache_invalidates_when_profile_config_changes(self) -> None:
        data = pd.DataFrame(
            {
                'patient_id': ['P1', 'P2', 'P3'],
                'cost': [10.0, 12.5, 15.0],
            }
        )
        data.attrs['dataset_cache_key'] = 'config-cache-key'
        cache_metrics = default_profile_cache_metrics()
        first_plan = {'profile_name': 'Standard', 'profile_sample_rows': 10_000, 'quality_sample_rows': 15_000}
        second_plan = {'profile_name': 'High Capacity', 'profile_sample_rows': 14_000, 'quality_sample_rows': 18_000}
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_root = Path(temp_dir)
            with patch('src.profiler.PROFILE_CACHE_ROOT', cache_root):
                first = build_structure_profile_bundle(data, sampling_plan=first_plan, cache_metrics=cache_metrics)
                second = build_structure_profile_bundle(data, sampling_plan=second_plan, cache_metrics=cache_metrics)

        self.assertNotEqual(first['structure_cache_key'], second['structure_cache_key'])
        self.assertEqual(cache_metrics['misses'], 4)

    def test_profile_cache_can_be_cleared_on_demand(self) -> None:
        data = pd.DataFrame(
            {
                'patient_id': ['P1', 'P2', 'P3'],
                'cost': [10.0, 12.5, 15.0],
            }
        )
        data.attrs['dataset_cache_key'] = 'clear-cache-key'
        cache_metrics = default_profile_cache_metrics()
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_root = Path(temp_dir)
            with patch('src.profiler.PROFILE_CACHE_ROOT', cache_root):
                build_structure_profile_bundle(data, cache_metrics=cache_metrics)
                self.assertTrue(any(cache_root.iterdir()))
                clear_profile_cache(cache_metrics)
                self.assertFalse(cache_root.exists())
        self.assertEqual(cache_metrics['clear_count'], 1)

    def test_column_detection_paginates_large_detection_tables(self) -> None:
        try:
            from ui import dataset_profile
        except ModuleNotFoundError:
            self.skipTest('streamlit is not installed in the lean unit-test environment')

        detection = pd.DataFrame(
            [
                {'column_name': f'field_{index}', 'inferred_type': 'text', 'confidence_score': 0.8}
                for index in range(120)
            ]
        )
        pipeline = {
            'structure': type('Structure', (), {'detection_table': detection})(),
            'semantic': {'mapping_table': pd.DataFrame()},
            'data_dictionary': pd.DataFrame(),
        }

        with patch.object(dataset_profile.st, 'subheader'), \
             patch.object(dataset_profile.st, 'selectbox', return_value=25), \
             patch.object(dataset_profile.st, 'number_input', return_value=2), \
             patch.object(dataset_profile.st, 'caption'), \
             patch.object(dataset_profile.st, 'markdown'), \
             patch('ui.dataset_profile.render_standards'), \
             patch('ui.dataset_profile.info_or_table') as info_or_table:
            dataset_profile.render_column_detection(pipeline, {})

        paged_detection = info_or_table.call_args_list[0].args[0]
        self.assertEqual(len(paged_detection), 25)
        self.assertEqual(paged_detection.iloc[0]['column_name'], 'field_25')

    def test_column_detection_skips_manual_mapping_editor_when_streamlit_stub_lacks_layout_primitives(self) -> None:
        try:
            from ui import dataset_profile
        except ModuleNotFoundError:
            self.skipTest('streamlit is not installed in the lean unit-test environment')

        detection = pd.DataFrame(
            [
                {'column_name': 'vis_en', 'inferred_type': 'datetime', 'confidence_score': 0.91},
                {'column_name': 'pat_id', 'inferred_type': 'text', 'confidence_score': 0.87},
            ]
        )
        pipeline = {
            'structure': type('Structure', (), {'detection_table': detection})(),
            'semantic': {'mapping_table': pd.DataFrame(), 'canonical_map': {'patient_id': 'pat_id'}},
            'data_dictionary': pd.DataFrame(),
            'data': pd.DataFrame(columns=['vis_en', 'pat_id']),
        }

        original_streamlit = dataset_profile.st
        fake_streamlit = type(
            'LeanStreamlitStub',
            (),
            {
                'session_state': {},
                'subheader': lambda *args, **kwargs: None,
                'selectbox': lambda *args, **kwargs: 25,
                'number_input': lambda *args, **kwargs: 1,
                'caption': lambda *args, **kwargs: None,
                'markdown': lambda *args, **kwargs: None,
            },
        )()

        try:
            dataset_profile.st = fake_streamlit
            with patch('ui.dataset_profile.render_standards'), \
                 patch('ui.dataset_profile.info_or_table') as info_or_table:
                dataset_profile.render_column_detection(pipeline, {})
        finally:
            dataset_profile.st = original_streamlit

        self.assertTrue(info_or_table.called)


if __name__ == '__main__':
    unittest.main()
