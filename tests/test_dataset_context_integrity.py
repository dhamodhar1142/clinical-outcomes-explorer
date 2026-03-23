from __future__ import annotations

import unittest
from types import SimpleNamespace

import pandas as pd

from src.pipeline import run_analysis_pipeline
from src.services.dataset_service import (
    PrimaryDatasetSelection,
    build_selection_from_active_bundle,
    resolve_primary_dataset_selection,
)
from src.visualization_performance import (
    build_visual_cache_diagnostics,
    build_visual_cache_key,
    get_cached_visual_payload,
    store_visual_payload_for_pipeline,
)


class DatasetContextIntegrityTests(unittest.TestCase):
    def test_uploaded_dataset_selection_reuses_cached_bundle_on_rerun(self) -> None:
        frame = pd.DataFrame([{'patient_id': 'P-1', 'value': 10}])
        selection = PrimaryDatasetSelection(
            source_mode='Uploaded dataset',
            data=frame,
            original_lookup={'patient_id': 'patient_id', 'value': 'value'},
            dataset_name='uploaded.csv',
            source_meta={'source_mode': 'Uploaded dataset', 'dataset_cache_key': 'upload-key-1'},
        )

        session_state = {
            'dataset_source_mode': 'Uploaded dataset',
            'active_dataset_bundle': {
                'source_mode': selection.source_mode,
                'data': selection.data,
                'original_lookup': selection.original_lookup,
                'dataset_name': selection.dataset_name,
                'source_meta': selection.source_meta,
            },
            'primary_upload': None,
        }

        resolved = resolve_primary_dataset_selection(
            session_state=session_state,
            load_selection=lambda: PrimaryDatasetSelection(
                source_mode='Uploaded dataset',
                data=selection.data,
                original_lookup=selection.original_lookup,
                dataset_name=selection.dataset_name,
                source_meta=selection.source_meta,
            ),
        )

        self.assertEqual(resolved.dataset_name, 'uploaded.csv')
        self.assertEqual(resolved.source_meta['dataset_cache_key'], 'upload-key-1')

    def test_uploaded_dataset_selection_restores_from_saved_upload_bytes(self) -> None:
        payload = (
            b'patient_id,admission_date,secondary_diagnosis_label,bmi\n'
            b'P-1,2024-01-01,,31.2\n'
            b'P-2,2024-01-03,Diabetes,29.4\n'
        )
        selection = build_selection_from_active_bundle(
            {
                'source_mode': 'Uploaded dataset',
                'data': None,
                'original_lookup': {},
                'dataset_name': 'uploaded.csv',
                'source_meta': {
                    'source_mode': 'Uploaded dataset',
                    'dataset_cache_key': 'upload-key-restored',
                    'sampling_override': 'auto',
                    'upload_status': 'ready',
                },
                'upload_file_name': 'uploaded.csv',
                'upload_file_bytes': payload,
                'upload_size_bytes': len(payload),
                'active_status': 'active',
            }
        )

        self.assertIsNotNone(selection)
        self.assertEqual(selection.dataset_name, 'uploaded.csv')
        self.assertEqual(selection.source_mode, 'Uploaded dataset')
        self.assertEqual(len(selection.data), 2)
        self.assertEqual(selection.source_meta['dataset_cache_key'], 'upload-key-restored')

    def test_uploaded_dataset_selection_does_not_fall_back_to_demo_when_source_not_explicitly_switched(self) -> None:
        payload = (
            b'patient_id,admission_date,secondary_diagnosis_label,bmi\n'
            b'P-1,2024-01-01,,31.2\n'
        )
        active_frame = pd.DataFrame(
            [{'patient_id': 'P-1', 'admission_date': '2024-01-01', 'secondary_diagnosis_label': None, 'bmi': 31.2}]
        )
        session_state = {
            'dataset_source_mode': '',
            'active_dataset_bundle': {
                'source_mode': 'Uploaded dataset',
                'data': None,
                'original_lookup': {},
                'dataset_name': 'uploaded.csv',
                'source_meta': {
                    'source_mode': 'Uploaded dataset',
                    'dataset_cache_key': 'upload-sticky',
                    'sampling_override': 'auto',
                    'upload_status': 'loading',
                },
                'upload_file_name': 'uploaded.csv',
                'upload_file_bytes': payload,
                'upload_size_bytes': len(payload),
                'active_status': 'active',
            },
            'primary_upload': None,
        }

        resolved = resolve_primary_dataset_selection(
            session_state=session_state,
            load_selection=lambda: PrimaryDatasetSelection(
                source_mode='Uploaded dataset',
                data=active_frame,
                original_lookup={'patient_id': 'patient_id', 'admission_date': 'admission_date', 'secondary_diagnosis_label': 'secondary_diagnosis_label', 'bmi': 'bmi'},
                dataset_name='uploaded.csv',
                source_meta={
                    'source_mode': 'Uploaded dataset',
                    'dataset_cache_key': 'upload-sticky',
                    'sampling_override': 'auto',
                    'upload_status': 'loading',
                },
            ),
        )

        self.assertEqual(resolved.source_mode, 'Uploaded dataset')
        self.assertEqual(resolved.dataset_name, 'uploaded.csv')

    def test_pipeline_preserves_uploaded_dataset_identity_through_preprocessing(self) -> None:
        frame = pd.DataFrame(
            [
                {'patient_id': 'P-1', 'admission_date': '2024-01-01', 'secondary_diagnosis_label': None, 'bmi': 31.2},
                {'patient_id': 'P-1', 'admission_date': '2024-01-03', 'secondary_diagnosis_label': 'Hypertension', 'bmi': 31.0},
            ]
        )
        frame.attrs['dataset_cache_key'] = 'upload-cache-42'

        pipeline = run_analysis_pipeline(
            frame,
            'uploaded.csv',
            {
                'source_mode': 'Uploaded dataset',
                'dataset_cache_key': 'upload-cache-42',
                'file_size_mb': 0.01,
            },
            demo_config={'synthetic_helper_mode': 'Off'},
        )

        self.assertEqual(pipeline['data'].attrs.get('dataset_cache_key'), 'upload-cache-42')
        self.assertEqual(pipeline['data'].attrs.get('dataset_identifier'), 'upload-cache-42')
        self.assertEqual(pipeline['dataset_runtime_diagnostics']['dataset_identifier'], 'upload-cache-42')
        self.assertEqual(pipeline['dataset_runtime_diagnostics']['row_count'], 2)

    def test_visual_cache_diagnostics_track_current_dataset_ownership(self) -> None:
        session_state: dict[str, object] = {}
        frame_one = pd.DataFrame({'event_date': pd.to_datetime(['2024-01-01']), 'value': [10]})
        frame_one.attrs['dataset_cache_key'] = 'dataset-a'
        frame_one.attrs['dataset_identifier'] = 'dataset-a'
        frame_one.attrs['dataset_name'] = 'uploaded-a.csv'
        pipeline_one = {
            'data': frame_one,
            'structure': SimpleNamespace(default_date_column='event_date', numeric_columns=['value']),
            'semantic': {'canonical_map': {}},
            'healthcare': {'survival_outcomes': {}, 'readmission': {}},
        }

        frame_two = pd.DataFrame({'event_date': pd.to_datetime(['2024-01-01']), 'value': [20]})
        frame_two.attrs['dataset_cache_key'] = 'dataset-b'
        frame_two.attrs['dataset_identifier'] = 'dataset-b'
        frame_two.attrs['dataset_name'] = 'uploaded-b.csv'
        pipeline_two = {
            'data': frame_two,
            'structure': SimpleNamespace(default_date_column='event_date', numeric_columns=['value']),
            'semantic': {'canonical_map': {}},
            'healthcare': {'survival_outcomes': {}, 'readmission': {}},
        }

        cache_key_one = build_visual_cache_key(pipeline_one, 'trend_analysis')
        cache_key_two = build_visual_cache_key(pipeline_two, 'trend_analysis')
        self.assertNotEqual(cache_key_one, cache_key_two)

        store_visual_payload_for_pipeline(session_state, pipeline_one, cache_key_one, {'records': 1})

        payload = get_cached_visual_payload(session_state, cache_key_one)
        diagnostics_one = build_visual_cache_diagnostics(session_state, pipeline_one, cache_key_one)
        diagnostics_two = build_visual_cache_diagnostics(session_state, pipeline_two, cache_key_one)

        self.assertEqual(payload, {'records': 1})
        self.assertTrue(diagnostics_one['cache_belongs_to_current_dataset'])
        self.assertFalse(diagnostics_two['cache_belongs_to_current_dataset'])


if __name__ == '__main__':
    unittest.main()
