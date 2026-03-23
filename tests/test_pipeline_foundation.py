from __future__ import annotations

import time
import unittest
from unittest.mock import patch

import pandas as pd

from src.healthcare_analysis import assess_healthcare_dataset
from src.readiness_engine import evaluate_analysis_readiness
from src.pipeline import AnalysisCancelledError, finalize_runtime_pipeline, run_analysis_pipeline
from src.schema_detection import detect_structure
from src.semantic_mapper import infer_semantic_mapping
from src.mapping_profiles import suggest_mapping_profile


class PipelineFoundationTests(unittest.TestCase):
    def test_encounter_style_abbreviations_map_to_healthcare_visit_fields(self):
        data = pd.DataFrame(
            [
                {
                    'pat_id': 'P-001',
                    'medt_id': 'E-1001',
                    'vis_en': '2024-01-01 08:00:00',
                    'vis_ex': '2024-01-03 11:00:00',
                    'vstat_cd': 'DIS',
                    'vstat_des': 'Discharged',
                    'vtype_cd': 'IP',
                    'vtype_des': 'Inpatient',
                    'rom_id': 'WARD-12',
                },
                {
                    'pat_id': 'P-002',
                    'medt_id': 'E-1002',
                    'vis_en': '2024-01-04 09:00:00',
                    'vis_ex': '2024-01-04 16:30:00',
                    'vstat_cd': 'OBS',
                    'vstat_des': 'Observation',
                    'vtype_cd': 'OP',
                    'vtype_des': 'Outpatient',
                    'rom_id': 'WARD-09',
                },
            ]
        )

        semantic = infer_semantic_mapping(data, detect_structure(data))

        self.assertEqual(semantic['canonical_map'].get('patient_id'), 'pat_id')
        self.assertEqual(semantic['canonical_map'].get('encounter_id'), 'medt_id')
        self.assertEqual(semantic['canonical_map'].get('admission_date'), 'vis_en')
        self.assertEqual(semantic['canonical_map'].get('discharge_date'), 'vis_ex')
        self.assertEqual(semantic['canonical_map'].get('encounter_status_code'), 'vstat_cd')
        self.assertEqual(semantic['canonical_map'].get('encounter_status'), 'vstat_des')
        self.assertEqual(semantic['canonical_map'].get('encounter_type_code'), 'vtype_cd')
        self.assertEqual(semantic['canonical_map'].get('encounter_type'), 'vtype_des')
        self.assertEqual(semantic['canonical_map'].get('room_id'), 'rom_id')
        self.assertGreaterEqual(float(semantic.get('healthcare_readiness_score', 0.0)), 0.6)

    def test_encounter_style_dataset_scores_as_healthcare_without_synthetic_support(self):
        canonical_map = {
            'patient_id': 'pat_id',
            'encounter_id': 'medt_id',
            'admission_date': 'vis_en',
            'discharge_date': 'vis_ex',
            'encounter_status_code': 'vstat_cd',
            'encounter_status': 'vstat_des',
            'encounter_type_code': 'vtype_cd',
            'encounter_type': 'vtype_des',
            'room_id': 'rom_id',
        }

        readiness = evaluate_analysis_readiness(canonical_map)
        healthcare = assess_healthcare_dataset(canonical_map)

        self.assertGreaterEqual(float(readiness.get('readiness_score', 0.0)), 0.7)
        self.assertEqual(healthcare.get('likely_dataset_type'), 'Encounter-oriented healthcare dataset')
        self.assertEqual(healthcare.get('synthetic_supported_healthcare_fields'), [])

    def test_native_encounter_fields_keep_segmentation_native_even_when_helper_exists(self):
        canonical_map = {
            'patient_id': 'pat_id',
            'encounter_id': 'medt_id',
            'admission_date': 'vis_en',
            'encounter_type': 'vtype_des',
            'encounter_status': 'vstat_des',
            'diagnosis_code': 'synthetic_diagnosis_code',
        }

        readiness = evaluate_analysis_readiness(canonical_map, synthetic_fields={'diagnosis_code'})
        table = readiness['readiness_table']
        segmentation = table[table['analysis_module'] == 'Diagnosis / Procedure Segmentation'].iloc[0]

        self.assertEqual(segmentation['status'], 'Available')
        self.assertEqual(segmentation['support_type'], 'Native')

    def test_manual_semantic_overrides_replace_auto_mapping(self):
        data = pd.DataFrame(
            [
                {'member_key': 'M-001', 'visit_start_raw': '2024-01-01', 'visit_end_raw': '2024-01-02'},
                {'member_key': 'M-002', 'visit_start_raw': '2024-01-03', 'visit_end_raw': '2024-01-04'},
            ]
        )
        structure = detect_structure(data)
        semantic = infer_semantic_mapping(
            data,
            structure,
            manual_overrides={
                'patient_id': 'member_key',
                'admission_date': 'visit_start_raw',
                'discharge_date': 'visit_end_raw',
            },
        )

        self.assertEqual(semantic['canonical_map'].get('patient_id'), 'member_key')
        self.assertEqual(semantic['canonical_map'].get('admission_date'), 'visit_start_raw')
        self.assertEqual(semantic['canonical_map'].get('discharge_date'), 'visit_end_raw')
        self.assertEqual(semantic['manual_overrides_applied']['patient_id'], 'member_key')
        mapping_table = semantic['mapping_table']
        manual_row = mapping_table[mapping_table['semantic_label'] == 'patient_id'].iloc[0]
        self.assertEqual(manual_row['mapping_source'], 'Manual override')
        self.assertTrue(manual_row['used_downstream'])

    def test_suggest_mapping_profile_recognizes_saved_dataset_family_profile(self):
        columns = ['member_key', 'visit_start_raw', 'visit_end_raw', 'visit_type_label', 'visit_status_label', 'room_code']
        suggestion = suggest_mapping_profile(
            columns,
            dataset_name='AMBIGUOUS_ENCOUNTER_VISITS.csv',
            user_profiles={},
        )

        self.assertIsNotNone(suggestion)
        self.assertEqual(suggestion['profile_name'], 'Encounter Raw Feed Template')
        self.assertEqual(suggestion['resolved_mappings']['patient_id'], 'member_key')

    def test_run_analysis_pipeline_returns_core_sections(self):
        data = pd.DataFrame(
            [
                {
                    'patient_id': 'P-001',
                    'age': 67,
                    'gender': 'F',
                    'diagnosis': 'Heart Failure',
                    'admission_date': '2024-01-01',
                    'discharge_date': '2024-01-05',
                    'length_of_stay': 4,
                    'readmission_flag': 1,
                    'department': 'Cardiology',
                    'cost': 18200,
                },
                {
                    'patient_id': 'P-002',
                    'age': 52,
                    'gender': 'M',
                    'diagnosis': 'COPD',
                    'admission_date': '2024-01-07',
                    'discharge_date': '2024-01-09',
                    'length_of_stay': 2,
                    'readmission_flag': 0,
                    'department': 'Pulmonology',
                    'cost': 9400,
                },
            ]
        )

        pipeline = run_analysis_pipeline(
            data,
            'unit-test-demo.csv',
            {
                'source_mode': 'Uploaded dataset',
                'description': 'Unit test dataset.',
                'best_for': 'Pipeline regression coverage.',
                'file_size_mb': 0.01,
            },
            demo_config={'synthetic_helper_mode': 'Off'},
            active_control_values={
                'report_mode': 'Executive Summary',
                'active_role': 'Researcher',
                'accuracy_benchmark_profile': 'Payer Claims',
                'accuracy_reporting_threshold_profile': 'Permissive',
                'accuracy_reporting_min_trust_score': 0.69,
                'accuracy_allow_directional_external_reporting': True,
            },
        )

        self.assertIn('overview', pipeline)
        self.assertIn('readiness', pipeline)
        self.assertIn('healthcare', pipeline)
        self.assertIn('dataset_intelligence', pipeline)
        self.assertIn('lineage', pipeline)
        self.assertIn('analysis_trust_summary', pipeline)
        self.assertIn('result_accuracy_summary', pipeline)
        self.assertTrue(pipeline['analysis_trust_summary'].get('available'))
        self.assertTrue(pipeline['result_accuracy_summary'].get('available'))
        self.assertIn('summary_text', pipeline['analysis_trust_summary'])
        self.assertTrue(str(pipeline['analysis_trust_summary'].get('summary_text', '')).strip())
        self.assertFalse(pipeline['result_accuracy_summary']['module_reporting_gates'].empty)
        self.assertEqual(pipeline['result_accuracy_summary']['benchmark_profile']['profile_name'], 'Payer Claims')
        self.assertEqual(pipeline['result_accuracy_summary']['reporting_policy']['profile_name'], 'Permissive')
        self.assertTrue(pipeline['result_accuracy_summary']['reporting_policy']['allow_directional_external_reporting'])
        self.assertIn('uncertainty_narrative', pipeline['result_accuracy_summary'])
        self.assertIn('metric_lineage_table', pipeline['result_accuracy_summary'])
        self.assertIn('approval_workflow', pipeline['result_accuracy_summary'])

    def test_run_analysis_pipeline_honors_manual_semantic_overrides(self):
        data = pd.DataFrame(
            [
                {
                    'member_key': 'M-001',
                    'visit_start_raw': '2024-01-01',
                    'visit_end_raw': '2024-01-02',
                    'visit_type_label': 'Observation',
                    'visit_status_label': 'Discharged',
                },
                {
                    'member_key': 'M-002',
                    'visit_start_raw': '2024-01-03',
                    'visit_end_raw': '2024-01-04',
                    'visit_type_label': 'Inpatient',
                    'visit_status_label': 'Completed',
                },
            ]
        )

        pipeline = run_analysis_pipeline(
            data,
            'manual-override-test.csv',
            {
                'source_mode': 'Uploaded dataset',
                'description': 'Manual override regression dataset.',
                'best_for': 'Manual semantic mapping coverage.',
                'file_size_mb': 0.01,
                'manual_semantic_overrides': {
                    'patient_id': 'member_key',
                    'admission_date': 'visit_start_raw',
                    'discharge_date': 'visit_end_raw',
                    'encounter_type': 'visit_type_label',
                    'encounter_status': 'visit_status_label',
                },
            },
            demo_config={'synthetic_helper_mode': 'Off'},
            active_control_values={'report_mode': 'Executive Summary'},
        )

        self.assertEqual(pipeline['semantic']['canonical_map'].get('patient_id'), 'member_key')
        self.assertEqual(pipeline['semantic']['manual_overrides_applied'].get('encounter_type'), 'visit_type_label')
        self.assertGreaterEqual(float(pipeline['readiness'].get('readiness_score', 0.0)), 0.45)

    def test_run_analysis_pipeline_supports_org_benchmark_pack_and_review_state(self):
        data = pd.DataFrame(
            [
                {'patient_id': 'P-001', 'admission_date': '2024-01-01', 'discharge_date': '2024-01-03', 'readmission': 1, 'length_of_stay': 2},
                {'patient_id': 'P-002', 'admission_date': '2024-01-04', 'discharge_date': '2024-01-06', 'readmission': 0, 'length_of_stay': 2},
            ]
        )
        pipeline = run_analysis_pipeline(
            data,
            'org-benchmark-pack.csv',
            {
                'source_mode': 'Uploaded dataset',
                'description': 'Org benchmark test.',
                'dataset_cache_key': 'org-benchmark-key',
            },
            active_control_values={
                'active_benchmark_pack_name': 'Pilot Benchmark Pack',
                'organization_benchmark_packs': {
                    'Pilot Benchmark Pack': {
                        'profile_family': 'organization-pack',
                        'rate_bands': {'Readmission rate': (0.0, 0.3), 'High-risk share': (0.0, 0.5)},
                        'numeric_bands': {'Average length of stay': (1.0, 5.0), 'Average cost': (100.0, 5000.0)},
                        'detail_note': 'Pilot pack.',
                    }
                },
                'dataset_review_approvals': {
                    'org-benchmark-key': {
                        'mapping_status': 'Approved',
                        'trust_gate_status': 'Approved',
                        'export_eligibility_status': 'Internal only',
                        'review_notes': 'Keep external use blocked.',
                        'reviewed_by_role': 'Data Steward',
                    }
                },
            },
        )

        accuracy = pipeline['result_accuracy_summary']
        self.assertEqual(accuracy['benchmark_profile']['profile_name'], 'Pilot Benchmark Pack')
        self.assertEqual(accuracy['approval_workflow']['mapping_status'], 'Approved')
        self.assertFalse(accuracy['metric_lineage_table'].empty)

    def test_finalize_runtime_pipeline_adds_runtime_views(self):
        pipeline = {
            'overview': {'rows': 2, 'columns': 10, 'analyzed_columns': 10},
            'quality': {'quality_score': 92.0},
            'readiness': {'readiness_score': 0.75, 'available_count': 4},
            'healthcare': {'healthcare_readiness_score': 0.7},
            'standards': {'badge_text': 'Moderate'},
            'remediation_context': {'synthetic_field_count': 0},
            'sample_info': {'sampling_applied': False},
            'dataset_intelligence': {'dataset_type_label': 'Healthcare dataset'},
            'use_case_detection': {'recommended_workflow': 'Healthcare Data Readiness'},
            'solution_packages': {'recommended_package_name': 'Core Review'},
        }

        final = finalize_runtime_pipeline(
            pipeline,
            dataset_name='unit-test-demo.csv',
            source_meta={'source_mode': 'Uploaded dataset', 'description': '', 'best_for': '', 'file_size_mb': 0.01},
            preflight={'warnings': []},
            column_validation={'warnings': []},
            job_runtime={'mode': 'sync'},
            heavy_task_catalog={'tasks': []},
            environment_checks={'checks': []},
            startup_readiness={'available': True},
            plan_awareness={'active_plan': 'Pro'},
            deployment_health_checks={'checks': []},
            performance_diagnostics={'cards': []},
            run_history=[],
            analysis_log=[],
            demo_config={},
        )

        self.assertIn('landing_summary', final)
        self.assertIn('audit_summary_bundle', final)
        self.assertEqual(final['source_meta']['source_mode'], 'Uploaded dataset')
        self.assertEqual(final['landing_summary']['startup_demo_flow'], {})

    def test_run_analysis_pipeline_can_cancel_between_stages(self):
        data = pd.DataFrame(
            [
                {
                    'patient_id': 'P-001',
                    'age': 67,
                    'gender': 'F',
                    'diagnosis': 'Heart Failure',
                    'admission_date': '2024-01-01',
                    'discharge_date': '2024-01-05',
                    'length_of_stay': 4,
                    'readmission_flag': 1,
                    'department': 'Cardiology',
                    'cost': 18200,
                },
            ]
        )
        with patch('src.pipeline.detect_structure', return_value=type('Structure', (), {'numeric_columns': [], 'identifier_columns': [], 'categorical_columns': [], 'default_date_column': 'admission_date'})()):
            with self.assertRaises(AnalysisCancelledError):
                run_analysis_pipeline(
                    data,
                    'unit-test-demo.csv',
                    {
                        'source_mode': 'Uploaded dataset',
                        'description': 'Unit test dataset.',
                        'best_for': 'Pipeline regression coverage.',
                        'file_size_mb': 0.01,
                    },
                    cancel_check=lambda: True,
                )

    def test_run_analysis_pipeline_falls_back_when_healthcare_analysis_times_out(self):
        data = pd.DataFrame(
            [
                {
                    'patient_id': 'P-001',
                    'age': 67,
                    'gender': 'F',
                    'diagnosis': 'Heart Failure',
                    'admission_date': '2024-01-01',
                    'discharge_date': '2024-01-05',
                    'length_of_stay': 4,
                    'readmission_flag': 1,
                    'department': 'Cardiology',
                    'cost': 18200,
                },
            ]
        )

        def _slow_healthcare(*args, **kwargs):
            time.sleep(0.05)
            return {'healthcare_readiness_score': 1.0}

        with patch('src.pipeline.run_healthcare_analysis', side_effect=_slow_healthcare):
            with patch('src.pipeline._healthcare_analysis_timeout_seconds', return_value=0.01):
                pipeline = run_analysis_pipeline(
                    data,
                    'unit-test-demo.csv',
                    {
                        'source_mode': 'Uploaded dataset',
                        'description': 'Unit test dataset.',
                        'best_for': 'Pipeline regression coverage.',
                        'file_size_mb': 0.01,
                    },
                    demo_config={'synthetic_helper_mode': 'Off'},
                    active_control_values={'report_mode': 'Executive Summary'},
                )

        self.assertTrue(pipeline['healthcare'].get('timeout_fallback'))
        self.assertIn('timed out', pipeline['healthcare'].get('timeout_reason', ''))
        self.assertTrue(pipeline['analysis_trust_summary'].get('timeout_fallback'))
        self.assertNotEqual(pipeline['analysis_trust_summary'].get('trust_level'), 'High')
        self.assertIn('timeout fallback', ' '.join(pipeline['analysis_trust_summary'].get('notes', [])).lower())


if __name__ == '__main__':
    unittest.main()
