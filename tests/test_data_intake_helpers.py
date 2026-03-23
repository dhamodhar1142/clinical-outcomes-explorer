from __future__ import annotations

import unittest

import pandas as pd

from src.data_intake_support import (
    apply_auto_mapping_suggestions,
    apply_mapping_template,
    build_data_profiling_summary,
    build_mapping_confidence_table,
    build_mapping_template,
    build_remap_board,
)
from src.profiler import analysis_sample_info, build_field_profile
from src.semantic_mapper import infer_semantic_mapping
from src.schema_detection import detect_structure


class DataIntakeHelperTests(unittest.TestCase):
    def test_mapping_confidence_table_and_remap_board_build_from_pipeline(self) -> None:
        pipeline = {
            'semantic': {
                'mapping_table': pd.DataFrame(
                    [
                        {
                            'original_column': 'admission_date',
                            'semantic_label': 'event_date',
                            'confidence_score': 0.91,
                            'confidence_label': 'High',
                            'used_downstream': True,
                        },
                        {
                            'original_column': 'department',
                            'semantic_label': 'department',
                            'confidence_score': 0.67,
                            'confidence_label': 'Medium',
                            'used_downstream': True,
                        },
                    ]
                ),
            },
            'field_profile': pd.DataFrame(
                [
                    {'column_name': 'admission_date', 'inferred_type': 'datetime'},
                    {'column_name': 'department', 'inferred_type': 'categorical'},
                    {'column_name': 'notes', 'inferred_type': 'text'},
                ]
            ),
        }

        confidence_table = build_mapping_confidence_table(pipeline)
        remap_board = build_remap_board(pipeline)

        self.assertEqual(len(confidence_table), 2)
        self.assertIn('confidence_score', confidence_table.columns)
        self.assertEqual(remap_board.iloc[0]['source_column'], 'admission_date')
        self.assertIn('mapped_field', remap_board.columns)
        self.assertIn('top_suggestions', confidence_table.columns)
        self.assertIn('suggested_field', remap_board.columns)

    def test_data_profiling_summary_handles_large_dataset_sampling(self) -> None:
        rows = 120000
        data = pd.DataFrame(
            {
                'patient_id': [f'P{index:06d}' for index in range(rows)],
                'cost_amount': [float(index % 5000) for index in range(rows)],
                'department': ['ER' if index % 2 == 0 else 'ICU' for index in range(rows)],
                'event_date': pd.date_range('2025-01-01', periods=rows, freq='h').astype(str),
            }
        )
        structure = detect_structure(data)
        field_profile = build_field_profile(data, structure)
        sample_info = analysis_sample_info(data)
        pipeline = {
            'overview': {'rows': rows},
            'field_profile': field_profile,
            'sample_info': sample_info,
        }

        profiling_summary = build_data_profiling_summary(pipeline)

        self.assertEqual(profiling_summary['summary_cards'][0]['value'], '120,000')
        self.assertEqual(profiling_summary['summary_cards'][3]['value'], 'Yes')
        self.assertFalse(profiling_summary['field_profile_preview'].empty)

    def test_auto_mapping_suggestions_and_templates_apply_to_board(self) -> None:
        board = pd.DataFrame(
            [
                {
                    'display_order': 1,
                    'source_column': 'admit_dt',
                    'mapped_field': 'Not mapped',
                    'confidence_score': 0.22,
                    'confidence_label': 'Low',
                    'inferred_type': 'datetime',
                    'suggested_field': 'admission_date',
                    'suggested_confidence': 0.88,
                },
                {
                    'display_order': 2,
                    'source_column': 'provider_npi',
                    'mapped_field': 'provider_id',
                    'confidence_score': 0.84,
                    'confidence_label': 'High',
                    'inferred_type': 'identifier',
                    'suggested_field': 'provider_id',
                    'suggested_confidence': 0.84,
                },
            ]
        )

        auto_applied = apply_auto_mapping_suggestions(board)
        self.assertEqual(auto_applied.iloc[0]['mapped_field'], 'admission_date')
        self.assertEqual(auto_applied.iloc[0]['confidence_label'], 'High')

        template = build_mapping_template(auto_applied, template_name='Healthcare template', dataset_type='Healthcare')
        new_board = pd.DataFrame(
            [
                {
                    'display_order': 1,
                    'source_column': 'admit_dt',
                    'mapped_field': 'Not mapped',
                    'confidence_score': 0.10,
                    'confidence_label': 'Low',
                    'inferred_type': 'datetime',
                    'suggested_field': 'admission_date',
                    'suggested_confidence': 0.88,
                }
            ]
        )
        templated = apply_mapping_template(new_board, template)
        self.assertEqual(templated.iloc[0]['mapped_field'], 'admission_date')

    def test_healthcare_semantic_mapping_adds_fuzzy_suggestions(self) -> None:
        data = pd.DataFrame(
            {
                'mrn': ['A1', 'A2'],
                'provider_npi': ['1111111111', '2222222222'],
                'admit_dt': ['2025-01-01', '2025-01-02'],
                'discharge_dt': ['2025-01-05', '2025-01-06'],
                'icd10_dx_code': ['I50.9', 'J44.1'],
                'cpt_code': ['99213', '99214'],
            }
        )
        structure = detect_structure(data)
        semantic = infer_semantic_mapping(data, structure)

        self.assertEqual(semantic['canonical_map'].get('patient_id'), 'mrn')
        self.assertEqual(semantic['canonical_map'].get('provider_id'), 'provider_npi')
        self.assertEqual(semantic['canonical_map'].get('diagnosis_code'), 'icd10_dx_code')
        self.assertFalse(semantic['suggestion_table'].empty)
        self.assertIn('auto_apply', semantic['suggestion_table'].columns)

    def test_generic_dataset_semantic_mapping_stays_generic_safe(self) -> None:
        data = pd.DataFrame(
            {
                'customer_id': ['C1', 'C2'],
                'invoice_date': ['2025-01-01', '2025-01-02'],
                'revenue': [100.0, 120.0],
            }
        )
        structure = detect_structure(data)
        semantic = infer_semantic_mapping(data, structure)

        self.assertIn('suggestion_table', semantic)
        self.assertTrue(all(field != 'diagnosis_code' for field in semantic['canonical_map']))


if __name__ == '__main__':
    unittest.main()
