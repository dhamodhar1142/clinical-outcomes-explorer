from __future__ import annotations

import unittest

import pandas as pd

from src.readiness_engine import evaluate_analysis_readiness
from src.schema_detection import detect_structure
from src.semantic_mapper import infer_semantic_mapping
from src.temporal_detection import augment_temporal_fields


class TemporalDetectionTests(unittest.TestCase):
    def test_year_only_dataset_gets_synthetic_event_date(self) -> None:
        data = pd.DataFrame(
            {
                'patient_id': [f'P{i:03d}' for i in range(30)],
                'year': [2018 + (i % 5) for i in range(30)],
                'cost_amount': [100 + i for i in range(30)],
            }
        )

        augmented, temporal_context = augment_temporal_fields(data)

        self.assertTrue(temporal_context.get('synthetic_date_created'))
        self.assertIn('event_date', augmented.columns)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(augmented['event_date']))
        self.assertTrue(augmented['event_date'].notna().any())

    def test_synthetic_event_date_unlocks_trend_readiness(self) -> None:
        data = pd.DataFrame(
            {
                'patient_id': [f'P{i:03d}' for i in range(30)],
                'year': [2020 + (i % 3) for i in range(30)],
                'readmission': [1 if i % 4 == 0 else 0 for i in range(30)],
            }
        )

        augmented, _ = augment_temporal_fields(data)
        structure = detect_structure(augmented)
        semantic = infer_semantic_mapping(augmented, structure)
        readiness = evaluate_analysis_readiness(semantic['canonical_map'])

        readiness_table = readiness['readiness_table']
        trend_row = readiness_table[readiness_table['analysis_module'] == 'Trend Analysis'].iloc[0]
        readmission_row = readiness_table[readiness_table['analysis_module'] == 'Readmission-Style Analysis'].iloc[0]

        self.assertEqual(trend_row['status'], 'Available')
        self.assertIn(readmission_row['status'], {'Available', 'Partial'})


if __name__ == '__main__':
    unittest.main()
