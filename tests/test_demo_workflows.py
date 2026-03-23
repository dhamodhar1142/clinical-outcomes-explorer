from __future__ import annotations

import unittest

from src.data_loader import DEMO_DATASETS, load_demo_dataset
from src.pipeline import run_analysis_pipeline


class DemoWorkflowIntegrationTests(unittest.TestCase):
    def test_all_demo_datasets_complete_core_multi_tab_pipeline(self) -> None:
        for dataset_name in DEMO_DATASETS:
            with self.subTest(dataset_name=dataset_name):
                data, _ = load_demo_dataset(dataset_name)
                data.attrs.setdefault('dataset_cache_key', f'demo::{dataset_name}')
                pipeline = run_analysis_pipeline(
                    data,
                    dataset_name,
                    {
                        'source_mode': 'Demo dataset',
                        'description': DEMO_DATASETS[dataset_name]['description'],
                        'best_for': DEMO_DATASETS[dataset_name]['best_for'],
                        'file_size_mb': 0.1,
                    },
                    demo_config={'synthetic_helper_mode': 'Auto'},
                    active_control_values={'report_mode': 'Executive Summary'},
                )

                self.assertIn('overview', pipeline)
                self.assertIn('structure', pipeline)
                self.assertIn('field_profile', pipeline)
                self.assertIn('quality', pipeline)
                self.assertIn('readiness', pipeline)
                self.assertIn('healthcare', pipeline)
                self.assertIn('lineage', pipeline)
                self.assertIn('executive_report_pack', pipeline)
                self.assertIn('stakeholder_export_bundle', pipeline)
                self.assertIn('profile_cache_summary', pipeline)
                self.assertFalse(pipeline['field_profile'].empty)
                self.assertFalse(pipeline['structure'].detection_table.empty)
                self.assertIn('quality_score', pipeline['quality'])
                self.assertIn('readiness_score', pipeline['readiness'])


if __name__ == '__main__':
    unittest.main()
