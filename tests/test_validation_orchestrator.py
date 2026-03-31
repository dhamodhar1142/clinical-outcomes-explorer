from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

from src.validation_orchestrator import (
    build_validation_execution_plan,
    build_validation_runtime_profile,
    recommended_validation_task,
    run_recommended_validation,
)


class ValidationOrchestratorTests(unittest.TestCase):
    def test_recommended_validation_task_maps_known_labels(self):
        self.assertEqual(recommended_validation_task('Release validation'), 'release')
        self.assertEqual(recommended_validation_task('Accessibility / UI audit'), 'accessibility')
        self.assertEqual(recommended_validation_task('Unknown validation'), 'quick')

    @patch('src.validation_orchestrator._is_cloud_runtime', return_value=False)
    def test_runtime_profile_supports_heavy_validation_locally(self, _mock_runtime):
        profile = build_validation_runtime_profile()
        self.assertEqual(profile['runtime_label'], 'Local / workstation')
        self.assertTrue(profile['supports_heavy_validation'])

    @patch('src.validation_orchestrator._is_cloud_runtime', return_value=True)
    def test_execution_plan_gates_heavy_validations_in_cloud(self, _mock_runtime):
        plan = build_validation_execution_plan(
            pd.DataFrame(
                [
                    {'validation': 'Quick validation', 'priority': 'Baseline', 'why': 'Fast check.'},
                    {'validation': 'Release validation', 'priority': 'High', 'why': 'Heavy release gate.'},
                ]
            )
        )
        quick = next(row for row in plan if row['validation'] == 'Quick validation')
        release = next(row for row in plan if row['validation'] == 'Release validation')
        self.assertTrue(quick['allowed'])
        self.assertFalse(release['allowed'])

    @patch('src.validation_orchestrator.subprocess.run')
    def test_run_recommended_validation_parses_artifacts(self, mock_run):
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = (
            'Validation complete: PASS\n'
            'Artifacts: C:\\artifacts\\run-123\n'
            'Markdown report: C:\\artifacts\\run-123\\validation_report.md\n'
        )
        mock_run.return_value.stderr = ''

        result = run_recommended_validation('Quick validation')

        self.assertEqual(result['task'], 'quick')
        self.assertEqual(result['status'], 'Passed')
        self.assertEqual(result['artifact_path'], 'C:\\artifacts\\run-123')
        self.assertEqual(result['markdown_report'], 'C:\\artifacts\\run-123\\validation_report.md')

    @patch('src.validation_orchestrator._is_cloud_runtime', return_value=True)
    def test_run_recommended_validation_blocks_heavy_tasks_in_cloud(self, _mock_runtime):
        result = run_recommended_validation('Release validation')
        self.assertEqual(result['status'], 'Blocked')
        self.assertEqual(result['task'], 'release')


if __name__ == '__main__':
    unittest.main()
