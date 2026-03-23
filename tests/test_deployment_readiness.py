from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from src.deployment_readiness import (
    build_config_guidance,
    build_environment_checks,
    build_launch_checklist,
    build_startup_readiness_summary,
)


class DeploymentReadinessTests(unittest.TestCase):
    def test_environment_checks_return_core_rows(self) -> None:
        result = build_environment_checks()
        table = result['checks_table']
        self.assertFalse(table.empty)
        self.assertIn('Streamlit entrypoint', table['check_area'].tolist())
        self.assertIn('Core dependencies manifest', table['check_area'].tolist())
        self.assertIn('Optional dependencies manifest', table['check_area'].tolist())
        self.assertIn('Development and test manifest', table['check_area'].tolist())
        self.assertIn('Persistence configuration', table['check_area'].tolist())
        self.assertIn('PostgreSQL persistence support', table['check_area'].tolist())
        self.assertIn('Artifact storage backend', table['check_area'].tolist())
        self.assertIn('Artifact storage health', table['check_area'].tolist())
        self.assertIn('Background job backend', table['check_area'].tolist())
        self.assertIn('Optional OpenAI configuration', table['check_area'].tolist())
        self.assertIn('Optional browser smoke tooling', table['check_area'].tolist())

    def test_environment_checks_report_postgres_and_worker_configuration(self) -> None:
        with patch.dict(
            os.environ,
            {
                'SMART_DATASET_ANALYZER_ENV': 'production',
                'SMART_DATASET_ANALYZER_DB_URL': 'postgresql://demo:secret@localhost:5432/smart_dataset',
                'SMART_DATASET_ANALYZER_JOB_BACKEND': 'worker',
                'SMART_DATASET_ANALYZER_JOB_MAX_WORKERS': '4',
                'SMART_DATASET_ANALYZER_SECRETS_SOURCE': 'secret-manager',
            },
            clear=False,
        ), patch('src.deployment_readiness.find_spec', side_effect=lambda name: object() if name == 'psycopg' else None):
            result = build_environment_checks()

        table = result['checks_table']
        env_row = table.loc[table['check_area'] == 'Deployment environment'].iloc[0]
        profile_row = table.loc[table['check_area'] == 'Deployment profile fit'].iloc[0]
        postgres_row = table.loc[table['check_area'] == 'PostgreSQL persistence support'].iloc[0]
        jobs_row = table.loc[table['check_area'] == 'Background job backend'].iloc[0]
        job_health_row = table.loc[table['check_area'] == 'Background job health'].iloc[0]
        secrets_row = table.loc[table['check_area'] == 'Secrets and config handling'].iloc[0]
        self.assertEqual(env_row['status'], 'Production')
        self.assertEqual(profile_row['status'], 'Ready')
        self.assertEqual(postgres_row['status'], 'Ready')
        self.assertIn('PostgreSQL URL', table.loc[table['check_area'] == 'Persistence configuration'].iloc[0]['status'])
        self.assertEqual(jobs_row['status'], 'Worker enabled')
        self.assertEqual(job_health_row['status'], 'Healthy')
        self.assertIn('4 configured worker slot', jobs_row['detail'])
        self.assertEqual(secrets_row['status'], 'Ready')

    def test_environment_checks_report_external_worker_health(self) -> None:
        with patch.dict(
            os.environ,
            {
                'SMART_DATASET_ANALYZER_ENV': 'staging',
                'SMART_DATASET_ANALYZER_JOB_BACKEND': 'rq',
                'SMART_DATASET_ANALYZER_JOB_QUEUE_URL': 'redis://localhost:6379/0',
                'SMART_DATASET_ANALYZER_JOB_QUEUE_NAME': 'smart-dataset-analyzer',
            },
            clear=False,
        ), patch(
            'src.deployment_readiness.build_job_backend_health',
            return_value={
                'status': 'Healthy',
                'detail': 'External queue "smart-dataset-analyzer" is reachable and currently has 1 queued job(s).',
            },
        ):
            result = build_environment_checks()

        table = result['checks_table']
        jobs_row = table.loc[table['check_area'] == 'Background job backend'].iloc[0]
        job_health_row = table.loc[table['check_area'] == 'Background job health'].iloc[0]
        self.assertEqual(jobs_row['status'], 'External worker enabled')
        self.assertEqual(job_health_row['status'], 'Healthy')

    def test_environment_checks_report_object_storage_health(self) -> None:
        fake_storage_service = type(
            '_FakeStorageService',
            (),
            {
                'status': type(
                    '_FakeStorageStatus',
                    (),
                    {
                        'mode': 's3',
                        'storage_target': 's3://smart-dataset-analyzer/pilot-artifacts',
                    },
                )(),
            },
        )()
        with patch.dict(
            os.environ,
            {
                'SMART_DATASET_ANALYZER_STORAGE_BACKEND': 's3',
                'SMART_DATASET_ANALYZER_STORAGE_BUCKET': 'smart-dataset-analyzer',
                'SMART_DATASET_ANALYZER_STORAGE_PREFIX': 'pilot-artifacts',
            },
            clear=False,
        ), patch(
            'src.deployment_readiness.build_storage_service',
            return_value=fake_storage_service,
        ), patch(
            'src.deployment_readiness.build_storage_backend_health',
            return_value={
                'status': 'Healthy',
                'detail': 'Object storage bucket smart-dataset-analyzer is reachable for artifact persistence.',
            },
        ):
            result = build_environment_checks()

        table = result['checks_table']
        storage_row = table.loc[table['check_area'] == 'Artifact storage backend'].iloc[0]
        storage_health_row = table.loc[table['check_area'] == 'Artifact storage health'].iloc[0]
        self.assertEqual(storage_row['status'], 'Object storage enabled')
        self.assertEqual(storage_health_row['status'], 'Healthy')

    def test_environment_checks_flag_production_without_persistent_backend(self) -> None:
        with patch.dict(
            os.environ,
            {
                'SMART_DATASET_ANALYZER_ENV': 'production',
                'SMART_DATASET_ANALYZER_DB_URL': '',
                'SMART_DATASET_ANALYZER_SQLITE_PATH': '',
            },
            clear=False,
        ):
            result = build_environment_checks()

        table = result['checks_table']
        profile_row = table.loc[table['check_area'] == 'Deployment profile fit'].iloc[0]
        secrets_row = table.loc[table['check_area'] == 'Secrets and config handling'].iloc[0]
        self.assertEqual(profile_row['status'], 'Warning')
        self.assertEqual(secrets_row['status'], 'Review')

    def test_startup_readiness_summary_reports_sampling(self) -> None:
        summary = build_startup_readiness_summary(
            {'blocked': False, 'warnings': []},
            {'warnings': ['Unnamed columns']},
            {'source_mode': 'Demo dataset', 'file_size_mb': 12.0},
            {'sampling_applied': True, 'profile_sample_rows': 20000, 'quality_sample_rows': 25000},
        )
        self.assertEqual(len(summary['summary_cards']), 4)
        self.assertTrue(any('Sampling is active' in note for note in summary['notes']))

    def test_launch_checklist_and_config_guidance_are_available(self) -> None:
        checklist = build_launch_checklist()
        guidance = build_config_guidance()
        self.assertFalse(checklist.empty)
        self.assertFalse(guidance.empty)
        self.assertIn('Select deployment environment profile', checklist['launch_step'].tolist())
        self.assertIn('Verify startup command', checklist['launch_step'].tolist())
        self.assertIn('Confirm optional package intent', checklist['launch_step'].tolist())
        self.assertIn('Prepare local validation environment', checklist['launch_step'].tolist())
        self.assertIn('Select persistence backend', checklist['launch_step'].tolist())
        self.assertIn('Select artifact storage backend', checklist['launch_step'].tolist())
        self.assertIn('Confirm worker backend intent', checklist['launch_step'].tolist())
        self.assertIn('Validate artifact storage health', checklist['launch_step'].tolist())
        self.assertIn('Require CI validation', checklist['launch_step'].tolist())
        self.assertIn('Gate releases on browser smoke', checklist['launch_step'].tolist())
        self.assertIn('Deployment environment', guidance['config_area'].tolist())
        self.assertIn('Python runtime', guidance['config_area'].tolist())
        self.assertIn('Dependency tiers', guidance['config_area'].tolist())
        self.assertIn('Optional package intent', guidance['config_area'].tolist())
        self.assertIn('Secrets handling', guidance['config_area'].tolist())
        self.assertIn('Persistence backend', guidance['config_area'].tolist())
        self.assertIn('Artifact storage', guidance['config_area'].tolist())
        self.assertIn('Object storage credentials', guidance['config_area'].tolist())
        self.assertIn('Background jobs', guidance['config_area'].tolist())
        self.assertIn('CI / release safety', guidance['config_area'].tolist())


if __name__ == '__main__':
    unittest.main()
