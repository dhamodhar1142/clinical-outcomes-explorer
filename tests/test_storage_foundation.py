from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.storage import build_storage_backend_health, build_storage_service


class StorageFoundationTests(unittest.TestCase):
    def test_storage_service_falls_back_when_backend_is_not_supported(self) -> None:
        service = build_storage_service(backend='ftp', storage_root='')
        self.assertFalse(service.enabled)
        self.assertEqual(service.status.mode, 'session')
        self.assertEqual(service.status.storage_target, 'session-only')

    def test_storage_service_falls_back_when_s3_bucket_is_missing(self) -> None:
        service = build_storage_service(backend='s3', storage_root='')
        self.assertFalse(service.enabled)
        self.assertEqual(service.status.mode, 'session')
        self.assertEqual(service.status.storage_target, 'session-only')

    def test_local_storage_service_saves_dataset_report_and_session_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            service = build_storage_service(storage_root=tmpdir)
            workspace_identity = {'workspace_id': 'Care Ops Workspace'}

            dataset_artifact = service.save_dataset_upload(
                workspace_identity,
                dataset_name='Admissions March',
                file_name='admissions.csv',
                payload=b'patient_id,readmission_flag\n1,1\n',
            )
            report_artifact = service.save_report_artifact(
                workspace_identity,
                dataset_name='Admissions March',
                report_type='Executive Report',
                file_name='executive_report.txt',
                payload='Executive summary ready.',
            )
            session_artifact = service.save_session_bundle(
                workspace_identity,
                dataset_name='Admissions March',
                file_name='smart_dataset_analyzer_session.json',
                payload='{"bundle_version": 1}',
            )

            self.assertTrue(dataset_artifact['stored'])
            self.assertTrue(report_artifact['stored'])
            self.assertTrue(session_artifact['stored'])
            self.assertTrue(Path(dataset_artifact['artifact_path']).exists())
            self.assertTrue(Path(report_artifact['artifact_path']).exists())
            self.assertTrue(Path(session_artifact['artifact_path']).exists())
            self.assertIn('/uploads/', dataset_artifact['relative_path'])
            self.assertIn('/reports/', report_artifact['relative_path'])
            self.assertIn('/session_bundles/', session_artifact['relative_path'])

    def test_storage_backend_health_reports_local_storage_ready(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            service = build_storage_service(storage_root=tmpdir)
            health = build_storage_backend_health(service)
        self.assertEqual(health['status'], 'Healthy')
        self.assertEqual(health['mode'], 'local')

    def test_s3_storage_service_saves_dataset_report_and_session_artifacts(self) -> None:
        class _FakeClient:
            def __init__(self) -> None:
                self.calls = []

            def put_object(self, **kwargs):
                self.calls.append(kwargs)

        class _FakeSession:
            def __init__(self, client) -> None:
                self._client = client

            def client(self, service_name, **kwargs):
                return self._client

        fake_client = _FakeClient()
        fake_boto3 = type(
            '_FakeBoto3',
            (),
            {'session': type('_FakeSessionModule', (), {'Session': lambda self=None: _FakeSession(fake_client)})},
        )()

        with patch('src.storage.import_module', return_value=fake_boto3):
            with patch.dict(
                'os.environ',
                {
                    'SMART_DATASET_ANALYZER_STORAGE_BUCKET': 'smart-dataset-analyzer',
                    'SMART_DATASET_ANALYZER_STORAGE_PREFIX': 'pilot-artifacts',
                    'SMART_DATASET_ANALYZER_STORAGE_ENDPOINT_URL': 'https://minio.local',
                },
                clear=False,
            ):
                service = build_storage_service(backend='s3')

        workspace_identity = {'workspace_id': 'Care Ops Workspace'}
        dataset_artifact = service.save_dataset_upload(
            workspace_identity,
            dataset_name='Admissions March',
            file_name='admissions.csv',
            payload=b'patient_id,readmission_flag\n1,1\n',
        )
        report_artifact = service.save_report_artifact(
            workspace_identity,
            dataset_name='Admissions March',
            report_type='Executive Report',
            file_name='executive_report.txt',
            payload='Executive summary ready.',
        )
        session_artifact = service.save_session_bundle(
            workspace_identity,
            dataset_name='Admissions March',
            file_name='smart_dataset_analyzer_session.json',
            payload='{"bundle_version": 1}',
        )

        self.assertTrue(service.enabled)
        self.assertEqual(service.status.mode, 's3')
        self.assertTrue(dataset_artifact['stored'])
        self.assertTrue(report_artifact['stored'])
        self.assertTrue(session_artifact['stored'])
        self.assertEqual(len(fake_client.calls), 3)
        self.assertTrue(dataset_artifact['artifact_path'].startswith('s3://smart-dataset-analyzer/'))
        self.assertIn('pilot-artifacts/', dataset_artifact['object_key'])
        self.assertIn('/uploads/', dataset_artifact['relative_path'])

    def test_storage_backend_health_reports_s3_reachability(self) -> None:
        class _FakeClient:
            def head_bucket(self, **kwargs):
                return {'ResponseMetadata': {'HTTPStatusCode': 200}}

            def put_object(self, **kwargs):
                return None

        class _FakeSession:
            def __init__(self, client) -> None:
                self._client = client

            def client(self, service_name, **kwargs):
                return self._client

        fake_client = _FakeClient()
        fake_boto3 = type(
            '_FakeBoto3',
            (),
            {'session': type('_FakeSessionModule', (), {'Session': lambda self=None: _FakeSession(fake_client)})},
        )()

        with patch('src.storage.import_module', return_value=fake_boto3):
            with patch.dict(
                'os.environ',
                {
                    'SMART_DATASET_ANALYZER_STORAGE_BUCKET': 'smart-dataset-analyzer',
                    'SMART_DATASET_ANALYZER_STORAGE_PREFIX': 'pilot-artifacts',
                },
                clear=False,
            ):
                service = build_storage_service(backend='s3')
        health = build_storage_backend_health(service)
        self.assertEqual(health['status'], 'Healthy')
        self.assertEqual(health['mode'], 's3')

    def test_storage_backend_health_reports_session_fallback(self) -> None:
        service = build_storage_service(backend='s3', storage_root='')
        health = build_storage_backend_health(service)
        self.assertEqual(health['status'], 'Session fallback')


if __name__ == '__main__':
    unittest.main()
