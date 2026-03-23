from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest.mock import patch


try:
    from fastapi.testclient import TestClient
except ImportError:  # pragma: no cover
    TestClient = None

from src.api.app import create_api_app


CSV_BYTES = b"patient_id,admission_date,discharge_date,diagnosis_code\n1,2024-01-01,2024-01-04,I10\n2,2024-01-05,2024-01-08,E11\n"


@unittest.skipIf(TestClient is None, 'FastAPI test dependencies are not installed.')
class APIFoundationTests(unittest.TestCase):
    def _build_client(self, **extra_env: str) -> tuple[TestClient, tempfile.TemporaryDirectory[str]]:
        temp_dir = tempfile.TemporaryDirectory()
        env = {
            'SMART_DATASET_ANALYZER_API_KEYS': 'test-key|Integration Client|owner|api-test-workspace|API Test Workspace',
            'SMART_DATASET_ANALYZER_STORAGE_BACKEND': 'local',
            'SMART_DATASET_ANALYZER_STORAGE_ROOT': temp_dir.name,
            'SMART_DATASET_ANALYZER_SQLITE_PATH': os.path.join(temp_dir.name, 'api.sqlite3'),
        }
        env.update(extra_env)
        self.addCleanup(temp_dir.cleanup)
        patcher = patch.dict(os.environ, env, clear=False)
        patcher.start()
        self.addCleanup(patcher.stop)
        client = TestClient(create_api_app())
        return client, temp_dir

    def test_openapi_and_health_endpoints(self) -> None:
        client, _ = self._build_client()

        health = client.get('/api/health')
        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json()['status'], 'ok')

        schema = client.get('/api/openapi.json')
        self.assertEqual(schema.status_code, 200)
        paths = schema.json()['paths']
        self.assertIn('/api/v1/datasets/upload', paths)
        self.assertIn('/api/v1/analysis/runs', paths)

    def test_dataset_upload_and_analysis_results(self) -> None:
        client, _ = self._build_client()
        headers = {'X-API-Key': 'test-key'}

        upload = client.post(
            '/api/v1/datasets/upload',
            headers=headers,
            files={'file': ('api-upload.csv', CSV_BYTES, 'text/csv')},
        )
        self.assertEqual(upload.status_code, 200)
        upload_payload = upload.json()
        self.assertEqual(upload_payload['row_count'], 2)
        self.assertTrue(upload_payload['artifact']['stored'])

        analysis = client.post(
            '/api/v1/analysis/runs',
            headers=headers,
            json={
                'dataset_name': 'api-upload.csv',
                'artifact_path': upload_payload['artifact']['artifact_path'],
                'relative_path': upload_payload['artifact']['relative_path'],
            },
        )
        self.assertEqual(analysis.status_code, 200)
        payload = analysis.json()
        self.assertEqual(payload['status']['status'], 'completed')
        self.assertEqual(payload['result']['pipeline']['overview']['rows'], 2)
        self.assertIn('quality_score', payload['result']['pipeline']['quality'])

    def test_rate_limiting(self) -> None:
        client, _ = self._build_client(
            SMART_DATASET_ANALYZER_API_RATE_LIMIT_PER_MINUTE='2',
            SMART_DATASET_ANALYZER_API_RATE_LIMIT_WINDOW_SECONDS='60',
        )
        headers = {'X-API-Key': 'test-key'}

        first = client.get('/api/v1/datasets', headers=headers)
        second = client.get('/api/v1/datasets', headers=headers)
        third = client.get('/api/v1/datasets', headers=headers)

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertEqual(third.status_code, 429)

    def test_webhook_callback_for_background_analysis(self) -> None:
        received: list[dict[str, object]] = []

        class _Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get('Content-Length', '0'))
                payload = json.loads(self.rfile.read(length) or b'{}')
                received.append(payload)
                self.send_response(200)
                self.end_headers()

            def log_message(self, format: str, *args) -> None:  # noqa: A003
                return

        server = HTTPServer(('127.0.0.1', 0), _Handler)
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        self.addCleanup(server.shutdown)
        self.addCleanup(server.server_close)

        client, _ = self._build_client(
            SMART_DATASET_ANALYZER_JOB_BACKEND='thread',
            SMART_DATASET_ANALYZER_JOB_MAX_WORKERS='1',
        )
        headers = {'X-API-Key': 'test-key'}
        webhook_url = f'http://127.0.0.1:{server.server_port}/callback'

        response = client.post(
            '/api/v1/analysis/runs/upload',
            headers=headers,
            data={'webhook_url': webhook_url},
            files={'file': ('api-upload.csv', CSV_BYTES, 'text/csv')},
        )
        self.assertIn(response.status_code, {200, 202})

        deadline = time.time() + 10
        while time.time() < deadline and not received:
            time.sleep(0.2)

        self.assertTrue(received, 'Expected webhook callback payload for the background analysis run.')
        self.assertIn(received[0]['status']['status'], {'completed', 'failed', 'cancelled'})


if __name__ == '__main__':
    unittest.main()
