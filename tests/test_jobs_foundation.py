from __future__ import annotations

import os
import time
import unittest
from unittest.mock import patch

from src.jobs import (
    JOB_BACKEND_ENV,
    JOB_HEALTHCHECK_TIMEOUT_ENV,
    JOB_QUEUE_URL_ENV,
    build_job_backend_health,
    build_heavy_task_catalog,
    build_job_runtime,
    build_job_status_view,
    cancel_job,
    force_cancel_job,
    get_job_result,
    get_job_status,
    run_managed_job,
    submit_job,
)


class JobsFoundationTests(unittest.TestCase):
    def test_job_runtime_defaults_to_synchronous_fallback(self) -> None:
        runtime = build_job_runtime()
        self.assertIn('mode', runtime)
        self.assertIn('status_label', runtime)
        self.assertFalse(runtime['backend_configured'])

    def test_heavy_task_catalog_lists_expected_operations(self) -> None:
        catalog = build_heavy_task_catalog()
        self.assertFalse(catalog.empty)
        self.assertIn('analysis_pipeline', catalog['task_key'].tolist())
        self.assertIn('predictive_modeling', catalog['task_key'].tolist())
        self.assertIn('report_generation', catalog['task_key'].tolist())

    def test_run_managed_job_tracks_completed_run_in_session_state(self) -> None:
        session_state: dict[str, object] = {'job_runs': []}
        runtime = {'backend_configured': False, 'mode': 'sync'}
        updates: list[tuple[float, str]] = []

        result = run_managed_job(
            session_state,
            runtime,
            task_key='report_generation',
            task_label='Executive Report',
            detail='Generated executive report.',
            stage_messages=['Starting...', 'Finishing...'],
            progress_callback=lambda value, message: updates.append((value, message)),
            runner=lambda: b'report-bytes',
        )

        self.assertEqual(result['result'], b'report-bytes')
        self.assertEqual(session_state['job_runs'][0]['status'], 'completed')
        self.assertEqual(session_state['job_runs'][0]['task_key'], 'report_generation')
        self.assertTrue(updates)

    def test_submit_job_stores_status_and_result(self) -> None:
        session_state: dict[str, object] = {'job_runs': []}
        runtime = {'backend_configured': False, 'mode': 'sync'}

        submission = submit_job(
            session_state,
            runtime,
            task_key='predictive_modeling',
            task_label='Predictive modeling comparison',
            detail='Prepared comparison-ready candidate models.',
            stage_messages=['Starting...', 'Done...'],
            runner=lambda: {'available': True},
        )

        status = get_job_status(session_state, submission['job_id'])
        result = get_job_result(session_state, submission['job_id'])

        self.assertEqual(submission['status'], 'completed')
        self.assertEqual(status['status'], 'completed')
        self.assertEqual(result, {'available': True})
        self.assertIn('created_at', status)
        self.assertIn('completed_at', status)

    def test_submit_job_records_queued_then_completed_for_worker_backend(self) -> None:
        session_state: dict[str, object] = {'job_runs': []}
        runtime = {'backend_configured': True, 'mode': 'worker', 'max_workers': 1}
        updates: list[tuple[float, str]] = []

        submission = submit_job(
            session_state,
            runtime,
            task_key='export_bundle',
            task_label='Bundle export',
            detail='Prepared export bundle.',
            stage_messages=['Queued...', 'Running...', 'Done...'],
            progress_callback=lambda value, message: updates.append((value, message)),
            runner=lambda: b'bundle',
        )

        self.assertEqual(submission['status'], 'queued')
        status = get_job_status(session_state, submission['job_id'])
        for _ in range(50):
            if status['status'] == 'completed':
                break
            time.sleep(0.02)
            status = get_job_status(session_state, submission['job_id'])
        self.assertEqual(status['status'], 'completed')
        self.assertEqual(get_job_result(session_state, submission['job_id']), b'bundle')
        self.assertTrue(status['backend_ready'])
        self.assertTrue(updates)
        self.assertIn('queued', updates[0][1].lower())

    def test_job_runtime_supports_external_rq_backend(self) -> None:
        with patch.dict(os.environ, {JOB_BACKEND_ENV: 'rq', JOB_QUEUE_URL_ENV: 'redis://localhost:6379/0'}, clear=False):
            runtime = build_job_runtime()
        self.assertTrue(runtime['backend_configured'])
        self.assertEqual(runtime['mode'], 'external')
        self.assertEqual(runtime['external_backend'], 'rq')

    def test_job_backend_health_reports_threaded_backend(self) -> None:
        health = build_job_backend_health({'backend_configured': True, 'mode': 'worker', 'max_workers': 3})
        self.assertEqual(health['status'], 'Healthy')
        self.assertEqual(health['backend'], 'threadpool')
        self.assertEqual(health['max_workers'], 3)

    def test_job_backend_health_reports_external_backend_reachability(self) -> None:
        class _FakeRedis:
            def ping(self):
                return True

        class _FakeQueue:
            count = 2

        class _FakeAdapter:
            def __init__(self) -> None:
                self._redis_connection = _FakeRedis()
                self._queue_cls = lambda name, connection=None: _FakeQueue()

        runtime = {
            'backend_configured': True,
            'mode': 'external',
            'external_backend': 'rq',
            'queue_url': 'redis://localhost:6379/0',
            'queue_name': 'smart-dataset-analyzer',
        }
        with patch('src.jobs._get_external_rq_adapter', return_value=_FakeAdapter()):
            health = build_job_backend_health(runtime)
        self.assertEqual(health['status'], 'Healthy')
        self.assertEqual(health['pending_jobs'], 2)

    def test_job_backend_health_reports_external_backend_unavailable(self) -> None:
        runtime = {
            'backend_configured': True,
            'mode': 'external',
            'external_backend': 'rq',
            'queue_url': 'redis://localhost:6379/0',
            'queue_name': 'smart-dataset-analyzer',
        }
        with patch('src.jobs._get_external_rq_adapter', side_effect=RuntimeError('queue unreachable')):
            health = build_job_backend_health(runtime)
        self.assertEqual(health['status'], 'Unavailable')
        self.assertIn('queue unreachable', health['detail'])

    def test_job_runtime_carries_healthcheck_timeout(self) -> None:
        with patch.dict(os.environ, {JOB_HEALTHCHECK_TIMEOUT_ENV: '3.5'}, clear=False):
            runtime = build_job_runtime()
        self.assertEqual(runtime['healthcheck_timeout_seconds'], 3.5)

    def test_submit_job_tracks_external_backend_completion(self) -> None:
        class _FakeExternalJob:
            def __init__(self) -> None:
                self.result = b'external-report'

            def get_status(self, refresh=True):
                return 'finished'

        class _FakeAdapter:
            def enqueue_registered_task(self, task_name, task_kwargs):
                return {
                    'external_job_id': 'external-1',
                    'queue_url': 'redis://localhost:6379/0',
                    'queue_name': 'smart-dataset-analyzer',
                }

            def fetch(self, external_job_id):
                return _FakeExternalJob()

            def cancel(self, external_job_id):
                return True

        session_state: dict[str, object] = {'job_runs': []}
        runtime = {
            'backend_configured': True,
            'mode': 'external',
            'external_backend': 'rq',
            'queue_url': 'redis://localhost:6379/0',
            'queue_name': 'smart-dataset-analyzer',
        }

        with patch('src.jobs._get_external_rq_adapter', return_value=_FakeAdapter()):
            submission = submit_job(
                session_state,
                runtime,
                task_key='report_generation',
                task_label='Executive Report',
                detail='Generated report.',
                task_name='report_generation_text',
                task_kwargs={
                    'report_label': 'Executive Report',
                    'dataset_name': 'demo.csv',
                    'overview': {'rows': 1, 'columns': 1, 'duplicate_rows': 0, 'missing_values': 0},
                    'quality': {'quality_score': 1.0},
                    'readiness': {'readiness_score': 1.0},
                    'healthcare': {'healthcare_readiness_score': 1.0},
                    'insights': {'summary_lines': []},
                    'action_recommendations': None,
                },
                runner=lambda: b'fallback-report',
            )
            status = get_job_status(session_state, submission['job_id'])
            result = get_job_result(session_state, submission['job_id'])

        self.assertEqual(submission['status'], 'queued')
        self.assertEqual(status['status'], 'completed')
        self.assertEqual(result, b'external-report')
        self.assertEqual(status['mode'], 'external')

    def test_cancel_job_cancels_external_backend_job(self) -> None:
        class _QueuedExternalJob:
            result = None

            def get_status(self, refresh=True):
                return 'queued'

        class _FakeAdapter:
            def fetch(self, external_job_id):
                return _QueuedExternalJob()

            def cancel(self, external_job_id):
                return True

        session_state: dict[str, object] = {
            'job_store': {
                'job-1': {
                    'job_id': 'job-1',
                    'task_key': 'report_generation',
                    'task_label': 'Executive Report',
                    'status': 'queued',
                    'mode': 'external',
                    'detail': 'Queued external report.',
                    'duration_seconds': 0.0,
                    'result': None,
                    'cancel_requested': False,
                    'external_backend': 'rq',
                    'external_job_id': 'external-1',
                    'queue_url': 'redis://localhost:6379/0',
                    'queue_name': 'smart-dataset-analyzer',
                }
            },
            'job_runs': [],
        }

        with patch('src.jobs._build_external_adapter_from_entry', return_value=_FakeAdapter()):
            cancelled = cancel_job(session_state, 'job-1')

        self.assertEqual(cancelled['status'], 'cancelled')
        self.assertTrue(cancelled['cancel_requested'])

    def test_cancel_job_marks_queued_job_cancelled(self) -> None:
        session_state: dict[str, object] = {}
        job_id = 'report-queued'
        session_state['job_store'] = {
            job_id: {
                'job_id': job_id,
                'task_key': 'report_generation',
                'task_label': 'Executive Report',
                'status': 'queued',
                'mode': 'worker',
                'detail': 'Queued report.',
                'duration_seconds': 0.0,
                'result': None,
                'cancel_requested': False,
            }
        }
        session_state['job_runs'] = []

        cancelled = cancel_job(session_state, job_id)

        self.assertEqual(cancelled['status'], 'cancelled')
        self.assertTrue(cancelled['cancel_requested'])
        self.assertEqual(session_state['job_runs'][0]['status'], 'cancelled')

    def test_cancel_job_cancels_background_future_before_execution(self) -> None:
        blocker_started = False
        blocker_release = False

        def blocking_runner() -> str:
            nonlocal blocker_started
            blocker_started = True
            while not blocker_release:
                time.sleep(0.01)
            return 'done'

        session_state: dict[str, object] = {'job_runs': []}
        runtime = {'backend_configured': True, 'mode': 'worker', 'max_workers': 1}

        first_job = submit_job(
            session_state,
            runtime,
            task_key='analysis_pipeline',
            task_label='Analysis pipeline',
            detail='Running blocker.',
            runner=blocking_runner,
        )
        for _ in range(50):
            status = get_job_status(session_state, first_job['job_id'])
            if status['status'] == 'running' or blocker_started:
                break
            time.sleep(0.02)

        second_job = submit_job(
            session_state,
            runtime,
            task_key='report_generation',
            task_label='Executive Report',
            detail='Queued report.',
            runner=lambda: 'report',
        )
        cancelled = cancel_job(session_state, second_job['job_id'])
        blocker_release = True

        self.assertEqual(cancelled['status'], 'cancelled')
        self.assertTrue(cancelled['cancel_requested'])

    def test_force_cancel_job_marks_running_job_cancelled_and_ignores_late_completion(self) -> None:
        session_state: dict[str, object] = {
            'job_store': {
                'job-1': {
                    'job_id': 'job-1',
                    'task_key': 'analysis_pipeline',
                    'task_label': 'Large dataset analysis',
                    'status': 'running',
                    'mode': 'worker',
                    'detail': 'Running analysis.',
                    'duration_seconds': 0.0,
                    'result': None,
                    'cancel_requested': True,
                    'context_fields': {},
                }
            },
            'job_runs': [],
        }

        forced = force_cancel_job(
            session_state,
            'job-1',
            reason='Cancellation timed out after 30 seconds and the analysis job was force-killed.',
        )

        self.assertEqual(forced['status'], 'cancelled')
        self.assertTrue(forced['force_cancelled'])
        self.assertEqual(forced['force_cancel_reason'], 'Cancellation timed out after 30 seconds and the analysis job was force-killed.')
        self.assertEqual(session_state['job_runs'][0]['status'], 'cancelled')

    def test_force_cancel_job_rotates_worker_executor_for_new_jobs(self) -> None:
        session_state: dict[str, object] = {'job_runs': []}
        runtime = {'backend_configured': True, 'mode': 'worker', 'max_workers': 1}

        first_job = submit_job(
            session_state,
            runtime,
            task_key='analysis_pipeline',
            task_label='Large dataset analysis',
            detail='Running analysis.',
            runner=lambda: time.sleep(0.2),
        )
        forced = force_cancel_job(
            session_state,
            first_job['job_id'],
            reason='Cancellation timed out after 30 seconds and the analysis job was force-killed.',
        )

        second_job = submit_job(
            session_state,
            runtime,
            task_key='analysis_pipeline',
            task_label='Large dataset analysis',
            detail='Running analysis again.',
            runner=lambda: 'fresh-run',
        )

        self.assertEqual(forced['status'], 'cancelled')
        status = get_job_status(session_state, second_job['job_id'])
        for _ in range(50):
            if status['status'] == 'completed':
                break
            time.sleep(0.02)
            status = get_job_status(session_state, second_job['job_id'])
        self.assertEqual(status['status'], 'completed')
        self.assertEqual(get_job_result(session_state, second_job['job_id']), 'fresh-run')

    def test_worker_jobs_survive_session_state_future_loss_between_polls(self) -> None:
        session_state: dict[str, object] = {'job_runs': []}
        runtime = {'backend_configured': True, 'mode': 'worker', 'max_workers': 1}

        submission = submit_job(
            session_state,
            runtime,
            task_key='analysis_pipeline',
            task_label='Large dataset analysis',
            detail='Running analysis.',
            runner=lambda: 'finished-after-rerun',
        )
        session_state.pop('job_futures', None)

        status = get_job_status(session_state, submission['job_id'])
        for _ in range(50):
            if status['status'] == 'completed':
                break
            time.sleep(0.02)
            status = get_job_status(session_state, submission['job_id'])

        self.assertEqual(status['status'], 'completed')
        self.assertEqual(get_job_result(session_state, submission['job_id']), 'finished-after-rerun')

    def test_job_status_view_handles_empty_history(self) -> None:
        table = build_job_status_view([])
        self.assertTrue(table.empty)


if __name__ == '__main__':
    unittest.main()
