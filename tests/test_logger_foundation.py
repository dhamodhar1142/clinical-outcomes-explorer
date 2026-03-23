from __future__ import annotations

import logging
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.logger import (
    DATADOG_API_KEY_ENV,
    DATADOG_SITE_ENV,
    ERROR_LOG_DIR_ENV,
    LOGGER_NAME,
    SENTRY_DSN_ENV,
    _DIAGNOSTIC_BUFFER,
    build_error_capture_status,
    build_log_context,
    build_support_diagnostics,
    configure_logging,
    ensure_platform_log_context,
    get_logger,
    log_platform_event,
    log_platform_exception,
)


class LoggerFoundationTests(unittest.TestCase):
    def setUp(self) -> None:
        _DIAGNOSTIC_BUFFER.clear()
        self._tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tempdir.cleanup)
        self.env_patch = patch.dict(os.environ, {ERROR_LOG_DIR_ENV: self._tempdir.name}, clear=False)
        self.env_patch.start()
        self.addCleanup(self.env_patch.stop)

    def test_configure_logging_returns_platform_logger(self):
        logger = configure_logging('INFO')
        self.assertEqual(logger.name, LOGGER_NAME)
        self.assertTrue(logger.handlers)

    def test_log_platform_event_writes_json_payload(self):
        logger = get_logger('unit')
        records: list[logging.LogRecord] = []

        class _Collector(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                records.append(record)

        handler = _Collector()
        logger.addHandler(handler)
        previous_level = logger.level
        logger.setLevel(logging.INFO)
        try:
            log_platform_event('unit_test_event', logger_name='unit', sample='value', count=2)
        finally:
            logger.removeHandler(handler)
            logger.setLevel(previous_level)

        self.assertTrue(records)
        self.assertIn('"event": "unit_test_event"', records[-1].getMessage())
        self.assertIn('"sample": "value"', records[-1].getMessage())

    def test_log_platform_exception_writes_error_payload(self):
        logger = get_logger('unit')
        records: list[logging.LogRecord] = []

        class _Collector(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                records.append(record)

        handler = _Collector()
        logger.addHandler(handler)
        previous_level = logger.level
        logger.setLevel(logging.INFO)
        try:
            try:
                raise ValueError('boom')
            except ValueError as error:
                log_platform_exception('unit_test_exception', error, logger_name='unit', stage='test')
        finally:
            logger.removeHandler(handler)
            logger.setLevel(previous_level)

        self.assertTrue(records)
        self.assertIn('"event": "unit_test_exception"', records[-1].getMessage())
        self.assertIn('"error_type": "ValueError"', records[-1].getMessage())

    def test_error_hook_is_invoked_when_configured(self):
        received: list[dict[str, object]] = []

        class _HookModule:
            @staticmethod
            def capture(payload):
                received.append(payload)

        with patch.dict(os.environ, {'SMART_DATASET_ANALYZER_ERROR_HOOK': 'fake_module:capture'}, clear=False), \
             patch('src.logger.import_module', return_value=_HookModule()):
            try:
                raise RuntimeError('hook me')
            except RuntimeError as error:
                log_platform_exception('hook_test_exception', error, logger_name='unit')

        self.assertTrue(received)
        self.assertEqual(received[0]['event'], 'hook_test_exception')

    def test_support_diagnostics_summarize_recent_events_and_errors(self):
        log_platform_event('diagnostic_event', logger_name='unit', workspace_id='workspace-a')
        try:
            raise ValueError('boom')
        except ValueError as error:
            log_platform_exception('diagnostic_error', error, logger_name='unit', operation_type='quality_review')

        diagnostics = build_support_diagnostics()
        self.assertGreaterEqual(len(diagnostics['summary_cards']), 5)
        self.assertFalse(diagnostics['diagnostics_table'].empty)
        self.assertTrue(any(card['label'] == 'Recent errors' for card in diagnostics['summary_cards']))
        self.assertFalse(diagnostics['error_frequency_table'].empty)

    def test_error_capture_status_reports_inline_mode_by_default(self):
        status = build_error_capture_status()
        self.assertEqual(status['status'], 'Inline only')

    def test_platform_log_context_assigns_session_and_request_ids(self):
        session_state: dict[str, object] = {}
        first = ensure_platform_log_context(session_state)
        second = ensure_platform_log_context(session_state)
        context = build_log_context(
            {
                **session_state,
                'workspace_identity': {
                    'workspace_id': 'workspace-a',
                    'workspace_name': 'Pilot Workspace',
                    'user_id': 'user-a',
                    'auth_mode': 'local',
                    'role': 'admin',
                },
            }
        )

        self.assertTrue(str(first['session_id']).startswith('session-'))
        self.assertNotEqual(first['request_id'], second['request_id'])
        self.assertEqual(context['workspace_id'], 'workspace-a')
        self.assertEqual(context['workspace_role'], 'admin')

    def test_log_platform_exception_persists_error_entries_for_30_day_store(self):
        try:
            raise RuntimeError('persist me')
        except RuntimeError as error:
            log_platform_exception(
                'persisted_error',
                error,
                logger_name='unit',
                operation_type='export_bundle',
                dataset_name='Demo Dataset',
                row_count=100,
            )

        files = list(Path(self._tempdir.name).glob('errors-*.jsonl'))
        self.assertEqual(len(files), 1)
        lines = files[0].read_text(encoding='utf-8').strip().splitlines()
        self.assertEqual(len(lines), 1)
        self.assertIn('"operation_type": "export_bundle"', lines[0])

    def test_support_diagnostics_builds_recurring_error_alerts(self):
        for _ in range(3):
            try:
                raise ValueError('same failure')
            except ValueError as error:
                log_platform_exception(
                    'analysis_failure',
                    error,
                    logger_name='unit',
                    operation_type='analysis_pipeline',
                )

        diagnostics = build_support_diagnostics()
        self.assertFalse(diagnostics['recurring_error_table'].empty)
        self.assertEqual(
            diagnostics['recurring_error_table'].iloc[0]['operation_type'],
            'analysis_pipeline',
        )

    def test_monitoring_targets_report_external_mode(self):
        with patch.dict(
            os.environ,
            {
                SENTRY_DSN_ENV: 'https://example@sentry.invalid/1',
                DATADOG_API_KEY_ENV: 'api-key',
                DATADOG_SITE_ENV: 'datadoghq.com',
            },
            clear=False,
        ):
            status = build_error_capture_status()
        self.assertEqual(status['status'], 'External monitoring configured')

    def test_log_platform_exception_dispatches_monitoring_targets_when_configured(self):
        class _Scope:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def set_extra(self, key, value):
                return None

        class _Hub:
            current = type('CurrentHub', (), {'client': object()})()

        class _SentryModule:
            Hub = _Hub

            @staticmethod
            def push_scope():
                return _Scope()

            @staticmethod
            def capture_exception(error):
                return None

        with patch.dict(
            os.environ,
            {
                SENTRY_DSN_ENV: 'https://example@sentry.invalid/1',
                DATADOG_API_KEY_ENV: 'api-key',
                DATADOG_SITE_ENV: 'datadoghq.com',
            },
            clear=False,
        ), patch('src.logger.import_module', return_value=_SentryModule()), patch('src.logger.urllib_request.urlopen') as mocked_urlopen:
            mocked_urlopen.return_value.__enter__.return_value = object()
            try:
                raise RuntimeError('monitor me')
            except RuntimeError as error:
                log_platform_exception('monitoring_error', error, logger_name='unit', operation_type='ui_render')

        mocked_urlopen.assert_called_once()


if __name__ == '__main__':
    unittest.main()
