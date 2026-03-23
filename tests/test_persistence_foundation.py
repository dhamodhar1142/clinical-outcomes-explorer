from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.persistence import PostgreSQLPersistenceBackend, build_persistence_service
from src.workspace import build_workspace_identity, persist_active_workspace_state, sync_workspace_views


class FakePostgresCursor:
    def __init__(self, store: dict[str, object]) -> None:
        self.store = store
        self.description: list[tuple[str]] = []
        self._rows: list[tuple[object, ...]] = []

    def execute(self, query: str, params: tuple[object, ...] = ()) -> None:
        self.store.setdefault('queries', []).append((query, params))
        normalized = ' '.join(query.split()).lower()

        if normalized.startswith('create table'):
            self.description = []
            self._rows = []
            return

        if 'insert into schema_migrations' in normalized:
            migrations = self.store.setdefault('schema_migrations', [])
            version = int(params[0])
            if all(existing[0] != version for existing in migrations):
                migrations.append((version, str(params[1])))
            self.description = []
            self._rows = []
            return

        if 'select version, applied_at' in normalized and 'from schema_migrations' in normalized:
            migrations = list(self.store.get('schema_migrations', []))
            self.description = [('version',), ('applied_at',)]
            self._rows = migrations
            return

        if normalized.startswith('select %s as value'):
            self.description = [('value',)]
            self._rows = [(params[0],)]
            return

        self.description = []
        self._rows = []

    def fetchall(self) -> list[tuple[object, ...]]:
        return list(self._rows)

    def close(self) -> None:
        return


class FakePostgresConnection:
    def __init__(self, store: dict[str, object]) -> None:
        self.store = store

    def cursor(self) -> FakePostgresCursor:
        return FakePostgresCursor(self.store)

    def commit(self) -> None:
        commits = int(self.store.get('commit_count', 0))
        self.store['commit_count'] = commits + 1

    def close(self) -> None:
        return


class PersistenceFoundationTests(unittest.TestCase):
    def test_persistence_service_falls_back_to_session_mode_without_database(self) -> None:
        service = build_persistence_service(database_url='', sqlite_path='')
        self.assertFalse(service.enabled)
        self.assertEqual(service.status.mode, 'session')
        self.assertIn('session-only', service.status.storage_target)
        schema_info = service.get_schema_info()
        self.assertEqual(schema_info['mode'], 'session')
        self.assertFalse(schema_info['is_current'])

    def test_postgres_url_falls_back_safely_without_driver(self) -> None:
        with patch('src.persistence._import_postgres_driver', side_effect=ImportError('missing driver')):
            service = build_persistence_service(database_url='postgresql://demo:demo@localhost:5432/smart_dataset')
        self.assertFalse(service.enabled)
        self.assertEqual(service.status.mode, 'session')
        self.assertTrue(any('no PostgreSQL driver is installed' in note for note in service.status.notes))

    def test_postgres_backend_translates_placeholders_and_redacts_target(self) -> None:
        store: dict[str, object] = {}

        def fake_connect(_database_url: str) -> FakePostgresConnection:
            return FakePostgresConnection(store)

        with patch('src.persistence._import_postgres_driver', return_value=('fake-psycopg', fake_connect)):
            backend = PostgreSQLPersistenceBackend('postgresql://demo:secret@localhost:5432/smart_dataset')
            backend.execute('SELECT ? AS value', ('ready',))
            rows = backend.fetch_all('SELECT ? AS value', ('healthy',))

        queries = [query for query, _params in store.get('queries', [])]
        self.assertTrue(any('%s AS value' in query for query in queries))
        self.assertEqual(rows[0]['value'], 'healthy')
        self.assertNotIn('secret', backend.storage_target)
        self.assertIn('***', backend.storage_target)

    def test_postgres_persistence_service_enables_when_driver_is_available(self) -> None:
        store: dict[str, object] = {}

        def fake_connect(_database_url: str) -> FakePostgresConnection:
            return FakePostgresConnection(store)

        with patch('src.persistence._import_postgres_driver', return_value=('fake-psycopg', fake_connect)):
            service = build_persistence_service(database_url='postgresql://demo:secret@localhost:5432/smart_dataset')

        self.assertTrue(service.enabled)
        self.assertEqual(service.status.mode, 'postgres')
        self.assertTrue(any('persisted in PostgreSQL' in note for note in service.status.notes))
        schema_info = service.get_schema_info()
        self.assertTrue(schema_info['is_current'])
        self.assertEqual(schema_info['mode'], 'postgres')

    def test_sqlite_persistence_service_saves_and_loads_workspace_documents(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            sqlite_path = Path(tmpdir) / 'workspace_state.sqlite3'
            service = build_persistence_service(sqlite_path=sqlite_path)
            identity = build_workspace_identity('Dana Analyst', 'Care Ops', True)
            schema_info = service.get_schema_info()
            self.assertTrue(schema_info['is_current'])
            self.assertIn(schema_info['current_schema_version'], schema_info['applied_versions'])
            self.assertIsNotNone(service.repositories)
            self.assertTrue(hasattr(service.repositories, 'datasets'))
            self.assertTrue(hasattr(service.repositories, 'reports'))

            service.save_workspace_state(
                identity,
                {
                    'saved_snapshots': {'snapshot-a': {'dataset_name': 'demo-a'}},
                    'workflow_packs': {'pack-a': {'summary': 'Pack A'}},
                    'collaboration_notes': [{'note_text': 'Review cost anomalies'}],
                    'beta_interest_submissions': [{'email': 'pilot@example.com'}],
                    'analysis_log': [{'event_type': 'Dataset Selected'}],
                    'run_history': [{'dataset_name': 'Healthcare Operations Demo'}],
                },
            )

            restored = service.load_workspace_state(identity)
            self.assertIn('snapshot-a', restored['saved_snapshots'])
            self.assertIn('pack-a', restored['workflow_packs'])
            self.assertEqual(restored['collaboration_notes'][0]['note_text'], 'Review cost anomalies')
            self.assertEqual(restored['analysis_log'][0]['event_type'], 'Dataset Selected')
            self.assertEqual(restored['run_history'][0]['dataset_name'], 'Healthcare Operations Demo')

            service.save_dataset_metadata(
                identity,
                {
                    'dataset_name': 'Healthcare Operations Demo',
                    'source_mode': 'Demo dataset',
                    'row_count': 5000,
                    'column_count': 18,
                    'file_size_mb': 0.65,
                    'description': 'Operations demo.',
                    'best_for': 'Readmission review.',
                },
            )
            service.record_usage_event(
                identity,
                {
                    'event_type': 'Dataset Selected',
                    'details': 'Selected demo dataset.',
                    'user_interaction': 'Dataset selection',
                    'analysis_step': 'Data intake',
                },
            )
            service.save_report_metadata(
                identity,
                {
                    'dataset_name': 'Healthcare Operations Demo',
                    'report_type': 'Executive Report',
                    'file_name': 'executive_report.txt',
                    'status': 'generated',
                },
            )

            datasets = service.list_dataset_metadata(identity)
            usage_events = service.list_usage_events(identity)
            reports = service.list_report_metadata(identity)
            summary = service.load_workspace_summary(identity)
            user_record = service.load_user_record(identity)
            workspace_record = service.load_workspace_record(identity)

            self.assertEqual(datasets[0]['dataset_name'], 'Healthcare Operations Demo')
            self.assertEqual(usage_events[0]['event_type'], 'Dataset Selected')
            self.assertEqual(reports[0]['report_type'], 'Executive Report')
            self.assertEqual(summary['dataset_count'], 1)
            self.assertEqual(summary['report_count'], 1)
            self.assertGreaterEqual(summary['user_count'], 1)
            self.assertEqual(user_record['display_name'], 'Dana Analyst')
            self.assertEqual(workspace_record['workspace_name'], 'Care Ops')

            service.save_user_settings(
                identity,
                {
                    'report_mode': 'Executive Summary',
                    'active_role': 'Analyst',
                },
            )
            service.save_dataset_version(
                identity,
                {
                    'dataset_name': 'Healthcare Operations Demo',
                    'version_hash': 'hash-v1',
                    'version_label': 'Healthcare Operations Demo | initial',
                    'source_mode': 'Demo dataset',
                    'row_count': 5000,
                    'column_count': 18,
                    'file_size_mb': 0.65,
                    'metadata_json': {'source_mode': 'Demo dataset'},
                },
            )
            service.save_workspace_snapshot(
                identity,
                {
                    'snapshot_name': 'Executive baseline',
                    'dataset_name': 'Healthcare Operations Demo',
                    'dataset_version_id': 'version-1',
                    'snapshot_payload': {'controls': {'report_mode': 'Executive Summary'}},
                    'created_by_user_id': identity['user_id'],
                },
            )
            service.upsert_collaboration_session(
                identity,
                {
                    'session_id': 'session-a',
                    'active_section': 'Data Intake',
                    'presence_state': 'active',
                },
            )

            settings = service.load_user_settings(identity)
            versions = service.list_dataset_versions(identity, 'Healthcare Operations Demo')
            snapshots = service.list_workspace_snapshots(identity)
            collaborators = service.list_collaboration_sessions(identity)

            self.assertEqual(settings['report_mode'], 'Executive Summary')
            self.assertEqual(versions[0]['version_hash'], 'hash-v1')
            self.assertTrue(versions[0]['is_active'])
            self.assertEqual(snapshots[0]['snapshot_name'], 'Executive baseline')
            self.assertEqual(collaborators[0]['active_section'], 'Data Intake')

    def test_workspace_sync_hydrates_and_persists_workspace_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            sqlite_path = Path(tmpdir) / 'workspace_state.sqlite3'
            service = build_persistence_service(sqlite_path=sqlite_path)
            identity = build_workspace_identity('User One', 'Workspace A', True)
            service.save_workspace_state(
                identity,
                {
                    'saved_snapshots': {'snapshot-a': {'dataset_name': 'demo-a'}},
                    'workflow_packs': {'pack-a': {'summary': 'Pack A'}},
                    'collaboration_notes': [{'note_text': 'Imported note'}],
                    'beta_interest_submissions': [],
                    'analysis_log': [{'event_type': 'Dataset Selected'}],
                    'run_history': [{'dataset_name': 'Demo A'}],
                },
            )

            session_state: dict[str, object] = {
                'persistence_service': service,
                'workspace_saved_snapshots': {},
                'workspace_workflow_packs': {},
                'workspace_collaboration_notes': {},
                'workspace_beta_interest_submissions': {},
                'workspace_analysis_logs': {},
                'workspace_run_history': {},
                'workspace_identity': identity,
            }
            sync_workspace_views(session_state)

            self.assertIn('snapshot-a', session_state['saved_snapshots'])
            self.assertIn('pack-a', session_state['workflow_packs'])
            self.assertEqual(session_state['collaboration_notes'][0]['note_text'], 'Imported note')
            self.assertEqual(session_state['analysis_log'][0]['event_type'], 'Dataset Selected')
            self.assertEqual(session_state['run_history'][0]['dataset_name'], 'Demo A')

            session_state['saved_snapshots']['snapshot-b'] = {'dataset_name': 'demo-b'}
            session_state['analysis_log'].append({'event_type': 'Export Generated'})
            persist_active_workspace_state(session_state)

            restored = service.load_workspace_state(identity)
            self.assertIn('snapshot-b', restored['saved_snapshots'])
            self.assertEqual(restored['analysis_log'][-1]['event_type'], 'Export Generated')


if __name__ == '__main__':
    unittest.main()
