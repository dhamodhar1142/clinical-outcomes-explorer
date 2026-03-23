from __future__ import annotations

import json
import os
import hashlib
import sqlite3
from abc import ABC, abstractmethod
from contextlib import closing
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib import import_module
from pathlib import Path
from typing import Any
from urllib.parse import ParseResult, urlparse, urlunparse

from src.models import (
    CollaborationSession,
    Dataset,
    DatasetVersion,
    PersistentWorkspaceSnapshot,
    Report,
    UsageEvent,
    User,
    UserSettings,
    Workspace,
)


SQLITE_PATH_ENV = 'SMART_DATASET_ANALYZER_SQLITE_PATH'
DATABASE_URL_ENV = 'SMART_DATASET_ANALYZER_DB_URL'
CURRENT_SCHEMA_VERSION = 3
SUPPORTED_DOCUMENT_TYPES = {
    'saved_snapshots',
    'workflow_packs',
    'collaboration_notes',
    'beta_interest_submissions',
    'analysis_log',
    'run_history',
}


def _utcnow() -> str:
    return datetime.now(UTC).isoformat()


def _parse_timestamp(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(str(value).replace('Z', '+00:00'))
    except ValueError:
        return None


def _json_default(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _normalize_document(document_type: str, payload: Any) -> Any:
    if document_type in {'saved_snapshots', 'workflow_packs'}:
        return payload if isinstance(payload, dict) else {}
    if document_type in {'collaboration_notes', 'beta_interest_submissions', 'analysis_log', 'run_history'}:
        return payload if isinstance(payload, list) else []
    return payload


def _stable_id(*parts: Any) -> str:
    raw = '::'.join(str(part or '') for part in parts)
    return hashlib.sha1(raw.encode('utf-8')).hexdigest()[:16]


def _redact_database_url(database_url: str) -> str:
    parsed = urlparse(database_url)
    if not parsed.password:
        return database_url
    netloc = parsed.hostname or ''
    if parsed.username:
        netloc = f'{parsed.username}:***@{netloc}'
    if parsed.port:
        netloc = f'{netloc}:{parsed.port}'
    return urlunparse(
        ParseResult(
            scheme=parsed.scheme,
            netloc=netloc,
            path=parsed.path,
            params=parsed.params,
            query=parsed.query,
            fragment=parsed.fragment,
        )
    )


def _translate_placeholders(query: str) -> str:
    return query.replace('?', '%s')


def _import_postgres_driver() -> tuple[str, Any]:
    try:
        psycopg = import_module('psycopg')
        return 'psycopg', psycopg.connect
    except ImportError:
        pass

    try:
        psycopg2 = import_module('psycopg2')
        return 'psycopg2', psycopg2.connect
    except ImportError as exc:
        raise ImportError(
            'PostgreSQL persistence requires the optional psycopg or psycopg2 package.'
        ) from exc


@dataclass(frozen=True)
class PersistenceStatus:
    enabled: bool
    mode: str
    storage_target: str
    notes: list[str]


class PersistenceBackend(ABC):
    def __init__(self, storage_target: str, mode: str) -> None:
        self.storage_target = storage_target
        self.mode = mode

    @abstractmethod
    def initialize(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def fetch_all(self, query: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def fetch_one(self, query: str, params: tuple[Any, ...] = ()) -> dict[str, Any] | None:
        raise NotImplementedError

    @abstractmethod
    def execute(self, query: str, params: tuple[Any, ...] = ()) -> None:
        raise NotImplementedError

    @abstractmethod
    def executescript(self, sql: str) -> None:
        raise NotImplementedError


class SQLitePersistenceBackend(PersistenceBackend):
    def __init__(self, sqlite_path: Path) -> None:
        self.sqlite_path = sqlite_path
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        super().__init__(str(sqlite_path), 'sqlite')
        self.initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.sqlite_path)
        connection.row_factory = sqlite3.Row
        return connection

    def initialize(self) -> None:
        self.executescript(
            '''
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version INTEGER PRIMARY KEY,
                applied_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS workspaces (
                workspace_id TEXT PRIMARY KEY,
                owner_user_id TEXT,
                display_name TEXT NOT NULL,
                workspace_name TEXT NOT NULL,
                signed_in INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                display_name TEXT NOT NULL,
                email TEXT NOT NULL DEFAULT '',
                auth_mode TEXT NOT NULL DEFAULT 'guest',
                signed_in INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS workspace_documents (
                workspace_id TEXT NOT NULL,
                document_type TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (workspace_id, document_type),
                FOREIGN KEY (workspace_id) REFERENCES workspaces(workspace_id)
            );

            CREATE TABLE IF NOT EXISTS dataset_metadata (
                workspace_id TEXT NOT NULL,
                dataset_name TEXT NOT NULL,
                source_mode TEXT NOT NULL,
                row_count INTEGER NOT NULL DEFAULT 0,
                column_count INTEGER NOT NULL DEFAULT 0,
                file_size_mb REAL NOT NULL DEFAULT 0,
                description TEXT NOT NULL DEFAULT '',
                best_for TEXT NOT NULL DEFAULT '',
                updated_at TEXT NOT NULL,
                PRIMARY KEY (workspace_id, dataset_name),
                FOREIGN KEY (workspace_id) REFERENCES workspaces(workspace_id)
            );

            CREATE TABLE IF NOT EXISTS usage_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                workspace_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                details TEXT NOT NULL DEFAULT '',
                user_interaction TEXT NOT NULL DEFAULT '',
                analysis_step TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                FOREIGN KEY (workspace_id) REFERENCES workspaces(workspace_id)
            );

            CREATE TABLE IF NOT EXISTS generated_reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                workspace_id TEXT NOT NULL,
                dataset_name TEXT NOT NULL,
                report_type TEXT NOT NULL,
                file_name TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'generated',
                generated_at TEXT NOT NULL,
                FOREIGN KEY (workspace_id) REFERENCES workspaces(workspace_id)
            );

            CREATE TABLE IF NOT EXISTS user_settings (
                user_id TEXT NOT NULL,
                workspace_id TEXT NOT NULL,
                settings_json TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (user_id, workspace_id)
            );

            CREATE TABLE IF NOT EXISTS dataset_versions (
                version_id TEXT PRIMARY KEY,
                workspace_id TEXT NOT NULL,
                dataset_name TEXT NOT NULL,
                version_hash TEXT NOT NULL,
                version_label TEXT NOT NULL DEFAULT '',
                source_mode TEXT NOT NULL DEFAULT '',
                row_count INTEGER NOT NULL DEFAULT 0,
                column_count INTEGER NOT NULL DEFAULT 0,
                file_size_mb REAL NOT NULL DEFAULT 0,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                is_active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS persistent_workspace_snapshots (
                snapshot_id TEXT PRIMARY KEY,
                workspace_id TEXT NOT NULL,
                snapshot_name TEXT NOT NULL,
                dataset_name TEXT NOT NULL DEFAULT '',
                dataset_version_id TEXT NOT NULL DEFAULT '',
                snapshot_payload_json TEXT NOT NULL,
                created_by_user_id TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS collaboration_sessions (
                workspace_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                display_name TEXT NOT NULL,
                session_id TEXT NOT NULL,
                active_section TEXT NOT NULL DEFAULT '',
                presence_state TEXT NOT NULL DEFAULT 'active',
                updated_at TEXT NOT NULL,
                PRIMARY KEY (workspace_id, user_id, session_id)
            );

            CREATE TABLE IF NOT EXISTS beta_interest_submissions (
                submission_id TEXT PRIMARY KEY,
                workspace_id TEXT NOT NULL,
                name TEXT NOT NULL,
                email TEXT NOT NULL,
                organization TEXT NOT NULL DEFAULT '',
                use_case TEXT NOT NULL,
                workspace_name TEXT NOT NULL DEFAULT '',
                submitted_by TEXT NOT NULL DEFAULT '',
                dataset_name TEXT NOT NULL DEFAULT '',
                dataset_source_mode TEXT NOT NULL DEFAULT '',
                capture_mode TEXT NOT NULL DEFAULT 'Database',
                follow_up_status TEXT NOT NULL DEFAULT 'New',
                contacted_at TEXT NOT NULL DEFAULT '',
                completed_at TEXT NOT NULL DEFAULT '',
                submitted_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            '''
        )
        self.execute(
            '''
            INSERT OR IGNORE INTO schema_migrations (version, applied_at)
            VALUES (?, ?)
            ''',
            (CURRENT_SCHEMA_VERSION, _utcnow()),
        )

    def fetch_all(self, query: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        with closing(self._connect()) as connection:
            rows = connection.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def fetch_one(self, query: str, params: tuple[Any, ...] = ()) -> dict[str, Any] | None:
        with closing(self._connect()) as connection:
            row = connection.execute(query, params).fetchone()
        return dict(row) if row is not None else None

    def execute(self, query: str, params: tuple[Any, ...] = ()) -> None:
        with closing(self._connect()) as connection:
            connection.execute(query, params)
            connection.commit()

    def executescript(self, sql: str) -> None:
        with closing(self._connect()) as connection:
            connection.executescript(sql)
            connection.commit()


class PostgreSQLPersistenceBackend(PersistenceBackend):
    def __init__(self, database_url: str) -> None:
        self.database_url = database_url
        self.driver_name, self._connect_fn = _import_postgres_driver()
        super().__init__(_redact_database_url(database_url), 'postgres')
        self.initialize()

    def _connect(self) -> Any:
        return self._connect_fn(self.database_url)

    def _execute_cursor(
        self,
        cursor: Any,
        query: str,
        params: tuple[Any, ...] = (),
    ) -> None:
        cursor.execute(_translate_placeholders(query), params)

    def initialize(self) -> None:
        self.executescript(
            '''
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version INTEGER PRIMARY KEY,
                applied_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS workspaces (
                workspace_id TEXT PRIMARY KEY,
                owner_user_id TEXT,
                display_name TEXT NOT NULL,
                workspace_name TEXT NOT NULL,
                signed_in BOOLEAN NOT NULL DEFAULT FALSE,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                display_name TEXT NOT NULL,
                email TEXT NOT NULL DEFAULT '',
                auth_mode TEXT NOT NULL DEFAULT 'guest',
                signed_in BOOLEAN NOT NULL DEFAULT FALSE,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS workspace_documents (
                workspace_id TEXT NOT NULL,
                document_type TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (workspace_id, document_type)
            );

            CREATE TABLE IF NOT EXISTS dataset_metadata (
                workspace_id TEXT NOT NULL,
                dataset_name TEXT NOT NULL,
                source_mode TEXT NOT NULL,
                row_count INTEGER NOT NULL DEFAULT 0,
                column_count INTEGER NOT NULL DEFAULT 0,
                file_size_mb DOUBLE PRECISION NOT NULL DEFAULT 0,
                description TEXT NOT NULL DEFAULT '',
                best_for TEXT NOT NULL DEFAULT '',
                updated_at TEXT NOT NULL,
                PRIMARY KEY (workspace_id, dataset_name)
            );

            CREATE TABLE IF NOT EXISTS usage_events (
                id BIGSERIAL PRIMARY KEY,
                workspace_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                details TEXT NOT NULL DEFAULT '',
                user_interaction TEXT NOT NULL DEFAULT '',
                analysis_step TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS generated_reports (
                id BIGSERIAL PRIMARY KEY,
                workspace_id TEXT NOT NULL,
                dataset_name TEXT NOT NULL,
                report_type TEXT NOT NULL,
                file_name TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'generated',
                generated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS user_settings (
                user_id TEXT NOT NULL,
                workspace_id TEXT NOT NULL,
                settings_json TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (user_id, workspace_id)
            );

            CREATE TABLE IF NOT EXISTS dataset_versions (
                version_id TEXT PRIMARY KEY,
                workspace_id TEXT NOT NULL,
                dataset_name TEXT NOT NULL,
                version_hash TEXT NOT NULL,
                version_label TEXT NOT NULL DEFAULT '',
                source_mode TEXT NOT NULL DEFAULT '',
                row_count INTEGER NOT NULL DEFAULT 0,
                column_count INTEGER NOT NULL DEFAULT 0,
                file_size_mb DOUBLE PRECISION NOT NULL DEFAULT 0,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                is_active BOOLEAN NOT NULL DEFAULT TRUE,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS persistent_workspace_snapshots (
                snapshot_id TEXT PRIMARY KEY,
                workspace_id TEXT NOT NULL,
                snapshot_name TEXT NOT NULL,
                dataset_name TEXT NOT NULL DEFAULT '',
                dataset_version_id TEXT NOT NULL DEFAULT '',
                snapshot_payload_json TEXT NOT NULL,
                created_by_user_id TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS collaboration_sessions (
                workspace_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                display_name TEXT NOT NULL,
                session_id TEXT NOT NULL,
                active_section TEXT NOT NULL DEFAULT '',
                presence_state TEXT NOT NULL DEFAULT 'active',
                updated_at TEXT NOT NULL,
                PRIMARY KEY (workspace_id, user_id, session_id)
            );

            CREATE TABLE IF NOT EXISTS beta_interest_submissions (
                submission_id TEXT PRIMARY KEY,
                workspace_id TEXT NOT NULL,
                name TEXT NOT NULL,
                email TEXT NOT NULL,
                organization TEXT NOT NULL DEFAULT '',
                use_case TEXT NOT NULL,
                workspace_name TEXT NOT NULL DEFAULT '',
                submitted_by TEXT NOT NULL DEFAULT '',
                dataset_name TEXT NOT NULL DEFAULT '',
                dataset_source_mode TEXT NOT NULL DEFAULT '',
                capture_mode TEXT NOT NULL DEFAULT 'Database',
                follow_up_status TEXT NOT NULL DEFAULT 'New',
                contacted_at TEXT NOT NULL DEFAULT '',
                completed_at TEXT NOT NULL DEFAULT '',
                submitted_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            '''
        )
        self.execute(
            '''
            INSERT INTO schema_migrations (version, applied_at)
            VALUES (?, ?)
            ON CONFLICT(version) DO NOTHING
            ''',
            (CURRENT_SCHEMA_VERSION, _utcnow()),
        )

    def fetch_all(self, query: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        with closing(self._connect()) as connection:
            with closing(connection.cursor()) as cursor:
                self._execute_cursor(cursor, query, params)
                rows = cursor.fetchall()
                columns = [column[0] for column in cursor.description or ()]
        return [dict(zip(columns, row, strict=False)) for row in rows]

    def fetch_one(self, query: str, params: tuple[Any, ...] = ()) -> dict[str, Any] | None:
        rows = self.fetch_all(query, params)
        return rows[0] if rows else None

    def execute(self, query: str, params: tuple[Any, ...] = ()) -> None:
        with closing(self._connect()) as connection:
            with closing(connection.cursor()) as cursor:
                self._execute_cursor(cursor, query, params)
            connection.commit()

    def executescript(self, sql: str) -> None:
        statements = [statement.strip() for statement in sql.split(';') if statement.strip()]
        if not statements:
            return
        with closing(self._connect()) as connection:
            with closing(connection.cursor()) as cursor:
                for statement in statements:
                    cursor.execute(statement)
            connection.commit()


class SchemaRepository:
    def __init__(self, backend: PersistenceBackend) -> None:
        self.backend = backend

    def get_schema_info(self) -> dict[str, Any]:
        versions = self.backend.fetch_all(
            '''
            SELECT version, applied_at
            FROM schema_migrations
            ORDER BY version ASC
            '''
        )
        applied_versions = [int(row['version']) for row in versions]
        return {
            'current_schema_version': CURRENT_SCHEMA_VERSION,
            'applied_versions': applied_versions,
            'is_current': CURRENT_SCHEMA_VERSION in applied_versions,
        }


class UserRepository:
    def __init__(self, backend: PersistenceBackend) -> None:
        self.backend = backend

    def ensure(self, identity: dict[str, Any]) -> None:
        self.backend.execute(
            '''
            INSERT INTO users (user_id, display_name, email, auth_mode, signed_in, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                display_name = excluded.display_name,
                email = excluded.email,
                auth_mode = excluded.auth_mode,
                signed_in = excluded.signed_in,
                updated_at = excluded.updated_at
            ''',
            (
                str(identity.get('user_id', 'guest-user')),
                str(identity.get('display_name', 'Guest User')),
                str(identity.get('email', '')),
                str(identity.get('auth_mode', 'guest')),
                1 if bool(identity.get('signed_in')) else 0,
                _utcnow(),
            ),
        )


class WorkspaceRepository:
    def __init__(self, backend: PersistenceBackend) -> None:
        self.backend = backend

    def ensure(self, identity: dict[str, Any]) -> None:
        self.backend.execute(
            '''
            INSERT INTO workspaces (workspace_id, owner_user_id, display_name, workspace_name, signed_in, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(workspace_id) DO UPDATE SET
                owner_user_id = excluded.owner_user_id,
                display_name = excluded.display_name,
                workspace_name = excluded.workspace_name,
                signed_in = excluded.signed_in,
                updated_at = excluded.updated_at
            ''',
            (
                str(identity.get('workspace_id', 'guest-demo-workspace')),
                str(identity.get('owner_user_id', identity.get('user_id', 'guest-user'))),
                str(identity.get('display_name', 'Guest User')),
                str(identity.get('workspace_name', 'Guest Demo Workspace')),
                1 if bool(identity.get('signed_in')) else 0,
                _utcnow(),
            ),
        )

    def load_summary(self, workspace_id: str) -> dict[str, Any]:
        workspace = self.backend.fetch_one(
            '''
            SELECT workspace_id, owner_user_id, display_name, workspace_name, signed_in, updated_at
            FROM workspaces
            WHERE workspace_id = ?
            ''',
            (workspace_id,),
        )
        user_count = self.backend.fetch_one('SELECT COUNT(*) AS count FROM users') or {'count': 0}
        dataset_count = self.backend.fetch_one(
            'SELECT COUNT(*) AS count FROM dataset_metadata WHERE workspace_id = ?',
            (workspace_id,),
        ) or {'count': 0}
        report_count = self.backend.fetch_one(
            'SELECT COUNT(*) AS count FROM generated_reports WHERE workspace_id = ?',
            (workspace_id,),
        ) or {'count': 0}
        usage_count = self.backend.fetch_one(
            'SELECT COUNT(*) AS count FROM usage_events WHERE workspace_id = ?',
            (workspace_id,),
        ) or {'count': 0}
        return {
            'workspace': workspace or {},
            'user_count': int(user_count['count']),
            'dataset_count': int(dataset_count['count']),
            'report_count': int(report_count['count']),
            'usage_event_count': int(usage_count['count']),
        }


class WorkspaceDocumentRepository:
    def __init__(self, backend: PersistenceBackend) -> None:
        self.backend = backend

    def load(self, workspace_id: str, document_type: str) -> Any:
        row = self.backend.fetch_one(
            '''
            SELECT payload_json
            FROM workspace_documents
            WHERE workspace_id = ? AND document_type = ?
            ''',
            (workspace_id, document_type),
        )
        if row is None:
            return _normalize_document(document_type, None)
        try:
            payload = json.loads(str(row['payload_json']))
        except json.JSONDecodeError:
            return _normalize_document(document_type, None)
        return _normalize_document(document_type, payload)

    def save(self, workspace_id: str, document_type: str, payload: Any) -> None:
        self.backend.execute(
            '''
            INSERT INTO workspace_documents (workspace_id, document_type, payload_json, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(workspace_id, document_type) DO UPDATE SET
                payload_json = excluded.payload_json,
                updated_at = excluded.updated_at
            ''',
            (
                workspace_id,
                document_type,
                json.dumps(_normalize_document(document_type, payload), default=_json_default),
                _utcnow(),
            ),
        )

    def load_all(self, workspace_id: str) -> dict[str, Any]:
        return {
            document_type: self.load(workspace_id, document_type)
            for document_type in SUPPORTED_DOCUMENT_TYPES
        }


class DatasetMetadataRepository:
    def __init__(self, backend: PersistenceBackend) -> None:
        self.backend = backend

    def save(self, record: Dataset) -> None:
        self.backend.execute(
            '''
            INSERT INTO dataset_metadata (
                workspace_id, dataset_name, source_mode, row_count, column_count, file_size_mb, description, best_for, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(workspace_id, dataset_name) DO UPDATE SET
                source_mode = excluded.source_mode,
                row_count = excluded.row_count,
                column_count = excluded.column_count,
                file_size_mb = excluded.file_size_mb,
                description = excluded.description,
                best_for = excluded.best_for,
                updated_at = excluded.updated_at
            ''',
            (
                record.workspace_id,
                record.dataset_name,
                record.source_mode,
                record.row_count,
                record.column_count,
                record.file_size_mb,
                record.description,
                record.best_for,
                record.updated_at,
            ),
        )

    def list(self, workspace_id: str) -> list[dict[str, Any]]:
        return self.backend.fetch_all(
            '''
            SELECT workspace_id, dataset_name, source_mode, row_count, column_count, file_size_mb, description, best_for, updated_at
            FROM dataset_metadata
            WHERE workspace_id = ?
            ORDER BY updated_at DESC, dataset_name ASC
            ''',
            (workspace_id,),
        )


class UsageEventRepository:
    def __init__(self, backend: PersistenceBackend) -> None:
        self.backend = backend

    def save(self, record: UsageEvent) -> None:
        self.backend.execute(
            '''
            INSERT INTO usage_events (
                workspace_id, user_id, event_type, details, user_interaction, analysis_step, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ''',
            (
                record.workspace_id,
                record.user_id,
                record.event_type,
                record.details,
                record.user_interaction,
                record.analysis_step,
                record.created_at,
            ),
        )

    def list(self, workspace_id: str, *, limit: int = 200) -> list[dict[str, Any]]:
        return self.backend.fetch_all(
            '''
            SELECT workspace_id, user_id, event_type, details, user_interaction, analysis_step, created_at
            FROM usage_events
            WHERE workspace_id = ?
            ORDER BY created_at DESC, id DESC
            LIMIT ?
            ''',
            (workspace_id, max(1, int(limit))),
        )


class ReportMetadataRepository:
    def __init__(self, backend: PersistenceBackend) -> None:
        self.backend = backend

    def save(self, record: Report) -> None:
        self.backend.execute(
            '''
            INSERT INTO generated_reports (
                workspace_id, dataset_name, report_type, file_name, status, generated_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            ''',
            (
                record.workspace_id,
                record.dataset_name,
                record.report_type,
                record.file_name,
                record.status,
                record.generated_at,
            ),
        )

    def list(self, workspace_id: str, *, limit: int = 100) -> list[dict[str, Any]]:
        return self.backend.fetch_all(
            '''
            SELECT workspace_id, dataset_name, report_type, file_name, status, generated_at
            FROM generated_reports
            WHERE workspace_id = ?
            ORDER BY generated_at DESC, id DESC
            LIMIT ?
            ''',
            (workspace_id, max(1, int(limit))),
        )


class UserSettingsRepository:
    def __init__(self, backend: PersistenceBackend) -> None:
        self.backend = backend

    def save(self, record: UserSettings) -> None:
        self.backend.execute(
            '''
            INSERT INTO user_settings (user_id, workspace_id, settings_json, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(user_id, workspace_id) DO UPDATE SET
                settings_json = excluded.settings_json,
                updated_at = excluded.updated_at
            ''',
            (
                record.user_id,
                record.workspace_id,
                json.dumps(record.settings_json, default=_json_default),
                record.updated_at,
            ),
        )

    def load(self, user_id: str, workspace_id: str) -> dict[str, Any]:
        row = self.backend.fetch_one(
            '''
            SELECT settings_json
            FROM user_settings
            WHERE user_id = ? AND workspace_id = ?
            ''',
            (user_id, workspace_id),
        )
        if row is None:
            return {}
        try:
            payload = json.loads(str(row['settings_json']))
        except json.JSONDecodeError:
            return {}
        return payload if isinstance(payload, dict) else {}


class DatasetVersionRepository:
    def __init__(self, backend: PersistenceBackend) -> None:
        self.backend = backend

    def save(self, record: DatasetVersion) -> None:
        if record.is_active:
            self.backend.execute(
                '''
                UPDATE dataset_versions
                SET is_active = 0
                WHERE workspace_id = ? AND dataset_name = ?
                ''',
                (record.workspace_id, record.dataset_name),
            )
        self.backend.execute(
            '''
            INSERT INTO dataset_versions (
                version_id, workspace_id, dataset_name, version_hash, version_label, source_mode, row_count, column_count, file_size_mb, metadata_json, is_active, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(version_id) DO UPDATE SET
                version_hash = excluded.version_hash,
                version_label = excluded.version_label,
                source_mode = excluded.source_mode,
                row_count = excluded.row_count,
                column_count = excluded.column_count,
                file_size_mb = excluded.file_size_mb,
                metadata_json = excluded.metadata_json,
                is_active = excluded.is_active,
                created_at = excluded.created_at
            ''',
            (
                record.version_id,
                record.workspace_id,
                record.dataset_name,
                record.version_hash,
                record.version_label,
                record.source_mode,
                record.row_count,
                record.column_count,
                record.file_size_mb,
                json.dumps(record.metadata_json, default=_json_default),
                1 if record.is_active else 0,
                record.created_at,
            ),
        )

    def list(self, workspace_id: str, dataset_name: str | None = None) -> list[dict[str, Any]]:
        query = '''
            SELECT version_id, workspace_id, dataset_name, version_hash, version_label, source_mode, row_count, column_count, file_size_mb, metadata_json, is_active, created_at
            FROM dataset_versions
            WHERE workspace_id = ?
        '''
        params: list[Any] = [workspace_id]
        if dataset_name:
            query += ' AND dataset_name = ?'
            params.append(dataset_name)
        query += ' ORDER BY created_at DESC, version_id DESC'
        rows = self.backend.fetch_all(query, tuple(params))
        hydrated: list[dict[str, Any]] = []
        for row in rows:
            try:
                metadata = json.loads(str(row.get('metadata_json', '{}')))
            except json.JSONDecodeError:
                metadata = {}
            hydrated.append(
                {
                    **row,
                    'metadata_json': metadata if isinstance(metadata, dict) else {},
                    'is_active': bool(row.get('is_active')),
                }
            )
        return hydrated


class PersistentSnapshotRepository:
    def __init__(self, backend: PersistenceBackend) -> None:
        self.backend = backend

    def save(self, record: PersistentWorkspaceSnapshot) -> None:
        self.backend.execute(
            '''
            INSERT INTO persistent_workspace_snapshots (
                snapshot_id, workspace_id, snapshot_name, dataset_name, dataset_version_id, snapshot_payload_json, created_by_user_id, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(snapshot_id) DO UPDATE SET
                snapshot_name = excluded.snapshot_name,
                dataset_name = excluded.dataset_name,
                dataset_version_id = excluded.dataset_version_id,
                snapshot_payload_json = excluded.snapshot_payload_json,
                created_by_user_id = excluded.created_by_user_id,
                created_at = excluded.created_at
            ''',
            (
                record.snapshot_id,
                record.workspace_id,
                record.snapshot_name,
                record.dataset_name,
                record.dataset_version_id,
                json.dumps(record.snapshot_payload, default=_json_default),
                record.created_by_user_id,
                record.created_at,
            ),
        )

    def list(self, workspace_id: str) -> list[dict[str, Any]]:
        rows = self.backend.fetch_all(
            '''
            SELECT snapshot_id, workspace_id, snapshot_name, dataset_name, dataset_version_id, snapshot_payload_json, created_by_user_id, created_at
            FROM persistent_workspace_snapshots
            WHERE workspace_id = ?
            ORDER BY created_at DESC, snapshot_name ASC
            ''',
            (workspace_id,),
        )
        hydrated: list[dict[str, Any]] = []
        for row in rows:
            try:
                payload = json.loads(str(row.get('snapshot_payload_json', '{}')))
            except json.JSONDecodeError:
                payload = {}
            hydrated.append(
                {
                    **row,
                    'snapshot_payload': payload if isinstance(payload, dict) else {},
                }
            )
        return hydrated

    def load(self, workspace_id: str, snapshot_id: str) -> dict[str, Any] | None:
        rows = self.list(workspace_id)
        for row in rows:
            if str(row.get('snapshot_id')) == snapshot_id:
                return row
        return None


class CollaborationSessionRepository:
    def __init__(self, backend: PersistenceBackend) -> None:
        self.backend = backend

    def upsert(self, record: CollaborationSession) -> None:
        self.backend.execute(
            '''
            INSERT INTO collaboration_sessions (
                workspace_id, user_id, display_name, session_id, active_section, presence_state, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(workspace_id, user_id, session_id) DO UPDATE SET
                display_name = excluded.display_name,
                active_section = excluded.active_section,
                presence_state = excluded.presence_state,
                updated_at = excluded.updated_at
            ''',
            (
                record.workspace_id,
                record.user_id,
                record.display_name,
                record.session_id,
                record.active_section,
                record.presence_state,
                record.updated_at,
            ),
        )

    def list(self, workspace_id: str, *, active_within_seconds: int = 900) -> list[dict[str, Any]]:
        cutoff = datetime.now(UTC).timestamp() - max(60, int(active_within_seconds))
        rows = self.backend.fetch_all(
            '''
            SELECT workspace_id, user_id, display_name, session_id, active_section, presence_state, updated_at
            FROM collaboration_sessions
            WHERE workspace_id = ?
            ORDER BY updated_at DESC
            ''',
            (workspace_id,),
        )
        hydrated: list[dict[str, Any]] = []
        for row in rows:
            updated_at = _parse_timestamp(str(row.get('updated_at', '')))
            if updated_at is not None and updated_at.timestamp() < cutoff:
                continue
            hydrated.append(row)
        return hydrated


class BetaInterestRepository:
    def __init__(self, backend: PersistenceBackend) -> None:
        self.backend = backend

    def save(self, submission: dict[str, Any]) -> None:
        self.backend.execute(
            '''
            INSERT INTO beta_interest_submissions (
                submission_id, workspace_id, name, email, organization, use_case,
                workspace_name, submitted_by, dataset_name, dataset_source_mode,
                capture_mode, follow_up_status, contacted_at, completed_at,
                submitted_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(submission_id) DO UPDATE SET
                name = excluded.name,
                email = excluded.email,
                organization = excluded.organization,
                use_case = excluded.use_case,
                workspace_name = excluded.workspace_name,
                submitted_by = excluded.submitted_by,
                dataset_name = excluded.dataset_name,
                dataset_source_mode = excluded.dataset_source_mode,
                capture_mode = excluded.capture_mode,
                follow_up_status = excluded.follow_up_status,
                contacted_at = excluded.contacted_at,
                completed_at = excluded.completed_at,
                submitted_at = excluded.submitted_at,
                updated_at = excluded.updated_at
            ''',
            (
                str(submission.get('submission_id', '')),
                str(submission.get('workspace_id', 'guest-demo-workspace')),
                str(submission.get('name', '')),
                str(submission.get('email', '')),
                str(submission.get('organization', '')),
                str(submission.get('use_case', '')),
                str(submission.get('workspace_name', '')),
                str(submission.get('submitted_by', '')),
                str(submission.get('dataset_name', '')),
                str(submission.get('dataset_source_mode', '')),
                str(submission.get('capture_mode', 'Database')),
                str(submission.get('follow_up_status', 'New')),
                str(submission.get('contacted_at', '')),
                str(submission.get('completed_at', '')),
                str(submission.get('submitted_at', _utcnow())),
                _utcnow(),
            ),
        )

    def list(self, workspace_id: str, *, limit: int = 250) -> list[dict[str, Any]]:
        return self.backend.fetch_all(
            '''
            SELECT submission_id, workspace_id, name, email, organization, use_case,
                   workspace_name, submitted_by, dataset_name, dataset_source_mode,
                   capture_mode, follow_up_status, contacted_at, completed_at,
                   submitted_at, updated_at
            FROM beta_interest_submissions
            WHERE workspace_id = ?
            ORDER BY submitted_at DESC, updated_at DESC
            LIMIT ?
            ''',
            (workspace_id, max(1, int(limit))),
        )

    def update_status(self, workspace_id: str, submission_id: str, *, follow_up_status: str, contacted_at: str = '', completed_at: str = '') -> None:
        self.backend.execute(
            '''
            UPDATE beta_interest_submissions
            SET follow_up_status = ?,
                contacted_at = ?,
                completed_at = ?,
                updated_at = ?
            WHERE workspace_id = ? AND submission_id = ?
            ''',
            (
                follow_up_status,
                contacted_at,
                completed_at,
                _utcnow(),
                workspace_id,
                submission_id,
            ),
        )


@dataclass(frozen=True)
class PersistenceRepositories:
    schema: SchemaRepository
    users: UserRepository
    workspaces: WorkspaceRepository
    documents: WorkspaceDocumentRepository
    datasets: DatasetMetadataRepository
    usage_events: UsageEventRepository
    reports: ReportMetadataRepository
    user_settings: UserSettingsRepository
    dataset_versions: DatasetVersionRepository
    snapshots: PersistentSnapshotRepository
    collaboration_sessions: CollaborationSessionRepository
    beta_interest: BetaInterestRepository


class PersistenceService:
    def __init__(self, repositories: PersistenceRepositories | None, status: PersistenceStatus) -> None:
        self.repositories = repositories
        self.status = status

    @property
    def enabled(self) -> bool:
        return self.repositories is not None and self.status.enabled

    def ensure_workspace(self, identity: dict[str, Any]) -> None:
        if not self.enabled or self.repositories is None:
            return
        self.repositories.users.ensure(identity)
        self.repositories.workspaces.ensure(identity)

    def load_workspace_state(self, identity: dict[str, Any]) -> dict[str, Any]:
        if not self.enabled or self.repositories is None:
            return {
                'saved_snapshots': {},
                'workflow_packs': {},
                'collaboration_notes': [],
                'beta_interest_submissions': [],
                'analysis_log': [],
                'run_history': [],
            }
        workspace_id = str(identity.get('workspace_id', 'guest-demo-workspace'))
        self.ensure_workspace(identity)
        return self.repositories.documents.load_all(workspace_id)

    def save_workspace_state(self, identity: dict[str, Any], state: dict[str, Any]) -> None:
        if not self.enabled or self.repositories is None:
            return
        workspace_id = str(identity.get('workspace_id', 'guest-demo-workspace'))
        self.ensure_workspace(identity)
        for document_type in SUPPORTED_DOCUMENT_TYPES:
            if document_type in state:
                self.repositories.documents.save(workspace_id, document_type, state.get(document_type))

    def save_dataset_metadata(self, identity: dict[str, Any], metadata: dict[str, Any]) -> None:
        if not self.enabled or self.repositories is None:
            return
        self.ensure_workspace(identity)
        record = Dataset(
            workspace_id=str(identity.get('workspace_id', 'guest-demo-workspace')),
            dataset_name=str(metadata.get('dataset_name', 'Current dataset')),
            source_mode=str(metadata.get('source_mode', 'unknown')),
            row_count=int(metadata.get('row_count', 0) or 0),
            column_count=int(metadata.get('column_count', 0) or 0),
            file_size_mb=float(metadata.get('file_size_mb', 0.0) or 0.0),
            description=str(metadata.get('description', '')),
            best_for=str(metadata.get('best_for', '')),
            updated_at=_utcnow(),
        )
        self.repositories.datasets.save(record)

    def list_dataset_metadata(self, identity: dict[str, Any]) -> list[dict[str, Any]]:
        if not self.enabled or self.repositories is None:
            return []
        return self.repositories.datasets.list(str(identity.get('workspace_id', 'guest-demo-workspace')))

    def record_usage_event(self, identity: dict[str, Any], event: dict[str, Any]) -> None:
        if not self.enabled or self.repositories is None:
            return
        self.ensure_workspace(identity)
        record = UsageEvent(
            workspace_id=str(identity.get('workspace_id', 'guest-demo-workspace')),
            user_id=str(identity.get('user_id', 'guest-user')),
            event_type=str(event.get('event_type', 'Unknown Event')),
            details=str(event.get('details', '')),
            user_interaction=str(event.get('user_interaction', '')),
            analysis_step=str(event.get('analysis_step', '')),
            created_at=_utcnow(),
        )
        self.repositories.usage_events.save(record)

    def list_usage_events(self, identity: dict[str, Any], limit: int = 200) -> list[dict[str, Any]]:
        if not self.enabled or self.repositories is None:
            return []
        return self.repositories.usage_events.list(
            str(identity.get('workspace_id', 'guest-demo-workspace')),
            limit=limit,
        )

    def save_report_metadata(self, identity: dict[str, Any], metadata: dict[str, Any]) -> None:
        if not self.enabled or self.repositories is None:
            return
        self.ensure_workspace(identity)
        record = Report(
            workspace_id=str(identity.get('workspace_id', 'guest-demo-workspace')),
            dataset_name=str(metadata.get('dataset_name', 'Current dataset')),
            report_type=str(metadata.get('report_type', 'Generated Report')),
            file_name=str(metadata.get('file_name', 'generated_report.txt')),
            status=str(metadata.get('status', 'generated')),
            generated_at=_utcnow(),
        )
        self.repositories.reports.save(record)

    def list_report_metadata(self, identity: dict[str, Any], limit: int = 100) -> list[dict[str, Any]]:
        if not self.enabled or self.repositories is None:
            return []
        return self.repositories.reports.list(
            str(identity.get('workspace_id', 'guest-demo-workspace')),
            limit=limit,
        )

    def load_workspace_summary(self, identity: dict[str, Any]) -> dict[str, Any]:
        if not self.enabled or self.repositories is None:
            return {
                'workspace': {},
                'user_count': 0,
                'dataset_count': 0,
                'report_count': 0,
                'usage_event_count': 0,
            }
        return self.repositories.workspaces.load_summary(str(identity.get('workspace_id', 'guest-demo-workspace')))

    def get_schema_info(self) -> dict[str, Any]:
        if not self.enabled or self.repositories is None:
            return {
                'current_schema_version': CURRENT_SCHEMA_VERSION,
                'applied_versions': [],
                'is_current': False,
                'mode': 'session',
            }
        info = self.repositories.schema.get_schema_info()
        info['mode'] = self.status.mode
        return info

    def load_user_record(self, identity: dict[str, Any]) -> dict[str, Any]:
        return asdict(
            User(
                user_id=str(identity.get('user_id', 'guest-user')),
                display_name=str(identity.get('display_name', 'Guest User')),
                email=str(identity.get('email', '')),
                auth_mode=str(identity.get('auth_mode', 'guest')),
                signed_in=bool(identity.get('signed_in', False)),
                role=str(identity.get('role', 'viewer')),
                updated_at=_utcnow(),
            )
        )

    def save_user_settings(self, identity: dict[str, Any], settings: dict[str, Any]) -> None:
        if not self.enabled or self.repositories is None:
            return
        self.ensure_workspace(identity)
        record = UserSettings(
            user_id=str(identity.get('user_id', 'guest-user')),
            workspace_id=str(identity.get('workspace_id', 'guest-demo-workspace')),
            settings_json=settings if isinstance(settings, dict) else {},
            updated_at=_utcnow(),
        )
        self.repositories.user_settings.save(record)

    def load_user_settings(self, identity: dict[str, Any]) -> dict[str, Any]:
        if not self.enabled or self.repositories is None:
            return {}
        return self.repositories.user_settings.load(
            str(identity.get('user_id', 'guest-user')),
            str(identity.get('workspace_id', 'guest-demo-workspace')),
        )

    def save_dataset_version(self, identity: dict[str, Any], version: dict[str, Any]) -> None:
        if not self.enabled or self.repositories is None:
            return
        self.ensure_workspace(identity)
        record = DatasetVersion(
            workspace_id=str(identity.get('workspace_id', 'guest-demo-workspace')),
            dataset_name=str(version.get('dataset_name', 'Current dataset')),
            version_id=str(version.get('version_id', ''))
            or _stable_id(
                identity.get('workspace_id', ''),
                version.get('dataset_name', 'Current dataset'),
                version.get('version_hash', ''),
                version.get('version_label', ''),
            ),
            version_hash=str(version.get('version_hash', ''))
            or _stable_id(
                version.get('dataset_name', 'Current dataset'),
                version.get('row_count', 0),
                version.get('column_count', 0),
                version.get('file_size_mb', 0.0),
                version.get('metadata_json', {}),
            ),
            version_label=str(version.get('version_label', '')),
            source_mode=str(version.get('source_mode', 'unknown')),
            row_count=int(version.get('row_count', 0) or 0),
            column_count=int(version.get('column_count', 0) or 0),
            file_size_mb=float(version.get('file_size_mb', 0.0) or 0.0),
            metadata_json=version.get('metadata_json', {}) if isinstance(version.get('metadata_json', {}), dict) else {},
            is_active=bool(version.get('is_active', True)),
            created_at=_utcnow(),
        )
        self.repositories.dataset_versions.save(record)

    def list_dataset_versions(self, identity: dict[str, Any], dataset_name: str | None = None) -> list[dict[str, Any]]:
        if not self.enabled or self.repositories is None:
            return []
        return self.repositories.dataset_versions.list(
            str(identity.get('workspace_id', 'guest-demo-workspace')),
            dataset_name=dataset_name,
        )

    def save_workspace_snapshot(self, identity: dict[str, Any], snapshot: dict[str, Any]) -> None:
        if not self.enabled or self.repositories is None:
            return
        self.ensure_workspace(identity)
        record = PersistentWorkspaceSnapshot(
            workspace_id=str(identity.get('workspace_id', 'guest-demo-workspace')),
            snapshot_id=str(snapshot.get('snapshot_id', ''))
            or _stable_id(
                identity.get('workspace_id', ''),
                snapshot.get('snapshot_name', 'Workspace snapshot'),
                snapshot.get('dataset_name', ''),
                snapshot.get('dataset_version_id', ''),
            ),
            snapshot_name=str(snapshot.get('snapshot_name', 'Workspace snapshot')),
            dataset_name=str(snapshot.get('dataset_name', '')),
            dataset_version_id=str(snapshot.get('dataset_version_id', '')),
            snapshot_payload=snapshot.get('snapshot_payload', {}) if isinstance(snapshot.get('snapshot_payload', {}), dict) else {},
            created_by_user_id=str(snapshot.get('created_by_user_id', identity.get('user_id', 'guest-user'))),
            created_at=_utcnow(),
        )
        self.repositories.snapshots.save(record)

    def list_workspace_snapshots(self, identity: dict[str, Any]) -> list[dict[str, Any]]:
        if not self.enabled or self.repositories is None:
            return []
        return self.repositories.snapshots.list(str(identity.get('workspace_id', 'guest-demo-workspace')))

    def load_workspace_snapshot(self, identity: dict[str, Any], snapshot_id: str) -> dict[str, Any] | None:
        if not self.enabled or self.repositories is None:
            return None
        return self.repositories.snapshots.load(
            str(identity.get('workspace_id', 'guest-demo-workspace')),
            snapshot_id,
        )

    def upsert_collaboration_session(self, identity: dict[str, Any], presence: dict[str, Any]) -> None:
        if not self.enabled or self.repositories is None:
            return
        self.ensure_workspace(identity)
        record = CollaborationSession(
            workspace_id=str(identity.get('workspace_id', 'guest-demo-workspace')),
            user_id=str(identity.get('user_id', 'guest-user')),
            display_name=str(identity.get('display_name', 'Guest User')),
            session_id=str(presence.get('session_id', identity.get('session_id', 'session'))),
            active_section=str(presence.get('active_section', '')),
            presence_state=str(presence.get('presence_state', 'active')),
            updated_at=_utcnow(),
        )
        self.repositories.collaboration_sessions.upsert(record)

    def list_collaboration_sessions(self, identity: dict[str, Any]) -> list[dict[str, Any]]:
        if not self.enabled or self.repositories is None:
            return []
        return self.repositories.collaboration_sessions.list(
            str(identity.get('workspace_id', 'guest-demo-workspace'))
        )

    def save_beta_interest_submission(self, identity: dict[str, Any], submission: dict[str, Any]) -> None:
        if not self.enabled or self.repositories is None:
            return
        self.ensure_workspace(identity)
        payload = dict(submission)
        payload['workspace_id'] = str(identity.get('workspace_id', payload.get('workspace_id', 'guest-demo-workspace')))
        self.repositories.beta_interest.save(payload)

    def list_beta_interest_submissions(self, identity: dict[str, Any], limit: int = 250) -> list[dict[str, Any]]:
        if not self.enabled or self.repositories is None:
            return []
        return self.repositories.beta_interest.list(
            str(identity.get('workspace_id', 'guest-demo-workspace')),
            limit=limit,
        )

    def update_beta_interest_submission_status(
        self,
        identity: dict[str, Any],
        submission_id: str,
        *,
        follow_up_status: str,
        contacted_at: str = '',
        completed_at: str = '',
    ) -> None:
        if not self.enabled or self.repositories is None:
            return
        self.repositories.beta_interest.update_status(
            str(identity.get('workspace_id', 'guest-demo-workspace')),
            submission_id,
            follow_up_status=follow_up_status,
            contacted_at=contacted_at,
            completed_at=completed_at,
        )

    def load_workspace_record(self, identity: dict[str, Any]) -> dict[str, Any]:
        return asdict(
            Workspace(
                workspace_id=str(identity.get('workspace_id', 'guest-demo-workspace')),
                workspace_name=str(identity.get('workspace_name', 'Guest Demo Workspace')),
                display_name=str(identity.get('display_name', 'Guest User')),
                owner_user_id=str(identity.get('owner_user_id', identity.get('user_id', 'guest-user'))),
                signed_in=bool(identity.get('signed_in', False)),
                plan_name=str(identity.get('plan_name', 'Pro')),
                updated_at=_utcnow(),
            )
        )


def _build_repository_bundle(backend: PersistenceBackend) -> PersistenceRepositories:
    return PersistenceRepositories(
        schema=SchemaRepository(backend),
        users=UserRepository(backend),
        workspaces=WorkspaceRepository(backend),
        documents=WorkspaceDocumentRepository(backend),
        datasets=DatasetMetadataRepository(backend),
        usage_events=UsageEventRepository(backend),
        reports=ReportMetadataRepository(backend),
        user_settings=UserSettingsRepository(backend),
        dataset_versions=DatasetVersionRepository(backend),
        snapshots=PersistentSnapshotRepository(backend),
        collaboration_sessions=CollaborationSessionRepository(backend),
        beta_interest=BetaInterestRepository(backend),
    )


def build_persistence_service(
    *,
    database_url: str | None = None,
    sqlite_path: str | os.PathLike[str] | None = None,
) -> PersistenceService:
    configured_database_url = (database_url or os.getenv(DATABASE_URL_ENV, '')).strip()
    configured_sqlite_path = str(sqlite_path or os.getenv(SQLITE_PATH_ENV, '')).strip()

    notes = [
        f'Set {SQLITE_PATH_ENV} to enable SQLite-backed persistence.',
        f'{DATABASE_URL_ENV} accepts sqlite:/// and postgresql:// URLs.',
    ]

    if configured_database_url:
        parsed = urlparse(configured_database_url)
        if configured_database_url.startswith('sqlite:///'):
            configured_sqlite_path = configured_database_url.replace('sqlite:///', '', 1)
        elif parsed.scheme in {'postgres', 'postgresql'}:
            try:
                backend = PostgreSQLPersistenceBackend(configured_database_url)
            except ImportError:
                return PersistenceService(
                    repositories=None,
                    status=PersistenceStatus(
                        enabled=False,
                        mode='session',
                        storage_target='session-only',
                        notes=notes + ['A PostgreSQL database URL was detected, but no PostgreSQL driver is installed. Install the optional persistence dependencies to enable it.'],
                    ),
                )
            except Exception as exc:
                return PersistenceService(
                    repositories=None,
                    status=PersistenceStatus(
                        enabled=False,
                        mode='session',
                        storage_target='session-only',
                        notes=notes + [f'A PostgreSQL database URL was detected, but the backend could not connect safely ({exc}). The app stayed in session-only mode.'],
                    ),
                )
            return PersistenceService(
                repositories=_build_repository_bundle(backend),
                status=PersistenceStatus(
                    enabled=True,
                    mode='postgres',
                    storage_target=backend.storage_target,
                    notes=notes + [f'Workspace state is persisted in PostgreSQL at {backend.storage_target} using {backend.driver_name}.'],
                ),
            )
        else:
            return PersistenceService(
                repositories=None,
                status=PersistenceStatus(
                    enabled=False,
                    mode='session',
                    storage_target='session-only',
                    notes=notes + ['The configured database URL is not supported yet, so the app stayed in session-only mode.'],
                ),
            )

    if not configured_sqlite_path:
        return PersistenceService(
            repositories=None,
            status=PersistenceStatus(
                enabled=False,
                mode='session',
                storage_target='session-only',
                notes=notes + ['No persistence database is configured, so workspace state remains session-scoped.'],
            ),
        )

    backend = SQLitePersistenceBackend(Path(configured_sqlite_path).expanduser())
    return PersistenceService(
        repositories=_build_repository_bundle(backend),
        status=PersistenceStatus(
            enabled=True,
            mode='sqlite',
            storage_target=backend.storage_target,
            notes=notes + [f'Workspace state is persisted in SQLite at {backend.storage_target}.'],
        ),
    )
