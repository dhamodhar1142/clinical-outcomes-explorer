from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import import_module
from pathlib import Path
from typing import Any

from src.runtime_paths import data_path

STORAGE_BACKEND_ENV = 'SMART_DATASET_ANALYZER_STORAGE_BACKEND'
STORAGE_ROOT_ENV = 'SMART_DATASET_ANALYZER_STORAGE_ROOT'
STORAGE_BUCKET_ENV = 'SMART_DATASET_ANALYZER_STORAGE_BUCKET'
STORAGE_PREFIX_ENV = 'SMART_DATASET_ANALYZER_STORAGE_PREFIX'
STORAGE_REGION_ENV = 'SMART_DATASET_ANALYZER_STORAGE_REGION'
STORAGE_ENDPOINT_ENV = 'SMART_DATASET_ANALYZER_STORAGE_ENDPOINT_URL'
STORAGE_ACCESS_KEY_ENV = 'SMART_DATASET_ANALYZER_STORAGE_ACCESS_KEY'
STORAGE_SECRET_KEY_ENV = 'SMART_DATASET_ANALYZER_STORAGE_SECRET_KEY'
DEFAULT_LOCAL_STORAGE_ROOT = data_path('storage')


def _utcnow() -> str:
    return datetime.now(UTC).isoformat()


def _slug(value: str | None) -> str:
    clean = ''.join(char.lower() if char.isalnum() else '-' for char in str(value or '').strip())
    while '--' in clean:
        clean = clean.replace('--', '-')
    return clean.strip('-') or 'artifact'


def _normalize_bytes(value: bytes | bytearray | str) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, bytearray):
        return bytes(value)
    return str(value).encode('utf-8')


def _normalize_storage_prefix(value: str | None) -> str:
    return str(value or '').strip().strip('/')


@dataclass(frozen=True)
class StorageStatus:
    enabled: bool
    mode: str
    storage_target: str
    notes: list[str]


class LocalArtifactRepository:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def save_bytes(self, relative_path: str, payload: bytes) -> dict[str, Any]:
        artifact_path = self.root / relative_path
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_bytes(payload)
        return {
            'relative_path': relative_path.replace('\\', '/'),
            'artifact_path': str(artifact_path),
            'size_bytes': len(payload),
            'stored_at': _utcnow(),
        }

    def load_bytes(self, *, relative_path: str = '', artifact_path: str = '') -> bytes:
        if artifact_path:
            candidate = Path(str(artifact_path))
        else:
            candidate = self.root / str(relative_path).replace('/', os.sep)
        return candidate.read_bytes()


class S3ArtifactRepository:
    def __init__(
        self,
        *,
        bucket_name: str,
        region_name: str | None = None,
        endpoint_url: str | None = None,
        access_key_id: str | None = None,
        secret_access_key: str | None = None,
        key_prefix: str = '',
    ) -> None:
        boto3 = import_module('boto3')
        session = boto3.session.Session()
        client_kwargs: dict[str, Any] = {}
        if region_name:
            client_kwargs['region_name'] = region_name
        if endpoint_url:
            client_kwargs['endpoint_url'] = endpoint_url
        if access_key_id:
            client_kwargs['aws_access_key_id'] = access_key_id
        if secret_access_key:
            client_kwargs['aws_secret_access_key'] = secret_access_key
        self.client = session.client('s3', **client_kwargs)
        self.bucket_name = bucket_name
        self.key_prefix = _normalize_storage_prefix(key_prefix)
        self.endpoint_url = endpoint_url or ''

    def _full_key(self, relative_path: str) -> str:
        cleaned = relative_path.replace('\\', '/').lstrip('/')
        if not self.key_prefix:
            return cleaned
        return f'{self.key_prefix}/{cleaned}'

    def save_bytes(self, relative_path: str, payload: bytes) -> dict[str, Any]:
        object_key = self._full_key(relative_path)
        self.client.put_object(Bucket=self.bucket_name, Key=object_key, Body=payload)
        artifact_path = f's3://{self.bucket_name}/{object_key}'
        return {
            'relative_path': relative_path.replace('\\', '/'),
            'artifact_path': artifact_path,
            'object_key': object_key,
            'bucket_name': self.bucket_name,
            'size_bytes': len(payload),
            'stored_at': _utcnow(),
            'endpoint_url': self.endpoint_url,
        }

    def load_bytes(self, *, relative_path: str = '', artifact_path: str = '') -> bytes:
        object_key = ''
        if artifact_path.startswith('s3://'):
            _, _, remainder = artifact_path.partition('s3://')
            bucket_name, _, object_key = remainder.partition('/')
            if bucket_name and bucket_name != self.bucket_name:
                raise FileNotFoundError(f'Artifact bucket mismatch for {artifact_path}.')
        if not object_key:
            object_key = self._full_key(relative_path)
        response = self.client.get_object(Bucket=self.bucket_name, Key=object_key)
        body = response.get('Body')
        return body.read() if body is not None else b''


class StorageService:
    def __init__(self, repository: Any | None, status: StorageStatus) -> None:
        self.repository = repository
        self.status = status

    @property
    def enabled(self) -> bool:
        return self.repository is not None and self.status.enabled

    def _workspace_prefix(self, workspace_identity: dict[str, Any] | None) -> str:
        workspace_id = str((workspace_identity or {}).get('workspace_id', 'guest-demo-workspace'))
        return _slug(workspace_id)

    def save_dataset_upload(
        self,
        workspace_identity: dict[str, Any] | None,
        *,
        dataset_name: str,
        file_name: str,
        payload: bytes | bytearray | str,
    ) -> dict[str, Any]:
        if not self.enabled or self.repository is None:
            return {'stored': False, 'mode': 'session', 'artifact_type': 'dataset_upload'}
        safe_name = _slug(file_name)
        relative_path = f"{self._workspace_prefix(workspace_identity)}/uploads/{_slug(dataset_name)}-{safe_name}.bin"
        saved = self.repository.save_bytes(relative_path, _normalize_bytes(payload))
        return {'stored': True, 'mode': self.status.mode, 'artifact_type': 'dataset_upload', **saved}

    def save_report_artifact(
        self,
        workspace_identity: dict[str, Any] | None,
        *,
        dataset_name: str,
        report_type: str,
        file_name: str,
        payload: bytes | bytearray | str,
    ) -> dict[str, Any]:
        if not self.enabled or self.repository is None:
            return {'stored': False, 'mode': 'session', 'artifact_type': 'report'}
        relative_path = (
            f"{self._workspace_prefix(workspace_identity)}/reports/"
            f"{_slug(dataset_name)}/{_slug(report_type)}-{_slug(file_name)}"
        )
        saved = self.repository.save_bytes(relative_path, _normalize_bytes(payload))
        return {'stored': True, 'mode': self.status.mode, 'artifact_type': 'report', **saved}

    def save_session_bundle(
        self,
        workspace_identity: dict[str, Any] | None,
        *,
        dataset_name: str,
        file_name: str,
        payload: bytes | bytearray | str,
    ) -> dict[str, Any]:
        if not self.enabled or self.repository is None:
            return {'stored': False, 'mode': 'session', 'artifact_type': 'session_bundle'}
        relative_path = (
            f"{self._workspace_prefix(workspace_identity)}/session_bundles/"
            f"{_slug(dataset_name)}-{_slug(file_name)}"
        )
        saved = self.repository.save_bytes(relative_path, _normalize_bytes(payload))
        return {'stored': True, 'mode': self.status.mode, 'artifact_type': 'session_bundle', **saved}

    def load_artifact_bytes(
        self,
        *,
        relative_path: str = '',
        artifact_path: str = '',
    ) -> bytes:
        if not self.enabled or self.repository is None:
            raise FileNotFoundError('Persistent artifact storage is not enabled for this environment.')
        if not hasattr(self.repository, 'load_bytes'):
            raise FileNotFoundError('The configured storage backend cannot retrieve stored artifacts.')
        return self.repository.load_bytes(relative_path=relative_path, artifact_path=artifact_path)


def build_storage_backend_health(storage_service: StorageService) -> dict[str, Any]:
    status = getattr(storage_service, 'status', None)
    mode = str(getattr(status, 'mode', 'session'))
    storage_target = str(getattr(status, 'storage_target', 'session-only'))
    notes = list(getattr(status, 'notes', []))
    if not storage_service.enabled or storage_service.repository is None:
        return {
            'status': 'Session fallback',
            'mode': mode,
            'storage_target': storage_target,
            'detail': 'Artifacts will stay in the active session because no persistent storage backend is currently enabled.',
            'notes': notes,
        }
    if mode == 'local':
        target = Path(storage_target)
        exists = target.exists()
        return {
            'status': 'Healthy' if exists else 'Unavailable',
            'mode': mode,
            'storage_target': storage_target,
            'detail': (
                f'Local artifact storage is ready at {storage_target}.'
                if exists
                else f'Local artifact storage target {storage_target} is not reachable.'
            ),
            'notes': notes,
        }
    if mode == 's3':
        repository = storage_service.repository
        try:
            bucket_name = str(getattr(repository, 'bucket_name', '') or '')
            response = repository.client.head_bucket(Bucket=bucket_name)
            return {
                'status': 'Healthy',
                'mode': mode,
                'storage_target': storage_target,
                'detail': f'Object storage bucket {bucket_name} is reachable for artifact persistence.',
                'notes': notes,
                'bucket_name': bucket_name,
                'response_metadata': dict(getattr(response, 'get', lambda *_: {})('ResponseMetadata', {})) if isinstance(response, dict) else {},
            }
        except Exception as error:
            return {
                'status': 'Unavailable',
                'mode': mode,
                'storage_target': storage_target,
                'detail': f'Object storage health check failed: {type(error).__name__}: {error}',
                'notes': notes,
            }
    return {
        'status': 'Unknown',
        'mode': mode,
        'storage_target': storage_target,
        'detail': 'Storage health could not be determined for the configured backend.',
        'notes': notes,
    }


def build_storage_service(
    *,
    backend: str | None = None,
    storage_root: str | os.PathLike[str] | None = None,
) -> StorageService:
    configured_backend = str(backend or os.getenv(STORAGE_BACKEND_ENV, 'local')).strip().lower() or 'local'
    configured_root = Path(str(storage_root or os.getenv(STORAGE_ROOT_ENV, '')).strip() or DEFAULT_LOCAL_STORAGE_ROOT).expanduser()
    configured_bucket = str(os.getenv(STORAGE_BUCKET_ENV, '')).strip()
    configured_prefix = _normalize_storage_prefix(os.getenv(STORAGE_PREFIX_ENV, ''))
    configured_region = str(os.getenv(STORAGE_REGION_ENV, '')).strip()
    configured_endpoint = str(os.getenv(STORAGE_ENDPOINT_ENV, '')).strip()
    configured_access_key = str(os.getenv(STORAGE_ACCESS_KEY_ENV, '')).strip()
    configured_secret_key = str(os.getenv(STORAGE_SECRET_KEY_ENV, '')).strip()
    notes = [
        f'Set {STORAGE_BACKEND_ENV} to switch storage backends in the future.',
        f'Set {STORAGE_ROOT_ENV} to change where local storage artifacts are written.',
        f'Set {STORAGE_BUCKET_ENV} and {STORAGE_PREFIX_ENV} to enable object-storage artifact persistence.',
    ]

    if configured_backend in {'s3', 'object', 'minio'}:
        if not configured_bucket:
            return StorageService(
                repository=None,
                status=StorageStatus(
                    enabled=False,
                    mode='session',
                    storage_target='session-only',
                    notes=notes + [f'The {configured_backend} backend was selected, but {STORAGE_BUCKET_ENV} is not configured. The app stayed in session-only artifact mode.'],
                ),
            )
        try:
            repository = S3ArtifactRepository(
                bucket_name=configured_bucket,
                region_name=configured_region or None,
                endpoint_url=configured_endpoint or None,
                access_key_id=configured_access_key or None,
                secret_access_key=configured_secret_key or None,
                key_prefix=configured_prefix,
            )
        except Exception as exc:
            return StorageService(
                repository=None,
                status=StorageStatus(
                    enabled=False,
                    mode='session',
                    storage_target='session-only',
                    notes=notes + [f'The object-storage backend could not be initialized safely ({exc}). The app stayed in session-only artifact mode.'],
                ),
            )
        target = f"s3://{configured_bucket}/{configured_prefix}".rstrip('/')
        if configured_endpoint:
            target = f'{target} via {configured_endpoint}'
        return StorageService(
            repository=repository,
            status=StorageStatus(
                enabled=True,
                mode='s3',
                storage_target=target,
                notes=notes + [f'Artifacts are stored in the object-storage bucket {configured_bucket}.'],
            ),
        )

    if configured_backend not in {'local', 'filesystem'}:
        return StorageService(
            repository=None,
            status=StorageStatus(
                enabled=False,
                mode='session',
                storage_target='session-only',
                notes=notes + ['The configured storage backend is not implemented yet, so the app stayed in session-only artifact mode.'],
            ),
        )

    repository = LocalArtifactRepository(configured_root)
    return StorageService(
        repository=repository,
        status=StorageStatus(
            enabled=True,
            mode='local',
            storage_target=str(configured_root),
            notes=notes + [f'Artifacts are stored on the local filesystem at {configured_root}.'],
        ),
    )
