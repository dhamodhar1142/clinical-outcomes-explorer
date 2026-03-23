from __future__ import annotations

import os
import re
import secrets
import sqlite3
import time
import urllib.parse
import urllib.request
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from src.persistence import DATABASE_URL_ENV, SQLITE_PATH_ENV
from src.workspace import build_workspace_identity

AUTH_PROVIDER_ENV = 'SMART_DATASET_ANALYZER_AUTH_PROVIDER'
AUTH_ISSUER_ENV = 'SMART_DATASET_ANALYZER_AUTH_ISSUER'
AUTH_AUDIENCE_ENV = 'SMART_DATASET_ANALYZER_AUTH_AUDIENCE'
AUTH_CLIENT_ID_ENV = 'SMART_DATASET_ANALYZER_AUTH_CLIENT_ID'
AUTH_CLIENT_SECRET_ENV = 'SMART_DATASET_ANALYZER_AUTH_CLIENT_SECRET'
AUTH_TENANT_ID_ENV = 'SMART_DATASET_ANALYZER_AUTH_TENANT_ID'
AUTH_REDIRECT_URI_ENV = 'SMART_DATASET_ANALYZER_AUTH_REDIRECT_URI'
AUTH_SCOPES_ENV = 'SMART_DATASET_ANALYZER_AUTH_SCOPES'
AUTH_SESSION_TIMEOUT_ENV = 'SMART_DATASET_ANALYZER_AUTH_SESSION_TIMEOUT_MINUTES'
AUTH_2FA_ISSUER_ENV = 'SMART_DATASET_ANALYZER_2FA_ISSUER'
AUTH_2FA_SECRET_ENV = 'SMART_DATASET_ANALYZER_2FA_SECRET'

OIDC_PROVIDER_PRESETS = {
    'okta': {
        'label': 'Okta',
        'scopes': ['openid', 'profile', 'email', 'offline_access'],
    },
    'azure-ad': {
        'label': 'Azure AD',
        'scopes': ['openid', 'profile', 'email', 'offline_access'],
    },
    'azure': {
        'label': 'Azure AD',
        'scopes': ['openid', 'profile', 'email', 'offline_access'],
    },
    'google-workspace': {
        'label': 'Google Workspace',
        'scopes': ['openid', 'profile', 'email'],
    },
    'google': {
        'label': 'Google Workspace',
        'scopes': ['openid', 'profile', 'email'],
    },
}

WORKSPACE_ROLE_OPTIONS = ['Owner', 'Admin', 'Analyst', 'Viewer']
WORKSPACE_ROLE_RANK = {
    'viewer': 1,
    'analyst': 2,
    'admin': 3,
    'owner': 4,
}
WORKSPACE_ROLE_PERMISSIONS = {
    'owner': {'workspace_manage', 'workspace_admin', 'workspace_write', 'workspace_read', 'dataset_upload', 'export_generate', 'export_download', 'settings_manage'},
    'admin': {'workspace_admin', 'workspace_write', 'workspace_read', 'dataset_upload', 'export_generate', 'export_download', 'settings_manage'},
    'analyst': {'workspace_write', 'workspace_read', 'dataset_upload', 'export_generate', 'export_download'},
    'viewer': {'workspace_read'},
}


def _utcnow() -> str:
    return datetime.now(UTC).isoformat()


def _utcnow_dt() -> datetime:
    return datetime.now(UTC)


def _parse_iso8601(value: str | None) -> datetime | None:
    text = str(value or '').strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace('Z', '+00:00'))
    except ValueError:
        return None


def _slug(value: str) -> str:
    slug = re.sub(r'[^a-z0-9]+', '-', value.strip().lower()).strip('-')
    return slug or 'demo'


def _normalize_email(value: str | None) -> str:
    return str(value or '').strip().lower()


def _normalize_provider(value: str | None) -> str:
    provider = str(value or '').strip().lower()
    return provider or 'guest'


def _normalize_provider_preset(value: str | None) -> str:
    provider = _normalize_provider(value)
    if provider in {'azuread', 'azure_ad'}:
        return 'azure-ad'
    if provider in {'googleworkspace', 'google_workspace'}:
        return 'google-workspace'
    return provider


def _session_timeout_minutes() -> int:
    raw = str(os.getenv(AUTH_SESSION_TIMEOUT_ENV, '720')).strip()
    try:
        return max(15, min(7 * 24 * 60, int(raw)))
    except ValueError:
        return 720


def _totp_secret() -> str:
    return str(os.getenv(AUTH_2FA_SECRET_ENV, '')).strip()


def _totp_issuer() -> str:
    return str(os.getenv(AUTH_2FA_ISSUER_ENV, 'Clinverity')).strip() or 'Clinverity'


def normalize_workspace_role(value: str | None) -> str:
    role = str(value or '').strip().lower()
    if role in WORKSPACE_ROLE_PERMISSIONS:
        return role
    return 'viewer'


def format_workspace_role(role: str | None) -> str:
    return normalize_workspace_role(role).title()


def get_workspace_role_permissions(role: str | None) -> set[str]:
    return set(WORKSPACE_ROLE_PERMISSIONS.get(normalize_workspace_role(role), WORKSPACE_ROLE_PERMISSIONS['viewer']))


def get_workspace_role_rank(role: str | None) -> int:
    return int(WORKSPACE_ROLE_RANK.get(normalize_workspace_role(role), WORKSPACE_ROLE_RANK['viewer']))


def workspace_role_at_least(role: str | None, minimum_role: str | None) -> bool:
    return get_workspace_role_rank(role) >= get_workspace_role_rank(minimum_role)


def workspace_can_access(role: str | None, permission: str) -> bool:
    return permission in get_workspace_role_permissions(role)


def workspace_identity_can_access(
    workspace_identity: dict[str, Any] | None,
    permission: str,
    *,
    allow_guest_demo: bool = True,
) -> bool:
    identity = workspace_identity or {}
    if allow_guest_demo and str(identity.get('auth_mode', 'guest')) == 'guest':
        return True
    if str(identity.get('auth_mode', 'guest')) != 'guest' and identity.get('membership_validated') is False:
        return False
    if normalize_workspace_role(identity.get('role')) == 'owner' and identity.get('ownership_validated') is False:
        return False
    return workspace_can_access(str(identity.get('role', 'viewer')), permission)


def workspace_identity_has_minimum_role(
    workspace_identity: dict[str, Any] | None,
    minimum_role: str,
    *,
    allow_guest_demo: bool = True,
) -> bool:
    identity = workspace_identity or {}
    if allow_guest_demo and str(identity.get('auth_mode', 'guest')) == 'guest':
        return True
    if str(identity.get('auth_mode', 'guest')) != 'guest' and identity.get('membership_validated') is False:
        return False
    if normalize_workspace_role(identity.get('role')) == 'owner' and identity.get('ownership_validated') is False:
        return False
    return workspace_role_at_least(str(identity.get('role', 'viewer')), minimum_role)


def workspace_identity_matches_workspace(
    workspace_identity: dict[str, Any] | None,
    workspace_id: str | None,
    *,
    allow_guest_demo: bool = True,
) -> bool:
    identity = workspace_identity or {}
    auth_mode = str(identity.get('auth_mode', 'guest'))
    if allow_guest_demo and auth_mode == 'guest':
        return True
    expected_workspace_id = str(workspace_id or '').strip()
    current_workspace_id = str(identity.get('workspace_id', '')).strip()
    if not expected_workspace_id or not current_workspace_id:
        return False
    if current_workspace_id != expected_workspace_id:
        return False
    membership_validated = identity.get('membership_validated')
    if membership_validated is False:
        return False
    if normalize_workspace_role(identity.get('role')) == 'owner' and identity.get('ownership_validated') is False:
        return False
    return True


def build_permission_denied_message(
    workspace_identity: dict[str, Any] | None,
    permission: str,
    *,
    resource_label: str,
) -> str:
    identity = workspace_identity or {}
    role_label = format_workspace_role(str(identity.get('role', 'viewer')))
    workspace_name = str(identity.get('workspace_name', 'Current workspace'))
    return (
        f"{role_label} access for '{workspace_name}' does not include {resource_label}. "
        f"This action requires the '{permission}' workspace permission."
    )


def build_workspace_role_message(
    workspace_identity: dict[str, Any] | None,
    *,
    minimum_role: str,
    resource_label: str,
) -> str:
    identity = workspace_identity or {}
    role_label = format_workspace_role(str(identity.get('role', 'viewer')))
    workspace_name = str(identity.get('workspace_name', 'Current workspace'))
    return (
        f"{resource_label.title()} in '{workspace_name}' requires at least the {format_workspace_role(minimum_role)} role. "
        f'The current signed-in workspace role is {role_label}.'
    )


def build_workspace_boundary_message(
    workspace_identity: dict[str, Any] | None,
    *,
    workspace_id: str | None,
    resource_label: str,
) -> str:
    identity = workspace_identity or {}
    workspace_name = str(identity.get('workspace_name', 'Current workspace'))
    current_workspace_id = str(identity.get('workspace_id', 'unknown-workspace'))
    expected_workspace_id = str(workspace_id or 'unknown-workspace')
    return (
        f"{resource_label.title()} is restricted to workspace '{workspace_name}'. "
        f'The active signed-in workspace scope ({current_workspace_id}) does not match the required workspace scope ({expected_workspace_id}).'
    )


def enforce_workspace_permission(
    workspace_identity: dict[str, Any] | None,
    permission: str,
    *,
    resource_label: str,
    allow_guest_demo: bool = True,
) -> None:
    if workspace_identity_can_access(workspace_identity, permission, allow_guest_demo=allow_guest_demo):
        return
    raise PermissionError(
        build_permission_denied_message(
            workspace_identity,
            permission,
            resource_label=resource_label,
        )
    )


def enforce_workspace_boundary(
    workspace_identity: dict[str, Any] | None,
    *,
    workspace_id: str | None,
    resource_label: str,
    allow_guest_demo: bool = True,
) -> None:
    if workspace_identity_matches_workspace(
        workspace_identity,
        workspace_id,
        allow_guest_demo=allow_guest_demo,
    ):
        return
    raise PermissionError(
        build_workspace_boundary_message(
            workspace_identity,
            workspace_id=workspace_id,
            resource_label=resource_label,
        )
    )


def enforce_workspace_minimum_role(
    workspace_identity: dict[str, Any] | None,
    minimum_role: str,
    *,
    resource_label: str,
    allow_guest_demo: bool = True,
) -> None:
    if workspace_identity_has_minimum_role(
        workspace_identity,
        minimum_role,
        allow_guest_demo=allow_guest_demo,
    ):
        return
    raise PermissionError(
        build_workspace_role_message(
            workspace_identity,
            minimum_role=minimum_role,
            resource_label=resource_label,
        )
    )


def build_workspace_security_summary(
    workspace_identity: dict[str, Any],
    auth_status: AuthStatus | None,
) -> dict[str, Any]:
    role = normalize_workspace_role(workspace_identity.get('role', 'viewer'))
    permissions = sorted(get_workspace_role_permissions(role))
    storage_target = str(getattr(auth_status, 'storage_target', 'session-only'))
    signed_in = str(workspace_identity.get('auth_mode', 'guest')) != 'guest'
    summary_cards = [
        {'label': 'Workspace Owner', 'value': str(workspace_identity.get('owner_label', 'Guest session'))},
        {'label': 'Workspace Role', 'value': format_workspace_role(role)},
        {'label': 'Auth Mode', 'value': str(workspace_identity.get('auth_mode', 'guest'))},
        {'label': 'Auth Provider', 'value': str(getattr(auth_status, 'provider_name', workspace_identity.get('provider', 'guest')))},
        {'label': 'Session Status', 'value': 'Active' if workspace_identity.get('session_id') else 'Guest session'},
        {'label': '2FA', 'value': 'Verified' if workspace_identity.get('mfa_verified') else 'Required' if workspace_identity.get('mfa_required') else 'Not required'},
        {'label': 'Storage Target', 'value': storage_target},
    ]
    boundaries_table = [
        {
            'control_area': 'Workspace ownership boundary',
            'status': 'Owned workspace' if signed_in else 'Guest workspace',
            'detail': 'Saved assets, notes, and usage are scoped to the active workspace identity.',
        },
        {
            'control_area': 'Tenant boundary',
            'status': 'Validated membership'
            if workspace_identity.get('membership_validated') is True
            else 'Guest demo scope'
            if not signed_in
            else 'Identity-scoped boundary',
            'detail': 'Signed-in actions can now enforce workspace-scope checks so assets, uploads, and exports stay inside the active workspace boundary.',
        },
        {
            'control_area': 'Membership source',
            'status': str(workspace_identity.get('membership_source', 'guest')).replace('-', ' ').title(),
            'detail': 'Workspace membership records now distinguish local, hosted-prepared, and session-scoped access paths so future external identity providers can map cleanly into tenant membership enforcement.',
        },
        {
            'control_area': 'Workspace role enforcement',
            'status': format_workspace_role(role),
            'detail': (
                'Current workspace permissions: ' + ', '.join(permissions)
                if permissions
                else 'No workspace permissions were resolved.'
            ),
        },
        {
            'control_area': 'Verified role state',
            'status': 'Verified'
            if (
                not signed_in
                or workspace_identity.get('membership_validated') is True
                or workspace_identity.get('membership_validated') is None
            )
            else 'Verification needed',
            'detail': 'Signed-in workspace actions now use explicit membership and minimum-role checks before allowing uploads, exports, and shared settings actions.',
        },
        {
            'control_area': 'Session management',
            'status': 'Tracked session' if workspace_identity.get('session_id') else 'Guest session',
            'detail': 'Signed-in sessions can now be tracked, expired, and revoked without changing the core workspace UX.',
        },
        {
            'control_area': 'Two-factor authentication',
            'status': 'Verified'
            if workspace_identity.get('mfa_verified')
            else 'Pending verification'
            if workspace_identity.get('mfa_required')
            else 'Optional',
            'detail': 'A TOTP-style second factor can be required for signed-in sessions when the platform is configured with a shared MFA secret.',
        },
        {
            'control_area': 'Persistence boundary',
            'status': storage_target,
            'detail': 'Workspace state can remain session-local or move into SQLite-backed foundations without changing the UI workflow.',
        },
        {
            'control_area': 'Dataset access',
            'status': 'Upload enabled' if workspace_identity_can_access(workspace_identity, 'dataset_upload') else 'Read-only',
            'detail': 'Signed-in workspaces can now apply role-aware upload boundaries while guest/demo mode stays available for unrestricted exploration.',
        },
        {
            'control_area': 'Export permissions',
            'status': 'Export enabled' if workspace_identity_can_access(workspace_identity, 'export_download') else 'Restricted',
            'detail': 'Export generation and download decisions now consider both audience role and workspace ownership permissions.',
        },
    ]
    notes = [
        str(workspace_identity.get('workspace_access_summary', 'Workspace access summary is not available yet.')),
        (
            f"Tenant scope: {workspace_identity.get('tenant_id') or 'demo-local'} | "
            f"Auth provider mode: {getattr(auth_status, 'provider_mode', 'guest')}"
        ),
        (
            f"OIDC provider: {getattr(auth_status, 'provider_name', 'guest')} | "
            f"Session expires: {workspace_identity.get('session_expires_at') or 'guest session'}"
        ),
        'This foundation is designed to support future tenant-aware RBAC and stronger ownership boundaries without removing guest/demo access.',
    ]
    return {
        'summary_cards': summary_cards,
        'boundaries_table': boundaries_table,
        'notes': notes,
    }


def build_signed_in_auth_session(
    display_name: str,
    email: str,
    *,
    provider: str = 'local-demo',
    auth_mode: str = 'local',
    role: str = 'owner',
) -> dict[str, Any]:
    clean_display = str(display_name or '').strip() or 'Local Demo User'
    clean_email = _normalize_email(email)
    normalized_role = normalize_workspace_role(role)
    user_id = f"{auth_mode}::{_slug(clean_email or clean_display)}"
    return {
        'signed_in': True,
        'auth_mode': auth_mode,
        'provider': provider,
        'status_label': 'Signed in',
        'user_id': user_id,
        'display_name': clean_display,
        'email': clean_email,
        'role': normalized_role,
        'role_label': format_workspace_role(normalized_role),
        'permissions': sorted(get_workspace_role_permissions(normalized_role)),
    }


def build_hosted_auth_session(
    *,
    display_name: str | None,
    email: str | None,
    provider_subject: str | None,
    provider: str,
    tenant_id: str | None = None,
    role: str = 'viewer',
) -> dict[str, Any]:
    clean_display = str(display_name or '').strip() or 'Hosted User'
    clean_email = _normalize_email(email)
    normalized_role = normalize_workspace_role(role)
    normalized_provider = _normalize_provider(provider)
    subject = str(provider_subject or '').strip() or _slug(clean_email or clean_display)
    user_id = f"{normalized_provider}::{_slug(subject)}"
    return {
        'signed_in': True,
        'auth_mode': 'hosted',
        'provider': normalized_provider,
        'status_label': 'Hosted sign-in',
        'user_id': user_id,
        'display_name': clean_display,
        'email': clean_email,
        'role': normalized_role,
        'role_label': format_workspace_role(normalized_role),
        'permissions': sorted(get_workspace_role_permissions(normalized_role)),
        'provider_subject': subject,
        'tenant_id': str(tenant_id or '').strip(),
    }


def _decode_base32(secret: str) -> bytes:
    import base64

    normalized = re.sub(r'[^A-Z2-7]', '', secret.strip().upper())
    if not normalized:
        return b''
    padding = '=' * ((8 - len(normalized) % 8) % 8)
    return base64.b32decode(normalized + padding, casefold=True)


def generate_totp_code(secret: str, *, timestamp: int | None = None, interval_seconds: int = 30, digits: int = 6) -> str:
    import hashlib
    import hmac

    key = _decode_base32(secret)
    if not key:
        return ''
    current_timestamp = int(timestamp if timestamp is not None else time.time())
    counter = current_timestamp // interval_seconds
    counter_bytes = counter.to_bytes(8, byteorder='big')
    digest = hmac.new(key, counter_bytes, hashlib.sha1).digest()
    offset = digest[-1] & 0x0F
    binary_code = int.from_bytes(digest[offset:offset + 4], byteorder='big') & 0x7FFFFFFF
    return str(binary_code % (10 ** digits)).zfill(digits)


def verify_two_factor_code(secret: str | None, code: str | None, *, timestamp: int | None = None, window: int = 1) -> bool:
    clean_secret = str(secret or '').strip()
    clean_code = str(code or '').strip()
    if not clean_secret or not clean_code:
        return False
    current_timestamp = int(timestamp if timestamp is not None else time.time())
    for step in range(-window, window + 1):
        candidate = generate_totp_code(clean_secret, timestamp=current_timestamp + (step * 30))
        if candidate and candidate == clean_code:
            return True
    return False


def build_oidc_authorization_url(
    metadata: OIDCProviderMetadata,
    *,
    state: str,
    nonce: str,
    prompt: str = 'select_account',
) -> str:
    query = urllib.parse.urlencode(
        {
            'client_id': metadata.client_id,
            'response_type': 'code',
            'redirect_uri': metadata.redirect_uri,
            'scope': ' '.join(metadata.scopes),
            'state': state,
            'nonce': nonce,
            'prompt': prompt,
        }
    )
    return f'{metadata.authorization_endpoint}?{query}'


def build_guest_auth_session() -> dict[str, Any]:
    return {
        'signed_in': False,
        'auth_mode': 'guest',
        'provider': 'guest',
        'status_label': 'Guest mode',
        'user_id': 'guest-user',
        'display_name': 'Guest User',
        'email': '',
        'role': 'viewer',
        'role_label': 'Viewer',
        'permissions': sorted(get_workspace_role_permissions('viewer')),
    }


@dataclass(frozen=True)
class AuthStatus:
    enabled: bool
    mode: str
    storage_target: str
    notes: list[str]
    provider_mode: str = 'guest'
    provider_name: str = 'guest'
    hosted_configured: bool = False
    tenant_id: str = ''


@dataclass(frozen=True)
class HostedAuthConfig:
    enabled: bool
    provider: str
    provider_label: str
    issuer: str
    audience: str
    client_id: str
    client_secret: str
    tenant_id: str
    redirect_uri: str
    scopes: list[str]
    notes: list[str]


@dataclass(frozen=True)
class OIDCProviderMetadata:
    provider: str
    provider_label: str
    issuer: str
    authorization_endpoint: str
    token_endpoint: str
    userinfo_endpoint: str
    jwks_uri: str
    scopes: list[str]
    client_id: str
    client_secret: str
    redirect_uri: str


@dataclass(frozen=True)
class AuthSessionRecord:
    session_id: str
    user_id: str
    workspace_id: str
    provider: str
    auth_mode: str
    created_at: str
    expires_at: str
    revoked_at: str
    mfa_required: bool
    mfa_verified: bool


@dataclass(frozen=True)
class WorkspaceInvitation:
    invitation_id: str
    workspace_id: str
    workspace_name: str
    email: str
    role: str
    invited_by_user_id: str
    invited_by_name: str
    provider: str
    tenant_id: str
    status: str
    created_at: str
    expires_at: str


@dataclass(frozen=True)
class WorkspaceMembership:
    workspace_id: str
    user_id: str
    role: str
    workspace_name: str
    membership_source: str
    provider: str
    tenant_id: str
    ownership_validated: bool


class LocalAuthRepository:
    def __init__(self, sqlite_path: Path) -> None:
        self.sqlite_path = sqlite_path
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.sqlite_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with closing(self._connect()) as connection:
            connection.executescript(
                '''
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    email TEXT NOT NULL UNIQUE,
                    display_name TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    external_subject TEXT NOT NULL DEFAULT '',
                    tenant_id TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS workspace_owners (
                    workspace_id TEXT PRIMARY KEY,
                    workspace_name TEXT NOT NULL,
                    owner_user_id TEXT NOT NULL,
                    tenant_id TEXT NOT NULL DEFAULT '',
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (owner_user_id) REFERENCES users(user_id)
                );

                CREATE TABLE IF NOT EXISTS workspace_members (
                    workspace_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    workspace_name TEXT NOT NULL,
                    membership_source TEXT NOT NULL DEFAULT 'local',
                    provider TEXT NOT NULL DEFAULT 'local-demo',
                    tenant_id TEXT NOT NULL DEFAULT '',
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (workspace_id, user_id),
                    FOREIGN KEY (user_id) REFERENCES users(user_id)
                );

                CREATE TABLE IF NOT EXISTS auth_sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    workspace_id TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    auth_mode TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    revoked_at TEXT NOT NULL DEFAULT '',
                    mfa_required INTEGER NOT NULL DEFAULT 0,
                    mfa_verified INTEGER NOT NULL DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS workspace_invitations (
                    invitation_id TEXT PRIMARY KEY,
                    workspace_id TEXT NOT NULL,
                    workspace_name TEXT NOT NULL,
                    email TEXT NOT NULL,
                    role TEXT NOT NULL,
                    invited_by_user_id TEXT NOT NULL,
                    invited_by_name TEXT NOT NULL,
                    provider TEXT NOT NULL DEFAULT 'local-demo',
                    tenant_id TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL DEFAULT 'pending',
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL
                );
                '''
            )
            self._ensure_column(connection, 'users', 'external_subject', "TEXT NOT NULL DEFAULT ''")
            self._ensure_column(connection, 'users', 'tenant_id', "TEXT NOT NULL DEFAULT ''")
            self._ensure_column(connection, 'workspace_owners', 'tenant_id', "TEXT NOT NULL DEFAULT ''")
            self._ensure_column(connection, 'workspace_members', 'membership_source', "TEXT NOT NULL DEFAULT 'local'")
            self._ensure_column(connection, 'workspace_members', 'provider', "TEXT NOT NULL DEFAULT 'local-demo'")
            self._ensure_column(connection, 'workspace_members', 'tenant_id', "TEXT NOT NULL DEFAULT ''")
            connection.commit()

    def _ensure_column(self, connection: sqlite3.Connection, table_name: str, column_name: str, definition: str) -> None:
        columns = {
            str(row['name'])
            for row in connection.execute(f'PRAGMA table_info({table_name})').fetchall()
        }
        if column_name in columns:
            return
        connection.execute(f'ALTER TABLE {table_name} ADD COLUMN {column_name} {definition}')

    def upsert_user(
        self,
        user_id: str,
        email: str,
        display_name: str,
        provider: str,
        *,
        external_subject: str = '',
        tenant_id: str = '',
    ) -> None:
        with closing(self._connect()) as connection:
            connection.execute(
                '''
                INSERT INTO users (user_id, email, display_name, provider, external_subject, tenant_id, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    email = excluded.email,
                    display_name = excluded.display_name,
                    provider = excluded.provider,
                    external_subject = excluded.external_subject,
                    tenant_id = excluded.tenant_id,
                    updated_at = excluded.updated_at
                ''',
                (user_id, email, display_name, provider, external_subject, tenant_id, _utcnow(), _utcnow()),
            )
            connection.commit()

    def upsert_workspace_owner(
        self,
        workspace_id: str,
        workspace_name: str,
        owner_user_id: str,
        *,
        tenant_id: str = '',
    ) -> None:
        with closing(self._connect()) as connection:
            connection.execute(
                '''
                INSERT INTO workspace_owners (workspace_id, workspace_name, owner_user_id, tenant_id, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(workspace_id) DO UPDATE SET
                    workspace_name = excluded.workspace_name,
                    owner_user_id = excluded.owner_user_id,
                    tenant_id = excluded.tenant_id,
                    updated_at = excluded.updated_at
                ''',
                (workspace_id, workspace_name, owner_user_id, tenant_id, _utcnow()),
            )
            connection.commit()

    def upsert_workspace_member(
        self,
        workspace_id: str,
        workspace_name: str,
        user_id: str,
        role: str,
        *,
        membership_source: str = 'local',
        provider: str = 'local-demo',
        tenant_id: str = '',
    ) -> None:
        with closing(self._connect()) as connection:
            connection.execute(
                '''
                INSERT INTO workspace_members (
                    workspace_id,
                    user_id,
                    role,
                    workspace_name,
                    membership_source,
                    provider,
                    tenant_id,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(workspace_id, user_id) DO UPDATE SET
                    role = excluded.role,
                    workspace_name = excluded.workspace_name,
                    membership_source = excluded.membership_source,
                    provider = excluded.provider,
                    tenant_id = excluded.tenant_id,
                    updated_at = excluded.updated_at
                ''',
                (
                    workspace_id,
                    user_id,
                    normalize_workspace_role(role),
                    workspace_name,
                    str(membership_source or 'local'),
                    _normalize_provider(provider or 'local-demo'),
                    str(tenant_id or ''),
                    _utcnow(),
                ),
            )
            connection.commit()

    def load_workspace_member_role(self, workspace_id: str, user_id: str) -> str | None:
        with closing(self._connect()) as connection:
            row = connection.execute(
                '''
                SELECT role
                FROM workspace_members
                WHERE workspace_id = ? AND user_id = ?
                ''',
                (workspace_id, user_id),
            ).fetchone()
        if row is None:
            return None
        return normalize_workspace_role(str(row['role']))

    def load_workspace_membership(self, workspace_id: str, user_id: str) -> WorkspaceMembership | None:
        with closing(self._connect()) as connection:
            row = connection.execute(
                '''
                SELECT workspace_id, user_id, role, workspace_name, membership_source, provider, tenant_id
                FROM workspace_members
                WHERE workspace_id = ? AND user_id = ?
                ''',
                (workspace_id, user_id),
            ).fetchone()
        if row is None:
            return None
        role = normalize_workspace_role(str(row['role']))
        return WorkspaceMembership(
            workspace_id=str(row['workspace_id']),
            user_id=str(row['user_id']),
            role=role,
            workspace_name=str(row['workspace_name']),
            membership_source=str(row['membership_source'] or 'local'),
            provider=_normalize_provider(str(row['provider'] or 'local-demo')),
            tenant_id=str(row['tenant_id'] or ''),
            ownership_validated=self.is_workspace_owner(str(row['workspace_id']), str(row['user_id'])),
        )

    def is_workspace_member(self, workspace_id: str, user_id: str) -> bool:
        with closing(self._connect()) as connection:
            row = connection.execute(
                '''
                SELECT 1
                FROM workspace_members
                WHERE workspace_id = ? AND user_id = ?
                ''',
                (workspace_id, user_id),
            ).fetchone()
        return row is not None

    def is_workspace_owner(self, workspace_id: str, user_id: str) -> bool:
        with closing(self._connect()) as connection:
            row = connection.execute(
                '''
                SELECT 1
                FROM workspace_owners
                WHERE workspace_id = ? AND owner_user_id = ?
                ''',
                (workspace_id, user_id),
            ).fetchone()
        return row is not None

    def create_auth_session(
        self,
        *,
        user_id: str,
        workspace_id: str,
        provider: str,
        auth_mode: str,
        expires_at: str,
        mfa_required: bool,
        mfa_verified: bool,
    ) -> AuthSessionRecord:
        session_id = secrets.token_urlsafe(18)
        with closing(self._connect()) as connection:
            connection.execute(
                '''
                INSERT INTO auth_sessions (
                    session_id, user_id, workspace_id, provider, auth_mode, created_at, expires_at, revoked_at, mfa_required, mfa_verified
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''',
                (
                    session_id,
                    user_id,
                    workspace_id,
                    provider,
                    auth_mode,
                    _utcnow(),
                    expires_at,
                    '',
                    int(mfa_required),
                    int(mfa_verified),
                ),
            )
            connection.commit()
        return self.get_auth_session(session_id)  # type: ignore[return-value]

    def get_auth_session(self, session_id: str) -> AuthSessionRecord | None:
        with closing(self._connect()) as connection:
            row = connection.execute(
                '''
                SELECT session_id, user_id, workspace_id, provider, auth_mode, created_at, expires_at, revoked_at, mfa_required, mfa_verified
                FROM auth_sessions
                WHERE session_id = ?
                ''',
                (session_id,),
            ).fetchone()
        if row is None:
            return None
        return AuthSessionRecord(
            session_id=str(row['session_id']),
            user_id=str(row['user_id']),
            workspace_id=str(row['workspace_id']),
            provider=str(row['provider']),
            auth_mode=str(row['auth_mode']),
            created_at=str(row['created_at']),
            expires_at=str(row['expires_at']),
            revoked_at=str(row['revoked_at'] or ''),
            mfa_required=bool(row['mfa_required']),
            mfa_verified=bool(row['mfa_verified']),
        )

    def list_auth_sessions(self, user_id: str, workspace_id: str | None = None, *, include_revoked: bool = False) -> list[AuthSessionRecord]:
        query = '''
            SELECT session_id, user_id, workspace_id, provider, auth_mode, created_at, expires_at, revoked_at, mfa_required, mfa_verified
            FROM auth_sessions
            WHERE user_id = ?
        '''
        params: list[Any] = [user_id]
        if workspace_id:
            query += ' AND workspace_id = ?'
            params.append(workspace_id)
        if not include_revoked:
            query += " AND revoked_at = ''"
        query += ' ORDER BY created_at DESC'
        with closing(self._connect()) as connection:
            rows = connection.execute(query, params).fetchall()
        return [
            AuthSessionRecord(
                session_id=str(row['session_id']),
                user_id=str(row['user_id']),
                workspace_id=str(row['workspace_id']),
                provider=str(row['provider']),
                auth_mode=str(row['auth_mode']),
                created_at=str(row['created_at']),
                expires_at=str(row['expires_at']),
                revoked_at=str(row['revoked_at'] or ''),
                mfa_required=bool(row['mfa_required']),
                mfa_verified=bool(row['mfa_verified']),
            )
            for row in rows
        ]

    def revoke_auth_session(self, session_id: str) -> None:
        with closing(self._connect()) as connection:
            connection.execute(
                '''
                UPDATE auth_sessions
                SET revoked_at = ?
                WHERE session_id = ?
                ''',
                (_utcnow(), session_id),
            )
            connection.commit()

    def prune_expired_sessions(self) -> int:
        with closing(self._connect()) as connection:
            before = connection.total_changes
            connection.execute(
                '''
                DELETE FROM auth_sessions
                WHERE expires_at < ?
                ''',
                (_utcnow(),),
            )
            connection.commit()
            return int(connection.total_changes - before)

    def create_workspace_invitation(
        self,
        *,
        workspace_id: str,
        workspace_name: str,
        email: str,
        role: str,
        invited_by_user_id: str,
        invited_by_name: str,
        provider: str,
        tenant_id: str,
        expires_at: str,
    ) -> WorkspaceInvitation:
        invitation_id = secrets.token_urlsafe(18)
        with closing(self._connect()) as connection:
            connection.execute(
                '''
                INSERT INTO workspace_invitations (
                    invitation_id, workspace_id, workspace_name, email, role, invited_by_user_id, invited_by_name, provider, tenant_id, status, created_at, expires_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''',
                (
                    invitation_id,
                    workspace_id,
                    workspace_name,
                    _normalize_email(email),
                    normalize_workspace_role(role),
                    invited_by_user_id,
                    invited_by_name,
                    _normalize_provider(provider),
                    str(tenant_id or ''),
                    'pending',
                    _utcnow(),
                    expires_at,
                ),
            )
            connection.commit()
        invitation = self.get_workspace_invitation(invitation_id)
        if invitation is None:
            raise RuntimeError('Workspace invitation could not be loaded after creation.')
        return invitation

    def get_workspace_invitation(self, invitation_id: str) -> WorkspaceInvitation | None:
        with closing(self._connect()) as connection:
            row = connection.execute(
                '''
                SELECT invitation_id, workspace_id, workspace_name, email, role, invited_by_user_id, invited_by_name, provider, tenant_id, status, created_at, expires_at
                FROM workspace_invitations
                WHERE invitation_id = ?
                ''',
                (invitation_id,),
            ).fetchone()
        if row is None:
            return None
        return WorkspaceInvitation(
            invitation_id=str(row['invitation_id']),
            workspace_id=str(row['workspace_id']),
            workspace_name=str(row['workspace_name']),
            email=str(row['email']),
            role=normalize_workspace_role(str(row['role'])),
            invited_by_user_id=str(row['invited_by_user_id']),
            invited_by_name=str(row['invited_by_name']),
            provider=str(row['provider']),
            tenant_id=str(row['tenant_id']),
            status=str(row['status']),
            created_at=str(row['created_at']),
            expires_at=str(row['expires_at']),
        )

    def list_workspace_invitations(self, workspace_id: str, *, include_accepted: bool = False) -> list[WorkspaceInvitation]:
        query = '''
            SELECT invitation_id, workspace_id, workspace_name, email, role, invited_by_user_id, invited_by_name, provider, tenant_id, status, created_at, expires_at
            FROM workspace_invitations
            WHERE workspace_id = ?
        '''
        params: list[Any] = [workspace_id]
        if not include_accepted:
            query += " AND status = 'pending'"
        query += ' ORDER BY created_at DESC'
        with closing(self._connect()) as connection:
            rows = connection.execute(query, params).fetchall()
        return [
            WorkspaceInvitation(
                invitation_id=str(row['invitation_id']),
                workspace_id=str(row['workspace_id']),
                workspace_name=str(row['workspace_name']),
                email=str(row['email']),
                role=normalize_workspace_role(str(row['role'])),
                invited_by_user_id=str(row['invited_by_user_id']),
                invited_by_name=str(row['invited_by_name']),
                provider=str(row['provider']),
                tenant_id=str(row['tenant_id']),
                status=str(row['status']),
                created_at=str(row['created_at']),
                expires_at=str(row['expires_at']),
            )
            for row in rows
        ]

    def accept_workspace_invitation(self, invitation_id: str, *, user_id: str) -> WorkspaceInvitation | None:
        invitation = self.get_workspace_invitation(invitation_id)
        if invitation is None:
            return None
        if invitation.status != 'pending':
            return invitation
        if (_parse_iso8601(invitation.expires_at) or _utcnow_dt()) < _utcnow_dt():
            with closing(self._connect()) as connection:
                connection.execute(
                    '''
                    UPDATE workspace_invitations
                    SET status = 'expired'
                    WHERE invitation_id = ?
                    ''',
                    (invitation_id,),
                )
                connection.commit()
            return self.get_workspace_invitation(invitation_id)
        self.upsert_workspace_member(
            invitation.workspace_id,
            invitation.workspace_name,
            user_id,
            invitation.role,
            membership_source='invitation',
            provider=invitation.provider,
            tenant_id=invitation.tenant_id,
        )
        with closing(self._connect()) as connection:
            connection.execute(
                '''
                UPDATE workspace_invitations
                SET status = 'accepted'
                WHERE invitation_id = ?
                ''',
                (invitation_id,),
            )
            connection.commit()
        return self.get_workspace_invitation(invitation_id)


class AuthService:
    def __init__(
        self,
        repository: LocalAuthRepository | None,
        status: AuthStatus,
        *,
        hosted_config: HostedAuthConfig | None = None,
    ) -> None:
        self.repository = repository
        self.status = status
        self.hosted_config = hosted_config

    def _session_expiry(self) -> str:
        return (_utcnow_dt() + timedelta(minutes=_session_timeout_minutes())).isoformat()

    def _oidc_metadata(self) -> OIDCProviderMetadata | None:
        if self.hosted_config is None or not self.hosted_config.provider:
            return None
        provider = _normalize_provider_preset(self.hosted_config.provider)
        label = self.hosted_config.provider_label or provider.title()
        issuer = str(self.hosted_config.issuer or '').rstrip('/')
        if not issuer or not self.hosted_config.client_id or not self.hosted_config.redirect_uri:
            return None
        authorization_endpoint = f'{issuer}/oauth2/v1/authorize'
        token_endpoint = f'{issuer}/oauth2/v1/token'
        userinfo_endpoint = f'{issuer}/oauth2/v1/userinfo'
        jwks_uri = f'{issuer}/oauth2/v1/keys'
        if provider in {'azure-ad', 'azure'}:
            authorization_endpoint = f'{issuer}/oauth2/v2.0/authorize'
            token_endpoint = f'{issuer}/oauth2/v2.0/token'
            userinfo_endpoint = 'https://graph.microsoft.com/oidc/userinfo'
            jwks_uri = f'{issuer}/discovery/v2.0/keys'
        elif provider in {'google-workspace', 'google'}:
            authorization_endpoint = 'https://accounts.google.com/o/oauth2/v2/auth'
            token_endpoint = 'https://oauth2.googleapis.com/token'
            userinfo_endpoint = 'https://openidconnect.googleapis.com/v1/userinfo'
            jwks_uri = 'https://www.googleapis.com/oauth2/v3/certs'
        return OIDCProviderMetadata(
            provider=provider,
            provider_label=label,
            issuer=issuer,
            authorization_endpoint=authorization_endpoint,
            token_endpoint=token_endpoint,
            userinfo_endpoint=userinfo_endpoint,
            jwks_uri=jwks_uri,
            scopes=list(self.hosted_config.scopes),
            client_id=self.hosted_config.client_id,
            client_secret=self.hosted_config.client_secret,
            redirect_uri=self.hosted_config.redirect_uri,
        )

    def build_oidc_start(self) -> dict[str, str] | None:
        metadata = self._oidc_metadata()
        if metadata is None:
            return None
        state = secrets.token_urlsafe(18)
        nonce = secrets.token_urlsafe(18)
        return {
            'provider': metadata.provider_label,
            'authorization_url': build_oidc_authorization_url(metadata, state=state, nonce=nonce),
            'state': state,
            'nonce': nonce,
            'redirect_uri': metadata.redirect_uri,
        }

    def _create_session_record(self, auth_session: dict[str, Any], workspace_identity: dict[str, Any], *, mfa_required: bool, mfa_verified: bool) -> dict[str, Any]:
        if self.repository is None or not self.status.enabled:
            auth_session['session_id'] = f"session::{secrets.token_urlsafe(12)}"
            auth_session['session_expires_at'] = self._session_expiry()
            auth_session['mfa_required'] = mfa_required
            auth_session['mfa_verified'] = mfa_verified
            workspace_identity['session_id'] = auth_session['session_id']
            workspace_identity['session_expires_at'] = auth_session['session_expires_at']
            workspace_identity['mfa_required'] = mfa_required
            workspace_identity['mfa_verified'] = mfa_verified
            return auth_session
        record = self.repository.create_auth_session(
            user_id=str(auth_session.get('user_id', 'guest-user')),
            workspace_id=str(workspace_identity.get('workspace_id', 'guest-demo-workspace')),
            provider=str(auth_session.get('provider', 'guest')),
            auth_mode=str(auth_session.get('auth_mode', 'guest')),
            expires_at=self._session_expiry(),
            mfa_required=mfa_required,
            mfa_verified=mfa_verified,
        )
        auth_session['session_id'] = record.session_id
        auth_session['session_expires_at'] = record.expires_at
        auth_session['mfa_required'] = record.mfa_required
        auth_session['mfa_verified'] = record.mfa_verified
        workspace_identity['session_id'] = record.session_id
        workspace_identity['session_expires_at'] = record.expires_at
        workspace_identity['mfa_required'] = record.mfa_required
        workspace_identity['mfa_verified'] = record.mfa_verified
        return auth_session

    def sign_in_local(
        self,
        display_name: str | None,
        email: str | None,
        workspace_name: str | None,
        workspace_role: str | None = None,
        *,
        two_factor_code: str | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        clean_display = str(display_name or '').strip() or 'Local Demo User'
        clean_email = _normalize_email(email)
        if not clean_email:
            local_slug = _slug(clean_display)
            clean_email = f'{local_slug}@local-demo.invalid'
        normalized_role = normalize_workspace_role(workspace_role or 'owner')
        session = build_signed_in_auth_session(
            clean_display,
            clean_email,
            provider='local-demo',
            auth_mode='local',
            role=normalized_role,
        )
        workspace_identity = self.build_workspace_identity(session, workspace_name)
        if self.repository is not None and self.status.enabled:
            self.repository.upsert_user(session['user_id'], clean_email, clean_display, 'local-demo')
            self.repository.upsert_workspace_owner(
                str(workspace_identity['workspace_id']),
                str(workspace_identity['workspace_name']),
                str(workspace_identity['owner_user_id']),
                tenant_id='',
            )
            self.repository.upsert_workspace_member(
                str(workspace_identity['workspace_id']),
                str(workspace_identity['workspace_name']),
                str(session['user_id']),
                normalized_role,
                membership_source='local',
                provider='local-demo',
                tenant_id='',
            )
        mfa_required = bool(_totp_secret())
        mfa_verified = True if not mfa_required else verify_two_factor_code(_totp_secret(), two_factor_code)
        session['mfa_issuer'] = _totp_issuer() if mfa_required else ''
        self._create_session_record(session, workspace_identity, mfa_required=mfa_required, mfa_verified=mfa_verified)
        return session, workspace_identity

    def sign_in_hosted(
        self,
        *,
        display_name: str | None,
        email: str | None,
        provider_subject: str | None,
        workspace_name: str | None,
        workspace_role: str | None = None,
        provider: str | None = None,
        tenant_id: str | None = None,
        two_factor_code: str | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        effective_provider = _normalize_provider(provider or getattr(self.hosted_config, 'provider', '') or 'hosted')
        effective_tenant_id = str(tenant_id or getattr(self.hosted_config, 'tenant_id', '') or '').strip()
        session = build_hosted_auth_session(
            display_name=display_name,
            email=email,
            provider_subject=provider_subject,
            provider=effective_provider,
            tenant_id=effective_tenant_id,
            role=normalize_workspace_role(workspace_role or 'viewer'),
        )
        workspace_identity = self.build_workspace_identity(session, workspace_name)
        if self.repository is not None and self.status.enabled:
            self.repository.upsert_user(
                str(session['user_id']),
                str(session.get('email', '')),
                str(session.get('display_name', 'Hosted User')),
                effective_provider,
                external_subject=str(session.get('provider_subject', '')),
                tenant_id=effective_tenant_id,
            )
            self.repository.upsert_workspace_member(
                str(workspace_identity['workspace_id']),
                str(workspace_identity['workspace_name']),
                str(session['user_id']),
                str(session.get('role', 'viewer')),
                membership_source='hosted',
                provider=effective_provider,
                tenant_id=effective_tenant_id,
            )
        mfa_required = bool(_totp_secret())
        mfa_verified = True if not mfa_required else verify_two_factor_code(_totp_secret(), two_factor_code)
        session['mfa_issuer'] = _totp_issuer() if mfa_required else ''
        self._create_session_record(session, workspace_identity, mfa_required=mfa_required, mfa_verified=mfa_verified)
        return session, workspace_identity

    def sign_out(self) -> tuple[dict[str, Any], dict[str, Any]]:
        guest = build_guest_auth_session()
        return guest, self.build_workspace_identity(guest, None)

    def list_active_sessions(self, workspace_identity: dict[str, Any]) -> list[dict[str, Any]]:
        if self.repository is None or not self.status.enabled:
            return []
        self.repository.prune_expired_sessions()
        sessions = self.repository.list_auth_sessions(
            str(workspace_identity.get('user_id', '')),
            str(workspace_identity.get('workspace_id', '')),
        )
        return [
            {
                'session_id': item.session_id,
                'provider': item.provider,
                'auth_mode': item.auth_mode,
                'created_at': item.created_at,
                'expires_at': item.expires_at,
                'mfa_required': item.mfa_required,
                'mfa_verified': item.mfa_verified,
            }
            for item in sessions
        ]

    def revoke_session(self, session_id: str) -> None:
        if self.repository is None or not self.status.enabled or not session_id:
            return
        self.repository.revoke_auth_session(session_id)

    def create_workspace_invitation(
        self,
        workspace_identity: dict[str, Any],
        *,
        email: str,
        role: str,
    ) -> dict[str, Any] | None:
        if self.repository is None or not self.status.enabled:
            return None
        invitation = self.repository.create_workspace_invitation(
            workspace_id=str(workspace_identity.get('workspace_id', 'guest-demo-workspace')),
            workspace_name=str(workspace_identity.get('workspace_name', 'Guest Demo Workspace')),
            email=email,
            role=role,
            invited_by_user_id=str(workspace_identity.get('user_id', 'guest-user')),
            invited_by_name=str(workspace_identity.get('display_name', 'Guest User')),
            provider=str(workspace_identity.get('provider', 'local-demo')),
            tenant_id=str(workspace_identity.get('tenant_id', '')),
            expires_at=(_utcnow_dt() + timedelta(days=7)).isoformat(),
        )
        return {
            'invitation_id': invitation.invitation_id,
            'email': invitation.email,
            'role': format_workspace_role(invitation.role),
            'status': invitation.status.title(),
            'expires_at': invitation.expires_at,
        }

    def list_workspace_invitations(self, workspace_identity: dict[str, Any]) -> list[dict[str, Any]]:
        if self.repository is None or not self.status.enabled:
            return []
        invitations = self.repository.list_workspace_invitations(str(workspace_identity.get('workspace_id', 'guest-demo-workspace')), include_accepted=True)
        return [
            {
                'invitation_id': item.invitation_id,
                'email': item.email,
                'role': format_workspace_role(item.role),
                'status': item.status.title(),
                'invited_by': item.invited_by_name,
                'provider': item.provider,
                'expires_at': item.expires_at,
            }
            for item in invitations
        ]

    def accept_workspace_invitation(self, auth_session: dict[str, Any], invitation_id: str) -> dict[str, Any] | None:
        if self.repository is None or not self.status.enabled:
            return None
        invitation = self.repository.accept_workspace_invitation(invitation_id, user_id=str(auth_session.get('user_id', '')))
        if invitation is None:
            return None
        auth_session['active_workspace_id'] = invitation.workspace_id
        return {
            'invitation_id': invitation.invitation_id,
            'workspace_id': invitation.workspace_id,
            'workspace_name': invitation.workspace_name,
            'status': invitation.status.title(),
            'role': format_workspace_role(invitation.role),
        }

    def build_workspace_identity(self, auth_session: dict[str, Any], workspace_name: str | None) -> dict[str, Any]:
        signed_in = bool(auth_session.get('signed_in'))
        normalized_role = normalize_workspace_role(auth_session.get('role', 'viewer'))
        auth_mode = str(auth_session.get('auth_mode') or 'guest')
        explicit_workspace_id = str(auth_session.get('active_workspace_id') or '').strip() or None
        identity = build_workspace_identity(
            str(auth_session.get('display_name') or ''),
            workspace_name,
            signed_in,
            user_id=str(auth_session.get('user_id') or '') or None,
            owner_user_id=str(auth_session.get('user_id') or '') or None,
            workspace_id=explicit_workspace_id,
            email=str(auth_session.get('email') or ''),
            auth_mode=auth_mode,
            role=normalized_role,
        )
        membership_validated: bool | None = None
        ownership_validated: bool | None = None
        membership_source = 'guest'
        tenant_id = str(auth_session.get('tenant_id') or '').strip()
        provider_subject = str(auth_session.get('provider_subject') or '').strip()
        if self.repository is not None and self.status.enabled and signed_in:
            workspace_id = str(identity['workspace_id'])
            user_id = str(identity['user_id'])
            membership = self.repository.load_workspace_membership(workspace_id, user_id)
            membership_validated = membership is not None
            if membership is not None:
                identity['workspace_id'] = membership.workspace_id
                identity['workspace_name'] = membership.workspace_name
                normalized_role = membership.role
                membership_source = membership.membership_source
                ownership_validated = membership.ownership_validated if normalized_role == 'owner' else False
                if membership.tenant_id:
                    tenant_id = membership.tenant_id
            elif normalized_role == 'owner':
                ownership_validated = self.repository.is_workspace_owner(workspace_id, user_id)
        elif signed_in:
            membership_source = 'session'
        identity['owner_label'] = (
            str(auth_session.get('display_name', 'Guest User'))
            if signed_in
            else 'Guest session'
        )
        identity['provider'] = str(auth_session.get('provider', 'guest'))
        identity['provider_subject'] = provider_subject
        identity['role'] = normalized_role
        identity['role_label'] = format_workspace_role(normalized_role)
        identity['permissions'] = sorted(get_workspace_role_permissions(normalized_role))
        identity['is_workspace_owner'] = bool(normalized_role == 'owner')
        identity['membership_validated'] = membership_validated
        identity['ownership_validated'] = ownership_validated
        identity['membership_source'] = membership_source
        identity['tenant_id'] = tenant_id
        identity['tenant_scope'] = str(identity.get('workspace_id', 'guest-demo-workspace'))
        identity['session_id'] = str(auth_session.get('session_id', ''))
        identity['session_expires_at'] = str(auth_session.get('session_expires_at', ''))
        identity['mfa_required'] = bool(auth_session.get('mfa_required', False))
        identity['mfa_verified'] = bool(auth_session.get('mfa_verified', False))
        identity['workspace_access_summary'] = (
            'Workspace owner with admin, write, and read access.'
            if normalized_role == 'owner'
            else 'Workspace admin with write and read access.'
            if normalized_role == 'admin'
            else 'Workspace analyst with write and read access.'
            if normalized_role == 'analyst'
            else 'Workspace viewer with read-only access.'
        )
        return identity


def build_auth_service(
    *,
    database_url: str | None = None,
    sqlite_path: str | os.PathLike[str] | None = None,
) -> AuthService:
    configured_provider = _normalize_provider(os.getenv(AUTH_PROVIDER_ENV, ''))
    configured_issuer = str(os.getenv(AUTH_ISSUER_ENV, '')).strip()
    configured_audience = str(os.getenv(AUTH_AUDIENCE_ENV, '')).strip()
    configured_client_id = str(os.getenv(AUTH_CLIENT_ID_ENV, '')).strip()
    configured_client_secret = str(os.getenv(AUTH_CLIENT_SECRET_ENV, '')).strip()
    configured_tenant_id = str(os.getenv(AUTH_TENANT_ID_ENV, '')).strip()
    configured_redirect_uri = str(os.getenv(AUTH_REDIRECT_URI_ENV, '')).strip()
    configured_scopes = [
        scope.strip()
        for scope in str(os.getenv(AUTH_SCOPES_ENV, '')).split()
        if scope.strip()
    ]
    configured_database_url = (database_url or os.getenv(DATABASE_URL_ENV, '')).strip()
    configured_sqlite_path = str(sqlite_path or os.getenv(SQLITE_PATH_ENV, '')).strip()
    notes = [
        'Guest mode stays available even when no database or external identity provider is configured.',
        'This foundation supports guest and local demo sign-in today and now exposes a hosted-auth preparation seam for future external identity integration.',
    ]
    hosted_notes: list[str] = []
    normalized_provider = _normalize_provider_preset(configured_provider)
    provider_preset = OIDC_PROVIDER_PRESETS.get(normalized_provider, {})
    hosted_enabled = normalized_provider not in {'', 'guest', 'local', 'local-demo'}
    hosted_config = HostedAuthConfig(
        enabled=hosted_enabled and bool(configured_issuer and configured_client_id and configured_redirect_uri),
        provider=normalized_provider if hosted_enabled else 'guest',
        provider_label=str(provider_preset.get('label', normalized_provider.title() if normalized_provider else 'Guest')),
        issuer=configured_issuer,
        audience=configured_audience,
        client_id=configured_client_id,
        client_secret=configured_client_secret,
        tenant_id=configured_tenant_id,
        redirect_uri=configured_redirect_uri,
        scopes=configured_scopes or list(provider_preset.get('scopes', ['openid', 'profile', 'email'])),
        notes=hosted_notes,
    )
    if hosted_enabled:
        if hosted_config.enabled:
            hosted_notes.append(
                f"Hosted auth preparation is configured for provider '{hosted_config.provider_label}' and can be wired to a future external sign-in callback without changing workspace flows."
            )
        else:
            hosted_notes.append(
                f"Hosted auth provider '{hosted_config.provider_label}' is selected, but issuer, client, or redirect settings are incomplete. The platform will stay on guest/local auth until that integration is finished."
            )

    if configured_database_url:
        if configured_database_url.startswith('sqlite:///'):
            configured_sqlite_path = configured_database_url.replace('sqlite:///', '', 1)
        else:
            return AuthService(
                repository=None,
                status=AuthStatus(
                    enabled=False,
                    mode='guest',
                    storage_target='session-only',
                    notes=notes + hosted_notes + ['The configured database URL is not supported by the local auth foundation, so auth stays session-local.'],
                    provider_mode='hosted-prepared' if hosted_enabled else 'guest',
                    provider_name=hosted_config.provider if hosted_enabled else 'guest',
                    hosted_configured=hosted_enabled,
                    tenant_id=configured_tenant_id,
                ),
                hosted_config=hosted_config if hosted_enabled else None,
            )

    if not configured_sqlite_path:
        return AuthService(
            repository=None,
            status=AuthStatus(
                enabled=False,
                mode='hosted-prepared' if hosted_enabled else 'guest',
                storage_target='session-only',
                notes=notes + hosted_notes + ['No auth database is configured, so the platform is using guest and session-local sign-in mode.'],
                provider_mode='hosted-prepared' if hosted_enabled else 'guest',
                provider_name=hosted_config.provider if hosted_enabled else 'guest',
                hosted_configured=hosted_enabled,
                tenant_id=configured_tenant_id,
            ),
            hosted_config=hosted_config if hosted_enabled else None,
        )

    sqlite_target = Path(configured_sqlite_path).expanduser()
    repository = LocalAuthRepository(sqlite_target)
    return AuthService(
        repository=repository,
        status=AuthStatus(
            enabled=True,
            mode='hybrid' if hosted_enabled else 'local',
            storage_target=str(sqlite_target),
            notes=notes + hosted_notes + [f'Local auth identities and workspace ownership are stored in SQLite at {sqlite_target}.'],
            provider_mode='hosted-prepared' if hosted_enabled else 'local',
            provider_name=hosted_config.provider if hosted_enabled else 'local-demo',
            hosted_configured=hosted_enabled,
            tenant_id=configured_tenant_id,
        ),
        hosted_config=hosted_config if hosted_enabled else None,
    )
