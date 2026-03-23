from __future__ import annotations

from collections.abc import MutableMapping
import re
from typing import Any

import src.logger as logger_module


def _slug(value: str) -> str:
    slug = re.sub(r'[^a-z0-9]+', '-', value.strip().lower()).strip('-')
    return slug or 'demo'


def build_workspace_identity(
    display_name: str | None,
    workspace_name: str | None,
    signed_in: bool,
    user_id: str | None = None,
    owner_user_id: str | None = None,
    workspace_id: str | None = None,
    email: str | None = None,
    auth_mode: str = 'guest',
    role: str | None = None,
) -> dict[str, str | bool]:
    clean_display = (display_name or '').strip() or 'Local Demo User'
    clean_workspace = (workspace_name or '').strip() or 'Personal Demo Workspace'
    identity_slug = _slug(user_id or clean_display)
    if signed_in:
        resolved_workspace_id = str(workspace_id or '').strip() or f"{_slug(clean_workspace)}::{identity_slug}"
        status_label = 'Signed in'
    else:
        clean_display = 'Guest User'
        clean_workspace = 'Guest Demo Workspace'
        resolved_workspace_id = 'guest-demo-workspace'
        status_label = 'Guest session'
    return {
        'signed_in': signed_in,
        'display_name': clean_display,
        'workspace_name': clean_workspace,
        'workspace_id': resolved_workspace_id,
        'status_label': status_label,
        'user_id': str(user_id or ('guest-user' if not signed_in else identity_slug)),
        'owner_user_id': str(owner_user_id or (user_id or ('guest-user' if not signed_in else identity_slug))),
        'owner_label': clean_display if signed_in else 'Guest session',
        'email': str(email or ''),
        'auth_mode': str(auth_mode or ('local' if signed_in else 'guest')),
        'role': str(role or ('owner' if signed_in else 'viewer')),
        'role_label': str(role or ('owner' if signed_in else 'viewer')).title(),
        'is_workspace_owner': bool((role or ('owner' if signed_in else 'viewer')) == 'owner'),
    }


def ensure_workspace_scope(
    session_state: MutableMapping[str, Any],
    store_key: str,
    identity: dict[str, Any],
) -> dict[str, Any]:
    scoped_store = session_state.setdefault(store_key, {})
    workspace_id = str(identity.get('workspace_id', 'guest-demo-workspace'))
    if workspace_id not in scoped_store:
        scoped_store[workspace_id] = {}
    return scoped_store[workspace_id]


def _ensure_workspace_list_scope(
    session_state: MutableMapping[str, Any],
    store_key: str,
    identity: dict[str, Any],
) -> list[Any]:
    scoped_store = session_state.setdefault(store_key, {})
    workspace_id = str(identity.get('workspace_id', 'guest-demo-workspace'))
    if workspace_id not in scoped_store or not isinstance(scoped_store.get(workspace_id), list):
        scoped_store[workspace_id] = []
    return scoped_store[workspace_id]


def _get_persistence_service(session_state: MutableMapping[str, Any]) -> Any | None:
    service = session_state.get('persistence_service')
    return service if service is not None else None


def persist_active_workspace_state(session_state: MutableMapping[str, Any]) -> None:
    identity = session_state.get('_active_workspace_identity') or session_state.get('workspace_identity') or build_workspace_identity(None, None, False)
    workspace_id = str(identity.get('workspace_id', 'guest-demo-workspace'))

    session_state.setdefault('workspace_saved_snapshots', {})[workspace_id] = dict(session_state.get('saved_snapshots', {}))
    session_state.setdefault('workspace_workflow_packs', {})[workspace_id] = dict(session_state.get('workflow_packs', {}))
    session_state.setdefault('workspace_collaboration_notes', {})[workspace_id] = list(session_state.get('collaboration_notes', []))
    session_state.setdefault('workspace_beta_interest_submissions', {})[workspace_id] = list(session_state.get('beta_interest_submissions', []))
    session_state.setdefault('workspace_analysis_logs', {})[workspace_id] = list(session_state.get('analysis_log', []))
    session_state.setdefault('workspace_run_history', {})[workspace_id] = list(session_state.get('run_history', []))

    service = _get_persistence_service(session_state)
    if service is not None and bool(getattr(service, 'enabled', False)):
        logger_module.log_platform_event(
            'workspace_state_persisted',
            logger_name='workspace',
            workspace_id=workspace_id,
            signed_in=bool(identity.get('signed_in', False)),
            snapshot_count=len(session_state['workspace_saved_snapshots'][workspace_id]),
            workflow_pack_count=len(session_state['workspace_workflow_packs'][workspace_id]),
            note_count=len(session_state['workspace_collaboration_notes'][workspace_id]),
        )
        service.save_workspace_state(
            identity,
            {
                'saved_snapshots': session_state['workspace_saved_snapshots'][workspace_id],
                'workflow_packs': session_state['workspace_workflow_packs'][workspace_id],
                'collaboration_notes': session_state['workspace_collaboration_notes'][workspace_id],
                'beta_interest_submissions': session_state['workspace_beta_interest_submissions'][workspace_id],
                'analysis_log': session_state['workspace_analysis_logs'][workspace_id],
                'run_history': session_state['workspace_run_history'][workspace_id],
            },
        )


def sync_workspace_views(session_state: MutableMapping[str, Any]) -> None:
    identity = session_state.get('workspace_identity') or build_workspace_identity(None, None, False)
    workspace_id = str(identity.get('workspace_id', 'guest-demo-workspace'))
    previous_workspace_id = str(session_state.get('_active_workspace_id', '')) or None
    if previous_workspace_id and previous_workspace_id != workspace_id:
        persist_active_workspace_state(session_state)

    service = _get_persistence_service(session_state)
    if service is not None and bool(getattr(service, 'enabled', False)):
        service.ensure_workspace(identity)
        loaded_workspace_ids = session_state.setdefault('_loaded_persisted_workspace_ids', [])
        if workspace_id not in loaded_workspace_ids:
            persisted_state = service.load_workspace_state(identity)
            session_state.setdefault('workspace_saved_snapshots', {})[workspace_id] = dict(persisted_state.get('saved_snapshots', {}))
            session_state.setdefault('workspace_workflow_packs', {})[workspace_id] = dict(persisted_state.get('workflow_packs', {}))
            session_state.setdefault('workspace_collaboration_notes', {})[workspace_id] = list(persisted_state.get('collaboration_notes', []))
            session_state.setdefault('workspace_beta_interest_submissions', {})[workspace_id] = list(persisted_state.get('beta_interest_submissions', []))
            session_state.setdefault('workspace_analysis_logs', {})[workspace_id] = list(persisted_state.get('analysis_log', []))
            session_state.setdefault('workspace_run_history', {})[workspace_id] = list(persisted_state.get('run_history', []))
            loaded_workspace_ids.append(workspace_id)
            logger_module.log_platform_event(
                'workspace_state_loaded',
                logger_name='workspace',
                workspace_id=workspace_id,
                signed_in=bool(identity.get('signed_in', False)),
                snapshot_count=len(session_state['workspace_saved_snapshots'][workspace_id]),
                workflow_pack_count=len(session_state['workspace_workflow_packs'][workspace_id]),
            )

    snapshots = ensure_workspace_scope(session_state, 'workspace_saved_snapshots', identity)
    packs = ensure_workspace_scope(session_state, 'workspace_workflow_packs', identity)
    notes = _ensure_workspace_list_scope(session_state, 'workspace_collaboration_notes', identity)
    beta_interest = _ensure_workspace_list_scope(session_state, 'workspace_beta_interest_submissions', identity)
    analysis_log = _ensure_workspace_list_scope(session_state, 'workspace_analysis_logs', identity)
    run_history = _ensure_workspace_list_scope(session_state, 'workspace_run_history', identity)
    session_state['saved_snapshots'] = snapshots
    session_state['workflow_packs'] = packs
    session_state['collaboration_notes'] = notes
    session_state['beta_interest_submissions'] = beta_interest
    session_state['analysis_log'] = analysis_log
    session_state['run_history'] = run_history
    session_state['_active_workspace_id'] = workspace_id
    session_state['_active_workspace_identity'] = dict(identity)
