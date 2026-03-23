from __future__ import annotations

import importlib
import sys
from typing import Any

import src.auth as auth_module
from src.collaboration_notes import append_collaboration_note


def _get_auth_module():
    global auth_module
    required_attrs = (
        'enforce_workspace_boundary',
        'enforce_workspace_permission',
        'enforce_workspace_minimum_role',
    )
    if all(hasattr(auth_module, name) for name in required_attrs):
        return auth_module
    sys.modules.pop('src.auth', None)
    auth_module = importlib.import_module('src.auth')
    return auth_module


def _ensure_workspace_asset_access(asset: dict[str, Any], workspace_identity: dict[str, Any], *, resource_label: str) -> None:
    active_auth = _get_auth_module()
    asset_workspace_id = str(asset.get('workspace_id', ''))
    active_auth.enforce_workspace_boundary(
        workspace_identity,
        workspace_id=asset_workspace_id or str(workspace_identity.get('workspace_id', '')),
        resource_label=resource_label,
    )
    if workspace_identity:
        active_auth.enforce_workspace_permission(
            workspace_identity,
            'workspace_read',
            resource_label=resource_label,
        )
    current_workspace_id = str(workspace_identity.get('workspace_id', ''))
    if asset_workspace_id and current_workspace_id and asset_workspace_id != current_workspace_id:
        raise PermissionError(
            f"{resource_label.title()} belongs to a different workspace and cannot be loaded into '{workspace_identity.get('workspace_name', 'Current workspace')}'."
        )


def save_snapshot_to_workspace(
    session_state: dict[str, Any],
    *,
    snapshot_name: str,
    dataset_name: str,
    controls: dict[str, Any],
    workspace_identity: dict[str, Any],
) -> dict[str, Any]:
    active_auth = _get_auth_module()
    active_auth.enforce_workspace_boundary(
        workspace_identity,
        workspace_id=str(workspace_identity.get('workspace_id', '')),
        resource_label='snapshot saves',
    )
    active_auth.enforce_workspace_permission(
        workspace_identity,
        'workspace_write',
        resource_label='snapshot saves',
    )
    active_auth.enforce_workspace_minimum_role(
        workspace_identity,
        'analyst',
        resource_label='snapshot saves',
    )
    snapshot_payload = {
        'dataset_name': dataset_name,
        'controls': controls,
        'workspace_id': str(workspace_identity.get('workspace_id', 'guest-demo-workspace')),
        'created_by_user_id': str(workspace_identity.get('user_id', 'guest-user')),
        'created_by_name': str(workspace_identity.get('display_name', 'Guest User')),
        'workspace_role': str(workspace_identity.get('role', 'viewer')),
    }
    session_state.setdefault('saved_snapshots', {})[snapshot_name] = snapshot_payload
    return snapshot_payload


def load_snapshot_into_session(session_state: dict[str, Any], snapshot_name: str, workspace_identity: dict[str, Any] | None = None) -> dict[str, Any]:
    snapshot = dict(session_state.get('saved_snapshots', {}).get(snapshot_name, {}))
    _ensure_workspace_asset_access(snapshot, workspace_identity or {}, resource_label='snapshot')
    for key, value in snapshot.get('controls', {}).items():
        session_state[key] = value
    return snapshot


def save_workflow_pack_to_workspace(
    session_state: dict[str, Any],
    *,
    workflow_name: str,
    details: dict[str, Any],
    summary: dict[str, Any],
    controls: dict[str, Any],
    workspace_identity: dict[str, Any],
) -> dict[str, Any]:
    active_auth = _get_auth_module()
    active_auth.enforce_workspace_boundary(
        workspace_identity,
        workspace_id=str(workspace_identity.get('workspace_id', '')),
        resource_label='workflow pack saves',
    )
    active_auth.enforce_workspace_permission(
        workspace_identity,
        'workspace_write',
        resource_label='workflow pack saves',
    )
    active_auth.enforce_workspace_minimum_role(
        workspace_identity,
        'analyst',
        resource_label='workflow pack saves',
    )
    workflow_payload = {
        'summary': summary,
        'details': details,
        'controls': controls,
        'workspace_id': str(workspace_identity.get('workspace_id', 'guest-demo-workspace')),
        'created_by_user_id': str(workspace_identity.get('user_id', 'guest-user')),
        'created_by_name': str(workspace_identity.get('display_name', 'Guest User')),
        'workspace_role': str(workspace_identity.get('role', 'viewer')),
    }
    session_state.setdefault('workflow_packs', {})[workflow_name] = workflow_payload
    return workflow_payload


def load_workflow_pack_into_session(session_state: dict[str, Any], workflow_name: str, workspace_identity: dict[str, Any] | None = None) -> dict[str, Any]:
    workflow_pack = dict(session_state.get('workflow_packs', {}).get(workflow_name, {}))
    _ensure_workspace_asset_access(workflow_pack, workspace_identity or {}, resource_label='workflow pack')
    for key, value in workflow_pack.get('controls', {}).items():
        session_state[key] = value
    return workflow_pack


def add_collaboration_note_to_workspace(
    session_state: dict[str, Any],
    *,
    target_type: str,
    target_name: str,
    note_text: str,
    workspace_identity: dict[str, Any],
    section_name: str = '',
) -> dict[str, Any]:
    active_auth = _get_auth_module()
    active_auth.enforce_workspace_boundary(
        workspace_identity,
        workspace_id=str(workspace_identity.get('workspace_id', '')),
        resource_label='collaboration notes',
    )
    active_auth.enforce_workspace_permission(
        workspace_identity,
        'workspace_write',
        resource_label='collaboration notes',
    )
    active_auth.enforce_workspace_minimum_role(
        workspace_identity,
        'analyst',
        resource_label='collaboration notes',
    )
    append_collaboration_note(
        session_state.setdefault('collaboration_notes', []),
        target_type,
        target_name,
        note_text,
        str(workspace_identity.get('display_name', 'Guest User')),
        str(workspace_identity.get('workspace_name', 'Guest Demo Workspace')),
        section_name=section_name or target_name,
        author_user_id=str(workspace_identity.get('user_id', 'guest-user')),
        workspace_id=str(workspace_identity.get('workspace_id', 'guest-demo-workspace')),
        workspace_role=str(workspace_identity.get('role', 'viewer')),
    )
    return dict(session_state.get('collaboration_notes', [])[-1])


__all__ = [
    'add_collaboration_note_to_workspace',
    'load_snapshot_into_session',
    'load_workflow_pack_into_session',
    'save_snapshot_to_workspace',
    'save_workflow_pack_to_workspace',
]
