from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class User:
    user_id: str
    display_name: str
    email: str = ''
    auth_mode: str = 'guest'
    signed_in: bool = False
    role: str = 'viewer'
    updated_at: str = ''


@dataclass(frozen=True)
class Workspace:
    workspace_id: str
    workspace_name: str
    display_name: str
    owner_user_id: str
    signed_in: bool = False
    plan_name: str = 'Pro'
    updated_at: str = ''


@dataclass(frozen=True)
class Dataset:
    workspace_id: str
    dataset_name: str
    source_mode: str
    row_count: int = 0
    column_count: int = 0
    file_size_mb: float = 0.0
    description: str = ''
    best_for: str = ''
    updated_at: str = ''


@dataclass(frozen=True)
class Snapshot:
    workspace_id: str
    snapshot_name: str
    dataset_name: str = ''
    controls: dict[str, Any] = field(default_factory=dict)
    created_by: str = ''
    updated_at: str = ''


@dataclass(frozen=True)
class WorkflowPack:
    workspace_id: str
    workflow_name: str
    summary: str = ''
    controls: dict[str, Any] = field(default_factory=dict)
    created_by: str = ''
    updated_at: str = ''


@dataclass(frozen=True)
class Note:
    workspace_id: str
    target_type: str
    target_name: str
    note_text: str
    author_name: str = ''
    created_at: str = ''


@dataclass(frozen=True)
class UsageEvent:
    workspace_id: str
    user_id: str
    event_type: str
    details: str = ''
    user_interaction: str = ''
    analysis_step: str = ''
    created_at: str = ''


@dataclass(frozen=True)
class Report:
    workspace_id: str
    dataset_name: str
    report_type: str
    file_name: str
    status: str = 'generated'
    generated_at: str = ''


@dataclass(frozen=True)
class UserSettings:
    user_id: str
    workspace_id: str
    settings_json: dict[str, Any] = field(default_factory=dict)
    updated_at: str = ''


@dataclass(frozen=True)
class DatasetVersion:
    workspace_id: str
    dataset_name: str
    version_id: str
    version_hash: str
    version_label: str = ''
    source_mode: str = ''
    row_count: int = 0
    column_count: int = 0
    file_size_mb: float = 0.0
    metadata_json: dict[str, Any] = field(default_factory=dict)
    is_active: bool = True
    created_at: str = ''


@dataclass(frozen=True)
class PersistentWorkspaceSnapshot:
    workspace_id: str
    snapshot_id: str
    snapshot_name: str
    dataset_name: str = ''
    dataset_version_id: str = ''
    snapshot_payload: dict[str, Any] = field(default_factory=dict)
    created_by_user_id: str = ''
    created_at: str = ''


@dataclass(frozen=True)
class CollaborationSession:
    workspace_id: str
    user_id: str
    display_name: str
    session_id: str
    active_section: str = ''
    presence_state: str = 'active'
    updated_at: str = ''
