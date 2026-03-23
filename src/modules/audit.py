from __future__ import annotations

from datetime import datetime


def log_audit_event(
    events: list[dict[str, object]],
    event_type: str,
    details: str,
    user_interaction: str = 'User action',
    analysis_step: str = 'Session activity',
    *,
    actor_context: dict[str, object] | None = None,
    resource_context: dict[str, object] | None = None,
) -> list[dict[str, object]]:
    updated = list(events)
    actor_context = actor_context or {}
    resource_context = resource_context or {}
    entry = {
        'event_type': event_type,
        'details': details,
        'user_interaction': user_interaction,
        'analysis_step': analysis_step,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'sequence': len(updated) + 1,
        'workspace_id': str(actor_context.get('workspace_id', 'guest-demo-workspace')),
        'workspace_name': str(actor_context.get('workspace_name', 'Guest Demo Workspace')),
        'user_id': str(actor_context.get('user_id', 'guest-user')),
        'workspace_role': str(actor_context.get('role_label', actor_context.get('role', 'Viewer'))),
        'owner_label': str(actor_context.get('owner_label', 'Guest session')),
        'session_id': str(actor_context.get('session_id', '')),
        'request_id': str(actor_context.get('request_id', '')),
        'resource_type': str(resource_context.get('resource_type', '')),
        'resource_name': str(resource_context.get('resource_name', '')),
        'action_outcome': str(resource_context.get('action_outcome', 'success')),
    }
    if not updated or updated[-1].get('event_type') != event_type or updated[-1].get('details') != details:
        updated.append(entry)
    return updated[-100:]
