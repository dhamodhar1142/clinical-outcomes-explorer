from __future__ import annotations

from datetime import datetime


def log_audit_event(events: list[dict[str, object]], event_type: str, details: str, user_interaction: str = 'User action', analysis_step: str = 'Session activity') -> list[dict[str, object]]:
    updated = list(events)
    entry = {
        'event_type': event_type,
        'details': details,
        'user_interaction': user_interaction,
        'analysis_step': analysis_step,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'sequence': len(updated) + 1,
    }
    if not updated or updated[-1].get('event_type') != event_type or updated[-1].get('details') != details:
        updated.append(entry)
    return updated[-100:]
