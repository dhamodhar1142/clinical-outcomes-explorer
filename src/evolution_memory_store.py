from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.runtime_paths import data_path


EVOLUTION_MEMORY_PATH = data_path('runtime', 'evolution_memory.json')


def load_evolution_memory(
    *,
    storage_service: Any | None = None,
    workspace_identity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if storage_service is not None and bool(getattr(storage_service, 'enabled', False)):
        try:
            payload = storage_service.load_runtime_state(
                workspace_identity,
                state_name='evolution_memory',
            )
            loaded = json.loads(payload.decode('utf-8'))
            if isinstance(loaded, dict):
                return loaded
        except (AttributeError, FileNotFoundError, OSError, UnicodeDecodeError, json.JSONDecodeError):
            pass
    path = Path(EVOLUTION_MEMORY_PATH)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def save_evolution_memory(
    memory: dict[str, Any] | None,
    *,
    storage_service: Any | None = None,
    workspace_identity: dict[str, Any] | None = None,
) -> bool:
    normalized = memory if isinstance(memory, dict) else {}
    serialized = json.dumps(normalized, indent=2)
    if storage_service is not None and bool(getattr(storage_service, 'enabled', False)):
        try:
            storage_service.save_runtime_state(
                workspace_identity,
                state_name='evolution_memory',
                payload=serialized,
            )
        except AttributeError:
            pass
    path = Path(EVOLUTION_MEMORY_PATH)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(serialized, encoding='utf-8')
        return True
    except OSError:
        return False


__all__ = ['EVOLUTION_MEMORY_PATH', 'load_evolution_memory', 'save_evolution_memory']
