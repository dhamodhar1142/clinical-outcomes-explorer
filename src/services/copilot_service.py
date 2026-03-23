from __future__ import annotations

from typing import Any


def _load_ai_copilot():
    from src import ai_copilot

    return ai_copilot


def execute_copilot_prompt(
    prompt: str,
    *,
    data,
    schema_context: dict[str, Any],
    persist_messages: bool = True,
) -> dict[str, Any]:
    ai_copilot = _load_ai_copilot()

    if persist_messages:
        ai_copilot.append_copilot_message('user', prompt)
    response = ai_copilot.run_copilot_question(prompt, data, schema_context)
    if persist_messages:
        ai_copilot.append_copilot_message('assistant', response.get('answer', ''), response)
    response['messages'] = ai_copilot.initialize_copilot_memory()
    return response


def plan_copilot_workflow(
    workflow_prompt: str,
    *,
    data,
    canonical_map: dict[str, Any],
    readiness: dict[str, Any],
    healthcare: dict[str, Any],
    remediation,
) -> dict[str, Any]:
    ai_copilot = _load_ai_copilot()
    return ai_copilot.plan_workflow_action(
        workflow_prompt,
        data,
        canonical_map,
        readiness,
        healthcare,
        remediation,
    )


__all__ = ['execute_copilot_prompt', 'plan_copilot_workflow']
