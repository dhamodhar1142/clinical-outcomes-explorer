from __future__ import annotations

from collections.abc import Callable
from typing import Any

from src.jobs import cancel_job, force_cancel_job, get_job_result, get_job_status, submit_job


def submit_background_task(
    session_state: dict[str, Any],
    *,
    job_runtime: dict[str, Any],
    job_id: str | None = None,
    task_key: str,
    task_label: str,
    runner: Callable[[], Any] | None = None,
    detail: str,
    stage_messages: list[str] | None = None,
    progress_callback: Callable[[float, str], None] | None = None,
    task_name: str | None = None,
    task_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return submit_job(
        session_state,
        job_runtime,
        job_id=job_id,
        task_key=task_key,
        task_label=task_label,
        runner=runner,
        detail=detail,
        stage_messages=stage_messages,
        progress_callback=progress_callback,
        task_name=task_name,
        task_kwargs=task_kwargs,
    )


def read_background_task(session_state: dict[str, Any], job_id: str) -> dict[str, Any]:
    return {
        'status': get_job_status(session_state, job_id),
        'result': get_job_result(session_state, job_id),
    }


def cancel_background_task(session_state: dict[str, Any], job_id: str) -> dict[str, Any]:
    return cancel_job(session_state, job_id)


def force_cancel_background_task(session_state: dict[str, Any], job_id: str, *, reason: str) -> dict[str, Any]:
    return force_cancel_job(session_state, job_id, reason=reason)


__all__ = ['cancel_background_task', 'force_cancel_background_task', 'read_background_task', 'submit_background_task']
