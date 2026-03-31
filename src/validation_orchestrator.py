from __future__ import annotations

import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

VALIDATION_NAME_TO_TASK = {
    'Quick validation': 'quick',
    'Full validation': 'full',
    'Release validation': 'release',
    'Accessibility / UI audit': 'accessibility',
    'Cross-dataset cache validation': 'cross-dataset-cache',
}


def _utcnow() -> str:
    return datetime.now(UTC).isoformat()


def _is_cloud_runtime() -> bool:
    root_hint = ROOT.as_posix().lower()
    return bool(
        os.getenv('STREAMLIT_SHARING_MODE')
        or os.getenv('STREAMLIT_CLOUD')
        or '/mount/src/' in root_hint
    )


def recommended_validation_task(validation_name: str) -> str:
    return VALIDATION_NAME_TO_TASK.get(str(validation_name).strip(), 'quick')


def build_validation_runtime_profile() -> dict[str, Any]:
    cloud_runtime = _is_cloud_runtime()
    return {
        'cloud_runtime': cloud_runtime,
        'runtime_label': 'Streamlit Cloud' if cloud_runtime else 'Local / workstation',
        'supports_heavy_validation': not cloud_runtime,
        'detail': (
            'This runtime can execute the full validation matrix directly.'
            if not cloud_runtime
            else 'This runtime is better suited to lighter checks. Heavy validation is recommended from a local or staging environment.'
        ),
    }


def build_validation_execution_plan(validation_recommendations: Any) -> list[dict[str, Any]]:
    profile = build_validation_runtime_profile()
    rows: list[dict[str, Any]] = []
    for row in getattr(validation_recommendations, 'to_dict', lambda **_: [])(orient='records'):
        validation_name = str(row.get('validation', '')).strip()
        task = recommended_validation_task(validation_name)
        heavy_task = task in {'full', 'release', 'cross-dataset-cache'}
        allowed = bool(profile.get('supports_heavy_validation', False) or not heavy_task)
        rows.append(
            {
                'validation': validation_name,
                'task': task,
                'priority': str(row.get('priority', 'Medium')),
                'allowed': allowed,
                'execution_posture': 'Run here' if allowed else 'Run locally or in staging',
                'why': str(row.get('why', '')).strip(),
                'gating_reason': (
                    'This validation is safe to run in the current runtime.'
                    if allowed
                    else 'This validation is intentionally gated because the current cloud runtime is optimized for lighter checks.'
                ),
            }
        )
    return rows


def run_validation_task(task: str) -> dict[str, Any]:
    started_at = _utcnow()
    command = [
        sys.executable,
        'scripts/run_validation_task.py',
        '--task',
        str(task).strip(),
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    stdout = str(completed.stdout or '')
    stderr = str(completed.stderr or '')
    artifact_path = ''
    markdown_report = ''
    for line in stdout.splitlines():
        if line.startswith('Artifacts:'):
            artifact_path = line.split(':', 1)[1].strip()
        elif line.startswith('Markdown report:'):
            markdown_report = line.split(':', 1)[1].strip()
    return {
        'task': str(task).strip(),
        'started_at': started_at,
        'finished_at': _utcnow(),
        'return_code': int(completed.returncode),
        'status': 'Passed' if completed.returncode == 0 else 'Failed',
        'artifact_path': artifact_path,
        'markdown_report': markdown_report,
        'stdout': stdout,
        'stderr': stderr,
    }


def run_recommended_validation(validation_name: str) -> dict[str, Any]:
    task = recommended_validation_task(validation_name)
    execution_plan = build_validation_execution_plan(
        type('Plan', (), {'to_dict': lambda self, orient='records': [{'validation': validation_name}]})()
    )
    plan_row = execution_plan[0] if execution_plan else {}
    if not bool(plan_row.get('allowed', True)):
        return {
            'task': task,
            'validation_name': str(validation_name).strip(),
            'started_at': _utcnow(),
            'finished_at': _utcnow(),
            'return_code': 2,
            'status': 'Blocked',
            'artifact_path': '',
            'markdown_report': '',
            'stdout': '',
            'stderr': str(plan_row.get('gating_reason', 'Validation is blocked in the current runtime.')),
            'gating_reason': str(plan_row.get('gating_reason', 'Validation is blocked in the current runtime.')),
        }
    result = run_validation_task(task)
    result['validation_name'] = str(validation_name).strip()
    return result


__all__ = [
    'build_validation_execution_plan',
    'build_validation_runtime_profile',
    'recommended_validation_task',
    'run_recommended_validation',
    'run_validation_task',
]
