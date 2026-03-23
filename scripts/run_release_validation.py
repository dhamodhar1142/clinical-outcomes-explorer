from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / 'artifacts' / 'validation'

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.e2e.fixture_registry import get_default_fixture, get_fixture


@dataclass
class ReleasePhaseResult:
    name: str
    command: list[str]
    status: str
    exit_code: int
    artifact_path: str = ''
    started_at: str = ''
    completed_at: str = ''
    output_path: str = ''
    notes: list[str] = field(default_factory=list)


@dataclass
class ReleaseValidationSummary:
    overall_status: str
    started_at: str
    completed_at: str
    selected_fixture: str
    release_artifact_dir: str
    phases: list[ReleasePhaseResult]
    commands_run: list[str]
    failures: list[dict[str, Any]] = field(default_factory=list)
    remaining_limitations: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ReleasePhaseSpec:
    name: str
    command: list[str]
    skip_reason: str | None = None
    parse_artifacts: bool = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run release-grade validation orchestration.')
    parser.add_argument('--fixture', default=None, help='Fixture key override. Defaults to the primary large fixture locally and the small fixture in CI mode.')
    parser.add_argument('--ci-mode', action='store_true', help='Use CI-friendly defaults and explicit skip/degrade reporting.')
    parser.add_argument('--skip-visual', action='store_true', help='Skip the visual regression phase.')
    parser.add_argument('--skip-soak', action='store_true', help='Skip the soak phase.')
    parser.add_argument('--soak-seconds', type=int, default=None, help='Override soak seconds.')
    parser.add_argument('--python-executable', default=None, help='Python executable to use. Defaults to the local .venv interpreter.')
    return parser.parse_args()


def _timestamp() -> str:
    return datetime.now(UTC).isoformat()


def _ensure_release_dir() -> Path:
    stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    path = ARTIFACT_ROOT / f'{stamp}-release-validation'
    path.mkdir(parents=True, exist_ok=True)
    return path


def _atomic_write(path: Path, content: str) -> None:
    temp_path = path.with_name(f'.{path.name}.tmp')
    temp_path.write_text(content, encoding='utf-8')
    temp_path.replace(path)


def _write_summary(summary: ReleaseValidationSummary, release_dir: Path) -> tuple[Path, Path]:
    json_path = release_dir / 'release_validation_summary.json'
    md_path = release_dir / 'release_validation_summary.md'
    _atomic_write(json_path, json.dumps(asdict(summary), indent=2, default=str))

    lines = [
        '# Release Validation Summary',
        '',
        f"- Overall result: `{summary.overall_status}`",
        f"- Started: `{summary.started_at}`",
        f"- Completed: `{summary.completed_at}`",
        f"- Selected fixture: `{summary.selected_fixture}`",
        f"- Release artifact directory: `{summary.release_artifact_dir}`",
        '',
        '## Phases',
    ]
    for phase in summary.phases:
        lines.append(f"- `{phase.status.upper()}` {phase.name}")
        lines.append(f"  - exit code: `{phase.exit_code}`")
        lines.append(f"  - command: `{' '.join(phase.command)}`")
        if phase.artifact_path:
            lines.append(f"  - artifacts: `{phase.artifact_path}`")
        if phase.output_path:
            lines.append(f"  - output log: `{phase.output_path}`")
        for note in phase.notes:
            lines.append(f"  - note: {note}")
    lines.extend(['', '## Commands Run'])
    lines.extend(f"- `{command}`" for command in summary.commands_run)
    lines.extend(['', '## Failures'])
    if summary.failures:
        for failure in summary.failures:
            lines.append(f"- `{failure.get('phase', 'unknown')}`: {failure.get('root_cause', '')}")
            if failure.get('artifact_path'):
                lines.append(f"  - artifacts: `{failure['artifact_path']}`")
    else:
        lines.append('- None.')
    lines.extend(['', '## Remaining Limitations'])
    if summary.remaining_limitations:
        lines.extend(f'- {item}' for item in summary.remaining_limitations)
    else:
        lines.append('- No additional limitations recorded.')
    _atomic_write(md_path, '\n'.join(lines))
    return json_path, md_path


def _parse_artifact_path(output: str) -> str:
    match = re.search(r'^Artifacts:\s+(.+)$', output, re.MULTILINE)
    return match.group(1).strip() if match else ''


def _run_phase(name: str, command: list[str], release_dir: Path, *, parse_artifacts: bool = True) -> ReleasePhaseResult:
    started_at = _timestamp()
    safe_name = re.sub(r'[^a-zA-Z0-9._-]+', '_', name.lower()).strip('_')
    output_path = release_dir / f'{safe_name}.log'
    print(f'==> {name}', flush=True)
    print(f"    Command: {' '.join(command)}", flush=True)
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    combined_output = '\n'.join(part for part in [completed.stdout, completed.stderr] if part)
    _atomic_write(output_path, combined_output)
    if completed.stdout:
        print(completed.stdout, end='' if completed.stdout.endswith('\n') else '\n', flush=True)
    if completed.stderr:
        print(completed.stderr, end='' if completed.stderr.endswith('\n') else '\n', file=sys.stderr, flush=True)
    artifact_path = _parse_artifact_path(combined_output) if parse_artifacts else ''
    notes: list[str] = []
    if not artifact_path:
        artifact_path = str(output_path)
        notes.append('Phase did not emit a dedicated artifact path; using the phase log as the phase artifact.')
    return ReleasePhaseResult(
        name=name,
        command=command,
        status='pass' if completed.returncode == 0 else 'fail',
        exit_code=int(completed.returncode),
        artifact_path=artifact_path,
        started_at=started_at,
        completed_at=_timestamp(),
        output_path=str(output_path),
        notes=notes,
    )


def _skipped_phase(name: str, command: list[str], reason: str) -> ReleasePhaseResult:
    now = _timestamp()
    return ReleasePhaseResult(
        name=name,
        command=command,
        status='skipped',
        exit_code=0,
        artifact_path='',
        started_at=now,
        completed_at=now,
        output_path='',
        notes=[reason],
    )


def _build_phase_specs(
    *,
    python_executable: str,
    fixture_key: str,
    ci_mode: bool,
    skip_visual: bool,
    skip_soak: bool,
    soak_seconds: int,
    baseline_ready: bool,
) -> list[ReleasePhaseSpec]:
    return [
        ReleasePhaseSpec(
            name='quick framework checks',
            command=[python_executable, '-m', 'unittest', 'tests.test_validation_automation', '-v'],
            parse_artifacts=False,
        ),
        ReleasePhaseSpec(
            name='full uploaded-dataset validation',
            command=[python_executable, '-u', 'scripts/run_dataset_validation.py', '--mode', 'full', '--fixture', fixture_key],
        ),
        ReleasePhaseSpec(
            name='visual regression',
            command=[python_executable, '-u', 'scripts/run_dataset_validation.py', '--mode', 'full', '--fixture', fixture_key, '--visual-regression'],
            skip_reason=(
                'Skipped because visual regression was disabled for this run.'
                if skip_visual
                else (
                    f'Skipped in CI mode because no baselines are available for fixture {fixture_key}.'
                    if ci_mode and not baseline_ready
                    else None
                )
            ),
        ),
        ReleasePhaseSpec(
            name='cross-dataset cache validation',
            command=[python_executable, 'scripts/run_cross_dataset_cache_validation.py'],
        ),
        ReleasePhaseSpec(
            name='accessibility audit',
            command=[python_executable, '-u', 'scripts/run_dataset_validation.py', '--mode', 'ui', '--fixture', fixture_key],
        ),
        ReleasePhaseSpec(
            name='5-minute soak test',
            command=[python_executable, '-u', 'scripts/run_dataset_validation.py', '--mode', 'soak', '--fixture', fixture_key, '--soak-seconds', str(soak_seconds), '--rerun-cycles', '3'],
            skip_reason='Skipped because soak was disabled for this run.' if skip_soak else None,
        ),
    ]


def main() -> int:
    args = parse_args()
    python_executable = args.python_executable or str((ROOT / '.venv' / 'Scripts' / 'python.exe').resolve())
    fixture = get_fixture(args.fixture) if args.fixture else (get_fixture('small-healthcare') if args.ci_mode else get_default_fixture())
    selected_fixture = str(fixture.resolve_path(require_exists=not args.ci_mode).resolve())
    release_dir = _ensure_release_dir()
    started_at = _timestamp()
    fixture_key = fixture.key
    soak_seconds = args.soak_seconds if args.soak_seconds is not None else (60 if args.ci_mode else 300)
    baseline_dir = ROOT / 'tests' / 'e2e' / 'baselines' / fixture.key
    baseline_ready = baseline_dir.exists() and any(baseline_dir.glob('*.png'))
    phases_spec = _build_phase_specs(
        python_executable=python_executable,
        fixture_key=fixture_key,
        ci_mode=args.ci_mode,
        skip_visual=args.skip_visual,
        skip_soak=args.skip_soak,
        soak_seconds=soak_seconds,
        baseline_ready=baseline_ready,
    )
    phase_results: list[ReleasePhaseResult] = []
    failures: list[dict[str, Any]] = []
    overall_status = 'PASS'
    remaining_limitations = [
        'Visual regression remains environment-sensitive and depends on the local baseline/browser/font stack.',
        'Accessibility checks are workflow-focused automation checks, not a full WCAG certification pass.',
    ]
    if args.ci_mode:
        remaining_limitations.append('CI mode uses a CI-friendly fixture and may skip or shorten environment-sensitive phases when configured.')
        if fixture.large_file:
            remaining_limitations.append('Large-fixture CI execution depends on fixture availability in the checkout environment.')
        else:
            remaining_limitations.append('CI mode is running against the small healthcare fixture instead of the large local default fixture.')
    try:
        for phase in phases_spec:
            if phase.skip_reason:
                result = _skipped_phase(phase.name, phase.command, phase.skip_reason)
            else:
                result = _run_phase(phase.name, phase.command, release_dir, parse_artifacts=phase.parse_artifacts)
            phase_results.append(result)
            summary = ReleaseValidationSummary(
                overall_status='PASS' if all(item.status != 'fail' for item in phase_results) else 'FAIL',
                started_at=started_at,
                completed_at=_timestamp(),
                selected_fixture=selected_fixture,
                release_artifact_dir=str(release_dir),
                phases=phase_results,
                commands_run=[' '.join(item.command) for item in phase_results],
                failures=failures,
                remaining_limitations=remaining_limitations,
            )
            _write_summary(summary, release_dir)
            if result.status != 'pass':
                if result.status == 'skipped':
                    continue
                overall_status = 'FAIL'
                failures.append(
                    {
                        'phase': phase.name,
                        'root_cause': f'Phase exited with code {result.exit_code}. See the phase log for details.',
                        'artifact_path': result.artifact_path,
                    }
                )
                break
    except Exception as error:
        overall_status = 'FAIL'
        failures.append(
            {
                'phase': 'release orchestrator',
                'root_cause': f'{type(error).__name__}: {error}',
                'artifact_path': str(release_dir),
            }
        )
    summary = ReleaseValidationSummary(
        overall_status=overall_status,
        started_at=started_at,
        completed_at=_timestamp(),
        selected_fixture=selected_fixture,
        release_artifact_dir=str(release_dir),
        phases=phase_results,
        commands_run=[' '.join(item.command) for item in phase_results],
        failures=failures,
        remaining_limitations=remaining_limitations,
    )
    json_path, md_path = _write_summary(summary, release_dir)
    print(f'Release validation complete: {summary.overall_status}', flush=True)
    print(f'Release artifacts: {release_dir}', flush=True)
    print(f'JSON summary: {json_path}', flush=True)
    print(f'Markdown summary: {md_path}', flush=True)
    return 0 if summary.overall_status == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
