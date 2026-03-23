from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.e2e.fixture_registry import get_default_fixture, get_fixture
from tests.e2e.validation_helpers import (
    build_failure_report,
    ensure_artifact_dir,
    run_validation,
    write_partial_state,
    write_report_files,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run reusable dataset-driven validation against the Streamlit app.')
    parser.add_argument('--dataset', required=False, help='Path to the dataset file to validate.')
    parser.add_argument('--fixture', required=False, default='default', help='Named dataset fixture from the registry. Defaults to the primary healthcare fixture.')
    parser.add_argument('--mode', choices=['quick', 'full', 'ui', 'soak'], default='quick', help='Validation depth.')
    parser.add_argument('--python-executable', default=sys.executable, help='Python executable used to launch Streamlit.')
    parser.add_argument('--visual-regression', action='store_true', help='Enable screenshot baseline comparison where configured.')
    parser.add_argument('--update-baselines', action='store_true', help='Update local visual baselines from the current run screenshots.')
    parser.add_argument('--soak-seconds', type=int, default=0, help='Additional soak duration in seconds.')
    parser.add_argument('--rerun-cycles', type=int, default=0, help='Additional tab/rerun stability cycles to execute.')
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    fixture = get_fixture(args.fixture)
    dataset_path = Path(args.dataset).resolve() if args.dataset else fixture.path.resolve()
    artifact_dir = ensure_artifact_dir(args.mode, dataset_path.name)
    command = ' '.join(sys.argv)
    write_partial_state(
        artifact_dir,
        {
            'status': 'runner-started',
            'dataset_path': str(dataset_path),
            'dataset_name': dataset_path.name,
            'mode': args.mode,
            'command': command,
        },
    )
    try:
        if not dataset_path.exists():
            raise FileNotFoundError(f'Dataset not found: {dataset_path}')
        report = run_validation(
            dataset_path,
            mode=args.mode,
            python_executable=args.python_executable,
            fixture=fixture,
            visual_regression=args.visual_regression,
            update_baselines=args.update_baselines,
            soak_seconds=args.soak_seconds,
            rerun_cycles=args.rerun_cycles,
        )
    except Exception as error:
        report = build_failure_report(
            dataset_path=dataset_path,
            mode=args.mode,
            artifact_dir=artifact_dir,
            command=command,
            commands_run=[command],
            last_completed_step='runner initialization',
            error=error,
        )
        write_report_files(report, artifact_dir)
        write_partial_state(
            artifact_dir,
            {
                'status': 'runner-failed',
                'dataset_path': str(dataset_path),
                'dataset_name': dataset_path.name,
                'mode': args.mode,
                'command': command,
                'artifact_dir': str(artifact_dir),
                'exception_type': type(error).__name__,
                'exception_message': str(error),
            },
        )
    report_json = Path(report.artifact_dir) / 'validation_report.json'
    report_md = Path(report.artifact_dir) / 'validation_report.md'
    print(f"Validation complete: {'PASS' if report.passed else 'FAIL'}")
    print(f'Artifacts: {report.artifact_dir}')
    print(f'JSON report: {report_json}')
    print(f'Markdown report: {report_md}')
    return 0 if report.passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
