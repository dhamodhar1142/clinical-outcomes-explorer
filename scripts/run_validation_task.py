from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


TASK_COMMANDS = {
    'quick': ['scripts/run_dataset_validation.py', '--mode', 'quick', '--fixture', 'small-healthcare'],
    'full': ['scripts/run_dataset_validation.py', '--mode', 'full', '--fixture', 'default'],
    'full-visual': ['scripts/run_dataset_validation.py', '--mode', 'full', '--fixture', 'default', '--visual-regression'],
    'soak-5m': ['scripts/run_dataset_validation.py', '--mode', 'soak', '--fixture', 'default', '--soak-seconds', '300', '--rerun-cycles', '3'],
    'accessibility': ['scripts/run_dataset_validation.py', '--mode', 'ui', '--fixture', 'default'],
    'cross-dataset-cache': ['scripts/run_cross_dataset_cache_validation.py'],
    'release': ['scripts/run_release_validation.py'],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Prompt-friendly validation task runner.')
    parser.add_argument('--task', choices=sorted(TASK_COMMANDS), required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    command = [sys.executable, *TASK_COMMANDS[args.task]]
    completed = subprocess.run(command, cwd=ROOT)
    return int(completed.returncode)


if __name__ == '__main__':
    raise SystemExit(main())
