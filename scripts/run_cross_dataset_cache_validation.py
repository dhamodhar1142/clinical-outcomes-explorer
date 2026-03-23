from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.e2e.validation_helpers import run_cross_dataset_cache_validation


def main() -> int:
    report = run_cross_dataset_cache_validation(python_executable=sys.executable)
    print(f"Cross-dataset cache validation complete: {'PASS' if report.passed else 'FAIL'}")
    print(f'Artifacts: {report.artifact_dir}')
    return 0 if report.passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
