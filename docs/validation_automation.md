# Dataset Validation Automation

This repository includes a reusable Codex-friendly validation workflow for uploaded datasets.

## Goals
- validate the real Streamlit app against uploaded datasets
- confirm uploaded-dataset authority across tabs and reruns
- capture screenshots, diagnostics, and reports
- support both quick and full validation modes

## Default full-workflow fixture
The default local full-validation healthcare fixture is:

```text
tests/fixtures/datasets/STG_EHP__VIST.csv
```

This fixture is intended for local/manual validation and is not meant to be a normal CI-tracked asset going forward.

Fixture expectations:
- dataset name: `STG_EHP__VIST.csv`
- dataset type: `healthcare`
- expected key source columns:
  - `REFR_NO`, `PAT_ID`, `MEDT_ID`, `VIS_EN`, `VIS_EX`, `VSTAT_CD`, `VSTAT_DES`, `VTYPE_CD`, `VTYPE_DES`, `ROM_ID`
- expected date columns:
  - `VIS_EN`, `VIS_EX`
- expected identifier fields:
  - `REFR_NO`, `PAT_ID`, `MEDT_ID`, `ROM_ID`
- expected categorical/code fields:
  - `VSTAT_DES`, `VTYPE_DES`, `VSTAT_CD`, `VTYPE_CD`
- expected approximate row count:
  - `917331`

This is a large-file fixture. The validator explicitly checks that large-file handling remains on the uploaded-dataset path and does not silently revert to demo/default mode.

## Default fixtures by entry point
- Quick validation:
  - fixture key: `small-healthcare`
  - dataset path: `tests/fixtures/datasets/SMALL_HEALTHCARE_VISITS.csv`
  - tracked in git and safe for CI
- Local full validation:
  - fixture key: `default`
  - dataset path: `tests/fixtures/datasets/STG_EHP__VIST.csv`
  - local/manual fixture
- Local release validation:
  - fixture key: `default`
  - dataset path: `tests/fixtures/datasets/STG_EHP__VIST.csv`
  - local/manual fixture
- CI quick validation:
  - fixture key: `small-healthcare`
- CI manual release validation:
  - fixture key: `small-healthcare` by default, overrideable to `default` when the environment and checkout support the large fixture

## Large fixture handling
Recommended hygiene:
- keep the small fixtures in git for CI and shared development
- keep `STG_EHP__VIST.csv` as a local/manual validation asset
- do not rely on the large fixture being present in fresh public clones

The validator resolves the local large fixture in this order:
1. `SMART_DATASET_ANALYZER_LARGE_FIXTURE_PATH`
2. `tests/fixtures/datasets/STG_EHP__VIST.csv`
3. `data/local_fixtures/STG_EHP__VIST.csv`

To restore local full validation on a fresh clone, place the file in either:

```text
tests/fixtures/datasets/STG_EHP__VIST.csv
```

or:

```text
data/local_fixtures/STG_EHP__VIST.csv
```

or set:

```powershell
$env:SMART_DATASET_ANALYZER_LARGE_FIXTURE_PATH="C:\path\to\STG_EHP__VIST.csv"
```

## Commands

PowerShell helpers:

```powershell
.\scripts\run_quick_validation.ps1
.\scripts\run_full_validation.ps1
.\scripts\run_release_validation.ps1
.\scripts\run_ui_visibility_audit.ps1
.\scripts\run_visual_regression.ps1
.\scripts\run_soak_test.ps1
.\scripts\run_accessibility_audit.ps1
.\scripts\run_cross_dataset_cache_validation.ps1
```

`run_full_validation.ps1` now:
- prints visible run progress
- uses unbuffered Python output
- opens the latest markdown report automatically when the run completes

Direct Python entry point:

```powershell
python scripts\run_dataset_validation.py --mode quick --fixture small-healthcare
python scripts\run_dataset_validation.py --mode full --fixture default
python scripts\run_dataset_validation.py --mode ui --fixture default
python scripts\run_dataset_validation.py --mode full --fixture default --visual-regression
python scripts\run_dataset_validation.py --mode soak --fixture default --soak-seconds 300 --rerun-cycles 3
python scripts\run_release_validation.py
```

Prompt-friendly task runner:

```powershell
python scripts\run_validation_task.py --task quick
python scripts\run_validation_task.py --task full
python scripts\run_validation_task.py --task full-visual
python scripts\run_validation_task.py --task soak-5m
python scripts\run_validation_task.py --task accessibility
python scripts\run_validation_task.py --task cross-dataset-cache
python scripts\run_validation_task.py --task release
```

Prompt-to-command mapping:
- `run full validation`
  - `python scripts\run_validation_task.py --task full`
- `run full validation with visual regression`
  - `python scripts\run_validation_task.py --task full-visual`
- `run 5-minute soak test`
  - `python scripts\run_validation_task.py --task soak-5m`
- `run accessibility audit`
  - `python scripts\run_validation_task.py --task accessibility`
- `run cross-dataset cache validation`
  - `python scripts\run_validation_task.py --task cross-dataset-cache`
- `run release validation`
  - `python scripts\run_validation_task.py --task release`

Override the dataset explicitly when needed:

```powershell
.\scripts\run_quick_validation.ps1 -Dataset .\path\to\other.csv
python scripts\run_dataset_validation.py --dataset .\path\to\other.csv --mode full --fixture <fixture-key>
```

## Modes
- `quick`: upload + core tab checks + diagnostics + screenshots for the main workflow
- `full`: upload + all major tabs + visibility audit + full evidence capture
- `ui`: focuses on browser-based visibility and critical control readability checks

Validation checks include:
- uploaded dataset remains the active source
- no fallback to demo/default mode
- row/column stability
- expected healthcare source columns present
- `VIS_EN` and `VIS_EX` recognized as date-like for the primary healthcare fixture
- Dataset Profile, Data Quality, Healthcare Analytics, Trend Analysis, Cohort Analysis, Key Insights, and Export Center remain bound to the uploaded dataset
- critical controls remain visible and readable

Expanded validation areas:
- browser environment readiness
- large-dataset fidelity with authoritative vs sampled metrics distinguished explicitly
- accessibility-oriented checks
- soak/rerun stability
- export artifact validation
- optional visual regression against local baselines

## Release validation
Release validation is the preferred final gate before major changes are considered complete.

PowerShell:

```powershell
.\scripts\run_release_validation.ps1
```

Python:

```powershell
python scripts\run_release_validation.py
python scripts\run_release_validation.py --ci-mode --fixture small-healthcare
```

Expected duration:
- approximately 20-25 minutes locally, depending on machine speed and browser startup

Phases:
1. quick framework checks
2. full uploaded-dataset validation
3. visual regression
4. cross-dataset cache validation
5. accessibility audit
6. 5-minute soak test

Release outputs:
- a per-phase artifact bundle from the underlying commands
- top-level release summary files:
  - `release_validation_summary.json`
  - `release_validation_summary.md`

These are written under:

```text
artifacts/validation/<timestamp>-release-validation/
```

Interpreting failures:
- if `quick framework checks` fails, treat it as a framework/regression issue before trusting later browser results
- if `full uploaded-dataset validation` fails, treat it as a functional product regression
- if `visual regression` fails, check whether the diffs are real UI changes or expected baseline drift
- if `cross-dataset cache validation` fails, treat it as dataset-identity/cache contamination
- if `accessibility audit` fails, review control visibility/focus diagnostics
- if `5-minute soak test` fails, treat it as a persistence/stability regression

## CI workflows
GitHub Actions workflows:

- `CI Quick`
  - workflow file: `.github/workflows/ci-quick.yml`
  - runs on `push` and `pull_request`
  - intended as the default routine gate for repository changes
- `CI Manual Release Validation`
  - workflow file: `.github/workflows/ci-manual-release-validation.yml`
  - runs on `workflow_dispatch`
  - intended as the heavier manual CI gate

What runs automatically on PRs:
- compile checks
- targeted validation/unit tests
- import sanity
- CI-friendly quick uploaded-dataset validation using the small healthcare fixture

What runs only manually:
- CI release validation orchestration
- CI-friendly full validation flow
- CI visual regression
- CI cross-dataset cache validation
- CI accessibility audit
- CI soak validation

CI-friendly vs local-only expectations:
- CI-friendly default fixture:
  - `small-healthcare`
- Local/manual preferred full fixture:
  - `default` / `STG_EHP__VIST.csv`
- Large-fixture validation may remain local/manual when the CI environment or checkout does not contain the large fixture.
- CI release validation uses `--ci-mode` so summaries can explicitly note fixture downgrades or skipped phases.
- CI visual regression may be skipped automatically when fixture baselines are not available for the selected CI fixture.

Recommended CI command equivalents:

```powershell
python -u scripts/run_dataset_validation.py --mode quick --fixture small-healthcare
python -u scripts/run_release_validation.py --ci-mode --fixture small-healthcare
```

CI artifacts:
- uploaded from `artifacts/validation`
- include reports, screenshots, logs, and diff artifacts when produced

How to interpret CI artifacts:
- quick CI artifacts are the first place to inspect for routine failures on PRs
- manual CI release artifacts include a top-level `release_validation_summary.json` and `release_validation_summary.md`
- if browser-heavy phases degrade or are skipped, the summary notes why instead of silently pretending they ran

## Fixture coverage
- `default` / `primary-healthcare`
  - `tests/fixtures/datasets/STG_EHP__VIST.csv`
  - large healthcare full-workflow fixture
- `small-healthcare`
  - `tests/fixtures/datasets/SMALL_HEALTHCARE_VISITS.csv`
  - small healthcare quick-run fixture
- `secondary-healthcare`
  - `tests/fixtures/datasets/ALT_EHP__VIST.csv`
  - second uploaded dataset for cache contamination tests
- `malformed-healthcare`
  - `tests/fixtures/datasets/MALFORMED_VISITS.csv`
  - malformed upload fixture for explicit safe-failure validation

## Artifacts
Phase-run outputs are written to:

```text
artifacts/validation/<timestamp>-<mode>-<dataset-name>/
```

Release-orchestration outputs are written to:

```text
artifacts/validation/<timestamp>-release-validation/
```

Expected files:
- `validation_report.json`
- `validation_report.md`
- tab screenshots when browser automation is available
- `streamlit_stdout.log`
- `streamlit_stderr.log`

## Browser automation
The validator prefers Playwright for real upload and tab navigation.

If Playwright is not installed, the runner falls back to a programmatic validation path that:
- loads the uploaded dataset through the pipeline
- checks dataset identity and downstream surfaces
- produces a report

When fallback is used, the report explicitly notes that UI visibility was not fully audited.

Browser readiness is reported explicitly:
- whether Playwright is installed
- whether Chromium can launch
- the exact degradation reason if fallback was required

To enable full browser validation locally:
1. install dev dependencies
2. run `python -m playwright install chromium`
3. rerun the validation command

## Visual regression
Visual regression is optional and uses local baselines in:

```text
tests/e2e/baselines/<fixture-key>/
```

To create or refresh baselines:

```powershell
python scripts\run_dataset_validation.py --mode full --fixture default --visual-regression --update-baselines
.\scripts\run_visual_regression.ps1
```

The recommended explicit baseline-generation command from PowerShell is:

```powershell
.\.venv\Scripts\python.exe -u scripts\run_dataset_validation.py --mode full --fixture default --visual-regression --update-baselines
```

Tracked visual surfaces for the default healthcare fixture:
- `data_intake.png`
- `overview.png`
- `column_detection.png`
- `quality_review.png`
- `readiness.png`
- `healthcare_intelligence.png`
- `trend_analysis.png`
- `cohort_analysis.png`
- `key_insights.png`
- `export_center.png`

When enabled, diff artifacts are written under the run artifact directory in:

```text
visual_diffs/
```

Visual runs use a practical screenshot-diff threshold so minor rendering noise does not fail the run unnecessarily. Baseline updates remain optional and do not affect normal full validation.
