# AGENTS.md

## Purpose
This repository contains a clinical data quality and analytics application. Agents working here must prioritize dataset integrity, uploaded-dataset persistence, UI visibility, and end-to-end workflow reliability.

## Core rules
- Treat the uploaded dataset as the single authoritative dataset until the user explicitly changes source.
- Never allow automatic fallback to demo/default dataset after upload.
- Always verify dataset identity across reruns, tab switches, and long-running analysis states.
- Prefer architecture fixes over one-off patches.
- Keep UI changes accessible, readable, and production-quality.
- Preserve performance for large datasets.

## Validation automation
Use the reusable dataset-validation workflow in this repo for real upload checks whenever practical.

Default fixtures by entry point:
- Quick validation:
  - fixture key: `small-healthcare`
  - dataset path: `tests/fixtures/datasets/SMALL_HEALTHCARE_VISITS.csv`
- Local full validation and local release validation:
  - fixture key: `default`
  - dataset path: `tests/fixtures/datasets/STG_EHP__VIST.csv`

Primary full-validation fixture details:
- `tests/fixtures/datasets/STG_EHP__VIST.csv`
- Dataset type: healthcare
- Treat this as the primary uploaded-dataset validation fixture unless the user explicitly provides a different dataset.
- Expected key source columns:
  - `REFR_NO`, `PAT_ID`, `MEDT_ID`, `VIS_EN`, `VIS_EX`, `VSTAT_CD`, `VSTAT_DES`, `VTYPE_CD`, `VTYPE_DES`, `ROM_ID`
- Expected date columns:
  - `VIS_EN`, `VIS_EX`
- Expected identifier fields:
  - `REFR_NO`, `PAT_ID`, `MEDT_ID`, `ROM_ID`
- Expected categorical/code fields:
  - `VSTAT_DES`, `VTYPE_DES`, `VSTAT_CD`, `VTYPE_CD`
- Expected approximate row count:
  - `917331`
- This is a large-file fixture, so validation should explicitly use the upload/streaming path and must not silently fall back to demo mode.

Primary commands:
- Quick validation: `.\scripts\run_quick_validation.ps1`
- Full validation: `.\scripts\run_full_validation.ps1`
- Release validation: `.\scripts\run_release_validation.ps1`
- UI visibility audit: `.\scripts\run_ui_visibility_audit.ps1`
- Visual regression: `.\scripts\run_visual_regression.ps1`
- Soak test: `.\scripts\run_soak_test.ps1`
- Accessibility audit: `.\scripts\run_accessibility_audit.ps1`
- Cross-dataset cache validation: `.\scripts\run_cross_dataset_cache_validation.ps1`
- Override dataset if needed: `.\scripts\run_quick_validation.ps1 -Dataset <path-to-dataset>`
- Prompt-friendly task runner: `python scripts/run_validation_task.py --task quick|full|full-visual|soak-5m|accessibility|cross-dataset-cache|release`
- Cross-platform entry point: `python scripts/run_dataset_validation.py --mode quick|full|ui|soak --fixture <fixture-key>`

CI expectations:
- quick CI checks are the default gate for routine changes
- release validation is the preferred final local/manual gate before major changes are considered complete
- CI may use the `small-healthcare` fixture instead of the large local default fixture when the environment is constrained
- CI visual regression may be skipped explicitly when matching baselines are not available for the CI fixture
- quick validation should use the `small-healthcare` fixture unless the user explicitly asks for a different dataset

When to use which command:
- generate or refresh baselines only when the UI intentionally changed and you want to accept the new visual state:
  - `python scripts/run_dataset_validation.py --mode full --fixture default --visual-regression --update-baselines`
- run visual regression after baseline creation or when checking for UI drift:
  - `.\scripts\run_visual_regression.ps1`
- run soak validation when checking long-lived uploaded-dataset persistence and rerun stability:
  - `.\scripts\run_soak_test.ps1`
- run release validation as the preferred final gate before major changes are considered complete:
  - `.\scripts\run_release_validation.ps1`

Artifacts are written to:
- phase runs: `artifacts/validation/<timestamp>-<mode>-<dataset-name>/`
- release orchestration: `artifacts/validation/<timestamp>-release-validation/`

Expected outputs:
- `validation_report.json`
- `validation_report.md`
- screenshots for critical tabs when browser automation is available
- Streamlit stdout/stderr logs when the browser workflow is used
- visual diff artifacts when visual regression is enabled and mismatches occur

## Required workflow for analysis/testing tasks
When asked to validate the app with a dataset:
1. Start the application.
2. Upload the provided dataset through the real UI when possible.
3. Wait for loading/progress to complete.
4. Verify the uploaded dataset remains active.
5. Check:
   - row count
   - column count
   - source mode
   - dataset identifier / cache key
   - field detection
   - expected healthcare source columns
   - expected date fields (`VIS_EN`, `VIS_EX`) detected as date-like when validating the primary healthcare fixture
   - mapping confidence / readiness surfaces
   - profiling
   - quality review
   - healthcare analytics
   - trend/cohort analysis
   - insights/export views
6. Confirm no tab reverts to demo/default context.
7. Capture screenshots and produce a concise report.
8. If issues are found, fix them, rerun tests, and summarize root cause.

Browser-first expectation:
- Prefer the browser automation workflow in `tests/e2e/validation_helpers.py`.
- If Playwright is unavailable, use the programmatic fallback only as a temporary validation path and clearly report that UI visibility was not fully audited.
- For the primary healthcare fixture, keep the uploaded dataset as the single authoritative source throughout the long-running large-file workflow.
- Always capture and report browser-environment readiness:
  - Playwright installed or not
  - browser launchability
  - any degradation reason when fallback is used

## UI/UX rules
- Reject low-contrast or nearly invisible controls.
- Ensure buttons, tabs, alerts, and form controls remain readable in the active theme.
- Prefer a cohesive design system over isolated style patches.

## Test philosophy
- Add regression tests for every confirmed bug.
- Keep deterministic logic in scripts/helpers where possible.
- Use browser automation for real upload and tab validation.
- Save artifacts for failures: screenshots, logs, and generated reports.
- Keep reusable validation logic in `tests/e2e/` and runnable entry points in `scripts/`.

## Definition of done
A task is not done unless:
- the uploaded dataset persists correctly,
- all major tabs use the uploaded dataset,
- critical controls are visible and usable,
- automated tests pass,
- and a short root-cause summary is written.
