# Visual Regression Baselines

This folder stores optional screenshot baselines for validation runs.

Tracked screens:
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

Baselines are organized by fixture key, for example:
- `tests/e2e/baselines/primary-healthcare/`

To create or refresh local baselines:

```powershell
.\.venv\Scripts\python.exe scripts\run_dataset_validation.py --mode full --fixture default --visual-regression --update-baselines
```

Visual regression remains optional. If baselines are missing, the run reports the condition explicitly instead of silently passing.
