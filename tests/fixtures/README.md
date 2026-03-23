# Validation Fixtures

Use this folder for deterministic uploaded-dataset validation fixtures that are exercised by the reusable automation workflow.

Suggested usage:
- small CSV fixture for quick upload validation
- representative healthcare upload fixture for full workflow validation
- edge-case fixtures for persistence, cache, and UI fallback regressions

Primary full-workflow fixture:
- `datasets/STG_EHP__VIST.csv`
- dataset type: `healthcare`
- expected approximate row count: `917331`
- expected key columns:
  - `REFR_NO`, `PAT_ID`, `MEDT_ID`, `VIS_EN`, `VIS_EX`, `VSTAT_CD`, `VSTAT_DES`, `VTYPE_CD`, `VTYPE_DES`, `ROM_ID`
- default wrapper commands use this dataset unless `-Dataset` is provided explicitly

The browser validation runner accepts any dataset path, so fixtures here are optional convenience assets rather than the only supported source.
