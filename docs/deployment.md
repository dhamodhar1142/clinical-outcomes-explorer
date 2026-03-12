# Deployment Notes

## Recommended runtime

- Python 3.11 or 3.12
- Streamlit app entrypoint: `app.py`

## Local setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Optional extras:

```bash
pip install -r requirements-optional.txt
```

## Launch

```bash
streamlit run app.py
```

## Streamlit deployment checklist

- ensure `app.py` is present at the repository root
- ensure `requirements.txt` is committed
- ensure the `data/` folder is included for built-in demo datasets
- optional packages are not required for app startup
- the app should degrade safely if:
  - `xgboost` is missing
  - `openai` is missing

## Validation commands

```bash
python -m compileall app.py src tests
python -m unittest tests.test_healthcare_regressions tests.test_modeling_serialization tests.test_temporal_detection tests.test_copilot_readmission_workflows tests.test_remediation_engine tests.test_decision_support tests.test_presentation_support tests.test_portfolio_support -v
```

## Deployment behavior notes

- large datasets are sampled safely for profiling and quality review
- synthetic helper fields are disclosed explicitly in the UI and reports
- governance, privacy, and standards outputs are readiness aids, not certification outputs
