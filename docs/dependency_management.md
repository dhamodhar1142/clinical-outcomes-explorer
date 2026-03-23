# Dependency Management

This project separates dependencies into three install tiers so the base
deployment stays lean and optional integrations remain explicit.

## Core runtime

Install the base application with:

```bash
pip install -r requirements.txt
```

This tier includes the packages required to launch the Streamlit app and run
the supported core analytics workflows.

## Optional integrations

Install optional packages with:

```bash
pip install -r requirements-optional.txt
```

This tier currently includes:

- `xgboost` for optional model-comparison support
- `openai` for enhanced Copilot responses when `OPENAI_API_KEY` is configured
- `playwright` for browser smoke tests and screenshot/demo-asset generation workflows

The app is designed to keep working when these packages are not installed.

## Deployment guidance

- Production-style installs should normally use only `requirements.txt`.
- Add `requirements-optional.txt` only when that environment intentionally needs OpenAI, XGBoost, or Playwright-backed workflows.
- `requirements-dev.txt` is the easiest local path when you want the full validation and demo toolchain in one environment.

The included `Dockerfile` follows the same pattern:

- installs `requirements.txt` by default
- installs `requirements-optional.txt` only when `INSTALL_OPTIONAL_DEPS=true`

## Full local development and validation

Install the broader local environment with:

```bash
pip install -r requirements-dev.txt
```

This includes the base runtime plus the optional integration packages so local
development, demos, and wider validation are easier to run from one setup.
