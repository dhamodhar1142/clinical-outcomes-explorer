FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV SMART_DATASET_ANALYZER_ENV=local
ENV SMART_DATASET_ANALYZER_SECRETS_SOURCE=environment
ENV SMART_DATASET_ANALYZER_JOB_BACKEND=
ENV SMART_DATASET_ANALYZER_JOB_MAX_WORKERS=2
ENV SMART_DATASET_ANALYZER_JOB_QUEUE_NAME=smart-dataset-analyzer
ENV SMART_DATASET_ANALYZER_JOB_HEALTHCHECK_TIMEOUT=2.0
ENV SMART_DATASET_ANALYZER_STORAGE_BACKEND=local
ENV SMART_DATASET_ANALYZER_STORAGE_ROOT=/app/data/storage
ENV SMART_DATASET_ANALYZER_STORAGE_PREFIX=

ARG INSTALL_OPTIONAL_DEPS=false

COPY requirements.txt /app/requirements.txt
COPY requirements-optional.txt /app/requirements-optional.txt
# Install only the base runtime by default. Optional integrations stay opt-in
# so production images do not silently expand their dependency surface area.
RUN pip install --no-cache-dir -r /app/requirements.txt
RUN if [ "$INSTALL_OPTIONAL_DEPS" = "true" ]; then pip install --no-cache-dir -r /app/requirements-optional.txt; fi

COPY . /app

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
