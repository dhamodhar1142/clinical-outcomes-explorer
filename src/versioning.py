from __future__ import annotations

import os


def current_build_label() -> str:
    for key in (
        'CLINVERITY_BUILD_LABEL',
        'SMART_DATASET_ANALYZER_BUILD_LABEL',
        'STREAMLIT_GIT_COMMIT',
        'STREAMLIT_GIT_BRANCH',
    ):
        value = str(os.getenv(key, '')).strip()
        if value:
            if key == 'STREAMLIT_GIT_COMMIT' and len(value) > 10:
                return f'Build {value[:10]}'
            return value
    return 'Community launch build'


__all__ = ['current_build_label']
