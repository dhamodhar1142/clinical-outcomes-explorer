from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / 'data'
DOCS_DIR = REPO_ROOT / 'docs'


def repo_path(*parts: str) -> Path:
    return REPO_ROOT.joinpath(*parts)


def data_path(*parts: str) -> Path:
    return DATA_DIR.joinpath(*parts)


def docs_path(*parts: str) -> Path:
    return DOCS_DIR.joinpath(*parts)

