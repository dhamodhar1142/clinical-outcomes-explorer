from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class StorageService:
    def __init__(self) -> None:
        self.root = Path(os.getenv("JOB_AGENT_DATA_DIR", "job-agent/data"))
        self.artifact_dir = Path(os.getenv("JOB_AGENT_ARTIFACT_DIR", self.root / "artifacts"))
        self.log_dir = Path(os.getenv("JOB_AGENT_LOG_DIR", self.root / "logs"))
        self.screenshot_dir = Path(os.getenv("JOB_AGENT_SCREENSHOT_DIR", self.root / "screenshots"))
        for folder in (self.root, self.artifact_dir, self.log_dir, self.screenshot_dir):
            folder.mkdir(parents=True, exist_ok=True)

    def application_dir(self, application_id: str) -> Path:
        path = self.artifact_dir / application_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def timestamp(self) -> str:
        return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")

    def write_text(self, path: Path, content: str) -> str:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return str(path)

    def write_json(self, path: Path, payload: dict[str, Any]) -> str:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return str(path)
