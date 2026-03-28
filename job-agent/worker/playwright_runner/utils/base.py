from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class StepLog:
    stage: str
    detail: str


@dataclass
class BaseApplier:
    plan: dict
    artifacts: dict[str, str] = field(default_factory=dict)
    logs: list[StepLog] = field(default_factory=list)

    async def log(self, stage: str, detail: str) -> None:
        self.logs.append(StepLog(stage=stage, detail=detail))

    async def run(self) -> dict:
        raise NotImplementedError
