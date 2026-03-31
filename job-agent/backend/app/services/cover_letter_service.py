from __future__ import annotations

from pathlib import Path

from ..models.schemas import CoverLetterRequest, CoverLetterResponse
from .storage_service import StorageService


class CoverLetterService:
    def __init__(self) -> None:
        self.storage = StorageService()

    def generate(self, payload: CoverLetterRequest) -> CoverLetterResponse:
        payload = CoverLetterRequest.model_validate(payload)
        body = (
            f"Dear Hiring Team,\n\n"
            f"I am excited to apply for the {payload.role} role at {payload.company}. My background combines hands-on analytics delivery, "
            f"clear business communication, and disciplined execution in data-heavy environments. After reviewing the role, I see a strong match "
            f"between your need for practical problem solving and my experience translating complex requirements into reporting, workflow improvement, "
            f"and decision-ready insights.\n\n"
            f"What stands out most about this opportunity is the emphasis on measurable outcomes, cross-functional collaboration, and dependable ownership. "
            f"In my recent work, I have supported stakeholders by improving reporting clarity, tightening data quality, and turning ambiguous operational questions "
            f"into structured analyses with actionable next steps. That same style shows up throughout my tailored resume, particularly in the way I approach "
            f"SQL-driven investigation, process improvement, and communication with non-technical partners.\n\n"
            f"I am especially motivated by roles where analytics is expected to do more than produce dashboards. The strongest teams use data to shape operations, "
            f"surface risk early, and create trust in the decisions that follow. That is the environment where I do my best work. I bring a practical mindset to data modeling, "
            f"quality checks, requirements gathering, and stakeholder alignment, with a focus on making outputs useful to the people who actually depend on them.\n\n"
            f"Your posting also points to the need for someone who can step into evolving priorities without losing execution discipline. I have consistently enjoyed that kind of ownership. "
            f"Whether the task is refining reporting logic, clarifying business definitions, or translating a broad request into an actionable plan, I aim to be the person who creates momentum, "
            f"reduces ambiguity, and follows through carefully.\n\n"
            f"I would bring a calm, execution-focused approach to {payload.company}: strong analytical judgment, reliable follow-through, and a genuine interest in "
            f"helping teams make better decisions faster. I would welcome the chance to discuss how my background can support the priorities outlined in this role.\n\n"
            f"Sincerely,\n{payload.candidate_name}"
        )
        markdown = "\n\n".join(body.split("\n\n"))
        return CoverLetterResponse(plain_text=body, markdown=markdown)

    def save(self, filename: str, content: str) -> str:
        path = self.storage.artifact_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(content, encoding="utf-8")
        return str(path)
