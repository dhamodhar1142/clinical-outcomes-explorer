from __future__ import annotations

from ..models.schemas import (
    ApplyPlanField,
    ApplyPlanQuestion,
    ApplyPlanRequest,
    ApplyPlanResponse,
    ApplyPlanStep,
)
from .screening_answer_service import ScreeningAnswerService


class ApplyPlanService:
    def __init__(self) -> None:
        self.answer_service = ScreeningAnswerService()

    def build(self, payload: ApplyPlanRequest) -> ApplyPlanResponse:
        portal_type = payload.portal_type or payload.job.portal_type
        fields = [
            ApplyPlanField(name="first_name", value_source="profile.full_name", required=True),
            ApplyPlanField(name="email", value_source="profile.email", required=True),
            ApplyPlanField(name="phone", value_source="profile.phone", required=True),
            ApplyPlanField(name="location", value_source="profile.location", required=False),
            ApplyPlanField(name="linkedin_url", value_source="profile.linkedin_url", required=False),
        ]
        questions: list[ApplyPlanQuestion] = self.answer_service.answer_common_questions(payload.profile)
        steps = [
            ApplyPlanStep(step_id="open_apply", action="navigate", description="Open the application URL."),
            ApplyPlanStep(step_id="resume_upload", action="upload", description="Upload tailored resume artifact."),
            ApplyPlanStep(step_id="cover_letter_upload", action="upload", description="Upload cover letter artifact."),
            ApplyPlanStep(step_id="profile_fill", action="fill_fields", description="Complete common applicant fields."),
            ApplyPlanStep(step_id="screening", action="answer_questions", description="Answer common screening questions."),
            ApplyPlanStep(step_id="review", action="review", description="Stop on review page for human approval."),
        ]
        if portal_type == "workday":
            steps.insert(1, ApplyPlanStep(step_id="workday_auth", action="account_gate", description="Handle login or account creation if needed."))
        elif portal_type in {"greenhouse", "lever"}:
            steps.insert(1, ApplyPlanStep(step_id="portal_specific_prep", action="portal_prep", description=f"Prepare {portal_type} application form and attachments."))
        return ApplyPlanResponse(
            portal_type=portal_type,
            required_files=["resume", "cover_letter"],
            fields=fields,
            questions=questions,
            steps=steps,
            submit_mode=payload.submit_mode,
        )
