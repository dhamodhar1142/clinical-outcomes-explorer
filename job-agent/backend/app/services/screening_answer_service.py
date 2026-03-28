from __future__ import annotations

from ..models.schemas import ApplyPlanQuestion, Profile


class ScreeningAnswerService:
    def answer_common_questions(self, profile: Profile) -> list[ApplyPlanQuestion]:
        auth = profile.work_authorization
        defaults = profile.default_answers
        return [
            ApplyPlanQuestion(
                question_type="work_authorization",
                value="Yes" if auth.authorized_us else "No",
                confidence=0.98,
                explanation="Derived from profile.work_authorization.authorized_us.",
            ),
            ApplyPlanQuestion(
                question_type="sponsorship",
                value="No" if not auth.need_sponsorship_now else "Yes",
                confidence=0.98,
                explanation="Derived from current sponsorship requirement.",
            ),
            ApplyPlanQuestion(
                question_type="disability",
                value=defaults.get("disability_status", "Decline to answer"),
                confidence=0.8,
                explanation="Falls back to safe default answers.",
            ),
            ApplyPlanQuestion(
                question_type="veteran",
                value=defaults.get("veteran_status", "Decline to answer"),
                confidence=0.8,
                explanation="Falls back to safe default answers.",
            ),
        ]
