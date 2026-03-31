from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


PortalType = Literal["workday", "greenhouse", "lever", "generic"]
SubmitMode = Literal["review_first", "auto_submit"]


class WorkAuthorization(BaseModel):
    authorized_us: bool = True
    need_sponsorship_now: bool = False
    need_sponsorship_future: bool = False


class Profile(BaseModel):
    profile_id: str = "default-profile"
    full_name: str
    email: str
    phone: str
    location: str
    linkedin_url: str | None = None
    website_url: str | None = None
    master_resume_text: str
    work_authorization: WorkAuthorization = Field(default_factory=WorkAuthorization)
    default_answers: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class JobParseRequest(BaseModel):
    source_url: str
    raw_text: str
    title_hint: str | None = None
    company_hint: str | None = None
    location_hint: str | None = None
    page_metadata: dict[str, Any] = Field(default_factory=dict)
    apply_links: list[str] = Field(default_factory=list)


class JobParseResponse(BaseModel):
    title: str
    company: str
    location: str
    responsibilities: list[str]
    requirements: list[str]
    keywords: list[str]
    apply_url: str | None = None
    portal_type: PortalType
    raw_text_preview: str
    source_url: str


class TailorResumeRequest(BaseModel):
    job_description: str
    master_resume: str
    profile: Profile
    parsed_job: JobParseResponse | None = None


class TailorResumeResponse(BaseModel):
    tailored_resume_text: str
    keywords: list[str]
    change_summary: list[str]
    prompt_used: str


class CoverLetterRequest(BaseModel):
    candidate_name: str
    company: str
    role: str
    full_job_description: str
    tailored_resume_text: str


class CoverLetterResponse(BaseModel):
    plain_text: str
    markdown: str
    saved_path: str | None = None


class ApplyPlanField(BaseModel):
    name: str
    value_source: str
    required: bool = False


class ApplyPlanQuestion(BaseModel):
    question_type: str
    value: str
    confidence: float = 0.5
    explanation: str | None = None


class ApplyPlanStep(BaseModel):
    step_id: str
    action: str
    description: str
    required: bool = True


class ApplyPlanRequest(BaseModel):
    job: JobParseResponse
    profile: Profile
    portal_type: PortalType | None = None
    submit_mode: SubmitMode = "review_first"


class ApplyPlanResponse(BaseModel):
    portal_type: PortalType
    required_files: list[str]
    fields: list[ApplyPlanField]
    questions: list[ApplyPlanQuestion]
    steps: list[ApplyPlanStep]
    submit_mode: SubmitMode


class RunApplicationRequest(BaseModel):
    source_url: str
    raw_text: str
    profile: Profile
    title_hint: str | None = None
    company_hint: str | None = None
    location_hint: str | None = None
    page_metadata: dict[str, Any] = Field(default_factory=dict)
    apply_links: list[str] = Field(default_factory=list)
    submit_mode: SubmitMode = "review_first"
    run_worker: bool = False


class ArtifactRecord(BaseModel):
    label: str
    path: str


class RunApplicationResponse(BaseModel):
    application_id: str
    parsed_job: JobParseResponse
    tailored_resume: TailorResumeResponse
    cover_letter: CoverLetterResponse
    apply_plan: ApplyPlanResponse
    artifacts: list[ArtifactRecord]
    worker_status: str
    logs: list[str]
