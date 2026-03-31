from __future__ import annotations

from ..models.schemas import (
    ApplyPlanRequest,
    ArtifactRecord,
    CoverLetterRequest,
    JobParseRequest,
    RunApplicationRequest,
    RunApplicationResponse,
    TailorResumeRequest,
)
from .apply_plan_service import ApplyPlanService
from .cover_letter_service import CoverLetterService
from .jd_parser import JDParserService
from .resume_renderer import ResumeRenderer
from .slugify import slugify
from .storage_service import StorageService
from .tailoring_service import ResumeTailoringService


class OrchestrationService:
    def __init__(self) -> None:
        self.parser = JDParserService()
        self.tailor = ResumeTailoringService()
        self.cover_letters = CoverLetterService()
        self.renderer = ResumeRenderer()
        self.plan_builder = ApplyPlanService()
        self.storage = StorageService()

    def run(self, payload: RunApplicationRequest) -> RunApplicationResponse:
        payload = RunApplicationRequest.model_validate(payload)
        logs = ["parsed_job:start"]
        parsed_job = self.parser.parse(
            JobParseRequest(
                source_url=payload.source_url,
                raw_text=payload.raw_text,
                title_hint=payload.title_hint,
                company_hint=payload.company_hint,
                location_hint=payload.location_hint,
                page_metadata=payload.page_metadata,
                apply_links=payload.apply_links,
            )
        )
        application_id = f"{self.storage.timestamp()}-{slugify(parsed_job.company)}-{slugify(parsed_job.title)}"
        logs.append("parsed_job:complete")

        tailored = self.tailor.tailor_resume(
            TailorResumeRequest(
                job_description=payload.raw_text,
                master_resume=payload.profile.master_resume_text,
                profile=payload.profile,
                parsed_job=parsed_job,
            )
        )
        logs.append("tailored_docs_ready")

        cover_letter = self.cover_letters.generate(
            CoverLetterRequest(
                candidate_name=payload.profile.full_name,
                company=parsed_job.company,
                role=parsed_job.title,
                full_job_description=payload.raw_text,
                tailored_resume_text=tailored.tailored_resume_text,
            )
        )
        logs.append("cover_letter_ready")

        plan = self.plan_builder.build(
            ApplyPlanRequest(job=parsed_job, profile=payload.profile, portal_type=parsed_job.portal_type, submit_mode=payload.submit_mode)
        )
        logs.append("apply_plan_ready")

        base_name = f"{slugify(parsed_job.company)}-{slugify(parsed_job.title)}"
        artifacts = [
            ArtifactRecord(label="resume_docx", path=self.renderer.save_resume_docx(application_id, f"{base_name}-resume.docx", tailored.tailored_resume_text)),
            ArtifactRecord(label="resume_pdf", path=self.renderer.save_resume_pdf(application_id, f"{base_name}-resume.pdf", tailored.tailored_resume_text)),
            ArtifactRecord(label="cover_letter_docx", path=self.renderer.save_cover_letter_docx(application_id, f"{base_name}-cover-letter.docx", cover_letter.plain_text)),
            ArtifactRecord(label="cover_letter_pdf", path=self.renderer.save_cover_letter_pdf(application_id, f"{base_name}-cover-letter.pdf", cover_letter.plain_text)),
            ArtifactRecord(label="tailoring_prompt", path=self.renderer.save_tailoring_prompt(application_id, "tailoring_prompt.txt", tailored.prompt_used)),
        ]
        logs.append("artifacts_saved")

        worker_status = "review_ready"
        if payload.run_worker:
            worker_status = "worker_queued"
            logs.append("worker_queued")

        return RunApplicationResponse(
            application_id=application_id,
            parsed_job=parsed_job,
            tailored_resume=tailored,
            cover_letter=cover_letter,
            apply_plan=plan,
            artifacts=artifacts,
            worker_status=worker_status,
            logs=logs,
        )
