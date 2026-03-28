from fastapi import APIRouter

from ..models.schemas import (
    CoverLetterRequest,
    CoverLetterResponse,
    TailorResumeRequest,
    TailorResumeResponse,
)
from ..services.cover_letter_service import CoverLetterService
from ..services.tailoring_service import ResumeTailoringService


router = APIRouter(tags=["tailoring"])
tailoring_service = ResumeTailoringService()
cover_letter_service = CoverLetterService()


@router.post("/tailor-resume", response_model=TailorResumeResponse)
def tailor_resume(payload: TailorResumeRequest) -> TailorResumeResponse:
    return tailoring_service.tailor_resume(payload)


@router.post("/generate-cover-letter", response_model=CoverLetterResponse)
def generate_cover_letter(payload: CoverLetterRequest) -> CoverLetterResponse:
    return cover_letter_service.generate(payload)
