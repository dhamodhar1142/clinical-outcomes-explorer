from fastapi import APIRouter

from ..models.schemas import JobParseRequest, JobParseResponse
from ..services.jd_parser import JDParserService


router = APIRouter(tags=["jobs"])
parser_service = JDParserService()


@router.post("/parse-job", response_model=JobParseResponse)
def parse_job(payload: JobParseRequest) -> JobParseResponse:
    return parser_service.parse(payload)
