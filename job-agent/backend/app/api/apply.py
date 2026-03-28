from fastapi import APIRouter

from ..models.schemas import (
    ApplyPlanRequest,
    ApplyPlanResponse,
    RunApplicationRequest,
    RunApplicationResponse,
)
from ..services.apply_plan_service import ApplyPlanService
from ..services.orchestration_service import OrchestrationService


router = APIRouter(tags=["apply"])
plan_service = ApplyPlanService()
orchestration_service = OrchestrationService()


@router.post("/build-apply-plan", response_model=ApplyPlanResponse)
def build_apply_plan(payload: ApplyPlanRequest) -> ApplyPlanResponse:
    return plan_service.build(payload)


@router.post("/run-application", response_model=RunApplicationResponse)
def run_application(payload: RunApplicationRequest) -> RunApplicationResponse:
    return orchestration_service.run(payload)
