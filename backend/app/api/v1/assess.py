"""Criteria assessment endpoint."""
from fastapi import APIRouter, Depends, Request

from app.core.security import require_api_key
from app.core.rate_limit import require_rate_limit
from app.schemas.clinician import ClinicianInputs
from app.schemas.criteria import CriteriaResponse
from app.services.criteria_service import CriteriaService

router = APIRouter(tags=["criteria"])


def get_criteria_service(request: Request) -> CriteriaService:
    return request.app.state.criteria


@router.post(
    "/assess",
    dependencies=[Depends(require_api_key), Depends(require_rate_limit)],
    response_model=CriteriaResponse,
)
def assess_case(
    inputs: ClinicianInputs,
    service: CriteriaService = Depends(get_criteria_service),
) -> CriteriaResponse:
    return service.assess(inputs)
