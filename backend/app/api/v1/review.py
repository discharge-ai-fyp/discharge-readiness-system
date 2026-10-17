"""Combined checklist and ML review endpoint."""
from fastapi import APIRouter, Depends, Request

from app.core.security import require_api_key
from app.core.rate_limit import require_rate_limit
from app.schemas.review import CaseInput, ReviewResponse
from app.services.review_service import ReviewService

router = APIRouter(tags=["review"])


def get_review_service(request: Request) -> ReviewService:
    return request.app.state.review


@router.post(
    "/review",
    dependencies=[Depends(require_api_key), Depends(require_rate_limit)],
    response_model=ReviewResponse,
)
def review_case(
    case: CaseInput,
    service: ReviewService = Depends(get_review_service),
) -> ReviewResponse:
    return service.review(case)
