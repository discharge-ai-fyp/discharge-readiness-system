"""ML facility-care proxy prediction endpoint."""
from fastapi import APIRouter, Depends, Request

from app.core.security import require_api_key
from app.core.rate_limit import require_rate_limit
from app.schemas.model import ModelFeatures, PredictionResponse
from app.services.model_service import ModelService

router = APIRouter(tags=["model"])


def get_model_service(request: Request) -> ModelService:
    return request.app.state.model


@router.post(
    "/predict",
    dependencies=[Depends(require_api_key), Depends(require_rate_limit)],
    response_model=PredictionResponse,
)
def predict_case(
    features: ModelFeatures,
    service: ModelService = Depends(get_model_service),
) -> PredictionResponse:
    return service.predict(features)
