"""Unauthenticated health endpoint for local runtime checks."""
from fastapi import APIRouter, Request

router = APIRouter(tags=["health"])


@router.get("/health")
def health(request: Request) -> dict[str, str]:
    return {
        "status": "ok",
        "criteria_version": request.app.state.criteria.version,
        "model_version": request.app.state.model.version,
    }
