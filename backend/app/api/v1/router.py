"""Compose versioned route modules."""
from fastapi import APIRouter

from app.api.v1 import assess, health, predict, review

api_router = APIRouter(prefix="/v1")
api_router.include_router(health.router)
api_router.include_router(assess.router)
api_router.include_router(predict.router)
api_router.include_router(review.router)
