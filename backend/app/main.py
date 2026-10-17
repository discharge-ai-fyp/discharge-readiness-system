"""FastAPI application factory.

Start locally with:
DISCHARGE_API_KEY=dev-key uvicorn app.main:create_app --factory --app-dir backend
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.adapters.audit_log import SQLiteAuditLog
from app.api.v1.router import api_router
from app.core.config import Settings
from app.core.errors import install_error_handlers
from app.core.logging import configure_logging, install_request_logging
from app.core.rate_limit import InMemoryRateLimiter
from app.services.criteria_service import CriteriaService
from app.services.model_service import ModelService
from app.services.review_service import ReviewService


def create_app(
    settings: Settings | None = None,
    *,
    model_service: ModelService | None = None,
) -> FastAPI:
    """Create an app with dependencies initialized once at startup."""
    settings = settings or Settings.from_env()
    app = FastAPI(
        title="Discharge Readiness Support API",
        version="0.1.0",
        description="Decision support prototype. The treating clinician makes the final decision.",
    )
    app.state.settings = settings
    app.state.criteria = CriteriaService(settings.criteria_path)
    app.state.model = model_service or ModelService(
        settings.model_path, settings.feature_spec_path, settings.model_metadata_path
    )
    app.state.audit_log = SQLiteAuditLog(settings.audit_db_path)
    app.state.review = ReviewService(app.state.criteria, app.state.model, app.state.audit_log)
    app.state.rate_limiter = InMemoryRateLimiter(settings.rate_limit_per_minute)
    configure_logging()
    install_request_logging(app)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["content-type", "x-api-key", "x-request-id"],
    )
    install_error_handlers(app)
    app.include_router(api_router)
    return app
