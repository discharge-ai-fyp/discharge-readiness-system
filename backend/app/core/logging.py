"""Request identifiers and compact structured runtime logs."""
from __future__ import annotations

import logging
import time
import uuid

from fastapi import FastAPI, Request

LOGGER = logging.getLogger("discharge_api")


def configure_logging() -> None:
    if not LOGGER.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        LOGGER.addHandler(handler)
        LOGGER.setLevel(logging.INFO)
        LOGGER.propagate = False


def install_request_logging(app: FastAPI) -> None:
    @app.middleware("http")
    async def add_request_id_and_log(request: Request, call_next):
        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        started = time.perf_counter()
        response = await call_next(request)
        response.headers["x-request-id"] = request_id
        LOGGER.info(
            "request_complete request_id=%s method=%s path=%s status=%s latency_ms=%s",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            int((time.perf_counter() - started) * 1000),
        )
        return response
