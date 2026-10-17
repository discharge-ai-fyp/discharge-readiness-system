"""Small API-key protection for the local academic prototype."""
from __future__ import annotations

import secrets

from fastapi import Header, Request

from app.core.errors import ApiError


def require_api_key(request: Request, x_api_key: str | None = Header(default=None)) -> None:
    expected = request.app.state.settings.api_key
    if not x_api_key or not secrets.compare_digest(x_api_key, expected):
        raise ApiError(401, "UNAUTHORIZED", "Missing or invalid API key")
