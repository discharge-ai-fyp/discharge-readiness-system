"""Generic raw-model input and prediction response contracts."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, RootModel


class ModelFeatures(RootModel[dict[str, Any]]):
    """Raw fields are validated against the locally trained feature spec."""


class FactorResponse(BaseModel):
    feature: str
    value: Any
    impact: float


class PredictionResponse(BaseModel):
    probability: float
    is_high: bool
    threshold: float
    threshold_policy: str
    model_version: str
    top_factors: list[FactorResponse]
    disclaimer: str
