"""Combined criteria and ML review contracts."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from app.domain.disagreement import FlagCode
from app.schemas.clinician import ClinicianInputs
from app.schemas.criteria import CriteriaResponse
from app.schemas.model import ModelFeatures, PredictionResponse


class CaseInput(BaseModel):
    """Both independent input groups required for a combined review."""

    model_config = ConfigDict(extra="forbid")

    clinician_inputs: ClinicianInputs
    model_features: ModelFeatures


class DisagreementFlag(BaseModel):
    code: FlagCode
    message: str | None


class ReviewResponse(BaseModel):
    criteria: CriteriaResponse
    ml: PredictionResponse
    flag: DisagreementFlag
    disclaimer: str
