"""Adapter-facing service for the existing YAML criteria engine."""
from __future__ import annotations

from criteria.engine import assess, load_criteria

from app.schemas.clinician import ClinicianInputs
from app.schemas.criteria import CriteriaResponse


class CriteriaService:
    """Load validated criteria once and assess API input without changing engine logic."""

    def __init__(self, criteria_path: str):
        self._criteria = load_criteria(criteria_path)

    @property
    def version(self) -> str:
        return self._criteria["version"]

    def assess(self, inputs: ClinicianInputs) -> CriteriaResponse:
        result = assess(inputs.model_dump(), cfg=self._criteria)
        return CriteriaResponse.from_engine(result)
