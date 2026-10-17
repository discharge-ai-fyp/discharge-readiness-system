"""Validated API access to the saved ML serving artifact."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from ml.src.inference import ArtifactContractError, Predictor

from app.core.errors import ApiError
from app.schemas.model import ModelFeatures, PredictionResponse

DISCLAIMER = (
    "Research-only facility-care discharge proxy. Decision support only; "
    "the treating clinician makes the final decision."
)


class ModelService:
    """Validate raw API input before calling the single serving implementation."""

    def __init__(
        self,
        model_path: str,
        feature_spec_path: str,
        metadata_path: str,
        *,
        predictor: Predictor | None = None,
    ):
        self._predictor = predictor or Predictor(model_path, feature_spec_path)
        self._spec = self._predictor.spec
        self._metadata = json.loads(Path(metadata_path).read_text())
        self._threshold = float(self._spec["high_risk_threshold"])
        self._version = self._metadata.get("git_revision", "unknown")

    @property
    def version(self) -> str:
        return self._version

    def _validate_values(self, features: dict[str, Any]) -> None:
        expected = set(self._spec["columns"])
        supplied = set(features)
        missing = sorted(expected - supplied)
        extra = sorted(supplied - expected)
        if missing or extra:
            parts = []
            if missing:
                parts.append(f"Missing model fields: {', '.join(missing)}")
            if extra:
                parts.append(f"Unexpected model fields: {', '.join(extra)}")
            raise ApiError(422, "VALIDATION_ERROR", "; ".join(parts), "model_features")

        for field in self._spec["numeric"]:
            value = features[field]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ApiError(422, "VALIDATION_ERROR", "Expected a number", f"model_features.{field}")
            if not math.isfinite(float(value)):
                raise ApiError(422, "VALIDATION_ERROR", "Expected a finite number", f"model_features.{field}")

        for field, allowed in self._spec["categorical"].items():
            value = features[field]
            if not isinstance(value, str) or value not in allowed:
                raise ApiError(
                    422,
                    "VALIDATION_ERROR",
                    "Value is not in the trained model's allowed categories",
                    f"model_features.{field}",
                )

    def predict(self, inputs: ModelFeatures) -> PredictionResponse:
        features = inputs.root
        self._validate_values(features)
        try:
            probability, factors = self._predictor.factor_dicts(features)
        except ArtifactContractError as error:
            raise ApiError(422, "VALIDATION_ERROR", str(error), "model_features") from error
        return PredictionResponse(
            probability=probability,
            is_high=probability >= self._threshold,
            threshold=self._threshold,
            threshold_policy=self._spec["threshold_policy"],
            model_version=self._version,
            top_factors=factors,
            disclaimer=DISCLAIMER,
        )
