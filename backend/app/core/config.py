"""Environment-backed backend settings."""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    """Runtime settings. Secrets are supplied only through environment variables."""

    api_key: str
    criteria_path: str = "criteria/criteria.yaml"
    model_path: str = "ml/models/facility_xgb.joblib"
    feature_spec_path: str = "ml/models/facility_xgb.features.json"
    model_metadata_path: str = "ml/models/facility_xgb.metadata.json"
    cors_origins: tuple[str, ...] = ("http://localhost:3000",)
    rate_limit_per_minute: int = 60
    audit_db_path: str = "backend/audit.db"

    @classmethod
    def from_env(cls) -> "Settings":
        api_key = os.getenv("DISCHARGE_API_KEY")
        if not api_key:
            raise RuntimeError("DISCHARGE_API_KEY must be set before starting the API")
        return cls(
            api_key=api_key,
            criteria_path=os.getenv("DISCHARGE_CRITERIA_PATH", "criteria/criteria.yaml"),
            model_path=os.getenv("DISCHARGE_MODEL_PATH", "ml/models/facility_xgb.joblib"),
            feature_spec_path=os.getenv(
                "DISCHARGE_FEATURE_SPEC_PATH", "ml/models/facility_xgb.features.json"
            ),
            model_metadata_path=os.getenv(
                "DISCHARGE_MODEL_METADATA_PATH", "ml/models/facility_xgb.metadata.json"
            ),
            cors_origins=tuple(
                origin.strip()
                for origin in os.getenv(
                    "DISCHARGE_CORS_ORIGINS", "http://localhost:3000"
                ).split(",")
                if origin.strip()
            ),
            rate_limit_per_minute=int(os.getenv("DISCHARGE_RATE_LIMIT_PER_MINUTE", "60")),
            audit_db_path=os.getenv("DISCHARGE_AUDIT_DB_PATH", "backend/audit.db"),
        )
