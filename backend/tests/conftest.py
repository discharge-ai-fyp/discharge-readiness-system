import json
from pathlib import Path
import sys

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT))

from app.core.config import Settings  # noqa: E402
from app.main import create_app  # noqa: E402
from app.services.model_service import ModelService  # noqa: E402


class StubPredictor:
    """Small deterministic serving double; real Predictor is covered in ml tests."""

    def __init__(self):
        self.spec = {
            "columns": ["age_mid", "gender"],
            "numeric": ["age_mid"],
            "categorical": {"gender": ["Female", "Male"]},
            "high_risk_threshold": 0.5,
            "threshold_policy": "test demo threshold",
        }

    def factor_dicts(self, features):
        probability = 0.8 if features["age_mid"] >= 70 else 0.2
        return probability, [
            {"feature": "age_mid", "value": features["age_mid"], "impact": 0.4},
            {"feature": "gender", "value": features["gender"], "impact": 0.1},
        ]


@pytest.fixture()
def client(tmp_path):
    metadata_path = tmp_path / "model.metadata.json"
    metadata_path.write_text(json.dumps({"git_revision": "test-model-revision"}))
    model = ModelService(
        "not-used.joblib",
        "not-used.features.json",
        str(metadata_path),
        predictor=StubPredictor(),
    )
    app = create_app(
        Settings(
            api_key="test-key",
            criteria_path=str(ROOT / "criteria/criteria.yaml"),
            audit_db_path=str(tmp_path / "audit.db"),
            rate_limit_per_minute=60,
        ),
        model_service=model,
    )
    return TestClient(app)


@pytest.fixture()
def headers() -> dict[str, str]:
    return {"x-api-key": "test-key"}
