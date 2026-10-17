import json

import numpy as np
import pandas as pd
import pytest

from src.inference import (
    ArtifactContractError,
    Predictor,
    build_feature_spec,
    feature_spec_path,
    json_value,
    metadata_path,
    source_column,
    write_artifacts,
)


def test_feature_spec_is_derived_from_training_columns():
    features = pd.DataFrame({"age_mid": [55, 65], "gender": ["Female", "Male"]})
    spec = build_feature_spec(
        features, np.array([0.1, 0.9]), target="facility", model_name="xgb"
    )

    assert spec["columns"] == ["age_mid", "gender"]
    assert spec["numeric"] == ["age_mid"]
    assert spec["categorical"]["gender"] == ["Female", "Male"]
    assert spec["high_risk_threshold"] == pytest.approx(0.74)


def test_artifact_paths_and_json_are_written(tmp_path):
    model = tmp_path / "facility_xgb.joblib"
    model.touch()
    features = pd.DataFrame({"age_mid": [55], "gender": ["Female"]})
    spec_file, meta_file = write_artifacts(
        model,
        features,
        np.array([0.4]),
        target="facility",
        model_name="xgb",
        metrics={"test_auc": 0.8},
        raw_data_path=None,
    )

    assert spec_file == feature_spec_path(model)
    assert meta_file == metadata_path(model)
    assert json.loads(spec_file.read_text())["columns"] == ["age_mid", "gender"]
    assert json.loads(meta_file.read_text())["metrics"]["test_auc"] == 0.8


@pytest.mark.parametrize(
    ("encoded", "expected"),
    [
        ("num__age_mid", "age_mid"),
        ("cat__payer_code_MC", "payer_code"),
        ("cat__medical_specialty_Emergency/Trauma", "medical_specialty"),
    ],
)
def test_encoded_feature_maps_to_original_column(encoded, expected):
    columns = ["age_mid", "payer_code", "medical_specialty"]
    assert source_column(encoded, columns) == expected


def test_json_value_converts_numpy_scalars():
    assert json_value(np.int64(75)) == 75
    assert json_value("Female") == "Female"


def test_predictor_rejects_missing_or_extra_features_without_loading_model():
    predictor = Predictor.__new__(Predictor)
    predictor.columns = ["age_mid", "gender"]

    with pytest.raises(ArtifactContractError, match="missing fields"):
        predictor._frame({"age_mid": 75})
    with pytest.raises(ArtifactContractError, match="unexpected fields"):
        predictor._frame({"age_mid": 75, "gender": "Female", "extra": True})
