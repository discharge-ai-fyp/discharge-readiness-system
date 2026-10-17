def test_health_is_available_without_auth(client):
    response = client.get("/v1/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "criteria_version": "0.2",
        "model_version": "test-model-revision",
    }


def test_assess_requires_an_api_key(client):
    response = client.post("/v1/assess", json={})

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "UNAUTHORIZED"


def test_assess_returns_unknown_for_missing_inputs(client, headers):
    response = client.post("/v1/assess", json={}, headers=headers)

    assert response.status_code == 200
    body = response.json()
    assert body["summary"]["unknown"] == 13
    assert body["summary"]["pass"] == 0
    assert set(body["summary"]["priority_unknown"]) == {
        "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C9", "C10", "C11"
    }


def test_assess_validates_ranges_and_rejects_extra_fields(client, headers):
    bad_range = client.post("/v1/assess", json={"spo2": 120}, headers=headers)
    extra_field = client.post("/v1/assess", json={"not_a_field": True}, headers=headers)

    assert bad_range.status_code == 422
    assert bad_range.json()["error"]["code"] == "VALIDATION_ERROR"
    assert bad_range.json()["error"]["field"] == "spo2"
    assert extra_field.status_code == 422
    assert extra_field.json()["error"]["field"] == "not_a_field"


def test_assess_preserves_criteria_engine_result(client, headers):
    payload = {"temperature_c": 38.1, "heart_rate": 88, "spo2": 95}
    response = client.post("/v1/assess", json=payload, headers=headers)

    assert response.status_code == 200
    items = {item["id"]: item for item in response.json()["items"]}
    assert items["C1"]["status"] == "fail"
    assert items["C2"]["status"] == "pass"
    assert items["C5"]["status"] == "pass"


def test_predict_requires_an_api_key(client):
    response = client.post("/v1/predict", json={"age_mid": 75, "gender": "Female"})

    assert response.status_code == 401


def test_predict_returns_probability_and_factors(client, headers):
    response = client.post(
        "/v1/predict", json={"age_mid": 75, "gender": "Female"}, headers=headers
    )

    assert response.status_code == 200
    body = response.json()
    assert body["probability"] == 0.8
    assert body["is_high"] is True
    assert body["threshold"] == 0.5
    assert body["model_version"] == "test-model-revision"
    assert body["top_factors"][0]["feature"] == "age_mid"


def test_predict_rejects_bad_feature_contract(client, headers):
    missing = client.post("/v1/predict", json={"age_mid": 75}, headers=headers)
    category = client.post(
        "/v1/predict", json={"age_mid": 75, "gender": "Unknown"}, headers=headers
    )
    numeric = client.post(
        "/v1/predict", json={"age_mid": "old", "gender": "Female"}, headers=headers
    )

    assert missing.status_code == 422
    assert missing.json()["error"]["field"] == "model_features"
    assert category.status_code == 422
    assert category.json()["error"]["field"] == "model_features.gender"
    assert numeric.status_code == 422
    assert numeric.json()["error"]["field"] == "model_features.age_mid"


def test_review_combines_criteria_and_ml_without_a_readiness_score(client, headers):
    response = client.post(
        "/v1/review",
        json={
            "clinician_inputs": {
                "temperature_c": 37.0,
                "heart_rate": 80,
                "resp_rate": 18,
                "systolic_bp": 120,
                "spo2": 97,
                "oral_intake_ok": True,
                "mental_status_normal": True,
                "discharge_plan_documented": True,
                "med_reconciliation_done": True,
                "followup_scheduled": True,
            },
            "model_features": {"age_mid": 75, "gender": "Female"},
        },
        headers=headers,
    )

    assert response.status_code == 200
    body = response.json()
    assert body["criteria"]["summary"]["priority_not_met"] == []
    assert body["criteria"]["summary"]["priority_unknown"] == []
    assert body["ml"]["is_high"] is True
    assert body["flag"]["code"] == "ML_HIGH_CRITERIA_MET"
    assert "readiness" not in body
    assert client.app.state.audit_log.count() == 1


def test_review_treats_missing_priority_input_as_open(client, headers):
    response = client.post(
        "/v1/review",
        json={
            "clinician_inputs": {},
            "model_features": {"age_mid": 55, "gender": "Female"},
        },
        headers=headers,
    )

    assert response.status_code == 200
    assert response.json()["flag"]["code"] == "CRITERIA_OPEN_ML_LOW"


def test_cors_preflight_allows_only_configured_local_frontend(client):
    response = client.options(
        "/v1/assess",
        headers={
            "origin": "http://localhost:3000",
            "access-control-request-method": "POST",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:3000"


def test_rate_limit_blocks_the_third_protected_request(client, headers):
    from app.core.rate_limit import InMemoryRateLimiter

    client.app.state.rate_limiter = InMemoryRateLimiter(limit_per_minute=2)
    first = client.post("/v1/assess", json={}, headers=headers)
    second = client.post("/v1/assess", json={}, headers=headers)
    third = client.post("/v1/assess", json={}, headers=headers)

    assert first.status_code == 200
    assert second.status_code == 200
    assert third.status_code == 429
    assert third.json()["error"]["code"] == "RATE_LIMITED"
