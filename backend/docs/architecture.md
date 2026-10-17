# Backend Architecture and Development Guide

## Architecture choice

The backend is a **modular monolith**: one FastAPI application with separate internal layers. This keeps the FYP prototype easy to run and test without premature microservices.

```text
HTTP client / future frontend
        |
        v
API routes                 HTTP, authentication, status codes
        |
        v
Services                   criteria/model orchestration
        |------------------|
        v                  v
Criteria engine      ML Predictor
criteria.engine      ml.src.inference
        |
        v
Domain rules and future audit adapter
```

## Folder structure

```text
backend/
├── app/
│   ├── main.py             # app factory and dependency startup
│   ├── core/               # config, errors, security
│   ├── api/v1/             # HTTP routes only
│   ├── schemas/            # request and response contracts
│   ├── services/           # application use-cases
│   ├── domain/             # pure business rules; added for review logic
│   └── adapters/           # storage/external boundaries; added when needed
├── docs/                   # team setup and architecture guides
├── tests/                  # API and contract tests
├── requirements.txt        # runtime packages
└── requirements-dev.txt    # test/lint packages
```

## Layer rules

| Layer | Owns | Must not own |
|---|---|---|
| `api` | HTTP routing, dependencies, status codes | clinical or ML decisions |
| `schemas` | Pydantic validation and public API shapes | model/file/database I/O |
| `services` | orchestration of criteria and ML calls | FastAPI-specific response handling |
| `domain` | deterministic review/disagreement rules | I/O or framework code |
| `adapters` | audit database and external integration wrappers | policy decisions |
| `core` | environment configuration, security, common errors | criteria/ML business logic |

## Milestones

| Milestone | Deliverable | Status |
|---|---|---|
| B0 | One ML inference implementation, feature spec, model metadata | Complete |
| B1 | App factory, health endpoint, criteria endpoint | Complete |
| B2 | Prediction service, `/v1/predict`, model contract tests | Complete |
| B3 | Combined `/v1/review` and disagreement rule | Complete |
| B4 | Audit log, CORS, rate limiting, Docker, CI | Implemented; Docker engine verification pending locally |

## Serving contract

Training writes three local artifacts beside the model:

```text
facility_xgb.joblib          # saved preprocessing pipeline + XGBoost classifier
facility_xgb.features.json   # exact raw feature fields, types/categories, demo threshold
facility_xgb.metadata.json   # training versions, metrics, seed and data hash
```

`ml/src/inference.py` is the only implementation that performs serving prediction and SHAP explanation. It transforms an input once, then uses the same sparse matrix for XGBoost and SHAP. If reconstructed SHAP probability differs from the model prediction, the request fails instead of returning misleading factors.

`/v1/predict` accepts a JSON object containing the exact fields named in `facility_xgb.features.json`. It rejects missing fields, unexpected fields, invalid numeric values, and unseen categorical values with a 422 response.

The model is a diabetic-dataset facility-discharge proxy. It is not an all-patient discharge-readiness predictor.

## API endpoints

| Endpoint | Authentication | Role |
|---|---|---|
| `GET /v1/health` | No | Service/criteria/model readiness status |
| `POST /v1/assess` | API key | Criteria checklist only |
| `POST /v1/predict` | API key | ML probability, demo threshold, SHAP factors |
| `POST /v1/review` | API key | Criteria, ML result, and review prompt |

## Non-negotiable safety rules

- Missing checklist data is always `unknown`, never `pass`.
- Model input is checked against the training artifact contract before prediction.
- The high/low category is a research/demo operating point, not a clinical discharge threshold.
- Never log API keys, raw patient-like input, free-text case IDs, or secrets.
- Configuration comes from environment variables; `.env` is ignored.
- Do not replace the separate criteria and ML views with a merged readiness score.

## Test strategy

| Test type | Covers |
|---|---|
| Criteria tests | YAML rules, threshold boundaries, unknown handling |
| Inference tests | feature spec, artifact metadata, field contract, source-factor mapping |
| API tests | auth, validation, endpoint response shapes |
| Model service tests | prediction response and API/offline contract equivalence |
| Integration tests | future clean, mixed, and concerning review cases |

## Operational protections (B4)

- CORS permits only configured browser origins; its local default is `http://localhost:3000`.
- Protected endpoints use an API key plus a 60-requests-per-minute, per-IP in-memory limiter by default.
- Every response gets an `X-Request-ID`; request-completion logs contain method, path, status, latency, and that ID only.
- `/v1/review` writes a privacy-safe SQLite audit event: time, endpoint, latency, artifact versions, flag, and probability. It does not write raw inputs, keys, or case IDs.
- The Docker image runs as an unprivileged user, while Compose mounts local model artifacts read-only.
- GitHub Actions runs lint/tests and confirms the Docker image builds. It does not use the ignored local model or UCI data.

The in-memory limiter and SQLite audit log are suitable for the single-process FYP demo. A production multi-instance deployment needs a shared rate limiter/API gateway, managed database, monitoring, secret manager, formal authentication, and clinical governance. `/v1/review` continues to present separate criteria and ML views rather than a merged readiness score.
