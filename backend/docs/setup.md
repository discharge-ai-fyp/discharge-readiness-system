# Backend Setup and Local Run Guide

## What this backend does

The backend exposes four currently available functions:

- `GET /v1/health` checks that the API and criteria configuration are loaded.
- `POST /v1/assess` runs the 13-item criteria checklist.
- `POST /v1/predict` returns the saved ML model's facility-care proxy probability and SHAP factors.
- `POST /v1/review` runs criteria and ML together, then returns a review prompt.

The prediction and review endpoints require locally trained model artifacts.

## Prerequisites

From the repository root, create/activate the documented Python environment and install dependencies:

```bash
source ml/venv312/bin/activate
uv pip install -r backend/requirements-dev.txt
```

The API imports both `backend/app` and the existing root-level `criteria` package. Start it from the repository root.

## Command map: what each command does and why

| Command | What it does | Why it is needed |
|---|---|---|
| `source ml/venv312/bin/activate` | Selects this project's Python environment | Ensures FastAPI, XGBoost, SHAP, and project dependencies use expected versions |
| `uv pip install -r backend/requirements-dev.txt` | Installs backend runtime, test, and lint packages | A new teammate needs FastAPI/Uvicorn/pytest before running the backend |
| `export DISCHARGE_API_KEY='...'` | Stores a temporary API secret in this terminal | The backend rejects unauthenticated calls; secrets do not go into source code |
| `PYTHONPATH=backend:.` | Adds `backend/` and repo root to Python import search path | Lets the API import both `app` and existing `criteria`/`ml` packages |
| `uvicorn ... --factory` | Calls `create_app()` to build/start FastAPI | The app factory checks configuration and loads criteria/model services once |
| `--reload` | Restarts local server after Python file changes | Development convenience only; do not use it for production |
| `python -m src.train facility xgb` | Trains the facility model and writes local artifacts | `/predict` needs the model, its exact feature contract, and metadata |
| `docker compose up --build` | Builds a reproducible API container and starts it | Lets every teammate run the same backend runtime without local Uvicorn setup |
| `docker compose down` | Stops the container while preserving its named audit-data volume | Use after a local Docker demo is complete |

## Configure a local API key

The API requires `DISCHARGE_API_KEY`. This is a local demo secret, not a GitHub password or a production credential.

```bash
export DISCHARGE_API_KEY='choose-a-local-demo-key'
```

The variable exists only in the current terminal. Do not commit it. A sample is in `backend/.env.example`; `.env` files remain ignored.

## Runtime safety settings

These values have local-demo defaults. Put any override in `backend/.env`, never in source code.

| Variable | Default | What it controls | Why it exists |
|---|---:|---|---|
| `DISCHARGE_CORS_ORIGINS` | `http://localhost:3000` | Which browser frontend origin may call the API | A browser blocks cross-origin calls unless the API explicitly allows the frontend origin |
| `DISCHARGE_RATE_LIMIT_PER_MINUTE` | `60` | Protected requests allowed per client IP per minute | Limits accidental loops and simple abuse during a demo |
| `DISCHARGE_AUDIT_DB_PATH` | `backend/audit.db` | Local SQLite audit-event database location | Keeps privacy-safe operational evidence separate from application code |

The audit database for `/v1/review` records only timestamp, endpoint, response latency, criteria/model versions, review flag, and probability. It never stores raw patient-like fields, API keys, or case IDs. The current rate limiter is in-memory, so it is for one local process only—not a multi-server production deployment.

## Start the server

```bash
PYTHONPATH=backend:. uvicorn app.main:create_app --factory --app-dir backend --reload
```

Expected output includes a URL such as `http://127.0.0.1:8000`.

Open interactive API documentation in a browser:

```text
http://127.0.0.1:8000/docs
```

`/docs` is Swagger UI: it lets teammates inspect endpoint request/response shapes. The actual model feature list is generated locally in `facility_xgb.features.json`, so `/predict` accepts a JSON object and validates it at runtime against that artifact.

Every response also has an `X-Request-ID` header. If a teammate reports an error, share that value and the approximate time; it lets you correlate the safe request-completion log without exposing request data.

## Test `GET /v1/health`

In a second terminal, activate the same environment and set the same key:

```bash
source ml/venv312/bin/activate
export DISCHARGE_API_KEY='choose-a-local-demo-key'
```

This checks that FastAPI started and loaded criteria/model services. It requires no API key:

```bash
curl http://127.0.0.1:8000/v1/health
```

Expected fields:

```json
{
  "status": "ok",
  "criteria_version": "0.2",
  "model_version": "..."
}
```

## Test `POST /v1/assess`

This endpoint runs the rule-based checklist only; it does not call ML. It is the easiest endpoint to test manually because all clinician fields are optional.

Criteria assessment requires the `x-api-key` header:

```bash
curl -X POST http://127.0.0.1:8000/v1/assess \
  -H 'content-type: application/json' \
  -H "x-api-key: $DISCHARGE_API_KEY" \
  -d '{"temperature_c":38.1,"heart_rate":88,"spo2":95}'
```

Expected behaviour:

- C1 is `fail` because temperature is above 37.8 C.
- C2 and C5 are `pass`.
- Fields not sent are `unknown`.

## Prepare model artifacts before `POST /v1/predict` or `POST /v1/review`

Run this from the repository root after UCI data setup:

```bash
cd ml
python -m src.train facility xgb
cd ..
```

What it produces and why:

| Local ignored artifact | Why the API needs it |
|---|---|
| `facility_xgb.joblib` | The saved preprocessing pipeline and XGBoost classifier |
| `facility_xgb.features.json` | Exact 44 raw fields, numeric fields, categories, and demo threshold |
| `facility_xgb.metadata.json` | Model version, library versions, seed, metric snapshot, data hash |

Do not commit these artifacts. Every developer creates them locally from the documented data and Python setup.

## Test `POST /v1/predict`

`/predict` needs all fields required by the trained model. Do not invent a partial payload manually. The command below creates a temporary valid payload from the first local UCI feature row:

```bash
PYTHONPATH=backend:. python - <<'PY' > /private/tmp/discharge-model-features.json
import json
from ml.src.data import make_dataset

features, _, _ = make_dataset("facility")
print(json.dumps(features.iloc[0].to_dict()))
PY
```

Then call the endpoint:

```bash
curl -X POST http://127.0.0.1:8000/v1/predict \
  -H 'content-type: application/json' \
  -H "x-api-key: $DISCHARGE_API_KEY" \
  --data @/private/tmp/discharge-model-features.json
```

The response contains the probability, whether it is above the **research/demo** threshold, top SHAP factors, and a disclaimer. It is not a discharge decision.

## Test `POST /v1/review`

`/review` takes two independent groups: `clinician_inputs` for the checklist and `model_features` for the ML model. This command builds a temporary combined demo request from the same local feature row:

```bash
PYTHONPATH=backend:. python - <<'PY' > /private/tmp/discharge-review.json
import json
from ml.src.data import make_dataset

features, _, _ = make_dataset("facility")
case = {
    "model_features": features.iloc[0].to_dict(),
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
        "followup_scheduled": True
    }
}
print(json.dumps(case))
PY
```

Send it:

```bash
curl -X POST http://127.0.0.1:8000/v1/review \
  -H 'content-type: application/json' \
  -H "x-api-key: $DISCHARGE_API_KEY" \
  --data @/private/tmp/discharge-review.json
```

`flag.code` has one of four meanings:

| Code | Meaning |
|---|---|
| `NONE` | Priority criteria are met and ML estimate is lower |
| `ML_HIGH_CRITERIA_MET` | Checklist is complete but ML estimate is higher; inspect factors/context |
| `CRITERIA_OPEN_ML_LOW` | ML estimate is lower but priority checklist fields are failed/unknown |
| `BOTH_REVIEW` | Both views need closer review |

The flag is a review prompt, never an automated discharge decision.

## Run through Docker Compose

Use Docker when you want a repeatable backend runtime. The ML artifacts are still deliberately local: Docker mounts `ml/models/` read-only, so first complete the training step above.

1. Start Docker Desktop and wait until its engine is running.
2. Create your untracked local environment file and choose a non-default demo key:

```bash
cp backend/.env.example backend/.env
```

3. Open `backend/.env` and set `DISCHARGE_API_KEY=your-local-demo-key`.
4. Build and start the API:

```bash
docker compose up --build
```

5. In another terminal, test it with the same `curl` commands above. The URL remains `http://127.0.0.1:8000`.
6. Stop the demo when finished:

```bash
docker compose down
```

`docker compose down` preserves the named audit-data volume. To intentionally discard local Docker audit events as well, use `docker compose down --volumes`; this is destructive to that local audit data only.

## What GitHub Actions checks

The workflow at `.github/workflows/backend-ci.yml` runs on pull requests and pushes to `dev` or `main` when backend, criteria, or ML-serving files change. It:

1. installs backend test dependencies;
2. runs focused lint checks for import/syntax errors;
3. runs backend API and criteria tests; and
4. builds the Docker image to catch container packaging errors.

It does not upload the local model or UCI dataset. Those remain ignored because they are generated/local research assets.

## Run checks

From the repository root:

```bash
ruff check backend/app backend/tests ml/src ml/tests criteria --select F,E9
PYTHONPATH=backend:. python -m pytest backend/tests -q
python -m pytest criteria -q
cd ml && python -m pytest -q
```

## Troubleshooting

| Problem | Fix |
|---|---|
| `DISCHARGE_API_KEY must be set` | Run `export DISCHARGE_API_KEY='...'` in the same terminal before starting Uvicorn. |
| `No module named app` | Start from repository root with `PYTHONPATH=backend:.` and `--app-dir backend`. |
| `401 Unauthorized` | Send the same key through the `x-api-key` request header. |
| Model artifact missing | Run `cd ml && python -m src.train facility xgb`. |
| `422 VALIDATION_ERROR` from `/predict` | Use all fields from the generated feature spec; values must match numeric/category contract. |
| XGBoost `libomp` error on macOS | Install `libomp` using Homebrew, then retry. |
| Docker cannot connect to `docker.sock` | Start Docker Desktop, wait for it to say the engine is running, then rerun `docker compose up --build`. |
