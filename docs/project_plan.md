# Project Plan

Status: local development plan | Last updated: 2026-10-17

## Delivery order

| Phase | Goal | Main deliverables | Completion check |
|---|---|---|---|
| P0 | Freeze honest scope | Clean design, model card, verified citations and limitations | Docs match repository reality |
| P1 | Complete ML reporting | Metadata, bootstrap intervals, sensitivity analysis, documented threshold | ML claims are evidence-backed |
| P2 | Create inference contract | `inference.py`, raw input schema, model-loader tests | API can call saved model locally |
| P3 | Build backend | FastAPI health, assess, predict, review; validation and tests | Local API test suite passes |
| P4 | Add safety/operations | API key, CORS, rate limit, minimal demo-only logging | No secrets/identifiers exposed/logged |
| P5 | Build frontend | Case form, criteria view, ML view, factors, disclaimer | Three demo cases work end-to-end |
| P6 | Integrated demo | Docker/local run instructions, integration tests, backup script | New teammate can run demo |

## Immediate priorities

1. Keep current ML/criteria source unchanged while documentation is cleaned.
2. Do not commit raw data, virtual environments, model artifacts, secrets, or accidental regenerated reports.
3. Define the inference input contract before a prediction endpoint or frontend form.
4. Keep criteria and ML outputs separate; use a review prompt instead of a merged readiness score.
5. Treat cloud deployment as optional after a working local demo.

## Suggested ownership

| Workstream | Responsibility |
|---|---|
| ML evidence and model card | ML owner |
| Criteria evidence and clinical wording | Criteria/data owner with supervisor input |
| API/inference contract | Backend owner |
| Dashboard and user flow | Frontend owner |
| Documentation, testing, release checklist | Shared team responsibility |

## Supervisor decisions needed

- Confirm that a diabetic-dataset facility-discharge proxy is acceptable for the academic prototype.
- Confirm that criteria are adapted decision support, not a validated universal protocol.
- Confirm that no real patient data enters the prototype without approval.
- Confirm that the treating clinician retains the final decision.
