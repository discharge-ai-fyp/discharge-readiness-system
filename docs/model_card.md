# Model Card — Facility-Care Discharge Proxy

Status: research prototype | Last updated: 2026-10-17

## Model summary

| Field | Description |
|---|---|
| Primary model | XGBoost classifier inside a scikit-learn preprocessing pipeline |
| Baseline | Logistic Regression inside the same preprocessing pattern |
| Output | Probability of historical facility-care discharge versus home discharge |
| Intended use | Research/demo support for a structured discharge-planning review |
| Not intended for | Clinical discharge decisions, triage, diagnosis, treatment, billing, or unvalidated real-patient use |

## Training data and target

The model uses the UCI Diabetes 130-US Hospitals dataset: diabetic inpatient encounters from 130 US hospitals from 1999–2008. It is neither a local Pakistani clinical dataset nor a generic all-patient dataset.

The primary target groups selected UCI discharge-disposition IDs into home versus facility. This is a project-defined proxy outcome, not a true label of discharge readiness or clinical care need.

## Inputs

Inputs include demographics, admission information, payer, medical specialty, encounter-level counts, broad diagnosis groups, diabetes-related medication fields, and selected measurement categories. Identifiers and explicit outcome fields are excluded. The facility model excludes length of stay because it can be influenced by disposition.

## Evaluation snapshot

| Model | AUC | PR-AUC | Brier |
|---|---:|---:|---:|
| Facility XGBoost | 0.799 | 0.547 | 0.132 |
| Facility Logistic Regression | 0.792 | 0.533 | 0.186 |
| Readmission XGBoost, exploratory | 0.648 | 0.207 | 0.096 |

Metrics are from one patient-level hold-out split with seed 42. They are not clinical-validation results.

## Explanation

TreeSHAP provides global feature importance and local factors for individual predictions. The implementation checks that SHAP reconstructed probabilities agree with pipeline predictions. SHAP values describe model behaviour; they do not establish medical causation.

## Limitations and risks

- Diabetic US inpatient encounters only, from 1999–2008.
- Discharge destination may reflect payer, social support, hospital policy, and bed availability as well as patient condition.
- No readiness label, detailed time-stamped clinical state, or numeric vital-sign series.
- Encounter counts may not be available at the same decision point in a live setting.
- Age and payer influence predictions, creating fairness and transportability concerns.
- No external, local, prospective, clinician, or real-world validation.

## Required before integrated demo claims

- Save versioned metadata next to trained models.
- Define/test raw API input schema from the trained pipeline.
- Add bootstrap confidence intervals and target sensitivity analysis.
- Use an explicit research/demo threshold for any high/low category.
- Obtain governance and validation before real-patient use.
