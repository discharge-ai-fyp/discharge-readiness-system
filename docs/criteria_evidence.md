# Criteria evidence (v0.2, draft)

Decision-support only. Not a validated clinical tool. The treating physician decides.

| ID | Criterion | Threshold / definition | Source | Input | Severity |
|---|---|---|---|---|---|
| C1 | Temperature | <= 37.8 C | Halm 2002, Methods | clinician | priority |
| C2 | Heart rate | <= 100/min | Halm 2002, Methods | clinician | priority |
| C3 | Respiratory rate | <= 24/min | Halm 2002, Methods | clinician | priority |
| C4 | Systolic BP | >= 90 mmHg | Halm 2002, Methods | clinician | priority |
| C5 | Oxygen saturation | SpO2 >= 90% (or PaO2 >= 60). Exceptions in paper: nasal-prong O2 needs >= 95%; baseline home-oxygen users not counted unstable. Engine checks the 90% cutoff only | Halm 2002, Methods | clinician | priority |
| C6 | Oral intake | able to eat, or long-term tube feeding resumed | Halm 2002, Methods | clinician | priority |
| C7 | Mental status | normal, or back to baseline for chronic dementia | Halm 2002, Methods | clinician | priority |
| C8 | A1C result available | A1C at admission if none in prior 3 months | ADA 2026 Sec.16 Rec.16.1 (B); 2025 Rec.16.1 | data (UCI A1Cresult) | advisory |
| C9 | Structured discharge plan | documented, tailored to the individual | ADA 2026 Rec.16.18 (B); 2025 Rec.16.16 | clinician | priority |
| C10 | Medication reconciliation | home and hospital lists cross-checked | ADA 2026 Sec.16 Transition (AHRQ minimum plan); RED Tool 3 Table 1, component 5 | clinician | priority |
| C11 | Follow-up appointment scheduled before discharge | scheduled (ADA: within 1 month after in-hospital hyper/hypoglycemia; 1-2 weeks if glycemic meds changed or control not optimal) | ADA 2026 Sec.16 Transition; RED Tool 3 Table 1, component 2 | clinician | priority |
| C12 | Plan for results pending at discharge | who reviews, when, how patient is told | ADA 2026 Sec.16 Transition; RED Tool 3 Table 1, component 3 | clinician | advisory |
| C13 | Post-discharge services and equipment arranged | arranged where needed | RED Tool 3 Table 1, component 4 | clinician | advisory |

## Notes
- C1-C7: Halm assessed these in the 24 h before discharge (most abnormal value of the day), in a
  community-acquired pneumonia cohort (680 patients, 1991-1994, low-risk patients oversampled).
  Adapted here, not validated for diabetic patients.
- Halm's own reading: 1 instability -> close follow-up and education; 2 or more -> usually stay,
  absent extenuating circumstances. Clinician decides. Sensitivity of 1+ instability was 27.5%,
  so no instability does not mean safe.
- Missing data is always `unknown`, never `pass`.
- ADA glucose targets (Rec. 16.4/16.5) are treatment targets, not discharge criteria. Not used.
- ADA 2026 Rec.16.18 (grade E part): if not discharged home, consider the facility's capability
  to manage diabetes. Candidate for a conditional criterion in a later version.
- No single "ready" score is produced.

## References
- Halm EA et al. Arch Intern Med 2002;162(11):1278-84. doi:10.1001/archinte.162.11.1278
- ADA Professional Practice Committee. Standards of Care in Diabetes 2026, Sec. 16. Diabetes Care 2026;49(Suppl 1):S339-S355. doi:10.2337/dc26-S016
- ADA Standards of Care in Diabetes 2025, Sec. 16. Diabetes Care 2025;48(Suppl 1):S321-S334. doi:10.2337/dc25-S016
- AHRQ Re-Engineered Discharge (RED) Toolkit, Tool 3, https://www.ahrq.gov/patient-safety/settings/hospital/red/toolkit/redtool3.html (reviewed March 2025)
