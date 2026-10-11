import pandas as pd

from src import data


def fake_df():
    d = ["250.1", "428", "V57", None]
    return pd.DataFrame({
        "encounter_id": [1, 2, 3, 4],
        "patient_nbr": [10, 11, 12, 13],
        "readmitted": ["NO", "<30", ">30", "NO"],
        "discharge_disposition_id": [1, 3, 11, 6],
        "weight": [None] * 4,
        "age": ["[70-80)", "[80-90)", "[60-70)", "[50-60)"],
        "time_in_hospital": [3, 5, 2, 4],
        "diag_1": d, "diag_2": d, "diag_3": d,
        "A1Cresult": [None] * 4,
        "max_glu_serum": [None] * 4,
        "payer_code": ["MC", None, "MD", "MC"],
        "race": ["Caucasian", None, "Asian", "Caucasian"],
        "medical_specialty": [None] * 4,
        "admission_type_id": [1, 2, 1, 3],
        "admission_source_id": [7, 1, 7, 4],
    })


def test_icd9_group():
    assert data.icd9_group("250.83") == "diabetes"
    assert data.icd9_group("428") == "circulatory"
    assert data.icd9_group("V57") == "other"
    assert data.icd9_group(None) == "missing"


def test_facility_target_and_leakage(monkeypatch):
    monkeypatch.setattr(data, "load_raw", fake_df)
    X, y, g = data.make_dataset("facility")
    assert len(X) == 3
    assert y.tolist() == [0, 1, 0]
    for leak in ["discharge_disposition_id", "readmitted", "time_in_hospital", "encounter_id", "patient_nbr", "age"]:
        assert leak not in X.columns
    assert X.age_mid.tolist() == [75, 85, 55]
    assert g.tolist() == [10, 11, 13]


def test_readmit_target(monkeypatch):
    monkeypatch.setattr(data, "load_raw", fake_df)
    X, y, _ = data.make_dataset("readmit")
    assert len(X) == 3
    assert y.tolist() == [0, 1, 0]
    assert "discharge_disposition_id" not in X.columns
