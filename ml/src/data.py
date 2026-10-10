from pathlib import Path
import pandas as pd

RAW = Path(__file__).resolve().parents[1] / "data" / "raw" / "uci" / "diabetic_data.csv"

HOME = [1, 6, 8]
FACILITY = [2, 3, 4, 5, 22, 23, 24, 30]
EXPIRED_HOSPICE = [11, 13, 14, 19, 20, 21]

AGE_MID = {f"[{a}-{a + 10})": a + 5 for a in range(0, 100, 10)}


def icd9_group(code):
    if pd.isna(code):
        return "missing"
    c = str(code)
    if c[0] in "VE":
        return "other"
    try:
        x = float(c)
    except ValueError:
        return "other"
    i = int(x)
    if 390 <= x < 460 or i == 785: return "circulatory"
    if 460 <= x < 520 or i == 786: return "respiratory"
    if 520 <= x < 580 or i == 787: return "digestive"
    if i == 250: return "diabetes"
    if 800 <= x < 1000: return "injury"
    if 710 <= x < 740: return "musculoskeletal"
    if 580 <= x < 630 or i == 788: return "genitourinary"
    if 140 <= x < 240: return "neoplasms"
    return "other"


def load_raw(path=RAW):
    return pd.read_csv(path, na_values="?", low_memory=False)


def make_dataset(target):
    df = load_raw()
    drop = ["encounter_id", "patient_nbr", "readmitted",
            "discharge_disposition_id", "weight", "age"]
    if target == "facility":
        df = df[df.discharge_disposition_id.isin(HOME + FACILITY)].copy()
        y = df.discharge_disposition_id.isin(FACILITY).astype(int)
        drop.append("time_in_hospital")
    elif target == "readmit":
        df = df[~df.discharge_disposition_id.isin(EXPIRED_HOSPICE)].copy()
        y = (df.readmitted == "<30").astype(int)
    else:
        raise ValueError("target must be 'facility' or 'readmit'")

    groups = df.patient_nbr
    df["age_mid"] = df["age"].map(AGE_MID)
    for c in ["diag_1", "diag_2", "diag_3"]:
        df[c] = df[c].map(icd9_group)
    for c in ["A1Cresult", "max_glu_serum", "payer_code", "race", "medical_specialty"]:
        df[c] = df[c].fillna("Not_measured")
    for c in ["admission_type_id", "admission_source_id"]:
        df[c] = df[c].astype(str)
    X = df.drop(columns=drop)
    return X, y, groups
