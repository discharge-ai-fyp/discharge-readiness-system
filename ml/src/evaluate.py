import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

from src.data import make_dataset

SEED = 42
ROOT = Path(__file__).resolve().parents[1]


def subgroup_auc(Xte, yte, p, col, min_n=300):
    out = {}
    for val, idx in Xte.groupby(col, observed=True).indices.items():
        yy = yte.values[idx]
        if len(idx) >= min_n and 0 < yy.sum() < len(yy):
            out[str(val)] = {"n": int(len(idx)), "auc": round(float(roc_auc_score(yy, p[idx])), 3)}
    return out


def main(target, model_name):
    X, y, g = make_dataset(target)
    _, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED).split(X, y, groups=g))
    Xte, yte = X.iloc[te].copy(), y.iloc[te]
    pipe = joblib.load(ROOT / "models" / f"{target}_{model_name}.joblib")
    p = pipe.predict_proba(Xte)[:, 1]

    df = pd.DataFrame({"p": p, "y": yte.values})
    df["decile"] = pd.qcut(df.p, 10, labels=False, duplicates="drop")
    calib = df.groupby("decile").agg(mean_pred=("p", "mean"), observed=("y", "mean"), n=("y", "size")).round(3)

    order = np.argsort(-p)
    ops = []
    for frac in (0.1, 0.2, 0.3):
        k = int(len(p) * frac)
        tp = df.y.values[order[:k]].sum()
        ops.append({"flag_top": frac, "precision": round(tp / k, 3),
                    "recall": round(tp / df.y.sum(), 3), "lift": round((tp / k) / df.y.mean(), 2)})

    Xte["age_band"] = pd.cut(Xte.age_mid, [0, 40, 60, 80, 100]).astype(str)
    sub = {c: subgroup_auc(Xte, yte, p, c) for c in ["gender", "race", "age_band"]}

    print("\nCALIBRATION (mean_pred vs observed):\n", calib)
    print("\nOPERATING POINTS:", *ops, sep="\n")
    print("\nSUBGROUP AUC:", json.dumps(sub, indent=2))
    (ROOT / "reports" / f"{target}_{model_name}_eval.json").write_text(
        json.dumps({"calibration": calib.reset_index().to_dict("records"),
                    "operating_points": ops, "subgroup_auc": sub}, indent=2))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
