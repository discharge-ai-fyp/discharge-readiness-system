import json
import sys
from pathlib import Path

import joblib
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from src.data import make_dataset

SEED = 42
ROOT = Path(__file__).resolve().parents[1]


def build_pipeline(X, model_name):
    num = X.select_dtypes(include="number").columns.tolist()
    cat = X.select_dtypes(exclude="number").columns.tolist()
    pre = ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=200), cat),
    ])
    if model_name == "logreg":
        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    else:
        clf = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                            subsample=0.8, colsample_bytree=0.8,
                            eval_metric="logloss", random_state=SEED, n_jobs=-1)
    return Pipeline([("pre", pre), ("clf", clf)])


def main(target, model_name):
    X, y, g = make_dataset(target)
    split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    tr, te = next(split.split(X, y, groups=g))
    Xtr, Xte, ytr, yte, gtr = X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te], g.iloc[tr]

    pipe = build_pipeline(Xtr, model_name)
    cv = cross_validate(pipe, Xtr, ytr, groups=gtr, cv=GroupKFold(5),
                        scoring={"auc": "roc_auc", "pr_auc": "average_precision"})
    pipe.fit(Xtr, ytr)
    p = pipe.predict_proba(Xte)[:, 1]

    metrics = {
        "target": target, "model": model_name,
        "n_train": len(Xtr), "n_test": len(Xte), "positive_rate": float(y.mean()),
        "cv_auc_mean": float(cv["test_auc"].mean()), "cv_auc_std": float(cv["test_auc"].std()),
        "cv_pr_auc_mean": float(cv["test_pr_auc"].mean()),
        "test_auc": float(roc_auc_score(yte, p)),
        "test_pr_auc": float(average_precision_score(yte, p)),
        "test_brier": float(brier_score_loss(yte, p)),
        "features": X.columns.tolist(),
    }
    (ROOT / "models").mkdir(exist_ok=True)
    (ROOT / "reports").mkdir(exist_ok=True)
    joblib.dump(pipe, ROOT / "models" / f"{target}_{model_name}.joblib")
    (ROOT / "reports" / f"{target}_{model_name}_metrics.json").write_text(json.dumps(metrics, indent=2))
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in metrics.items() if k != "features"})


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
