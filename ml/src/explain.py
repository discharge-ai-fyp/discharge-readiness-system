import json
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy import sparse
from sklearn.model_selection import GroupShuffleSplit

from src.data import make_dataset

SEED = 42
ROOT = Path(__file__).resolve().parents[1]


def transform(pipe, X):
    """Run the fitted preprocessor and keep the SAME (sparse) matrix the model saw."""
    Z = pipe.named_steps["pre"].transform(X)
    if not sparse.issparse(Z):
        Z = sparse.csr_matrix(Z)
    names = [n.split("__", 1)[1] for n in pipe.named_steps["pre"].get_feature_names_out()]
    return Z.tocsr(), names


def top_factors(sv_row, names, k=5):
    idx = np.argsort(-np.abs(sv_row))[:k]
    return [{"feature": names[i], "shap": round(float(sv_row[i]), 3)} for i in idx]


def check_consistency(pipe, explainer, Z, sv, Xs, tol=1e-3):
    """SHAP must reproduce the pipeline's own probabilities, otherwise explanations are wrong."""
    base = float(np.ravel(explainer.expected_value)[0])
    recon_p = 1.0 / (1.0 + np.exp(-(sv.sum(axis=1) + base)))
    pipe_p = pipe.predict_proba(Xs)[:, 1]
    diff = float(np.abs(recon_p - pipe_p).max())
    print(f"\nCONSISTENCY: max |recon_p - pipeline_p| = {diff:.2e} "
          f"| mean pipeline p = {pipe_p.mean():.3f} | mean recon p = {recon_p.mean():.3f}")
    if diff > tol:
        raise RuntimeError("SHAP is not consistent with the pipeline output. Do not use these explanations.")


def main(target, model_name="xgb", n=2000):
    X, y, g = make_dataset(target)
    _, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED).split(X, y, groups=g))
    Xs = X.iloc[te].sample(n, random_state=SEED)

    pipe = joblib.load(ROOT / "models" / f"{target}_{model_name}.joblib")
    Z, names = transform(pipe, Xs)
    explainer = shap.TreeExplainer(pipe.named_steps["clf"])
    sv = explainer.shap_values(Z)
    check_consistency(pipe, explainer, Z, sv, Xs)
    p = pipe.predict_proba(Xs)[:, 1]

    imp = pd.Series(np.abs(sv).mean(axis=0), index=names).sort_values(ascending=False)
    print("\nGLOBAL IMPORTANCE (mean |SHAP|, log-odds scale):\n", imp.head(15).round(3))

    fig_dir = ROOT / "reports" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    shap.summary_plot(sv, Z.toarray(), feature_names=names, show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(fig_dir / f"{target}_{model_name}_shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    hi, lo = int(np.argmax(p)), int(np.argmin(p))
    examples = {
        "highest_risk": {"p": round(float(p[hi]), 3), "factors": top_factors(sv[hi], names)},
        "lowest_risk": {"p": round(float(p[lo]), 3), "factors": top_factors(sv[lo], names)},
    }
    print("\nEXAMPLE EXPLANATIONS:", json.dumps(examples, indent=2))
    (ROOT / "reports" / f"{target}_{model_name}_shap.json").write_text(
        json.dumps({"global_importance": imp.head(20).round(4).to_dict(), "examples": examples}, indent=2)
    )


if __name__ == "__main__":
    main(sys.argv[1], *(sys.argv[2:3] or ["xgb"]))