"""Reusable model inference and artifact-contract helpers.

Training exports a feature specification next to each model. API code uses this
module so preprocessing, prediction, and SHAP explanations stay aligned.
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class ArtifactContractError(ValueError):
    """Raised when request features do not match the trained model contract."""


@dataclass(frozen=True)
class Factor:
    """One original input column's signed contribution to a prediction."""

    feature: str
    value: Any
    impact: float


def feature_spec_path(model_path: Path) -> Path:
    """Return the standard feature-spec path for a saved model."""
    return model_path.with_suffix(".features.json")


def metadata_path(model_path: Path) -> Path:
    """Return the standard metadata path for a saved model."""
    return model_path.with_suffix(".metadata.json")


def _git_revision() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() or "unknown"


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_feature_spec(
    features: pd.DataFrame,
    test_probabilities: np.ndarray,
    *,
    target: str,
    model_name: str,
) -> dict[str, Any]:
    """Build the API feature contract from the exact training dataframe.

    The threshold is an 80th-percentile *demo operating point*, not a clinical
    cutoff. The API must label it accordingly.
    """
    numeric = features.select_dtypes(include="number").columns.tolist()
    categorical_columns = [column for column in features.columns if column not in numeric]
    categories = {
        column: sorted(str(value) for value in features[column].dropna().unique())
        for column in categorical_columns
    }
    return {
        "schema_version": "1.0",
        "target": target,
        "model_name": model_name,
        "columns": features.columns.tolist(),
        "numeric": numeric,
        "categorical": categories,
        "high_risk_threshold": float(np.quantile(test_probabilities, 0.80)),
        "threshold_policy": (
            "80th percentile of held-out predictions; research/demo operating point only"
        ),
    }


def build_metadata(
    *,
    target: str,
    model_name: str,
    metrics: dict[str, Any],
    raw_data_path: Path | None,
) -> dict[str, Any]:
    """Record enough context to reproduce and identify a trained artifact."""
    import sklearn
    import xgboost

    return {
        "schema_version": "1.0",
        "target": target,
        "model_name": model_name,
        "trained_at": datetime.now(UTC).isoformat(),
        "git_revision": _git_revision(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "library_versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
        },
        "raw_data_sha256": _sha256(raw_data_path) if raw_data_path else None,
        "metrics": metrics,
    }


def write_artifacts(
    model_path: Path,
    features: pd.DataFrame,
    test_probabilities: np.ndarray,
    *,
    target: str,
    model_name: str,
    metrics: dict[str, Any],
    raw_data_path: Path | None,
) -> tuple[Path, Path]:
    """Write feature-spec and metadata JSON beside a saved model."""
    spec_file = feature_spec_path(model_path)
    meta_file = metadata_path(model_path)
    spec_file.write_text(
        json.dumps(
            build_feature_spec(
                features, test_probabilities, target=target, model_name=model_name
            ),
            indent=2,
        )
    )
    meta_file.write_text(
        json.dumps(
            build_metadata(
                target=target,
                model_name=model_name,
                metrics=metrics,
                raw_data_path=raw_data_path,
            ),
            indent=2,
        )
    )
    return spec_file, meta_file


def source_column(transformed_name: str, columns: list[str]) -> str:
    """Map an encoded feature name back to its original input column."""
    name = transformed_name.split("__", 1)[-1]
    for column in sorted(columns, key=len, reverse=True):
        if name == column or name.startswith(f"{column}_"):
            return column
    return name


def json_value(value: Any) -> Any:
    """Convert NumPy scalar values so API responses remain JSON serializable."""
    return value.item() if isinstance(value, np.generic) else value


class Predictor:
    """Load one trained sklearn pipeline and return a probability plus factors."""

    def __init__(self, model_path: str | Path, spec_path: str | Path):
        import joblib
        import shap
        from scipy import sparse

        self.model_path = Path(model_path)
        self.spec_path = Path(spec_path)
        self.pipe = joblib.load(self.model_path)
        self.spec = json.loads(self.spec_path.read_text())
        self.columns: list[str] = self.spec["columns"]
        self.pre = self.pipe.named_steps["pre"]
        self.clf = self.pipe.named_steps["clf"]
        self._sparse = sparse
        self._explainer = shap.TreeExplainer(self.clf)
        self._output_names = self.pre.get_feature_names_out().tolist()

    def _frame(self, features: dict[str, Any]) -> pd.DataFrame:
        expected = set(self.columns)
        supplied = set(features)
        missing = sorted(expected - supplied)
        extra = sorted(supplied - expected)
        if missing or extra:
            parts = []
            if missing:
                parts.append(f"missing fields: {missing}")
            if extra:
                parts.append(f"unexpected fields: {extra}")
            raise ArtifactContractError("; ".join(parts))
        return pd.DataFrame([{column: features[column] for column in self.columns}])

    def predict(self, features: dict[str, Any], top_k: int = 5) -> tuple[float, list[Factor]]:
        """Predict one case and explain it on the exact model input matrix."""
        frame = self._frame(features)
        matrix = self.pre.transform(frame)
        if not self._sparse.issparse(matrix):
            matrix = self._sparse.csr_matrix(matrix)
        matrix = matrix.tocsr()

        probability = float(self.clf.predict_proba(matrix)[0, 1])
        shap_values = np.asarray(self._explainer.shap_values(matrix))
        if shap_values.ndim != 2 or shap_values.shape[0] != 1:
            raise RuntimeError("Expected one-row binary SHAP values from the classifier")
        contributions = shap_values[0]
        baseline = float(np.ravel(self._explainer.expected_value)[0])
        reconstructed = float(1.0 / (1.0 + np.exp(-(contributions.sum() + baseline))))
        if abs(probability - reconstructed) > 1e-4:
            raise RuntimeError(
                "SHAP explanation does not match model prediction: "
                f"{probability:.6f} vs {reconstructed:.6f}"
            )

        by_column: dict[str, float] = {}
        for name, impact in zip(self._output_names, contributions, strict=True):
            column = source_column(name, self.columns)
            by_column[column] = by_column.get(column, 0.0) + float(impact)
        ranked = sorted(by_column.items(), key=lambda item: abs(item[1]), reverse=True)[:top_k]
        row = frame.iloc[0]
        return probability, [
            Factor(feature=column, value=row[column], impact=impact)
            for column, impact in ranked
        ]

    def factor_dicts(self, features: dict[str, Any], top_k: int = 5) -> tuple[float, list[dict[str, Any]]]:
        """Convenience shape for API adapters."""
        probability, factors = self.predict(features, top_k=top_k)
        return probability, [
            {**asdict(factor), "value": json_value(factor.value)} for factor in factors
        ]
