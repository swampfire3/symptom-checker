"""Model candidates, evaluation helpers, and model-bundle artifacts."""

import json
import os
import platform
import tempfile
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import joblib
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.tree import DecisionTreeClassifier

from src.preprocessing import TARGET_COLUMN

SCHEMA_VERSION = 2
BUNDLE_REQUIRED_KEYS = ("schema_version", "model_name", "model", "feature_names", "labels")
MODEL_PREFERENCE = ("random_forest", "logreg", "tree_baseline", "knn_jaccard")
METRIC_FIELDS = ("precision", "recall", "f1-score")


def _as_boolean(features):
    return features.astype(bool)


def build_candidates():
    """Return the named estimators compared during training."""
    return {
        "tree_baseline": DecisionTreeClassifier(
            max_depth=12, random_state=42, class_weight="balanced"
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100, random_state=42, class_weight="balanced"
        ),
        "logreg": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "knn_jaccard": make_pipeline(
            FunctionTransformer(_as_boolean, validate=False),
            KNeighborsClassifier(n_neighbors=1, metric="jaccard"),
        ),
    }


def cross_validate_candidates(X, y, folds=5, repeats=5, seed=42, candidates=None):
    """Score each candidate with repeated stratified cross-validation."""
    if folds < 2:
        raise ValueError("Cross-validation needs at least two folds.")
    if repeats < 1:
        raise ValueError("Cross-validation needs at least one repeat.")
    smallest_class = int(y.value_counts().min())
    if smallest_class < folds:
        raise ValueError(
            f"Cannot use {folds} folds: the smallest disease class has only "
            f"{smallest_class} unique symptom combination(s). Lower --cv-folds."
        )

    candidates = dict(candidates if candidates is not None else build_candidates())
    splitter = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats, random_state=seed)
    results = {}
    for name, estimator in candidates.items():
        scores = cross_val_score(estimator, X, y, cv=splitter, n_jobs=-1, error_score="raise")
        results[name] = {
            "mean_accuracy": float(scores.mean()),
            "std_accuracy": float(scores.std()),
            "min_accuracy": float(scores.min()),
            "max_accuracy": float(scores.max()),
            "fold_scores": [float(score) for score in scores],
            "folds": folds,
            "repeats": repeats,
        }
    return results


def select_model(results, requested=None):
    """Pick a model by mean accuracy, preferring the documented default on ties."""
    if not results:
        raise ValueError("No model results were supplied.")
    if requested is not None:
        if requested not in results:
            raise ValueError(
                f"Unknown model {requested!r}. Available models: {', '.join(sorted(results))}."
            )
        return requested

    def rank(name):
        preference = (
            MODEL_PREFERENCE.index(name) if name in MODEL_PREFERENCE else len(MODEL_PREFERENCE)
        )
        return (-round(results[name]["mean_accuracy"], 6), preference, name)

    return sorted(results, key=rank)[0]


def deduplicate(features, labels):
    """Drop duplicate normalized feature/label rows before any split."""
    if TARGET_COLUMN in features.columns:
        raise ValueError(
            f"Features must not already contain the {TARGET_COLUMN!r} target column."
        )
    if len(features) != len(labels):
        raise ValueError("Features and labels must have the same number of rows.")

    unique = features.copy()
    unique[TARGET_COLUMN] = labels.to_numpy()
    unique = unique.drop_duplicates()
    if unique.empty:
        raise ValueError("No rows remain after removing duplicate feature/label pairs.")
    unique_labels = unique.pop(TARGET_COLUMN).rename(labels.name or "y")
    return unique, unique_labels


def split_dataset(features, labels, test_size=0.2, seed=42):
    """Create a reproducible stratified holdout after deduplication."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")
    counts = labels.value_counts()
    if counts.min() < 2:
        raise ValueError(
            "Some diseases have fewer than two unique combinations; "
            "a stratified holdout is not reliable."
        )
    return train_test_split(
        features,
        labels,
        test_size=test_size,
        random_state=seed,
        stratify=labels,
    )


def _json_safe_report(report, label_order):
    per_class = {}
    for label in label_order:
        detail = report.get(str(label))
        if detail is None:
            continue
        per_class[str(label)] = {
            field: float(detail[field]) for field in METRIC_FIELDS
        }
        per_class[str(label)]["support"] = int(detail["support"])
    return per_class


def fit_and_evaluate(estimator, X_train, X_test, y_train, y_test):
    """Fit a fresh estimator and return it with a JSON-serializable report."""
    model = clone(estimator)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    label_order = sorted(set(y_train) | set(y_test))
    report = classification_report(
        y_test,
        predictions,
        labels=label_order,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_test, predictions, labels=label_order)
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "test_rows": int(len(y_test)),
        "labels": [str(label) for label in label_order],
        "per_class": _json_safe_report(report, label_order),
        "confusion_matrix": matrix.tolist(),
    }
    return model, metrics


def sha256_file(path):
    """Return the SHA-256 hex digest of a file."""
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_version(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


def build_bundle(
    model,
    model_name,
    feature_names,
    labels,
    metrics,
    dataset,
    trained_at=None,
):
    """Assemble the self-describing artifact that Predictor loads."""
    bundle = {
        "schema_version": SCHEMA_VERSION,
        "model_name": str(model_name),
        "model": model,
        "feature_names": [str(name) for name in feature_names],
        "labels": [str(label) for label in labels],
        "metrics": metrics,
        "dataset": dataset,
        "trained_at": trained_at or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "library_versions": {
            "python": platform.python_version(),
            "scikit-learn": _package_version("scikit-learn"),
            "pandas": _package_version("pandas"),
            "numpy": _package_version("numpy"),
            "joblib": _package_version("joblib"),
        },
    }
    validate_bundle(bundle)
    return bundle


def validate_bundle(bundle):
    """Raise a helpful error when a model bundle is missing or outdated."""
    if not isinstance(bundle, dict):
        raise ValueError(
            "Model bundle must be a mapping. Retrain with `python train_model.py`."
        )
    missing = [key for key in BUNDLE_REQUIRED_KEYS if key not in bundle]
    if missing:
        raise ValueError(
            f"Model bundle is missing {', '.join(missing)}. "
            "Retrain with `python train_model.py`."
        )
    if bundle["schema_version"] != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported model bundle schema {bundle['schema_version']!r}; "
            f"expected {SCHEMA_VERSION}. Retrain with `python train_model.py`."
        )
    if not isinstance(bundle["feature_names"], (list, tuple)) or not bundle["feature_names"]:
        raise ValueError(
            "Model bundle has no feature names. Retrain with `python train_model.py`."
        )
    if not isinstance(bundle["labels"], (list, tuple)) or not bundle["labels"]:
        raise ValueError(
            "Model bundle has no disease labels. Retrain with `python train_model.py`."
        )


def _temporary_path(path):
    """Create a unique sibling path so a completed write can be swapped in."""
    handle, name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    os.close(handle)
    return Path(name)


def save_bundle(bundle, model_path, metrics_path=None):
    """Write the bundle and optional metrics, replacing each file atomically.

    Readers never observe a half-written bundle: both documents are written to
    temporary files first and then moved into place. The metrics document is
    derived from the same bundle, so its ``trained_at`` and dataset hash show
    whether a model and metrics file came from the same training run.
    """
    validate_bundle(bundle)
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    metrics_text = None
    if metrics_path is not None:
        metrics_path = Path(metrics_path)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        document = {key: value for key, value in bundle.items() if key != "model"}
        metrics_text = json.dumps(document, indent=2, sort_keys=True) + "\n"

    temporary_model = _temporary_path(model_path)
    temporary_metrics = (
        _temporary_path(metrics_path) if metrics_path is not None else None
    )
    try:
        joblib.dump(bundle, temporary_model)
        if temporary_metrics is not None:
            temporary_metrics.write_text(metrics_text, encoding="utf-8")
        os.replace(temporary_model, model_path)
        if temporary_metrics is not None:
            os.replace(temporary_metrics, metrics_path)
    finally:
        temporary_model.unlink(missing_ok=True)
        if temporary_metrics is not None:
            temporary_metrics.unlink(missing_ok=True)

    return model_path, metrics_path
