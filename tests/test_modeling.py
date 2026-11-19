import json

import joblib
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from src.modeling import (
    SCHEMA_VERSION,
    build_bundle,
    build_candidates,
    cross_validate_candidates,
    deduplicate,
    fit_and_evaluate,
    save_bundle,
    select_model,
    sha256_file,
    split_dataset,
    validate_bundle,
)
from src.preprocessing import SYMPTOM_COLUMNS, prepare_dataset


def make_frame(rows):
    """Build the selected dataset schema from (disease, symptoms) rows."""
    data = []
    for disease, symptoms in rows:
        padded = list(symptoms) + [""] * (len(SYMPTOM_COLUMNS) - len(symptoms))
        data.append({"Disease": disease, **dict(zip(SYMPTOM_COLUMNS, padded))})
    return pd.DataFrame(data, columns=["Disease", *SYMPTOM_COLUMNS])


def dataset_rows():
    patterns = {
        "Alpha": [
            ("fever",),
            ("fever", "cough"),
            ("fever", "cough", "fatigue"),
            ("cough", "fatigue"),
        ],
        "Beta": [
            ("headache",),
            ("headache", "nausea"),
            ("headache", "nausea", "rash"),
            ("nausea", "rash"),
        ],
        "Gamma": [
            ("itching",),
            ("itching", "rash"),
            ("itching", "rash", "fatigue"),
            ("rash", "fatigue"),
        ],
    }
    return [
        (disease, symptoms)
        for disease, combinations in patterns.items()
        for symptoms in combinations
    ]


def small_candidates():
    return {
        "tree_baseline": DecisionTreeClassifier(max_depth=2, random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=5, random_state=42),
    }


def signatures(features, labels):
    return {
        (str(label), tuple(name for name, present in zip(features.columns, row) if present))
        for label, row in zip(labels, features.itertuples(index=False, name=None))
    }


def test_build_candidates_are_picklable(tmp_path):
    candidates = build_candidates()
    assert set(candidates) == {
        "tree_baseline",
        "random_forest",
        "logreg",
        "knn_jaccard",
    }
    for name, estimator in candidates.items():
        path = tmp_path / f"{name}.joblib"
        joblib.dump(estimator, path)
        joblib.load(path)


def test_cross_validation_reports_every_candidate():
    X, y = prepare_dataset(make_frame(dataset_rows()))
    results = cross_validate_candidates(
        X,
        y,
        folds=2,
        repeats=2,
        seed=7,
        candidates=small_candidates(),
    )

    assert set(results) == {"tree_baseline", "random_forest"}
    for result in results.values():
        assert len(result["fold_scores"]) == 4
        assert result["folds"] == 2
        assert result["repeats"] == 2
        assert 0 <= result["min_accuracy"] <= result["max_accuracy"] <= 1
    json.dumps(results)


def test_cross_validation_rejects_insufficient_class_examples():
    X, y = prepare_dataset(make_frame([("Alpha", ["fever"]), ("Beta", ["cough"])]))
    with pytest.raises(ValueError, match="smallest disease class"):
        cross_validate_candidates(X, y, folds=2, repeats=1, candidates=small_candidates())


@pytest.mark.parametrize("folds, repeats, message", [(1, 1, "two folds"), (2, 0, "one repeat")])
def test_cross_validation_rejects_invalid_settings(folds, repeats, message):
    X, y = prepare_dataset(make_frame(dataset_rows()))
    with pytest.raises(ValueError, match=message):
        cross_validate_candidates(
            X,
            y,
            folds=folds,
            repeats=repeats,
            candidates=small_candidates(),
        )


def test_select_model_prefers_random_forest_on_ties():
    results = {
        "tree_baseline": {"mean_accuracy": 0.500},
        "logreg": {"mean_accuracy": 1.000},
        "random_forest": {"mean_accuracy": 1.000},
    }
    assert select_model(results) == "random_forest"
    assert select_model(results, "logreg") == "logreg"
    assert select_model(results, "tree_baseline") == "tree_baseline"
    with pytest.raises(ValueError, match="Unknown model"):
        select_model(results, "missing")


def test_deduplicate_removes_copy_rows_only():
    X, y = prepare_dataset(
        make_frame(
            [
                ("Alpha", ["fever", "cough"]),
                ("Alpha", ["cough", "fever"]),
                ("Beta", ["headache"]),
            ]
        )
    )
    unique_X, unique_y = deduplicate(X, y)

    assert len(unique_X) == 2
    assert list(unique_y) == ["Alpha", "Beta"]
    assert unique_y.name == "y"
    assert len(X) == 3


def test_split_does_not_leak_duplicate_combinations():
    X, y = prepare_dataset(
        make_frame(dataset_rows() + [("Alpha", ["fever", "cough", "fatigue"])])
    )
    unique_X, unique_y = deduplicate(X, y)
    x_train, x_test, y_train, y_test = split_dataset(unique_X, unique_y, test_size=0.25, seed=7)

    assert signatures(x_train, y_train).isdisjoint(signatures(x_test, y_test))
    assert len(x_train) + len(x_test) == len(unique_X)


def test_split_rejects_single_combination_class():
    X, y = prepare_dataset(
        make_frame([("Alpha", ["fever"]), ("Beta", ["cough"]), ("Beta", ["rash"])])
    )
    with pytest.raises(ValueError, match="fewer than two unique combinations"):
        split_dataset(X, y, test_size=0.5)


def test_fit_and_evaluate_returns_serializable_report():
    X, y = prepare_dataset(make_frame(dataset_rows()))
    x_train, x_test, y_train, y_test = split_dataset(X, y, test_size=0.25, seed=7)
    model, report = fit_and_evaluate(
        DecisionTreeClassifier(random_state=42), x_train, x_test, y_train, y_test
    )

    assert isinstance(model, DecisionTreeClassifier)
    assert 0 <= report["accuracy"] <= 1
    assert 0 <= report["macro_f1"] <= 1
    assert report["test_rows"] == len(y_test)
    assert len(report["confusion_matrix"]) == len(report["labels"])
    assert sum(item["support"] for item in report["per_class"].values()) == len(y_test)
    json.dumps(report)


def test_bundle_round_trip_writes_model_and_metrics(tmp_path):
    bundle = build_bundle(
        DecisionTreeClassifier(random_state=42),
        "tree_baseline",
        ["fever", "cough"],
        ["Alpha", "Beta"],
        {"holdout": {"accuracy": 1.0}},
        {"filename": "dataset.csv"},
        trained_at="2026-01-01T00:00:00+00:00",
    )
    model_path, metrics_path = save_bundle(
        bundle, tmp_path / "model.joblib", tmp_path / "metrics.json"
    )

    loaded = joblib.load(model_path)
    validate_bundle(loaded)
    assert loaded["schema_version"] == SCHEMA_VERSION
    assert loaded["feature_names"] == ["fever", "cough"]
    assert loaded["labels"] == ["Alpha", "Beta"]
    assert loaded["trained_at"] == "2026-01-01T00:00:00+00:00"
    assert loaded["library_versions"]["scikit-learn"]

    document = json.loads(metrics_path.read_text())
    assert "model" not in document
    assert document["dataset"]["filename"] == "dataset.csv"
    assert document["metrics"]["holdout"]["accuracy"] == 1.0


def test_save_bundle_keeps_previous_files_when_a_write_fails(tmp_path, monkeypatch):
    bundle = build_bundle(
        DecisionTreeClassifier(random_state=42),
        "tree_baseline",
        ["fever", "cough"],
        ["Alpha", "Beta"],
        {"holdout": {"accuracy": 1.0}},
        {"filename": "dataset.csv"},
        trained_at="2026-01-01T00:00:00+00:00",
    )
    model_path, metrics_path = save_bundle(
        bundle, tmp_path / "model.joblib", tmp_path / "metrics.json"
    )
    original_model = model_path.read_bytes()
    original_metrics = metrics_path.read_text()

    def fail_to_write(*args, **kwargs):
        raise OSError("simulated disk failure")

    monkeypatch.setattr("src.modeling.joblib.dump", fail_to_write)
    replacement = build_bundle(
        DecisionTreeClassifier(random_state=42),
        "tree_baseline",
        ["fever"],
        ["Alpha"],
        {},
        {"filename": "other.csv"},
        trained_at="2026-02-02T00:00:00+00:00",
    )
    with pytest.raises(OSError, match="simulated disk failure"):
        save_bundle(replacement, model_path, metrics_path)

    assert model_path.read_bytes() == original_model
    assert metrics_path.read_text() == original_metrics
    assert {path.name for path in tmp_path.iterdir()} == {"model.joblib", "metrics.json"}


def test_validate_bundle_rejects_missing_keys_and_wrong_schema():
    with pytest.raises(ValueError, match="missing"):
        validate_bundle({"schema_version": SCHEMA_VERSION})

    bundle = build_bundle(
        DecisionTreeClassifier(random_state=42),
        "tree_baseline",
        ["fever"],
        ["Alpha"],
        {},
        {"filename": "dataset.csv"},
    )
    bundle["schema_version"] = 99
    with pytest.raises(ValueError, match="Unsupported model bundle schema"):
        validate_bundle(bundle)


def test_sha256_file_hashes_contents(tmp_path):
    path = tmp_path / "dataset.csv"
    path.write_bytes(b"abc")
    assert (
        sha256_file(path)
        == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )
