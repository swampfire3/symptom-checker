"""Preprocessing and data-quality checks for the selected symptom CSV format."""

import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

TARGET_COLUMN = "Disease"
SYMPTOM_COLUMNS = [f"Symptom_{number}" for number in range(1, 18)]
SUPPORTING_FILES = {
    "severity": "Symptom-severity.csv",
    "description": "symptom_Description.csv",
    "precautions": "symptom_precaution.csv",
}


def normalize_symptom(value):
    """Canonicalize symptom text to lowercase underscore-separated names."""
    value = str(value).strip().lower()
    value = re.sub(r"[\s-]+", "_", value)
    return re.sub(r"_+", "_", value).strip("_")


def _check_columns(dataframe):
    expected = {TARGET_COLUMN, *SYMPTOM_COLUMNS}
    missing = sorted(expected - set(dataframe.columns))
    unexpected = sorted(set(dataframe.columns) - expected)
    if missing or unexpected:
        raise ValueError(
            f"Unexpected dataset schema. Missing columns: {missing or 'none'}; "
            f"unexpected columns: {unexpected or 'none'}."
        )


def prepare_dataset(dataframe, target_column=TARGET_COLUMN):
    """Convert the selected wide text dataset into binary X and disease y.

    Empty symptom cells are padding and become all-zero entries. Formatting
    variants that normalize to the same name are reported as collisions; they
    are merged only by the documented deterministic normalization rule.
    """
    if target_column != TARGET_COLUMN:
        raise ValueError(f"This dataset uses {TARGET_COLUMN!r} as its target column.")
    if dataframe.empty:
        raise ValueError("The primary dataset is empty.")
    _check_columns(dataframe)

    target = dataframe[TARGET_COLUMN]
    labels = target.astype("string").str.strip()
    invalid_labels = labels.isna() | labels.eq("")
    if invalid_labels.any():
        raise ValueError(
            f"Found {int(invalid_labels.sum())} rows with missing or blank disease labels; "
            "inspect and resolve them before training."
        )
    y = labels.astype(str).rename("y")

    variants = defaultdict(set)
    normalized_rows = []
    for row_number, row in enumerate(dataframe[SYMPTOM_COLUMNS].itertuples(index=False, name=None), start=2):
        present = set()
        for column, value in zip(SYMPTOM_COLUMNS, row):
            if pd.isna(value) or (isinstance(value, str) and not value.strip()):
                continue
            if not isinstance(value, str):
                raise ValueError(
                    f"Invalid symptom value at CSV row {row_number}, column {column}: "
                    f"expected text or empty padding, got {value!r}."
                )
            original = value.strip()
            normalized = normalize_symptom(original)
            if not normalized:
                raise ValueError(f"Symptom at CSV row {row_number}, column {column} normalizes to empty.")
            variants[normalized].add(original)
            present.add(normalized)
        normalized_rows.append(present)

    collisions = {name: sorted(values) for name, values in variants.items() if len(values) > 1}
    if collisions:
        print("Normalization collisions (format variants intentionally canonicalized):")
        for name, values in sorted(collisions.items()):
            print(f"  {name}: {values}")

    feature_names = sorted(variants)
    if not feature_names:
        raise ValueError("No non-empty symptom values were found.")
    X = pd.DataFrame(
        [{name: int(name in present) for name in feature_names} for present in normalized_rows],
        columns=feature_names,
        index=dataframe.index,
        dtype="int8",
    )
    if X.empty or y.empty:
        raise ValueError("Preprocessing produced an empty dataset.")
    if y.nunique() < 2:
        raise ValueError("At least two disease classes are required for training.")
    return X, y


def disease_combination_counts(X, y):
    """Count distinct normalized symptom sets per disease."""
    counts = defaultdict(set)
    for label, row in zip(y, X.itertuples(index=False, name=None)):
        signature = tuple(name for name, present in zip(X.columns, row) if present)
        counts[label].add(signature)
    return {label: len(signatures) for label, signatures in sorted(counts.items())}


def duplicate_summary(X, y):
    """Return full-row and normalized disease/symptom-signature duplicate counts."""
    frame = X.copy()
    frame.insert(0, TARGET_COLUMN, y.to_numpy())
    signatures = pd.Series(
        [(label, tuple(name for name, present in zip(X.columns, row) if present))
         for label, row in zip(y, X.itertuples(index=False, name=None))],
        index=X.index,
    )
    return {
        "rows": len(frame),
        "exact_duplicate_rows": int(frame.duplicated().sum()),
        "unique_disease_symptom_combinations": int(signatures.nunique()),
        "duplicate_combination_rows": int(len(signatures) - signatures.nunique()),
    }


def inspect_supporting_files(raw_dir, disease_labels, symptom_names):
    """Report sidecar label/symptom mismatches without merging labels or features."""
    raw_dir = Path(raw_dir)
    canonical_labels = set(disease_labels)
    report = {}
    for kind, filename in SUPPORTING_FILES.items():
        path = raw_dir / filename
        if not path.is_file():
            report[kind] = {"missing_file": filename}
            continue
        data = pd.read_csv(path)
        detail = {"rows": len(data)}
        if kind == "severity":
            if "Symptom" not in data.columns:
                detail["error"] = "Missing Symptom column"
            else:
                raw_symptoms = data["Symptom"].dropna().astype(str)
                normalized_symptoms = raw_symptoms.map(normalize_symptom)
                side_symptoms = {value for value in normalized_symptoms if value}
                detail["symptoms_not_in_main"] = sorted(side_symptoms - set(symptom_names))
                detail["main_symptoms_missing_weight"] = sorted(set(symptom_names) - side_symptoms)
                detail["normalization_collisions"] = {
                    normalized: sorted(set(raw_symptoms[normalized_symptoms == normalized]))
                    for normalized in side_symptoms
                    if raw_symptoms[normalized_symptoms == normalized].nunique() > 1
                }
                if "weight" in data.columns:
                    weights = pd.to_numeric(data["weight"], errors="coerce")
                    detail["invalid_weight_rows"] = int(weights.isna().sum())
        else:
            if "Disease" not in data.columns:
                detail["error"] = "Missing Disease column"
            else:
                side_labels = set(data["Disease"].dropna().astype(str).str.strip())
                detail["label_mismatches"] = {
                    "in_main_not_supporting": sorted(canonical_labels - side_labels),
                    "in_supporting_not_main": sorted(side_labels - canonical_labels),
                }
                detail["duplicate_label_rows"] = int(data["Disease"].dropna().astype(str).str.strip().duplicated().sum())
                text_columns = [column for column in data.columns if column != "Disease"]
                detail["rows_with_missing_text"] = int(data[text_columns].isna().any(axis=1).sum())
        report[kind] = detail
    return report
