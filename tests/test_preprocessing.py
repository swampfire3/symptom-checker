import pandas as pd
import pytest

from src.preprocessing import (
    SYMPTOM_COLUMNS,
    disease_combination_counts,
    duplicate_summary,
    inspect_supporting_files,
    normalize_symptom,
    prepare_dataset,
)


def make_frame(rows):
    """Build the selected dataset schema from (disease, symptoms) rows."""
    data = []
    for disease, symptoms in rows:
        padded = list(symptoms) + [""] * (len(SYMPTOM_COLUMNS) - len(symptoms))
        data.append({"Disease": disease, **dict(zip(SYMPTOM_COLUMNS, padded))})
    return pd.DataFrame(data, columns=["Disease", *SYMPTOM_COLUMNS])


def test_prepare_dataset_turns_text_symptoms_into_binary_features_and_y():
    frame = make_frame([
        (" Alpha ", [" Fever", "skin rash"]),
        ("Beta", ["cough"]),
    ])
    X, y = prepare_dataset(frame)

    assert list(X.columns) == ["cough", "fever", "skin_rash"]
    assert X.loc[0].to_dict() == {"cough": 0, "fever": 1, "skin_rash": 1}
    assert X.loc[1].to_dict() == {"cough": 1, "fever": 0, "skin_rash": 0}
    assert list(y) == ["Alpha", "Beta"]
    assert y.name == "y"
    assert set(X.to_numpy().ravel()) <= {0, 1}


def test_normalization_standardizes_whitespace_and_separators():
    assert normalize_symptom("  Dischromic _patches ") == "dischromic_patches"
    assert normalize_symptom("High-fever") == "high_fever"


def test_empty_symptom_cells_are_padding():
    X, _ = prepare_dataset(make_frame([("Alpha", ["fever", "", "  "]), ("Beta", ["cough"])]))
    assert X.loc[0, "fever"] == 1
    assert X.loc[0, "cough"] == 0
    assert X.shape == (2, 2)


def test_rejects_wrong_or_unexpected_schema():
    frame = make_frame([("Alpha", ["fever"]), ("Beta", ["cough"])])
    with pytest.raises(ValueError, match="unexpected columns"):
        prepare_dataset(frame.assign(extra="not supported"))
    with pytest.raises(ValueError, match="uses 'Disease'"):
        prepare_dataset(frame, "prognosis")


@pytest.mark.parametrize("frame, message", [
    (make_frame([]), "empty"),
    (make_frame([(None, ["fever"]), ("Beta", ["cough"])]), "missing or blank disease labels"),
    (make_frame([("Alpha", ["fever"]), ("Beta", ["cough"])])
     .assign(Symptom_1=[1, "cough"]), "expected text or empty padding"),
])
def test_rejects_invalid_datasets(frame, message):
    with pytest.raises(ValueError, match=message):
        prepare_dataset(frame)


def test_duplicate_audit_and_per_disease_unique_combinations():
    X, y = prepare_dataset(make_frame([
        ("Alpha", ["fever", "cough"]),
        ("Alpha", ["cough", "fever"]),
        ("Alpha", ["headache"]),
        ("Beta", ["fever", "cough"]),
    ]))
    assert duplicate_summary(X, y) == {
        "rows": 4,
        "exact_duplicate_rows": 1,
        "unique_disease_symptom_combinations": 3,
        "duplicate_combination_rows": 1,
    }
    assert disease_combination_counts(X, y) == {"Alpha": 2, "Beta": 1}


def test_normalization_collision_is_reported(capsys):
    prepare_dataset(make_frame([
        ("Alpha", ["high fever"]),
        ("Beta", ["high_fever"]),
    ]))
    assert "high_fever" in capsys.readouterr().out


def test_supporting_files_report_mismatches_without_merging(tmp_path):
    pd.DataFrame({"Symptom": ["fever"], "weight": [3]}).to_csv(
        tmp_path / "Symptom-severity.csv", index=False
    )
    pd.DataFrame({"Disease": ["Alpha typo"], "Description": ["Description"]}).to_csv(
        tmp_path / "symptom_Description.csv", index=False
    )
    pd.DataFrame({"Disease": ["Alpha"], "Precaution_1": ["Ask a professional"]}).to_csv(
        tmp_path / "symptom_precaution.csv", index=False
    )

    report = inspect_supporting_files(tmp_path, ["Alpha", "Beta"], ["fever", "cough"])
    assert report["severity"]["symptoms_not_in_main"] == []
    assert report["severity"]["main_symptoms_missing_weight"] == ["cough"]
    assert report["description"]["label_mismatches"] == {
        "in_main_not_supporting": ["Alpha", "Beta"],
        "in_supporting_not_main": ["Alpha typo"],
    }
    assert report["precautions"]["label_mismatches"]["in_main_not_supporting"] == ["Beta"]
