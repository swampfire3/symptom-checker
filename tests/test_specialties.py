from pathlib import Path

import pandas as pd
import pytest

from src.specialties import (
    EMERGENCY_LABELS,
    EMERGENCY_THRESHOLD,
    SPECIALTY_BY_LABEL,
    SPECIALTY_THRESHOLD,
    suggest_care,
)

DATASET = Path(__file__).resolve().parents[1] / "data" / "raw" / "dataset.csv"


def ranked(*items):
    return [{"label": label, "probability": probability} for label, probability in items]


def test_every_dataset_label_has_a_specialty():
    labels = set(pd.read_csv(DATASET)["Disease"].astype(str).str.strip())
    assert labels == set(SPECIALTY_BY_LABEL)


def test_emergency_labels_are_mapped():
    assert EMERGENCY_LABELS <= set(SPECIALTY_BY_LABEL)


def test_confident_prediction_returns_a_specialty():
    result = suggest_care(ranked(("Migraine", 0.9), ("Hypertension", 0.05)))
    assert result == {
        "kind": "specialty",
        "label": "Migraine",
        "probability": 0.9,
        "specialty": "Neurology",
        "reason": "confident",
    }


def test_emergency_label_uses_the_lower_threshold():
    assert EMERGENCY_THRESHOLD < SPECIALTY_THRESHOLD
    result = suggest_care(ranked(("Heart attack", 0.35), ("GERD", 0.2)))
    assert result["kind"] == "emergency"
    assert result["reason"] == "emergency-label"


def test_emergency_label_below_its_threshold_abstains():
    result = suggest_care(ranked(("Heart attack", 0.2), ("GERD", 0.1)))
    assert result["kind"] == "uncertain"
    assert result["reason"] == "low-confidence"


def test_low_confidence_abstains():
    result = suggest_care(ranked(("Migraine", 0.3), ("Hypertension", 0.2)))
    assert result["kind"] == "uncertain"
    assert result["reason"] == "low-confidence"


def test_missing_probability_abstains():
    result = suggest_care([{"label": "Migraine", "probability": None}])
    assert result["kind"] == "uncertain"
    assert result["reason"] == "no-probability"


def test_empty_predictions_abstain():
    assert suggest_care([])["reason"] == "no-prediction"
    assert suggest_care(None)["reason"] == "no-prediction"


def test_unmapped_label_abstains_without_crashing():
    result = suggest_care(ranked(("Unknown condition", 0.99)))
    assert result["kind"] == "uncertain"
    assert result["reason"] == "unmapped-label"


def test_only_the_top_prediction_drives_the_outcome():
    result = suggest_care(ranked(("Migraine", 0.9), ("Heart attack", 0.08)))
    assert result["kind"] == "specialty"


@pytest.mark.parametrize("threshold,emergency", [(0, 0.3), (1.1, 0.3), (0.5, 0), (0.5, 2)])
def test_invalid_thresholds_are_rejected(threshold, emergency):
    with pytest.raises(ValueError):
        suggest_care(ranked(("Migraine", 0.9)), threshold=threshold, emergency_threshold=emergency)
