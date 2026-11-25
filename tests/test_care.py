import pytest

from src.care import MESSAGES, TITLES, build_care, describe_care


def ranked(*items):
    return [
        {"label": label, "probability": probability}
        for label, probability in items
    ]


def test_confident_prediction_names_the_mapped_specialty():
    care = build_care(ranked(("Migraine", 0.9)), "random_forest")

    assert care["kind"] == "specialty"
    assert care["specialty"] == "Neurology"
    assert care["reason"] == "confident"
    assert care["title"] == TITLES["specialty"]
    assert "Neurology" in care["message"]


def test_emergency_prediction_does_not_offer_a_routine_specialty():
    care = build_care(ranked(("Heart attack", 0.5)), "random_forest")

    assert care["kind"] == "emergency"
    assert care["specialty"] is None
    assert care["title"] == TITLES["emergency"]
    assert "emergency services" in care["message"]
    assert "over-flags" in care["message"]
    assert "Specialist to read about" not in care["title"]


def test_low_confidence_prediction_abstains_with_explanation():
    care = build_care(ranked(("Migraine", 0.2)), "random_forest")

    assert care["kind"] == "uncertain"
    assert care["reason"] == "low-confidence"
    assert care["title"] == TITLES["uncertain"]
    assert "below the confidence gate" in care["message"]


def test_uncalibrated_model_withholds_guidance_before_gating():
    care = build_care(ranked(("Migraine", 0.99)), "logreg")

    assert care["kind"] == "unavailable"
    assert care["reason"] == "model-not-calibrated"
    assert care["label"] is None
    assert care["specialty"] is None
    assert care["title"] == TITLES["unavailable"]


def test_unmapped_label_gets_its_own_copy():
    care = build_care(ranked(("Unknown condition", 0.99)), "random_forest")

    assert care["kind"] == "uncertain"
    assert care["reason"] == "unmapped-label"
    assert "no specialist mapping" in care["message"]


def test_invalid_probability_gets_its_own_copy():
    care = build_care(
        [{"label": "Migraine", "probability": float("nan")}],
        "random_forest",
    )

    assert care["kind"] == "uncertain"
    assert care["reason"] == "invalid-probability"
    assert "cannot be interpreted" in care["message"]


def test_every_gate_reason_has_copy():
    reasons = {
        "confident",
        "emergency-label",
        "emergency-low-confidence",
        "low-confidence",
        "unmapped-label",
        "no-probability",
        "invalid-probability",
        "no-prediction",
        "model-not-calibrated",
    }
    assert reasons == set(MESSAGES)
    assert set(TITLES) == {"emergency", "specialty", "uncertain", "unavailable"}


def kind_for_reason(reason):
    if reason == "model-not-calibrated":
        return "unavailable"
    if reason.startswith("emergency"):
        return "emergency"
    if reason == "confident":
        return "specialty"
    return "uncertain"


@pytest.mark.parametrize("reason", sorted(MESSAGES))
def test_described_copy_leaves_no_placeholders(reason):
    care = describe_care(
        {
            "kind": kind_for_reason(reason),
            "reason": reason,
            "label": "Example condition",
            "specialty": "Neurology",
        }
    )

    assert care["message"].strip()
    assert "{" not in care["message"]
    assert "}" not in care["message"]
    assert care["caveat"]


@pytest.mark.parametrize("reason", ["confident", "emergency-label"])
def test_described_copy_names_the_predicted_label(reason):
    care = describe_care(
        {
            "kind": kind_for_reason(reason),
            "reason": reason,
            "label": "Example condition",
            "specialty": "Neurology",
        }
    )

    assert "Example condition" in care["message"]


def test_described_copy_names_the_specialty():
    care = describe_care(
        {
            "kind": "specialty",
            "reason": "confident",
            "label": "Migraine",
            "specialty": "Neurology",
        }
    )

    assert "Neurology" in care["message"]
    assert care["caveat"].startswith("Educational mapping")


def test_unknown_outcome_falls_back_without_crashing():
    care = describe_care({"kind": "mystery", "reason": "mystery"})

    assert care["kind"] == "uncertain"
    assert care["title"] == TITLES["uncertain"]
    assert care["message"]
