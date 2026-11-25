"""Turn care-gate outcomes into interface copy.

The gating logic lives in ``src/specialties.py``. This module only decides what
the page says about an outcome. Every message is educational and editorial: it
names a specialty as a topic to read about, never as a referral, diagnosis, or
treatment plan.

The thresholds used by the gate were measured on the random-forest bundle. For
any other model build the interface withholds care guidance instead of applying
cut-offs that do not describe that model's scores.
"""

from src.specialties import is_calibrated, suggest_care

TITLES = {
    "emergency": "Urgent care may be appropriate",
    "specialty": "Specialist to read about",
    "uncertain": "Not enough confidence for a care suggestion",
    "unavailable": "Care suggestions are unavailable for this model",
}

MESSAGES = {
    "confident": (
        "The model's top label is {label}. If that label were correct, it "
        "would usually be managed in {specialty}. This is an editorial mapping "
        "for learning, not a referral."
    ),
    "emergency-label": (
        "The model's top label is {label}, a condition this project routes to "
        "urgent care. This demo deliberately over-flags emergencies and cannot "
        "diagnose you. If symptoms are severe, new, or worrying, contact "
        "emergency services or a clinician now instead of relying on this page."
    ),
    "emergency-low-confidence": (
        "The model's top label is a condition this project routes to urgent "
        "care, but its score is too low to act on. Do not read that as "
        "reassurance: if symptoms are severe or worsening, seek urgent help."
    ),
    "low-confidence": (
        "The top score is below the confidence gate, so no specialty is named. "
        "With only one or two symptoms the model is usually guessing; try "
        "adding more symptoms, or treat the labels above as weak suggestions."
    ),
    "unmapped-label": (
        "The model's top label has no specialist mapping in this project, so "
        "no care suggestion is shown."
    ),
    "no-probability": (
        "The model did not return a probability for its top label, so the "
        "confidence gate cannot run and no care suggestion is shown."
    ),
    "invalid-probability": (
        "The model returned a score that cannot be interpreted, so no care "
        "suggestion is shown."
    ),
    "no-prediction": (
        "The model returned no predictions, so there is no care suggestion."
    ),
    "model-not-calibrated": (
        "This model build has not been calibrated for the care gate. The "
        "specialist mapping was measured on the random-forest bundle, so the "
        "page withholds it rather than guess."
    ),
}

GENERIC_MESSAGE = "No care suggestion is available for this result."

CAVEAT = (
    "Educational mapping, not a referral or medical advice. "
    "For real concerns, contact a qualified clinician."
)


def build_care(predictions, model_name):
    """Return the gate outcome plus display copy for the web interface."""
    if not is_calibrated(model_name):
        outcome = {
            "kind": "unavailable",
            "label": None,
            "probability": None,
            "specialty": None,
            "reason": "model-not-calibrated",
        }
    else:
        outcome = suggest_care(predictions)
    return describe_care(outcome)


def describe_care(outcome):
    """Attach a title and message to a gate outcome."""
    kind = outcome.get("kind")
    if kind not in TITLES:
        kind = "uncertain"
    template = MESSAGES.get(outcome.get("reason"), GENERIC_MESSAGE)
    message = template.format(
        label=outcome.get("label") or "an unknown condition",
        specialty=outcome.get("specialty") or "the relevant specialty",
    )
    return {
        **outcome,
        "kind": kind,
        "title": TITLES[kind],
        "message": message,
        "caveat": CAVEAT,
    }
