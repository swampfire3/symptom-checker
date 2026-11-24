"""Educational care-pathway suggestions built on top of model predictions.

This module does not diagnose or triage. It is a deterministic, editorial
lookup table plus a confidence gate: it maps a predicted condition to the kind
of specialist that usually manages it, and it abstains when the model's score
is too low to justify naming one. The mapping was written for this project and
has not been reviewed by clinicians.

The thresholds were measured on the default random-forest bundle, whose scores
are the only ones they describe. Other supported models (logistic regression,
the single tree, Jaccard k-NN) have different score scales and need their own
calibration before these cut-offs are trusted. See ``docs/reliability.md``.
"""

import math

# Keys must match the dataset labels exactly, including the upstream typos.
# Do not "fix" the spelling here without updating the dataset and bundle.
SPECIALTY_BY_LABEL = {
    "(vertigo) Paroymsal  Positional Vertigo": "ENT",  # sic: double space
    "AIDS": "Infectious diseases",
    "Acne": "Dermatology",
    "Alcoholic hepatitis": "Hepatology",
    "Allergy": "Allergy and immunology",
    "Arthritis": "Rheumatology",
    "Bronchial Asthma": "Pulmonology",
    "Cervical spondylosis": "Orthopaedics",
    "Chicken pox": "Infectious diseases",
    "Chronic cholestasis": "Hepatology",
    "Common Cold": "General practice",
    "Dengue": "Infectious diseases",
    "Diabetes": "Endocrinology",
    "Dimorphic hemmorhoids(piles)": "Colorectal surgery",  # sic
    "Drug Reaction": "Allergy and immunology",
    "Fungal infection": "Dermatology",
    "GERD": "Gastroenterology",
    "Gastroenteritis": "Gastroenterology",
    "Heart attack": "Emergency medicine",
    "Hepatitis B": "Hepatology",
    "Hepatitis C": "Hepatology",
    "Hepatitis D": "Hepatology",
    "Hepatitis E": "Hepatology",
    "Hypertension": "Cardiology",
    "Hyperthyroidism": "Endocrinology",
    "Hypoglycemia": "Endocrinology",
    "Hypothyroidism": "Endocrinology",
    "Impetigo": "Dermatology",
    "Jaundice": "Hepatology",
    "Malaria": "Infectious diseases",
    "Migraine": "Neurology",
    "Osteoarthristis": "Orthopaedics",  # sic
    "Paralysis (brain hemorrhage)": "Emergency medicine",
    "Peptic ulcer diseae": "Gastroenterology",  # sic
    "Pneumonia": "Pulmonology",
    "Psoriasis": "Dermatology",
    "Tuberculosis": "Pulmonology",
    "Typhoid": "Infectious diseases",
    "Urinary tract infection": "Urology",
    "Varicose veins": "Vascular surgery",
    "hepatitis A": "Hepatology",
}

# Conditions where the educational message should point to urgent care instead
# of a routine specialty. These are intentionally routed at a lower score
# threshold than routine specialties: a missed emergency is worse than an
# unnecessary caution.
EMERGENCY_LABELS = frozenset({"Heart attack", "Paralysis (brain hemorrhage)"})

SPECIALTY_THRESHOLD = 0.6
EMERGENCY_THRESHOLD = 0.3

# The thresholds above were measured on this model's score scale. Other
# supported candidates produce different scales, so callers should check this
# before presenting care guidance from a non-default bundle.
CALIBRATED_MODEL = "random_forest"


def is_calibrated(model_name):
    """Return whether the thresholds describe this model's score scale."""
    return str(model_name) == CALIBRATED_MODEL


def suggest_care(
    predictions,
    threshold=SPECIALTY_THRESHOLD,
    emergency_threshold=EMERGENCY_THRESHOLD,
):
    """Return an emergency, specialty, or uncertain outcome.

    ``predictions`` is an ordered sequence of ``{"label", "probability"}``
    mappings, best first, as produced by ``Predictor.top_predictions``. The
    result is a mapping with ``kind`` (``"emergency"``, ``"specialty"``, or
    ``"uncertain"``), ``label``, ``probability``, ``specialty``, and a short
    ``reason`` code.
    """
    if not 0 < threshold <= 1:
        raise ValueError("threshold must be between 0 and 1.")
    if not 0 < emergency_threshold <= 1:
        raise ValueError("emergency_threshold must be between 0 and 1.")

    ranked = [
        item
        for item in predictions or []
        if isinstance(item, dict) and isinstance(item.get("label"), str)
    ]
    if not ranked:
        return _result("uncertain", reason="no-prediction")

    top = ranked[0]
    label = top["label"]
    probability = top.get("probability")
    if probability is None:
        return _result("uncertain", label=label, reason="no-probability")
    try:
        probability = float(probability)
    except (TypeError, ValueError):
        return _result("uncertain", label=label, reason="invalid-probability")
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        return _result("uncertain", label=label, reason="invalid-probability")

    if label in EMERGENCY_LABELS:
        if probability >= emergency_threshold:
            return _result(
                "emergency",
                label=label,
                probability=probability,
                reason="emergency-label",
            )
        # An emergency label that misses its own lower bar must never fall
        # through to the routine specialty gate, even if the caller has
        # configured the routine threshold below the emergency threshold.
        return _result(
            "uncertain",
            label=label,
            probability=probability,
            reason="emergency-low-confidence",
        )
    if probability < threshold:
        return _result(
            "uncertain",
            label=label,
            probability=probability,
            reason="low-confidence",
        )

    specialty = SPECIALTY_BY_LABEL.get(label)
    if specialty is None:
        return _result(
            "uncertain",
            label=label,
            probability=probability,
            reason="unmapped-label",
        )
    return _result(
        "specialty",
        label=label,
        probability=probability,
        specialty=specialty,
        reason="confident",
    )


def _result(kind, label=None, probability=None, specialty=None, reason=""):
    return {
        "kind": kind,
        "label": label,
        "probability": probability,
        "specialty": specialty,
        "reason": reason,
    }
