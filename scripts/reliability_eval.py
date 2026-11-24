"""Reproduce the reliability tables behind the care-suggestion gates.

The care layer only names a specialty when the model is confident, so the
thresholds need evidence rather than intuition. This script builds deliberately
sparse inputs from the dataset's unique symptom patterns, scores them with a
trained bundle, and reports how coverage and accuracy trade off as the gates
move.

Usage:
    python scripts/reliability_eval.py
    python scripts/reliability_eval.py --samples-per-pattern 5 --seed 42
    python scripts/reliability_eval.py --json models/reliability.json
"""

import argparse
import itertools
import json
import random
import sys
from pathlib import Path

# Allow both `python scripts/reliability_eval.py` and `python -m scripts.reliability_eval`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import pandas as pd

from src.modeling import validate_bundle
from src.preprocessing import prepare_dataset
from src.predictor import MODEL_PATH
from src.specialties import (
    EMERGENCY_LABELS,
    EMERGENCY_THRESHOLD,
    SPECIALTY_THRESHOLD,
    suggest_care,
)


def load_patterns(data_path):
    """Return sorted (label, symptom tuple) pairs after duplicate removal."""
    features, labels = prepare_dataset(pd.read_csv(data_path))
    names = list(features.columns)
    patterns = set()
    for label, row in zip(labels, features.itertuples(index=False, name=None)):
        patterns.add((label, tuple(name for name, present in zip(names, row) if present)))
    return sorted(patterns)


def sample_inputs(patterns, sizes, per_pattern, rng):
    """Yield (label, subset) pairs with a fixed number of samples per size."""
    for label, symptoms in patterns:
        ordered = sorted(symptoms)
        for size in sizes:
            if size > len(ordered):
                continue
            combinations = list(itertools.combinations(ordered, size))
            for subset in rng.sample(combinations, min(per_pattern, len(combinations))):
                yield label, subset


def score_inputs(bundle, inputs):
    """Return one record per input with its ranked top-3 predictions."""
    model = bundle["model"]
    if not hasattr(model, "predict_proba"):
        raise SystemExit("The bundle's model does not expose predict_proba.")

    frame = pd.DataFrame(
        [{name: int(name in subset) for name in bundle["feature_names"]} for _, subset in inputs],
        columns=bundle["feature_names"],
    )
    probabilities = model.predict_proba(frame)
    classes = [str(name) for name in model.classes_]

    records = []
    for (label, subset), row in zip(inputs, probabilities):
        ranked = sorted(zip(classes, row), key=lambda item: item[1], reverse=True)
        records.append(
            {
                "size": len(subset),
                "label": label,
                "predictions": [
                    {"label": name, "probability": float(score)}
                    for name, score in ranked[:3]
                ],
            }
        )
    return records


def _format_row(name, covered, total, correct, extra=""):
    coverage = covered / total if total else 0.0
    accuracy = correct / covered if covered else 0.0
    wrong = covered - correct
    print(f"  {name:<26} {coverage:>8.2%} {accuracy:>10.2%} {wrong:>10}{extra}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/raw/dataset.csv")
    parser.add_argument("--model", default=str(MODEL_PATH))
    parser.add_argument("--sizes", default="1,2,3", help="comma-separated subset sizes")
    parser.add_argument("--samples-per-pattern", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--json", help="optional path for a JSON report")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.is_file():
        raise SystemExit(f"No model bundle at {model_path}. Run `python train_model.py` first.")
    bundle = joblib.load(model_path)
    validate_bundle(bundle)

    sizes = [int(value) for value in args.sizes.split(",") if value.strip()]
    rng = random.Random(args.seed)
    inputs = list(sample_inputs(load_patterns(args.data), sizes, args.samples_per_pattern, rng))
    records = score_inputs(bundle, inputs)

    total = len(records)
    print(f"Care-suggestion reliability: {bundle['model_name']}")
    print(f"  inputs: {total} sparse samples (sizes {sizes}, seed {args.seed})")

    print("\nAccuracy by input size")
    print(f"  {'size':<26} {'share':>8} {'top-1':>10}")
    for size in sizes:
        subset = [record for record in records if record["size"] == size]
        if not subset:
            continue
        correct = sum(
            1 for record in subset if record["predictions"][0]["label"] == record["label"]
        )
        print(f"  {str(size) + ' symptom(s)':<26} {len(subset) / total:>8.2%} {correct / len(subset):>10.2%}")

    print("\nSpecialty gate sweep")
    print(f"  {'threshold':<26} {'coverage':>8} {'accuracy':>10} {'wrong':>10}")
    for threshold in (0.4, 0.5, 0.6, 0.7):
        covered = correct = 0
        for record in records:
            top = record["predictions"][0]
            if top["probability"] >= threshold:
                covered += 1
                correct += int(top["label"] == record["label"])
        _format_row(f"top score >= {threshold:.2f}", covered, total, correct)

    print("\nEmergency gate sweep")
    print(f"  {'threshold':<26} {'recall':>8} {'false alarms':>14}")
    true_emergencies = [record for record in records if record["label"] in EMERGENCY_LABELS]
    ordinary = [record for record in records if record["label"] not in EMERGENCY_LABELS]
    for threshold in (0.3, 0.4, 0.5):
        flagged = [
            record
            for record in records
            if record["predictions"][0]["label"] in EMERGENCY_LABELS
            and record["predictions"][0]["probability"] >= threshold
        ]
        caught = sum(1 for record in flagged if record["label"] in EMERGENCY_LABELS)
        false_alarms = sum(1 for record in flagged if record["label"] not in EMERGENCY_LABELS)
        recall = caught / len(true_emergencies) if true_emergencies else 0.0
        rate = false_alarms / len(ordinary) if ordinary else 0.0
        print(f"  {'top score >= ' + format(threshold, '.2f'):<26} {recall:>8.1%} {rate:>14.1%}")

    outcomes = [suggest_care(record["predictions"]) for record in records]
    specialty = [
        (record, outcome)
        for record, outcome in zip(records, outcomes)
        if outcome["kind"] == "specialty"
    ]
    emergency = [
        (record, outcome)
        for record, outcome in zip(records, outcomes)
        if outcome["kind"] == "emergency"
    ]
    specialty_correct = sum(1 for record, _ in specialty if record["predictions"][0]["label"] == record["label"])
    emergency_caught = sum(1 for record, _ in emergency if record["label"] in EMERGENCY_LABELS)
    emergency_false = sum(1 for record, _ in emergency if record["label"] not in EMERGENCY_LABELS)
    uncertain = sum(1 for outcome in outcomes if outcome["kind"] == "uncertain")

    print(
        f"\nSelected policy (specialty {SPECIALTY_THRESHOLD:.2f}, "
        f"emergency {EMERGENCY_THRESHOLD:.2f})"
    )
    print(
        f"  specialty suggestions: {len(specialty)} ({len(specialty) / total:.1%}), "
        f"top-1 correct: {specialty_correct / len(specialty):.1%}"
        if specialty
        else "  specialty suggestions: 0"
    )
    print(
        f"  emergency flags: {len(emergency)}, "
        f"true emergencies caught: {emergency_caught}/{len(true_emergencies)}, "
        f"false alarms: {emergency_false}"
    )
    print(f"  abstained: {uncertain} ({uncertain / total:.1%})")

    if args.json:
        report = {
            "model_name": bundle["model_name"],
            "dataset": bundle.get("dataset", {}),
            "thresholds": {
                "specialty": SPECIALTY_THRESHOLD,
                "emergency": EMERGENCY_THRESHOLD,
            },
            "inputs": {
                "total": total,
                "sizes": sizes,
                "samples_per_pattern": args.samples_per_pattern,
                "seed": args.seed,
            },
            "selected_policy": {
                "specialty_suggestions": len(specialty),
                "specialty_top1_correct": specialty_correct,
                "emergency_flags": len(emergency),
                "emergency_caught": emergency_caught,
                "emergency_false_alarms": emergency_false,
                "abstained": uncertain,
            },
        }
        Path(args.json).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"\nWrote {args.json}")


if __name__ == "__main__":
    main()
