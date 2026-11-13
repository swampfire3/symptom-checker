"""Train, benchmark, evaluate, and save a symptom-classification model."""

import argparse
from pathlib import Path

import pandas as pd

from src.modeling import (
    build_bundle,
    build_candidates,
    cross_validate_candidates,
    deduplicate,
    fit_and_evaluate,
    save_bundle,
    select_model,
    sha256_file,
    split_dataset,
)
from src.preprocessing import (
    TARGET_COLUMN,
    disease_combination_counts,
    duplicate_summary,
    inspect_supporting_files,
    prepare_dataset,
)

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
MODEL_PATH = ROOT / "models" / "symptom_disease_model.joblib"
DATA_PATH = RAW_DIR / "dataset.csv"
MODEL_CHOICES = {
    "tree": "tree_baseline",
    "forest": "random_forest",
    "logreg": "logreg",
    "knn": "knn_jaccard",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=DATA_PATH,
        help="Primary dataset CSV (default: data/raw/dataset.csv)",
    )
    parser.add_argument(
        "--target",
        default=TARGET_COLUMN,
        choices=[TARGET_COLUMN],
        help="Disease-label column",
    )
    parser.add_argument(
        "--model",
        default="forest",
        choices=["auto", *MODEL_CHOICES],
        help="Model to train (default: forest; use auto for highest cross-validation accuracy)",
    )
    parser.add_argument("--cv-folds", type=int, default=5, help="Cross-validation folds")
    parser.add_argument("--cv-repeats", type=int, default=5, help="Cross-validation repeats")
    parser.add_argument("--test-size", type=float, default=0.2, help="Holdout fraction")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--out",
        type=Path,
        default=MODEL_PATH,
        help="Model bundle path (default: models/symptom_disease_model.joblib)",
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=None,
        help="Metrics JSON path (default: metrics.json next to the model bundle)",
    )
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Run the audit and evaluation without saving a model bundle",
    )
    return parser.parse_args()


def print_benchmark(results, chosen):
    print("Cross-validation results:")
    ranked = sorted(results.items(), key=lambda item: -item[1]["mean_accuracy"])
    for name, result in ranked:
        marker = "*" if name == chosen else " "
        print(
            f" {marker} {name:16} {result['mean_accuracy']:.3f} "
            f"+/- {result['std_accuracy']:.3f} "
            f"(min {result['min_accuracy']:.3f})"
        )


def main():
    args = parse_args()

    csv_path = args.data
    if not csv_path.is_file():
        raise SystemExit(f"Dataset file does not exist: {csv_path}")

    dataframe = pd.read_csv(csv_path)
    features, labels = prepare_dataset(dataframe, args.target)

    print("Duplicate audit:", duplicate_summary(features, labels))
    print("Unique disease-symptom combinations per disease:")
    for disease, count in disease_combination_counts(features, labels).items():
        print(f"  {disease}: {count}")
    print("Supporting-file alignment (metadata only):")
    print(inspect_supporting_files(csv_path.parent, labels.unique(), features.columns))

    # Split only after deduplicating normalized feature/label rows. Otherwise
    # copies of one example can land in both partitions and leak into evaluation.
    unique_features, unique_labels = deduplicate(features, labels)
    removed = len(features) - len(unique_features)
    print(
        f"Training on {len(unique_features)} unique rows "
        f"({removed} duplicate row(s) removed) across {unique_labels.nunique()} diseases."
    )

    candidates = build_candidates()
    requested = None if args.model == "auto" else MODEL_CHOICES[args.model]
    results = cross_validate_candidates(
        unique_features,
        unique_labels,
        folds=args.cv_folds,
        repeats=args.cv_repeats,
        seed=args.seed,
        candidates=candidates,
    )
    chosen = select_model(results, requested)
    best_mean = max(result["mean_accuracy"] for result in results.values())
    print(
        f"\nRepeated stratified cross-validation "
        f"({args.cv_folds}-fold x {args.cv_repeats} repeats, "
        f"seed {args.seed}):"
    )
    print_benchmark(results, chosen)
    print(f"Selected model: {chosen}")
    if requested is not None and results[chosen]["mean_accuracy"] < best_mean - 1e-12:
        print(f"Note: another candidate scored higher, but --model {args.model} was honored.")

    x_train, x_test, y_train, y_test = split_dataset(
        unique_features,
        unique_labels,
        test_size=args.test_size,
        seed=args.seed,
    )
    model, holdout = fit_and_evaluate(
        candidates[chosen], x_train, x_test, y_train, y_test
    )
    print(
        f"\nHoldout accuracy: {holdout['accuracy']:.3f} | "
        f"macro F1: {holdout['macro_f1']:.3f} | "
        f"rows: {holdout['test_rows']}"
    )
    print("Per-class holdout results:")
    per_class = pd.DataFrame(holdout["per_class"]).T
    per_class["support"] = per_class["support"].astype(int)
    print(per_class.round(3).to_string())

    print(
        "\nNote: these scores reflect separability of this dataset's unique "
        "symptom combinations, not clinical accuracy."
    )

    if args.evaluate_only:
        print("Evaluation only; no model bundle was saved.")
        return

    dataset_metadata = {
        "filename": csv_path.name,
        "sha256": sha256_file(csv_path),
        "csv_rows": int(len(dataframe)),
        "unique_rows": int(len(unique_features)),
        "duplicate_rows_removed": int(removed),
        "feature_count": int(unique_features.shape[1]),
        "class_count": int(unique_labels.nunique()),
        "target_column": TARGET_COLUMN,
    }
    bundle = build_bundle(
        model=model,
        model_name=chosen,
        feature_names=unique_features.columns,
        labels=sorted(unique_labels.unique()),
        metrics={"cross_validation": results, "holdout": holdout},
        dataset=dataset_metadata,
    )
    metrics_path = args.metrics or args.out.with_name("metrics.json")
    model_path, written_metrics = save_bundle(bundle, args.out, metrics_path)
    print(f"Saved model bundle to {model_path}")
    print(f"Saved metrics to {written_metrics}")


if __name__ == "__main__":
    main()
