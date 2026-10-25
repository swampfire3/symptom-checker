"""Train and evaluate a Decision Tree from a CSV in data/raw/."""

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_PATH, help="Primary dataset CSV (default: data/raw/dataset.csv)")
    parser.add_argument("--target", default=TARGET_COLUMN, choices=[TARGET_COLUMN], help="Disease-label column")
    args = parser.parse_args()

    csv_path = args.data
    if not csv_path.is_file():
        parser.error(f"Dataset file does not exist: {csv_path}")

    dataframe = pd.read_csv(csv_path)
    features, labels = prepare_dataset(dataframe, args.target)
    duplicates = duplicate_summary(features, labels)
    print("Duplicate audit:", duplicates)
    print("Unique disease-symptom combinations per disease:")
    for disease, count in disease_combination_counts(features, labels).items():
        print(f"  {disease}: {count}")
    print("Supporting-file alignment (metadata only):")
    print(inspect_supporting_files(RAW_DIR, labels.unique(), features.columns))

    # Split only after deduplicating normalized feature/label rows. Otherwise
    # copies of one example can land in both partitions and leak into evaluation.
    unique_rows = features.copy()
    unique_rows[TARGET_COLUMN] = labels.to_numpy()
    unique_rows = unique_rows.drop_duplicates()
    labels = unique_rows.pop(TARGET_COLUMN).rename("y")
    features = unique_rows
    stratify = labels if labels.value_counts().min() >= 2 else None
    if stratify is None:
        parser.error("Some diseases have fewer than two unique combinations; a stratified holdout is not reliable.")
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=stratify,
    )
    model = DecisionTreeClassifier(max_depth=12, random_state=42, class_weight="balanced")
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    print(f"Accuracy: {accuracy_score(y_test, predictions):.3f}")
    print("Classification report:\n", classification_report(y_test, predictions, zero_division=0))
    print("Confusion matrix (label order):", sorted(labels.unique()))
    print(confusion_matrix(y_test, predictions, labels=sorted(labels.unique())))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_names": list(features.columns)}, MODEL_PATH)
    print(f"Saved model bundle to {MODEL_PATH}")


if __name__ == "__main__":
    main()
