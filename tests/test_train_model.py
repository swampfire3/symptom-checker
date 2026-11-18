import json
import subprocess
import sys
from pathlib import Path

import joblib
import pandas as pd

from src.modeling import validate_bundle
from src.preprocessing import SYMPTOM_COLUMNS

ROOT = Path(__file__).resolve().parents[1]


def write_smoke_dataset(path):
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
    rows = []
    for disease, combinations in patterns.items():
        for symptoms in combinations:
            padded = list(symptoms) + [""] * (len(SYMPTOM_COLUMNS) - len(symptoms))
            rows.append({"Disease": disease, **dict(zip(SYMPTOM_COLUMNS, padded))})
    pd.DataFrame(rows, columns=["Disease", *SYMPTOM_COLUMNS]).to_csv(path, index=False)


def test_training_cli_saves_bundle_and_metrics(tmp_path):
    dataset = tmp_path / "dataset.csv"
    model_path = tmp_path / "model.joblib"
    metrics_path = tmp_path / "metrics.json"
    write_smoke_dataset(dataset)

    completed = subprocess.run(
        [
            sys.executable,
            "train_model.py",
            "--data",
            str(dataset),
            "--model",
            "tree",
            "--cv-folds",
            "2",
            "--cv-repeats",
            "1",
            "--test-size",
            "0.25",
            "--out",
            str(model_path),
            "--metrics",
            str(metrics_path),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "Selected model: tree_baseline" in completed.stdout
    assert model_path.is_file()
    assert metrics_path.is_file()

    bundle = joblib.load(model_path)
    validate_bundle(bundle)
    assert bundle["model_name"] == "tree_baseline"
    assert bundle["dataset"]["unique_rows"] == 12
    assert bundle["dataset"]["csv_rows"] == 12
    assert bundle["labels"] == ["Alpha", "Beta", "Gamma"]

    document = json.loads(metrics_path.read_text())
    assert "model" not in document
    assert document["metrics"]["holdout"]["test_rows"] == 3
    assert "tree_baseline" in document["metrics"]["cross_validation"]
