"""Load a trained model bundle and create predictions."""

from pathlib import Path

import joblib
import pandas as pd

from src.preprocessing import normalize_symptom

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "symptom_disease_model.joblib"


class ModelNotReadyError(RuntimeError):
    """Raised when a model has not yet been trained or saved."""


class Predictor:
    def __init__(self, model_path=MODEL_PATH):
        self.model_path = Path(model_path)
        self._bundle = None

    def _load(self):
        if self._bundle is None:
            if not self.model_path.is_file():
                raise ModelNotReadyError(
                    "The prediction model is not available yet. Add a dataset to "
                    "data/raw/ and run python train_model.py first."
                )
            self._bundle = joblib.load(self.model_path)
        return self._bundle

    @property
    def symptoms(self):
        return self._load()["feature_names"]

    def predict(self, selected_symptoms):
        bundle = self._load()
        if not isinstance(selected_symptoms, (list, tuple, set)):
            raise ValueError("Symptoms must be supplied as a list.")
        normalized_selected = {normalize_symptom(symptom) for symptom in selected_symptoms}
        if "" in normalized_selected:
            raise ValueError("Symptoms cannot be blank.")
        valid = set(bundle["feature_names"])
        unknown = sorted(normalized_selected - valid)
        if unknown:
            raise ValueError("Unknown symptom(s): " + ", ".join(unknown))
        row = {name: int(name in normalized_selected) for name in bundle["feature_names"]}
        features = pd.DataFrame([row], columns=bundle["feature_names"])
        return str(bundle["model"].predict(features)[0])
