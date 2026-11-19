"""Load a trained model bundle and create predictions."""

import pickle
from pathlib import Path

import joblib
import pandas as pd

from src.modeling import validate_bundle
from src.preprocessing import normalize_symptom

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "symptom_disease_model.joblib"


class ModelNotReadyError(RuntimeError):
    """Raised when a model has not yet been trained or saved."""


class Predictor:
    def __init__(self, model_path=MODEL_PATH):
        self.model_path = Path(model_path)
        self._bundle = None
        self._bundle_stamp = None

    def _load(self):
        try:
            stamp = self.model_path.stat().st_mtime_ns
        except FileNotFoundError:
            self._bundle = None
            self._bundle_stamp = None
            raise ModelNotReadyError(
                "The prediction model is not available yet. Add a dataset to "
                "data/raw/ and run python train_model.py first."
            ) from None
        if self._bundle is not None and stamp == self._bundle_stamp:
            return self._bundle
        try:
            bundle = joblib.load(self.model_path)
            validate_bundle(bundle)
        except (
            OSError,
            EOFError,
            KeyError,
            ImportError,
            pickle.UnpicklingError,
            ValueError,
        ) as exc:
            raise ModelNotReadyError(
                f"The saved model cannot be used: {exc}"
            ) from exc
        self._bundle = bundle
        self._bundle_stamp = stamp
        return bundle

    @property
    def symptoms(self):
        return self._load()["feature_names"]

    @property
    def labels(self):
        return list(self._load()["labels"])

    @property
    def model_name(self):
        return self._load()["model_name"]

    @staticmethod
    def _build_features(selected_symptoms, feature_names):
        if not isinstance(selected_symptoms, (list, tuple, set)):
            raise ValueError("Symptoms must be supplied as a list.")
        normalized_selected = {
            normalize_symptom(symptom) for symptom in selected_symptoms
        }
        if "" in normalized_selected:
            raise ValueError("Symptoms cannot be blank.")
        valid = set(feature_names)
        unknown = sorted(normalized_selected - valid)
        if unknown:
            raise ValueError("Unknown symptom(s): " + ", ".join(unknown))
        row = {name: int(name in normalized_selected) for name in feature_names}
        return pd.DataFrame([row], columns=feature_names)

    def predict(self, selected_symptoms):
        bundle = self._load()
        features = self._build_features(selected_symptoms, bundle["feature_names"])
        return str(bundle["model"].predict(features)[0])

    def top_predictions(self, selected_symptoms, k=3):
        """Return the most likely labels, with probabilities when available."""
        if not isinstance(k, int) or k < 1:
            raise ValueError("k must be a positive integer.")
        bundle = self._load()
        features = self._build_features(selected_symptoms, bundle["feature_names"])
        model = bundle["model"]
        if not hasattr(model, "predict_proba") or not hasattr(model, "classes_"):
            return [
                {
                    "label": str(model.predict(features)[0]),
                    "probability": None,
                }
            ]

        probabilities = model.predict_proba(features)[0]
        ranked = sorted(
            zip(model.classes_, probabilities),
            key=lambda item: item[1],
            reverse=True,
        )
        return [
            {"label": str(label), "probability": float(probability)}
            for label, probability in ranked[:k]
            if probability > 0
        ]
