import joblib
import pytest

from app import app
from src.predictor import Predictor


class CaptureModel:
    def __init__(self, prediction="Example disease"):
        self.prediction = prediction
        self.received = None

    def predict(self, features):
        self.received = features.copy()
        return [self.prediction]


def test_home_page_explains_missing_model():
    app.config["TESTING"] = True
    response = app.test_client().get("/")
    assert response.status_code == 200
    assert b"Medical disclaimer" in response.data


def test_prediction_requires_symptom():
    app.config["TESTING"] = True
    response = app.test_client().post("/predict", json={"symptoms": []})
    assert response.status_code == 400


def test_predictor_uses_training_normalization_and_feature_order(tmp_path):
    model = CaptureModel()
    model_path = tmp_path / "test-model.joblib"
    joblib.dump({"model": model, "feature_names": ["cough", "high_fever"]}, model_path)
    predictor = Predictor(model_path)

    assert predictor.predict([" High-fever "]) == "Example disease"
    loaded_model = predictor._bundle["model"]
    assert list(loaded_model.received.columns) == ["cough", "high_fever"]
    assert loaded_model.received.iloc[0].to_dict() == {"cough": 0, "high_fever": 1}


def test_predictor_rejects_unknown_symptoms(tmp_path):
    model_path = tmp_path / "test-model.joblib"
    joblib.dump({"model": CaptureModel(), "feature_names": ["cough"]}, model_path)
    with pytest.raises(ValueError, match="Unknown symptom"):
        Predictor(model_path).predict(["unknown symptom"])
