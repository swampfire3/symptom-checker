import os

import joblib
import pytest

import app as app_module
from app import app, predictor
from src.modeling import build_bundle, save_bundle
from src.predictor import ModelNotReadyError, Predictor


class CaptureModel:
    def __init__(self, prediction="Example disease"):
        self.prediction = prediction
        self.received = None

    def predict(self, features):
        self.received = features.copy()
        return [self.prediction]


class ProbabilityModel(CaptureModel):
    classes_ = ["Alpha", "Beta", "Gamma"]

    def predict_proba(self, features):
        self.received = features.copy()
        return [[0.05, 0.75, 0.20]]


class StubPredictor:
    symptoms = ["cough", "high_fever"]

    def __init__(self, predictions=None):
        self.predictions = predictions if predictions is not None else [
            {"label": "Alpha", "probability": 0.75},
            {"label": "Beta", "probability": 0.15},
            {"label": "Gamma", "probability": 0.10},
        ]
        self.selected = None

    def top_predictions(self, selected_symptoms, k=3):
        self.selected = list(selected_symptoms)
        return self.predictions[:k]


def write_bundle(path, model, feature_names=("cough", "high_fever"), labels=("Example disease", "Other")):
    bundle = build_bundle(
        model,
        "test_model",
        list(feature_names),
        list(labels),
        {},
        {"filename": "test.csv"},
        trained_at="2026-01-01T00:00:00+00:00",
    )
    joblib.dump(bundle, path)
    return path


def test_home_page_explains_missing_model(tmp_path, monkeypatch):
    app.config["TESTING"] = True
    monkeypatch.setattr(predictor, "model_path", tmp_path / "missing.joblib")
    monkeypatch.setattr(predictor, "_bundle", None)

    response = app.test_client().get("/")
    assert response.status_code == 200
    assert b"Model not trained yet" in response.data
    assert b"Medical disclaimer" in response.data


def test_prediction_requires_symptom():
    app.config["TESTING"] = True
    response = app.test_client().post("/predict", json={"symptoms": []})
    assert response.status_code == 400


def test_form_prediction_renders_ranked_scores(monkeypatch):
    app.config["TESTING"] = True
    monkeypatch.setattr(app_module, "predictor", StubPredictor())

    response = app.test_client().post("/predict", data={"symptoms": ["cough"]})

    assert response.status_code == 200
    assert b"Top dataset labels" in response.data
    assert b"Alpha" in response.data
    assert b"75.0%" in response.data
    assert b"Beta" in response.data
    assert b"15.0%" in response.data
    assert b"confidence" in response.data


def test_form_prediction_passes_every_selected_symptom(monkeypatch):
    app.config["TESTING"] = True
    stub = StubPredictor()
    monkeypatch.setattr(app_module, "predictor", stub)

    response = app.test_client().post(
        "/predict", data={"symptoms": ["cough", "high_fever"]}
    )

    assert response.status_code == 200
    assert stub.selected == ["cough", "high_fever"]


def test_json_prediction_returns_top_three(monkeypatch):
    app.config["TESTING"] = True
    monkeypatch.setattr(app_module, "predictor", StubPredictor())

    response = app.test_client().post("/predict", json={"symptoms": ["cough"]})

    assert response.status_code == 200
    assert response.get_json() == {
        "prediction": "Alpha",
        "predictions": [
            {"label": "Alpha", "probability": 0.75},
            {"label": "Beta", "probability": 0.15},
            {"label": "Gamma", "probability": 0.10},
        ],
    }


def test_form_result_handles_missing_probability(monkeypatch):
    app.config["TESTING"] = True
    monkeypatch.setattr(
        app_module,
        "predictor",
        StubPredictor([{"label": "Alpha", "probability": None}]),
    )

    response = app.test_client().post("/predict", data={"symptoms": ["cough"]})

    assert response.status_code == 200
    assert b"Alpha" in response.data
    assert b"prediction-score" not in response.data


def test_form_validation_renders_inline_error(monkeypatch):
    app.config["TESTING"] = True
    monkeypatch.setattr(app_module, "predictor", StubPredictor())

    response = app.test_client().post("/predict", data={})

    assert response.status_code == 400
    assert b"Select at least one symptom." in response.data
    assert b"Medical disclaimer" in response.data


def test_predictor_uses_training_normalization_and_feature_order(tmp_path):
    model = CaptureModel()
    predictor = Predictor(write_bundle(tmp_path / "test-model.joblib", model))

    assert predictor.predict([" High-fever "]) == "Example disease"
    assert predictor.symptoms == ["cough", "high_fever"]
    assert predictor.labels == ["Example disease", "Other"]
    assert predictor.model_name == "test_model"
    loaded_model = predictor._bundle["model"]
    assert list(loaded_model.received.columns) == ["cough", "high_fever"]
    assert loaded_model.received.iloc[0].to_dict() == {"cough": 0, "high_fever": 1}


def test_predictor_reloads_when_the_bundle_file_changes(tmp_path):
    model_path = tmp_path / "test-model.joblib"
    predictor = Predictor(model_path)

    first = build_bundle(
        CaptureModel("Alpha"),
        "test_model",
        ["cough"],
        ["Alpha"],
        {},
        {"filename": "test.csv"},
    )
    save_bundle(first, model_path)
    assert predictor.predict(["cough"]) == "Alpha"

    second = build_bundle(
        CaptureModel("Beta"),
        "test_model",
        ["cough"],
        ["Beta"],
        {},
        {"filename": "test.csv"},
    )
    save_bundle(second, model_path)
    # A coarse filesystem could reuse the same timestamp, so make it distinct.
    stamp = model_path.stat().st_mtime_ns
    os.utime(model_path, ns=(stamp + 1_000_000_000, stamp + 1_000_000_000))

    assert predictor.predict(["cough"]) == "Beta"


def test_predictor_rejects_unknown_symptoms(tmp_path):
    model_path = write_bundle(
        tmp_path / "test-model.joblib",
        CaptureModel(),
        feature_names=("cough",),
        labels=("Example disease",),
    )
    with pytest.raises(ValueError, match="Unknown symptom"):
        Predictor(model_path).predict(["unknown symptom"])


def test_top_predictions_returns_sorted_probabilities(tmp_path):
    predictor = Predictor(
        write_bundle(
            tmp_path / "test-model.joblib",
            ProbabilityModel(),
            feature_names=("cough",),
            labels=("Alpha", "Beta", "Gamma"),
        )
    )
    assert predictor.top_predictions(["cough"], k=2) == [
        {"label": "Beta", "probability": 0.75},
        {"label": "Gamma", "probability": 0.20},
    ]


def test_top_predictions_falls_back_without_probabilities(tmp_path):
    predictor = Predictor(
        write_bundle(
            tmp_path / "test-model.joblib",
            CaptureModel(),
            feature_names=("cough",),
            labels=("Example disease",),
        )
    )
    assert predictor.top_predictions(["cough"]) == [
        {"label": "Example disease", "probability": None}
    ]


def test_top_predictions_rejects_invalid_k(tmp_path):
    predictor = Predictor(
        write_bundle(
            tmp_path / "test-model.joblib",
            CaptureModel(),
            feature_names=("cough",),
            labels=("Example disease",),
        )
    )
    with pytest.raises(ValueError, match="positive integer"):
        predictor.top_predictions(["cough"], k=0)


def test_predictor_rejects_legacy_bundle(tmp_path):
    model_path = tmp_path / "legacy.joblib"
    joblib.dump({"model": CaptureModel(), "feature_names": ["cough"]}, model_path)
    with pytest.raises(ModelNotReadyError, match="missing"):
        Predictor(model_path).predict(["cough"])


def test_predictor_rejects_wrong_schema_version(tmp_path):
    bundle = build_bundle(
        CaptureModel(),
        "test_model",
        ["cough"],
        ["Example disease"],
        {},
        {"filename": "test.csv"},
    )
    bundle["schema_version"] = 99
    model_path = tmp_path / "wrong-schema.joblib"
    joblib.dump(bundle, model_path)
    with pytest.raises(ModelNotReadyError, match="Unsupported model bundle schema"):
        Predictor(model_path).predict(["cough"])


def test_predictor_rejects_unreadable_bundle(tmp_path):
    model_path = tmp_path / "broken.joblib"
    model_path.write_text("not a joblib file")
    with pytest.raises(ModelNotReadyError, match="cannot be used"):
        Predictor(model_path).predict(["cough"])
