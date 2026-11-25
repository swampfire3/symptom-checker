"""Flask web application for the educational symptom checker."""

from flask import Flask, jsonify, render_template, request

from src.care import build_care
from src.predictor import ModelNotReadyError, Predictor

app = Flask(__name__)
predictor = Predictor()


def _render_index(**context):
    try:
        symptoms = predictor.symptoms
        model_available = True
    except ModelNotReadyError:
        symptoms = []
        model_available = False
    return render_template(
        "index.html",
        symptoms=symptoms,
        model_available=model_available,
        **context,
    )


@app.get("/")
def index():
    return _render_index()


@app.post("/predict")
def predict():
    if request.is_json:
        data = request.get_json(silent=True) or {}
        selected = data.get("symptoms", [])
    else:
        selected = request.form.getlist("symptoms")
    if isinstance(selected, str):
        selected = [selected]
    if not isinstance(selected, list) or not selected:
        message = "Select at least one symptom."
        if request.is_json:
            return jsonify(error=message), 400
        return _render_index(error=message), 400

    try:
        predictions = predictor.top_predictions(selected, k=3)
        care = build_care(predictions, predictor.model_name)
    except ModelNotReadyError as exc:
        if request.is_json:
            return jsonify(error=str(exc)), 503
        return _render_index(error=str(exc)), 503
    except ValueError as exc:
        if request.is_json:
            return jsonify(error=str(exc)), 400
        return _render_index(error=str(exc), selected_symptoms=selected), 400

    if request.is_json:
        return jsonify(
            prediction=predictions[0]["label"] if predictions else None,
            predictions=predictions,
            care=care,
        )
    return _render_index(
        selected_symptoms=selected,
        predictions=predictions,
        care=care,
    )


if __name__ == "__main__":
    app.run(debug=True)
