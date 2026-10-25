"""Flask web application for the educational symptom checker."""

from flask import Flask, jsonify, render_template, request

from src.predictor import ModelNotReadyError, Predictor

app = Flask(__name__)
predictor = Predictor()


@app.get("/")
def index():
    try:
        symptoms = predictor.symptoms
        model_available = True
    except ModelNotReadyError:
        symptoms = []
        model_available = False
    return render_template(
        "index.html", symptoms=symptoms, model_available=model_available
    )


@app.post("/predict")
def predict():
    data = request.get_json(silent=True) if request.is_json else request.form
    selected = data.get("symptoms", []) if data else []
    if isinstance(selected, str):
        selected = [selected]
    if not isinstance(selected, list) or not selected:
        message = "Select at least one symptom."
        return (jsonify(error=message), 400) if request.is_json else (message, 400)

    try:
        disease = predictor.predict(selected)
    except ModelNotReadyError as exc:
        return (jsonify(error=str(exc)), 503) if request.is_json else (str(exc), 503)
    except ValueError as exc:
        return (jsonify(error=str(exc)), 400) if request.is_json else (str(exc), 400)

    if request.is_json:
        return jsonify(prediction=disease)
    return render_template(
        "index.html",
        symptoms=predictor.symptoms,
        model_available=True,
        selected_symptoms=selected,
        prediction=disease,
    )


if __name__ == "__main__":
    app.run(debug=True)
