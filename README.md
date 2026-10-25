# Educational AI Symptom Checker

A student project that demonstrates a Decision Tree classifier behind a Flask interface. **It is not a medical device and must not be used to diagnose, treat, or rule out illness.**

## Requirements and setup

Use Python 3.10 or newer. From the project root:

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

## Add your dataset

Put the CSV you have selected in `data/raw/`. Check its source, license, and format before use. The starter trainer assumes one row per example, binary symptom columns (`0`/`1`), and a disease target column named `prognosis`. If your dataset differs, update `src/preprocessing.py` and pass the actual target column. Do not commit dataset files unless its license permits redistribution.

```bash
python train_model.py --target prognosis
# Or specify a CSV path:
python train_model.py --data data/raw/your_dataset.csv --target Disease
```

The script holds out a test subset, prints accuracy, a per-class report, and a confusion matrix, then saves the model bundle to `models/symptom_disease_model.joblib`. Review class counts and evaluation metrics; a small or imbalanced dataset can make the reported results misleading. Do not claim clinical accuracy.

## Run the web app

```bash
flask --app app run --debug
```

Open the local address Flask prints. The symptom list is loaded from the model's feature names, and form input is checked against that list. Debug mode is for local development only.

## Tests

```bash
pytest
```

## Project map

- `app.py`: Flask page and prediction endpoint.
- `src/preprocessing.py`: dataset checks and feature preparation.
- `src/predictor.py`: model loading and input conversion.
- `train_model.py`: split, train, evaluate, and save workflow.
- `templates/` and `static/`: browser interface and styling.
- `tests/`: initial checks for preprocessing and request validation.
- `data/raw/`: locally added source CSV; ignored by Git by default.

## Team workflow

Use a private GitHub repository, feature branches, and reviewed pull requests. Keep generated model files, virtual environments, secrets, and restricted data out of commits. Teammates should follow the same Python/dependency setup and document any dataset-specific preprocessing changes.

## Important limitations

This prototype only predicts labels represented by the chosen dataset. Dataset provenance and quality may be uncertain, and a Decision Tree may overfit. Real symptoms and medical conditions are more complex than a checkbox dataset. The output is an educational model result—not medical guidance. For real health concerns, contact a qualified professional; seek emergency care for urgent symptoms.
