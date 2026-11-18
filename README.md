# Educational AI Symptom Checker

[![Tests](https://github.com/swampfire3/symptom-checker/actions/workflows/tests.yml/badge.svg)](https://github.com/swampfire3/symptom-checker/actions/workflows/tests.yml)

A student project that demonstrates a tree-based classifier behind a Flask interface. Training compares several models and saves a random-forest bundle by default. **It is not a medical device and must not be used to diagnose, treat, or rule out illness.**

## Requirements and setup

Use Python 3.12 or newer (CI tests Python 3.12 and 3.14). From the project root:

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

## Dataset format

Put the CSV you have selected in `data/raw/`. The trainer expects a wide text format:

- one `Disease` target column;
- symptom columns named `Symptom_1` through `Symptom_17`;
- one symptom name per cell (for example `high_fever` or `skin rash`);
- empty cells used as padding for rows with fewer than 17 symptoms.

Symptom text is normalized to lowercase underscore-separated names, empty padding is ignored, and duplicate disease/symptom rows are removed before evaluation. Source files stay unchanged. Dataset provenance and redistribution rights are unresolved; see `DATASET_ATTRIBUTION.md`. Do not commit dataset files.

## Train and evaluate

```bash
python train_model.py
```

The script runs a duplicate and supporting-file audit, compares four candidates with repeated stratified cross-validation, evaluates the selected model on a held-out subset, and saves:

- `models/symptom_disease_model.joblib` — the model bundle;
- `models/metrics.json` — cross-validation and holdout metrics.

Candidates are a single decision tree (the teaching baseline), a random forest (the default), logistic regression, and Jaccard K-nearest-neighbours. Useful options:

```bash
python train_model.py --model auto          # choose the highest cross-validation score
python train_model.py --model tree          # train the single-tree baseline
python train_model.py --cv-folds 5 --cv-repeats 5 --seed 42
python train_model.py --evaluate-only       # run metrics without saving a bundle
python train_model.py --data data/raw/other.csv
```

### Current benchmark

The source CSV contains 4,920 rows but only **304 unique disease/symptom combinations** because 4,616 rows are duplicates. On those 304 unique rows, with 5-fold cross-validation repeated 5 times (seed 42):

| Model | Cross-validation accuracy |
|---|---:|
| Random forest (default) | 1.000 ± 0.000 |
| Logistic regression | 1.000 ± 0.000 |
| KNN, Jaccard, k=1 | 1.000 ± 0.000 |
| Decision tree baseline, depth 12 | 0.238 ± 0.040 |

The 20% holdout (61 rows) also scores 1.000 accuracy and 1.000 macro F1 with seed 42. These numbers show that the unique symptom sets in this dataset are separable. **They do not show clinical accuracy.** See `MODEL_CARD.md` for evaluation details and limitations.

## Run the web app

```bash
flask --app app run --debug
```

Open the local address Flask prints. The symptom list is loaded from the model's feature names, and form input is checked against that list. Results rank the model's three highest-scoring labels and show each score; these are uncalibrated model outputs, not medical probabilities. Validation errors stay on the page so selections can be corrected. Debug mode is for local development only.

For a JSON response:

```bash
curl -s http://127.0.0.1:5000/predict \
  -H 'Content-Type: application/json' \
  -d '{"symptoms":["high_fever","cough"]}'
```

The response keeps `prediction` as the top label for compatibility and adds `predictions`, a ranked list of up to three `{"label", "probability"}` entries.

## Tests

```bash
pytest
```

GitHub Actions runs the same test suite on Python 3.12 and 3.14 for every push to `master` and every pull request.

## Project map

- `app.py`: Flask page and prediction endpoint.
- `src/preprocessing.py`: dataset checks, normalization, and duplicate audits.
- `src/modeling.py`: candidate models, cross-validation, holdout evaluation, and bundle validation.
- `src/predictor.py`: model loading, input conversion, and top-3 probability support.
- `train_model.py`: audit, benchmark, evaluate, and save workflow.
- `templates/` and `static/`: browser interface and styling.
- `tests/`: preprocessing, modeling, bundle, CLI, and request-validation checks.
- `data/raw/`: locally added source CSVs; ignored by Git.
- `models/`: generated bundle and metrics; ignored by Git.
- `.github/workflows/tests.yml`: GitHub Actions test workflow.
- `CONTRIBUTING.md`: setup, branch, pull-request, and commit conventions.
- `MODEL_CARD.md`: intended use, evaluation, and limitations.

## Team workflow

Use feature branches and pull requests. `master` is protected: changes must pass the GitHub Actions test workflow and resolve review conversations before merging. Keep generated model files, virtual environments, secrets, and restricted data out of commits. Teammates should follow the same Python/dependency setup and document any dataset-specific preprocessing changes.

See `CONTRIBUTING.md` for setup, branch, pull-request, and commit conventions.

## Important limitations

This prototype only predicts labels represented by the chosen dataset. Dataset provenance and quality may be uncertain, the data is highly duplicated, and the unique symptom patterns are unrealistically separable. Real symptoms and medical conditions are more complex than a checkbox dataset. The output is an educational model result—not medical guidance. For real health concerns, contact a qualified professional; seek emergency care for urgent symptoms.
