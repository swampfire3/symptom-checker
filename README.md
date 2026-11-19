# Educational AI Symptom Checker

[![Tests](https://github.com/swampfire3/symptom-checker/actions/workflows/tests.yml/badge.svg)](https://github.com/swampfire3/symptom-checker/actions/workflows/tests.yml)

A student project that demonstrates a tree-based classifier behind a Flask interface. Training compares several models and saves a random-forest bundle by default. **It is not a medical device and must not be used to diagnose, treat, or rule out illness.**

## Quickstart

```bash
git clone https://github.com/swampfire3/symptom-checker.git
cd symptom-checker
make setup     # create .venv and install the pinned dependencies
make train     # write models/symptom_disease_model.joblib
make run       # serve http://127.0.0.1:5000
```

`make test` runs the suite and `make help` lists the targets. On systems without
`make`, run the equivalent commands in the next two sections. The dataset is
included in `data/raw/`; the trained model is generated locally and is not
committed.

## Requirements and setup

Use Python 3.12 or newer (CI tests Python 3.12 and 3.14). `requirements.lock`
pins the exact versions that were tested together, and `requirements.txt`
declares the supported ranges. From the project root:

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
python -m pip install -r requirements.lock
```

## Dataset format

The four CSVs are included in `data/raw/`; see `data/raw/README.md` for attribution. The trainer expects a wide text format:

- one `Disease` target column;
- symptom columns named `Symptom_1` through `Symptom_17`;
- one symptom name per cell (for example `high_fever` or `skin rash`);
- empty cells used as padding for rows with fewer than 17 symptoms.

Symptom text is normalized to lowercase underscore-separated names, empty padding is ignored, and duplicate disease/symptom rows are removed before evaluation. Source files stay unchanged. The dataset is third-party content used under **CC BY-SA 4.0**; attribution, file hashes, change notes, and ShareAlike obligations are recorded in `DATASET_ATTRIBUTION.md`. `tests/test_dataset_files.py` fails if a tracked file changes without its recorded hash being updated.

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

Train the model first (see above); without a bundle the page shows a setup message instead of predictions.

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
- `data/raw/`: tracked source CSVs and their attribution note.
- `models/`: generated bundle and metrics; ignored by Git.
- `.github/workflows/tests.yml`: GitHub Actions test workflow.
- `CONTRIBUTING.md`: setup, branch, pull-request, and commit conventions.
- `MODEL_CARD.md`: intended use, evaluation, and limitations.

## Team workflow

Use feature branches and pull requests. `master` is protected: changes must pass the GitHub Actions test workflow and resolve review conversations before merging. Keep generated model files, virtual environments, secrets, and unvetted data out of commits. Teammates should follow the same Python/dependency setup and document any dataset-specific preprocessing changes.

See `CONTRIBUTING.md` for setup, branch, pull-request, and commit conventions.

## Licensing

- **Code** — MIT; see `LICENSE`. This covers the Python source, tests, templates, static assets, CI configuration, and `requirements.txt`.
- **Documentation** — CC BY-NC-SA 4.0; see `LICENSE-DOCS`. This covers the Markdown documentation and the prose inside it.
- **Dataset** — third-party content under CC BY-SA 4.0, not owned by this project; see `DATASET_ATTRIBUTION.md` for attribution and obligations.

The documentation license is noncommercial. It does not make the model clinically valid, and the disclaimer below still applies to any use.

## Important limitations

This prototype only predicts labels represented by the chosen dataset. The dataset is small and highly duplicated, and its unique symptom patterns are unrealistically separable. Real symptoms and medical conditions are more complex than a checkbox dataset. The output is an educational model result—not medical guidance. For real health concerns, contact a qualified professional; seek emergency care for urgent symptoms.
