# Contributing

## Setup

1. Use Python 3.12 or newer.
2. Create and activate a virtual environment.
3. Install dependencies from `requirements.txt`.
4. Run `pytest` before changing anything to confirm the starting state.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
pytest
```

## Branch and pull request workflow

- Create a short-lived branch from `master`, for example `feature/clear-selection` or `fix/form-validation`.
- Keep each branch focused on one change.
- Run the full test suite locally before opening a pull request.
- Open a pull request with a short summary and the checks completed.
- Wait for the GitHub Actions `Tests` workflow to pass before merging.
- Merge with a squash merge or a clear merge commit, then delete the branch.
- Keep `master` in a working state.

`master` is protected: changes must go through a pull request, the `Tests`
checks must pass, and conversations must be resolved before merging. Direct
pushes and force pushes are blocked, including for administrators.

## What not to commit

- `data/raw/` or `data/processed/` files: excluded by default to keep the repository small. The dataset is third-party CC BY-SA 4.0 content, so any committed copy must keep the attribution and change notes in `DATASET_ATTRIBUTION.md`.
- `models/` artifacts: regenerate them with `python train_model.py`.
- `.venv/`, `__pycache__/`, `.pytest_cache/`, coverage output, or editor files.
- Secrets, tokens, `.env` files, or personal credentials.

## Training and evaluation changes

- Do not claim clinical accuracy. Metrics describe this dataset's separability only.
- Keep deduplication before any train/test split.
- Update `README.md` and `MODEL_CARD.md` when the model, metrics, commands, or limitations change.
- Keep the medical disclaimer visible in every interface that shows a prediction.
- Preserve the dataset attribution notes in `DATASET_ATTRIBUTION.md`.

## Commit messages

Use a short imperative subject line, for example:

```text
Add clear-selection button
Fix duplicate-row leakage in evaluation
Update model card benchmark table
```

Add a body when the reason or trade-off is not obvious from the subject.

## Licensing of contributions

By opening a pull request you agree that your contribution is licensed under
the terms already in the repository:

- source code, tests, templates, and configuration under **MIT** (`LICENSE`);
- documentation under **CC BY-NC-SA 4.0** (`LICENSE-DOCS`).

Do not paste in code, text, datasets, or images that you cannot license on
those terms. Third-party material needs its own attribution and license note.

## Reporting problems

Open an issue with:

- what you expected;
- what happened;
- the command or page you used;
- relevant output, with secrets and personal data removed.
