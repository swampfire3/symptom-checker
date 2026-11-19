# Convenience targets for the documented local workflow.
# Run `make` or `make help` to list them.

PYTHON ?= python
VENV ?= .venv
BIN := $(VENV)/bin

.PHONY: help setup train run test clean

help:
	@echo "setup  Create .venv and install the pinned dependencies"
	@echo "train  Train the model bundle into models/"
	@echo "run    Start the Flask development server"
	@echo "test   Run the test suite"
	@echo "clean  Remove generated model files and caches"

setup:
	$(PYTHON) -m venv $(VENV)
	$(BIN)/python -m pip install --upgrade pip
	$(BIN)/python -m pip install -r requirements.lock

train:
	$(BIN)/python train_model.py

run:
	$(BIN)/flask --app app run --debug

test:
	$(BIN)/python -m pytest -q

clean:
	rm -f models/symptom_disease_model.joblib models/metrics.json models/data_audit.json
	rm -rf .pytest_cache __pycache__ src/__pycache__ tests/__pycache__
