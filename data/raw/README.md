# Raw dataset files

These four CSVs are third-party content and are **not** original to this
project. They are redistributed under the terms below.

## Attribution

- Dataset: **Disease Symptom Prediction**
- Source: <https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset>
- Uploader: Kaggle user `itachi9604`
- License: **CC BY-SA 4.0** — <https://creativecommons.org/licenses/by-sa/4.0/>

## Changes made to the upstream files

- Header rows were added to `dataset.csv`, `symptom_Description.csv`, and
  `symptom_precaution.csv`.
- `Symptom-severity.csv` was renamed from `Symptom_severity.csv`, given a
  header row, and includes an extra `prognosis,5` entry.
- Contents are otherwise unchanged.

`DATASET_ATTRIBUTION.md` records the file hashes and the full license
obligations. Tests in `tests/test_dataset_files.py` fail if a tracked file
changes without the recorded hash being updated.
