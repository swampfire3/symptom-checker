# Dataset source, attribution, and license

## Status: license confirmed by the project owner

On 2026-11-18 the project owner confirmed that the Kaggle listing for this
dataset states **CC BY-SA 4.0** and adopted that license for the four CSVs used
here. This file records the attribution and the obligations that come with it;
it is a record of the project's decision, not an independent legal opinion.

- Dataset title: **Disease Symptom Prediction**
- Source: <https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset>
- Uploader: Kaggle handle **`itachi9604`** (no separate legal name was verified)
- License: **Creative Commons Attribution-ShareAlike 4.0 International**
  - Summary: <https://creativecommons.org/licenses/by-sa/4.0/>
  - Legal code: <https://creativecommons.org/licenses/by-sa/4.0/legalcode>

| File | Size | SHA-256 |
|---|---:|---|
| `data/raw/dataset.csv` | 632,202 bytes | `ebbd391c4ba4d64f57a00eb3d0a55f0ca9b920b0c5de6b9af4234e72519c9618` |
| `data/raw/Symptom-severity.csv` | 2,332 bytes | `cb5a84b5ebde81bb18738b7e7733bcbc5abdc667b4376893fb1774d8c444bc8d` |
| `data/raw/symptom_Description.csv` | 11,025 bytes | `e0581662e815509596707ebb5b31efb44539562b451ecef8c1246aa713b7d832` |
| `data/raw/symptom_precaution.csv` | 3,488 bytes | `49371294708232b928f68fc60e9837e5cab8b90d450b7bd90f7305795bb6d311` |

### Suggested attribution notice

When redistributing these files, keep a notice equivalent to:

> "Disease Symptom Prediction" by Kaggle user `itachi9604`, licensed under
> CC BY-SA 4.0 (<https://creativecommons.org/licenses/by-sa/4.0/>). Files
> reformatted as described in the project's `DATASET_ATTRIBUTION.md`.

Do not imply that the uploader, Kaggle, or Creative Commons endorses this
project.

## Obligations under CC BY-SA 4.0

- **Attribution:** credit the uploader, name the license, and link to it.
- **Changes:** state that the files were modified and describe how.
- **ShareAlike:** distribute adaptations of the dataset under CC BY-SA 4.0 or a
  compatible license.
- **No additional restrictions:** do not add legal or technical terms that
  prevent others from exercising the license rights.
- **No warranty:** the data is provided as-is; the licenses give no guarantee of
  accuracy, and none is claimed here.

## Redistribution and version notes

The four CSVs are tracked in Git under `data/raw/` so a fresh clone can train
and run the demo. They are redistributed under CC BY-SA 4.0 with the attribution
above, and `data/raw/README.md` repeats the notice for readers who open only that
directory. `tests/test_dataset_files.py` checks the hashes, so an unintended
edit fails the test suite; change the hash here and in that test together only
when the change is deliberate.

The local files are a reformatted copy, not a byte-identical copy, of the CSVs
in the uploader's GitHub repository. They add header rows, and
`Symptom-severity.csv` includes an extra `prognosis,5` row. The hashes above pin
the exact version this project trains on, so results can be reproduced and the
differences can be described accurately.

Two upstream repositories linked to the dataset were also inspected. One
displays GPL-3.0 metadata but contains no CSV files; another contains the CSVs
but declares no license. Neither changes the Kaggle listing's CC BY-SA 4.0
terms, but they are the reason the exact source of each file is recorded here
rather than inferred from repository metadata.

## Project changes to the data

The original CSVs are left unchanged on disk. The project preprocessing code
trims disease-label whitespace; normalizes symptom text to lowercase,
underscore-separated names; treats empty symptom cells as padding; converts
the symptom list to binary features; and provides duplicate and alignment
audits. Duplicate removal is applied before training and evaluation; it
does not rewrite these source files. No differently spelled disease labels
are silently merged. Severity weights, descriptions, and precautions remain
separate supporting metadata, not model training features.

The training workflow removes 4,616 duplicate rows before splitting and
evaluates on the 304 remaining unique disease/symptom combinations. Metrics
about this dataset therefore describe those unique combinations, not the raw
row count.
