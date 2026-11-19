"""Integrity checks for the tracked dataset files.

The four CSVs are redistributed under CC BY-SA 4.0 and are a reformatted copy of
the upstream files. Their hashes are recorded in DATASET_ATTRIBUTION.md so the
exact version used for training stays reproducible. If a file is changed on
purpose, update both the attribution table and this mapping in the same commit.
"""

import pytest

from src.modeling import sha256_file

RAW_DIR = "data/raw"

EXPECTED_SHA256 = {
    "dataset.csv": "ebbd391c4ba4d64f57a00eb3d0a55f0ca9b920b0c5de6b9af4234e72519c9618",
    "Symptom-severity.csv": "cb5a84b5ebde81bb18738b7e7733bcbc5abdc667b4376893fb1774d8c444bc8d",
    "symptom_Description.csv": "e0581662e815509596707ebb5b31efb44539562b451ecef8c1246aa713b7d832",
    "symptom_precaution.csv": "49371294708232b928f68fc60e9837e5cab8b90d450b7bd90f7305795bb6d311",
}


@pytest.mark.parametrize("filename,expected", sorted(EXPECTED_SHA256.items()))
def test_tracked_dataset_file_matches_recorded_hash(filename, expected):
    path = f"{RAW_DIR}/{filename}"
    assert sha256_file(path) == expected, (
        f"{path} no longer matches DATASET_ATTRIBUTION.md; "
        "update the recorded hash deliberately if the change is intended"
    )
