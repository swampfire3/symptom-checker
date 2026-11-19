"""Integrity checks for the tracked dataset files.

The four CSVs are redistributed under CC BY-SA 4.0 and are a reformatted copy of
the upstream files. Their hashes are recorded in DATASET_ATTRIBUTION.md, which is
the single source of truth for this test. If a file is changed on purpose, update
the attribution table and the file in the same commit.
"""

import re
from pathlib import Path

import pytest

from src.modeling import sha256_file

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
ATTRIBUTION = ROOT / "DATASET_ATTRIBUTION.md"

# A table row shaped like: | `data/raw/dataset.csv` | 632,202 bytes | `abc...` |
HASH_ROW = re.compile(
    r"^\|\s*`(?P<path>data/raw/[^`]+)`\s*\|[^|]*\|\s*`(?P<digest>[0-9a-f]{64})`\s*\|$",
    re.MULTILINE,
)


def recorded_hashes():
    """Return the path-to-hash mapping published in the attribution document."""
    text = ATTRIBUTION.read_text(encoding="utf-8")
    return {
        match["path"]: match["digest"] for match in HASH_ROW.finditer(text)
    }


RECORDED = recorded_hashes()


def test_attribution_lists_every_tracked_dataset_file():
    tracked = {path.relative_to(ROOT).as_posix() for path in RAW_DIR.glob("*.csv")}
    assert set(RECORDED) == tracked, (
        "DATASET_ATTRIBUTION.md and data/raw/ disagree about which CSVs are tracked"
    )


@pytest.mark.parametrize("relative_path", sorted(RECORDED))
def test_tracked_dataset_file_matches_recorded_hash(relative_path):
    assert sha256_file(ROOT / relative_path) == RECORDED[relative_path], (
        f"{relative_path} no longer matches DATASET_ATTRIBUTION.md; "
        "update the recorded hash deliberately if the change is intended"
    )
