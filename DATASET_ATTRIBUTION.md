# Dataset source, attribution, and license review

## Source identified for the four CSVs

The filenames and dataset structure match the Kaggle dataset titled **Disease
Symptom Prediction**, uploaded under the Kaggle handle **`itachi9604`**:

- Dataset page: <https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset>
- Uploader handle: `itachi9604` (a separate legal or preferred name was not
  independently verified)
- Kaggle's displayed license: **CC BY-SA 4.0**, not CC BY 4.0.
- License text: <https://creativecommons.org/licenses/by-sa/4.0/>

The four local files reviewed against this source are:

| File | Size |
|---|---:|
| `data/raw/dataset.csv` | 632,202 bytes |
| `data/raw/Symptom-severity.csv` | 2,332 bytes |
| `data/raw/symptom_Description.csv` | 11,025 bytes |
| `data/raw/symptom_precaution.csv` | 3,488 bytes |

## Attribution and license conditions

If the dataset can be redistributed under the Kaggle listing's stated license,
attribute it as **Disease Symptom Prediction, Kaggle uploader `itachi9604`**,
link to the source page above, link to the CC BY-SA 4.0 license, and identify
changes made. Do not imply that the uploader or Kaggle endorses this project.
CC BY-SA requires appropriate credit, a license link, an indication of
modifications, and ShareAlike licensing for adaptations. Preserve any source
notices and comply with the license's complete terms.

## Coverage is not fully established; keep source files untracked for now

The Kaggle page reports CC BY-SA 4.0 for its dataset listing, and its file
collection is the identified source for these CSVs. However, that listing alone
does not establish that the uploader had redistribution rights to every
third-party description, precaution, or other incorporated content. The
available local files do not include per-file provenance or rights statements.
There is also a GitHub repository linked to the dataset whose repository
metadata displays GPL-3.0; that does not resolve the licensing of these CSVs
and should not be treated as replacing the Kaggle file license. The difference
between the reported CC BY 4.0 and the Kaggle page's CC BY-SA 4.0 also needs to
be resolved before distribution.

Therefore this project does **not** assert that all four CSVs are cleared for
redistribution. They remain excluded by `.gitignore`. Before tracking them,
verify the exact source/version for each file, the uploader's rights in any
third-party material, any underlying source acknowledgements, and whether the
Kaggle license selection covers the complete bundle. If that cannot be
confirmed, obtain permission or use a dataset with clear file-level rights.

## Project changes to the data

The original CSVs are left unchanged. The project preprocessing code trims
disease-label whitespace; normalizes symptom text to lowercase,
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
