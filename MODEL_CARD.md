# Model card: educational symptom checker

## Overview

- **Purpose:** classroom demonstration of a machine-learning workflow behind a web interface.
- **Not a medical device.** The output is not a diagnosis, triage decision, or treatment recommendation.
- **Default model:** random forest with 100 trees, balanced class weights, and random seed 42.
- **Input:** 131 binary symptom features derived from the dataset's text symptom columns.
- **Output:** one of 41 disease labels. The web interface shows the three highest-scoring labels with their model scores.

## Intended use

Intended for demonstrating data checks, model comparison, evaluation, artifact saving, and a small Flask interface. It must not be used by patients, clinicians, or anyone else to make health decisions. No real-patient validation has been performed.

## Data

- Four CSVs whose structure matches the Kaggle dataset **Disease Symptom Prediction**, uploaded by `itachi9604`.
- The Kaggle listing's **CC BY-SA 4.0** is the project's confirmed license for these files. Attribution, hashes, and change notes are in `DATASET_ATTRIBUTION.md`; the raw files remain untracked by default.
- The primary CSV has 4,920 rows and 41 disease classes. Only **304 disease/symptom combinations are unique**; 4,616 rows are duplicates.
- Each class is represented by 5–10 unique symptom combinations.
- Preprocessing trims disease labels, normalizes symptom text, treats empty cells as padding, and creates binary features. Duplicate rows are removed before any split. Source files are not rewritten.

## Evaluation

The model is evaluated only on the 304 unique rows, never on duplicate copies.

- **Cross-validation:** 5-fold stratified cross-validation repeated 5 times, seed 42.
- **Holdout:** 20% stratified holdout (61 rows), seed 42, after deduplication.

| Model | Cross-validation accuracy |
|---|---:|
| Random forest (default) | 1.000 ± 0.000 |
| Logistic regression | 1.000 ± 0.000 |
| KNN, Jaccard, k=1 | 1.000 ± 0.000 |
| Decision tree baseline, depth 12 | 0.238 ± 0.040 |

Holdout accuracy is 1.000 and macro F1 is 1.000 for the random forest with these settings.

These scores mean every unique symptom combination in this dataset maps cleanly to one disease label. That is a property of the dataset, not evidence of real-world diagnostic performance. The scores are not a claim of clinical validity.

## Care suggestions

The web interface adds an educational care-suggestion panel below the ranked labels. It is an editorial lookup table plus a confidence gate, not a referral service.

- Every one of the 41 dataset labels maps to a specialty, for example Migraine → Neurology.
- `Heart attack` and `Paralysis (brain hemorrhage)` are routed to an urgent-care message instead of a routine specialty. Emergency routing deliberately uses a lower score bar than routine specialties, so it over-flags; the panel says so.
- A specialty is named only when the top score is at least 0.60. Emergency labels use 0.30. Below the gate the panel says there is not enough confidence and names no specialty.
- The thresholds were measured on the random-forest bundle only. For any other model build the interface withholds care guidance rather than apply cut-offs that do not describe that model's scores.
- Every state is labelled as an educational mapping, not a referral or medical advice.

On 2,672 deliberately sparse inputs the gate abstained 84.8% of the time, and all 249 specialty suggestions it made matched the true dataset label. Emergency flags caught 62 of 74 true emergencies with 95 false alarms. These numbers describe gate behaviour on this constructed dataset, not clinical performance. See `docs/reliability.md`.

The mapping and the interface copy were written for this project and have not been reviewed by clinicians.

## Limitations

- The dataset appears highly templated: 94% of rows are duplicates, and each unique symptom set belongs to a single disease.
- Real patients can share symptoms across conditions; this dataset contains no such ambiguity.
- Severity weights, symptom duration, demographics, vital signs, and test results are not used as model inputs.
- The model has no "unknown" or "not enough information" outcome.
- Random-forest probabilities are not calibrated for medical decision-making.
- Class labels and symptom names come from the dataset and may contain typos.
- There is no external validation, fairness analysis, or clinical review.
- The dataset is third-party content under CC BY-SA 4.0, so any redistributed copy must keep the attribution, license link, change notes, and ShareAlike terms recorded in `DATASET_ATTRIBUTION.md`.

## Safety and legal notes

- Keep the medical disclaimer visible wherever predictions are shown.
- Do not deploy this as a public health tool.
- Do not collect or store real user symptom data for this demo.
- Do not describe the model as accurate, validated, or diagnostic.

## Retraining

```bash
python train_model.py
python train_model.py --model auto
python train_model.py --evaluate-only
pytest
```

Generated artifacts are written to `models/`, which is intentionally untracked. The metrics document includes the dataset SHA-256, row counts, model names, library versions, and both evaluation reports.
