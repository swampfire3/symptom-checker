# Care-suggestion reliability

The care layer names a specialty only when the model is confident. These tables
are the evidence behind its two thresholds, and they are reproducible with
`scripts/reliability_eval.py`.

## Method

- Source: the 304 unique disease/symptom patterns in `data/raw/dataset.csv`.
- Inputs: random subsets of 1, 2, and 3 symptoms, three samples per size per
  pattern, seed 7 — **2,672 inputs**.
- Model: the saved random-forest bundle, top-3 probabilities.
- The inputs are deliberately sparse. A real user selecting five or more
  symptoms will clear the gate far more often than these numbers suggest.

## Accuracy by input size

| Symptoms selected | Share of inputs | Top-1 accuracy |
|---:|---:|---:|
| 1 | 34.1% | 37.1% |
| 2 | 34.1% | 60.0% |
| 3 | 31.7% | 68.4% |

Confidence is tracking correctness, which is what makes a gate useful.

## Choosing the gate

| Policy | Coverage | Accuracy when it answers | Confident but wrong |
|---|---:|---:|---:|
| No gate | 100% | 54.8% | 1,207 |
| Top score ≥ 0.6 | 10.8% | 97.6% | 7 |
| Top-1 vs top-2 margin ≥ 0.3 only | 25.9% | 88.4% | 80 |
| Top score ≥ 0.6 **and** margin ≥ 0.15 | 10.8% | 97.6% | 7 |

A margin alone is unsafe: it answers far more often while being right less
often. Once the absolute threshold is applied, the margin adds nothing on this
dataset, so the shipped policy uses the threshold only.

`SPECIALTY_THRESHOLD` is therefore **0.6**.

## Emergency routing

Emergency labels use a lower bar than routine specialties, because a missed
emergency is worse than an unnecessary caution.

| Emergency threshold | Recall on true emergencies | False alarms / non-emergency inputs |
|---:|---:|---:|
| 0.30 | 83.8% | 3.7% |
| 0.40 | 71.6% | 1.5% |
| 0.50 | 48.6% | 0.5% |

`EMERGENCY_THRESHOLD` is **0.30**. The false-alarm column is the deliberate
cost: the system over-triages rather than staying quiet.

## Selected policy

With specialty 0.60 and emergency 0.30 on the 2,672 inputs:

- **249 specialty suggestions (9.3%)**, of which **249 were correct**.
- **157 emergency flags**, catching **62 of 74** true emergencies, with **95
  false alarms** (3.7% of non-emergency inputs).
- **2,266 abstentions (84.8%)**.

Of the 288 inputs scoring at or above 0.60, 39 had an emergency label as their
top prediction and were routed to the emergency outcome (32 correct, 7 wrong);
the other 249 became specialty suggestions, all correct. The 157 emergency
flags also include lower-confidence flags between 0.30 and 0.60, so they are
not a subset of those 288.

Abstention is the normal outcome for sparse input; that is the intended
behaviour, not a failure.

## Caveats

- **These estimates are in-distribution.** The inputs are symptom subsets of
  patterns the model was trained on, so they measure how the shipped gate
  behaves on the shipped model, not how it generalises to unseen conditions.
  The pattern holdout in `train_model.py` is the generalisation check.
- **The thresholds describe the random-forest bundle.** Logistic regression,
  the decision tree, and Jaccard k-NN produce differently scaled scores; rerun
  this harness and recalibrate before using the gate with them.
- The mapping from condition to specialty is editorial. It was written for this
  project and has not been reviewed by clinicians.
- These thresholds are product decisions about how often the system should
  speak, not clinical cut-offs.
- The inputs are synthetic subsets of a constructed dataset. No real patient
  data was used, and the numbers do not transfer to clinical settings.

## Reproduce

```bash
python train_model.py
python scripts/reliability_eval.py
python scripts/reliability_eval.py --samples-per-pattern 5 --seed 42 --json models/reliability.json
```
