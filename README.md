# How much does bad input data change a model's predictions?

The experiment trains two credit-default models on clean data. It then corrupts the test data with ten realistic pipeline errors, such as stale feeds, unit mix-ups, swapped columns and silent zero-fills. For each error it measures two things: how much the model's predictions change, and whether common data checks would have caught the corrupted batch.

**Headline finding:** the most damaging error was a stale repayment status, where this month's `PAY_0` was copied from last month's `PAY_2`. With 20% of rows corrupted, it flipped the predicted class for 36% of the changed rows under gradient boosting and 27% under logistic regression. Yet AUC dropped by less than 0.01 on average. Basic checks (schema, nulls, range, allowed codes), the PSI distribution check and the prediction-drift check all missed it in every repeat. Only a month-over-month consistency check caught it, and only from 20% severity upwards.

## Reproduce it

You need Python 3.15.2 (the version used for the published run) and Jupyter or another notebook runner.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install jupyter                # if you don't already have a notebook runner
jupyter notebook bad_input_data_experiment.ipynb
```

Run all cells. The notebook downloads the dataset from the UCI repository, so you need an internet connection. A full run takes a few minutes. It writes its results to `outputs/` and overwrites the files committed there.

The first code cell runs `pip install` without version pins. If you have already installed `requirements.txt`, skip that cell so the pinned versions stay in place. The notebook prints the library versions it ran with, so you can confirm them.

All randomness is seeded (`SEED = 42`, and each repeat derives its own seed from it), so a run with the same library versions should reproduce the committed numbers. Other versions may shift results slightly.

## Data

Yeh, I. (2009). *Default of Credit Card Clients* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C55S3H

The dataset is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). It isn't committed here; the notebook fetches it with `ucimlrepo` (dataset id 350). To use a local copy instead, set `DATA_SOURCE = "file"` and `LOCAL_PATH` in the configuration cell.

The documentation lists education codes 1–4 and marital status codes 1–3, but the data contains other values. The notebook maps those to the "other" category (4 and 3 respectively) before training.

## What's in each file

| File | What it is |
|---|---|
| `bad_input_data_experiment.ipynb` | The full experiment, executed with 10 repeats and its outputs saved, so you can read every number without running anything. |
| `requirements.txt` | Pinned library versions from the published run. |
| `outputs/results.csv` | One row per corruption × severity × repeat × model (600 rows). This is the raw data every other table and figure comes from. |
| `outputs/summary_20pct.csv` | At 20% severity, averaged over repeats: AUC drop, flip rate, and the share of repeats in which each check fired. |
| `outputs/flip_direction_20pct.csv` | At 20% severity: which way predictions flipped (to low risk or to high risk), and how many actual defaulters were newly missed or caught. |
| `outputs/detection_by_severity.csv` | For gradient boosting: the share of repeats in which each check fired, at 5%, 20% and 50% severity. |
| `outputs/impact_vs_detection.png` | The notebook's chart of flip rate against detection tier at 20% severity. |

### Columns in `results.csv`

- `auc_drop`: clean AUC minus corrupted AUC (higher is worse).
- `flip_rate_corrupted`: share of *changed* rows whose predicted class flipped at the 0.5 threshold.
- `mean_abs_change`: average absolute change in predicted probability across all rows.
- `flips_to_low_risk` / `flips_to_high_risk`: counts of changed rows that flipped from default to no default, and the reverse.
- `defaulters_newly_missed` / `defaulters_newly_caught`: the same counts, restricted to customers who actually defaulted.
- `caught_by_basic`, `caught_by_psi`, `caught_by_consistency`, `caught_by_prediction_drift`: whether each check tier flagged the batch.
- `prediction_psi`: PSI of the model's output probabilities against the reference distribution.
- `checks_flagged`: which checks fired, and on which columns.

`rows_changed` can be lower than the severity implies. For example, `PAY_0` already equals `PAY_2` in about 77% of rows, so copying one into the other changes only some of the rows picked.

## Settings and scope

**Models.** `HistGradientBoostingClassifier` (handles missing values natively) and logistic regression (median imputation, scaling, one-hot encoded categoricals). Both are trained once on a 70% stratified split of clean data. Only the 30% test set is corrupted.

**Corruptions.** Ten, each applied to 5%, 20% and 50% of test rows, with **10 repeats** per combination using different random seeds.

**Checks and thresholds.** All checks use the training data as the reference and see the whole corrupted batch. They don't know which rows were changed.

| Check | Flags a batch when |
|---|---|
| Null rate | a column's null rate rises more than **1 percentage point** above training |
| Range | more than **1%** of a column's values fall outside the training min–max |
| Allowed values | a categorical or status column contains a code not seen in training |
| Distribution (PSI) | a column's PSI against training exceeds **0.2** (10 quantile bins) |
| Month-over-month consistency | the share of rows where a month equals, or exceeds, the previous month differs from training by more than **4 standard errors** |
| Prediction drift | PSI of the model's output probabilities against out-of-fold training predictions exceeds **0.1** |

A prediction counts as flipped when it crosses the **0.5** class threshold.

The clean test set passes every check (asserted in the notebook), so none of the checks raise false alarms on clean data at these thresholds.

**Limits.** This is one dataset, two models, ten hand-picked corruptions and one class threshold. The consistency check covers only three column pairs. Prediction drift can stay quiet when errors push predictions in both directions and cancel out in the overall distribution. Treat the results as evidence about this setup, not as general rates.

## Credit

The approach builds on **JENGA**, a framework for studying how data errors affect ML model predictions: Schelter, S., Rukat, T. and Biessmann, F. (2021). *JENGA: A Framework to Study the Impact of Data Errors on the Predictions of Machine Learning Models.* EDBT 2021. https://github.com/schelterlabs/jenga

## Licence

Code: MIT, see [LICENSE](LICENSE). Dataset: CC BY 4.0, credited above.
