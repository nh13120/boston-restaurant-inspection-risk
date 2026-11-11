# Data dictionary: `data/processed/model_table.parquet`

One row per routine inspection of an "Eating & Drinking" establishment in Boston, 2016-01-02 to 2026-11-10. Re-inspections are excluded. 58,564 rows.

`data/processed/inspections_full.parquet` has the same columns, plus `licstatus`, for all 161,962 inspections since April 2006, re-inspections included. Rows before 2016 serve only as history for the features.

Source: [Food Establishment Inspections](https://data.boston.gov/dataset/food-establishment-inspections), Analyze Boston, Open Data Commons Public Domain Dedication and License (PDDL).

## Identifiers and outcome

| Column | Type | Meaning |
|---|---|---|
| `inspection_id` | text | `<licenseno>_<YYYYMMDD>`. Unique. |
| `licenseno` | text | License number of the establishment. |
| `date` | date | Inspection date in Boston time (the source stores UTC). |
| `year`, `split` | int, text | Calendar year and its split: `train` (2016 to 2022), `val` (2023), `test` (2024 onward). In the full table, rows before 2016 are `history`. |
| `label_fail` | 0/1 | The outcome to predict. 1 if the inspection carried any fail code (see `DECISIONS.md` D1.2). A 0 means "no follow-up required", not "no violations": it includes `HE_Filed` visits, which usually list one or two minor violations. |
| `result`, `result_codes` | text | `fail` or `pass`, and the original codes joined with a vertical bar. |

## Features (known before the inspection)

"Earlier inspections" always means earlier inspections of the same license, re-inspections included unless the name says "routine".

| Column | Type | Meaning |
|---|---|---|
| `prior_n` | int | Number of earlier inspections on record. The record starts in April 2006, so the count is understated for restaurants that opened before then. |
| `prior_n_3y` | int | Number of earlier inspections in the previous three years. |
| `is_first` | 0/1 | 1 if this is the license's first inspection on record. |
| `prior_fail_rate` | float | Share of earlier inspections that failed. Missing for a first inspection. |
| `prior_routine_fail_rate` | float | Share of earlier routine inspections that failed. |
| `last_fail` | 0/1 | Whether the previous inspection failed. Usually 0, because the previous visit is often a passed re-inspection. |
| `last_n_viol` | int | Violations found at the previous inspection. |
| `last_n_lvl3` | int | Level 3 (most serious, `***`) violations found at the previous inspection. |
| `last_routine_fail` | 0/1 | Whether the last routine inspection failed. |
| `last_routine_n_viol` | int | Violations found at the last routine inspection. |
| `last_routine_n_lvl3` | int | Level 3 violations found at the last routine inspection. |
| `days_since_last` | int | Days since the previous inspection. |
| `license_age_days` | int | Days since the license's first inspection on record, or since its issue date when that is earlier (D2.2). |
| `month` | int | Calendar month of the inspection, 1 to 12. |
| `is_takeout` | 0/1 | 1 if the establishment type is "Eating & Drinking w/ Take Out". |
| `neighborhood` | text | Neighborhood from the `city` column. "Boston" covers downtown. |
| `zip` | text | Zip code. |
| `prior_comments` | text | Inspector comments from the previous inspection, personal details masked. Empty for a first inspection or when the previous visit had no comments. |

`is_takeout`, `neighborhood` and `zip` are the values on the pull date. The source keeps one value per license.

## Outcome-side columns (never used as features)

These describe the inspection being predicted. They are kept to build the next row's history and for checks. `src/models.py` refuses them as features.

| Column | Type | Meaning |
|---|---|---|
| `n_viol`, `n_lvl1`, `n_lvl2`, `n_lvl3` | int | Violations found at this inspection, in total and by level (`*`, `**`, `***`). |
| `n_rechecked_ok` | int | Earlier violations re-checked at this inspection and found corrected. |
| `comments` | text | Inspector comments from this inspection, personal details masked. |
| `is_reinspection` | 0/1 | 1 if the previous inspection failed 30 days ago or less. Always 0 in the model table. |

## Descriptive columns

| Column | Type | Meaning |
|---|---|---|
| `businessname` | text | Business name. Used only in the priority list. |
| `descript` | text | Establishment type. |
| `license_issue_date` | date | Current license issue date in the snapshot. |
| `licstatus` | text | Full table only. License status on the pull date (`Active`, `Inactive` or `Deleted`). It describes today, so it is never a feature. |

The owner-name columns of the source (`legalowner`, `namelast`, `namefirst`) were never downloaded.
