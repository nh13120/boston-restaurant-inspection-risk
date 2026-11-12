# Spec checklist: 15.071 Project 1

Source: `Project.pdf`, sections "From a problem to an AI solution", "Project 1: data, modeling, and evaluation" and "Deliverables". Each item is a yes/no check. The "Where" column says which file answers it.

## The four graded questions

| # | Check | Where |
|---|---|---|
| Q1 | Is the problem stated, with a target stakeholder, the current process or decision, and the opportunity for improvement? | README |
| Q2 | Is appropriate data acquired and an AI solution designed for it? | notebooks 01 and 03 |
| Q3 | Is the solution evaluated rigorously against appropriate benchmarks? | notebooks 02 to 04, `results/results.csv` |
| Q4 | Is technical performance translated into managerial or societal impact? | notebook 04, `results/impact.csv` |

## Problem and prediction task

| # | Check | Where |
|---|---|---|
| P1 | Is there a named stakeholder (Boston ISD) and a decision they make (which restaurants to inspect first)? | README |
| P2 | Is the prediction task well defined: one unit of prediction, a stated outcome, stated inputs? | README, `docs/data_dictionary.md` |
| P3 | Are the inputs limited to what ISD would know before the inspection (no leakage)? | `src/features.py`, `docs/REVIEW_LOG.md` (Gate 1) |
| P4 | Is the scope focused enough to analyze in a few weeks, and is a simple model preferred when it performs as well? | `docs/DECISIONS.md` |

## Data

| # | Check | Where |
|---|---|---|
| D1 | Is the dataset public, with its source, license and pull date recorded? | README, `data/raw/pull_metadata.json` |
| D2 | Is the repo free of confidential, proprietary or personally identifiable information (no owner names in any saved file)? | `src/data_pull.py`, `docs/REVIEW_LOG.md` (Gate 1) |
| D3 | Is the data preparation documented and reproducible (cleaning, collapsing to one row per inspection, label, features)? | notebook 01, `src/features.py` |
| D4 | Is there a data dictionary? | `docs/data_dictionary.md` |

## Modeling and validation

| # | Check | Where |
|---|---|---|
| M1 | Does the project combine machine learning on tabular data with deep learning on unstructured data? | notebook 03 (LightGBM + BERT embeddings) |
| M2 | Is model selection rigorous: a time-based split, tuning on validation only, the test period scored once? | notebook 03, `docs/FROZEN_CONFIG.json`, `docs/REVIEW_LOG.md` (Gate 2) |
| M3 | Is there a predictive baseline (logistic regression)? | notebook 02 |
| M4 | Is there a business-rule benchmark (rank by past failure rate) and a status-quo benchmark (random order)? | notebook 02 |
| M5 | Are alternative modeling approaches compared (tabular only vs. tabular plus text)? | notebook 03 |
| M6 | Do reported metrics carry uncertainty (bootstrap confidence intervals)? | `results/results.csv` |
| M7 | Is the work reproducible (fixed seed, relative paths, notebooks pass Restart & Run All)? | `docs/REVIEW_LOG.md` (Gate 4) |

## Impact and communication

| # | Check | Where |
|---|---|---|
| I1 | Is impact quantified in policy terms (extra failures found per year, inspections saved) at a fixed capacity? | `results/impact.csv` |
| I2 | Does every impact number trace to the data or a logged assumption, with no invented dollar figures? | `src/impact.py`, `docs/REVIEW_LOG.md` (Gate 3) |
| I3 | Are there visualizations that show performance and impact? | `results/figures/` |
| I4 | Are limitations stated honestly? | README |

## Deliverables

| # | Check | Where |
|---|---|---|
| R1 | Is there a GitHub repository with executable code? | this repo |
| R2 | Are there instructions for reproducing the results? | README |
| R3 | Is there a one-page README addressed to a data scientist or ML engineer at the target organization? | README |
| R4 | Do the outputs support the 5 to 10 slide deck and the two-minute video (figures, results table, impact numbers)? These two are made separately. | `results/` |
