# Which Boston restaurants should be inspected first?

Boston's Inspectional Services Department (ISD) makes about 6,000 routine restaurant inspections a year, and 59% end in a failure. This repo ranks upcoming inspections by risk of failing, so planners can schedule the riskiest restaurants first each quarter and find problems sooner. We could not confirm how ISD orders visits today: "random order" is our assumption.

## Data

[Food Establishment Inspections](https://data.boston.gov/dataset/food-establishment-inspections), Analyze Boston, PDDL license. Snapshot of 2026-11-11 from the CKAN SQL API: "Eating & Drinking" inspections with `resultdttm < '2026-11-11'` (785,315 violation rows since 2006; query in `docs/pull_metadata.json`). Owner names are never downloaded. `data/` is git-ignored and must stay unpublished: raw comments contain names and emails.

## Setup and run

```bash
python3.13 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt     # all exact versions: requirements-lock.txt
```

macOS, before the notebooks: LightGBM needs OpenMP. Run `brew install libomp` or the two-line fix in `docs/setup_macos.md`.

```bash
cd notebooks
for nb in 01_data 02_baselines 03_models 04_results; do
  jupyter nbconvert --to notebook --execute --inplace $nb.ipynb
done
```

About 6 minutes on an M-series laptop. A first run also pulls the data (about 10 minutes) and downloads a 90 MB text encoder. To match our numbers exactly, get `inspections_snapshot.parquet` from the repo owner and place it in `data/raw/`.

## Results on the test years (2024-01-02 to 2026-11-10, 17,171 inspections)

Hit rate is the share of the riskiest 20% of inspections that end in a failure. Settings were frozen on 2023 before the test years were scored.

| Method | Hit rate (95% CI) | Failures in the first 20% | AUC |
|---|---|---|---|
| B0 Random order (assumed status quo) | 59.0% (58 to 60) | 20.0% | 0.500 |
| B1 Past failure rate (business rule) | 80.3% (78 to 82) | 27.2% | 0.698 |
| B2 Logistic regression on history | 87.0% (85 to 88) | 29.5% | 0.736 |
| M1 LightGBM on history (final model) | 88.1% (86 to 89) | 29.9% | 0.748 |
| M2 LightGBM on history + comment embeddings | 88.8% (87 to 90) | 30.1% | 0.747 |

ISD makes every routine inspection sooner or later, so the yearly total (about 3,544 failed inspections) is the same in any order. Ranking changes when problems are found. In the first 20% of visits (1,201 a year), M1 finds 349 more failed inspections than a random order (CI 333 to 364), about 320 more than the order ISD actually followed, and 40 to 95 more than a hand rule. Those visits hold 40% of all serious (level 3) violations, against 20% in random order. Within a quarter, a visit that ends in a failure comes 1.1 weeks sooner on average, and one that ends in a pass 1.6 weeks later.

We recommend ranking by inspection history (a hand rule already reaches 80%), using the model without zip code, and returning sooner to restaurants that pass but rank high: 60% of them failed their next routine inspection, against 43% of other passes (`results/after_a_pass.csv`).

Figures are in `results/figures/`. `results/priority_list.csv` ranks 3,163 active restaurants; read `results/priority_list_note.md` first.

## Modeling choices

- One row per routine inspection, label 1 = any fail code. Visits within 30 days of a failure are re-inspections, dropped after the features are built.
- Features use only earlier inspections of the same license (`shift(1)`, cross-checked by hand), with history since 2006.
- Time split: train 2016 to 2022, validation 2023, test 2024 onward. Final fits use 2016 to 2023.
- Tabular plus deep learning: M2 adds the previous inspection's comments, embedded with a frozen `all-MiniLM-L6-v2` (mean-pooled, 32 PCA components).
- Intervals are bootstraps over whole restaurants. M1 was fixed as the final model on validation, where text added nothing.

## Limitations

- The baselines are assumptions, and 20% is an illustrative capacity. By year the gain is 346, 407 and 288 against random order and 334, 393 and 234 against ISD's actual order (`results/results_by_year.csv`).
- The algorithm and the text matter little. M1 leads logistic regression by 1.1 points (CI 0.3 to 1.9), and the text model leads M1 by 0.7 (CI 0.1 to 1.4): comments mostly restate the violation count M1 already uses.
- Zip code is an input, so neighborhoods with similar fail rates get different shares of early visits (`results/neighborhood_table.csv`). Dropping it cost 0.5 points on validation (within noise), but the shipped model and list still use it. Scores are estimates, not findings about a business, and name masking in comments is best effort.

## Repo map

`notebooks/` 01 data, 02 baselines, 03 models, 04 results. `src/` logic. `preds/` scores. `results/` tables, figures, priority list. `docs/` decisions, data dictionary, frozen config.
