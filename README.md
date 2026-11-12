# Which Boston restaurants should be inspected first?

Boston's Inspectional Services Department (ISD) makes about 6,000 routine restaurant inspections a year, and 59% end in a failure. This repo ranks upcoming inspections by risk of failing, so planners can schedule the riskiest restaurants first each quarter and find serious violations sooner. We could not confirm how ISD orders visits today: "random order" is our assumption.

## Data

[Food Establishment Inspections](https://data.boston.gov/dataset/food-establishment-inspections), Analyze Boston, PDDL license, pulled on 2026-11-11 through the CKAN SQL API (query in `docs/pull_metadata.json`). Owner names are never downloaded. `data/` is git-ignored and must stay unpublished: raw comments contain names and emails.

## Setup and run

```bash
python3.13 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt     # macOS: then follow docs/setup_macos.md (LightGBM needs OpenMP)
cd notebooks
for nb in 01_data 02_baselines 03_models 04_results; do
  jupyter nbconvert --to notebook --execute --inplace $nb.ipynb
done
```

About 6 minutes on an M-series laptop, plus a first data pull (10 minutes) and a 90 MB encoder download. For our exact numbers (verified on Apple Silicon), get `inspections_snapshot.parquet` from the repo owner and place it in `data/raw/`.

## Results on the test years (2024 to Nov 2026, 17,171 inspections)

Each method ranks the inspections, and we look at the first 20%. Settings were frozen on 2023 before the test years were scored.

| Method | Visits that find a failure (95% CI) | Share of all failures found | Share of serious violations found |
|---|---|---|---|
| B0 Random order (assumed status quo) | 59.0% (58 to 60) | 20.0% | 20% |
| B1 Past failure rate (business rule) | 80.3% (78 to 82) | 27.2% | 33% |
| B2 Logistic regression on history | 87.0% (85 to 88) | 29.5% | 39% |
| M1 LightGBM on history (final model) | 88.1% (86 to 89) | 29.9% | 40% |
| M2 LightGBM on history + comment embeddings | 88.8% (87 to 90) | 30.1% | 40% |

ISD makes every routine inspection sooner or later, so the yearly total (about 3,544 failed inspections) is the same in any order. Ranking changes when problems are found. In the first 20% of visits (1,201 a year), M1 finds 349 more failed inspections than a random order (CI 333 to 364) and 42 to 94 more than a hand rule. Within a quarter, a visit that ends in a failure comes 1.1 weeks sooner (0.8 to 0.9 with a hand rule), and one that ends in a pass 1.6 weeks later.

We recommend ranking by inspection history, where a hand rule already brings most of the gain, and returning sooner to restaurants that pass but rank high: 60% failed their next routine inspection, against 43% of other passes. Zip code and neighborhood are model inputs: Dorchester gets 46% of its inspections among the first visits and Roxbury 19%, at the same fail rate. Dropping them cost 0.5 points on 2023, within noise. We suggest ISD does so, but the numbers and `results/priority_list.csv` here include location.

## Modeling choices

- One row per routine inspection, label 1 = any fail code. "Routine" is inferred: visits within 30 days of a failure are re-inspections, dropped after the features are built.
- History features use only earlier inspections of the same license (records since 2006). Location and type are today's values.
- Time split: train 2016 to 2022, validation 2023, test 2024 onward. Final fits use 2016 to 2023.
- Deep learning on text: M2 adds the previous inspection's comments, embedded with a frozen `all-MiniLM-L6-v2` (32 PCA components).
- Intervals are bootstraps over whole restaurants. M1 was fixed as the final model on validation, where text added nothing.

## Limitations

- The status quo is assumed, and 20% is an illustrative capacity. M1's lead over the rule is 7.8 points overall and 3.3 to 3.7 on two robustness slices. Hit rates fell from 90.1% in 2024 to 84.9% in 2026.
- The algorithm and the text add little: M1 leads logistic regression by 1.1 points (CI 0.3 to 1.9), and the text model leads M1 by 0.7 (CI 0.1 to 1.4).
- Scores are estimates, not findings about a business. The priority list names businesses, so read `results/priority_list_note.md` first. Name masking in comments is best effort.

## Repo map

`notebooks/` 01 data, 02 baselines, 03 models, 04 results. `src/` logic. `preds/` scores. `results/` tables, 8 figures, priority list. `docs/` decisions, data dictionary, frozen config.
