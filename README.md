# Which Boston restaurants should be inspected first?

Boston's Inspectional Services Department (ISD) makes about 6,000 routine restaurant inspections a year with a fixed staff, and 59% of them end in a failure. This repo ranks inspections by their risk of failing, so that the first visits go where failures are most likely and problems are found sooner.

## Data

[Food Establishment Inspections](https://data.boston.gov/dataset/food-establishment-inspections), Analyze Boston, PDDL license. Snapshot of 2026-11-11 from the CKAN SQL API: every "Eating & Drinking" inspection with `resultdttm < '2026-11-11'` (785,315 violation rows since 2006; query and fingerprint in `docs/pull_metadata.json`). Owner names are never downloaded. `data/` is git-ignored and must stay unpublished: raw comments contain names and emails.

## Setup and run

Python 3.13. About 6 minutes on an Apple-silicon laptop, plus 10 for the first data pull and a 90 MB encoder download.

```bash
python3.13 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cd notebooks
for nb in 01_data 02_baselines 03_models 04_results; do
  jupyter nbconvert --to notebook --execute --inplace $nb.ipynb
done
```

macOS: LightGBM needs OpenMP. Run `brew install libomp`, or point LightGBM at scikit-learn's copy:

```bash
LIB=$(python -c "import importlib.util as u, pathlib as p; print(p.Path(u.find_spec('lightgbm').origin).parent/'lib/lib_lightgbm.dylib')")
install_name_tool -add_rpath "@loader_path/../../sklearn/.dylibs" "$LIB" && codesign --force --sign - "$LIB"
```

Notebook 01 pulls the data when `data/raw/` is empty and warns if the city has edited rows since our snapshot. Exact numbers need that snapshot.

## Results on the test years (2024-01-02 to 2026-11-10, 17,171 inspections)

Hit rate is the share of the riskiest 20% of inspections that end in a failure. Settings were frozen on the validation year (2023) before the test years were scored.

| Method | Hit rate (95% CI) | Share of failures in the first 20% | AUC |
|---|---|---|---|
| B0 Random order (assumed status quo) | 59.0% (58 to 60) | 20.0% | 0.500 |
| B1 Past failure rate (business rule) | 80.3% (78 to 82) | 27.2% | 0.698 |
| B2 Logistic regression on history | 87.0% (85 to 88) | 29.5% | 0.736 |
| M1 LightGBM on history (final model) | 88.1% (86 to 89) | 29.9% | 0.748 |
| M2 LightGBM on history + comment embeddings | 88.8% (87 to 90) | 30.1% | 0.747 |

ISD makes every routine inspection sooner or later, so the yearly total of about 3,540 failed inspections is the same in any order. Ranking changes when they are found. In the first 20% of visits (1,201 a year), M1 finds 349 more failed inspections than a random order (CI 333 to 364; 346, 407 and 288 by year) and 94 more than the rule (79 to 109). Reordered within each quarter, a visit that ends in a failure comes 1.1 weeks sooner on average and one that ends in a pass 1.6 weeks later (the gain is 0.4 weeks within months, 4.4 within years). `results/priority_list.csv` scores the next inspection of 3,163 active restaurants.

## Modeling choices

- One row per routine inspection. Label 1 = any fail code; 0 = no follow-up required (minor violations possible). Visits within 30 days of a failure are re-inspections, dropped after the features are built.
- Features use only earlier inspections of the same license (`shift(1)`, cross-checked by hand), with history since 2006. The model table starts in 2016.
- Time split: train 2016 to 2022, validation 2023, test 2024 onward. Final fits use 2016 to 2023.
- Text: the previous inspection's comments, embedded with a frozen `all-MiniLM-L6-v2`, mean-pooled, reduced to 32 PCA components.
- Intervals are bootstraps over whole restaurants. They do not cover change from year to year (2026 is the weakest year). M1 was fixed as the final model on validation.

## Limitations

- "Random order" as today's schedule is an assumption. The visits ISD actually made first each quarter failed 62% of the time, which puts the gain nearer 320 than 349. Against a simple rule the gain is 40 to 95 failed inspections a year: ranking by violations at the last routine inspection alone reaches 81.9%, and M1's lead over B1 falls to 3.7 points without late follow-ups (4% of rows) and to 3.3 counting each restaurant once.
- The algorithm matters little. M1 beats logistic regression by 1.1 points (CI 0.3 to 1.9). The text model is 0.7 ahead of M1 (CI 0.1 to 1.4). That interval excludes zero with every bootstrap seed, although the edge is only about twice M2's 0.3-point noise between devices. We kept M1 because the choice was fixed before the test, where text added nothing.
- Scores and "model factors" are model outputs, not causes or findings about a business. Zip code is an input, so some neighborhoods rank higher. Dropping location cost about 0.3 points on validation, so ISD can choose to leave it out. The test set is the inspections ISD chose to make. Nothing here measures prevented illness or dollars. Location and type are pull-date values, and name masking in comments is best effort.

## Repo map

`notebooks/` 01 data, 02 baselines, 03 models, 04 results. `src/` logic (`data_pull`, `features`, `evaluate`, `models`, `text`, `impact`). `preds/` scores per inspection. `results/` tables, `figures/`, priority list. `docs/` decisions, data dictionary, frozen config, review log.
