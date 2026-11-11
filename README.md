# Which Boston restaurants should be inspected first?

Boston's Inspectional Services Department (ISD) makes about 6,000 routine restaurant inspections a year with a fixed staff, and 59% of them find a failure. This repo ranks upcoming inspections by their risk of failing, so that the first 20% of visits go where failures are most likely.

## Data

[Food Establishment Inspections](https://data.boston.gov/dataset/food-establishment-inspections), Analyze Boston, Open Data Commons PDDL. Snapshot pulled on 2026-11-11 through the CKAN SQL API: every "Eating & Drinking" inspection with `resultdttm < '2026-11-11'` (785,315 violation rows since 2006; the exact query and a content fingerprint are in `docs/pull_metadata.json`). The owner-name columns are never downloaded. `data/` is git-ignored and must stay unpublished, because raw comments contain names and emails.

## Setup and run

Python 3.13. About 10 minutes on an Apple-silicon laptop, plus about 10 minutes for the first data pull and a 90 MB encoder download.

```bash
python3.13 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cd notebooks
for nb in 01_data 02_baselines 03_models 04_results; do
  jupyter nbconvert --to notebook --execute --inplace $nb.ipynb
done
```

macOS only: LightGBM needs the OpenMP library. Run `brew install libomp`, or, to leave the system untouched, point LightGBM at scikit-learn's copy:

```bash
LIB=$(python -c "import importlib.util as u, pathlib as p; print(p.Path(u.find_spec('lightgbm').origin).parent/'lib/lib_lightgbm.dylib')")
install_name_tool -add_rpath "@loader_path/../../sklearn/.dylibs" "$LIB" && codesign --force --sign - "$LIB"
```

Notebook 01 pulls the data when `data/raw/` is empty and warns if the city has edited rows since our snapshot. Exact numbers need the original snapshot.

## Results on the test years (2024-01-02 to 2026-11-10, 17,171 inspections)

Capacity is fixed at the riskiest 20% of inspections. Hit rate is the share of those visits that find a failure. Settings were frozen on the validation year (2023) before the test years were scored once.

| Model | Hit rate (95% CI) | Share of failures caught | AUC |
|---|---|---|---|
| B0 Random order (today's routine schedule) | 59.0% (57.8 to 59.9) | 20.0% | 0.500 |
| B1 Rank by past failure rate (business rule) | 80.3% (78.5 to 81.7) | 27.2% | 0.698 |
| B2 Logistic regression on history | 87.0% (85.2 to 88.2) | 29.5% | 0.736 |
| M1 LightGBM on history (final model) | 88.1% (86.4 to 89.4) | 29.9% | 0.748 |
| M2 LightGBM on history + comment embeddings | 88.8% (87.4 to 90.2) | 30.1% | 0.747 |

With the same 1,201 visits a year, M1 finds about 349 more failed inspections than random order and 94 more than the rule. It matches the rule's yield with 116 fewer visits a year. Reordering visits within each quarter would reach a failing restaurant 1.1 weeks sooner on average. `results/priority_list.csv` scores the next inspection of 3,163 active restaurants.

## Modeling choices

- One row per routine inspection. The label is 1 for any fail code. A 0 means "no follow-up required" and can include minor violations. Visits within 30 days of a failure are re-inspections and are dropped after the features are built.
- Features use only earlier inspections of the same license (`shift(1)` per license, checked against a by-hand recomputation). History back to 2006 feeds the features, and the model table starts in 2016.
- Time split: train 2016 to 2022, validation 2023, test 2024 onward. Final fits use 2016 to 2023.
- Text: the previous inspection's comments, embedded with a frozen `all-MiniLM-L6-v2`, mean-pooled, reduced to 32 PCA components fitted on the fitting rows.
- Intervals are bootstraps over whole restaurants. M1 was chosen on validation, where the text model was no better.

## Limitations

- The algorithm matters little. M1 beats logistic regression by 1.1 points (CI 0.3 to 1.9) and trails the text model by 0.7 (CI 0.1 to 1.4), a gap that validation did not show and that is close to M2's device noise of 0.3 points. The gain over "the rule" depends on the rule: ranking by violations at the last routine inspection alone reaches 81.9%.
- The test set is the inspections ISD chose to make, and outcomes are assumed not to depend on visit timing. The numbers describe finding failures sooner, and say nothing about preventing illness. Risk scores and drivers are model outputs, not findings about a business. About 4% of rows are late follow-ups. Without them B1 rises to 84.8% and M1's lead over it narrows to 3.7 points.
- Zip, neighborhood and type are values on the pull date. Names in comments are masked with regexes, on a best-effort basis. Dollar impact is not estimated, because we have no sourced cost per missed failure.

## Repo map

`notebooks/` 01 data, 02 baselines, 03 models, 04 results. `src/` logic (`data_pull`, `features`, `evaluate`, `models`, `text`, `impact`). `preds/` scores per inspection. `results/` tables, `figures/`, priority list. `docs/` decisions, data dictionary, frozen config, review log.
