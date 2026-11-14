# Which Boston restaurants should be inspected first?

Written for a data scientist or ML engineer at Boston's Inspectional Services Department (ISD). ISD makes about 6,000 routine restaurant inspections a year, and 59% end in a failure. The model ranks upcoming inspections by risk of failing, so planners can schedule the riskiest restaurants first.

## How it works

1. Data (`src/data_pull.py`): [Food Establishment Inspections](https://data.boston.gov/dataset/food-establishment-inspections) from Analyze Boston (PDDL license), pulled on 2026-11-11 without owner names. Keep `data/` unpublished: raw comments contain names and emails.
2. Features (`src/features.py`): one row per routine inspection (re-inspections within 30 days of a failure are dropped), label 1 = any fail code. The 17 features use only the license's earlier inspections, plus month and today's type and location.
3. Models (`src/models.py`, `src/text.py`, `src/finetune.py`): M1, the final model, is LightGBM on those features. M2 adds the previous inspection's comments, embedded with a frozen `all-MiniLM-L6-v2`, and M3 fine-tunes that encoder.
4. Evaluation (`src/evaluate.py`): train 2016 to 2022, validation 2023, test 2024 onward (17,171 inspections). Settings were frozen (`docs/FROZEN_CONFIG.json`) before the test years were scored. Final fits use 2016 to 2023.

## Reproduce

```bash
python3.13 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt     # macOS: then follow docs/setup_macos.md (LightGBM needs OpenMP)
cd notebooks && for nb in 01_data 02_baselines 03_models 04_results; do
  jupyter nbconvert --to notebook --execute --inplace $nb.ipynb
done
```

About 10 minutes on an M-series laptop including the data pull, plus a 90 MB encoder download. The city still edits the table: a pull two days after ours had 34 more rows and moved each hit rate by less than 0.01 point. For our exact numbers (verified on Apple Silicon), install from `requirements-lock.txt`, get `inspections_snapshot.parquet` from the repo owner and place it in `data/raw/`. Notebook 05 (M3) is optional: about 40 minutes on a GPU.

## Results on the first 20% of visits, test years 2024 to Nov 2026

| Method | Visits that find a failure (95% CI) | Share of all failures found | Share of serious violations found |
|---|---|---|---|
| B0 Random order (assumed status quo) | 59.0% (58 to 60) | 20.0% | 20% |
| B1 Past failure rate (business rule) | 80.3% (78 to 82) | 27.2% | 33% |
| B2 Logistic regression | 87.0% (85 to 88) | 29.5% | 39% |
| M1 LightGBM (final model) | 88.1% (86 to 89) | 29.9% | 40% |
| M2 M1 + comment embeddings | 88.8% (87 to 90) | 30.1% | 40% |
| M3 M1 + fine-tuned text encoder | 89.1% (88 to 90) | 30.2% | 40% |

Intervals are bootstraps over whole restaurants. The yearly total (about 3,544 failed inspections) is the same in any order: ranking changes when they are found. In the first 20% of visits (1,201 a year), M1 finds 349 more of them than a random order (CI 333 to 364) and 42 to 94 more than a hand rule.

## Limitations

- We could not confirm how ISD orders visits today: "random order" is our assumption, and 20% is an illustrative capacity. Hit rates fell from 90.1% in 2024 to 84.9% in 2026.
- M1 was fixed as the final model on validation, where text added nothing. On test it leads logistic regression by 1.1 points (CI 0.3 to 1.9), and M3, added later with its own freeze, leads M1 by 1.0 (CI 0.3 to 1.8).
- Zip code and neighborhood are inputs: Dorchester gets 46% of its inspections among the first visits and Roxbury 19%, at the same fail rate. Dropping them cost 0.5 points on 2023, within noise, and we suggest ISD does so.
- Scores are estimates, not findings about a business: read `results/priority_list_note.md` before using `results/priority_list.csv`. See also `docs/DECISIONS.md` and, for charts, [START_HERE.md](START_HERE.md).
