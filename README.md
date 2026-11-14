# Which Boston restaurants should be inspected first?

For data scientists and ML engineers at Boston's Inspectional Services Department (ISD). The model ranks ISD's routine restaurant inspections (about 6,000 a year, 59% end in a failure) by risk of failing, so the riskiest can be visited first. Charts, recommendations and limits are in [START_HERE.md](START_HERE.md).

## How it works

1. Data (`src/data_pull.py`): [Food Establishment Inspections](https://data.boston.gov/dataset/food-establishment-inspections) from Analyze Boston (PDDL license), pulled on 2026-11-11 without owner names.
2. Features (`src/features.py`, [data dictionary](docs/data_dictionary.md)): one row per routine inspection (re-inspections within 30 days of a failure are dropped), label 1 = any fail result code. Of the 17 inputs, 13 use only the license's earlier inspections. The rest are the month, and the type and location as of the pull date.
3. Models (`src/models.py`, `src/text.py`, `src/finetune.py`): M1, the final model, is LightGBM on those inputs. The deep-learning models add the inspector comments: M2 through a frozen `all-MiniLM-L6-v2` encoder, M3 through a fine-tuned one.
4. Evaluation (`src/evaluate.py`): train 2016 to 2022, validation 2023, test 2024 onward (17,171 inspections). Settings come from a 12-point grid scored on validation precision@20%. They were frozen (`docs/FROZEN_CONFIG.json`) before the test years were scored, and models were then refitted on 2016 to 2023.

## Reproduce

```bash
python3.13 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt     # macOS: LightGBM also needs OpenMP, see docs/setup.md
cd notebooks && jupyter nbconvert --to notebook --execute --inplace 0[1-4]_*.ipynb
```

About 10 minutes on an M-series laptop, with internet access. [docs/setup.md](docs/setup.md) covers our exact files and other hardware. Notebook 05 (M3) is optional.

## Results on the first 20% of visits, test years 2024 to Nov 2026

| Method | Visits that find a failure (precision@20%, 95% CI) | Share of serious violations found† |
|---|---|---|
| B0 Random order (assumed) | 59.0% (58 to 60) | 20% |
| B1 Past failure rate (simple rule) | 80.3% (78 to 82) | 33% |
| B2 Logistic regression | 87.0% (85 to 88) | 39% |
| M1 LightGBM (final model) | 88.1% (86 to 89) | 40% |
| M2 M1 + comment embeddings | 88.8% (87 to 90) | 40% |
| M3 M1 + fine-tuned text encoder† | 89.1% (88 to 90) | 40% |

Intervals are bootstraps over restaurants. Paired differences: M1 leads B1 by 7.8 points (6.5 to 9.1) and B2 by 1.1 (0.3 to 1.9), and M3 leads M1 by 1.0 (0.3 to 1.8). The yearly total (about 3,544 failed inspections) is the same in any order: ranking changes when they are found. In the first 20% of visits (1,201 a year), M1 finds 349 more of them than a random order (CI 333 to 364) and 42† to 94 more than a simple rule.

† Added after the first test scoring: extra reporting on the frozen predictions, or, for M3, a model built afterwards with its own freeze.

## Limitations

- "Random order" is an assumption, and the test measures ranking quality on past visits, not a simulated schedule.
- The final model omits comments: they added nothing measurable on 2023, and about one point on the test years, seen after the freeze. M3 is the choice for a team that wants that point and accepts a neural network.
- Location is an input the model barely relies on (scrambling it costs 0.2 points†), but it shifts early visits between neighborhoods. We suggest dropping it. A refit without it was checked on 2023 only.
- Scores are estimates, not findings about a business.
