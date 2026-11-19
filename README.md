# Which Boston restaurants should be inspected first?

For data scientists and ML engineers at Boston's Inspectional Services Department (ISD). Grades set how often ISD visits a restaurant, but the published schedule does not say which visit comes first. This repo ranks its 6,000 routine inspections a year (59% fail) by risk of failing. Charts and recommendations: [START_HERE.md](START_HERE.md).

## How it works

1. Data (`src/data_pull.py`, `src/features.py`): Analyze Boston's [Food Establishment Inspections](https://data.boston.gov/dataset/food-establishment-inspections) (PDDL), pulled 2026-11-11, no owner names. One row per routine inspection, label 1 = any fail code, 17 inputs ([data dictionary](docs/data_dictionary.md)): earlier inspections, the month, and type and location as of the pull.
2. Tabular models (`src/models.py`, `src/compare.py`): two rules, logistic regression, LightGBM (M1), four others.
3. Deep learning on text (`src/text.py`, `src/finetune.py`): earlier inspector comments through a MiniLM encoder, frozen (M2) or fine-tuned (M3), added to M1's inputs.
4. Evaluation (`src/evaluate.py`): train 2016 to 2022, tune on 2023, test on 2024 to Nov 2026, settings frozen before the test (`docs/FROZEN_CONFIG*.json`).

## Reproduce ([setup, notebooks 05 and 06, run order](docs/setup.md))

```bash
python3.13 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt     # macOS: see docs/setup.md for OpenMP
cd notebooks && jupyter nbconvert --to notebook --execute --inplace 0[1-4]_*.ipynb   # about 10 minutes
```

## Results: share of the first 20% of visits that find a failure (test years)

| Method | Hit rate (95% CI) | Serious violations found † |
|---|---|---|
| No ranking (reference point) | 59.0% (58 to 60) | 20% |
| Rule: past failure rate | 80.3% (78 to 82) | 33% |
| Rule: violations at the last routine visit | 81.9% (80 to 83) | 39% |
| Logistic regression | 87.0% (85 to 88) | 39% |
| **M1: LightGBM, history (final model)** | **88.1% (86 to 89)** | **40%** |
| M1 without location † | 88.7% (87 to 90) | 40% |
| M3: M1 + fine-tuned comments † | 89.1% (88 to 90) | 40% |

† Added after the first test scoring: settings fixed beforehand, or reporting on the frozen predictions. All models: `results/readme_table.csv`.

- M1 leads the past-failure rule by 7.8 points (6.5 to 9.1), or 3.7† without late follow-ups, and by 94% to 73%† in the first 5% of visits. In the first 20% it finds 42† to 94 more failed inspections a year than a rule. Failures are found 1.1 weeks† sooner than in ISD's actual order (rules: 0.8 to 0.9), and the yearly total is unchanged.
- Comments alone reach 84%†. On top of the history they add one point (M3: 1.0, 0.3 to 1.8), which needs a neural network that reads them.
- Why M1: it was fixed before the test, and no algorithm on the same inputs beats its hit rate beyond noise. LASSO†, with four dominant weights, ties M1 at 20% but trails by 2 points in the first 10%.
- Deploy M1 without location: within noise of M1 (+0.6 points), and Dorchester and Roxbury, which fail equally often, get 35% and 24% of their inspections in the first visits†, against 46% and 19%.

## Limitations

ISD already inspects lower-graded restaurants more often ([schedule](https://www.boston.gov/departments/inspectional-services/how-we-grade-restaurants-and-food-trucks)), so the rules are the comparison to today. The test ranks past visits, not a simulated schedule. The hit rate fell from 90.1% (2024) to 84.9% (2026): refit yearly. Scores are estimates, not findings.
