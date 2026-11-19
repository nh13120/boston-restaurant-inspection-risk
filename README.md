# Which Boston restaurants should be inspected first?

For data scientists and ML engineers at Boston's Inspectional Services Department (ISD). A restaurant's grade sets how often ISD visits it, but nothing sets which visit comes first. This repo ranks ISD's 6,000 routine inspections a year (59% fail) by risk of failing. Charts, recommendations and limits are in [START_HERE.md](START_HERE.md).

## How it works

1. Data (`src/data_pull.py`, `src/features.py`): Analyze Boston's [Food Establishment Inspections](https://data.boston.gov/dataset/food-establishment-inspections) (PDDL), pulled 2026-11-11 without owner names. One row per routine inspection, label 1 = any fail code, 17 inputs known before the visit ([data dictionary](docs/data_dictionary.md)).
2. Tabular models (`src/models.py`, `src/compare.py`): two rules, logistic regression, LightGBM (M1), four others.
3. Deep learning on text (`src/text.py`, `src/finetune.py`): earlier inspector comments through a MiniLM encoder, frozen (M2) or fine-tuned (M3), added to M1's inputs.
4. Evaluation (`src/evaluate.py`): train 2016 to 2022, tune on 2023, test on 2024 to Nov 2026 (17,171 inspections), with settings frozen before each model saw the test years (`docs/FROZEN_CONFIG*.json`).

## Reproduce

```bash
python3.13 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt     # macOS: see docs/setup.md for OpenMP
cd notebooks && jupyter nbconvert --to notebook --execute --inplace 0[1-4]_*.ipynb
```

About 10 minutes on an M-series laptop. [docs/setup.md](docs/setup.md) covers notebooks 05 and 06 and the run order.

## Results: share of the first 20% of visits that find a failure (test years)

| Method | Hit rate (95% CI) | Serious (level 3) violations found |
|---|---|---|
| No ranking (reference point) | 59.0% (58 to 60) | 20% |
| Rule: past failure rate | 80.3% (78 to 82) | 33% |
| Rule: most violations at the last routine inspection | 81.9% (80 to 83) | 39% |
| Logistic regression | 87.0% (85 to 88) | 39% |
| **M1: LightGBM on the history (final model)** | **88.1% (86 to 89)** | **40%** |
| M1 without location † | 88.7% (87 to 90) | 40% |
| M3: M1 + fine-tuned comment encoder † | 89.1% (88 to 90) | 40% |

† Added after the first test scoring, with its own frozen settings. Intervals resample restaurants. `results/readme_table.csv` adds ridge, LASSO, random forest, XGBoost (86.9% to 88.3%) and M2 (88.8%).

- M1 leads the past-failure rule by 7.8 points (6.5 to 9.1), or 3.7 without late follow-ups, and by 94% to 73% in the first 5% of visits. Each year it finds 42 to 94 more failed inspections than a rule in those visits, 1.1 weeks sooner, with the yearly total unchanged.
- Comments alone reach 84%. On top of the history they add one point (1.0, 0.3 to 1.8), as they mostly restate violation counts.
- Why M1: no algorithm on the same inputs beats it beyond noise. M3's point needs a neural network that reads comments, and LASSO, with four dominant weights, ties M1 at 20% but is 2 points behind in the first 10%.
- For use we recommend M1 without location: within noise of M1, and Dorchester and Roxbury, which fail equally often, get 35% and 24% of their inspections in the first visits, against 46% and 19%.

## Limitations

- ISD already inspects lower-graded restaurants more often ([schedule](https://www.boston.gov/departments/inspectional-services/how-we-grade-restaurants-and-food-trucks)), so the rules are the comparison to today. The test ranks past visits and does not simulate a schedule.
- The hit rate fell from 90.1% in 2024 to 84.9% in 2026: refit yearly. Scores are estimates, not findings.
