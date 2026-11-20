# Which Boston restaurants should be inspected first?

For data scientists and ML engineers at Boston's Inspectional Services Department (ISD). ISD visits a restaurant [about twice a year](https://www.boston.gov/departments/inspectional-services/how-get-food-service-permit), and no published rule says which visit comes first. This repo ranks its 6,000 routine inspections a year (59% fail) by risk of failing. Charts and recommendations: [START_HERE.md](START_HERE.md).

## How it works

1. Data (`src/data_pull.py`, `src/features.py`): Analyze Boston's [Food Establishment Inspections](https://data.boston.gov/dataset/food-establishment-inspections) (PDDL), pulled 2026-11-11, no owner names. One row per routine inspection, label 1 = any fail code, 17 inputs ([data dictionary](docs/data_dictionary.md)): earlier inspections, the month, and type and location as of the pull.
2. Tabular models (`src/models.py`, `src/compare.py`): 2 rules, logistic regression, LightGBM (M1), 4 more.
3. Deep learning on text (`src/text.py`, `src/finetune.py`): earlier inspector comments through a MiniLM encoder, frozen (M2) or fine-tuned (M3), added to M1's inputs.
4. Evaluation (`src/evaluate.py`): train 2016 to 2022, tune on 2023, test on 2024 to Nov 2026, settings frozen before the test (`docs/FROZEN_CONFIG*.json`).

## Reproduce ([macOS setup, notebooks 05 and 06, run order](docs/setup.md))

```bash
python3.13 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
cd notebooks && jupyter nbconvert --to notebook --execute --inplace 0[1-4]_*.ipynb   # about 10 minutes
```

## Results: share of the first 20% of visits that find a failure (test years)

| Method | Hit rate (95% CI) | Serious violations found † |
|---|---|---|
| No ranking (reference point) | 59.0% (58 to 60) | 20% |
| Rule: past failure rate | 80.3% (78 to 82) | 33% |
| Rule: violations at the last routine visit | 81.9% (80 to 83) | 39% |
| Logistic regression | 87.0% (85 to 88) | 39% |
| Ridge, LASSO, random forest, XGBoost † | 86.9% to 88.3% | 39% to 40% |
| **M1: LightGBM, history (final model)** | **88.1% (86 to 89)** | **40%** |
| M1 without location † | 88.7% (87 to 90) | 40% |
| M3: M1 + fine-tuned comments † | 89.1% (88 to 90) | 40% |

† Added after the first test scoring; nothing was tuned on the test years. All models: `results/readme_table.csv`.

- M1 leads the past-failure rule by 7.8 points (6.5 to 9.1), or 3.7† without late follow-ups, and by 94% to 73%† in the first 5% of visits.
- Comments alone reach 84%† and add one point to M1 (M3: 1.0, 0.3 to 1.8), at the cost of a neural network.
- Why M1: it was fixed before the test, and no algorithm on the same inputs beats its hit rate beyond noise. LASSO†, with four dominant weights, ties M1 at 20% but trails by 2 points in the first 10%.
- Deploy M1 without location: within noise of M1 (+0.6 points), and Dorchester and Roxbury, which fail equally often, get 35% and 24% of their inspections in the first visits†, against 46% and 19%.

## Impact and limits

With the same inspectors and yearly total, ordering visits by M1 finds 324† more failed inspections a year in the first 20% of visits than the order ISD actually used, and failures surface 1.1 weeks sooner (rules: 0.8 to 0.9†). Against a simple rule it still finds 42† to 94 more, and matches the past-failure rule's count with 10% fewer visits (116 a year).

Limits: ISD's actual order is our estimate from visit dates (61.5%† of its first visits failed, 59.0% of all visits). The test reorders past visits, so run a pilot first. Refit yearly: the hit rate fell from 90.1% (2024) to 84.9% (2026).

Nicolas Hong, Louis Neubauer, Robin Cen, Jose Plaza Carreras. MIT 15.071 Project 1, Fall 2026. [MIT license](LICENSE).
