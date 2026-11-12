# How to read `priority_list.csv`

The list ranks 3,163 active Boston restaurants by a model's estimate of the chance that their next routine inspection fails, as of 2026-11-11. The estimate comes from public inspection records only.

- `estimated_risk_pct` is a model estimate. It is not a finding about the business, and a high value does not mean the restaurant is unsafe today.
- `top_3_model_factors` are the inputs that raise the estimate the most. They explain the model. They are not causes.
- Location is one of the inputs, so restaurants in some neighborhoods rank higher on average.
- The list ranks by risk only. It does not know which restaurants are due for a visit. Use `days_since_last_visit` to combine risk with ISD's own schedule.
- The list is a planning aid for inspectors. It is not a rating of businesses.
- To refresh it: set `FORCE_PULL = True` in notebook 01, move `SNAPSHOT_END` in `src/data_pull.py` to today, and rerun the four notebooks.
