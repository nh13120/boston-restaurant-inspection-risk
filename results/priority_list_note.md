# How to read `priority_list.csv`

The list ranks 3,163 active Boston restaurants by a model's estimate of the chance that their next routine inspection fails, as of 2026-11-11. The estimate comes from public inspection records only.

- `estimated_risk_pct` is a model estimate. It is not a finding about the business, and a high value does not mean the restaurant is unsafe today.
- `top_3_model_factors` are the inputs that raise the estimate the most. They explain the model. They are not causes.
- Location is one of the inputs, so restaurants in some neighborhoods rank higher on average.
- The list is a suggestion for where an inspector might look first. It should not be used to rate or compare businesses in public.
