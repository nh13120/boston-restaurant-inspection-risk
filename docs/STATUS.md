# Status: pipeline done, pushed (private), all four review gates run

Test years (2024 to Nov 2026, 17,171 routine inspections, 59% fail). Share of the first 20% of visits that find a failure:

- random order 59%, past-failure-rate rule 80%
- logistic regression 87%, LightGBM (final) 88%, + frozen text 88.8%, + fine-tuned text 89.1% (added later)

Same yearly total, found earlier: about 350 more failures in the first 20% of visits than random order, 42 to 94 more than a hand rule.

Caveats: "random order = today" is assumed. Model adds 1 point over logistic regression, text 1 more. Zip code is an input.

Needed: a source on ISD's scheduling, a cost per failure, a decision on the named priority list.
