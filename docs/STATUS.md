# Status: pipeline finished, repo ready for review

Test years (2024 to Nov 2026, 17,171 routine inspections, 59% fail), first 20% of visits ranked by risk:

- random order: 59% of visits find a failure
- past-failure-rate rule: 80%
- logistic regression: 87%
- LightGBM on history (final): 88%
- LightGBM + comment embeddings: 89%

Impact: about 350 more failed inspections found in the first 20% of visits per year than random order, 94 more than the rule. Same yearly total, found earlier (1.1 weeks).

Honest caveats: the algorithm adds about 1 point over logistic regression, text 0.7. "Random order = today" is an assumption. Zip code is an input.

Still needed: a sourced cost per failure, and a decision on publishing the named priority list.
