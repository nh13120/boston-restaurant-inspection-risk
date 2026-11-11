# Decisions log

Each entry: what we chose, the alternatives, and one line of reasoning.

## Phase 0: setup

**D0.1 Environment.** Chose a local `.venv` built with Python 3.13. Alternative: reuse the class venv at `ai-edge/ai-edge.venv`. That folder does not exist on this machine, so there was nothing to reuse.

**D0.2 Patterns reused from the lecture notebooks.**

- Lecture 6 (`~/Desktop/Semester 1/ai-edge/Lecture 6/15.071_Lecture6_code/15.071_Lecture6_code/15.071_Lecture6.ipynb`): `LGBMClassifier(random_state=42, verbosity=-1, n_jobs=-1)`, a small explicit parameter grid, an `evaluate(...)` helper that returns one row per model, and results collected into a single DataFrame. The lecture tunes with `GridSearchCV` and grouped folds. We loop over the grid by hand because our split is by time: the validation set is the year 2023, which `GridSearchCV` would not respect without extra setup.
- Lecture 11 (`~/Desktop/Semester 1/ai-edge/Lecture 11/15.071_Lecture11_code/15.071_Lecture11_code/15.071_Lecture11.ipynb`): `sklearn.decomposition.PCA` to compress embeddings, and the rule that any fitted transform (scaler, PCA) is fit on training rows only.
- Lecture 12 (`~/Desktop/Semester 1/ai-edge/Lecture 12/15.071_Lecture12_code/15.071_Lecture12_code/15.071_Lecture12.ipynb`): `AutoTokenizer` and `AutoModel` with the encoder frozen under `torch.no_grad()`, batches moved to `DEVICE` (`mps` if available), mean pooling as `(hidden * mask).sum(1) / mask.sum(1)`, features cached to disk behind a `recompute` flag, and the final refit on train + validation before scoring test.

**D0.3 Seed.** 42 everywhere, as the project brief requires. The lectures use 1234 in places, and the brief takes priority.

**D0.4 LightGBM on macOS.** LightGBM needs the OpenMP library `libomp.dylib`, which this Mac does not have system-wide. Chose to add scikit-learn's bundled copy to LightGBM's search path inside `.venv` (`install_name_tool -add_rpath "@loader_path/../../sklearn/.dylibs"` on `lib_lightgbm.dylib`, then `codesign --force --sign -`). Alternative: `brew install libomp`, the standard fix, which the README recommends to other users. We did not run it here because it changes the system outside this repo.

## Phase 1: data pull and profile

**D1.1 Columns pulled.** The brief's list, with three adjustments found in the 5-row peek. The zip column is called `zip` (there is no `zip_code`). There is no neighborhood column, so we use `city`, which holds Boston neighborhood names (Dorchester, Roxbury, and so on; "Boston" covers downtown). We also pull `viol_status` and `licstatus`. Alternative: pull only the brief's columns. `viol_status` is needed to count violations correctly (D2.1) and `licstatus` to list active restaurants in Phase 7. The pull names its columns one by one because `_full_text` embeds owner names, so `SELECT *` would leak them.

**D1.2 Result codes.** From the `result` value counts:

- Fail: `HE_Fail`, `HE_FailExt`, `HE_Hearing`, `HE_TSOP`, `HE_Closure`, `HE_VolClos`, `HE_FAILNOR`, and `Fail` (129 rows before 2010).
- Pass: `HE_Pass`, `HE_Filed`, and the rare `Pass` and `NoViol`.
- Administrative, dropped: `HE_NotReq`, `HE_OutBus`, `HE_Misc`, `HE_Hold`, `Closed`, `DATAERR` (19,464 rows).

Reasoning: `HE_Filed` visits list violations but ISD does not require a follow-up, so for the decision "was this visit worth making first" they count as a pass. The alternative, counting `HE_Filed` as a fail, would turn the label into "any violation at all". Voluntary closures (`HE_VolClos`) and suspensions (`HE_TSOP`) carry mostly failed violation rows, so they are fails. `HE_NotReq` rows have no violations and arrive in timestamped batches, which marks them as bookkeeping records. `HE_Hold` (11 rows) and `DATAERR` (34 rows) appear only before 2010 and have no clear outcome.

**D1.3 What one inspection is.** One license number on one calendar date, in Boston time. Alternative: one exact timestamp. Several hundred license-days have more than one timestamp, usually the same visit saved twice, so the date is the safer key. When the codes on one day disagree (369 inspections), the inspection is a fail if any code is a fail code.

**D1.4 Dates.** `resultdttm` is UTC text such as `2018-03-20 14:54:25+00`, and some rows add fractional seconds. We parse the first 19 characters, which works for every row, and convert to `America/New_York` before taking the date (changed at Gate 1, see G1.6).

**D1.5 The fixed split works.** The model table runs from 2016-01-02 to 2026-11-10. Routine inspections per split: train 35,872 (2016 to 2022), validation 5,521 (2023), test 17,171 (2024-01-02 to 2026-11-10). 2020 has about a quarter fewer inspections than neighboring years because of the pandemic. We keep it, as the brief asks.

## Phase 2: model table

**D2.1 Counting violations.** `n_viol` and the level counts include only rows with `viol_status = Fail`. Rows with status `Pass` are earlier violations re-checked and found corrected (a re-inspection lists the same violations again with status `Pass`). They are counted separately in `n_rechecked_ok`, which is not a model feature. Alternative: count every violation row, which would give a passed re-inspection as many "violations" as the failure before it.

**D2.2 License age.** `license_age_days` counts from the license's first inspection on record. The snapshot only holds each license's current issue date, which for 7,220 of 7,916 licenses is later than their first inspection (licenses get re-issued), so it would bring a future date into old rows. Alternative tried first: the earlier of the issue date and the first inspection. The Gate 2 reviewer pointed out that this still switched on a pull-date value for 5% of rows, so the issue date is now left out of the feature altogether.

**D2.3 Re-inspection rule.** A visit is a re-inspection when the previous inspection failed and it comes within 30 days. After a failure the next visit comes a median of 8 days later and 95.7% come within 30 days, which matches ISD's published 30-day rule. From 2016 on, the rule removes 40,360 of 98,924 inspections (40.8%), most of them passes (19% fail). It is applied after the history features are computed. About 4% of post-failure visits come later than 30 days and stay in the table (1,807 rows). Some of them are late follow-ups. Notebook 04 checks whether the results hold without them.

**D2.4 Administrative rows leave before history is computed.** They are not inspections, so they should not count in `prior_n` or reset `days_since_last`.

**D2.5 Privacy masking.** `clean()` masks emails, phone numbers and personal names in comments with `[EMAIL]`, `[PHONE]` and `[NAME]`. Names are found three ways: a known first name followed by a capitalized word, anywhere in the text; a name context ("PIC Jane stated", "owner: Jane Doe", "spoke with", "certificate posted for", "Jane stated"); and after an honorific. The first names come from `src/first_names.txt` (about 1,300 names: the BSD `propernames` list plus common US first names, minus names that are also ordinary words, foods, brands or streets). It changes 1,905 of 324,517 distinct comments. Alternative: a named-entity model, which is heavier than this project needs. The masking is best effort. Three reviewers with their own name lists each still found some names (see G1.7 and G1.8), so the comment text stays in git-ignored `data/` and the README lists this as a limitation.

**D2.6 Extra static feature.** `is_takeout` (from `descript`, the establishment type) joins the feature list. The brief lists type among the static fields.

## Gate 1 changes (data and leakage audit)

The reviewer found no leakage. These decisions answer its findings. Details are in `REVIEW_LOG.md`.

**G1.1 History before 2016.** The brief filters the pull to 2016 onward. With that filter, 60% of 2016 rows looked like a restaurant's first inspection, and `prior_n` meant something different every year. Chose to pull all years (from April 2006) and use the inspections before 2016 as history only. The model table and the split still start in 2016. Alternative: drop 2016 from training as a burn-in year, which would change the fixed split. After the change, `is_first` is 5 to 6% in every year.

**G1.2 Routine-only history features.** Half of all routine inspections are preceded by a passed re-inspection, so `last_fail` was 1 on only 3% of rows and pointed the wrong way. Added `last_routine_fail`, `last_routine_n_viol`, `last_routine_n_lvl3` and `prior_routine_fail_rate`, which look back to the last routine inspection. The brief's original features stay. Alternative: leave this to the improvement loop. The reviewer rated it a major finding, so it is fixed here. On validation, M1's hit rate went from 87.8% to 90.5%.

**G1.3 Windowed count.** Added `prior_n_3y`, the number of inspections in the previous three years. Unlike `prior_n`, it does not grow with calendar time (about 7 to 8 in every year). `prior_n` stays, as the brief lists it.

**G1.4 A frozen, checkable snapshot.** The pull now has an upper bound (`resultdttm < '2026-11-11'`), and its metadata records the source, the license and a fingerprint of the content. `docs/pull_metadata.json` is the committed reference. A fresh pull is compared with it and prints a warning if the city has edited rows since. Alternative: commit the data, which the brief forbids.

**G1.5 `licstatus` leaves the model table.** It describes the license today, so it would be future information as a feature. It stays in `inspections_full.parquet` for the priority list. `models.py` also refuses any outcome-side column as a feature. The other static fields (`zip`, `neighborhood`, type) are also as of the pull date. We accept that, because a restaurant's location and type rarely change, and the README says so.

**G1.6 Smaller fixes.** Dates are converted to Boston time before the calendar date is taken. 224 exact duplicate rows are dropped. The leakage check now compares all 13 history columns with a by-hand recomputation for 300 random restaurants. `requirements.txt` is pinned. The lecture paths in D0.2 no longer show the local user name.

**G1.7 Round 2: masking again, and what "pass" means.** The second review found at least 94 comments with names that the first rewrite missed: common names absent from the list (Tony, Mike, Mary), names in capitals, and forms such as "PIC Tom stated". We extended the name list and the contexts (D2.5). Run against the reviewer's own name list, the scan now leaves 3 name-like strings in 324,528 comment pieces. Notebook 01 no longer claims that nothing personal remains. The reviewer also asked us to state that a "pass" means "no follow-up required": 97% of `HE_Filed` visits list at least one violation. The data dictionary now says so. Two things were noted as limitations and left unchanged: the city sometimes edits comment text after the inspection date (13 model rows mention a later date), and history starts in 2006, so `prior_n` is understated for the oldest restaurants.

## Phase 4 and 5: models on validation

**D4.1 Grid.** `num_leaves` 7, 15, 31; `learning_rate` 0.02, 0.05; `n_estimators` 100, 500 (12 combinations), with early stopping after 50 rounds without a better validation AUC. `n_estimators` is an upper limit, and the frozen value is the number of trees early stopping kept. Alternative: caps of 300 and 1000, which early stopping never reached, so half the grid was a duplicate. Selection is on validation `precision_at_k`, ties broken by AUC, then by the smaller model.

**D4.2 Deterministic training.** LightGBM runs with `n_jobs=1`, `deterministic=True` and `force_row_wise=True`, so a rerun gives identical numbers. The lecture uses `n_jobs=-1`. The data is small enough that one thread takes seconds.

**D4.3 Ties in the ranking.** When two inspections have the same score, `evaluate.py` orders them with a seeded random draw. Otherwise a rule with many tied scores, such as B1, would be judged by the row order of the table.

**D5.1 Embedding run.** 100 comments take about 0.4 seconds on `mps`, so the full run takes under a minute. We keep the 256-token truncation (7% of comments are longer) and embed each distinct text once. The cache stores a hash of every text, so it is rebuilt when the text changes.

**D5.2 PCA.** 32 components, fitted on training rows that have comments (59.8% of variance kept). Rows without comments get a zero vector and `has_prior_comments = 0`, as the brief asks.

**D5.3 Text adds no measurable value.** On validation, M2 minus M1 is +0.1 points of hit rate (95% CI -1.3 to +0.9). Before the routine-only features were added (G1.2), text gave +1.7 points. The comments mostly describe the violations of the last routine inspection, and `last_routine_n_viol` now carries that signal directly. M2's selected settings change with tiny numeric differences, because its 12 grid combinations sit within one point of each other.

## Improvement loop (one round, validation only)

An experimenter subagent analyzed the validation errors and proposed changes. The rule set in advance: keep a change only if validation `precision_at_k` improves by more than the half-width of M1's bootstrap interval (about 2 points).

What the error analysis found: M1 and M2 make the same mistakes (1,025 of 1,104 top rows in common). Almost every top-ranked row had a failed last routine inspection. The blind spot is restaurants whose last routine inspection was clean (0 to 2 violations): 41% of them still fail, and they hold more than half of the failures the model misses.

**IL.1 Routine-violation memory: tested, not kept.** Two features, `r3_mean_viol` (mean violations over the last three routine inspections) and `prev2_routine_n_viol` (violations at the routine inspection before the last one). Validation hit rate went from 90.49% to 91.30%, a gain of 0.82 points (95% CI -0.45 to +1.90), below the 1.95-point threshold. The experimenter's backtests on 2019 to 2022 also showed gains of 0.5 to 1.2 points each year, so the idea is probably real but small. The code stays in `features.candidate_memory_features` and notebook 03 reruns the test. It is the first thing to try with more data.

**IL.2 Comment truncation: tested by the experimenter, not proposed.** M2's validation hit rate at 32, 64, 128, 256 and 512 tokens was 89.8%, 90.1%, 89.7%, 90.2% and 90.4%. Every difference from 256 is within noise, and none beats M1. We keep 256.

**IL.3 Other feature groups the experimenter tried and dropped.** Ten follow-up-cycle features (89.8%), four result-code features (90.2%) and two comment-size features (89.8%), against M1's 90.5%. None was proposed.

No third proposal was made, and there is no second round, because more rounds would overfit the validation year.

## Gate 1 round 3 and Gate 2 changes

**G1.8 Gate 1 round 3 (no blocker, no major).** The third reviewer still found about 62 comments with a name (single first names after "PIC", "X stated", lower-case names around certificates). Added those three contexts. Masking some ordinary words as `[NAME]` is the price ("PIC Products" can become "PIC [NAME]"); it only touches the text model. Left as is: notebook 01 still prints the fail rate of every split, test years included, because the brief asks for it. These are label summaries, and no modeling decision uses them.

**G2.1 A fair business rule (B1).** The brief sets first inspections to "the train base rate". The routine fail rate (0.60) is on the wrong scale: `prior_fail_rate` counts re-inspections, which mostly pass, so it averages 0.37. With 0.60, every new restaurant landed in B1's top 20% and B1 scored 68.9% on validation. Chose the mean `prior_fail_rate` of the fitting rows as the fallback, which puts new restaurants in the middle of the ranking. B1 then scores 80.4%. Alternative: keep the brief's wording, which would double the claimed gain over the rule.

**G2.2 Ties, and random order as an exact average.** `evaluate()` takes the expected number of hits over all ways of breaking a tie at the cutoff, so B1 (many restaurants share a rate of exactly 0.5) is judged by its average. Before, one seeded random draw decided. B0 uses the same idea: every inspection gets the same score, so its hit rate is exactly the fail rate, which is what a random order achieves on average. The brief says "random scores"; one random draw would add noise to the baseline for no benefit. Curves still need an order and break ties with a random stream derived from seed 42.

**G2.3 Which model we carry forward.** The brief names M1 as the main model and M2 as its text extension. Rule, applied on validation before the test is scored: keep M1 unless M2 beats it beyond noise. M2 does not (+0.1 points, 95% CI -1.2 to +1.0), so the final model is M1. M1 beats the past-failure-rate rule by 10.1 points (CI +7.8 to +12.2). Its lead over logistic regression is 1.5 points (CI 0.0 to +2.9), which is not distinguishable from noise, and the notebooks and README say so. Alternative tried and dropped after the round 2 review: a ladder that started from logistic regression and moved up when an interval excluded zero. It picked M1 on a margin of one inspection, and the answer changed with the bootstrap seed. We keep M1 because the brief designates it and because tree models give per-restaurant drivers. A team that prefers logistic regression loses nothing measurable.

**G2.6 Confidence intervals resample whole restaurants.** Inspections of one restaurant are not independent (most restaurants appear more than once per year), so the bootstrap draws license numbers and takes all their inspections. Single metrics use 500 resamples, as the brief asks, and paired differences use 2,000. Alternative: resample single inspections, which gives almost the same widths here.

**G2.7 Refit on train + validation for the test run.** The brief asks for this, PCA included. "PCA fitted on train only" therefore means: on validation, fitted on 2016 to 2022; for the test run, fitted on 2016 to 2023. No test row is ever used to fit anything. The frozen config states it.

**G2.8 A second simple rule, for honesty.** Ranking by the number of violations at the last routine inspection scores 84.4% on validation, 4 points above B1. B1 stays the business-rule benchmark because the brief defines it. Notebooks 02 and 04 show the second rule as a sensitivity row, and the README says that the gain "over the rule" depends on the rule.

**G2.4 M2 depends on the device.** Embeddings computed on `mps` and on CPU differ in the sixth decimal. That is enough to change which of M2's near-tied grid settings wins and to move its hit rate by about 0.3 points. The conclusion (text adds nothing measurable) holds on both. We keep `mps` as the brief asks, pin the encoder revision, and state the caveat in the README. M1, the final model, does not use embeddings and reruns bit for bit.

**G2.5 Text examples.** The brief asks for failed inspections that M2 ranked in its top 10% and M1 "much lower". There are none: the largest move is about 10 percentile points and the median is near zero. `results/text_examples.md` now says so and shows the three largest moves.
