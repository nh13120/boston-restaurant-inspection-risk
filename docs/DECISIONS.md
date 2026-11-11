# Decisions log

Each entry: what we chose, the alternatives, and one line of reasoning.

## Phase 0: setup

**D0.1 Environment.** Chose a local `.venv` built with Python 3.13. Alternative: reuse the class venv at `ai-edge/ai-edge.venv`. That folder does not exist on this machine, so there was nothing to reuse.

**D0.2 Patterns reused from the lecture notebooks.**

- Lecture 6 (`/Users/nicolash/Desktop/Semester 1/ai-edge/Lecture 6/15.071_Lecture6_code/15.071_Lecture6_code/15.071_Lecture6.ipynb`): `LGBMClassifier(random_state=42, verbosity=-1, n_jobs=-1)`, a small explicit parameter grid, an `evaluate(...)` helper that returns one row per model, and results collected into a single DataFrame. The lecture tunes with `GridSearchCV` and grouped folds. We loop over the grid by hand because our split is by time: the validation set is the year 2023, which `GridSearchCV` would not respect without extra setup.
- Lecture 11 (`/Users/nicolash/Desktop/Semester 1/ai-edge/Lecture 11/15.071_Lecture11_code/15.071_Lecture11_code/15.071_Lecture11.ipynb`): `sklearn.decomposition.PCA` to compress embeddings, and the rule that any fitted transform (scaler, PCA) is fit on training rows only.
- Lecture 12 (`/Users/nicolash/Desktop/Semester 1/ai-edge/Lecture 12/15.071_Lecture12_code/15.071_Lecture12_code/15.071_Lecture12.ipynb`): `AutoTokenizer` and `AutoModel` with the encoder frozen under `torch.no_grad()`, batches moved to `DEVICE` (`mps` if available), mean pooling as `(hidden * mask).sum(1) / mask.sum(1)`, features cached to disk behind a `recompute` flag, and the final refit on train + validation before scoring test.

**D0.3 Seed.** 42 everywhere, as the project brief requires. The lectures use 1234 in places, and the brief takes priority.

**D0.4 LightGBM on macOS.** LightGBM needs the OpenMP library `libomp.dylib`, which this Mac does not have system-wide. Chose to add scikit-learn's bundled copy to LightGBM's search path inside `.venv` (`install_name_tool -add_rpath "@loader_path/../../sklearn/.dylibs"` on `lib_lightgbm.dylib`, then `codesign --force --sign -`). Alternative: `brew install libomp`, the standard fix, which the README recommends to other users. We did not run it here because it changes the system outside this repo.

## Phase 1: data pull and profile

**D1.1 Columns pulled.** The brief's list, with three adjustments found in the 5-row peek. The zip column is called `zip` (there is no `zip_code`). There is no neighborhood column, so we use `city`, which holds Boston neighborhood names (Dorchester, Roxbury, and so on; "Boston" covers downtown). We also pull `viol_status` and `licstatus`. Alternative: pull only the brief's columns. `viol_status` is needed to count violations correctly (D2.1) and `licstatus` to list active restaurants in Phase 7. The pull names its columns one by one because `_full_text` embeds owner names, so `SELECT *` would leak them.

**D1.2 Result codes.** From the `result` value counts:

- Fail: `HE_Fail`, `HE_FailExt`, `HE_Hearing`, `HE_TSOP`, `HE_Closure`, `HE_VolClos`, `HE_FAILNOR`.
- Pass: `HE_Pass`, `HE_Filed`, and the rare `Pass` and `NoViol` (1 row each).
- Administrative, dropped: `HE_NotReq`, `HE_OutBus`, `HE_Misc`, `Closed` (12,457 rows).

Reasoning: `HE_Filed` visits list violations but ISD does not require a follow-up, so for the decision "was this visit worth making first" they count as a pass. The alternative, counting `HE_Filed` as a fail, would turn the label into "any violation at all". Voluntary closures (`HE_VolClos`) and suspensions (`HE_TSOP`) carry mostly failed violation rows, so they are fails. `HE_NotReq` rows have no violations and arrive in timestamped batches, which marks them as bookkeeping records.

**D1.3 What one inspection is.** One license number on one calendar date. Alternative: one exact timestamp. 555 license-days have more than one timestamp, usually the same visit saved twice, so the date is the safer key. When the codes on one day disagree (241 inspections), the inspection is a fail if any code is a fail code.

**D1.4 Dates.** `resultdttm` is text such as `2018-03-20 14:54:25+00`, and about 9,400 rows add fractional seconds. We parse the first 19 characters, which works for every row.

**D1.5 The fixed split works.** Data runs from 2016-01-02 to 2026-11-10. Routine inspections per split: train 35,915 (2016 to 2022), validation 5,520 (2023), test 17,168 (2024-01-02 to 2026-11-10). 2020 has about a quarter fewer inspections than neighboring years because of the pandemic. We keep it, as the brief asks.

## Phase 2: model table

**D2.1 Counting violations.** `n_viol` and the level counts include only rows with `viol_status = Fail`. Rows with status `Pass` are earlier violations re-checked and found corrected (a re-inspection lists the same violations again with status `Pass`). They are counted separately in `n_rechecked_ok`, which is not a model feature. Alternative: count every violation row, which would give a passed re-inspection as many "violations" as the failure before it.

**D2.2 License age.** The snapshot holds each license's current issue date, which for 2,674 licenses is later than their first inspection (re-issued licenses and pre-opening visits). Using it directly would bring a future date into old rows. The clock therefore starts at the earlier of the issue date and the first inspection in our data. Alternative: set those rows to missing, but then the missing flag itself would reveal a later re-issue.

**D2.3 Re-inspection rule.** A visit is a re-inspection when the previous inspection failed and it comes within 30 days. After a failure the next visit comes a median of 8 days later and 95.7% come within 30 days, which matches ISD's published 30-day rule. The rule removes 40,304 of 98,907 inspections (40.7%), most of them passes (19% fail). It is applied after the history features are computed. About 4% of post-failure visits come later than 30 days and stay in the table. Some of them are late follow-ups.

**D2.4 Administrative rows leave before history is computed.** They are not inspections, so they should not count in `prior_n` or reset `days_since_last`.

**D2.5 Privacy masking.** `clean()` masks emails, phone numbers, honorifics with a name (`Mr. X`), names after "spoke with" or "per", and two-word names after a role word such as "owner" or "PIC". It changes 1,124 of 191,948 distinct comments. Alternative: a named-entity model, which is heavier than this project needs. A regex cannot catch every name, so the README lists that as a limitation. The source data is already public.

**D2.6 Known weak spots in the history features, kept as specified.** Most routine inspections are preceded by a passed re-inspection, so `last_fail` is 0 for 97% of rows and `last_n_viol` is usually 0. The features follow the brief. Whether routine-only versions help is a question for the improvement loop. History also starts in 2016, so `prior_n` undercounts for restaurants that opened earlier, and `is_first` in the early years often means "first in our data".

**D2.7 Extra static feature.** `is_takeout` (from `descript`, the establishment type) joins the feature list. The brief lists type among the static fields.
