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

**D2.2 License age.** The snapshot holds each license's current issue date, which for most licenses (7,220 of 7,916) is later than their first inspection on record: licenses get re-issued, and visits happen before opening. Using it directly would bring a future date into old rows. The clock therefore starts at the earlier of the issue date and the first inspection in our data, so in practice the feature is mostly "days since the first inspection on record". Alternative: set those rows to missing, but then the missing flag itself would reveal a later re-issue.

**D2.3 Re-inspection rule.** A visit is a re-inspection when the previous inspection failed and it comes within 30 days. After a failure the next visit comes a median of 8 days later and 95.7% come within 30 days, which matches ISD's published 30-day rule. From 2016 on, the rule removes 40,360 of 98,924 inspections (40.8%), most of them passes (19% fail). It is applied after the history features are computed. About 4% of post-failure visits come later than 30 days and stay in the table (1,807 rows). Some of them are late follow-ups. Notebook 04 checks whether the results hold without them.

**D2.4 Administrative rows leave before history is computed.** They are not inspections, so they should not count in `prior_n` or reset `days_since_last`.

**D2.5 Privacy masking.** `clean()` masks emails, phone numbers and personal names in comments with `[EMAIL]`, `[PHONE]` and `[NAME]`. Names are found three ways: after a context word ("spoke with", "per", "owner", "PIC", "Inspector", "certificate for"), after an honorific, and anywhere a known first name is followed by a capitalized word. The first names come from `src/first_names.txt` (958 names from the BSD `propernames` list, minus names that are also common words, brands or streets). It changes 1,543 of 324,517 distinct comments. Alternative: a named-entity model, which is heavier than this project needs. A regex cannot catch every name, so the README lists that as a limitation. The source data is already public.

**D2.6 Extra static feature.** `is_takeout` (from `descript`, the establishment type) joins the feature list. The brief lists type among the static fields.

## Gate 1 changes (data and leakage audit)

The reviewer found no leakage. These decisions answer its findings. Details are in `REVIEW_LOG.md`.

**G1.1 History before 2016.** The brief filters the pull to 2016 onward. With that filter, 60% of 2016 rows looked like a restaurant's first inspection, and `prior_n` meant something different every year. Chose to pull all years (from April 2006) and use the inspections before 2016 as history only. The model table and the split still start in 2016. Alternative: drop 2016 from training as a burn-in year, which would change the fixed split. After the change, `is_first` is 5 to 6% in every year.

**G1.2 Routine-only history features.** Half of all routine inspections are preceded by a passed re-inspection, so `last_fail` was 1 on only 3% of rows and pointed the wrong way. Added `last_routine_fail`, `last_routine_n_viol`, `last_routine_n_lvl3` and `prior_routine_fail_rate`, which look back to the last routine inspection. The brief's original features stay. Alternative: leave this to the improvement loop. The reviewer rated it a major finding, so it is fixed here. On validation, M1's hit rate went from 87.8% to 90.5%.

**G1.3 Windowed count.** Added `prior_n_3y`, the number of inspections in the previous three years. Unlike `prior_n`, it does not grow with calendar time (about 7 to 8 in every year). `prior_n` stays, as the brief lists it.

**G1.4 A frozen, checkable snapshot.** The pull now has an upper bound (`resultdttm < '2026-11-11'`), and its metadata records the source, the license and a fingerprint of the content. `docs/pull_metadata.json` is the committed reference. A fresh pull is compared with it and prints a warning if the city has edited rows since. Alternative: commit the data, which the brief forbids.

**G1.5 `licstatus` leaves the model table.** It describes the license today, so it would be future information as a feature. It stays in `inspections_full.parquet` for the priority list. `models.py` also refuses any outcome-side column as a feature. The other static fields (`zip`, `neighborhood`, type) are also as of the pull date. We accept that, because a restaurant's location and type rarely change, and the README says so.

**G1.6 Smaller fixes.** Dates are converted to Boston time before the calendar date is taken. 224 exact duplicate rows are dropped. The leakage check now compares all 13 history columns with a by-hand recomputation for 300 random restaurants. `requirements.txt` is pinned. The lecture paths in D0.2 no longer show the local user name.
