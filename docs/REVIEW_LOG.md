# Review log

Each gate is reviewed by a fresh subagent that receives only the file paths, `SPEC_CHECKLIST.md` and the gate checklist. Findings are tagged BLOCKER, MAJOR or MINOR. Every BLOCKER and MAJOR is fixed, and a MINOR is fixed when it takes under five minutes.

## Gate 1: data and leakage audit

### Round 1

Verdict: 0 BLOCKER, 4 MAJOR, 8 MINOR. All six checklist items passed except privacy. The reviewer recomputed the nine history features from the raw snapshot for all 58,603 rows and found no mismatch.

| # | Tag | Finding | What we did |
|---|---|---|---|
| 1 | MAJOR | At least 34 full names, about 35 role-plus-first-name mentions and 13 emails written with spaces escaped the masking. The notebook's privacy check used the same patterns as the masker, so it could not see them. | Rewrote the masker: names after titles and certificates, first names from a name list, full names anywhere, emails with spaces, phones without separators. The notebook now scans with looser patterns and reports what remains. An independent dictionary scan finds no residual full names. |
| 2 | MAJOR | `last_fail`, `last_n_viol` and `last_n_lvl3` carried almost no signal, because the previous visit is usually a passed re-inspection. | Added `last_routine_fail`, `last_routine_n_viol`, `last_routine_n_lvl3` and `prior_routine_fail_rate`. Kept the original features. |
| 3 | MAJOR | History was cut at 2016, so 60% of 2016 rows had `is_first = 1` and `prior_n` drifted from 0.9 to 20.7 across years. | Pulled all years since 2006 as history only. Added the windowed count `prior_n_3y`. `is_first` is now 5 to 6% in every year. |
| 4 | MAJOR | The pull had no upper date bound, no checksum and unpinned requirements, so a fresh clone could not reproduce the data. | Added `resultdttm < '2026-11-11'`, a content fingerprint with a committed reference (`docs/pull_metadata.json`), and pinned `requirements.txt`. |
| 5 | MINOR | Local paths with the macOS user name in `DECISIONS.md`. | Replaced with `~`. |
| 6 | MINOR | Dates were UTC dates, not Boston dates. | Converted to `America/New_York` before taking the date. |
| 7 | MINOR | License attributes are as-of-pull values, and `licstatus` sat in the model table. | Dropped `licstatus` from the model table, added a refusal list in `models.py`, documented the as-of-pull fields. |
| 8 | MINOR | The 30-day rule misclassifies 2 to 3% of rows (late follow-ups stay in). | Left the rule as the brief defines it. Notebook 04 adds a robustness check that removes likely late follow-ups. |
| 9 | MINOR | 154 exact duplicate raw rows. | `clean()` drops exact duplicates (224 in the full-history snapshot). |
| 10 | MINOR | The built-in leakage check covered 3 licenses and 3 columns. | Added a by-hand recomputation of all 13 history columns for 300 random restaurants. |
| 11 | MINOR | License not recorded; two small inaccuracies in the data dictionary. | Source and license are in the pull metadata and the dictionary; both lines corrected. |
| 12 | MINOR | Notebooks 02 and 03 had already run on the table under review. | Reran both after the fixes. |

Deliberately left: the 30-day cutoff itself (finding 8), which the brief fixes.

### Round 2

Same reviewer prompt, fresh subagent. Verdict: 0 BLOCKER, 1 MAJOR, 6 MINOR. The reviewer recomputed 20 feature columns and `prior_comments` for all 58,564 model rows and all 161,962 inspections with no mismatch, confirmed that the snapshot hash matches the reference and that `build_model_table` rebuilds both saved tables exactly, and found no escaped emails or phones.

| # | Tag | Finding | What we did |
|---|---|---|---|
| 1 | MAJOR | Name masking still missed at least 94 comments: common names absent from the list, names in capitals, "PIC Tom stated", "owner of restaurant: X Y". Notebook 01 said no personal details remained, which was false. | Added about 350 common first names, capitalized and lower-case matching in name contexts, five more contexts, and a cleanup for surnames left after a mask. With the reviewer's own name list, 3 name-like strings remain in 324,528 comment pieces. Rewrote the notebook text to say the masking is best effort. |
| 2 | MINOR | The raw snapshot holds unmasked comments. | It stays git-ignored. The README says `data/` must never be published. |
| 3 | MINOR | 1,807 model rows follow a failed visit by more than 30 days, and many are late re-inspections. | Limitation logged (D2.3). Notebook 04 reports the results without those rows. We did not filter on `n_rechecked_ok`, which is outcome-side. |
| 4 | MINOR | Comment text is sometimes edited after the inspection date (13 model rows mention a later date). | Logged as a limitation (G1.7). The final model does not use text. |
| 5 | MINOR | Some attributes are as of the pull date, and the feature refusal list was incomplete. | Added `license_issue_date`, `businessname`, `descript`, `inspection_id` and `licenseno` to `NEVER_FEATURES`. The dictionary notes the 2006 start of the record. |
| 6 | MINOR | The snapshot check trusted the metadata file. | `pull_inspections` now hashes the parquet content on load and compares it with `docs/pull_metadata.json`. |
| 7 | MINOR | "Pass" means "no follow-up required", not "no violations". | Stated in the data dictionary, and in the README. |

Deliberately left: the 30-day rule and the label definition, which the brief fixes.

### Round 3

Same prompt, fresh subagent. Verdict: 0 BLOCKER, 0 MAJOR, 7 MINOR. Gate 1 passed. The reviewer matched 22 numeric columns, the categorical columns and both comment columns on all 58,564 model rows and all 161,962 inspections, rebuilt both saved tables exactly, and verified the snapshot hash.

| # | Tag | Finding | What we did |
|---|---|---|---|
| 1 | MINOR | About 62 comments still carry a name (first names after a role word, "X stated", lower-case names around certificates). Tracked files are clean. | Added those three contexts, a lower-case honorific and a fix for a half-masked email. The dictionary now says "best effort". `data/` stays unpublished. |
| 2 | MINOR | Late follow-up visits remain in the model table (1,807 rows). | Limitation. Notebook 04 reports results without them. |
| 3 | MINOR | Notebook 01 prints the fail rate of the test years. | Left: the brief asks for the fail rate per split and per year. Added a note that these are label summaries and no decision uses them. |
| 4 | MINOR | One prior comment looks edited after the predicted inspection. | Limitation (G1.7). |
| 5 | MINOR | The "Eating & Drinking" filter is the license type on the pull date. | Stated in the data dictionary. |
| 6 | MINOR | Exact reproduction needs the original snapshot; Python version not recorded. | Stated in the README. |
| 7 | MINOR | 251 failed inspections have a violation count of 0. | Stated in the data dictionary. |

## Gate 2: methodology review

### Round 1

Verdict: 0 BLOCKER, 4 MAJOR, 6 MINOR. No leakage found: all 15 numeric features and `prior_comments` are identical for the 41,393 train and validation rows when every 2024+ row is removed first. No test file or test-period metric exists. The reviewer reproduced B1 and M1's validation metrics to four decimals.

| # | Tag | Finding | What we did |
|---|---|---|---|
| 1 | MAJOR | B1 was handicapped: first inspections got the routine fail rate (0.60), above 96% of real `prior_fail_rate` values, so all of them landed in B1's top 20%. Fair value on validation is 80.4%, not 68.9%. | Fallback is now the mean `prior_fail_rate` of the fitting rows. `evaluate()` takes the expected value over ties (G2.1, G2.2). |
| 2 | MAJOR | M2 is not reproducible across devices, and its selected settings flip on float noise. | Pinned the encoder revision. Documented the device caveat in the config, DECISIONS and README. M1, the final model, is unaffected (G2.4). |
| 3 | MAJOR | The settings to be frozen named no model to deploy, no fitting rows and no comparisons, and M1 was never tested against B2. | Added a pre-specified parsimony rule and the paired comparisons to notebook 03 and to the config (G2.3). |
| 4 | MAJOR | `text_examples.md` said "ranked much higher"; the largest move was about 5 points. | Rewritten with the measured moves and the overlap of the two top-20% lists (G2.5). |
| 5 | MINOR | Validation numbers of M1 and M2 are post-selection. | Stated in notebook 03. |
| 6 | MINOR | `license_age_days` switched on a pull-date value for 5% of rows. | Now counted from the first inspection on record only (D2.2). |
| 7 | MINOR | Notebook 01 prints test-year fail rates. | Left, with a note (see Gate 1 round 3, finding 3). |
| 8 | MINOR | B0 is one random draw and shared its random stream with the tie-breaker. | Ties no longer use a random draw in the metrics; curves use their own seed. B0 stays one random draw, as the brief asks, and the notebooks state that its expected hit rate is the fail rate. |
| 9 | MINOR | Late re-inspections remain in the table. | Limitation, as above. |
| 10 | MINOR | "12 history features" in a notebook text; there are 17. | Fixed. |

### Round 2

Same prompt, fresh subagent. Verdict: 0 BLOCKER, 1 MAJOR, 8 MINOR. All five prediction files reproduced exactly with the reviewer's own metric code. Refitting B2, M1 and M2 on the training rows gave identical scores, the train-only PCA was confirmed by exact refit, and the features matched a by-hand rebuild for all 5,521 validation rows.

| # | Tag | Finding | What we did |
|---|---|---|---|
| 1 | MAJOR | The ladder rule chose M1 over logistic regression on a margin of one inspection: the interval's lower bound was 0.0009 and moved to 0.0000 with another seed or with resampling by license. The rule was also written after the numbers were known. | Dropped the ladder. M1 is carried forward because the brief names it, unless M2 beats it beyond noise. Notebook 03 prints that M1's lead over logistic regression is not distinguishable from noise. All paired comparisons use 2,000 resamples of whole restaurants (G2.3, G2.6). |
| 2 | MINOR | The text ablation also differs in tuned settings. | Reworded, and added a same-settings comparison (text added to M1's settings: -0.3 points, CI -1.1 to +0.5). |
| 3 | MINOR | M2's lead over M1 is numeric noise and device-dependent. | Reported as "equal within noise". The device caveat is in the config and README. We keep `mps`, as the brief asks. |
| 4 | MINOR | Refitting the PCA on train + validation for the test run conflicts with a literal reading of "PCA on train only". | The brief asks for that refit. Stated in the frozen config and in G2.7. |
| 5 | MINOR | Test-year label summaries in notebook 01. | Left, as before: the brief asks for them. |
| 6 | MINOR | B0 was one random draw. | B0 now scores every row the same, and the metric averages over all random orders: hit rate equals the fail rate (G2.2). |
| 7 | MINOR | A one-feature rule beats B1 by 4 points. | Added as a sensitivity row in notebooks 02 and 04 (G2.8). |
| 8 | MINOR | Validation was reused across rounds, which the notebook did not say. | Notebook 03 now says every validation number is optimistic and only test counts. |
| 9 | MINOR | Tie-breaker seed 43; top-20% count off by one in the text examples. | The tie-breaker stream is derived from seed 42. The examples use the same cutoff as `evaluate`. |

### Round 3

Same prompt, fresh subagent. Verdict: 0 BLOCKER, 1 MAJOR, 7 MINOR. This was the third and last round, and with no blocker the gate is cleared. The reviewer's own past-only recomputation matched all 15 numeric features and `prior_comments` on all 41,393 train and validation rows, all four fitted models refit to the saved predictions exactly, and no test-scoring code or file exists in any commit.

| # | Tag | Finding | What we did |
|---|---|---|---|
| 1 | MAJOR | "M1 is not distinguishable from logistic regression" depended on the resample count: the interval's lower end sits on zero and the yes/no flag flipped in 4 of 6 settings. | Replaced the flag with three verdicts (beyond noise, borderline, within noise), raised paired resamples to 10,000, and wrote the wording rule into the frozen config. M1 against logistic regression now reads "borderline" (G2.9). |
| 2 | MINOR | Test protocol not fully pinned: pooled or per year, no B0 row in the differences, uncommitted `impact.py`. | The config fixes "pooled headline plus per-year breakdown". B0 is in the differences table. `impact.py` is committed before the freeze. |
| 3 | MINOR | Early stopping also watched log-loss. | `first_metric_only=True`. The selected settings did not change. |
| 4 | MINOR | The notebook did not state M2's device noise. | Stated next to the ablation in notebook 03, and in the README. |
| 5 | MINOR | Test-year fail rates printed in notebook 01. | The per-year table now stops at 2023. The split table keeps one line per split, as the brief asks. |
| 6 | MINOR | B0's interval is that of the fail rate; the gains figure used one random tie-break. | Gains curves now use the expected value over ties, so B0 is the exact diagonal and B1's curve agrees with the table. |
| 7 | MINOR | Three evidence standards for "keep it". | Notebook 03 says why the improvement loop's bar is higher. Model comparisons all use the three verdicts. |
| 8 | MINOR | The test-run refit changes what "training rows" means for the PCA. | Notebook 04 asserts that nothing is fitted on a row dated 2024 or later. The config and README state the definition. |

Gate 2 cleared. Settings frozen in `docs/FROZEN_CONFIG.json`.

## Gate 3: results and impact critic

### Round 1

Verdict: 0 BLOCKER, 2 MAJOR, 10 MINOR. The reviewer reproduced every number in `results/*.csv`, the README and the four figures from `preds/` and the labels with independent code, refitted M1 from the frozen settings and matched the saved test predictions exactly, and found no dollar figures and no tuning on the test years.

| # | Tag | Finding | What we did |
|---|---|---|---|
| 1 | MAJOR | "More failures found" read as net new failures, but the yearly total (about 3,544) is the same in any order. The 349 is a timing gain within the first 20% of visits. | Retitled the figure, renamed the `impact.csv` rows, added the unchanged total to the figure, the table and the README (G3.1). |
| 2 | MAJOR | "Random order (today)" was shown as fact. It is an assumption, and the visits ISD actually made first failed 61.7% of the time, not 59.0%. | Relabelled as "assumed status quo", stated in the README limitations, and added the actual-order comparison (+323) to `impact.csv` (G3.2). |
| 3 | MINOR | M2's lead over M1 on test was soft-pedalled. | The README and notebook now say: 0.7 points ahead, beyond noise by the frozen rule, borderline across seeds, about twice its device noise, kept M1 because the choice was fixed on validation. |
| 4 | MINOR | Impact numbers had no uncertainty, and 94 is the top of a range. | Intervals in `impact.csv`; README quotes 40 to 95 (G3.3). |
| 5 | MINOR | "1.1 weeks sooner" omitted its mirror image and its dependence on the window. | Both are in `impact.csv` (G3.3). |
| 6 | MINOR | The top 20% is mostly repeat visits; per restaurant the lead over B1 is 3.3 points. | Added the slice to `robustness.csv` and the README. Wording says "failed inspections" and "visits", not "failing restaurants". |
| 7 | MINOR | Gains chart not a 5-second read. | Three curves, one callout, plain names. |
| 8 | MINOR | Hit-rate chart had jargon and an overlapping label. | Plain names, the chosen model outlined, a note on the 95% ranges, the overlapping label removed. |
| 9 | MINOR | Importance chart overstated what it measures. | Top 8 inputs, axis "share of the model's use of each input", note "signals, not causes". |
| 10 | MINOR | Priority-list drivers included counter-intuitive and location-based reasons. | Column renamed `top_3_model_factors`, nonsense factors skipped, location flagged in the README and notebook (G3.4). |
| 11 | MINOR | README said "scored once"; notebook 04 ran more than once. | Reworded. D6.3 explains that reruns changed presentation and added robustness rows, with identical predictions. |
| 12 | MINOR | Interval ends quoted to 0.1 point from 500 resamples. | README rounds interval ends to whole points. |

Deliberately left: location stays in the model (changing inputs after the test would break the freeze), with a recommendation to ISD.

### Round 2

Same prompt, fresh subagent. Verdict: 0 BLOCKER, 0 MAJOR, 11 MINOR. Gate 3 passed. All 15 headline values matched to 1e-16, all 13 impact rows reproduced, M1 and B2 refitted from the frozen config gave the saved test predictions exactly, and the test predictions and `results.csv` are unchanged since the scoring commit.

| # | Tag | Finding | What we did |
|---|---|---|---|
| 1 | MINOR | The yearly gain (346, 407, 288) varies more than its interval (333 to 364) suggests. | Yearly gains added to `results_by_year.csv`, notebook 04 and the README, with a sentence on what the interval covers. |
| 2 | MINOR | The impact figure headlined only the most favourable comparison. | The callout now reads "about 320 to 350 more than the assumed current order, and 94 more than the simple rule". |
| 3 | MINOR | The "(chosen)" bar is not the tallest, and the figure did not say why. | Footnote added: chosen on 2023 data, the comment model's edge appeared only afterwards. |
| 4 | MINOR | "Borderline across bootstrap seeds" understated M2's lead. | Reworded: the interval excludes zero with every seed. |
| 5 | MINOR | "Inspections saved" contradicts "ISD makes every inspection". | Row renamed "fewer visits needed to find as many failed inspections as the rule". |
| 6 | MINOR | The 1.1 weeks was quoted without its window dependence. | README adds 0.4 within months, 4.4 within years, and that passing visits move 1.6 weeks later. |
| 7 | MINOR | Gains-curve axes said "the year's" for a pooled 2.86-year curve. | Axis labels corrected. |
| 8 | MINOR | Random order stated as fact in two places. | Reworded as an assumption in notebook 04 and `models.py`. |
| 9 | MINOR | 61.7% in the log against 61.6% in the table; same-day ties broken by row order. | Same-day ties are now averaged, as in `evaluate()`: 61.5% and +324. Log corrected. |
| 10 | MINOR | The location caveat did not say how cheap location is to remove. | README: dropping location cost about 0.3 points on validation. |
| 11 | MINOR | The priority list names real businesses with a risk percentage and no caveat in the file. | Column renamed `estimated_risk_pct`; `results/priority_list_note.md` sits next to the file. Business names stay, as the brief asks. Flagged for the team before the repo goes public. |
