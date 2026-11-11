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
