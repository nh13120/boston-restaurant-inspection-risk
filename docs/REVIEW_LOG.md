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
