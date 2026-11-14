# What is in this folder

Every file here is written by a notebook, so a full run regenerates all of them. Files ending in `_val`, or starting with `validation_`, come from the validation year (2023), which was used to choose settings. All other tables come from the test years (2024 to November 2026), which were scored once after the settings were frozen.

In the tables, `precision_at_k` is the hit rate: the share of the first 20% of visits that find a failure. `catch_at_k` is the share of all failures those visits find. Columns ending in `_lo` and `_hi` are the ends of a 95% interval.

## Headline results (test years)

| File | What it holds | Notebook |
|---|---|---|
| `results.csv` | The five planned models: hit rate, share of failures found and AUC, with intervals | 04 |
| `finetune_test.csv` | The same table with the fine-tuned text model (M3) and a text-only model added | 05 |
| `model_differences.csv` | Differences between pairs of models, with intervals and a verdict | 04 |
| `results_by_year.csv` | Hit rate and extra failures found, for each test year | 04 |
| `robustness.csv` | The same metrics on all test inspections and on two stricter slices | 04 |
| `rule_sensitivity.csv` | A second hand rule (most violations at the last routine inspection first) against the final model | 04 |
| `permutation_importance.csv` | How far the hit rate falls when each input is scrambled: what each input adds to the ranking | 04 |
| `feature_importance.csv` | How much the final model used each input while training (LightGBM split gain). It favors inputs with many values, such as zip code, so use `permutation_importance.csv` for what each input adds | 04 |

## Impact

| File | What it holds | Notebook |
|---|---|---|
| `impact.csv` | Every impact number quoted in the README, with a line on how it was computed | 04 |
| `capacity_sensitivity.csv` | Hit rates when the first 5%, 10%, 30% or 50% of visits are counted, not 20% | 04 |
| `calibration.csv` | Predicted risk against the observed fail rate, in ten groups | 04 |
| `after_a_pass.csv` | What happens next to restaurants that passed but ranked as risky | 04 |
| `neighborhood_table.csv` | The share of each neighborhood's inspections that land in the first visits: under the model, under the same model with location scrambled, and under a hand rule that ignores location | 04 |
| `priority_list.csv` | Active restaurants ranked by estimated risk. Read `priority_list_note.md` first | 04 |
| `priority_list_note.md` | How to read the priority list, and what it does not mean | 04 |

## Choosing settings (validation year)

| File | What it holds | Notebook |
|---|---|---|
| `validation_baselines.csv` | The three baselines on 2023 | 02 |
| `rule_sensitivity_val.csv` | The two hand rules on 2023 | 02 |
| `lgbm_grid_val.csv` | Every LightGBM setting that was tried | 03 |
| `validation_results.csv` | The five planned models on 2023 | 03 |
| `validation_differences.csv` | Differences between pairs of models on 2023 | 03 |
| `feature_importance_val.csv` | Split gain for the models fitted for validation (same caveat as `feature_importance.csv`) | 03 |
| `improvement_loop_val.csv` | The one extra feature idea that was tested, and whether it was kept | 03 |
| `location_sensitivity_val.csv` | The final model without zip code and neighborhood | 03 |
| `text_ablation_val.csv` | What the comment text adds, in several variants | 03 |
| `text_examples.md` | The inspections where the text model and the history model disagree most | 03 |
| `finetune_val.csv` | The fine-tuned text model against the others on 2023 | 05 |
| `finetune_like_for_like_val.csv` | Two choices of which comments to feed the text model | 05 |
| `selected_settings.json` | The settings chosen on 2023. Identical to the frozen copy in `docs/FROZEN_CONFIG.json` | 03 |
| `m3_selected_settings.json` | The same for M3. Identical to `docs/FROZEN_CONFIG_M3.json` | 05 |

## Figures

| File in `figures/` | What it shows | Notebook |
|---|---|---|
| `hit_rate_bar.png` | Hit rate of each method on the test years | 04 |
| `impact_bar.png` | Failed inspections found per year in the first 20% of visits | 04 |
| `serious_violations_bar.png` | Share of serious violations found in the first 20% of visits | 04 |
| `text_value_bar.png` | What the inspector comments add on top of the history | 04 |
| `neighborhood_first_visits.png` | Share of first visits by neighborhood, with location as is and scrambled | 04 |
| `feature_importance.png` | How far the hit rate falls when each input is scrambled | 04 |
| `gains_curve_test.png` | Failures found as more visits are made, test years | 04 |
| `gains_curve_val.png` | The same curve on 2023, all models | 03 |
| `gains_curve_val_baselines.png` | The same curve on 2023, baselines only | 02 |
| `reinspection_gap.png` | Days between a failed inspection and the next visit, which supports the 30-day cutoff | 01 |
