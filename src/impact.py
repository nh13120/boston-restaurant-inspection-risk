"""Translate model performance into things ISD can act on.

All functions take the true outcomes and the risk scores of a set of past inspections
(the test period) and return plain numbers: failures found, inspections saved, weeks gained.
"""
import numpy as np
import pandas as pd

from src.evaluate import K, _ranked, evaluate

# Cost of a failure that goes undetected. We do not have a sourced figure, so these stay
# None and the dollar lines are skipped. Never fill them with a guess.
COST_PER_FAILURE_LOW = None
COST_PER_FAILURE_HIGH = None

MONTH_NAMES = ["January", "February", "March", "April", "May", "June", "July", "August",
               "September", "October", "November", "December"]


def yearly_volume(dates):
    """Inspections per year over the period covered by `dates` (period length in years)."""
    dates = pd.to_datetime(pd.Series(dates))
    years = ((dates.max() - dates.min()).days + 1) / 365.25
    return len(dates) / years, years


def failures_found_per_year(y, score, inspections_per_year, k=K):
    """Failures found per year when ISD visits the top k share by `score`."""
    visits = k * inspections_per_year
    return evaluate(y, score, k)["precision_at_k"] * visits


def inspections_to_match(y, score_model, target_failures):
    """Smallest number of visits, in the model's risk order, that finds `target_failures` failures."""
    found = np.cumsum(_ranked(y, score_model))
    return int(np.searchsorted(found, target_failures) + 1)


def weeks_earlier(dates, y, score, window="Q"):
    """Time to discovery.

    Keep the number of inspections ISD actually made each week, but within each window
    (calendar quarter by default) hand out the visit slots in risk order instead of the
    actual order. Returns how many weeks earlier a failing restaurant is visited, on average.

    Assumption: the outcome of a visit does not depend on which week of the window it happens.
    """
    df = pd.DataFrame({"date": pd.to_datetime(pd.Series(dates)).to_numpy(), "y": np.asarray(y),
                       "score": np.asarray(score, dtype=float)})
    df["window"] = df["date"].dt.to_period(window)
    shifts = []
    for _, w in df.groupby("window"):
        actual_slots = np.sort(w["date"].to_numpy())                 # the visit slots that existed
        by_risk = w.sort_values("score", ascending=False, kind="stable")
        new_date = pd.Series(actual_slots, index=by_risk.index)       # riskiest gets the earliest slot
        days_gained = (w["date"] - new_date.loc[w.index]).dt.days
        shifts.append(days_gained[w["y"] == 1])
    gained = pd.concat(shifts)
    return {"mean_weeks_earlier_failures": gained.mean() / 7, "median_weeks_earlier_failures": gained.median() / 7,
            "n_failures": int(len(gained)), "n_windows": int(df["window"].nunique())}


def impact_table(y, dates, scores, best, k=K):
    """One tidy table of impact numbers. `scores` = {model name: test scores}; `best` = name of the model to deploy."""
    y = np.asarray(y)
    per_year, years = yearly_volume(dates)
    visits = k * per_year
    fail_rate = y.mean()
    found = {name: failures_found_per_year(y, s, per_year, k) for name, s in scores.items()}
    # Random order: averaged over every possible random order, the hit rate equals the fail rate.
    found_random = fail_rate * visits

    # Inspections needed by the best model to find what the rule (B1) finds with the full capacity.
    n_test_capacity = int(round(k * len(y)))
    b1_found_in_test = evaluate(y, scores["B1_past_fail_rate"], k)["precision_at_k"] * n_test_capacity
    needed_in_test = inspections_to_match(y, scores[best], b1_found_in_test)
    saved_per_year = (n_test_capacity - needed_in_test) / years

    rows = [
        ("Routine inspections per year", per_year, "inspections", f"{len(y):,} test inspections over {years:.2f} years"),
        ("Fail rate in the test period", fail_rate, "share", "share of test inspections that failed"),
        (f"Visits per year at {k:.0%} capacity", visits, "inspections", f"{k:.0%} x inspections per year"),
        ("Failures found per year, random order", found_random, "failures", "fail rate x visits (expected value of a random order)"),
        ("Failures found per year, past-failure-rate rule (B1)", found["B1_past_fail_rate"], "failures", "B1 test hit rate x visits"),
        (f"Failures found per year, {best}", found[best], "failures", f"{best} test hit rate x visits"),
        ("Extra failures found per year vs. random order", found[best] - found_random, "failures", "difference of the two lines above"),
        ("Extra failures found per year vs. the rule (B1)", found[best] - found["B1_past_fail_rate"], "failures", "difference of the two lines above"),
        ("Inspections saved per year to match the rule (B1)", saved_per_year, "inspections",
         f"{best} needs {needed_in_test:,} test visits to find the {b1_found_in_test:.0f} failures B1 finds in {n_test_capacity:,}"),
    ]
    out = pd.DataFrame(rows, columns=["quantity", "value", "unit", "how_computed"])

    if COST_PER_FAILURE_LOW is not None and COST_PER_FAILURE_HIGH is not None:
        extra = found[best] - found_random
        out.loc[len(out)] = ("Value of extra failures found per year vs. random, low", extra * COST_PER_FAILURE_LOW,
                             "dollars", "extra failures x COST_PER_FAILURE_LOW")
        out.loc[len(out)] = ("Value of extra failures found per year vs. random, high", extra * COST_PER_FAILURE_HIGH,
                             "dollars", "extra failures x COST_PER_FAILURE_HIGH")
    return out


# --- Priority list -------------------------------------------------------------
def next_inspection_rows(full, as_of, add_history_features, reinspection_days=30, active_within_days=730):
    """One row per active restaurant describing its NEXT inspection as of `as_of`.

    We append a placeholder inspection dated `as_of` to each restaurant's history and reuse the
    same feature code as training, so the features mean exactly the same thing.
    """
    as_of = pd.Timestamp(as_of).normalize()
    last = full.sort_values("date").groupby("licenseno").tail(1)
    recent = (as_of - last["date"]).dt.days <= active_within_days
    active = last[(last["licstatus"] == "Active") & recent].copy()

    placeholder = active.copy()
    placeholder["date"] = as_of
    placeholder["inspection_id"] = placeholder["licenseno"] + "_next"
    # Outcome-side columns are unknown for a future visit. They are never used as features.
    placeholder[["label_fail", "n_viol", "n_lvl1", "n_lvl2", "n_lvl3", "n_rechecked_ok"]] = 0
    placeholder["comments"] = ""

    base_cols = [c for c in placeholder.columns if c in full.columns]
    history = full[full["licenseno"].isin(active["licenseno"])][base_cols]
    combined = add_history_features(pd.concat([history, placeholder[base_cols]], ignore_index=True))
    nxt = combined[combined["inspection_id"].str.endswith("_next")].copy()

    # A restaurant that failed in the last 30 days is due a re-inspection anyway: not a ranking decision.
    due_reinspection = (nxt["last_fail"] == 1) & (nxt["days_since_last"] <= reinspection_days)
    print(f"priority list: {len(nxt):,} active restaurants; {int(due_reinspection.sum())} already due a "
          f"re-inspection are left out")
    return nxt[~due_reinspection].reset_index(drop=True)


def _describe(feature, row):
    """One feature and its value, in words a manager can read."""
    v = row.get(feature)
    if feature == "prior_fail_rate":
        return "no inspection history" if pd.isna(v) else f"failed {v:.0%} of past inspections"
    if feature == "last_fail":
        return "no previous inspection" if pd.isna(v) else ("failed the previous inspection" if v == 1 else "passed the previous inspection")
    if feature == "days_since_last":
        return "never inspected before" if pd.isna(v) else f"{v:.0f} days since the last visit"
    if feature == "license_age_days":
        return f"license held for {v / 365.25:.1f} years"
    if feature == "prior_n":
        return f"{v:.0f} past inspections on record"
    if feature == "last_n_viol":
        return "no previous inspection" if pd.isna(v) else f"{v:.0f} violations at the previous visit"
    if feature == "last_n_lvl3":
        return "no previous inspection" if pd.isna(v) else f"{v:.0f} serious violations at the previous visit"
    if feature == "month":
        return f"time of year ({MONTH_NAMES[int(v) - 1]})"
    if feature == "is_takeout":
        return "offers take-out" if v == 1 else "no take-out"
    if feature == "is_first":
        return "first inspection on record" if v == 1 else "has been inspected before"
    if feature == "location":
        return f"location ({row.get('neighborhood')}, zip {row.get('zip')})"
    if feature == "text":
        return "wording of the last inspection report"
    return f"{feature} = {v}"


def top_drivers(model, X, rows, text_features=(), n=3):
    """For each row, the n features that push its risk up the most, in plain words.

    Uses LightGBM's built-in contributions (pred_contrib): how much each feature moved this
    restaurant's score away from the average.
    """
    contrib = pd.DataFrame(model.predict_proba(X, pred_contrib=True)[:, :-1], columns=X.columns, index=X.index)
    # Columns that describe one thing are added up into one driver.
    groups = {"text": [c for c in text_features if c in contrib.columns],
              "location": [c for c in ("zip", "neighborhood") if c in contrib.columns]}
    for name, cols in groups.items():
        if cols:
            contrib[name] = contrib[cols].sum(axis=1)
            contrib = contrib.drop(columns=cols)
    out = []
    for idx, c in contrib.iterrows():
        top = c[c > 0].sort_values(ascending=False).head(n)
        out.append("; ".join(_describe(f, rows.loc[idx]) for f in top.index))
    return out
