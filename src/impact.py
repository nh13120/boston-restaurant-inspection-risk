"""Translate model performance into things ISD can act on.

All functions take the true outcomes and the risk scores of a set of past inspections
(the test period) and return plain numbers: failures found, inspections saved, weeks gained.
"""
import numpy as np
import pandas as pd

from src.evaluate import K, _expected_ranked, evaluate

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
    found = np.cumsum(_expected_ranked(y, score_model))
    return int(np.searchsorted(found, target_failures) + 1)


def first_visits_hit_rate(dates, y, score=None, k=K, window="Q"):
    """Hit rate of the first k share of visits in each window (calendar quarter).

    With `score=None` the first visits are the ones ISD actually made first (by date): a rough
    picture of today's order. With a score, they are the riskiest k share of that window.
    """
    df = pd.DataFrame({"date": pd.to_datetime(pd.Series(dates)).to_numpy(), "y": np.asarray(y, dtype=float)})
    # "Earliest first" is a ranking like any other: the earlier the date, the higher the score.
    # evaluate() averages over visits made on the same day, as it does for any tied score.
    df["rank_score"] = -df["date"].astype("int64").astype(float) if score is None else np.asarray(score, dtype=float)
    hits = visits = 0.0
    for _, w in df.groupby(df["date"].dt.to_period(window)):
        n_top = int(round(k * len(w)))
        hits += evaluate(w["y"].to_numpy(), w["rank_score"].to_numpy(), k)["precision_at_k"] * n_top
        visits += n_top
    return hits / visits


def weeks_earlier(dates, y, score, window="Q"):
    """Time to discovery.

    Keep the number of inspections ISD actually made each week, but within each window
    (calendar quarter by default) hand out the visit slots in risk order instead of the
    actual order. Returns how many weeks earlier a visit that ends in a failure happens, on
    average, and how many weeks later a visit that ends in a pass happens.

    Assumption: the outcome of a visit does not depend on which week of the window it happens.
    """
    df = pd.DataFrame({"date": pd.to_datetime(pd.Series(dates)).to_numpy(), "y": np.asarray(y),
                       "score": np.asarray(score, dtype=float)})
    df["window"] = df["date"].dt.to_period(window)
    gained = []
    for _, w in df.groupby("window"):
        actual_slots = np.sort(w["date"].to_numpy())                 # the visit slots that existed
        by_risk = w.sort_values("score", ascending=False, kind="stable")
        new_date = pd.Series(actual_slots, index=by_risk.index)       # riskiest gets the earliest slot
        gained.append((w["date"] - new_date.loc[w.index]).dt.days)
    gained = pd.concat(gained)
    fails, passes = gained[df["y"] == 1], gained[df["y"] == 0]
    return {"mean_weeks_earlier_failures": fails.mean() / 7, "median_weeks_earlier_failures": fails.median() / 7,
            "mean_weeks_later_passes": -passes.mean() / 7,
            "n_failures": int(len(fails)), "n_windows": int(df["window"].nunique())}


def impact_table(y, dates, scores, best, k=K, diff_ci=None):
    """One tidy table of impact numbers. `scores` = {model name: test scores}; `best` = the final model.
    `diff_ci` = {other model: (low, high)} for the hit-rate difference best minus other, to carry
    the uncertainty over to the yearly counts."""
    y = np.asarray(y)
    per_year, years = yearly_volume(dates)
    visits = k * per_year
    fail_rate = y.mean()
    found = {name: failures_found_per_year(y, s, per_year, k) for name, s in scores.items()}
    # Random order: averaged over every possible random order, the hit rate equals the fail rate.
    found_random = fail_rate * visits

    # Inspections needed by the final model to find what the rule (B1) finds with the full capacity.
    n_test_capacity = int(round(k * len(y)))
    b1_found_in_test = evaluate(y, scores["B1_past_fail_rate"], k)["precision_at_k"] * n_test_capacity
    needed_in_test = inspections_to_match(y, scores[best], b1_found_in_test)
    saved_per_year = (n_test_capacity - needed_in_test) / years

    # A rough picture of today's order: the visits ISD actually made first in each quarter.
    actual_first = first_visits_hit_rate(dates, y, None, k)
    model_in_quarters = first_visits_hit_rate(dates, y, scores[best], k)

    first = f"in the first {k:.0%} of visits"
    rows = [
        ("Routine inspections per year", per_year, f"{len(y):,} test inspections over {years:.2f} years"),
        ("Fail rate in the test period", fail_rate, "share of test inspections that failed"),
        ("Failed inspections per year in total, in any visit order", fail_rate * per_year,
         "fail rate x inspections per year; visiting in a different order does not change this total"),
        (f"Visits per year in the first {k:.0%}", visits, f"{k:.0%} x inspections per year"),
        (f"Failed inspections found per year {first}, random order", found_random,
         "fail rate x visits (average over all random orders; ASSUMES today's schedule is no better than random)"),
        (f"Failed inspections found per year {first}, past-failure-rate rule (B1)", found["B1_past_fail_rate"],
         "B1 test hit rate x visits"),
        (f"Failed inspections found per year {first}, {best}", found[best], f"{best} test hit rate x visits"),
        (f"Extra failed inspections found per year {first} vs. random order", found[best] - found_random,
         "difference of the lines above: failures found EARLIER, not additional failures"),
        (f"Extra failed inspections found per year {first} vs. the rule (B1)", found[best] - found["B1_past_fail_rate"],
         "difference of the lines above: failures found EARLIER, not additional failures"),
        ("Fewer visits needed per year to find as many failed inspections as the rule (B1) finds in its first 20%", saved_per_year,
         f"{best} needs {needed_in_test:,} test visits to find the {b1_found_in_test:.0f} failures B1 finds in {n_test_capacity:,}"),
        (f"Hit rate of the visits ISD actually made first (first {k:.0%} of each quarter, by date)", actual_first,
         "a rough picture of today's order; seasonality within a quarter can also move it"),
        (f"Extra failed inspections found per year {first} vs. ISD's actual order", (model_in_quarters - actual_first) * visits,
         f"({best} ranked within each quarter, hit rate {model_in_quarters:.3f}, minus the line above) x visits"),
    ]
    out = pd.DataFrame(rows, columns=["quantity", "value", "how_computed"])
    out["ci_lo"], out["ci_hi"] = np.nan, np.nan
    for other, label in [("B0_random", "vs. random order"), ("B1_past_fail_rate", "vs. the rule (B1)")]:
        if diff_ci and other in diff_ci:
            row = out["quantity"].str.startswith("Extra failed") & out["quantity"].str.endswith(label)
            out.loc[row, ["ci_lo", "ci_hi"]] = [diff_ci[other][0] * visits, diff_ci[other][1] * visits]

    if COST_PER_FAILURE_LOW is not None and COST_PER_FAILURE_HIGH is not None:
        extra = found[best] - found_random
        out.loc[len(out)] = ("Value of failed inspections found earlier per year vs. random, low",
                             extra * COST_PER_FAILURE_LOW, "extra failures x COST_PER_FAILURE_LOW", np.nan, np.nan)
        out.loc[len(out)] = ("Value of failed inspections found earlier per year vs. random, high",
                             extra * COST_PER_FAILURE_HIGH, "extra failures x COST_PER_FAILURE_HIGH", np.nan, np.nan)
    return out[["quantity", "value", "ci_lo", "ci_hi", "how_computed"]]


def severity_in_first_visits(n_serious, score, k=K):
    """How much of the serious (level 3) violations the first k share of visits finds.

    `n_serious` is the number of level 3 violations each inspection FOUND. It is an outcome,
    used here only to measure results, never as a feature. Returns (share of all serious
    violations found in the first k, share of those visits that find at least one).
    """
    n_serious = np.asarray(n_serious, dtype=float)
    n_top = int(round(k * len(n_serious)))
    counts = _expected_ranked(n_serious, score)[:n_top]
    any_serious = _expected_ranked((n_serious > 0).astype(float), score)[:n_top]
    return counts.sum() / n_serious.sum(), any_serious.mean()


def plot_share_bar(shares, best, title, ylabel, footnote, path=None):
    """Three bars (random order, the rule, the model) for a share between 0 and 1."""
    import textwrap

    import matplotlib.pyplot as plt
    from src.evaluate import FIGURE_LABELS, MODEL_COLORS, set_plot_style

    set_plot_style()
    names = ["B0_random", "B1_past_fail_rate", best]
    fig, ax = plt.subplots(figsize=(9, 5.8))
    bars = ax.bar([FIGURE_LABELS[n] for n in names], [shares[n] * 100 for n in names],
                  color=[MODEL_COLORS[n] for n in names])
    for bar, n in zip(bars, names):
        ax.text(bar.get_x() + bar.get_width() / 2, shares[n] * 100 + 1, f"{shares[n]:.0%}", ha="center",
                fontsize=15, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, max(shares[n] for n in names) * 100 * 1.25)
    ax.set_title(title)
    ax.text(0.0, -0.3, textwrap.fill(footnote, 98), transform=ax.transAxes, fontsize=11, va="top")
    fig.tight_layout()
    if path:
        fig.savefig(path, bbox_inches="tight")
    return fig


def capacity_table(y, scores, capacities=(0.05, 0.10, 0.20, 0.30, 0.50)):
    """Hit rate of each method at several capacities, with random order and the best possible."""
    y = np.asarray(y)
    rows = []
    for k in capacities:
        row = {"capacity": k, "random_order": y.mean(), "best_possible": min(1.0, y.mean() / k)}
        row.update({name: evaluate(y, s, k)["precision_at_k"] for name, s in scores.items() if name != "B0_random"})
        rows.append(row)
    return pd.DataFrame(rows)


def calibration_table(y, score, bins=10):
    """Does a predicted risk of 80% fail about 80% of the time? Ten equal groups by predicted risk."""
    df = pd.DataFrame({"y": np.asarray(y), "score": np.asarray(score, dtype=float)})
    df["group"] = pd.qcut(df["score"].rank(method="first"), bins, labels=False) + 1
    return df.groupby("group").agg(n=("y", "size"), mean_predicted_risk=("score", "mean"),
                                   observed_fail_rate=("y", "mean")).reset_index()


def first_visit_probability(score, k=K):
    """Chance that each inspection is among the first k share of visits when ranked by `score`.
    1 above the cutoff, 0 below, and a fair share of the remaining slots for scores tied at it."""
    score = np.asarray(score, dtype=float)
    n_top = int(round(k * len(score)))
    cutoff = np.sort(score)[::-1][n_top - 1]
    above, tied = score > cutoff, score == cutoff
    return np.where(above, 1.0, np.where(tied, (n_top - above.sum()) / tied.sum(), 0.0))


def neighborhood_table(neighborhood, y, score, other=None, k=K, min_rows=200):
    """For each neighborhood: its fail rate, and the share of its inspections that land in the
    first k share of visits. `other` = {column label: another score} adds the same share under
    another ranking, for comparison (for example a rule that does not use location)."""
    df = pd.DataFrame({"neighborhood": np.asarray(neighborhood).astype(str), "y": np.asarray(y),
                       "in_first": first_visit_probability(score, k)})
    agg = {"inspections": ("y", "size"), "fail_rate": ("y", "mean"), "share_in_first_visits": ("in_first", "mean")}
    for label, other_score in (other or {}).items():
        df[label] = first_visit_probability(other_score, k)
        agg[label] = (label, "mean")
    out = df.groupby("neighborhood").agg(**agg).reset_index()
    hits = (df["y"] * df["in_first"]).groupby(df["neighborhood"]).sum() / df.groupby("neighborhood")["in_first"].sum()
    out["hit_rate_in_first_visits"] = hits.reindex(out["neighborhood"]).to_numpy()
    return out[out["inspections"] >= min_rows].sort_values("share_in_first_visits", ascending=False).reset_index(drop=True)


def after_a_pass(scored, table, k=K):
    """What happens after a restaurant PASSES: does a high risk score still mean something?

    `scored` = test inspections with columns licenseno, date, label_fail, score.
    `table`  = all routine inspections (the model table). For each test inspection that passed,
    we look up the same restaurant's next routine inspection and report how often it failed,
    separately for passes the model ranked in its riskiest k share and for the other passes.
    """
    nxt = table.sort_values(["licenseno", "date"])[["licenseno", "date", "label_fail"]].copy()
    nxt["next_date"] = nxt.groupby("licenseno")["date"].shift(-1)
    nxt["next_fail"] = nxt.groupby("licenseno")["label_fail"].shift(-1)
    df = scored.merge(nxt[["licenseno", "date", "next_date", "next_fail"]], on=["licenseno", "date"], how="left")
    df["high_risk"] = first_visit_probability(df["score"], k) > 0.5
    passed = df[(df["label_fail"] == 0) & df["next_fail"].notna()].copy()
    passed["days_to_next"] = (passed["next_date"] - passed["date"]).dt.days
    out = passed.groupby("high_risk").agg(passes_with_a_later_inspection=("next_fail", "size"),
                                          next_inspection_fail_rate=("next_fail", "mean"),
                                          median_days_to_next_inspection=("days_to_next", "median")).reset_index()
    out["group"] = np.where(out["high_risk"], f"passed, but ranked in the riskiest {k:.0%}", "passed, ranked lower")
    return out[["group", "passes_with_a_later_inspection", "next_inspection_fail_rate", "median_days_to_next_inspection"]]


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
    """One feature and its value, in words a manager can read. Returns None when the value is
    not something a reader would accept as a reason for a HIGH score ("previous visit passed")."""
    v = row.get(feature)
    missing = pd.isna(v)
    if feature == "prior_fail_rate":
        return None if missing else f"failed {v:.0%} of all past visits"
    if feature == "prior_routine_fail_rate":
        return None if missing else f"failed {v:.0%} of past routine inspections"
    if feature == "last_routine_n_viol":
        return None if missing else f"{v:.0f} violations at the last routine inspection"
    if feature == "last_routine_n_lvl3":
        return None if missing or v == 0 else f"{v:.0f} serious violations at the last routine inspection"
    if feature == "last_routine_fail":
        return "failed the last routine inspection" if v == 1 else None
    if feature == "last_fail":
        return "previous visit failed" if v == 1 else None
    if feature == "last_n_viol":
        return None if missing or v == 0 else f"{v:.0f} violations at the previous visit"
    if feature == "last_n_lvl3":
        return None if missing or v == 0 else f"{v:.0f} serious violations at the previous visit"
    if feature == "days_since_last":
        return "never inspected before" if missing else f"{v:.0f} days since the last visit"
    if feature == "license_age_days":
        return f"on record for {v / 365.25:.1f} years"
    if feature == "prior_n":
        return f"{v:.0f} past visits on record"
    if feature == "prior_n_3y":
        return f"{v:.0f} visits in the last 3 years"
    if feature == "month":
        return f"time of year ({MONTH_NAMES[int(v) - 1]})"
    if feature == "is_takeout":
        return "offers take-out" if v == 1 else None
    if feature == "is_first":
        return "first inspection on record" if v == 1 else None
    if feature == "location":
        return f"location ({row.get('neighborhood')}, zip {row.get('zip')})"
    if feature == "text":
        return "wording of the last inspection report"
    return f"{feature} = {v}"


def top_drivers(model, X, rows, text_features=(), n=3):
    """For each row, the n inputs that push its score up the most, in plain words.

    Uses LightGBM's built-in contributions (pred_contrib): how much each feature moved this
    restaurant's score away from the average. These are factors in the model's score. They
    are not causes, and not findings about the business.
    """
    contrib = pd.DataFrame(model.predict_proba(X, pred_contrib=True)[:, :-1], columns=X.columns, index=X.index)
    # Columns that describe one thing are added up into one factor.
    groups = {"text": [c for c in text_features if c in contrib.columns],
              "location": [c for c in ("zip", "neighborhood") if c in contrib.columns]}
    for name, cols in groups.items():
        if cols:
            contrib[name] = contrib[cols].sum(axis=1)
            contrib = contrib.drop(columns=cols)
    out = []
    for idx, c in contrib.iterrows():
        ranked = c[c > 0].sort_values(ascending=False).index
        words = [d for d in (_describe(f, rows.loc[idx]) for f in ranked) if d]
        out.append("; ".join(words[:n]))
    return out


def plot_impact_bar(impact, best, title, path=None, k=K):
    """Failed inspections found in the first k share of visits: random order vs. the rule vs. the model."""
    import matplotlib.pyplot as plt
    from src.evaluate import FIGURE_LABELS, MODEL_COLORS, set_plot_style

    set_plot_style()
    value = impact.set_index("quantity")["value"]
    found = {q.rsplit(", ", 1)[1]: v for q, v in value.items() if q.startswith("Failed inspections found per year in the first")}
    rows = [(FIGURE_LABELS["B0_random"], found["random order"], MODEL_COLORS["B0_random"]),
            (FIGURE_LABELS["B1_past_fail_rate"], found["past-failure-rate rule (B1)"], MODEL_COLORS["B1_past_fail_rate"]),
            (FIGURE_LABELS[best], found[best], MODEL_COLORS[best])]
    visits = value[[q for q in value.index if q.startswith("Visits per year")][0]]
    total = value["Failed inspections per year in total, in any visit order"]

    fig, ax = plt.subplots(figsize=(9, 5.8))
    bars = ax.bar([r[0] for r in rows], [r[1] for r in rows], color=[r[2] for r in rows])
    for bar, (_, v, _) in zip(bars, rows):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 12, f"{v:,.0f}", ha="center", fontsize=15, fontweight="bold")
    extra_random = rows[2][1] - rows[0][1]
    extra_actual = value[[q for q in value.index if q.endswith("vs. ISD's actual order")][0]]
    extra_rule = rows[2][1] - rows[1][1]
    lo, hi = sorted([round(extra_actual, -1), round(extra_random, -1)])
    ax.text(0.02, 0.97, f"About {lo:,.0f} to {hi:,.0f} more than the assumed current order,\n"
                        f"and {extra_rule:,.0f} more than the simple rule",
            transform=ax.transAxes, va="top", fontsize=13, color=MODEL_COLORS[best], fontweight="bold")
    ax.set_ylabel(f"Failed inspections found per year\nin the first {k:.0%} of visits")
    ax.set_ylim(0, max(r[1] for r in rows) * 1.32)
    ax.set_title(title)
    ax.text(0.0, -0.3, f"Each bar: the first {visits:,.0f} of about {visits / k:,.0f} routine inspections a year. Over the whole year the total\n"
                        f"(about {total:,.0f} failed inspections) is the same in any order. The gain is finding them earlier.",
            transform=ax.transAxes, fontsize=11)
    fig.tight_layout()
    if path:
        fig.savefig(path, bbox_inches="tight")
    return fig
