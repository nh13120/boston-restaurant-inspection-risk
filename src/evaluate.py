"""Metrics for ranking inspections by risk, with capacity fixed at the top k share.

precision_at_k (the "hit rate") = share of the visited inspections that find a failure.
catch_at_k                      = share of all failures that the visited inspections find.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata

SEED = 42
K = 0.20   # ISD capacity: visit the riskiest 20% first

# One display name and one colour per model, shared by every figure.
MODEL_LABELS = {
    "B0_random": "Random order (today)",
    "B1_past_fail_rate": "Past failure rate (rule)",
    "B2_logreg": "Logistic regression",
    "M1_lgbm": "LightGBM, history",
    "M2_lgbm_text": "LightGBM, history + text",
}
MODEL_COLORS = {
    "B0_random": "#9a9a9a", "B1_past_fail_rate": "#dd8452", "B2_logreg": "#8172b3",
    "M1_lgbm": "#4c72b0", "M2_lgbm_text": "#2a9d8f",
}


def _auc(y, score):
    """Area under the ROC curve from ranks (the Mann-Whitney formula; tied scores get the
    average rank). Same value as sklearn's roc_auc_score, several times faster, which matters
    for 10,000 bootstrap resamples."""
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    return (rankdata(score)[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def _expected_ranked(y, score):
    """Labels sorted from highest to lowest score. Inspections that share a score are replaced
    by their group's fail rate: the average over every random order of the tied rows. A rule
    with many tied scores (such as B1, or the constant B0) gets no credit from row order."""
    y, score = np.asarray(y, dtype=float), np.asarray(score, dtype=float)
    order = np.argsort(-score, kind="stable")
    y_sorted, s_sorted = pd.Series(y[order]), score[order]
    return y_sorted.groupby(s_sorted, sort=False).transform("mean").to_numpy()


def evaluate(y, score, k=K):
    """AUC plus the two capacity metrics at the top k share.

    When several inspections share the score at the cutoff, we take the expected number of
    failures over every random way of breaking the tie. A rule with many tied scores (such as
    B1) is then judged by its average, not by one lucky or unlucky draw.
    """
    y, score = np.asarray(y), np.asarray(score, dtype=float)
    n = len(y)
    n_top = max(1, int(round(k * n)))
    cutoff = np.sort(score)[::-1][n_top - 1]            # score of the last visit made
    above, tied = score > cutoff, score == cutoff
    hits = y[above].sum() + (n_top - above.sum()) * y[tied].mean()
    return {
        "auc": _auc(y, score),
        "catch_at_k": hits / y.sum(),
        "precision_at_k": hits / n_top,
        "n": n,
    }


def gains_curve(y, score):
    """Share of failures caught as a function of the share of inspections made (expected value
    over ties, so it agrees with evaluate() at every capacity)."""
    y_sorted = _expected_ranked(y, score)
    n = len(y_sorted)
    return pd.DataFrame({
        "share_inspected": np.arange(0, n + 1) / n,
        "share_failures_caught": np.concatenate([[0], np.cumsum(y_sorted)]) / y_sorted.sum(),
    })


METRICS = ["auc", "catch_at_k", "precision_at_k"]


def _resamples(n_rows, n, seed, groups=None):
    """Yield n bootstrap samples of row positions.

    Without `groups`, rows are drawn with replacement. With `groups` (one label per row, here
    the license number), whole restaurants are drawn with replacement and all their inspections
    come along. That is the honest unit: inspections of one restaurant are not independent.
    """
    rng = np.random.default_rng(seed)
    if groups is None:
        for _ in range(n):
            yield rng.integers(0, n_rows, n_rows)
        return
    codes = pd.factorize(np.asarray(groups))[0]
    order = np.argsort(codes, kind="stable")
    rows_of = np.split(order, np.cumsum(np.bincount(codes))[:-1])    # row positions of each restaurant
    for _ in range(n):
        picked = rng.integers(0, len(rows_of), len(rows_of))
        yield np.concatenate([rows_of[g] for g in picked])


def bootstrap_all(y, score, n=500, seed=SEED, k=K, groups=None):
    """95% confidence interval of every metric: {metric: (low, high)}."""
    y, score = np.asarray(y), np.asarray(score, dtype=float)
    values = pd.DataFrame([evaluate(y[idx], score[idx], k) for idx in _resamples(len(y), n, seed, groups)])
    return {m: tuple(np.percentile(values[m], [2.5, 97.5])) for m in METRICS}


def bootstrap_ci(y, score, metric, n=500, seed=SEED, k=K, groups=None):
    """95% confidence interval for one metric: resample n times (see _resamples)."""
    return bootstrap_all(y, score, n=n, seed=seed, k=k, groups=groups)[metric]


def paired_differences(y, score_a, score_b, n=10_000, seed=SEED, k=K, groups=None):
    """metric(a) - metric(b) with a 95% CI, using the SAME resamples for both models.

    Returns {metric: (difference, low, high)}.
    """
    y, score_a, score_b = np.asarray(y), np.asarray(score_a, dtype=float), np.asarray(score_b, dtype=float)
    point_a, point_b = evaluate(y, score_a, k), evaluate(y, score_b, k)
    diffs = []
    for idx in _resamples(len(y), n, seed, groups):
        a, b = evaluate(y[idx], score_a[idx], k), evaluate(y[idx], score_b[idx], k)
        diffs.append({m: a[m] - b[m] for m in METRICS})
    diffs = pd.DataFrame(diffs)
    return {m: (point_a[m] - point_b[m], *np.percentile(diffs[m], [2.5, 97.5])) for m in METRICS}


def verdict(diff, lo, hi):
    """Words for a paired difference, fixed before the test was scored.

    Look at the end of the 95% interval nearest to zero, on the side of the difference:
      - "beyond noise": that end is clearly away from zero (by more than 10% of the interval's width);
      - "borderline":   that end is about zero (within 10% of the width, either side);
      - "within noise": the interval comfortably contains zero.
    A yes/no flag on "interval excludes zero" would flip with the number of resamples when the
    end sits on zero. The middle category avoids that.
    """
    margin = 0.1 * (hi - lo)
    near = lo if diff >= 0 else -hi
    if near > margin:
        return "beyond noise"
    return "borderline" if near >= -margin else "within noise"


def bootstrap_diff_ci(y, score_a, score_b, metric, n=10_000, seed=SEED, k=K, groups=None):
    """95% CI for metric(a) - metric(b) on the same resamples (see paired_differences)."""
    return paired_differences(y, score_a, score_b, n=n, seed=seed, k=k, groups=groups)[metric][1:]


def difference_table(y, scores, model, others, groups=None, n=10_000):
    """Paired differences `model` minus each of `others`, one row per pair and metric."""
    rows = []
    for other in others:
        for metric, (diff, lo, hi) in paired_differences(y, scores[model], scores[other], n=n, groups=groups).items():
            rows.append({"model": model, "minus": other, "metric": metric, "difference": diff,
                         "ci_lo": lo, "ci_hi": hi, "verdict": verdict(diff, lo, hi)})
    return pd.DataFrame(rows)


def results_table(y, scores, n_boot=500, groups=None):
    """One row per model with the three metrics and their bootstrap CIs. `scores` = {name: score}."""
    rows = []
    for name, score in scores.items():
        row = {"model": name, **evaluate(y, score)}
        for metric, (lo, hi) in bootstrap_all(y, score, n=n_boot, groups=groups).items():
            row[f"{metric}_lo"], row[f"{metric}_hi"] = lo, hi
        rows.append(row)
    return pd.DataFrame(rows)


def set_plot_style():
    """Readable, slide-ready defaults."""
    plt.rcParams.update({
        "font.size": 13, "axes.titlesize": 16, "axes.titleweight": "bold", "axes.labelsize": 13,
        "legend.fontsize": 12, "xtick.labelsize": 12, "ytick.labelsize": 12,
        "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 100, "savefig.dpi": 160,
    })


def plot_gains(y, scores, title, path=None, k=K):
    """Gains curves for several models. `scores` = {model name: score array}."""
    set_plot_style()
    fig, ax = plt.subplots(figsize=(9, 6))
    for name, score in scores.items():
        curve = gains_curve(y, score)
        ax.plot(curve["share_inspected"] * 100, curve["share_failures_caught"] * 100,
                label=MODEL_LABELS.get(name, name), color=MODEL_COLORS.get(name),
                linewidth=2.5 if name.startswith("M") else 1.8,
                linestyle="--" if name == "B0_random" else "-")
    ax.axvline(k * 100, color="black", linestyle=":", linewidth=1.2)
    ax.text(k * 100 + 1, 3, f"Capacity: {k:.0%} of inspections", fontsize=11)
    ax.set_xlabel("Share of inspections made, riskiest first (%)")
    ax.set_ylabel("Share of all failures found (%)")
    ax.set_title(title)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    if path:
        fig.savefig(path)
    return fig


def self_test():
    """Sanity checks: a perfect score and a random score must give the textbook values."""
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(SEED)
    for base_rate in [0.1, 0.59]:
        y = (rng.random(20_000) < base_rate).astype(int)
        perfect = evaluate(y, y.astype(float))
        expected = min(1.0, K / y.mean())
        assert abs(perfect["catch_at_k"] - expected) < 0.01, (perfect, expected)
        assert perfect["auc"] == 1.0
        noisy = y + rng.random(len(y)).round(1)          # a score with many ties
        assert abs(_auc(y, noisy) - roc_auc_score(y, noisy)) < 1e-12, "rank-based AUC must equal sklearn's"
        assert abs(perfect["precision_at_k"] - min(1.0, y.mean() / K)) < 0.01
        random = evaluate(y, rng.random(len(y)))
        assert abs(random["catch_at_k"] - K) < 0.02, random
        assert abs(random["precision_at_k"] - y.mean()) < 0.02, random
        assert abs(random["auc"] - 0.5) < 0.02
    # A constant score is the average random order: its hit rate must equal the fail rate exactly,
    # whatever the row order.
    constant = evaluate(np.sort(y)[::-1], np.zeros(len(y)))
    assert abs(constant["precision_at_k"] - y.mean()) < 1e-12, constant
    # Half the rows tied at the cutoff: the expected value sits between best and worst case.
    y_small, tied_scores = np.array([1, 1, 0, 0, 1, 0, 0, 0, 0, 0]), np.array([9, 5, 5, 5, 5, 1, 1, 1, 1, 1.0])
    assert abs(evaluate(y_small, tied_scores, k=0.3)["precision_at_k"] - (1 + 2 * 0.5) / 3) < 1e-12
    lo, hi = bootstrap_ci(y[:2000], rng.random(2000), "precision_at_k", n=100)
    assert lo < y[:2000].mean() < hi
    # Clustered resampling: with one row per group it is an ordinary bootstrap; a score minus
    # itself is exactly zero on every resample.
    groups = np.repeat(np.arange(500), 4)
    lo, hi = bootstrap_ci(y[:2000], rng.random(2000), "precision_at_k", n=100, groups=groups)
    assert lo < y[:2000].mean() < hi
    s_same = rng.random(2000)
    assert paired_differences(y[:2000], s_same, s_same, n=50, groups=groups)["precision_at_k"] == (0.0, 0.0, 0.0)
    # The gains curve agrees with evaluate() at the capacity, ties included.
    curve = gains_curve(y_small, tied_scores)
    assert abs(curve["share_failures_caught"][3] - evaluate(y_small, tied_scores, k=0.3)["catch_at_k"]) < 1e-12
    assert verdict(0.10, 0.08, 0.12) == "beyond noise" and verdict(0.015, 0.0, 0.029) == "borderline"
    assert verdict(0.015, 0.0009, 0.029) == "borderline" and verdict(0.001, -0.012, 0.010) == "within noise"
    print("evaluate.py self-test passed: perfect score gives catch_at_k = min(1, k / base rate); "
          "random and constant scores give about k")


if __name__ == "__main__":
    self_test()


# --- Slide-ready figures -------------------------------------------------------
FEATURE_LABELS = {
    "prior_n": "Number of past inspections",
    "prior_n_3y": "Inspections in the last 3 years",
    "is_first": "First inspection on record",
    "prior_fail_rate": "Share of past inspections failed",
    "prior_routine_fail_rate": "Share of past routine inspections failed",
    "last_fail": "Previous visit failed",
    "last_n_viol": "Violations at the previous visit",
    "last_n_lvl3": "Serious violations at the previous visit",
    "last_routine_fail": "Last routine inspection failed",
    "last_routine_n_viol": "Violations at the last routine inspection",
    "last_routine_n_lvl3": "Serious violations at the last routine inspection",
    "days_since_last": "Days since the previous visit",
    "license_age_days": "Years on record",
    "month": "Month of the year",
    "is_takeout": "Offers take-out",
    "neighborhood": "Neighborhood",
    "zip": "Zip code",
    "has_prior_comments": "Previous visit has comments",
}


def plot_hit_rate_bar(results, title, path=None, base_rate=None, k=K):
    """Hit rate at capacity k for each model, with 95% bootstrap intervals. `results` = results_table output."""
    set_plot_style()
    fig, ax = plt.subplots(figsize=(9, 5.5))
    names = list(results["model"])
    hit = results["precision_at_k"].to_numpy() * 100
    err = np.vstack([hit - results["precision_at_k_lo"].to_numpy() * 100,
                     results["precision_at_k_hi"].to_numpy() * 100 - hit])
    bars = ax.bar([MODEL_LABELS.get(n, n).replace(", ", ",\n").replace(" (", "\n(") for n in names], hit,
                  yerr=err, capsize=5, color=[MODEL_COLORS.get(n, "#4c72b0") for n in names])
    for bar, value, high in zip(bars, hit, results["precision_at_k_hi"].to_numpy() * 100):
        ax.text(bar.get_x() + bar.get_width() / 2, high + 1.5, f"{value:.0f}%", ha="center", fontsize=14, fontweight="bold")
    if base_rate is not None:
        ax.axhline(base_rate * 100, color="black", linestyle=":", linewidth=1.2)
        ax.text(len(names) - 0.5, base_rate * 100 - 4, f"Overall fail rate {base_rate:.0%}", ha="right", fontsize=11)
    ax.set_ylabel(f"Visits that find a failure (%)")
    ax.set_ylim(0, 105)
    ax.set_title(title)
    ax.tick_params(axis="x", labelsize=11)
    fig.tight_layout()
    if path:
        fig.savefig(path)
    return fig


def plot_importance(importance, title, path=None):
    """Horizontal bars of the top features. `importance` = models.importance_table output."""
    set_plot_style()
    imp = importance.iloc[::-1]
    labels = [FEATURE_LABELS.get(f, f.replace("text_pc", "Comment text, component ")) for f in imp["feature"]]
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(imp) + 1.8))
    ax.barh(labels, imp["share_of_gain"] * 100, color="#4c72b0")
    for y, value in enumerate(imp["share_of_gain"] * 100):
        ax.text(value + 0.5, y, f"{value:.0f}%" if value >= 1 else "<1%", va="center", fontsize=11)
    ax.set_xlabel("Share of the model's predictive power (%)")
    ax.set_title(title)
    fig.tight_layout()
    if path:
        fig.savefig(path)
    return fig
