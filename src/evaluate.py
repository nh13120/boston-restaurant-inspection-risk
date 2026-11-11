"""Metrics for ranking inspections by risk, with capacity fixed at the top k share.

precision_at_k (the "hit rate") = share of the visited inspections that find a failure.
catch_at_k                      = share of all failures that the visited inspections find.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

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


def _ranked(y, score, seed=SEED + 1):
    """Labels sorted from highest to lowest score, for curves. Ties are broken at random with
    their own fixed seed, so a model that gives many rows the same score gets no credit from
    row order."""
    y, score = np.asarray(y), np.asarray(score, dtype=float)
    tie_breaker = np.random.default_rng(seed).random(len(y))
    order = np.lexsort((tie_breaker, -score))   # sort by -score, then by the tie breaker
    return y[order]


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
        "auc": roc_auc_score(y, score),
        "catch_at_k": hits / y.sum(),
        "precision_at_k": hits / n_top,
        "n": n,
    }


def gains_curve(y, score):
    """Share of failures caught as a function of the share of inspections made."""
    y_sorted = _ranked(y, score)
    n = len(y_sorted)
    return pd.DataFrame({
        "share_inspected": np.arange(0, n + 1) / n,
        "share_failures_caught": np.concatenate([[0], np.cumsum(y_sorted)]) / y_sorted.sum(),
    })


def bootstrap_ci(y, score, metric, n=500, seed=SEED, k=K):
    """95% confidence interval for one metric: resample inspections with replacement n times."""
    y, score = np.asarray(y), np.asarray(score, dtype=float)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        values.append(evaluate(y[idx], score[idx], k)[metric])
    return tuple(np.percentile(values, [2.5, 97.5]))


def bootstrap_diff_ci(y, score_a, score_b, metric, n=500, seed=SEED, k=K):
    """95% CI for metric(a) - metric(b), using the SAME resampled rows for both models.
    If the interval excludes 0, the difference is larger than the sampling noise."""
    y, score_a, score_b = np.asarray(y), np.asarray(score_a, dtype=float), np.asarray(score_b, dtype=float)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        diffs.append(evaluate(y[idx], score_a[idx], k)[metric] - evaluate(y[idx], score_b[idx], k)[metric])
    return tuple(np.percentile(diffs, [2.5, 97.5]))


def results_table(y, scores, n_boot=500):
    """One row per model with the three metrics and their bootstrap CIs. `scores` = {name: score}."""
    rows = []
    for name, score in scores.items():
        row = {"model": name, **evaluate(y, score)}
        for metric in ["auc", "catch_at_k", "precision_at_k"]:
            row[f"{metric}_lo"], row[f"{metric}_hi"] = bootstrap_ci(y, score, metric, n=n_boot)
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
    rng = np.random.default_rng(SEED)
    for base_rate in [0.1, 0.59]:
        y = (rng.random(20_000) < base_rate).astype(int)
        perfect = evaluate(y, y.astype(float))
        expected = min(1.0, K / y.mean())
        assert abs(perfect["catch_at_k"] - expected) < 0.01, (perfect, expected)
        assert perfect["auc"] == 1.0
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
    assert lo < y.mean() < hi
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
