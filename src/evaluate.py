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


def _ranked(y, score, seed=SEED):
    """Labels sorted from highest to lowest score. Ties are broken at random (fixed seed),
    so a model that gives many rows the same score gets no credit from row order."""
    y, score = np.asarray(y), np.asarray(score, dtype=float)
    tie_breaker = np.random.default_rng(seed).random(len(y))
    order = np.lexsort((tie_breaker, -score))   # sort by -score, then by the tie breaker
    return y[order]


def evaluate(y, score, k=K):
    """AUC plus the two capacity metrics at the top k share."""
    y_sorted = _ranked(y, score)
    n = len(y_sorted)
    n_top = max(1, int(round(k * n)))
    hits = y_sorted[:n_top].sum()
    return {
        "auc": roc_auc_score(y, score),
        "catch_at_k": hits / y_sorted.sum(),
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
    # A constant score must behave like a random one (ties broken at random, not by row order).
    y_sorted_by_label = np.sort(y)[::-1]
    constant = evaluate(y_sorted_by_label, np.zeros(len(y)))
    assert abs(constant["precision_at_k"] - y.mean()) < 0.02, constant
    lo, hi = bootstrap_ci(y[:2000], rng.random(2000), "precision_at_k", n=100)
    assert lo < y.mean() < hi
    print("evaluate.py self-test passed: perfect score gives catch_at_k = min(1, k / base rate); "
          "random and constant scores give about k")


if __name__ == "__main__":
    self_test()
