"""Wider model comparison: ridge and LASSO logistic regression, random forest, XGBoost, and
LightGBM without location.

Same label, inputs, splits and metrics as src/models.py. The four added algorithms read the matrix
that logistic regression (B2) already uses: the 15 numeric inputs plus one 0/1 column per zip code
and neighborhood. Settings are chosen on the validation year only (see notebook 06).
"""
import textwrap
from itertools import product

import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.evaluate import evaluate
from src.models import HISTORY_FEATURES, SEED, TARGET, fit_lgbm, linear_matrix

# Small, explicit grids. Each list runs from the
# simplest setting to the most flexible, because ties on validation go to the simpler setting.
# C is the inverse of the penalty strength: a small C means a strong penalty.
C_VALUES = [0.001, 0.01, 0.1, 1, 10, 100]
GRIDS = {
    "C1_ridge_logreg": {"C": C_VALUES},
    "C2_lasso_logreg": {"C": C_VALUES},
    "C3_random_forest": {"n_estimators": [100, 500], "max_depth": [6, 10, None], "max_features": ["sqrt", 0.5]},
    # n_estimators is an upper limit here: early stopping can use fewer trees, as for LightGBM.
    "C4_xgboost": {"max_depth": [2, 4, 6], "learning_rate": [0.02, 0.05], "n_estimators": [100, 500]},
}
FIXED = {
    "C1_ridge_logreg": "L2 penalty, lbfgs solver, same preparation as B2 (B2 is this model at C=1)",
    "C2_lasso_logreg": "L1 penalty, saga solver, same preparation as B2",
    "C3_random_forest": "min_samples_leaf=20, one-hot columns, missing values handled by the trees",
    "C4_xgboost": "hist trees, early stopping on validation AUC (50 rounds), one-hot columns",
}
# M1 with its frozen settings and without zip code and neighborhood: the version the project recommends.
NO_LOCATION = "M1_lgbm_no_location"
NO_LOCATION_FEATURES = [f for f in HISTORY_FEATURES if f not in ("zip", "neighborhood")]

LABELS = {
    "C1_ridge_logreg": "Ridge logistic regression", "C2_lasso_logreg": "LASSO logistic regression",
    "C3_random_forest": "Random forest", "C4_xgboost": "XGBoost",
    NO_LOCATION: "LightGBM without location",
    "S1_last_routine_violations": "Rule: most violations at the last routine inspection",
    "M3_lgbm_finetuned": "LightGBM, history + fine-tuned text",
}

# From scikit-learn 1.8 on, l1_ratio alone selects the penalty. Older versions would ignore it
# and fit ridge under the LASSO name, with only a warning.
assert tuple(int(part) for part in sklearn.__version__.split(".")[:2]) >= (1, 8), "src/compare.py needs scikit-learn 1.8 or later"


def _logreg(C, l1):
    """B2's pipeline with a tunable penalty. l1=False is ridge (L2), l1=True is LASSO (L1)."""
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(C=C, l1_ratio=1.0 if l1 else 0.0, solver="saga" if l1 else "lbfgs",
                                     max_iter=5000 if l1 else 2000, random_state=SEED)),
    ])


def fit(name, train, params, val=None):
    """Fit one model. `val` is used only by XGBoost, to stop adding trees when validation AUC stalls."""
    X, y = linear_matrix(train), train[TARGET]
    if name == "C1_ridge_logreg":
        return _logreg(params["C"], l1=False).fit(X, y)
    if name == "C2_lasso_logreg":
        return _logreg(params["C"], l1=True).fit(X, y)
    if name == "C3_random_forest":
        # n_jobs=1 keeps reruns identical, like the LightGBM settings in src/models.py.
        return RandomForestClassifier(**params, min_samples_leaf=20, random_state=SEED, n_jobs=1).fit(X, y)
    if name == "C4_xgboost":
        model = XGBClassifier(**params, tree_method="hist", eval_metric="auc", random_state=SEED, n_jobs=1,
                              early_stopping_rounds=50 if val is not None else None)
        if val is None:
            return model.fit(X, y)
        return model.fit(X, y, eval_set=[(linear_matrix(val), val[TARGET])], verbose=False)
    raise ValueError(name)


def score(model, df):
    """Risk score per inspection: the predicted probability of failing."""
    return model.predict_proba(linear_matrix(df))[:, 1]


def grid_search(name, train, val):
    """Try every setting in GRIDS[name], select on validation precision_at_k.

    Returns (table of all settings, best settings). Ties go to the higher AUC, then to the setting
    listed first (the simpler one). For XGBoost, n_estimators becomes the number of trees early
    stopping actually used, so the test refit needs no validation set.
    """
    settings = [dict(zip(GRIDS[name], values)) for values in product(*GRIDS[name].values())]
    rows = []
    for params in settings:
        model = fit(name, train, params, val=val)
        row = {"model": name, **params, **evaluate(val[TARGET], score(model, val))}
        if name == "C4_xgboost":
            row["best_iteration"] = model.best_iteration + 1
        rows.append(row)
    table = pd.DataFrame(rows)       # an empty max_depth cell means "no limit"
    best = (table.assign(listed=range(len(table)))
            .sort_values(["precision_at_k", "auc", "listed"], ascending=[False, False, True]).index[0])
    best_params = dict(settings[best])
    if name == "C4_xgboost":
        best_params["n_estimators"] = int(table.loc[best, "best_iteration"])
    return table, best_params


def fit_no_location(train, lgbm_params):
    """LightGBM on the history features without zip code and neighborhood, with M1's settings."""
    return fit_lgbm(train, NO_LOCATION_FEATURES, lgbm_params)


def score_no_location(model, df):
    return model.predict_proba(df[NO_LOCATION_FEATURES])[:, 1]


def plot_comparison(table, path=None, final="M1_lgbm", xmin=78.0):
    """Hit rate of every method with its 95% range, as dots on a narrow axis.

    The learned models sit within about two points of each other, which a bar chart from 0 to 100
    cannot show. Methods below `xmin` are left off the axis and named in the footnote.
    `table` = results_table output with a `label` column.
    """
    import matplotlib.pyplot as plt
    from src.evaluate import K, set_plot_style

    set_plot_style()
    table = table.assign(hit=table["precision_at_k"] * 100)
    shown = table[table["hit"] >= xmin].sort_values("hit").reset_index(drop=True)
    left_out = table[table["hit"] < xmin]
    is_final = (shown["model"] == final).to_numpy()
    colors = ["#c44e52" if f else "#4c72b0" for f in is_final]

    fig, ax = plt.subplots(figsize=(10, 0.5 * len(shown) + 2.4))
    rows = range(len(shown))
    ax.hlines(rows, shown["precision_at_k_lo"] * 100, shown["precision_at_k_hi"] * 100, color=colors, lw=2.5)
    ax.scatter(shown["hit"], rows, s=90, color=colors, zorder=3)
    for row, value, high in zip(rows, shown["hit"], shown["precision_at_k_hi"] * 100):
        ax.text(high + 0.25, row, f"{value:.1f}%", va="center", fontsize=12, fontweight="bold" if is_final[row] else "normal")
    ax.axvline(shown.loc[is_final, "hit"].iloc[0], color="#c44e52", lw=1, ls=":")
    ax.set_yticks(list(rows))
    ax.set_yticklabels([label + (" (final model)" if f else "") for label, f in zip(shown["label"], is_final)], fontsize=12)
    ax.set_xlim(xmin, max(shown["precision_at_k_hi"] * 100) + 2.2)
    ax.set_xlabel("Visits that find a failure (%)")
    ax.set_title(f"The learned models are within about two points of each other\n(first {K:.0%} of visits, test years)")
    off = "; ".join(f"{label} {hit:.1f}%" for label, hit in zip(left_out["label"], left_out["hit"]))
    footnote = (f"Dots: hit rate. Lines: 95% ranges from resampling restaurants. The axis starts at {xmin:.0f}%."
                + (f" Off the axis: {off}." if off else "") + " † Added after the first test scoring.")
    ax.annotate(textwrap.fill(footnote, 110), xy=(0, 0), xycoords="axes fraction", xytext=(0, -52), textcoords="offset points",
                fontsize=10.5, va="top")
    fig.tight_layout()
    if path:
        fig.savefig(path, bbox_inches="tight")
    return fig
