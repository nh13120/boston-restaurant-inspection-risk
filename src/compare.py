"""Wider model comparison: ridge and LASSO logistic regression, random forest, XGBoost.

Same label, inputs, splits and metrics as src/models.py. Every model reads the matrix that
logistic regression (B2) already uses: the 15 numeric inputs plus one 0/1 column per zip code and
neighborhood. Settings are chosen on the validation year only (see notebook 06).
"""
from itertools import product

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.evaluate import evaluate
from src.models import SEED, TARGET, linear_matrix

# Small, explicit grids, modeled on the course's (Lectures 4 and 6, HW4). Each list runs from the
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
LABELS = {
    "C1_ridge_logreg": "Ridge logistic regression", "C2_lasso_logreg": "LASSO logistic regression",
    "C3_random_forest": "Random forest", "C4_xgboost": "XGBoost",
}


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
    best = table.sort_values(["precision_at_k", "auc"], ascending=False, kind="stable").index[0]
    best_params = dict(settings[best])
    if name == "C4_xgboost":
        best_params["n_estimators"] = int(table.loc[best, "best_iteration"])
    return table, best_params
