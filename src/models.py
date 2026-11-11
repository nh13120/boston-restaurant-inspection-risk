"""Baselines (B0, B1, B2) and the LightGBM models (M1, M2).

Every model returns one risk score per inspection. Higher = more likely to fail.
"""
from itertools import product

import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.evaluate import evaluate

SEED = 42
TARGET = "label_fail"

# Everything here is known before the inspection (see docs/data_dictionary.md).
NUM_FEATURES = ["prior_n", "prior_n_3y", "is_first", "prior_fail_rate", "prior_routine_fail_rate",
                "last_fail", "last_n_viol", "last_n_lvl3",
                "last_routine_fail", "last_routine_n_viol", "last_routine_n_lvl3",
                "days_since_last", "license_age_days", "month", "is_takeout"]
CAT_FEATURES = ["neighborhood", "zip"]
HISTORY_FEATURES = NUM_FEATURES + CAT_FEATURES

# Columns that describe the inspection being predicted, or today's license status.
# Using any of them as a feature would be leakage, so fitting refuses them.
NEVER_FEATURES = {"label_fail", "result", "result_codes", "n_viol", "n_lvl1", "n_lvl2", "n_lvl3",
                  "n_rechecked_ok", "comments", "licstatus", "is_reinspection", "split", "year", "date",
                  "license_issue_date", "businessname", "descript", "inspection_id", "licenseno"}
assert not NEVER_FEATURES & set(HISTORY_FEATURES)

# Lecture 6 style: a small, explicit grid (3 x 2 x 2 = 12 combinations).
LGBM_GRID = {
    "num_leaves": [7, 15, 31],
    "learning_rate": [0.02, 0.05],
    "n_estimators": [100, 500],   # an upper limit: early stopping can use fewer trees
}
# n_jobs=1 and deterministic=True make reruns give identical numbers; the data is small enough.
LGBM_FIXED = dict(random_state=SEED, verbosity=-1, n_jobs=1, deterministic=True, force_row_wise=True)


def load_model_table(path="../data/processed/model_table.parquet"):
    """Load the table and mark the categorical columns, so every split shares the same categories."""
    df = pd.read_parquet(path)
    # Plain floats: LightGBM rejects the Arrow-backed integer columns that pandas 3 can produce.
    df[NUM_FEATURES] = df[NUM_FEATURES].astype("float64")
    for col in CAT_FEATURES:
        df[col] = df[col].astype("category")
    return df


# --- Baselines ---------------------------------------------------------------
def score_random(df, seed=SEED):
    """B0: the current routine schedule ignores risk. Every inspection gets the same score, and
    evaluate() averages over every possible random order of tied rows. The result is exactly
    what a random order achieves on average: a hit rate equal to the fail rate and an AUC of 0.5."""
    return np.full(len(df), 0.5)


def score_simple_rule(df):
    """Sensitivity check: another rule an inspector could apply by hand, "most violations at the
    last routine inspection first". Restaurants with no routine history go last."""
    return df["last_routine_n_viol"].fillna(-1).to_numpy()


def past_fail_rate_fallback(fit_rows):
    """Score for a restaurant with no history: the average past failure rate of the fitting rows.
    It must be on the same scale as prior_fail_rate (which counts re-inspections, so it averages
    about 0.37). The routine fail rate (about 0.60) would put every new restaurant at the top."""
    return float(fit_rows["prior_fail_rate"].mean())


def score_past_fail_rate(df, fallback):
    """B1: rank by the share of past inspections failed. A first inspection has no past,
    so it gets `fallback` (see past_fail_rate_fallback)."""
    return df["prior_fail_rate"].fillna(fallback).to_numpy()


def linear_matrix(df):
    """Numeric features plus one 0/1 column per neighborhood and zip, for logistic regression."""
    dummies = pd.get_dummies(df[CAT_FEATURES], dtype=float)   # categories are fixed, so columns are stable
    return pd.concat([df[NUM_FEATURES].astype(float), dummies], axis=1)


def fit_logreg(train):
    """B2: impute -> scale -> logistic regression."""
    pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),   # add_indicator flags "was missing"
        ("scale", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000, random_state=SEED)),
    ])
    return pipe.fit(linear_matrix(train), train[TARGET])


# --- LightGBM ----------------------------------------------------------------
def fit_lgbm(train, features, params, val=None):
    """Fit one LightGBM. With `val`, training stops early when validation stops improving."""
    assert not NEVER_FEATURES & set(features), "outcome-side column used as a feature"
    model = LGBMClassifier(**LGBM_FIXED, **params)
    if val is None:
        return model.fit(train[features], train[TARGET])
    return model.fit(
        train[features], train[TARGET],
        eval_X=val[features], eval_y=val[TARGET], eval_metric="auc",
        callbacks=[lgb.early_stopping(stopping_rounds=50, first_metric_only=True, verbose=False)],   # stop on AUC only
    )


def grid_search_lgbm(train, val, features, grid=LGBM_GRID):
    """Try every grid combination, select on validation precision_at_k.

    Returns (table of all combinations, best settings with n_estimators = trees actually used).
    """
    rows = []
    for num_leaves, learning_rate, n_estimators in product(*grid.values()):
        params = dict(num_leaves=num_leaves, learning_rate=learning_rate, n_estimators=n_estimators)
        model = fit_lgbm(train, features, params, val=val)
        best_iter = model.best_iteration_ or n_estimators
        score = model.predict_proba(val[features])[:, 1]   # uses the best iteration automatically
        rows.append({**params, "best_iteration": best_iter, **evaluate(val[TARGET], score)})
    table = pd.DataFrame(rows)
    # Ties on precision_at_k go to the higher AUC, then to the simpler model (pandas sort is stable).
    best = table.sort_values(["precision_at_k", "auc", "num_leaves"], ascending=[False, False, True]).iloc[0]
    best_params = dict(num_leaves=int(best["num_leaves"]), learning_rate=float(best["learning_rate"]),
                       n_estimators=int(best["best_iteration"]))
    return table, best_params


def importance_table(model, features, top=15):
    """Top features by total gain (how much each feature reduced the error across all splits)."""
    gain = model.booster_.feature_importance(importance_type="gain")
    out = pd.DataFrame({"feature": features, "gain": gain}).sort_values("gain", ascending=False)
    out["share_of_gain"] = out["gain"] / out["gain"].sum()
    return out.head(top).reset_index(drop=True)


def save_preds(df, score, path):
    pd.DataFrame({"inspection_id": df["inspection_id"].to_numpy(), "score": score}).to_csv(path, index=False)
