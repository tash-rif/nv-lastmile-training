"""
train.py
========
Model training functions for LightGBM, Random Forest, and Logistic Regression.
Each function accepts pre-transformed feature arrays (i.e. after the
ColumnTransformer has been applied).
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import json
import joblib
import shutil
import subprocess
import numpy as np
import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.metrics import f1_score, roc_auc_score, make_scorer

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _subsample_stratified(
    X: np.ndarray,
    y: np.ndarray,
    n: int,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return a stratified subsample of (X, y) of at most n rows."""
    if n >= len(X):
        return X, y
    rng = np.random.default_rng(random_state)
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    ratio = len(idx1) / len(y)
    n1 = min(len(idx1), max(1, int(n * ratio)))
    n0 = min(len(idx0), n - n1)
    sel = np.concatenate([
        rng.choice(idx0, n0, replace=False),
        rng.choice(idx1, n1, replace=False),
    ])
    rng.shuffle(sel)
    print(f"   Subsampled {len(sel):,} rows for search (from {len(X):,} total, "
          f"failure rate={y[sel].mean():.3f})")
    return X[sel], y[sel]


def _print_cv_table(cv_results: Dict, top_n: int = 5) -> None:
    """Print a formatted table of the top GridSearch results."""
    results_df = pd.DataFrame(cv_results)
    cols = [c for c in results_df.columns if c.startswith("param_") or c in
            ("mean_test_score", "std_test_score", "rank_test_score")]
    results_df = results_df[cols].sort_values("rank_test_score").head(top_n)
    print(results_df.to_string(index=False))


# ---------------------------------------------------------------------------
# LightGBM
# ---------------------------------------------------------------------------

def train_lgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    params: Dict,
    checkpoint_path: Optional[str] = None,
) -> "lgb.LGBMClassifier":
    """
    Fit a LightGBM classifier with the supplied hyperparameters.

    Parameters
    ----------
    X_train : array-like
        Pre-transformed feature matrix.
    y_train : array-like
        Binary target vector.
    params : dict
        Hyperparameters to pass to LGBMClassifier (e.g. from Optuna search).
        Must NOT include 'objective' or 'metric' — those are set internally.
    checkpoint_path : str or None
        If given, load from this path if it exists; save to it after training.

    Returns
    -------
    Fitted lgb.LGBMClassifier.
    """
    try:
        import lightgbm as lgb
    except ImportError:
        raise ImportError("lightgbm not installed. Run: pip install lightgbm")

    if checkpoint_path and os.path.exists(checkpoint_path):
        model = joblib.load(checkpoint_path)
        print(f"   Loaded LightGBM checkpoint from {checkpoint_path}")
        return model

    params = {**params}
    params.setdefault("objective", "binary")
    params.setdefault("metric", "auc")
    params.setdefault("verbosity", 1)
    params.setdefault("n_jobs", 4)

    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    params.setdefault("scale_pos_weight", n_neg / max(n_pos, 1))

    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train)
    print(f"   ✅ LightGBM trained — n_estimators={params.get('n_estimators', 100)}, "
          f"num_leaves={params.get('num_leaves', 31)}")

    if checkpoint_path:
        os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
        joblib.dump(model, checkpoint_path)
        print(f"   Saved LightGBM checkpoint to {checkpoint_path}")

    return model


def run_optuna_lgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_trials: int = 30,
    random_state: int = 42,
    search_sample_size: Optional[int] = None,
    n_splits: int = 3,
    n_jobs: int = 4,
    checkpoint_dir: Optional[str] = None,
) -> Dict:
    """
    Run an Optuna hyperparameter search for LightGBM using a stratified
    subsample and 3-fold CV with per-fold early stopping.

    Parameters
    ----------
    X_train : array-like
    y_train : array-like
    n_trials : int
    random_state : int
    search_sample_size : int or None
        If set, subsample this many rows (stratified) for the search.
    n_splits : int
        CV folds during search (default 3).
    n_jobs : int
        Threads for each LightGBM model during search (default 4).
    checkpoint_dir : str or None
        Directory to persist the Optuna SQLite study and best-params JSON.
        If set, a crashed search resumes from the last completed trial.

    Returns
    -------
    dict : Best hyperparameters (including best_n_estimators from early stopping).
    """
    try:
        import optuna
        import lightgbm as lgb
    except ImportError:
        raise ImportError("Install optuna and lightgbm: pip install optuna lightgbm")

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # --- checkpoint setup ---
    storage = None
    best_params_path = None
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
        best_params_path = os.path.join(checkpoint_dir, "lgbm_best_params.json")
        if os.path.exists(best_params_path):
            with open(best_params_path) as f:
                best = json.load(f)
            print(f"   Loaded cached Optuna results from {best_params_path}")
            return best
        storage = f"sqlite:///{os.path.join(checkpoint_dir, 'lgbm_optuna.db')}"

    X_search, y_search = _subsample_stratified(
        X_train, y_train, search_sample_size or len(X_train), random_state
    )

    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "binary",
            "metric": "auc",
            "verbosity": 1,
            "n_jobs": n_jobs,
            # High ceiling — early stopping decides the real count
            "n_estimators": 1000,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 20, 150),
            "max_depth": trial.suggest_int("max_depth", 4, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        }
        n_neg = int((y_search == 0).sum())
        n_pos = int((y_search == 1).sum())
        params["scale_pos_weight"] = trial.suggest_float(
            "scale_pos_weight", 1.0, max(n_neg / max(n_pos, 1), 2.0)
        )

        cv_splits = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        fold_f1s = []
        best_iters = []

        for tr_idx, val_idx in cv_splits.split(X_search, y_search):
            X_tr, X_val = X_search[tr_idx], X_search[val_idx]
            y_tr, y_val = y_search[tr_idx], y_search[val_idx]

            model = lgb.LGBMClassifier(**params)
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                callbacks=[
                    lgb.early_stopping(50, verbose=False),
                    lgb.log_evaluation(100),
                ],
            )
            preds = model.predict(X_val)
            fold_f1s.append(f1_score(y_val, preds, pos_label=1, zero_division=0))
            best_iters.append(model.best_iteration_ or params["n_estimators"])

        trial.set_user_attr("best_n_estimators", int(np.mean(best_iters)))
        return float(np.mean(fold_f1s))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
        storage=storage,
        study_name="lgbm_optuna",
        load_if_exists=True,
    )

    completed = sum(
        1 for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    )
    remaining = max(0, n_trials - completed)
    if completed:
        print(f"   Resuming Optuna: {completed} trials already done, {remaining} remaining...")
    study.optimize(objective, n_trials=remaining, show_progress_bar=True)

    best = study.best_params
    best["n_estimators"] = study.best_trial.user_attrs.get("best_n_estimators", 300)
    print(f"\n✅ Optuna search complete — best F1: {study.best_value:.4f}")
    print(f"   Best params: {best}")

    if best_params_path:
        with open(best_params_path, "w") as f:
            json.dump(best, f, indent=2)
        print(f"   Saved best params to {best_params_path}")

    return best


# ---------------------------------------------------------------------------
# Random Forest
# ---------------------------------------------------------------------------

def train_rf(
    X_train: np.ndarray,
    y_train: np.ndarray,
    params: Dict,
) -> RandomForestClassifier:
    """Fit a RandomForestClassifier with the supplied hyperparameters."""
    params = {**params}
    params.setdefault("n_estimators", 200)
    params.setdefault("class_weight", "balanced")
    params.setdefault("n_jobs", -1)
    params.setdefault("random_state", 42)

    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    print(f"   ✅ Random Forest trained — n_estimators={params['n_estimators']}, "
          f"max_depth={params.get('max_depth', 'None')}")
    return model


def grid_search_rf(
    X_train: np.ndarray,
    y_train: np.ndarray,
    param_grid: Dict,
    random_state: int = 42,
    search_sample_size: Optional[int] = None,
    n_splits: int = 3,
    model_n_jobs: int = 4,
    gs_n_jobs: int = 1,
) -> Tuple[RandomForestClassifier, Dict]:
    """
    Run GridSearchCV for Random Forest on a stratified subsample.

    Parameters
    ----------
    X_train, y_train : array-like
    param_grid : dict
    random_state : int
    search_sample_size : int or None
    n_splits : int  CV folds during search (default 3).
    model_n_jobs : int  Threads per RF model (default 4).
    gs_n_jobs : int  Parallel param combos in GridSearchCV (default 1 — sequential
                     to avoid multiplying memory with model_n_jobs).

    Returns
    -------
    (best_model_refit_on_full, best_params)
    """
    X_search, y_search = _subsample_stratified(
        X_train, y_train, search_sample_size or len(X_train), random_state
    )

    f1_scorer = make_scorer(f1_score, pos_label=1, zero_division=0)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    base = RandomForestClassifier(n_jobs=model_n_jobs, random_state=random_state)

    cleaned_grid = {}
    for k, vals in param_grid.items():
        cleaned_grid[k] = [None if v == "null" else v for v in vals]

    gs = GridSearchCV(
        base, cleaned_grid,
        scoring=f1_scorer, cv=cv,
        n_jobs=gs_n_jobs, refit=False, verbose=1,
    )
    gs.fit(X_search, y_search)

    print(f"\n   GridSearch RF — best F1: {gs.best_score_:.4f}")
    print(f"   Best params: {gs.best_params_}")
    _print_cv_table(gs.cv_results_, top_n=5)

    # Refit best params on the FULL training set
    print("   Refitting RF on full training set...")
    best_params = {**gs.best_params_}
    best_params.setdefault("n_jobs", model_n_jobs)
    best_params.setdefault("random_state", random_state)
    best_model = RandomForestClassifier(**best_params)
    best_model.fit(X_train, y_train)

    return best_model, gs.best_params_


# ---------------------------------------------------------------------------
# Logistic Regression
# ---------------------------------------------------------------------------

def train_lr(
    X_train: np.ndarray,
    y_train: np.ndarray,
    params: Dict,
) -> LogisticRegression:
    """Fit a LogisticRegression classifier."""
    params = {**params}
    params.setdefault("class_weight", "balanced")
    params.setdefault("solver", "lbfgs")
    params.setdefault("max_iter", 1000)
    params.setdefault("n_jobs", -1)
    params.setdefault("random_state", 42)

    model = LogisticRegression(**params)
    model.fit(X_train, y_train)
    print(f"   ✅ Logistic Regression trained — C={params.get('C', 1.0)}")
    return model


def grid_search_lr(
    X_train: np.ndarray,
    y_train: np.ndarray,
    param_grid: Dict,
    random_state: int = 42,
    search_sample_size: Optional[int] = None,
    n_splits: int = 3,
) -> Tuple[LogisticRegression, Dict]:
    """
    Run GridSearchCV for Logistic Regression on a stratified subsample.

    Parameters
    ----------
    X_train, y_train : array-like
    param_grid : dict
    random_state : int
    search_sample_size : int or None
    n_splits : int  CV folds during search (default 3).

    Returns
    -------
    (best_model_refit_on_full, best_params)
    """
    X_search, y_search = _subsample_stratified(
        X_train, y_train, search_sample_size or len(X_train), random_state
    )

    f1_scorer = make_scorer(f1_score, pos_label=1, zero_division=0)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    base = LogisticRegression(
        solver="lbfgs", max_iter=1000,
        n_jobs=-1, random_state=random_state,
    )

    gs = GridSearchCV(
        base, param_grid,
        scoring=f1_scorer, cv=cv,
        n_jobs=-1, refit=False, verbose=0,
    )
    gs.fit(X_search, y_search)

    print(f"\n   GridSearch LR — best F1: {gs.best_score_:.4f}")
    print(f"   Best params: {gs.best_params_}")
    _print_cv_table(gs.cv_results_, top_n=5)

    # Refit best params on the FULL training set
    print("   Refitting LR on full training set...")
    best_params = {**gs.best_params_, "solver": "lbfgs", "max_iter": 1000,
                   "n_jobs": -1, "random_state": random_state}
    best_model = LogisticRegression(**best_params)
    best_model.fit(X_train, y_train)

    return best_model, gs.best_params_


# ---------------------------------------------------------------------------
# Cross-validation helper (final reporting — keeps 5-fold for accuracy)
# ---------------------------------------------------------------------------

def cross_validate_model(
    model,
    X: np.ndarray,
    y: np.ndarray,
    model_name: str = "Model",
    random_state: int = 42,
    cv_n_jobs: int = 1,
) -> Dict:
    """
    Run 5-fold stratified CV and return summary metrics (AUC-ROC + F1).
    Used for final reporting only — not called during hyperparameter search.

    cv_n_jobs controls sklearn's parallelism across folds. Default 1 (sequential)
    avoids OOM when the model itself is already multi-threaded (e.g. LightGBM).
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    scoring = {
        "roc_auc": "roc_auc",
        "f1": make_scorer(f1_score, pos_label=1, zero_division=0),
    }
    results = cross_validate(
        model, X, y, cv=cv, scoring=scoring, n_jobs=cv_n_jobs, error_score=0.0
    )

    summary = {
        "model_name": model_name,
        "mean_auc": float(np.mean(results["test_roc_auc"])),
        "std_auc":  float(np.std(results["test_roc_auc"])),
        "mean_f1":  float(np.mean(results["test_f1"])),
        "std_f1":   float(np.std(results["test_f1"])),
    }

    print(
        f"   {model_name:25s}  "
        f"AUC={summary['mean_auc']:.4f}±{summary['std_auc']:.4f}  "
        f"F1={summary['mean_f1']:.4f}±{summary['std_f1']:.4f}"
    )
    return summary
