"""
train.py
========
Model training functions for LightGBM, Random Forest, and Logistic Regression.
Each function accepts pre-transformed feature arrays (i.e. after the
ColumnTransformer has been applied).
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

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
# LightGBM
# ---------------------------------------------------------------------------

def train_lgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    params: Dict,
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

    Returns
    -------
    Fitted lgb.LGBMClassifier.
    """
    try:
        import lightgbm as lgb
    except ImportError:
        raise ImportError("lightgbm not installed. Run: pip install lightgbm")

    # Always override these for classification
    params = {**params}
    params.setdefault("objective", "binary")
    params.setdefault("metric", "auc")
    params.setdefault("verbosity", -1)
    params.setdefault("n_jobs", -1)

    # If running on Google Colab with a T4 GPU, force LightGBM to use that GPU.
    # Detection: presence of google.colab and nvidia-smi reporting a T4 device.
    def _colab_has_t4() -> bool:
        try:
            import google.colab  # type: ignore
            in_colab = True
        except Exception:
            in_colab = False
        if not in_colab:
            return False
        try:
            if shutil.which("nvidia-smi") is None:
                return False
            out = subprocess.check_output([
                "nvidia-smi", "--query-gpu=name", "--format=csv,noheader"
            ], stderr=subprocess.DEVNULL)
            names = out.decode().strip().splitlines()
            for n in names:
                if "T4" in n or "Tesla T4" in n:
                    return True
        except Exception:
            return False
        return False

    if _colab_has_t4():
        params["device"] = "gpu"
        params["device_type"] = "gpu"
        params.setdefault("gpu_platform_id", 0)
        params.setdefault("gpu_device_id", 0)

    # Compute class weight
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    params.setdefault("scale_pos_weight", n_neg / max(n_pos, 1))

    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train)
    print(f"   ✅ LightGBM trained — n_estimators={params.get('n_estimators', 100)}, "
          f"num_leaves={params.get('num_leaves', 31)}")
    return model


def run_optuna_lgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_trials: int = 50,
    random_state: int = 42,
) -> Dict:
    """
    Run an Optuna hyperparameter search for LightGBM, optimising F1 on the
    failure class via 5-fold stratified cross-validation.

    Parameters
    ----------
    X_train : array-like
        Pre-transformed feature matrix.
    y_train : array-like
        Binary target vector.
    n_trials : int
        Number of Optuna trials.
    random_state : int
        Random seed.

    Returns
    -------
    dict : Best hyperparameters found.
    """
    try:
        import optuna
        import lightgbm as lgb
    except ImportError:
        raise ImportError("Install optuna and lightgbm: pip install optuna lightgbm")

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    f1_scorer = make_scorer(f1_score, pos_label=1, zero_division=0)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "binary",
            "metric": "auc",
            "verbosity": -1,
            "n_jobs": -1,
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 20, 150),
            "max_depth": trial.suggest_int("max_depth", 4, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        }
        n_neg = int((y_train == 0).sum())
        n_pos = int((y_train == 1).sum())
        params["scale_pos_weight"] = trial.suggest_float(
            "scale_pos_weight", 1.0, max(n_neg / max(n_pos, 1), 2.0)
        )

        # Force GPU only when on Colab with T4
        try:
            import google.colab  # type: ignore
            in_colab = True
        except Exception:
            in_colab = False
        if in_colab:
            try:
                if shutil.which("nvidia-smi") is not None:
                    out = subprocess.check_output([
                        "nvidia-smi", "--query-gpu=name", "--format=csv,noheader"
                    ], stderr=subprocess.DEVNULL)
                    if any("T4" in s or "Tesla T4" in s for s in out.decode().splitlines()):
                        params["device"] = "gpu"
                        params["device_type"] = "gpu"
                        params.setdefault("gpu_platform_id", 0)
                        params.setdefault("gpu_device_id", 0)
            except Exception:
                pass

        model = lgb.LGBMClassifier(**params)
        scores = cross_validate(
            model, X_train, y_train,
            cv=cv, scoring=f1_scorer,
            n_jobs=-1, error_score=0.0,
        )
        return float(np.mean(scores["test_score"]))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    print(f"\n✅ Optuna search complete — best F1: {study.best_value:.4f}")
    print(f"   Best params: {best}")
    return best


# ---------------------------------------------------------------------------
# Random Forest
# ---------------------------------------------------------------------------

def train_rf(
    X_train: np.ndarray,
    y_train: np.ndarray,
    params: Dict,
) -> RandomForestClassifier:
    """
    Fit a RandomForestClassifier with the supplied hyperparameters.

    Parameters
    ----------
    X_train : array-like
    y_train : array-like
    params : dict
        Passed directly to RandomForestClassifier.

    Returns
    -------
    Fitted RandomForestClassifier.
    """
    params = {**params}
    params.setdefault("n_estimators", 300)
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
) -> Tuple[RandomForestClassifier, Dict]:
    """
    Run GridSearchCV for Random Forest, optimising F1 on the failure class.

    Parameters
    ----------
    X_train, y_train : array-like
    param_grid : dict
        Grid passed to GridSearchCV (e.g. from config.yaml).
    random_state : int

    Returns
    -------
    (best_model, best_params)
    """
    f1_scorer = make_scorer(f1_score, pos_label=1, zero_division=0)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    base = RandomForestClassifier(n_jobs=-1, random_state=random_state)

    # Replace null strings from yaml with Python None
    cleaned_grid = {}
    for k, vals in param_grid.items():
        cleaned_grid[k] = [None if v == "null" else v for v in vals]

    gs = GridSearchCV(
        base, cleaned_grid,
        scoring=f1_scorer, cv=cv,
        n_jobs=-1, refit=True, verbose=0,
    )
    gs.fit(X_train, y_train)

    print(f"\n   GridSearch RF — best F1: {gs.best_score_:.4f}")
    print(f"   Best params: {gs.best_params_}")

    _print_cv_table(gs.cv_results_, top_n=5)
    return gs.best_estimator_, gs.best_params_


# ---------------------------------------------------------------------------
# Logistic Regression
# ---------------------------------------------------------------------------

def train_lr(
    X_train: np.ndarray,
    y_train: np.ndarray,
    params: Dict,
) -> LogisticRegression:
    """
    Fit a LogisticRegression classifier.

    Parameters
    ----------
    X_train, y_train : array-like
    params : dict

    Returns
    -------
    Fitted LogisticRegression.
    """
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
) -> Tuple[LogisticRegression, Dict]:
    """
    Run GridSearchCV for Logistic Regression.

    Returns
    -------
    (best_model, best_params)
    """
    f1_scorer = make_scorer(f1_score, pos_label=1, zero_division=0)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    base = LogisticRegression(
        solver="lbfgs", max_iter=1000,
        n_jobs=-1, random_state=random_state,
    )

    gs = GridSearchCV(
        base, param_grid,
        scoring=f1_scorer, cv=cv,
        n_jobs=-1, refit=True, verbose=0,
    )
    gs.fit(X_train, y_train)

    print(f"\n   GridSearch LR — best F1: {gs.best_score_:.4f}")
    print(f"   Best params: {gs.best_params_}")

    _print_cv_table(gs.cv_results_, top_n=5)
    return gs.best_estimator_, gs.best_params_


# ---------------------------------------------------------------------------
# Cross-validation helper
# ---------------------------------------------------------------------------

def cross_validate_model(
    model,
    X: np.ndarray,
    y: np.ndarray,
    model_name: str = "Model",
    random_state: int = 42,
) -> Dict:
    """
    Run 5-fold stratified cross-validation and return summary metrics.

    Metrics reported:
    - mean AUC-ROC
    - mean F1 (failure class)
    - std of both

    Parameters
    ----------
    model : sklearn estimator (unfitted)
    X, y : array-like
    model_name : str
    random_state : int

    Returns
    -------
    dict with keys: model_name, mean_auc, std_auc, mean_f1, std_f1
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    scoring = {
        "roc_auc": "roc_auc",
        "f1": make_scorer(f1_score, pos_label=1, zero_division=0),
    }
    results = cross_validate(
        model, X, y, cv=cv, scoring=scoring, n_jobs=-1, error_score=0.0
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


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _print_cv_table(cv_results: Dict, top_n: int = 5) -> None:
    """Print a formatted table of the top GridSearch results."""
    results_df = pd.DataFrame(cv_results)
    cols = [c for c in results_df.columns if c.startswith("param_") or c in
            ("mean_test_score", "std_test_score", "rank_test_score")]
    results_df = results_df[cols].sort_values("rank_test_score").head(top_n)
    print(results_df.to_string(index=False))
