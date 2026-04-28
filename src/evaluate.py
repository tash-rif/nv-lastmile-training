"""
evaluate.py
===========
Model evaluation utilities for the last-mile delivery failure prediction pipeline.

Functions
---------
evaluate_model        : compute AUC-ROC, precision, recall, F1, Brier score
plot_roc_curves       : plot all model ROC curves on a single figure
plot_feature_importance : bar chart of top-N LightGBM feature importances
write_evaluation_report : write evaluation_report.json
select_champion       : pick best model by F1, break ties with AUC-ROC
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

PALETTE = {
    "lgbm": "#E63946",   # red
    "rf":   "#457B9D",   # steel blue
    "lr":   "#2A9D8F",   # teal
}

plt.rcParams.update({
    "figure.dpi": 120,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "DejaVu Sans",
})


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
    threshold: float = 0.5,
    feature_names: Optional[List[str]] = None,
) -> Dict:
    """
    Evaluate a fitted binary classifier on the held-out test set.

    Metrics computed
    ----------------
    - AUC-ROC
    - Precision (failure class = 1)
    - Recall    (failure class = 1)
    - F1        (failure class = 1)
    - Brier score
    - Confusion matrix (absolute counts)

    Parameters
    ----------
    model        : fitted sklearn-compatible classifier with predict_proba
    X_test       : pre-transformed feature matrix (numpy array)
    y_test       : true binary labels
    model_name   : display name, e.g. "lgbm"
    threshold    : decision threshold (default 0.5)
    feature_names : column names for display (optional)

    Returns
    -------
    dict with keys: model_name, auc_roc, precision, recall, f1, brier,
                    confusion_matrix, y_prob, y_pred, threshold
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    auc   = float(roc_auc_score(y_test, y_prob))
    prec  = float(precision_score(y_test, y_pred, pos_label=1, zero_division=0))
    rec   = float(recall_score(y_test, y_pred, pos_label=1, zero_division=0))
    f1    = float(f1_score(y_test, y_pred, pos_label=1, zero_division=0))
    brier = float(brier_score_loss(y_test, y_prob))
    cm    = confusion_matrix(y_test, y_pred).tolist()

    print(f"\n{'─'*55}")
    print(f"  {model_name.upper():^51}")
    print(f"{'─'*55}")
    print(f"  AUC-ROC : {auc:.4f}")
    print(f"  Precision (failure): {prec:.4f}")
    print(f"  Recall    (failure): {rec:.4f}")
    print(f"  F1        (failure): {f1:.4f}")
    print(f"  Brier score        : {brier:.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred,
                                 target_names=["success", "failure"],
                                 zero_division=0))

    return {
        "model_name": model_name,
        "auc_roc":    auc,
        "precision":  prec,
        "recall":     rec,
        "f1":         f1,
        "brier":      brier,
        "confusion_matrix": cm,
        "y_prob":     y_prob,
        "y_pred":     y_pred,
        "threshold":  threshold,
    }


# ---------------------------------------------------------------------------
# ROC curve plot
# ---------------------------------------------------------------------------

def plot_roc_curves(
    models: Dict,           # {"lgbm": fitted_model, "rf": ..., "lr": ...}
    results: Dict,          # output of evaluate_model keyed by model name
    y_test: np.ndarray,
    save_path: str,
) -> None:
    """
    Plot ROC curves for all models on a single figure and save as PNG.

    Parameters
    ----------
    models   : dict of fitted models (unused directly — y_prob from results)
    results  : dict keyed by model name, each value = evaluate_model output
    y_test   : true binary labels
    save_path: full file path for the saved PNG
    """
    fig, ax = plt.subplots(figsize=(7, 6))

    for name, res in results.items():
        fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
        color = PALETTE.get(name, "#555555")
        ax.plot(fpr, tpr,
                color=color, lw=2,
                label=f"{name.upper()}  (AUC = {res['auc_roc']:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random baseline")
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("ROC Curves — Delivery Failure Prediction", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    ax.grid(alpha=0.3)

    _save_figure(fig, save_path)
    print(f"✅ ROC curves saved → {save_path}")


# ---------------------------------------------------------------------------
# Confusion matrix plot
# ---------------------------------------------------------------------------

def plot_confusion_matrices(
    results: Dict,
    save_path: str,
) -> None:
    """
    Plot side-by-side normalised confusion matrices for all models.

    Parameters
    ----------
    results   : dict keyed by model name, each value = evaluate_model output
    save_path : full file path for the saved PNG
    """
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (name, res) in zip(axes, results.items()):
        cm = np.array(res["confusion_matrix"])
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["success", "failure"],
        )
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(f"{name.upper()}", fontsize=12, fontweight="bold")

    fig.suptitle("Confusion Matrices (Test Set)", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    _save_figure(fig, save_path)
    print(f"✅ Confusion matrices saved → {save_path}")


# ---------------------------------------------------------------------------
# Feature importance (LightGBM)
# ---------------------------------------------------------------------------

def plot_feature_importance(
    lgbm_model,
    feature_names: List[str],
    save_path: str,
    top_n: int = 20,
) -> None:
    """
    Plot a horizontal bar chart of the top-N LightGBM feature importances
    (gain-based).

    Parameters
    ----------
    lgbm_model   : fitted lgb.LGBMClassifier
    feature_names: list of feature column names
    save_path    : full file path for the saved PNG
    top_n        : number of features to display
    """
    importance = lgbm_model.feature_importances_
    fi_df = (
        pd.DataFrame({"feature": feature_names, "importance": importance})
        .sort_values("importance", ascending=False)
        .head(top_n)
        .sort_values("importance", ascending=True)   # flip for horizontal bar
    )

    fig, ax = plt.subplots(figsize=(9, max(4, top_n * 0.38)))
    bars = ax.barh(fi_df["feature"], fi_df["importance"],
                   color="#E63946", alpha=0.85, edgecolor="white")

    # Value labels
    for bar, val in zip(bars, fi_df["importance"]):
        ax.text(val + fi_df["importance"].max() * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:,.0f}", va="center", fontsize=8)

    ax.set_xlabel("Importance (Gain)", fontsize=11)
    ax.set_title(f"LightGBM — Top {top_n} Feature Importances", fontsize=13, fontweight="bold")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    plt.tight_layout()
    _save_figure(fig, save_path)
    print(f"✅ Feature importance plot saved → {save_path}")


# ---------------------------------------------------------------------------
# Calibration curve
# ---------------------------------------------------------------------------

def plot_calibration_curves(
    results: Dict,
    y_test: np.ndarray,
    save_path: str,
) -> None:
    """
    Plot calibration curves (reliability diagrams) for all models.
    Well-calibrated models have curves close to the diagonal.

    Parameters
    ----------
    results   : dict keyed by model name, each value = evaluate_model output
    y_test    : true binary labels
    save_path : full file path for the saved PNG
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated", lw=1.2)

    for name, res in results.items():
        fraction_pos, mean_pred = calibration_curve(
            y_test, res["y_prob"], n_bins=10, strategy="uniform"
        )
        color = PALETTE.get(name, "#555555")
        ax.plot(mean_pred, fraction_pos, marker="o", color=color,
                lw=2, ms=5, label=name.upper())

    ax.set_xlabel("Mean Predicted Probability", fontsize=11)
    ax.set_ylabel("Fraction of Positives", fontsize=11)
    ax.set_title("Calibration Curves", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    _save_figure(fig, save_path)
    print(f"✅ Calibration curves saved → {save_path}")


# ---------------------------------------------------------------------------
# Champion model selection
# ---------------------------------------------------------------------------

def select_champion(results: Dict) -> str:
    """
    Select the champion model: highest F1 on failure class; AUC-ROC as tiebreaker.

    Parameters
    ----------
    results : dict keyed by model name (evaluate_model outputs)

    Returns
    -------
    str : name of the champion model (e.g. "lgbm")
    """
    ranked = sorted(
        results.items(),
        key=lambda kv: (round(kv[1]["f1"], 4), round(kv[1]["auc_roc"], 4)),
        reverse=True,
    )
    champion = ranked[0][0]

    print("\n📊 Model Ranking (by F1, then AUC-ROC):")
    print(f"  {'Model':10s}  {'F1':>7s}  {'AUC-ROC':>8s}  {'Precision':>10s}  {'Recall':>7s}")
    print(f"  {'─'*10}  {'─'*7}  {'─'*8}  {'─'*10}  {'─'*7}")
    for name, res in ranked:
        marker = " ← CHAMPION" if name == champion else ""
        print(f"  {name:10s}  {res['f1']:7.4f}  {res['auc_roc']:8.4f}  "
              f"{res['precision']:10.4f}  {res['recall']:7.4f}{marker}")

    return champion


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

def write_evaluation_report(
    results: Dict,
    champion: str,
    feature_names: List[str],
    test_set_size: int,
    failure_rate_test: float,
    save_path: str,
    threshold: float = 0.5,
) -> Dict:
    """
    Write evaluation_report.json with the standard schema.

    Parameters
    ----------
    results          : dict keyed by model name (evaluate_model outputs)
    champion         : name of the champion model
    feature_names    : ordered list of feature column names
    test_set_size    : number of rows in the test set
    failure_rate_test: proportion of failures in test set
    save_path        : full file path for the JSON output
    threshold        : decision threshold used

    Returns
    -------
    dict : the report object (also written to disk)
    """
    report = {
        "champion_model": champion,
        "models": {
            name: {
                "auc_roc":   round(res["auc_roc"], 6),
                "precision": round(res["precision"], 6),
                "recall":    round(res["recall"], 6),
                "f1":        round(res["f1"], 6),
                "brier":     round(res["brier"], 6),
            }
            for name, res in results.items()
        },
        "test_set_size":    test_set_size,
        "failure_rate_test": round(failure_rate_test, 6),
        "feature_names":    feature_names,
        "threshold_used":   threshold,
    }

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"✅ Evaluation report saved → {save_path}")
    return report


# ---------------------------------------------------------------------------
# Champion copy helper
# ---------------------------------------------------------------------------

def copy_champion_model(
    champion_name: str,
    outputs_dir: str,
) -> str:
    """
    Copy the champion model joblib to champion_model.joblib.

    Parameters
    ----------
    champion_name : e.g. "lgbm"
    outputs_dir   : path to the outputs directory

    Returns
    -------
    str : path to the champion_model.joblib file
    """
    src = Path(outputs_dir) / f"model_{champion_name}.joblib"
    dst = Path(outputs_dir) / "champion_model.joblib"

    if not src.exists():
        raise FileNotFoundError(f"Expected model file not found: {src}")

    shutil.copy2(src, dst)
    print(f"✅ Champion model copied: {src.name} → champion_model.joblib")
    return str(dst)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _save_figure(fig: plt.Figure, path: str) -> None:
    """Save figure, creating parent dirs if needed."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
