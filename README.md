# nv-lastmile-training
### Ninja Van Capstone 2026 — Last-Mile Delivery Failure Prediction
#### Model Training Pipeline (Google Colab)

This repository contains a 5-notebook Colab pipeline that trains three machine learning models to predict last-mile delivery failure, then exports the fitted artifacts (`.joblib` files) for use in the [`nv-lastmile-demo`](https://github.com/YOUR_ORG/nv-lastmile-demo) web app.

---

## Quick Start — Open in Colab

Run the notebooks **in order** by clicking the badges below:

| # | Notebook | Open | Purpose |
|---|---|---|---|
| 01 | `01_data_download.ipynb` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_ORG/nv-lastmile-training/blob/main/notebooks/01_data_download.ipynb) | Download LaDe + Amazon datasets |
| 02 | `02_eda.ipynb` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_ORG/nv-lastmile-training/blob/main/notebooks/02_eda.ipynb) | Exploratory data analysis |
| 03 | `03_feature_engineering.ipynb` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_ORG/nv-lastmile-training/blob/main/notebooks/03_feature_engineering.ipynb) | Feature engineering + preprocessor |
| 04 | `04_model_training.ipynb` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_ORG/nv-lastmile-training/blob/main/notebooks/04_model_training.ipynb) | Train LightGBM, RF, LR |
| 05 | `05_model_evaluation.ipynb` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_ORG/nv-lastmile-training/blob/main/notebooks/05_model_evaluation.ipynb) | Evaluate + export champion model |

> ⚠️ **Update the badge URLs** — replace `YOUR_ORG` with your actual GitHub organisation/username after pushing this repo.

---

## Setup: Colab Secrets (Required Before Running)

The pipeline uses two credentials stored as **Colab Secrets** (not hardcoded). Set these up before running Notebook 01:

1. Open any notebook in Colab
2. Click the **🔑 key icon** in the left sidebar to open Secrets
3. Add the following secrets:

| Secret Name | Value | Where to get it |
|---|---|---|
| `HF_TOKEN` | Your HuggingFace access token | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) — create a **Read** token |
| `KAGGLE_KEY` | Full JSON contents of your `kaggle.json` | [kaggle.com/account](https://www.kaggle.com/account) → API → Create New Token → copy the entire file contents |

**KAGGLE_KEY format** — paste the full JSON string exactly as it appears in `kaggle.json`:
```json
{"username": "your_username", "key": "abc123def456..."}
```

> Both secrets are accessed via `google.colab.userdata.get(...)` — they are never written to the notebook output or committed to git.

---

## Execution Order & Expected Outputs

### Notebook 01 — Data Download (~10 min)
- **What it does:** Installs all dependencies, authenticates, downloads LaDe from HuggingFace and Amazon Delivery from Kaggle.
- **Outputs written to disk:**
  - `/content/data/raw/lade/lade_train.parquet`, `lade_test.parquet` (LaDe splits)
  - `/content/data/raw/amazon/amazon_delivery.parquet`
- **Expected console output:** Row counts and schemas for both datasets.

### Notebook 02 — EDA (~5 min)
- **What it does:** Generates 8 exploratory plots covering class imbalance, temporal failure patterns, courier workload, missing values, and feature correlations.
- **Outputs written to disk:**
  - `/content/outputs/eda/*.png` — 8 plot files
- **Key findings to note:** Failure rate (~15–25%), peak failure hours, city-level variance.

### Notebook 03 — Feature Engineering (~10 min)
- **What it does:** Applies 6 feature engineering modules, fits the sklearn `ColumnTransformer`, creates 80/20 stratified train/test splits, optionally generates SMOTE-balanced set.
- **Outputs written to disk:**
  - `/content/data/processed/features_train.parquet`
  - `/content/data/processed/features_test.parquet`
  - `/content/outputs/preprocessor.joblib` ← **needed by demo app**
  - `/content/outputs/feature_names.json` ← **needed by demo app**
  - `/content/outputs/class_weights.json`
- **Feature count:** 20 features across 6 groups (temporal, courier, address, geo, parcel, recipient).

### Notebook 04 — Model Training (~30–60 min on CPU, ~10–15 min on GPU)
- **What it does:** Trains LightGBM (Optuna, 50 trials), Random Forest (GridSearchCV), and Logistic Regression (GridSearchCV) — all via 5-fold stratified CV optimising F1 on the failure class.
- **Outputs written to disk:**
  - `/content/outputs/model_lgbm.joblib`
  - `/content/outputs/model_rf.joblib`
  - `/content/outputs/model_lr.joblib`
  - `/content/outputs/cv_results.json`
  - `/content/outputs/cv_results_comparison.png`

### Notebook 05 — Model Evaluation (~5 min)
- **What it does:** Evaluates all three models on the held-out test set, selects the champion (best F1), writes the evaluation report, and lets you download all artifacts.
- **Outputs written to disk:**
  - `/content/outputs/champion_model.joblib` ← **primary artifact for demo app**
  - `/content/outputs/evaluation_report.json`
  - `/content/outputs/roc_curves.png`
  - `/content/outputs/confusion_matrices.png`
  - `/content/outputs/feature_importance.png`
  - `/content/outputs/calibration_curves.png`
- **Target metrics:** AUC-ROC ≥ 0.78, Precision ≥ 0.70 on failure class.

---

## Hardware Requirements

| Tier | Runtime | Notebook 04 Time | Notes |
|---|---|---|---|
| **CPU** (free Colab) | Standard | ~45–60 min | Fully supported. Optuna trials slower. |
| **GPU T4** (free Colab) | GPU → T4 | ~10–15 min | Recommended. LightGBM uses GPU tree method automatically. |
| **GPU A100** (Colab Pro) | GPU → A100 | ~5–8 min | Fastest option. |

To enable GPU: **Runtime → Change runtime type → T4 GPU**

> All notebooks are fully CPU-compatible. GPU is optional and only meaningfully speeds up Notebook 04 (LightGBM Optuna search).

---

## Downloading Artifacts from Colab

Notebook 05 includes a **Download Artifacts** cell that calls `google.colab.files.download()` for all 7 artifact files. Run that cell and your browser will prompt you to save each file.

Alternatively, from any notebook:

```python
from google.colab import files
from pathlib import Path

for f in Path("/content/outputs").glob("*.joblib"):
    files.download(str(f))

files.download("/content/outputs/feature_names.json")
files.download("/content/outputs/evaluation_report.json")
```

---

## Repository Structure

```
nv-lastmile-training/
├── README.md
├── requirements.txt                  # Pinned dependencies (also installed in Notebook 01)
├── notebooks/
│   ├── 01_data_download.ipynb        # Step 1: download raw data
│   ├── 02_eda.ipynb                  # Step 2: exploratory analysis
│   ├── 03_feature_engineering.ipynb  # Step 3: feature matrix + preprocessor
│   ├── 04_model_training.ipynb       # Step 4: train + tune all models
│   └── 05_model_evaluation.ipynb     # Step 5: evaluate + export champion
├── src/
│   ├── __init__.py
│   ├── data_loader.py     # HuggingFace + Kaggle download helpers
│   ├── features.py        # Feature engineering pipeline (20 features)
│   ├── train.py           # Training functions (LightGBM, RF, LR + Optuna/GridSearch)
│   └── evaluate.py        # Evaluation metrics, plots, report generation
├── config/
│   └── config.yaml        # All paths, hyperparameter grids, and flags
└── outputs/               # Git-ignored; populated after running the pipeline
```

---

## Primary Dataset

| Dataset | Source | Size | Licence |
|---|---|---|---|
| **LaDe** (Cainiao-AI/LaDe) | [HuggingFace](https://huggingface.co/datasets/Cainiao-AI/LaDe) | ~10.67M packages | See HuggingFace dataset card |
| Amazon Delivery | [Kaggle](https://kaggle.com/datasets/sujalsuthar/amazon-delivery-dataset) | ~45k records | CC0 |

> Raw data is downloaded at runtime and never committed to this repository.

---

## Handoff to Demo App

Once Notebook 05 completes, copy these files into `nv-lastmile-demo/models/`:

```
/content/outputs/champion_model.joblib  →  nv-lastmile-demo/models/champion_model.joblib
/content/outputs/preprocessor.joblib   →  nv-lastmile-demo/models/preprocessor.joblib
/content/outputs/feature_names.json    →  nv-lastmile-demo/models/feature_names.json
```

The demo app ships with placeholder models trained on synthetic data so it can run immediately. Replacing those files with the real Colab-trained models upgrades it to production quality with no code changes.

---

## Configuration (`config/config.yaml`)

Key settings you may want to adjust:

| Key | Default | Description |
|---|---|---|
| `data.test_size` | `0.2` | Fraction of data held out for evaluation |
| `data.random_state` | `42` | Global random seed for reproducibility |
| `features.use_smote` | `false` | Set `true` to enable SMOTE oversampling |
| `models.lgbm.optuna_trials` | `50` | Increase for better LightGBM tuning (slower) |
| `models.rf.param_grid` | see file | GridSearch grid for Random Forest |
| `models.lr.param_grid` | see file | GridSearch grid for Logistic Regression |

---

*Ninja Van Capstone 2026 | Report compiled April 2026*
