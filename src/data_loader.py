"""
data_loader.py
==============
Helpers for downloading and loading datasets from HuggingFace (LaDe) and
Kaggle (Amazon Delivery). All functions are designed to be called from
Colab notebooks but are importable as a plain Python module.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ensure_dir(path: str) -> Path:
    """Create directory if it doesn't exist and return a Path object."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _pip_install(package: str) -> None:
    """Install a pip package at runtime (useful inside Colab cells)."""
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])


# ---------------------------------------------------------------------------
# LaDe dataset (HuggingFace)
# ---------------------------------------------------------------------------

def download_lade(output_dir: str, hf_token: Optional[str] = None) -> pd.DataFrame:
    """
    Download the LaDe last-mile delivery dataset from HuggingFace and save
    raw parquet files to ``output_dir/lade/``.

    LaDe contains ~10.67 M packages across 6 months of real-world courier
    operations from Cainiao (Alibaba Logistics).

    Parameters
    ----------
    output_dir : str
        Root directory for raw data (e.g. ``/content/data/raw``).
    hf_token : str, optional
        HuggingFace access token. If None, attempts to use the
        ``HF_TOKEN`` environment variable.

    Returns
    -------
    pd.DataFrame
        Combined dataframe of all LaDe splits (train + test).
    """
    try:
        from datasets import load_dataset
        from huggingface_hub import login
    except ImportError:
        _pip_install("datasets huggingface_hub")
        from datasets import load_dataset
        from huggingface_hub import login

    token = hf_token or os.environ.get("HF_TOKEN")
    if token:
        login(token=token, add_to_git_credential=False)
        print("✅ HuggingFace authenticated")
    else:
        print("⚠️  No HF_TOKEN found — attempting anonymous download")

    print("⏳ Downloading LaDe dataset from HuggingFace (this may take a few minutes)…")
    dataset = load_dataset("Cainiao-AI/LaDe", trust_remote_code=True)
    print(f"   Splits available: {list(dataset.keys())}")

    out_dir = _ensure_dir(os.path.join(output_dir, "lade"))
    dfs = []
    for split_name, split_data in dataset.items():
        df_split = split_data.to_pandas()
        df_split["_split"] = split_name
        path = out_dir / f"lade_{split_name}.parquet"
        df_split.to_parquet(path, index=False)
        print(f"   Saved {split_name}: {len(df_split):,} rows → {path}")
        dfs.append(df_split)

    df_all = pd.concat(dfs, ignore_index=True)
    print(f"\n✅ LaDe download complete — total rows: {len(df_all):,}")
    return df_all


# ---------------------------------------------------------------------------
# Amazon Delivery dataset (Kaggle)
# ---------------------------------------------------------------------------

def _configure_kaggle(kaggle_creds: Optional[Dict[str, str]] = None) -> None:
    """
    Set up Kaggle API credentials either from the supplied dict or from the
    ``KAGGLE_KEY`` environment variable (JSON string).

    The ``KAGGLE_KEY`` secret in Colab should contain the full JSON content
    of your kaggle.json file:
        {"username": "you", "key": "abc123..."}
    """
    creds = kaggle_creds

    if creds is None:
        raw = os.environ.get("KAGGLE_KEY", "")
        if raw:
            try:
                creds = json.loads(raw)
            except json.JSONDecodeError:
                raise ValueError(
                    "KAGGLE_KEY environment variable is not valid JSON. "
                    "It should contain the contents of your kaggle.json file."
                )

    if creds:
        kaggle_dir = Path.home() / ".kaggle"
        kaggle_dir.mkdir(exist_ok=True)
        kaggle_json = kaggle_dir / "kaggle.json"
        with open(kaggle_json, "w") as f:
            json.dump(creds, f)
        kaggle_json.chmod(0o600)
        print("✅ Kaggle credentials written to ~/.kaggle/kaggle.json")
    else:
        # Try existing ~/.kaggle/kaggle.json
        if not (Path.home() / ".kaggle" / "kaggle.json").exists():
            raise FileNotFoundError(
                "No Kaggle credentials found. Either:\n"
                "  1. Set KAGGLE_KEY Colab Secret (JSON string from kaggle.json), or\n"
                "  2. Manually place kaggle.json in ~/.kaggle/"
            )
        print("ℹ️  Using existing ~/.kaggle/kaggle.json")


def download_amazon_kaggle(
    output_dir: str,
    kaggle_creds: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """
    Download the Amazon Delivery dataset from Kaggle and save to
    ``output_dir/amazon/``.

    Dataset: sujalsuthar/amazon-delivery-dataset

    Parameters
    ----------
    output_dir : str
        Root directory for raw data (e.g. ``/content/data/raw``).
    kaggle_creds : dict, optional
        Dict with keys ``username`` and ``key``. If None, falls back to
        KAGGLE_KEY env var or ~/.kaggle/kaggle.json.

    Returns
    -------
    pd.DataFrame
        Loaded Amazon delivery dataframe.
    """
    try:
        import kaggle  # noqa: F401
    except ImportError:
        _pip_install("kaggle")

    _configure_kaggle(kaggle_creds)

    out_dir = _ensure_dir(os.path.join(output_dir, "amazon"))
    print("⏳ Downloading Amazon Delivery dataset from Kaggle…")

    subprocess.check_call(
        [
            sys.executable, "-m", "kaggle",
            "datasets", "download",
            "-d", "sujalsuthar/amazon-delivery-dataset",
            "-p", str(out_dir),
            "--unzip",
        ]
    )

    # Find the CSV
    csv_files = list(out_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {out_dir} after Kaggle download.")

    df = pd.read_csv(csv_files[0])
    # Save as parquet for consistency
    parquet_path = out_dir / "amazon_delivery.parquet"
    df.to_parquet(parquet_path, index=False)
    print(f"✅ Amazon delivery: {len(df):,} rows saved → {parquet_path}")
    return df


# ---------------------------------------------------------------------------
# Generic loader
# ---------------------------------------------------------------------------

def load_raw(dataset: str, data_dir: str) -> pd.DataFrame:
    """
    Load a raw dataset from disk.

    Parameters
    ----------
    dataset : str
        One of ``"lade"`` or ``"amazon"``.
    data_dir : str
        Root raw data directory (e.g. ``/content/data/raw``).

    Returns
    -------
    pd.DataFrame
        Combined dataframe for the requested dataset.
    """
    dataset = dataset.lower().strip()
    base = Path(data_dir)

    if dataset == "lade":
        folder = base / "lade"
        files = sorted(folder.glob("*.parquet"))
        if not files:
            raise FileNotFoundError(
                f"No parquet files found in {folder}. Run download_lade() first."
            )
        dfs = [pd.read_parquet(f) for f in files]
        df = pd.concat(dfs, ignore_index=True)
        print(f"Loaded LaDe: {len(df):,} rows from {len(files)} file(s)")
        return df

    elif dataset == "amazon":
        path = base / "amazon" / "amazon_delivery.parquet"
        if not path.exists():
            # Fallback to CSV
            csv_path = base / "amazon" / "amazon_delivery.csv"
            if csv_path.exists():
                df = pd.read_csv(csv_path)
            else:
                raise FileNotFoundError(
                    f"Amazon delivery file not found at {path}. "
                    "Run download_amazon_kaggle() first."
                )
        else:
            df = pd.read_parquet(path)
        print(f"Loaded Amazon delivery: {len(df):,} rows")
        return df

    else:
        raise ValueError(f"Unknown dataset '{dataset}'. Choose from: 'lade', 'amazon'.")


# ---------------------------------------------------------------------------
# Schema inspection utility
# ---------------------------------------------------------------------------

def print_schema_summary(df: pd.DataFrame, name: str = "Dataset") -> None:
    """
    Print a summary of column dtypes, null counts, and sample values.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to inspect.
    name : str
        Display name for the dataset.
    """
    print(f"\n{'='*60}")
    print(f"  {name} — {len(df):,} rows × {len(df.columns)} cols")
    print(f"{'='*60}")
    summary = pd.DataFrame({
        "dtype":    df.dtypes,
        "nulls":    df.isnull().sum(),
        "null_%":   (df.isnull().mean() * 100).round(2),
        "sample":   df.iloc[0] if len(df) > 0 else pd.Series(dtype=object),
    })
    print(summary.to_string())
    print()
