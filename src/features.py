"""
features.py
===========
Feature engineering for the last-mile delivery failure prediction model.

All functions are pure transformations on a pandas DataFrame.
The ``build_full_feature_matrix`` function is the main entry point called
from ``03_feature_engineering.ipynb``.

LaDe Column Reference (key fields used here)
----------------------------------------------
- courier_id          : unique courier identifier
- order_id            : unique order/parcel identifier
- recipient_id        : unique recipient identifier (may be hashed)
- city                : city code
- district            : district/sub-area code
- ds                  : date string (YYYY-MM-DD)
- start_time          : earliest acceptable delivery timestamp
- end_time            : latest acceptable delivery timestamp
- got_time            : time courier received the task
- finish_time         : actual delivery completion time (NaN if not done)
- delivery_status     : outcome label (accepted/failed/etc.)
- package_weight      : float, kg
- declared_value      : float, monetary value
- address             : address string
"""

from __future__ import annotations

import re
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUMERIC_FEATURES: List[str] = [
    "hour_of_day",
    "day_of_week",
    "days_since_order",
    "time_window_duration_mins",
    "time_window_start_hour",
    "time_window_end_hour",
    "courier_daily_parcel_count",
    "courier_historical_failure_rate",
    "courier_experience_days",
    "address_length_chars",
    "address_token_count",
    "recipient_historical_failure_rate",
    "recipient_attempts_count",
]

BINARY_FEATURES: List[str] = [
    "is_weekend",
    "is_narrow_window",
    "has_unit_number",
]

CATEGORICAL_FEATURES: List[str] = [
    "city_encoded",
    "district_encoded",
    "package_weight_bucket",
    "declared_value_bucket",
]

ALL_FEATURE_COLS: List[str] = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES

TARGET_COL = "delivery_failed"

# Keyword hints for unit/floor numbers in addresses
_UNIT_KEYWORDS = re.compile(
    r"\b(unit|apt|apartment|floor|fl|#|blk|block|lot|room|rm|suite|ste|no\.?)\b",
    re.IGNORECASE,
)

# Weight bucket edges (kg)
_WEIGHT_BINS = [0, 1, 5, 15, float("inf")]
_WEIGHT_LABELS = ["light", "medium", "heavy", "oversized"]

# Declared value bucket edges (currency units — dataset-agnostic)
_VALUE_BINS = [0, 100, 500, float("inf")]
_VALUE_LABELS = ["low", "medium", "high"]


# ---------------------------------------------------------------------------
# Target derivation
# ---------------------------------------------------------------------------

def derive_target(df: pd.DataFrame) -> pd.Series:
    """
    Derive binary ``delivery_failed`` label from LaDe's ``delivery_status`` column.

    LaDe status values observed:
    - "DELIVERED"  / "delivered"   → success (0)
    - "FAILED"     / "failed"       → failure (1)
    - "RETURNED"                    → failure (1)
    - Other / NaN                   → NaN (dropped later)

    Returns
    -------
    pd.Series of int (0/1) with same index as df.
    """
    if "delivery_status" in df.columns:
        status = df["delivery_status"].astype(str).str.upper().str.strip()
        failed = status.isin(["FAILED", "RETURNED", "EXCEPTION", "REFUSED"])
        success = status.isin(["DELIVERED", "SUCCESS", "COMPLETED"])
        result = pd.Series(np.nan, index=df.index)
        result[failed] = 1
        result[success] = 0
        return result.astype("Int64")

    # Fallback: if finish_time is NaN → failed
    if "finish_time" in df.columns:
        return df["finish_time"].isna().astype(int)

    raise KeyError(
        "Cannot derive target: neither 'delivery_status' nor 'finish_time' column found."
    )


# ---------------------------------------------------------------------------
# Temporal features
# ---------------------------------------------------------------------------

def build_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build time-based features from LaDe timestamp columns.

    New columns added (all original columns preserved):
    - hour_of_day             : int 0-23 (from start_time or got_time)
    - day_of_week             : int 0 (Mon) – 6 (Sun)
    - is_weekend              : int 0/1
    - days_since_order        : int, days between order date and delivery date
    - time_window_duration_mins : float, end_time - start_time in minutes
    - time_window_start_hour  : int 0-23
    - time_window_end_hour    : int 0-23
    - is_narrow_window        : int 0/1 (1 if window < 120 min)

    Parameters
    ----------
    df : pd.DataFrame
        Raw LaDe dataframe. Must contain at least one of:
        ``start_time``, ``got_time``, ``ds``.

    Returns
    -------
    pd.DataFrame with new temporal feature columns.
    """
    df = df.copy()

    # --- Parse timestamps ---
    for col in ["start_time", "end_time", "got_time", "finish_time"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Reference datetime for hour_of_day and day_of_week
    ref_col = next(
        (c for c in ["start_time", "got_time"] if c in df.columns), None
    )
    if ref_col is None:
        # Try constructing from date string 'ds'
        if "ds" in df.columns:
            df["_ref_dt"] = pd.to_datetime(df["ds"], errors="coerce")
        else:
            df["_ref_dt"] = pd.NaT
    else:
        df["_ref_dt"] = df[ref_col]

    df["hour_of_day"] = df["_ref_dt"].dt.hour.fillna(12).astype(int)
    df["day_of_week"] = df["_ref_dt"].dt.dayofweek.fillna(2).astype(int)  # 0=Mon
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # Days since order
    if "ds" in df.columns and ref_col is not None:
        order_date = pd.to_datetime(df["ds"], errors="coerce")
        df["days_since_order"] = (
            df["_ref_dt"].dt.normalize() - order_date.dt.normalize()
        ).dt.days.clip(lower=0).fillna(2)
    else:
        df["days_since_order"] = 2

    # Time window features
    if "start_time" in df.columns and "end_time" in df.columns:
        df["time_window_duration_mins"] = (
            (df["end_time"] - df["start_time"]).dt.total_seconds() / 60
        ).clip(lower=0).fillna(120)
        df["time_window_start_hour"] = df["start_time"].dt.hour.fillna(9).astype(int)
        df["time_window_end_hour"] = df["end_time"].dt.hour.fillna(21).astype(int)
    else:
        df["time_window_duration_mins"] = 120.0
        df["time_window_start_hour"] = 9
        df["time_window_end_hour"] = 21

    df["is_narrow_window"] = (df["time_window_duration_mins"] < 120).astype(int)

    df.drop(columns=["_ref_dt"], inplace=True, errors="ignore")
    return df


# ---------------------------------------------------------------------------
# Courier features (expanding window — leakage-safe)
# ---------------------------------------------------------------------------

def build_courier_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build courier-level aggregate features using an expanding window
    (sorted by date) to prevent data leakage.

    Requires columns: ``courier_id``, ``ds``, and ``delivery_failed`` target.

    New columns:
    - courier_daily_parcel_count        : number of deliveries for this courier on this day
    - courier_historical_failure_rate   : expanding mean failure rate up to (but not including) current date
    - courier_experience_days           : number of distinct dates the courier has been active

    Parameters
    ----------
    df : pd.DataFrame
        Must have been processed by ``build_temporal_features`` and have
        ``delivery_failed`` column.

    Returns
    -------
    pd.DataFrame with new courier feature columns.
    """
    df = df.copy()

    if "courier_id" not in df.columns:
        df["courier_id"] = "unknown"
    if "ds" not in df.columns:
        df["ds"] = "2023-01-01"
    df["ds"] = pd.to_datetime(df["ds"], errors="coerce").dt.normalize()

    # Sort for expanding window correctness
    df = df.sort_values(["courier_id", "ds"]).reset_index(drop=True)

    # Daily parcel count
    daily_counts = (
        df.groupby(["courier_id", "ds"])
        .size()
        .reset_index(name="courier_daily_parcel_count")
    )
    df = df.merge(daily_counts, on=["courier_id", "ds"], how="left")
    df["courier_daily_parcel_count"] = df["courier_daily_parcel_count"].fillna(1)

    # Expanding historical failure rate (shifted by 1 day to avoid leakage)
    target_col = TARGET_COL if TARGET_COL in df.columns else None
    if target_col:
        # Per-courier daily failure rate, then expanding mean over past days
        daily_fail = (
            df.groupby(["courier_id", "ds"])[target_col]
            .mean()
            .reset_index()
            .rename(columns={target_col: "_daily_fail_rate"})
        )
        daily_fail = daily_fail.sort_values(["courier_id", "ds"])
        # Expanding mean — shift so we don't include today
        daily_fail["courier_historical_failure_rate"] = (
            daily_fail.groupby("courier_id")["_daily_fail_rate"]
            .transform(lambda x: x.shift(1).expanding().mean())
        )
        daily_fail.drop(columns=["_daily_fail_rate"], inplace=True)
        df = df.merge(daily_fail, on=["courier_id", "ds"], how="left")
        # Fill NaN for courier's first day with global mean
        global_mean = df[target_col].mean() if target_col in df.columns else 0.15
        df["courier_historical_failure_rate"] = (
            df["courier_historical_failure_rate"].fillna(global_mean)
        )
    else:
        df["courier_historical_failure_rate"] = 0.15

    # Courier experience: distinct active days up to current date
    experience = (
        df.groupby("courier_id")["ds"]
        .transform(lambda x: x.rank(method="dense").astype(int))
    )
    df["courier_experience_days"] = experience.fillna(1).astype(int)

    return df


# ---------------------------------------------------------------------------
# Address features
# ---------------------------------------------------------------------------

def build_address_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build proxy address quality features from the raw address string.

    New columns:
    - address_length_chars  : total character length of address string
    - has_unit_number       : 1 if address contains unit/floor/apt keywords
    - address_token_count   : number of whitespace-separated tokens

    Parameters
    ----------
    df : pd.DataFrame
        Must have an ``address`` column (or similar). Falls back gracefully
        if the column is absent.

    Returns
    -------
    pd.DataFrame with new address feature columns.
    """
    df = df.copy()

    # Try to find address column
    addr_col = next(
        (c for c in ["address", "receiver_address", "addr", "location"] if c in df.columns),
        None,
    )

    if addr_col is None:
        df["address_length_chars"] = 50
        df["has_unit_number"] = 0
        df["address_token_count"] = 7
        return df

    addr = df[addr_col].fillna("").astype(str)
    df["address_length_chars"] = addr.str.len()
    df["has_unit_number"] = addr.str.contains(_UNIT_KEYWORDS).astype(int)
    df["address_token_count"] = addr.str.split().str.len().fillna(0).astype(int)

    return df


# ---------------------------------------------------------------------------
# Recipient features (expanding window)
# ---------------------------------------------------------------------------

def build_recipient_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build recipient-level aggregate features using an expanding window.

    New columns:
    - recipient_historical_failure_rate : expanding mean failure rate for this recipient
    - recipient_attempts_count          : number of prior delivery attempts for this recipient

    Parameters
    ----------
    df : pd.DataFrame
        Must have ``recipient_id`` (or ``receiver_id``) column and
        ``delivery_failed`` target column.

    Returns
    -------
    pd.DataFrame with new recipient feature columns.
    """
    df = df.copy()

    # Normalise recipient id column name
    recip_col = next(
        (c for c in ["recipient_id", "receiver_id", "customer_id", "user_id"] if c in df.columns),
        None,
    )
    if recip_col is None:
        df["recipient_historical_failure_rate"] = 0.15
        df["recipient_attempts_count"] = 0
        return df

    if "ds" not in df.columns:
        df["ds"] = "2023-01-01"
    df["ds"] = pd.to_datetime(df["ds"], errors="coerce").dt.normalize()
    df = df.sort_values([recip_col, "ds"]).reset_index(drop=True)

    target_col = TARGET_COL if TARGET_COL in df.columns else None

    if target_col:
        # Expanding failure rate per recipient (shift to avoid leakage)
        df["recipient_historical_failure_rate"] = (
            df.groupby(recip_col)[target_col]
            .transform(lambda x: x.shift(1).expanding().mean())
        )
        global_mean = df[target_col].mean() if target_col in df.columns else 0.15
        df["recipient_historical_failure_rate"] = (
            df["recipient_historical_failure_rate"].fillna(global_mean)
        )
    else:
        df["recipient_historical_failure_rate"] = 0.15

    # Cumulative attempt count (prior attempts for this recipient)
    df["recipient_attempts_count"] = (
        df.groupby(recip_col).cumcount()  # 0-indexed count of prior rows
    )

    return df


# ---------------------------------------------------------------------------
# Geo / categorical features
# ---------------------------------------------------------------------------

def build_geo_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode city and district as string category columns for the
    ColumnTransformer's OrdinalEncoder to handle.

    The ColumnTransformer in ``get_preprocessor`` will apply OrdinalEncoder
    to these columns, so we just ensure they exist and are string-typed here.

    New columns:
    - city_encoded     : string city label (for OrdinalEncoder)
    - district_encoded : string district label (for OrdinalEncoder)

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame with geo columns normalised.
    """
    df = df.copy()

    city_col = next(
        (c for c in ["city", "city_id", "city_code"] if c in df.columns), None
    )
    district_col = next(
        (c for c in ["district", "district_id", "region"] if c in df.columns), None
    )

    df["city_encoded"] = (
        df[city_col].fillna("Unknown").astype(str) if city_col else "Unknown"
    )
    df["district_encoded"] = (
        df[district_col].fillna("Unknown").astype(str) if district_col else "Unknown"
    )

    return df


def build_parcel_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bin package weight and declared value into categorical buckets.

    New columns:
    - package_weight_bucket  : "light" / "medium" / "heavy" / "oversized"
    - declared_value_bucket  : "low" / "medium" / "high"

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame with bucket columns added.
    """
    df = df.copy()

    weight_col = next(
        (c for c in ["package_weight", "weight", "item_weight"] if c in df.columns), None
    )
    value_col = next(
        (c for c in ["declared_value", "item_price", "price", "value"] if c in df.columns),
        None,
    )

    if weight_col and pd.api.types.is_numeric_dtype(df[weight_col]):
        df["package_weight_bucket"] = pd.cut(
            df[weight_col].fillna(1.0).clip(lower=0),
            bins=_WEIGHT_BINS,
            labels=_WEIGHT_LABELS,
            right=False,
        ).astype(str)
    else:
        df["package_weight_bucket"] = "medium"

    if value_col and pd.api.types.is_numeric_dtype(df[value_col]):
        df["declared_value_bucket"] = pd.cut(
            df[value_col].fillna(100.0).clip(lower=0),
            bins=_VALUE_BINS,
            labels=_VALUE_LABELS,
            right=False,
        ).astype(str)
    else:
        df["declared_value_bucket"] = "medium"

    return df


# ---------------------------------------------------------------------------
# Main pipeline entry point
# ---------------------------------------------------------------------------

def build_full_feature_matrix(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Apply all feature engineering steps in sequence and return the final
    feature matrix and target series.

    Steps (in order):
    1. Derive target (delivery_failed)
    2. Temporal features
    3. Address features
    4. Geo + parcel buckets
    5. Recipient features (requires target → must come after step 1)
    6. Courier features (requires target → must come after step 1)
    7. Select final feature columns

    Parameters
    ----------
    df : pd.DataFrame
        Raw LaDe dataframe (as loaded by data_loader.load_raw).

    Returns
    -------
    X : pd.DataFrame  — feature matrix with columns = ALL_FEATURE_COLS
    y : pd.Series     — binary target (delivery_failed)
    """
    print("🔧 Building feature matrix…")

    # Step 1: target
    df = df.copy()
    df[TARGET_COL] = derive_target(df)
    # Drop rows where target could not be determined
    n_before = len(df)
    df = df.dropna(subset=[TARGET_COL])
    if len(df) < n_before:
        print(f"   Dropped {n_before - len(df):,} rows with unknown target")

    # Step 2-4: deterministic features
    df = build_temporal_features(df)
    df = build_address_features(df)
    df = build_geo_features(df)
    df = build_parcel_features(df)

    # Step 5-6: expanding window (requires sorted data + target)
    df = build_recipient_features(df)
    df = build_courier_features(df)

    # Step 7: select and coerce columns
    missing = [c for c in ALL_FEATURE_COLS if c not in df.columns]
    if missing:
        print(f"   ⚠️  Adding zero-fill for missing columns: {missing}")
        for c in missing:
            df[c] = 0

    X = df[ALL_FEATURE_COLS].copy()
    y = df[TARGET_COL].astype(int)

    # Coerce binary features to int
    for col in BINARY_FEATURES:
        X[col] = X[col].astype(int)

    print(f"✅ Feature matrix: {X.shape[0]:,} rows × {X.shape[1]} features")
    print(f"   Failure rate: {y.mean():.3f} ({y.sum():,} failed / {len(y):,} total)")

    return X, y


# ---------------------------------------------------------------------------
# Preprocessor factory
# ---------------------------------------------------------------------------

def get_preprocessor(
    numeric_cols: List[str] = None,
    categorical_cols: List[str] = None,
) -> ColumnTransformer:
    """
    Build an unfitted sklearn ColumnTransformer preprocessor.

    Numeric pipeline  : SimpleImputer(median)  → StandardScaler
    Categorical pipeline : SimpleImputer(most_frequent) → OrdinalEncoder

    Binary (0/1) features are treated as numeric.

    Parameters
    ----------
    numeric_cols : list of str, optional
        Defaults to ``NUMERIC_FEATURES + BINARY_FEATURES``.
    categorical_cols : list of str, optional
        Defaults to ``CATEGORICAL_FEATURES``.

    Returns
    -------
    sklearn.compose.ColumnTransformer (unfitted)
    """
    if numeric_cols is None:
        numeric_cols = NUMERIC_FEATURES + BINARY_FEATURES
    if categorical_cols is None:
        categorical_cols = CATEGORICAL_FEATURES

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocessor
