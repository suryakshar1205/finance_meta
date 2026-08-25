"""
Data Cleaning and Preprocessing Module
======================================
Implements rigorous financial data validation, chronological sorting,
duplicate resolution, missing-value handling, and consistency checks.
"""

import os
import logging
from typing import Tuple, Dict, Any
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def clean_market_data(
    df: pd.DataFrame,
    save_path: str = "data/processed/nifty50_daily_processed.csv"
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Clean and validate raw OHLCV financial market time series.

    Parameters
    ----------
    df : pd.DataFrame
        Raw market data with DateTimeIndex or 'Date' column.
    save_path : str
        Destination path for cleaned output.

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, Any]]
        Cleaned DataFrame and execution log audit dictionary.
    """
    audit = {
        "initial_rows": len(df),
        "duplicate_rows_removed": 0,
        "missing_values_imputed": 0,
        "invalid_price_anomalies": 0,
        "calendar_gaps_detected": 0
    }
    
    cleaned = df.copy()

    # 1. Ensure DatetimeIndex
    if not isinstance(cleaned.index, pd.DatetimeIndex):
        if "Date" in cleaned.columns:
            cleaned["Date"] = pd.to_datetime(cleaned["Date"])
            cleaned.set_index("Date", inplace=True)
        else:
            cleaned.index = pd.to_datetime(cleaned.index)

    cleaned.index.name = "Date"

    # 2. Chronological Sorting
    cleaned = cleaned.sort_index()

    # 3. Duplicate Detection and Removal
    duplicates = cleaned.index.duplicated(keep="first")
    n_dupes = duplicates.sum()
    if n_dupes > 0:
        logger.warning(f"Detected and dropped {n_dupes} duplicate date entries.")
        cleaned = cleaned[~duplicates]
        audit["duplicate_rows_removed"] = int(n_dupes)

    # 4. Missing Value Inspection
    missing_counts = cleaned.isna().sum()
    total_missing = int(missing_counts.sum())
    if total_missing > 0:
        logger.info(f"Missing values found before imputation:\n{missing_counts[missing_counts > 0]}")
        # Forward fill prices (last observed market state), then backward fill if leading NaNs
        cleaned = cleaned.ffill().bfill()
        audit["missing_values_imputed"] = total_missing

    # 5. Data Consistency Checks
    # Non-positive prices check
    price_cols = ["Open", "High", "Low", "Close"]
    for col in price_cols:
        if (cleaned[col] <= 0).any():
            invalid_mask = cleaned[col] <= 0
            n_inv = invalid_mask.sum()
            audit["invalid_price_anomalies"] += int(n_inv)
            logger.error(f"Found {n_inv} non-positive prices in column '{col}'. Replacing with forward-fill.")
            cleaned.loc[invalid_mask, col] = np.nan
            cleaned[col] = cleaned[col].ffill()

    # High-Low Consistency Check: High must be >= max(Open, Close), Low must be <= min(Open, Close)
    high_invalid = (cleaned["High"] < cleaned["Low"]) | \
                   (cleaned["High"] < cleaned["Open"]) | \
                   (cleaned["High"] < cleaned["Close"])
    if high_invalid.any():
        n_high_inv = int(high_invalid.sum())
        logger.warning(f"Fixing {n_high_inv} High price inconsistencies to max(High, Open, Close).")
        cleaned.loc[high_invalid, "High"] = cleaned.loc[high_invalid, ["High", "Open", "Close"]].max(axis=1)
        audit["invalid_price_anomalies"] += n_high_inv

    low_invalid = (cleaned["Low"] > cleaned["High"]) | \
                  (cleaned["Low"] > cleaned["Open"]) | \
                  (cleaned["Low"] > cleaned["Close"])
    if low_invalid.any():
        n_low_inv = int(low_invalid.sum())
        logger.warning(f"Fixing {n_low_inv} Low price inconsistencies to min(Low, Open, Close).")
        cleaned.loc[low_invalid, "Low"] = cleaned.loc[low_invalid, ["Low", "Open", "Close"]].min(axis=1)
        audit["invalid_price_anomalies"] += n_low_inv

    # Non-negative volume
    if "Volume" in cleaned.columns:
        cleaned["Volume"] = cleaned["Volume"].clip(lower=0)

    # 6. Calendar Gap Analysis (Business days check)
    full_bday_index = pd.date_range(start=cleaned.index.min(), end=cleaned.index.max(), freq="B")
    missing_bdays = full_bday_index.difference(cleaned.index)
    audit["calendar_gaps_detected"] = len(missing_bdays)
    audit["final_rows"] = len(cleaned)
    audit["date_range"] = f"{cleaned.index.min().strftime('%Y-%m-%d')} to {cleaned.index.max().strftime('%Y-%m-%d')}"

    logger.info(
        f"Data cleaning complete: {audit['initial_rows']} raw rows -> {audit['final_rows']} valid rows "
        f"over {audit['date_range']} ({audit['calendar_gaps_detected']} non-trading business days/holidays)."
    )

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cleaned.to_csv(save_path)
        logger.info(f"Cleaned dataset persisted to {save_path}")

    return cleaned, audit
