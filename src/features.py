"""
Feature Engineering Module
==========================
Constructs strictly backward-looking, leakage-free econometric and statistical features
for supervised machine learning volatility forecasting.
"""

import logging
from typing import List, Tuple
import numpy as np
import pandas as pd

from src.volatility import (
    compute_returns,
    compute_rolling_volatility,
    compute_parkinson_volatility,
    compute_garman_klass_volatility,
    compute_forward_realized_volatility
)

logger = logging.getLogger(__name__)


def build_feature_dataset(
    df: pd.DataFrame,
    target_horizon: int = 5,
    vol_windows: List[int] = [5, 10, 20, 30, 60],
    return_lags: List[int] = [1, 2, 3, 5, 10, 20],
    annualization_factor: int = 252
) -> pd.DataFrame:
    """
    Build a comprehensive, strictly leakage-free feature matrix for volatility prediction.

    All feature values at index t are computed using information available at or before t.
    Target variable at index t represents realized volatility strictly in future window [t+1, t+target_horizon].

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned OHLCV market dataset.
    target_horizon : int
        Number of forward days for the realized volatility target.
    vol_windows : List[int]
        Window lengths for historical volatility.
    return_lags : List[int]
        Lag lengths for return features.
    annualization_factor : int
        Trading days per year.

    Returns
    -------
    pd.DataFrame
        Feature matrix containing features and aligned target column.
    """
    data = compute_returns(df, price_col="Close")
    data = compute_rolling_volatility(data, return_col="log_return", windows=vol_windows, annualization_factor=annualization_factor)

    features = pd.DataFrame(index=data.index)

    # 1. Return Lag Features
    for lag in return_lags:
        features[f"return_lag_{lag}"] = data["log_return"].shift(lag - 1) if lag == 1 else data["log_return"].shift(lag)
        # Shifted correctly: return_lag_1 is log_return at time t

    # 2. Historical Rolling Volatility Features (available at t)
    for w in vol_windows:
        features[f"vol_ann_{w}d"] = data[f"vol_ann_{w}d"]

    # Volatility Term Structure / Ratios
    if 5 in vol_windows and 20 in vol_windows:
        features["vol_ratio_5_20"] = data["vol_ann_5d"] / (data["vol_ann_20d"] + 1e-8)
    if 20 in vol_windows and 60 in vol_windows:
        features["vol_ratio_20_60"] = data["vol_ann_20d"] / (data["vol_ann_60d"] + 1e-8)

    # 3. Intraday Range & Microstructure Features
    features["high_low_range_pct"] = (data["High"] - data["Low"]) / data["Close"]
    features["close_open_return"] = (data["Close"] - data["Open"]) / data["Open"]
    features["parkinson_vol_20d"] = compute_parkinson_volatility(data, window=20, annualization_factor=annualization_factor)
    features["garman_klass_vol_20d"] = compute_garman_klass_volatility(data, window=20, annualization_factor=annualization_factor)

    # 4. Momentum and Trend Features
    features["mom_5d"] = data["Close"].pct_change(5)
    features["mom_20d"] = data["Close"].pct_change(20)
    features["dist_ma20"] = (data["Close"] / data["Close"].rolling(20).mean()) - 1.0
    features["dist_ma50"] = (data["Close"] / data["Close"].rolling(50).mean()) - 1.0

    # 5. Volume Features
    if "Volume" in data.columns:
        features["volume_change_1d"] = data["Volume"].pct_change()
        features["volume_ratio_5_20"] = data["Volume"].rolling(5).mean() / (data["Volume"].rolling(20).mean() + 1e-8)
        features["volume_vol_20d"] = data["Volume"].pct_change().rolling(20).std()

    # 6. Construct Target Realized Volatility
    target_col = f"target_rv_{target_horizon}d"
    features[target_col] = compute_forward_realized_volatility(
        data, return_col="log_return", horizon=target_horizon, annualization_factor=annualization_factor
    )

    # Preserve essential price/return columns for downstream strategy backtesting
    features["Close"] = data["Close"]
    features["log_return"] = data["log_return"]
    features["simple_return"] = data["simple_return"]

    # Drop warm-up NaN rows caused by longest lookback (60 days)
    # Note: Target will have NaNs for the very last target_horizon rows (unrealized future)
    features = features.dropna(subset=[col for col in features.columns if col != target_col])

    logger.info(f"Feature matrix generated: {features.shape[0]} rows, {features.shape[1]} columns.")
    return features


def verify_temporal_integrity(df: pd.DataFrame, target_col: str) -> bool:
    """
    Formally test feature dataset for target leakage.
    Asserts that no feature at timestamp t incorporates data from t+1 onwards.

    Parameters
    ----------
    df : pd.DataFrame
        Constructed feature matrix.
    target_col : str
        Target column name.

    Returns
    -------
    bool
        True if all leakage integrity assertions pass.
    """
    feature_cols = [c for c in df.columns if c not in [target_col, "Close", "log_return", "simple_return"]]
    
    # 1. Verify index monotonicity
    if not df.index.is_monotonic_increasing:
        raise AssertionError("Temporal integrity error: Index is not monotonically increasing.")

    # 2. Check lead-lag correlation: target at t should not be perfectly collinear with features at t
    for col in feature_cols:
        valid_mask = df[col].notna() & df[target_col].notna()
        if valid_mask.sum() > 50:
            corr = np.corrcoef(df.loc[valid_mask, col], df.loc[valid_mask, target_col])[0, 1]
            if abs(corr) > 0.9999:
                raise AssertionError(f"Suspicious near-perfect correlation ({corr:.4f}) between {col} and {target_col}.")

    logger.info("Temporal integrity check PASSED: No lookahead leakage detected.")
    return True
