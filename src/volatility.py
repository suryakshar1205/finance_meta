"""
Volatility Modeling and Estimators Module
==========================================
Computes simple & log returns, rolling sample volatilities, advanced intraday-range
volatility estimators (Parkinson, Garman-Klass), and forward realized volatility targets.
"""

import logging
from typing import List, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_returns(df: pd.DataFrame, price_col: str = "Close") -> pd.DataFrame:
    """
    Calculate simple and logarithmic returns for a given price series.

    Theoretical definition:
    - Simple Return: R_t = P_t / P_{t-1} - 1
    - Log Return:    r_t = ln(P_t / P_{t-1}) = ln(1 + R_t)

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing price series.
    price_col : str
        Name of the closing price column.

    Returns
    -------
    pd.DataFrame
        Original DataFrame with 'simple_return' and 'log_return' columns added.
    """
    out = df.copy()
    out["simple_return"] = out[price_col].pct_change()
    out["log_return"] = np.log(out[price_col] / out[price_col].shift(1))
    return out


def compute_rolling_volatility(
    df: pd.DataFrame,
    return_col: str = "log_return",
    windows: List[int] = [5, 10, 20, 30, 60],
    annualization_factor: int = 252
) -> pd.DataFrame:
    """
    Compute rolling sample standard deviation of returns across multiple time horizons.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with return series.
    return_col : str
        Column name of returns to calculate volatility on.
    windows : List[int]
        Rolling window lengths in trading days.
    annualization_factor : int
        Number of trading days per year (standard: 252).

    Returns
    -------
    pd.DataFrame
        DataFrame with daily and annualized rolling volatility columns.
    """
    out = df.copy()
    sqrt_ann = np.sqrt(annualization_factor)

    for w in windows:
        # Daily rolling standard deviation (unbiased estimator with ddof=1)
        out[f"vol_daily_{w}d"] = out[return_col].rolling(window=w, min_periods=w).std(ddof=1)
        # Annualized rolling volatility
        out[f"vol_ann_{w}d"] = out[f"vol_daily_{w}d"] * sqrt_ann

    return out


def compute_parkinson_volatility(
    df: pd.DataFrame,
    window: int = 20,
    annualization_factor: int = 252
) -> pd.Series:
    """
    Calculate Parkinson (1980) extreme-value volatility estimator using High and Low prices.
    
    Formula:
    sigma^2 = (1 / (4 * ln(2))) * sum(ln(High_i / Low_i)^2) / N

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'High' and 'Low' price columns.
    window : int
        Rolling window length.
    annualization_factor : int
        Trading days per year.

    Returns
    -------
    pd.Series
        Annualized Parkinson volatility series.
    """
    hl_ratio = np.log(df["High"] / df["Low"])
    factor = 1.0 / (4.0 * np.log(2.0))
    daily_var = factor * (hl_ratio ** 2).rolling(window=window, min_periods=window).mean()
    return np.sqrt(daily_var * annualization_factor)


def compute_garman_klass_volatility(
    df: pd.DataFrame,
    window: int = 20,
    annualization_factor: int = 252
) -> pd.Series:
    """
    Calculate Garman-Klass (1980) volatility estimator incorporating Open, High, Low, Close.
    
    Formula:
    daily_var = 0.5 * ln(H/L)^2 - (2*ln(2) - 1) * ln(C/O)^2

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'Open', 'High', 'Low', 'Close' price columns.
    window : int
        Rolling window length.
    annualization_factor : int
        Trading days per year.

    Returns
    -------
    pd.Series
        Annualized Garman-Klass volatility series.
    """
    log_hl = np.log(df["High"] / df["Low"])
    log_co = np.log(df["Close"] / df["Open"])
    gk_const = 2.0 * np.log(2.0) - 1.0
    daily_var = 0.5 * (log_hl ** 2) - gk_const * (log_co ** 2)
    daily_var_clipped = daily_var.clip(lower=0)
    rolling_var = daily_var_clipped.rolling(window=window, min_periods=window).mean()
    return np.sqrt(rolling_var * annualization_factor)


def compute_forward_realized_volatility(
    df: pd.DataFrame,
    return_col: str = "log_return",
    horizon: int = 5,
    annualization_factor: int = 252
) -> pd.Series:
    """
    Compute strictly forward-looking realized volatility target over [t+1, t+k].

    Definition:
    RV_{t, t+k} = sqrt( sum_{i=1}^k r_{t+i}^2 ) * sqrt(252 / k)

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with log return series.
    return_col : str
        Name of log return column.
    horizon : int
        Forecast horizon (k days into future).
    annualization_factor : int
        Annualization factor (252).

    Returns
    -------
    pd.Series
        Target forward realized volatility aligned at timestamp t.
    """
    sq_returns = df[return_col] ** 2
    # Rolling forward sum from t+1 to t+k:
    # shift(-horizon) rolling window shifted back
    forward_sum_sq = sq_returns.iloc[::-1].rolling(window=horizon, min_periods=horizon).sum().iloc[::-1].shift(-1)
    
    # Realized volatility annualized
    target_vol = np.sqrt(forward_sum_sq * (annualization_factor / horizon))
    target_vol.name = f"target_rv_{horizon}d"
    return target_vol


class HistoricalVolatilityBaseline:
    """
    Baseline benchmark that forecasts future volatility using past rolling historical volatility.
    """
    def __init__(self, window: int = 20, annualization_factor: int = 252):
        self.window = window
        self.annualization_factor = annualization_factor

    def predict(self, df: pd.DataFrame, return_col: str = "log_return") -> pd.Series:
        """
        Predict future volatility as the current rolling historical volatility at time t.
        """
        sqrt_ann = np.sqrt(self.annualization_factor)
        pred = df[return_col].rolling(window=self.window, min_periods=self.window).std(ddof=1) * sqrt_ann
        pred.name = f"pred_hist_vol_{self.window}d"
        return pred
