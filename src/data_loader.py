"""
Data Acquisition Module for Financial Time Series
=================================================
Handles reliable fetching, caching, and persistence of raw OHLCV market data.
Supports primary (NIFTY 50), backup, and external robustness benchmark indices.
"""

import os
import logging
from typing import Optional
import pandas as pd
import yfinance as yf
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config/config.yaml") -> dict:
    """
    Load central project YAML configuration.

    Parameters
    ----------
    config_path : str
        Path to YAML configuration file.

    Returns
    -------
    dict
        Parsed configuration dictionary.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def fetch_market_data(
    ticker: str = "^NSEI",
    start_date: str = "2010-01-01",
    end_date: str = "2024-12-31",
    save_path: Optional[str] = "data/raw/nifty50_daily_raw.csv",
    force_reload: bool = False
) -> pd.DataFrame:
    """
    Fetch daily OHLCV historical time series from Yahoo Finance with disk caching.

    Parameters
    ----------
    ticker : str
        Market index or asset ticker symbol (e.g. '^NSEI' for NIFTY 50).
    start_date : str
        Start date formatted as 'YYYY-MM-DD'.
    end_date : str
        End date formatted as 'YYYY-MM-DD'.
    save_path : Optional[str]
        Local file path to cache raw CSV.
    force_reload : bool
        If True, re-downloads even if local cached file exists.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: Date, Open, High, Low, Close, Adj Close, Volume.
    """
    if save_path and os.path.exists(save_path) and not force_reload:
        logger.info(f"Loading cached raw data from {save_path}")
        df = pd.read_csv(save_path, parse_dates=["Date"])
        df.set_index("Date", inplace=True)
        return df

    logger.info(f"Downloading historical data for {ticker} ({start_date} to {end_date})...")
    
    # Attempt download via yfinance
    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=False)
    except Exception as e:
        logger.error(f"Failed to download {ticker}: {e}")
        data = pd.DataFrame()

    if data.empty:
        # Fallback tickers for NIFTY
        alt_tickers = ["^NSEI", "NIFTY_50.NS", "^NSEBANK"]
        for alt in alt_tickers:
            if alt != ticker:
                logger.info(f"Attempting fallback download with ticker '{alt}'...")
                try:
                    data = yf.download(alt, start=start_date, end=end_date, progress=False, auto_adjust=False)
                    if not data.empty:
                        ticker = alt
                        break
                except Exception:
                    continue

    if data.empty:
        raise ValueError(f"Could not retrieve historical data for ticker '{ticker}'. Please verify internet connection.")

    # Flatten MultiIndex columns if returned by yfinance >= 0.2.30
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] for col in data.columns]

    data.index.name = "Date"
    data = data.sort_index()

    # Ensure required columns are present
    required_cols = ["Open", "High", "Low", "Close", "Volume"]
    for col in required_cols:
        if col not in data.columns:
            raise KeyError(f"Required column '{col}' missing from downloaded market data.")

    if "Adj Close" not in data.columns:
        data["Adj Close"] = data["Close"]

    # Cache locally
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        data.to_csv(save_path)
        logger.info(f"Saved {len(data)} observations to {save_path}")

    return data
