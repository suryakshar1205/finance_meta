"""
Time-Series Validation and Walk-Forward Evaluation Engine
=========================================================
Implements strict chronological train/validation/test splitting and an expanding/rolling
walk-forward out-of-sample forecasting pipeline across all competing volatility models.
"""

import logging
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.volatility import HistoricalVolatilityBaseline
from src.garch import GARCHModel
from src.ml_models import RandomForestVolatilityModel, XGBoostVolatilityModel
from src.hybrid import HybridGARCHMLModel

logger = logging.getLogger(__name__)


def chronological_split(
    df: pd.DataFrame,
    train_end: str = "2018-12-31",
    val_end: str = "2020-12-31"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split time series strictly chronologically into Train, Validation, and Test partitions.

    Parameters
    ----------
    df : pd.DataFrame
        Time-series DataFrame with DateTimeIndex.
    train_end : str
        End date of training period.
    val_end : str
        End date of validation period.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        train_df, val_df, test_df
    """
    train_df = df.loc[:train_end].copy()
    val_df = df.loc[train_end:val_end].iloc[1:].copy()
    test_df = df.loc[val_end:].iloc[1:].copy()

    logger.info(
        f"Chronological split completed:\n"
        f"  Train: {len(train_df)} rows ({train_df.index.min().date()} to {train_df.index.max().date()})\n"
        f"  Val:   {len(val_df)} rows ({val_df.index.min().date()} to {val_df.index.max().date()})\n"
        f"  Test:  {len(test_df)} rows ({test_df.index.min().date()} to {test_df.index.max().date()})"
    )
    return train_df, val_df, test_df


def run_walk_forward_evaluation(
    feature_df: pd.DataFrame,
    target_col: str,
    feature_cols: List[str],
    config: Dict[str, Any]
) -> pd.DataFrame:
    """
    Execute rigorous expanding/rolling walk-forward out-of-sample evaluation
    across all 5 core models:
      1. Historical Volatility (Baseline)
      2. GARCH(1,1)
      3. Random Forest
      4. XGBoost
      5. Hybrid GARCH + ML

    Parameters
    ----------
    feature_df : pd.DataFrame
        Complete feature matrix with target and return series.
    target_col : str
        Target column name (e.g. 'target_rv_5d').
    feature_cols : List[str]
        List of predictor feature names.
    config : Dict[str, Any]
        Configuration dictionary.

    Returns
    -------
    pd.DataFrame
        Standardized DataFrame containing: Date, Actual, Predicted, Model.
    """
    wf_cfg = config.get("walk_forward", {})
    initial_train_window = wf_cfg.get("initial_train_window", 1500)
    refit_interval = wf_cfg.get("refit_interval", 20)
    step_size = wf_cfg.get("step_size", 1)
    scheme = wf_cfg.get("scheme", "expanding")
    target_horizon = config.get("features", {}).get("target_horizon", 5)

    # Filter valid rows where target is present
    valid_data = feature_df.dropna(subset=[target_col] + feature_cols).copy()
    n_total = len(valid_data)

    if n_total <= initial_train_window + 50:
        raise ValueError(f"Insufficient observations ({n_total}) for initial train window ({initial_train_window}).")

    logger.info(
        f"Initiating walk-forward evaluation: Total={n_total} days, "
        f"Initial Train={initial_train_window} days, Scheme={scheme}, Refit Frequency={refit_interval} days."
    )

    # Model instances
    rf_params = config.get("ml_models", {}).get("random_forest", {})
    xgb_params = config.get("ml_models", {}).get("xgboost", {})
    hist_baseline = HistoricalVolatilityBaseline(window=20)
    
    # Store predictions
    records = []

    # Model states for periodic retraining
    rf_model = None
    xgb_model = None
    hybrid_model = None
    garch_model = None

    test_indices = list(range(initial_train_window, n_total, step_size))
    
    for idx_step, t in enumerate(test_indices):
        # Determine training window
        start_t = 0 if scheme == "expanding" else (t - initial_train_window)
        train_slice = valid_data.iloc[start_t:t]
        current_obs = valid_data.iloc[t:t+1]
        eval_date = valid_data.index[t]
        actual_val = float(current_obs[target_col].values[0])

        X_train = train_slice[feature_cols]
        y_train = train_slice[target_col]
        ret_train = train_slice["log_return"]
        X_test = current_obs[feature_cols]
        all_ret_up_to_t = valid_data.iloc[:t+1]["log_return"]

        # 1. Historical Volatility Baseline
        hist_pred = float(hist_baseline.predict(valid_data.iloc[:t+1]).iloc[-1])
        records.append({
            "Date": eval_date, "Actual": actual_val, "Predicted": hist_pred, "Model": "Historical Volatility"
        })

        # 2. Refit periodic models (GARCH, RF, XGB, Hybrid)
        if idx_step % refit_interval == 0 or rf_model is None:
            # GARCH(1,1)
            garch_model = GARCHModel(
                p=config.get("garch", {}).get("p", 1),
                q=config.get("garch", {}).get("q", 1),
                dist=config.get("garch", {}).get("dist", "StudentsT")
            )
            garch_model.fit(ret_train)

            # Random Forest
            rf_model = RandomForestVolatilityModel(**rf_params)
            rf_model.fit(X_train, y_train)

            # XGBoost
            xgb_model = XGBoostVolatilityModel(**xgb_params)
            xgb_model.fit(X_train, y_train)

            # Hybrid GARCH + ML
            hybrid_model = HybridGARCHMLModel(
                base_learner="xgboost",
                ml_params=xgb_params,
                garch_p=config.get("garch", {}).get("p", 1),
                garch_q=config.get("garch", {}).get("q", 1),
                garch_dist=config.get("garch", {}).get("dist", "StudentsT")
            )
            hybrid_model.fit(X_train, y_train, ret_train)

        # 3. Generate Predictions for current timestamp t
        # GARCH forecast
        garch_pred = float(garch_model.forecast_horizon_volatility(horizon=target_horizon))
        records.append({
            "Date": eval_date, "Actual": actual_val, "Predicted": garch_pred, "Model": "GARCH"
        })

        # Random Forest forecast
        rf_pred = float(rf_model.predict(X_test)[0])
        records.append({
            "Date": eval_date, "Actual": actual_val, "Predicted": rf_pred, "Model": "Random Forest"
        })

        # XGBoost forecast
        xgb_pred = float(xgb_model.predict(X_test)[0])
        records.append({
            "Date": eval_date, "Actual": actual_val, "Predicted": xgb_pred, "Model": "XGBoost"
        })

        # Hybrid forecast
        hybrid_pred = float(hybrid_model.predict(X_test, all_ret_up_to_t)[0])
        records.append({
            "Date": eval_date, "Actual": actual_val, "Predicted": hybrid_pred, "Model": "Hybrid GARCH+ML"
        })

    results_df = pd.DataFrame(records)
    logger.info(f"Walk-forward evaluation complete: Generated {len(results_df)} total forecast records.")
    return results_df
