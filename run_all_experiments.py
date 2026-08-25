"""
Master Reproducible Experiment Pipeline
=======================================
Executes the end-to-end quantitative volatility research study:
  Data Acquisition -> Preprocessing -> Feature Engineering -> EDA ->
  Walk-Forward Forecasting -> Statistical Evaluation (DM Test) ->
  Economic Volatility-Targeting Backtesting -> Robustness Analysis ->
  Publication Figures and Tables Export.
"""

import os
import sys
import logging
import json
import matplotlib
matplotlib.use("Agg")
import pandas as pd
import numpy as np
import yaml

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath("."))

from src.data_loader import load_config, fetch_market_data
from src.preprocessing import clean_market_data
from src.features import build_feature_dataset, verify_temporal_integrity
from src.volatility import compute_returns, compute_rolling_volatility, HistoricalVolatilityBaseline
from src.garch import GARCHModel
from src.ml_models import RandomForestVolatilityModel, XGBoostVolatilityModel
from src.hybrid import HybridGARCHMLModel
from src.validation import chronological_split, run_walk_forward_evaluation
from src.evaluation import (
    compute_forecast_metrics_table,
    run_pairwise_dm_tests,
    compute_mae,
    compute_rmse,
    compute_qlike
)
from src.backtesting import compare_all_strategies
from src.visualization import (
    plot_price_and_returns,
    plot_return_distribution,
    plot_volatility_clustering_acf,
    plot_rolling_volatilities,
    plot_forecast_comparison,
    plot_cumulative_equity_and_drawdowns,
    plot_feature_importances,
    plot_gross_vs_net_sharpe
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 80)
    logger.info("STARTING QUANTITATIVE VOLATILITY FORECASTING EXPERIMENT PIPELINE")
    logger.info("=" * 80)

    # 1. Load Configuration
    config = load_config("config/config.yaml")
    seed = config["project"]["random_seed"]
    np.random.seed(seed)

    # Prepare Directories
    for d in [
        config["data"]["raw_dir"],
        config["data"]["processed_dir"],
        config["results"]["forecasts_dir"],
        config["results"]["tables_dir"],
        config["results"]["figures_dir"]
    ]:
        os.makedirs(d, exist_ok=True)

    # 2. Data Acquisition
    logger.info("Step 1: Data Acquisition")
    raw_df = fetch_market_data(
        ticker=config["data"]["ticker"],
        start_date=config["data"]["start_date"],
        end_date=config["data"]["end_date"],
        save_path=config["data"]["raw_file"],
        force_reload=False
    )

    # 3. Data Cleaning and Preprocessing
    logger.info("Step 2: Data Preprocessing & Validation")
    clean_df, audit = clean_market_data(raw_df, save_path=config["data"]["processed_file"])
    with open("results/tables/data_audit_summary.json", "w") as f:
        json.dump(audit, f, indent=2)

    # 4. Feature Engineering & Target Construction
    logger.info("Step 3: Feature Engineering & Target Construction")
    target_k = config["features"]["target_horizon"]
    feature_df = build_feature_dataset(
        clean_df,
        target_horizon=target_k,
        vol_windows=config["features"]["volatility_windows"],
        return_lags=config["features"]["return_lags"],
        annualization_factor=config["features"]["annualization_factor"]
    )
    target_col = f"target_rv_{target_k}d"
    verify_temporal_integrity(feature_df, target_col=target_col)

    # 5. Exploratory Data Analysis & Diagnostic Plots
    logger.info("Step 4: Generating EDA Plots and Descriptive Statistics")
    plot_price_and_returns(feature_df, save_path="results/figures/01_price_and_returns.png")
    plot_return_distribution(feature_df["log_return"], save_path="results/figures/02_return_distribution.png")
    plot_volatility_clustering_acf(feature_df["log_return"], lags=40, save_path="results/figures/03_volatility_clustering_acf.png")
    plot_rolling_volatilities(feature_df, windows=[5, 20, 60], save_path="results/figures/04_rolling_volatilities.png")

    # Descriptive Statistics Table
    ret_series = feature_df["log_return"].dropna() * 100.0
    desc_stats = pd.DataFrame([{
        "Index": config["data"]["name"],
        "Start Date": str(feature_df.index.min().date()),
        "End Date": str(feature_df.index.max().date()),
        "Observations": len(ret_series),
        "Mean Return (%)": ret_series.mean(),
        "Daily Std (%)": ret_series.std(ddof=1),
        "Annualized Vol (%)": ret_series.std(ddof=1) * np.sqrt(252),
        "Skewness": ret_series.skew(),
        "Excess Kurtosis": ret_series.kurtosis(),
        "Min Return (%)": ret_series.min(),
        "Max Return (%)": ret_series.max()
    }])
    desc_stats.to_csv("results/tables/descriptive_statistics.csv", index=False)

    # 6. Chronological Split & Baseline In-Sample Inspection
    train_end = config["splits"]["train_end"]
    val_end = config["splits"]["val_end"]
    train_df, val_df, test_df = chronological_split(feature_df, train_end=train_end, val_end=val_end)

    feature_cols = [
        c for c in feature_df.columns
        if c not in [target_col, "Close", "log_return", "simple_return"] and not c.startswith("target_")
    ]

    # In-sample GARCH estimation for parameter diagnostics
    garch_init = GARCHModel(p=1, q=1, dist="StudentsT").fit(train_df["log_return"])
    with open("results/tables/garch_insample_parameters.json", "w") as f:
        json.dump(garch_init.params_summary, f, indent=2)
    logger.info(f"GARCH(1,1) In-sample fitted parameters: {garch_init.params_summary}")

    # Initial Feature Importances from Train set
    rf_init = RandomForestVolatilityModel(**config["ml_models"]["random_forest"]).fit(train_df[feature_cols], train_df[target_col])
    xgb_init = XGBoostVolatilityModel(**config["ml_models"]["xgboost"]).fit(train_df[feature_cols], train_df[target_col])
    hyb_init = HybridGARCHMLModel(
        base_learner="xgboost",
        ml_params=config["ml_models"]["xgboost"]
    ).fit(train_df[feature_cols], train_df[target_col], train_df["log_return"])

    importances_dict = {
        "Random Forest": rf_init.get_feature_importances(),
        "XGBoost": xgb_init.get_feature_importances(),
        "Hybrid GARCH+ML": hyb_init.get_feature_importances()
    }
    plot_feature_importances(importances_dict, save_path="results/figures/05_feature_importances.png")
    
    # Save feature importances table
    feat_imp_df = pd.DataFrame(importances_dict).fillna(0.0)
    feat_imp_df.to_csv("results/tables/feature_importances.csv")

    # 7. Out-of-Sample Walk-Forward Evaluation
    logger.info("Step 5: Running Walk-Forward Out-of-Sample Forecasting Engine")
    forecasts_df = run_walk_forward_evaluation(
        feature_df=feature_df,
        target_col=target_col,
        feature_cols=feature_cols,
        config=config
    )
    forecasts_df.to_csv("results/forecasts/walk_forward_forecasts.csv", index=False)

    # 8. Statistical Forecast Evaluation
    logger.info("Step 6: Statistical Forecast Evaluation (MAE, RMSE, QLIKE)")
    metrics_table = compute_forecast_metrics_table(forecasts_df)
    metrics_table.to_csv("results/tables/forecast_metrics.csv", index=False)
    logger.info(f"Forecast Accuracy Metrics:\n{metrics_table.to_string(index=False)}")

    # 9. Diebold-Mariano Tests
    logger.info("Step 7: Executing Pairwise Diebold-Mariano Tests")
    dm_vs_garch = run_pairwise_dm_tests(forecasts_df, horizon=target_k, benchmark_model="GARCH")
    dm_vs_hist = run_pairwise_dm_tests(forecasts_df, horizon=target_k, benchmark_model="Historical Volatility")
    dm_all = pd.concat([dm_vs_garch, dm_vs_hist], ignore_index=True)
    dm_all.to_csv("results/tables/dm_test_results.csv", index=False)
    logger.info(f"Diebold-Mariano Test Results:\n{dm_all.to_string(index=False)}")

    # 10. Economic Risk Management & Portfolio Backtesting
    logger.info("Step 8: Volatility-Targeting Strategy Backtesting & Transaction Cost Attribution")
    strat_summary, daily_bt_results = compare_all_strategies(
        forecasts_df=forecasts_df,
        prices=clean_df["Close"],
        config=config
    )
    strat_summary.to_csv("results/tables/portfolio_metrics.csv", index=False)
    logger.info(f"Strategy Performance Summary:\n{strat_summary.to_string(index=False)}")

    # 11. Visualizations: Forecasts, Equity Curves, Drawdowns, Cost Impact
    logger.info("Step 9: Generating Publication-Grade Result Figures")
    plot_forecast_comparison(forecasts_df, save_path="results/figures/06_forecast_comparison.png")
    plot_cumulative_equity_and_drawdowns(daily_bt_results, save_path="results/figures/07_cumulative_wealth_and_drawdowns.png")
    plot_gross_vs_net_sharpe(strat_summary, save_path="results/figures/08_gross_vs_net_sharpe.png")

    # 12. Robustness Analysis Across Horizons & Regimes
    logger.info("Step 10: Robustness Analysis across Forecast Horizons and Market Regimes")
    robustness_records = []
    for k_robust in [1, 5, 10, 20]:
        t_col = f"target_rv_{k_robust}d"
        feat_rob = build_feature_dataset(
            clean_df,
            target_horizon=k_robust,
            vol_windows=config["features"]["volatility_windows"],
            return_lags=config["features"]["return_lags"],
            annualization_factor=config["features"]["annualization_factor"]
        )
        f_cols = [c for c in feat_rob.columns if c not in [t_col, "Close", "log_return", "simple_return"] and not c.startswith("target_")]
        
        # Fast walk-forward run with step_size=5 for multi-horizon grid
        cfg_fast = config.copy()
        cfg_fast["walk_forward"]["step_size"] = 5
        cfg_fast["features"]["target_horizon"] = k_robust
        
        wf_rob = run_walk_forward_evaluation(feat_rob, target_col=t_col, feature_cols=f_cols, config=cfg_fast)
        m_table = compute_forecast_metrics_table(wf_rob)
        m_table["Horizon_Days"] = k_robust
        robustness_records.append(m_table)

    robust_df = pd.concat(robustness_records, ignore_index=True)
    robust_df.to_csv("results/tables/robustness_horizons_metrics.csv", index=False)

    # Regime Analysis (High Volatility vs Low Volatility Regimes)
    median_vol = forecasts_df["Actual"].median()
    forecasts_df["Regime"] = np.where(forecasts_df["Actual"] > median_vol, "High Volatility", "Low Volatility")

    regime_metrics = []
    for reg, grp in forecasts_df.groupby("Regime"):
        t_reg = compute_forecast_metrics_table(grp)
        t_reg["Regime"] = reg
        regime_metrics.append(t_reg)

    regime_df = pd.concat(regime_metrics, ignore_index=True)
    regime_df.to_csv("results/tables/regime_forecast_metrics.csv", index=False)

    logger.info("=" * 80)
    logger.info("MASTER EXPERIMENT PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("All forecasts, tables, metrics, and publication figures persisted to 'results/'")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
