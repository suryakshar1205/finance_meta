"""
Refinement Experiments and Final Package Generation
===================================================
Executes advanced econometric diagnostics, Mincer-Zarnowitz calibration,
full Diebold-Mariano matrices, transaction cost sensitivity grids, rebalance
frequency analysis, and exports the frozen results/final/ repository.
"""

import os
import sys
import json
import shutil
import logging
import pandas as pd
import numpy as np
import yaml

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath("."))

from src.data_loader import load_config
from src.garch import GARCHModel
from src.evaluation import (
    compute_forecast_metrics_table,
    generate_dm_comparison_matrix,
    compute_mincer_zarnowitz_calibration,
    run_pairwise_dm_tests
)
from src.backtesting import (
    compare_all_strategies,
    run_transaction_cost_sensitivity,
    run_rebalance_frequency_analysis,
    run_turnover_controlled_ml_backtest
)
from src.visualization import (
    plot_dm_heatmap,
    plot_transaction_cost_sensitivity,
    plot_turnover_vs_net_sharpe
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 80)
    logger.info("STARTING QUANTITATIVE REFINEMENT & SENSITIVITY EXPERIMENT SUITE")
    logger.info("=" * 80)

    # 1. Load Frozen Configuration
    config = load_config("config/final_experiment.yaml")
    os.makedirs("results/final", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)
    os.makedirs("results/tables", exist_ok=True)

    # 2. Load Processed Data and Baseline Walk-Forward Forecasts
    processed_df = pd.read_csv("data/processed/nifty50_daily_processed.csv", parse_dates=["Date"], index_col="Date")
    from src.volatility import compute_returns
    processed_df = compute_returns(processed_df)

    forecasts_df = pd.read_csv("results/forecasts/walk_forward_forecasts.csv", parse_dates=["Date"])
    prices = processed_df["Close"]

    logger.info(f"Loaded {len(forecasts_df)} walk-forward forecast records across models: {forecasts_df['Model'].unique().tolist()}")

    # 3. GARCH Econometric Diagnostics (In-sample & Residual Tests)
    logger.info("Step 1: GARCH Residual Diagnostics (Ljung-Box & ARCH-LM Tests)")
    train_returns = processed_df.loc[:config["splits"]["train_end"]]["log_return"]
    garch_model = GARCHModel(p=1, q=1, dist="StudentsT").fit(train_returns)
    garch_diag = garch_model.run_residual_diagnostics(lags=10)
    
    garch_params_df = pd.DataFrame([{
        "Model": "GARCH(1,1) StudentsT",
        "Omega": garch_model.params_summary.get("omega"),
        "Alpha": garch_model.params_summary.get("alpha"),
        "Beta": garch_model.params_summary.get("beta"),
        "Persistence (Alpha+Beta)": garch_model.params_summary.get("persistence"),
        "Half_Life_Days": garch_model.params_summary.get("half_life_days"),
        "AIC": garch_model.params_summary.get("aic"),
        "BIC": garch_model.params_summary.get("bic"),
        "Ljung_Box_Resid_Q_Stat": garch_diag.get("ljung_box_resid_stat"),
        "Ljung_Box_Resid_Pval": garch_diag.get("ljung_box_resid_pval"),
        "Ljung_Box_SqResid_Q_Stat": garch_diag.get("ljung_box_sq_resid_stat"),
        "Ljung_Box_SqResid_Pval": garch_diag.get("ljung_box_sq_resid_pval"),
        "ARCH_LM_Stat": garch_diag.get("arch_lm_stat"),
        "ARCH_LM_Pval": garch_diag.get("arch_lm_pval")
    }])
    garch_params_df.to_csv("results/final/final_model_parameters.csv", index=False)
    logger.info(f"GARCH Diagnostics: Ljung-Box Q p-val = {garch_diag.get('ljung_box_sq_resid_pval'):.4f}, ARCH-LM p-val = {garch_diag.get('arch_lm_pval'):.4f}")

    # 4. Forecast Accuracy Metrics
    logger.info("Step 2: Computing Final Out-of-Sample Accuracy Metrics")
    metrics_table = compute_forecast_metrics_table(forecasts_df)
    metrics_table.to_csv("results/final/final_forecast_metrics.csv", index=False)
    metrics_table.to_csv("results/tables/forecast_metrics.csv", index=False)

    # 5. Full Pairwise Diebold-Mariano Matrix & Heatmap
    logger.info("Step 3: Generating Full Diebold-Mariano Matrix & Heatmap")
    dm_stat_mat, dm_pval_mat = generate_dm_comparison_matrix(forecasts_df, horizon=config["features"]["target_horizon"])
    dm_stat_mat.to_csv("results/final/final_dm_comparison_matrix.csv")
    dm_stat_mat.to_csv("results/tables/dm_comparison_matrix.csv")
    plot_dm_heatmap(dm_stat_mat, dm_pval_mat, save_path="results/figures/09_dm_comparison_heatmap.png")

    dm_vs_garch = run_pairwise_dm_tests(forecasts_df, horizon=config["features"]["target_horizon"], benchmark_model="GARCH")
    dm_vs_garch.to_csv("results/final/final_dm_tests.csv", index=False)

    # 6. Mincer-Zarnowitz Forecast Calibration & Bias Analysis
    logger.info("Step 4: Performing Mincer-Zarnowitz Calibration & Forecast Bias Analysis")
    calibration_df = compute_mincer_zarnowitz_calibration(forecasts_df)
    calibration_df.to_csv("results/final/final_calibration_analysis.csv", index=False)
    calibration_df.to_csv("results/tables/calibration_analysis.csv", index=False)
    logger.info(f"Calibration Analysis:\n{calibration_df[['Model', 'Mean_Bias', 'MZ_Alpha', 'MZ_Beta', 'MZ_R_Squared']].to_string(index=False)}")

    # 7. Economic Evaluation: Volatility-Targeting Strategy
    logger.info("Step 5: Dynamic Volatility Targeting Strategy Backtest")
    strat_summary, daily_results = compare_all_strategies(forecasts_df, prices, config)
    strat_summary.to_csv("results/final/final_portfolio_metrics.csv", index=False)
    strat_summary.to_csv("results/tables/portfolio_metrics.csv", index=False)

    # 8. Transaction Cost Sensitivity Analysis Grid
    logger.info("Step 6: Transaction Cost Sensitivity Grid (0 to 50 bps)")
    cost_grid = config["portfolio"]["cost_sensitivity_grid"]
    cost_df = run_transaction_cost_sensitivity(
        prices=prices,
        forecasts_df=forecasts_df,
        cost_grid=cost_grid,
        slippage_bps=config["portfolio"]["slippage_bps"],
        target_vol=config["portfolio"]["target_volatility"],
        rf=config["portfolio"]["risk_free_rate"]
    )
    cost_df.to_csv("results/final/final_transaction_cost_sensitivity.csv", index=False)
    cost_df.to_csv("results/tables/transaction_cost_sensitivity.csv", index=False)
    plot_transaction_cost_sensitivity(cost_df, save_path="results/figures/10_transaction_cost_sensitivity.png")

    # 9. Rebalancing Frequency and Turnover Analysis
    logger.info("Step 7: Rebalancing Frequency Analysis (Daily, Weekly, Biweekly)")
    freq_df = run_rebalance_frequency_analysis(
        prices=prices,
        forecasts_df=forecasts_df,
        frequencies=config["portfolio"]["rebalance_frequencies"],
        target_vol=config["portfolio"]["target_volatility"],
        cost_bps=config["portfolio"]["transaction_cost_bps"],
        slippage_bps=config["portfolio"]["slippage_bps"],
        rf=config["portfolio"]["risk_free_rate"]
    )
    freq_df.to_csv("results/final/final_turnover_analysis.csv", index=False)
    freq_df.to_csv("results/tables/turnover_analysis.csv", index=False)
    plot_turnover_vs_net_sharpe(strat_summary, save_path="results/figures/11_turnover_vs_net_sharpe.png")

    # 10. Exploratory Extension: Turnover-Dampened / EMA-Smoothed ML
    logger.info("Step 8: Exploratory Extension: Turnover-Controlled / EMA-Smoothed ML Strategy")
    smoothed_df = run_turnover_controlled_ml_backtest(
        prices=prices,
        forecasts_df=forecasts_df,
        smoothing_spans=config["portfolio"]["ema_smoothing_spans"],
        target_vol=config["portfolio"]["target_volatility"],
        cost_bps=config["portfolio"]["transaction_cost_bps"],
        slippage_bps=config["portfolio"]["slippage_bps"],
        rf=config["portfolio"]["risk_free_rate"]
    )
    smoothed_df.to_csv("results/final/final_smoothed_ml_extension.csv", index=False)
    smoothed_df.to_csv("results/tables/smoothed_ml_extension.csv", index=False)

    # 11. Volatility Regime Breakdown (Terciles: Low, Medium, High)
    logger.info("Step 9: Volatility Regime Breakdown (Terciles)")
    terciles = forecasts_df["Actual"].quantile([0.333, 0.667])
    q_low, q_high = terciles.iloc[0], terciles.iloc[1]

    def assign_regime(val):
        if val <= q_low:
            return "Low Volatility (T1)"
        elif val <= q_high:
            return "Medium Volatility (T2)"
        else:
            return "High Volatility (T3)"

    forecasts_df["Regime_Tercile"] = forecasts_df["Actual"].apply(assign_regime)

    regime_records = []
    for reg in ["Low Volatility (T1)", "Medium Volatility (T2)", "High Volatility (T3)"]:
        grp = forecasts_df[forecasts_df["Regime_Tercile"] == reg]
        t_reg = compute_forecast_metrics_table(grp)
        t_reg["Regime"] = reg
        t_reg["Regime_Range"] = f"[{grp['Actual'].min():.3f}, {grp['Actual'].max():.3f}]"
        regime_records.append(t_reg)

    regime_df = pd.concat(regime_records, ignore_index=True)
    regime_df.to_csv("results/final/final_regime_metrics.csv", index=False)
    regime_df.to_csv("results/tables/regime_metrics.csv", index=False)

    # 12. Freeze Final Configuration Copy
    shutil.copy("config/final_experiment.yaml", "results/final/final_configuration.yaml")

    logger.info("=" * 80)
    logger.info("REFINEMENT EXPERIMENT SUITE COMPLETED SUCCESSFULLY")
    logger.info("All final publication tables and figures persisted in results/final/ and results/figures/")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
