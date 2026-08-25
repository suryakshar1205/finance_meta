"""
Economic Risk Management and Portfolio Backtesting Module
=========================================================
Implements dynamic volatility targeting, leverage constraints, realistic transaction
costs & slippage deductions, and comprehensive institutional risk-adjusted performance attribution.
"""

import logging
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def run_volatility_targeting_backtest(
    prices: pd.Series,
    predicted_vol: pd.Series,
    target_vol: float = 0.15,
    max_leverage: float = 1.5,
    min_leverage: float = 0.0,
    risk_free_rate: float = 0.05,
    transaction_cost_bps: float = 10.0,
    slippage_bps: float = 5.0,
    annualization_factor: int = 252
) -> pd.DataFrame:
    """
    Backtest dynamic volatility targeting strategy for a given volatility forecasting model.

    Targeting Rule:
      w_t = clip( target_vol / predicted_vol_t, min_leverage, max_leverage )
    
    Execution:
      Portfolio Gross Return: R_gross,t = w_{t-1} * R_asset,t + (1 - w_{t-1}) * (r_f / 252)
      Turnover Cost:          Cost_t = |w_t - w_{t-1}| * (cost_bps + slippage_bps) / 10000
      Portfolio Net Return:   R_net,t = R_gross,t - Cost_t

    Parameters
    ----------
    prices : pd.Series
        Asset close price series.
    predicted_vol : pd.Series
        Predicted forward annualized volatility aligned with dates.
    target_vol : float
        Target annualized portfolio volatility (e.g. 0.15 for 15%).
    max_leverage : float
        Upper bound on asset weight.
    min_leverage : float
        Lower bound on asset weight.
    risk_free_rate : float
        Annual risk-free benchmark rate (e.g. 0.05).
    transaction_cost_bps : float
        Transaction fee in basis points (e.g. 10.0 = 0.10%).
    slippage_bps : float
        Slippage in basis points (e.g. 5.0 = 0.05%).
    annualization_factor : int
        Trading days per year.

    Returns
    -------
    pd.DataFrame
        Detailed daily backtest time series: weights, gross return, net return,
        drawdowns, turnover, cumulative wealth.
    """
    common_idx = prices.index.intersection(predicted_vol.index).sort_values()
    p = prices.loc[common_idx]
    v_pred = predicted_vol.loc[common_idx]

    daily_rf = risk_free_rate / annualization_factor
    total_cost_pct = (transaction_cost_bps + slippage_bps) / 10000.0

    asset_returns = p.pct_change().fillna(0.0)

    # 1. Compute target weights based on predicted volatility (known at t-1)
    raw_weights = target_vol / (v_pred.replace(0, np.nan).fillna(target_vol))
    target_weights = raw_weights.clip(lower=min_leverage, upper=max_leverage)

    # Shift weights by 1 day to ensure zero lookahead (position held from t-1 into t)
    weights = target_weights.shift(1).fillna(1.0)

    # 2. Daily returns and turnover
    weight_changes = weights.diff().abs().fillna(weights.iloc[0])
    cost_deductions = weight_changes * total_cost_pct

    # Gross return: invested asset return + uninvested cash interest
    gross_returns = weights * asset_returns + (1.0 - weights) * daily_rf
    net_returns = gross_returns - cost_deductions

    # 3. Cumulative equity curves (starting at 1.0)
    cum_gross = (1.0 + gross_returns).cumprod()
    cum_net = (1.0 + net_returns).cumprod()
    cum_benchmark = (1.0 + asset_returns).cumprod()

    # 4. Drawdowns
    running_max_gross = cum_gross.cummax()
    drawdown_gross = (cum_gross - running_max_gross) / running_max_gross

    running_max_net = cum_net.cummax()
    drawdown_net = (cum_net - running_max_net) / running_max_net

    df_out = pd.DataFrame({
        "Close": p,
        "Asset_Return": asset_returns,
        "Predicted_Vol": v_pred,
        "Weight": weights,
        "Turnover": weight_changes,
        "Cost_Deduction": cost_deductions,
        "Gross_Return": gross_returns,
        "Net_Return": net_returns,
        "Cum_Gross": cum_gross,
        "Cum_Net": cum_net,
        "Cum_Benchmark": cum_benchmark,
        "Drawdown_Gross": drawdown_gross,
        "Drawdown_Net": drawdown_net
    }, index=common_idx)

    return df_out


def compute_performance_metrics(
    daily_returns: pd.Series,
    drawdown_series: pd.Series,
    turnover_series: Optional[pd.Series] = None,
    risk_free_rate: float = 0.05,
    annualization_factor: int = 252
) -> Dict[str, float]:
    """
    Calculate comprehensive institutional risk-adjusted performance metrics.

    Parameters
    ----------
    daily_returns : pd.Series
        Series of daily strategy net or gross returns.
    drawdown_series : pd.Series
        Drawdown series.
    turnover_series : Optional[pd.Series]
        Daily turnover series.
    risk_free_rate : float
        Annual risk-free rate.
    annualization_factor : int
        Trading days per year.

    Returns
    -------
    Dict[str, float]
        Dictionary of performance indicators.
    """
    clean_ret = daily_returns.dropna()
    N = len(clean_ret)
    if N < 20:
        return {}

    daily_rf = risk_free_rate / annualization_factor
    cum_return = (1.0 + clean_ret).prod() - 1.0
    cagr = (1.0 + cum_return) ** (annualization_factor / N) - 1.0
    ann_vol = clean_ret.std(ddof=1) * np.sqrt(annualization_factor)

    excess_returns = clean_ret - daily_rf
    sharpe_ratio = (excess_returns.mean() / (clean_ret.std(ddof=1) + 1e-8)) * np.sqrt(annualization_factor)

    # Sortino Ratio (Downside deviation)
    downside_returns = clean_ret[clean_ret < 0]
    downside_std = downside_returns.std(ddof=1) * np.sqrt(annualization_factor) if len(downside_returns) > 2 else 1e-4
    sortino_ratio = (cagr - risk_free_rate) / (downside_std + 1e-8)

    max_drawdown = float(drawdown_series.min())
    calmar_ratio = cagr / abs(max_drawdown) if abs(max_drawdown) > 1e-6 else np.nan

    ann_turnover = float(turnover_series.mean() * annualization_factor) if turnover_series is not None else 0.0

    return {
        "Total_Return": float(cum_return),
        "CAGR": float(cagr),
        "Annualized_Vol": float(ann_vol),
        "Sharpe_Ratio": float(sharpe_ratio),
        "Sortino_Ratio": float(sortino_ratio),
        "Max_Drawdown": float(max_drawdown),
        "Calmar_Ratio": float(calmar_ratio),
        "Annual_Turnover": float(ann_turnover)
    }


def compare_all_strategies(
    forecasts_df: pd.DataFrame,
    prices: pd.Series,
    config: Dict[str, Any]
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Run backtest comparison across all competing models + Passive Buy-and-Hold benchmark.

    Parameters
    ----------
    forecasts_df : pd.DataFrame
        Walk-forward forecasts DataFrame.
    prices : pd.Series
        Asset close price series.
    config : Dict[str, Any]
        Configuration parameters.

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]
        Summary performance metrics table, dictionary of daily backtest DataFrames per model.
    """
    p_cfg = config.get("portfolio", {})
    target_vol = p_cfg.get("target_volatility", 0.15)
    max_leverage = p_cfg.get("max_leverage", 1.5)
    min_leverage = p_cfg.get("min_leverage", 0.0)
    rf = p_cfg.get("risk_free_rate", 0.05)
    cost_bps = p_cfg.get("transaction_cost_bps", 10.0)
    slip_bps = p_cfg.get("slippage_bps", 5.0)

    models = forecasts_df["Model"].unique()
    daily_results = {}
    metrics_list = []

    # 1. Evaluate each Volatility Targeting Strategy
    for m in models:
        sub = forecasts_df[forecasts_df["Model"] == m].set_index("Date")["Predicted"].dropna()
        bt_df = run_volatility_targeting_backtest(
            prices=prices,
            predicted_vol=sub,
            target_vol=target_vol,
            max_leverage=max_leverage,
            min_leverage=min_leverage,
            risk_free_rate=rf,
            transaction_cost_bps=cost_bps,
            slippage_bps=slip_bps
        )
        daily_results[m] = bt_df

        m_gross = compute_performance_metrics(bt_df["Gross_Return"], bt_df["Drawdown_Gross"], bt_df["Turnover"], rf)
        m_net = compute_performance_metrics(bt_df["Net_Return"], bt_df["Drawdown_Net"], bt_df["Turnover"], rf)

        metrics_list.append({
            "Strategy": f"VolTarget ({m})",
            "CAGR_Gross": m_gross.get("CAGR", np.nan),
            "CAGR_Net": m_net.get("CAGR", np.nan),
            "Vol_Net": m_net.get("Annualized_Vol", np.nan),
            "Sharpe_Gross": m_gross.get("Sharpe_Ratio", np.nan),
            "Sharpe_Net": m_net.get("Sharpe_Ratio", np.nan),
            "Sortino_Net": m_net.get("Sortino_Ratio", np.nan),
            "Max_Drawdown_Net": m_net.get("Max_Drawdown", np.nan),
            "Calmar_Net": m_net.get("Calmar_Ratio", np.nan),
            "Annual_Turnover": m_net.get("Annual_Turnover", np.nan)
        })

    # 2. Add Passive Buy-and-Hold Benchmark (100% allocation on identical test window)
    if daily_results:
        ref_df = list(daily_results.values())[0]
        bh_ret = ref_df["Asset_Return"]
        bh_dd = (ref_df["Cum_Benchmark"] - ref_df["Cum_Benchmark"].cummax()) / ref_df["Cum_Benchmark"].cummax()
        bh_metrics = compute_performance_metrics(bh_ret, bh_dd, pd.Series(0.0, index=bh_ret.index), rf)

        metrics_list.append({
            "Strategy": "Buy & Hold (Passive 100%)",
            "CAGR_Gross": bh_metrics.get("CAGR", np.nan),
            "CAGR_Net": bh_metrics.get("CAGR", np.nan),
            "Vol_Net": bh_metrics.get("Annualized_Vol", np.nan),
            "Sharpe_Gross": bh_metrics.get("Sharpe_Ratio", np.nan),
            "Sharpe_Net": bh_metrics.get("Sharpe_Ratio", np.nan),
            "Sortino_Net": bh_metrics.get("Sortino_Ratio", np.nan),
            "Max_Drawdown_Net": bh_metrics.get("Max_Drawdown", np.nan),
            "Calmar_Net": bh_metrics.get("Calmar_Ratio", np.nan),
            "Annual_Turnover": 0.0
        })

    summary_df = pd.DataFrame(metrics_list)
    return summary_df, daily_results


def run_transaction_cost_sensitivity(
    prices: pd.Series,
    forecasts_df: pd.DataFrame,
    cost_grid: List[float] = [0.0, 5.0, 10.0, 15.0, 20.0, 30.0, 50.0],
    slippage_bps: float = 5.0,
    target_vol: float = 0.15,
    rf: float = 0.05
) -> pd.DataFrame:
    """
    Evaluate strategy Net Sharpe and Net CAGR across a spectrum of transaction cost scenarios.
    """
    models = forecasts_df["Model"].unique()
    records = []

    for cost_bps in cost_grid:
        for m in models:
            sub = forecasts_df[forecasts_df["Model"] == m].set_index("Date")["Predicted"].dropna()
            bt_df = run_volatility_targeting_backtest(
                prices=prices,
                predicted_vol=sub,
                target_vol=target_vol,
                risk_free_rate=rf,
                transaction_cost_bps=cost_bps,
                slippage_bps=slippage_bps
            )
            m_net = compute_performance_metrics(bt_df["Net_Return"], bt_df["Drawdown_Net"], bt_df["Turnover"], rf)
            
            records.append({
                "Cost_bps": cost_bps,
                "Total_Friction_bps": cost_bps + slippage_bps,
                "Model": m,
                "Net_Sharpe": m_net.get("Sharpe_Ratio", np.nan),
                "Net_CAGR": m_net.get("CAGR", np.nan),
                "Annual_Turnover": m_net.get("Annual_Turnover", np.nan),
                "Max_Drawdown": m_net.get("Max_Drawdown", np.nan)
            })

    return pd.DataFrame(records)


def run_rebalance_frequency_analysis(
    prices: pd.Series,
    forecasts_df: pd.DataFrame,
    frequencies: List[str] = ["Daily", "Weekly (5d)", "Biweekly (10d)"],
    target_vol: float = 0.15,
    cost_bps: float = 10.0,
    slippage_bps: float = 5.0,
    rf: float = 0.05
) -> pd.DataFrame:
    """
    Evaluate performance of volatility targeting under different rebalancing cadences.
    """
    models = forecasts_df["Model"].unique()
    records = []

    freq_step_map = {
        "Daily": 1,
        "Weekly (5d)": 5,
        "Biweekly (10d)": 10
    }

    for freq_name in frequencies:
        step = freq_step_map.get(freq_name, 1)
        for m in models:
            sub = forecasts_df[forecasts_df["Model"] == m].set_index("Date")["Predicted"].dropna()
            
            # Subsample predicted vol at rebalance dates and forward-fill weights
            common_idx = prices.index.intersection(sub.index).sort_values()
            p = prices.loc[common_idx]
            v = sub.loc[common_idx]

            raw_w = (target_vol / v.replace(0, np.nan).fillna(target_vol)).clip(0.0, 1.5)
            
            # Hold constant between rebalance steps
            rebal_mask = np.zeros(len(raw_w), dtype=bool)
            rebal_mask[::step] = True
            w_rebal = raw_w.copy()
            w_rebal[~rebal_mask] = np.nan
            w_periodic = w_rebal.ffill().shift(1).fillna(1.0)

            # Compute backtest with periodic weights
            ret = p.pct_change().fillna(0.0)
            daily_rf = rf / 252.0
            total_cost_pct = (cost_bps + slippage_bps) / 10000.0

            turnover = w_periodic.diff().abs().fillna(w_periodic.iloc[0])
            cost_deduction = turnover * total_cost_pct
            net_ret = w_periodic * ret + (1.0 - w_periodic) * daily_rf - cost_deduction

            cum_net = (1.0 + net_ret).cumprod()
            dd_net = (cum_net - cum_net.cummax()) / cum_net.cummax()

            m_net = compute_performance_metrics(net_ret, dd_net, turnover, rf)

            records.append({
                "Rebalance_Frequency": freq_name,
                "Model": m,
                "Net_CAGR": m_net.get("CAGR", np.nan),
                "Net_Sharpe": m_net.get("Sharpe_Ratio", np.nan),
                "Annual_Turnover": m_net.get("Annual_Turnover", np.nan),
                "Max_Drawdown": m_net.get("Max_Drawdown", np.nan)
            })

    return pd.DataFrame(records)


def run_turnover_controlled_ml_backtest(
    prices: pd.Series,
    forecasts_df: pd.DataFrame,
    smoothing_spans: List[int] = [1, 3, 5, 10],
    target_vol: float = 0.15,
    cost_bps: float = 10.0,
    slippage_bps: float = 5.0,
    rf: float = 0.05
) -> pd.DataFrame:
    """
    Exploratory research extension: Test if exponentially smoothing ML volatility forecasts
    dampens excessive turnover and restores net economic alpha.
    """
    ml_models = [m for m in forecasts_df["Model"].unique() if m in ["XGBoost", "Random Forest", "Hybrid GARCH+ML"]]
    records = []

    for span in smoothing_spans:
        for m in ml_models:
            sub = forecasts_df[forecasts_df["Model"] == m].set_index("Date")["Predicted"].dropna()
            
            # Apply EMA smoothing to raw volatility forecast
            smoothed_v = sub.ewm(span=span, adjust=False).mean() if span > 1 else sub

            bt_df = run_volatility_targeting_backtest(
                prices=prices,
                predicted_vol=smoothed_v,
                target_vol=target_vol,
                risk_free_rate=rf,
                transaction_cost_bps=cost_bps,
                slippage_bps=slippage_bps
            )
            m_net = compute_performance_metrics(bt_df["Net_Return"], bt_df["Drawdown_Net"], bt_df["Turnover"], rf)

            records.append({
                "Model": m,
                "EMA_Smoothing_Span": span,
                "Net_CAGR": m_net.get("CAGR", np.nan),
                "Net_Sharpe": m_net.get("Sharpe_Ratio", np.nan),
                "Annual_Turnover": m_net.get("Annual_Turnover", np.nan),
                "Max_Drawdown": m_net.get("Max_Drawdown", np.nan)
            })

    return pd.DataFrame(records)

