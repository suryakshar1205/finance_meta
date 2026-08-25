"""
Publication-Grade Financial Visualization Module
================================================
Generates standardized, aesthetically refined charts and empirical diagnostic plots
adhering to academic quantitative finance standards.
"""

import os
import logging
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.graphics.tsaplots import plot_acf
import scipy.stats as stats

logger = logging.getLogger(__name__)

# Standard Style Configuration
plt.rcParams.update({
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 13,
    "figure.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--"
})

PALETTE = {
    "Historical Volatility": "#7f7f7f",
    "GARCH": "#1f77b4",
    "Random Forest": "#2ca02c",
    "XGBoost": "#ff7f0e",
    "Hybrid GARCH+ML": "#9467bd",
    "Buy & Hold": "#333333",
    "Actual": "#111111"
}


def plot_price_and_returns(df: pd.DataFrame, save_path: Optional[str] = None) -> plt.Figure:
    """Plot asset closing price level and daily log return series."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    
    ax1.plot(df.index, df["Close"], color="#1f77b4", lw=1.5, label="NIFTY 50 Close Index")
    ax1.set_title("NIFTY 50 Index: Historical Price Dynamics (2010 - 2024)")
    ax1.set_ylabel("Index Level (INR)")
    ax1.legend(loc="upper left")

    ax2.plot(df.index, df["log_return"] * 100, color="#d62728", lw=0.6, alpha=0.8, label="Daily Log Returns (%)")
    ax2.axhline(0, color="black", lw=0.8, linestyle="--")
    ax2.set_title("Daily Log Returns")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Return (%)")
    ax2.legend(loc="upper left")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_return_distribution(returns: pd.Series, save_path: Optional[str] = None) -> plt.Figure:
    """Plot return histogram, fitted normal distribution, and descriptive statistics summary."""
    clean_ret = returns.dropna() * 100.0  # percentage
    mu, std = clean_ret.mean(), clean_ret.std(ddof=1)
    skew = clean_ret.skew()
    kurt = clean_ret.kurtosis()  # excess kurtosis

    fig, ax = plt.subplots(figsize=(9, 5.5))
    n, bins, _ = ax.hist(clean_ret, bins=80, density=True, alpha=0.6, color="#4682b4", edgecolor="white", label="Empirical Density")
    
    # Gaussian overlay
    x = np.linspace(clean_ret.min(), clean_ret.max(), 500)
    p = stats.norm.pdf(x, mu, std)
    ax.plot(x, p, "r--", lw=2, label=f"Normal Fit ($\mu$={mu:.2f}%, $\sigma$={std:.2f}%)")

    stats_text = (
        f"Mean: {mu:.3f}%\n"
        f"Std Dev: {std:.3f}%\n"
        f"Skewness: {skew:.3f}\n"
        f"Excess Kurtosis: {kurt:.3f}\n"
        f"Observations: {len(clean_ret):,}"
    )
    ax.text(
        0.03, 0.95, stats_text, transform=ax.transAxes, verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="gray", alpha=0.9)
    )

    ax.set_title("NIFTY 50 Daily Return Distribution (Fat Tails & Leptokurtosis)")
    ax.set_xlabel("Daily Log Return (%)")
    ax.set_ylabel("Probability Density")
    ax.legend(loc="upper right")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_volatility_clustering_acf(returns: pd.Series, lags: int = 40, save_path: Optional[str] = None) -> plt.Figure:
    """Plot Autocorrelation Functions for Returns, Squared Returns, and Absolute Returns."""
    clean_ret = returns.dropna()
    sq_ret = clean_ret ** 2
    abs_ret = np.abs(clean_ret)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    plot_acf(clean_ret, lags=lags, ax=ax1, alpha=0.05, title="ACF: Raw Returns (No Linear Memory)")
    ax1.set_ylabel("Autocorrelation")

    plot_acf(sq_ret, lags=lags, ax=ax2, alpha=0.05, title="ACF: Squared Returns (Strong Volatility Persistence)")
    ax2.set_ylabel("Autocorrelation")

    plot_acf(abs_ret, lags=lags, ax=ax3, alpha=0.05, title="ACF: Absolute Returns (Long-Memory ARCH Effect)")
    ax3.set_ylabel("Autocorrelation")
    ax3.set_xlabel("Lag (Trading Days)")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_rolling_volatilities(df: pd.DataFrame, windows: List[int] = [5, 20, 60], save_path: Optional[str] = None) -> plt.Figure:
    """Plot multi-horizon rolling annualized historical volatility series."""
    fig, ax = plt.subplots(figsize=(11, 5.5))
    colors = ["#2ca02c", "#1f77b4", "#9467bd"]

    for w, c in zip(windows, colors):
        col_name = f"vol_ann_{w}d"
        if col_name in df.columns:
            ax.plot(df.index, df[col_name] * 100, label=f"{w}-Day Rolling Volatility", color=c, lw=1.2)

    ax.set_title("NIFTY 50 Multi-Horizon Rolling Annualized Volatility")
    ax.set_xlabel("Date")
    ax.set_ylabel("Annualized Volatility (%)")
    ax.legend(loc="upper right")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_forecast_comparison(forecasts_df: pd.DataFrame, save_path: Optional[str] = None) -> plt.Figure:
    """Plot actual realized volatility alongside out-of-sample model predictions."""
    pivot_df = forecasts_df.pivot(index="Date", columns="Model", values="Predicted")
    actual_series = forecasts_df.groupby("Date")["Actual"].first()

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(actual_series.index, actual_series.values * 100, color="black", lw=1.8, label="Actual Realized Volatility", zorder=5)

    for m in pivot_df.columns:
        ax.plot(
            pivot_df.index, pivot_df[m] * 100,
            label=m,
            color=PALETTE.get(m, None),
            lw=1.1,
            alpha=0.85
        )

    ax.set_title("Out-of-Sample Volatility Forecasts vs. Actual Realized Target (NIFTY 50)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Annualized Volatility (%)")
    ax.legend(loc="upper right", framealpha=0.95)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_cumulative_equity_and_drawdowns(
    daily_results: Dict[str, pd.DataFrame],
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot cumulative strategy wealth (net of fees) and corresponding drawdown trajectories."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})

    # Add Buy & Hold benchmark
    first_df = list(daily_results.values())[0]
    ax1.plot(first_df.index, first_df["Cum_Benchmark"], color="#333333", linestyle="--", lw=1.5, label="Buy & Hold (100%)")
    ax2.plot(first_df.index, (first_df["Cum_Benchmark"] - first_df["Cum_Benchmark"].cummax()) / first_df["Cum_Benchmark"].cummax() * 100,
             color="#333333", linestyle="--", lw=1.2, label="Buy & Hold")

    for model_name, df in daily_results.items():
        c = PALETTE.get(model_name, None)
        ax1.plot(df.index, df["Cum_Net"], label=f"VolTarget ({model_name})", color=c, lw=1.4)
        ax2.plot(df.index, df["Drawdown_Net"] * 100, label=model_name, color=c, lw=1.1)

    ax1.set_title("Economic Evaluation: Volatility-Targeted Portfolio Growth (Net of Transaction Costs)")
    ax1.set_ylabel("Cumulative Wealth ($1.0 Base)")
    ax1.legend(loc="upper left", framealpha=0.9)

    ax2.set_title("Strategy Drawdown Profiles (%)")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Drawdown (%)")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_feature_importances(importance_dict: Dict[str, pd.Series], save_path: Optional[str] = None) -> plt.Figure:
    """Plot feature importance ranking across Random Forest, XGBoost, and Hybrid models."""
    n_models = len(importance_dict)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 6), sharey=False)
    if n_models == 1:
        axes = [axes]

    for ax, (model_name, importances) in zip(axes, importance_dict.items()):
        top_feats = importances.head(10).iloc[::-1]
        ax.barh(top_feats.index, top_feats.values, color="#1f77b4", edgecolor="black", alpha=0.8)
        ax.set_title(f"{model_name}\nTop 10 Feature Importances")
        ax.set_xlabel("Relative Importance")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_gross_vs_net_sharpe(summary_df: pd.DataFrame, save_path: Optional[str] = None) -> plt.Figure:
    """Plot Gross vs Net Sharpe ratio to illuminate transaction cost drag."""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    x = np.arange(len(summary_df))
    width = 0.35

    ax.bar(x - width/2, summary_df["Sharpe_Gross"], width, label="Gross Sharpe (Before Costs)", color="#4682b4", edgecolor="black")
    ax.bar(x + width/2, summary_df["Sharpe_Net"], width, label="Net Sharpe (After Costs & Slippage)", color="#d95f02", edgecolor="black")

    ax.set_xticks(x)
    ax.set_xticklabels(summary_df["Strategy"], rotation=25, ha="right")
    ax.set_ylabel("Annualized Sharpe Ratio")
    ax.set_title("Economic Significance: Impact of Transaction Costs on Portfolio Sharpe Ratios")
    ax.axhline(0, color="black", lw=0.8, linestyle="--")
    ax.legend(loc="upper right")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_dm_heatmap(dm_stat_matrix: pd.DataFrame, dm_pval_matrix: pd.DataFrame, save_path: Optional[str] = None) -> plt.Figure:
    """Plot pairwise Diebold-Mariano test statistic and p-value matrix heatmap."""
    fig, ax = plt.subplots(figsize=(8, 6.5))
    
    # Text annotation with DM Stat and p-value
    data = dm_stat_matrix.values
    models = list(dm_stat_matrix.index)
    
    im = ax.imshow(data, cmap="coolwarm", vmin=-4.0, vmax=4.0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("DM Test Statistic (>0 implies Column Superior to Row)")

    ax.set_xticks(np.arange(len(models)))
    ax.set_yticks(np.arange(len(models)))
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.set_yticklabels(models)

    for i in range(len(models)):
        for j in range(len(models)):
            if i == j:
                text = "—"
            else:
                stat_val = dm_stat_matrix.iloc[i, j]
                pval_val = dm_pval_matrix.iloc[i, j]
                stars = "***" if pval_val < 0.01 else ("**" if pval_val < 0.05 else ("*" if pval_val < 0.1 else ""))
                text = f"{stat_val:.2f}{stars}\n(p={pval_val:.3f})"
            ax.text(j, i, text, ha="center", va="center", color="black" if abs(data[i, j]) < 2.5 else "white", fontsize=8)

    ax.set_title("Pairwise Diebold-Mariano Comparison Matrix (HLN Corrected)\n*** p<0.01, ** p<0.05, * p<0.10")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_transaction_cost_sensitivity(cost_df: pd.DataFrame, save_path: Optional[str] = None) -> plt.Figure:
    """Plot Net Sharpe Ratio vs. Transaction Cost (bps) across all models."""
    fig, ax = plt.subplots(figsize=(10, 5.5))
    
    for m in cost_df["Model"].unique():
        sub = cost_df[cost_df["Model"] == m].sort_values("Cost_bps")
        c = PALETTE.get(m, None)
        ax.plot(sub["Cost_bps"], sub["Net_Sharpe"], marker="o", lw=1.6, label=f"VolTarget ({m})", color=c)

    ax.axhline(0, color="black", linestyle="--", lw=0.8)
    ax.set_xlabel("Transaction Cost per One-Way Trade (Basis Points)")
    ax.set_ylabel("Annualized Net Sharpe Ratio")
    ax.set_title("Transaction Cost Sensitivity: Net Sharpe vs. Execution Friction")
    ax.legend(loc="upper right", framealpha=0.95)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_turnover_vs_net_sharpe(strat_df: pd.DataFrame, save_path: Optional[str] = None) -> plt.Figure:
    """Scatter plot illustrating the Empirical Turnover-Friction Tradeoff."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    
    sub = strat_df[strat_df["Strategy"] != "Buy & Hold (Passive 100%)"].copy()

    for _, row in sub.iterrows():
        strat = row["Strategy"]
        turnover = row["Annual_Turnover"]
        sharpe = row["Sharpe_Net"]
        ax.scatter(turnover, sharpe, s=120, edgecolor="black", label=strat)
        ax.annotate(strat.replace("VolTarget (", "").replace(")", ""), (turnover + 0.3, sharpe + 0.005), fontsize=9)

    ax.set_xlabel("Annualized Portfolio Turnover")
    ax.set_ylabel("Annualized Net Sharpe Ratio (After Costs & Slippage)")
    ax.set_title("The Turnover Penalty: High Trading Frequency Erodes Risk-Adjusted Returns")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
    return fig

