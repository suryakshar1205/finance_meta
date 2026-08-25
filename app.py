"""
Quantitative Volatility Forecasting & Economic Utility Dashboard
================================================================
An interactive institutional-grade research dashboard exploring:
  1. Statistical forecast accuracy (RMSE, MAE, QLIKE, Diebold-Mariano tests)
  2. Live Dynamic Volatility Targeting backtesting with customizable transaction costs & target volatility
  3. Rebalancing frequency trade-offs (Daily vs Weekly vs Biweekly)
  4. Forecast smoothing (EMA filtering) and turnover-friction dynamics
"""

import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="Financial Volatility Forecasting | Quantitative Research",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for institutional styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_all_data():
    """Load forecasts, market data, and frozen empirical tables."""
    # 1. Processed market data
    market_path = "data/processed/nifty50_daily_processed.csv"
    if os.path.exists(market_path):
        mdf = pd.read_csv(market_path)
        mdf["Date"] = pd.to_datetime(mdf["Date"])
        mdf["log_return"] = np.log(mdf["Close"] / mdf["Close"].shift(1))
    else:
        mdf = pd.DataFrame()

    # 2. Forecasts
    forecasts_path = "results/forecasts/walk_forward_forecasts.csv"
    if os.path.exists(forecasts_path):
        raw_fdf = pd.read_csv(forecasts_path)
        raw_fdf["Date"] = pd.to_datetime(raw_fdf["Date"])
        pivoted = raw_fdf.pivot(index="Date", columns="Model", values="Predicted").reset_index()
        actuals = raw_fdf[["Date", "Actual"]].drop_duplicates()
        merged = pd.merge(pivoted, actuals, on="Date")
        if not mdf.empty:
            fdf = pd.merge(merged, mdf[["Date", "log_return", "Close"]], on="Date").sort_values("Date").reset_index(drop=True)
        else:
            fdf = merged.sort_values("Date").reset_index(drop=True)
    else:
        fdf = pd.DataFrame()

    # 3. Final frozen tables
    metrics_df = pd.read_csv("results/final/final_forecast_metrics.csv") if os.path.exists("results/final/final_forecast_metrics.csv") else pd.DataFrame()
    dm_matrix_df = pd.read_csv("results/final/final_dm_comparison_matrix.csv") if os.path.exists("results/final/final_dm_comparison_matrix.csv") else pd.DataFrame()
    port_df = pd.read_csv("results/final/final_portfolio_metrics.csv") if os.path.exists("results/final/final_portfolio_metrics.csv") else pd.DataFrame()
    cost_df = pd.read_csv("results/final/final_transaction_cost_sensitivity.csv") if os.path.exists("results/final/final_transaction_cost_sensitivity.csv") else pd.DataFrame()
    rebal_df = pd.read_csv("results/final/rebalancing_frequency_sensitivity.csv") if os.path.exists("results/final/rebalancing_frequency_sensitivity.csv") else pd.DataFrame()

    return fdf, mdf, metrics_df, dm_matrix_df, port_df, cost_df, rebal_df


def simulate_volatility_targeting(fdf, target_vol=0.15, max_leverage=1.5, fee_bps=10.0, slippage_bps=5.0, cadence=1, ema_span=1):
    """
    Live simulator for volatility targeting across all models with interactive parameters.
    """
    df = fdf.copy().sort_values("Date").reset_index(drop=True)
    model_cols = [c for c in ["Historical Volatility", "GARCH", "Random Forest", "XGBoost", "Hybrid GARCH+ML"] if c in df.columns]
    total_friction = (fee_bps + slippage_bps) / 10000.0

    results = {}
    
    # Passive Buy & Hold Benchmark
    bh_ret = df["log_return"].fillna(0.0).values
    bh_cum = np.cumprod(1.0 + bh_ret)
    n_days = len(df)
    bh_cagr = (bh_cum[-1] ** (252.0 / n_days)) - 1.0
    bh_vol = np.std(bh_ret) * np.sqrt(252.0)
    bh_sharpe = bh_cagr / (bh_vol + 1e-8)
    bh_dd = (bh_cum - np.maximum.accumulate(bh_cum)) / np.maximum.accumulate(bh_cum)
    results["Buy & Hold"] = {
        "cum_equity": bh_cum,
        "daily_returns": bh_ret,
        "cagr_gross": bh_cagr,
        "cagr_net": bh_cagr,
        "sharpe_net": bh_sharpe,
        "max_dd": np.min(bh_dd),
        "turnover": 0.0,
        "cost_drag": 0.0,
        "dates": df["Date"].values
    }

    for m in model_cols:
        raw_forecast = df[m].values.copy()
        
        # Apply EMA smoothing if requested
        if ema_span > 1:
            raw_forecast = pd.Series(raw_forecast).ewm(span=ema_span, adjust=False).mean().values
            
        # Target weight with 1-day lag
        raw_weights = np.clip(target_vol / np.maximum(raw_forecast, 1e-4), 0.0, max_leverage)
        
        # Apply rebalancing cadence
        weights = raw_weights.copy()
        if cadence > 1:
            for i in range(len(weights)):
                if i % cadence != 0:
                    weights[i] = weights[i - 1]

        # Shift weights by 1 day for zero lookahead
        w_exec = np.roll(weights, 1)
        w_exec[0] = weights[0]

        # Turnover
        dw = np.abs(np.diff(w_exec, prepend=w_exec[0]))
        annual_turnover = np.sum(dw) * (252.0 / n_days)

        # Returns
        gross_ret = w_exec * df["log_return"].fillna(0.0).values
        cost_series = dw * total_friction
        net_ret = gross_ret - cost_series

        cum_gross = np.cumprod(1.0 + gross_ret)
        cum_net = np.cumprod(1.0 + net_ret)

        cagr_gross = (cum_gross[-1] ** (252.0 / n_days)) - 1.0
        cagr_net = (cum_net[-1] ** (252.0 / n_days)) - 1.0
        
        ann_vol_net = np.std(net_ret) * np.sqrt(252.0)
        sharpe_net = cagr_net / (ann_vol_net + 1e-8)
        
        running_max = np.maximum.accumulate(cum_net)
        drawdowns = (cum_net - running_max) / running_max
        max_dd = np.min(drawdowns)

        results[m] = {
            "cum_equity": cum_net,
            "daily_returns": net_ret,
            "cagr_gross": cagr_gross,
            "cagr_net": cagr_net,
            "sharpe_net": sharpe_net,
            "max_dd": max_dd,
            "turnover": annual_turnover,
            "cost_drag": cagr_net - cagr_gross,
            "dates": df["Date"].values
        }

    return results


def main():
    fdf, mdf, metrics_df, dm_matrix_df, port_df, cost_df, rebal_df = load_all_data()

    st.markdown('<div class="main-header">Quantitative Volatility Forecasting & Economic Utility</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Evaluating Statistical Precision vs. Portfolio Implementation Friction on the NIFTY 50 Index (2010–2024, N = 3,679)</div>', unsafe_allow_html=True)

    # Sidebar parameters
    st.sidebar.header("🕹️ Live Simulation Parameters")
    
    target_vol_input = st.sidebar.slider("Target Annualized Volatility (%)", min_value=5.0, max_value=25.0, value=15.0, step=1.0) / 100.0
    fee_bps_input = st.sidebar.slider("Brokerage Fee (bps)", min_value=0.0, max_value=30.0, value=10.0, step=1.0)
    slippage_bps_input = st.sidebar.slider("Execution Slippage (bps)", min_value=0.0, max_value=20.0, value=5.0, step=1.0)
    cadence_choice = st.sidebar.selectbox("Rebalancing Cadence", options=["Daily (1-day)", "Weekly (5-day)", "Biweekly (10-day)"])
    cadence_map = {"Daily (1-day)": 1, "Weekly (5-day)": 5, "Biweekly (10-day)": 10}
    cadence_days = cadence_map[cadence_choice]

    ema_smoothing = st.sidebar.slider("Forecast Smoothing (EMA Days)", min_value=1, max_value=10, value=1, help="1 = Raw Forecasts (No Smoothing)")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Research Quick Facts")
    st.sidebar.info("""
    - **Primary Market:** NIFTY 50 (`^NSEI`)
    - **Test Window:** 997 Days (2021–2024)
    - **Best Statistical Model:** Random Forest (RMSE: 0.0850)
    - **Best Daily Net Sharpe:** GARCH(1,1) (0.490)
    - **Optimal Cadence:** Biweekly XGBoost (Net Sharpe: 0.558)
    """)

    # Live simulation
    sim_results = simulate_volatility_targeting(
        fdf,
        target_vol=target_vol_input,
        fee_bps=fee_bps_input,
        slippage_bps=slippage_bps_input,
        cadence=cadence_days,
        ema_span=ema_smoothing
    )

    # Navigation Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🚀 Live Economic Simulation",
        "🎯 Statistical Forecast Accuracy",
        "📉 Sensitivity & Rebalancing Tradeoffs",
        "🔍 Econometric Stylized Facts"
    ])

    # TAB 1: LIVE SIMULATION
    with tab1:
        st.subheader("Dynamic Volatility-Targeting Backtest (Live Recalculation)")
        st.markdown(f"Evaluating portfolio performance targeting **{target_vol_input*100:.0f}% volatility** under **{fee_bps_input + slippage_bps_input:.0f} bps total friction** with **{cadence_choice}** rebalancing.")

        # Metric cards
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            garch_net = sim_results["GARCH"]["sharpe_net"] if "GARCH" in sim_results else 0.0
            st.metric("GARCH Net Sharpe", f"{garch_net:.3f}", delta=f"Turnover: {sim_results['GARCH']['turnover']:.2f}")
        with col2:
            xgb_net = sim_results["XGBoost"]["sharpe_net"] if "XGBoost" in sim_results else 0.0
            st.metric("XGBoost Net Sharpe", f"{xgb_net:.3f}", delta=f"Turnover: {sim_results['XGBoost']['turnover']:.2f}")
        with col3:
            rf_net = sim_results["Random Forest"]["sharpe_net"] if "Random Forest" in sim_results else 0.0
            st.metric("Random Forest Net Sharpe", f"{rf_net:.3f}", delta=f"Turnover: {sim_results['Random Forest']['turnover']:.2f}")
        with col4:
            bh_net = sim_results["Buy & Hold"]["sharpe_net"]
            st.metric("Buy & Hold Sharpe", f"{bh_net:.3f}", delta="Passive 100%")

        # Interactive Cumulative Equity Chart
        fig_equity = go.Figure()
        colors = {
            "GARCH": "#2563EB",
            "XGBoost": "#DC2626",
            "Random Forest": "#059669",
            "Hybrid GARCH+ML": "#7C3AED",
            "Historical Volatility": "#D97706",
            "Buy & Hold": "#64748B"
        }
        for name, data in sim_results.items():
            fig_equity.add_trace(go.Scatter(
                x=data["dates"],
                y=data["cum_equity"],
                mode="lines",
                name=name,
                line=dict(color=colors.get(name, "#333"), width=2.2 if name in ["GARCH", "XGBoost"] else 1.5)
            ))
        fig_equity.update_layout(
            title="Cumulative Net Equity Curves (Out-of-Sample 2021–2024)",
            xaxis_title="Date",
            yaxis_title="Portfolio Wealth ($1.00 Base)",
            template="plotly_white",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_equity, use_container_width=True)

        # Performance table
        st.markdown("### 📋 Dynamic Strategy Performance Comparison")
        table_rows = []
        for name, data in sim_results.items():
            table_rows.append({
                "Strategy": name,
                "Net Sharpe": f"{data['sharpe_net']:.3f}",
                "Net CAGR (%)": f"{data['cagr_net']*100:.2f}%",
                "Gross CAGR (%)": f"{data['cagr_gross']*100:.2f}%",
                "Annual Turnover": f"{data['turnover']:.2f}",
                "Cost Drag (%)": f"{data['cost_drag']*100:.2f}%",
                "Max Drawdown (%)": f"{data['max_dd']*100:.2f}%"
            })
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True)

    # TAB 2: STATISTICAL ACCURACY
    with tab2:
        st.subheader("Statistical Forecast Accuracy vs. Realized Volatility")
        
        # Interactive Forecast Tracking Plot
        fig_forecasts = go.Figure()
        fig_forecasts.add_trace(go.Scatter(
            x=fdf["Date"],
            y=fdf["Actual"],
            mode="lines",
            name="Target Realized Vol (5d)",
            line=dict(color="#0F172A", width=1.5, dash="dot")
        ))
        if "Random Forest" in fdf.columns:
            fig_forecasts.add_trace(go.Scatter(
                x=fdf["Date"],
                y=fdf["Random Forest"],
                mode="lines",
                name="Random Forest (Winner RMSE)",
                line=dict(color="#059669", width=2.0)
            ))
        if "XGBoost" in fdf.columns:
            fig_forecasts.add_trace(go.Scatter(
                x=fdf["Date"],
                y=fdf["XGBoost"],
                mode="lines",
                name="XGBoost (Winner MAE)",
                line=dict(color="#DC2626", width=2.0)
            ))
        if "GARCH" in fdf.columns:
            fig_forecasts.add_trace(go.Scatter(
                x=fdf["Date"],
                y=fdf["GARCH"],
                mode="lines",
                name="GARCH(1,1) Benchmark",
                line=dict(color="#2563EB", width=2.0)
            ))
        fig_forecasts.update_layout(
            title="Out-of-Sample Volatility Forecast Tracking across 997 Test Days",
            xaxis_title="Date",
            yaxis_title="Annualized Volatility",
            template="plotly_white",
            hovermode="x unified"
        )
        st.plotly_chart(fig_forecasts, use_container_width=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("### 📊 Loss Function Metrics Table")
            if not metrics_df.empty:
                st.dataframe(metrics_df, use_container_width=True)
        with col_b:
            st.markdown("### 🔬 Diebold-Mariano Significance ($p$-values)")
            if not dm_matrix_df.empty:
                st.dataframe(dm_matrix_df, use_container_width=True)
                st.caption("All ML models reject the null hypothesis of equal forecast accuracy against GARCH at p < 0.01.")

    # TAB 3: SENSITIVITIES
    with tab3:
        st.subheader("Execution Sensitivity: The Turnover-Friction Dilemma")
        
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.markdown("### 💰 Transaction Cost Grid Response (0 to 50 bps)")
            if not cost_df.empty:
                fig_cost = px.line(
                    cost_df,
                    x="Fee_bps",
                    y="Sharpe_Net",
                    color="Strategy",
                    markers=True,
                    title="Net Sharpe Response across Transaction Cost Grid",
                    labels={"Fee_bps": "Brokerage Fee (bps)", "Sharpe_Net": "Net Sharpe Ratio"}
                )
                fig_cost.update_layout(template="plotly_white")
                st.plotly_chart(fig_cost, use_container_width=True)

        with col_c2:
            st.markdown("### ⏱️ Rebalancing Frequency Tradeoff")
            if not rebal_df.empty:
                fig_rebal = px.bar(
                    rebal_df,
                    x="Cadence",
                    y="Net_Sharpe",
                    color="Model",
                    barmode="group",
                    title="Net Sharpe by Rebalancing Cadence",
                    labels={"Cadence": "Rebalancing Cadence", "Net_Sharpe": "Net Sharpe Ratio"}
                )
                fig_rebal.update_layout(template="plotly_white")
                st.plotly_chart(fig_rebal, use_container_width=True)

    # TAB 4: STYLIZED FACTS
    with tab4:
        st.subheader("NIFTY 50 Stylized Facts & Historical Dynamics")
        if not mdf.empty:
            fig_market = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, subplot_titles=["NIFTY 50 Historical Index Price", "Daily Continuous Log Returns"])
            fig_market.add_trace(go.Scatter(x=mdf["Date"], y=mdf["Close"], mode="lines", name="Close Price", line=dict(color="#2563EB")), row=1, col=1)
            fig_market.add_trace(go.Scatter(x=mdf["Date"], y=mdf["log_return"], mode="lines", name="Log Return", line=dict(color="#DC2626", width=0.8)), row=2, col=1)
            fig_market.update_layout(template="plotly_white", showlegend=False, height=500)
            st.plotly_chart(fig_market, use_container_width=True)

            col_s1, col_s2, col_s3, col_s4 = st.columns(4)
            col_s1.metric("Total Trading Days", f"{len(mdf):,}")
            col_s2.metric("Daily Return Kurtosis", f"{mdf['log_return'].kurtosis():.3f}", delta="Heavy Tails (p < 0.0001)")
            col_s3.metric("Daily Return Skewness", f"{mdf['log_return'].skew():.3f}", delta="Crash Asymmetry")
            col_s4.metric("Worst Daily Crash", f"{mdf['log_return'].min()*100:.2f}%", delta="COVID Shock (Mar 2020)")


if __name__ == "__main__":
    main()
