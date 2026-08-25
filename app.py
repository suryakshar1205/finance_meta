"""
Quantitative Volatility Forecasting & Economic Utility Dashboard
================================================================
An institutional-grade research dashboard structured into 5 clean tabs:
  1. Overview (Research question, 5 models, project flow, 3 findings)
  2. Market & Data (NIFTY 50 index dynamics, stylized facts, tooltips)
  3. Forecasting Models (Statistical loss space: RMSE, MAE, QLIKE, DM matrix)
  4. Portfolio Simulation (Baseline box, Forecast->Portfolio flow, 4 controls, live simulation, "Why did this change?" explanation)
  5. Final Verdict (Executive summary for viva presentation)
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
    page_title="Volatility Forecasting vs Economic Utility | Quantitative Research",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.1rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 0.2rem;
    }
    .research-quote {
        font-size: 1.15rem;
        font-weight: 500;
        color: #1E40AF;
        background-color: #EFF6FF;
        border-left: 4px solid #3B82F6;
        padding: 12px 18px;
        border-radius: 4px;
        margin: 15px 0 25px 0;
    }
    .baseline-box {
        background-color: #F8FAFC;
        border: 1px solid #CBD5E1;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 20px;
    }
    .verdict-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
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


def simulate_volatility_targeting(fdf, target_vol=0.15, max_leverage=1.5, total_cost_bps=15.0, cadence=1, ema_span=1):
    """Live simulator for volatility targeting across all models with interactive parameters."""
    df = fdf.copy().sort_values("Date").reset_index(drop=True)
    model_cols = [c for c in ["Historical Volatility", "GARCH", "Random Forest", "XGBoost", "Hybrid GARCH+ML"] if c in df.columns]
    total_friction = total_cost_bps / 10000.0

    results = {}
    n_days = len(df)
    
    # Passive Buy & Hold Benchmark
    bh_ret = df["log_return"].fillna(0.0).values
    bh_cum = np.cumprod(1.0 + bh_ret)
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
        
        # EMA smoothing
        if ema_span > 1:
            raw_forecast = pd.Series(raw_forecast).ewm(span=ema_span, adjust=False).mean().values
            
        # Target weight with 1-day lag
        raw_weights = np.clip(target_vol / np.maximum(raw_forecast, 1e-4), 0.0, max_leverage)
        
        # Cadence filter
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

    # Title & Subtitle
    st.markdown('<div class="main-title">A Comparative & Hybrid Framework for Volatility Forecasting</div>', unsafe_allow_html=True)
    st.markdown('<div class="research-quote">"Can better volatility forecasts lead to better investment performance after transaction costs?"</div>', unsafe_allow_html=True)

    # 4 INTERACTIVE CONTROLS IN SIDEBAR (Improvement 7)
    st.sidebar.header("🕹️ Interactive Simulation Controls")
    st.sidebar.caption("Adjust parameters to explore turnover & cost dynamics:")
    
    target_vol_input = st.sidebar.slider(
        "Target Volatility ⓘ",
        min_value=5, max_value=25, value=15, step=1, format="%d%%",
        help="Annualized target volatility level for dynamic leverage sizing."
    ) / 100.0

    total_cost_input = st.sidebar.slider(
        "Transaction Cost ⓘ",
        min_value=0, max_value=50, value=15, step=1, format="%d bps",
        help="Total execution friction per unit position turnover (fee + slippage in basis points. 1 bps = 0.01%)."
    )

    cadence_choice = st.sidebar.selectbox(
        "Rebalancing Cadence ⓘ",
        options=["Daily", "Weekly (5-day)", "Biweekly (10-day)"],
        index=0,
        help="How frequently the portfolio adjusts position weights to match updated volatility forecasts."
    )
    cadence_map = {"Daily": 1, "Weekly (5-day)": 5, "Biweekly (10-day)": 10}
    cadence_days = cadence_map[cadence_choice]

    ema_smoothing = st.sidebar.slider(
        "Forecast Smoothing (EMA) ⓘ",
        min_value=1, max_value=10, value=1, format="%d days",
        help="Applies an exponential moving average filter to smooth volatility forecasts and reduce turnover. 1d = Raw Forecast."
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📚 Key Terminology Guide")
    st.sidebar.markdown("""
    - **Volatility ⓘ:** Standard deviation of asset price returns measuring turbulence.
    - **RMSE ⓘ:** Root Mean Squared Error (penalizes large forecast errors heavily).
    - **MAE ⓘ:** Mean Absolute Error (average absolute forecast error).
    - **QLIKE ⓘ:** Quasi-Likelihood scale-invariant loss function.
    - **Sharpe Ratio ⓘ:** Return generated per unit of realized volatility.
    - **CAGR ⓘ:** Compound Annual Growth Rate.
    - **Turnover ⓘ:** Frequency of position changes per year. Higher turnover = higher fees.
    - **Drawdown ⓘ:** Peak-to-trough percentage decline in portfolio wealth.
    """)

    # 5 CLEAN TABS (Improvement 2)
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "1. Overview",
        "2. Market & Data",
        "3. Forecasting Models",
        "4. Portfolio Simulation",
        "5. Final Verdict"
    ])

    # =========================================================================
    # TAB 1: OVERVIEW (Improvements 1 & 4)
    # =========================================================================
    with tab1:
        st.subheader("Project Overview")
        
        # Project Flow Diagram (Improvement 1)
        st.markdown("### 🔄 End-to-End Research Pipeline")
        st.code("""
Market Data (NIFTY 50 OHLCV)
         ↓
Volatility Forecasting (5 Models)
         ↓
Forecast Evaluation (RMSE, MAE, QLIKE, DM Test)
         ↓
Portfolio Simulation (15% Dynamic Vol Targeting)
         ↓
Transaction Costs (15 bps Total Execution Friction)
         ↓
Final Verdict (Statistical Accuracy vs Economic Utility)
        """, language="text")

        st.markdown("---")
        
        # 5 Models Explanation Table (Improvement 4)
        st.markdown("### 🤖 The 5 Evaluated Volatility Models")
        model_table_data = [
            {"Model": "Historical Volatility", "Type": "Baseline", "Simple Explanation": "Uses recent 20-day historical volatility as the future forecast."},
            {"Model": "GARCH(1,1)", "Type": "Econometric", "Simple Explanation": "Models how volatility changes and persists gradually over time (mean-reversion)."},
            {"Model": "Random Forest", "Type": "Machine Learning", "Simple Explanation": "Uses 200 de-correlated decision trees to identify complex non-linear price patterns."},
            {"Model": "XGBoost", "Type": "Machine Learning", "Simple Explanation": "Uses boosted trees that learn sequentially from past prediction mistakes."},
            {"Model": "Hybrid Model", "Type": "Novel Proposed", "Simple Explanation": "Combines GARCH conditional volatility estimates directly into an XGBoost learner."}
        ]
        st.dataframe(pd.DataFrame(model_table_data), use_container_width=True)

        st.markdown("---")

        # 3 Key Findings (Improvement 1)
        st.markdown("### 💡 3 Key Research Findings")
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            st.info("""
            **1. ML Wins on Forecasting**  
            Random Forest and XGBoost reduced volatility forecast errors by **15% to 16%** over GARCH ($p < 0.01$).
            """)
        with col_f2:
            st.warning("""
            **2. GARCH Wins on Daily Net Return**  
            In daily portfolio trading, GARCH achieved higher Net Sharpe (**0.490 vs 0.416**) due to far lower turnover (**2.47 vs 17.31**).
            """)
        with col_f3:
            st.success("""
            **3. Rebalancing Solves the Gap**  
            Switching ML to **Weekly or Biweekly rebalancing** cuts turnover by >50%, lifting XGBoost Net Sharpe to **0.558** (beating GARCH).
            """)

    # =========================================================================
    # TAB 2: MARKET & DATA
    # =========================================================================
    with tab2:
        st.subheader("NIFTY 50 Market Benchmark & Dataset")
        st.markdown("Analyzing daily trading sessions of India's benchmark **NIFTY 50 Index (`^NSEI`)** from **2010 to 2024** ($N = 3,679$ trading days).")

        if not mdf.empty:
            # Interactive Chart
            fig_m = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, subplot_titles=["NIFTY 50 Index Price (INR)", "Daily Continuous Log Returns"])
            fig_m.add_trace(go.Scatter(x=mdf["Date"], y=mdf["Close"], mode="lines", name="Close Price", line=dict(color="#2563EB")), row=1, col=1)
            fig_m.add_trace(go.Scatter(x=mdf["Date"], y=mdf["log_return"], mode="lines", name="Log Return", line=dict(color="#DC2626", width=0.8)), row=2, col=1)
            fig_m.update_layout(template="plotly_white", showlegend=False, height=450)
            st.plotly_chart(fig_m, use_container_width=True)

            # Metrics
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Historical Sample Size ⓘ", f"{len(mdf):,} Days", help="15 years of daily continuous trading data (2010–2024)")
            c2.metric("Out-of-Sample Test Window ⓘ", "997 Days", help="Strictly out-of-sample expanding walk-forward test period (2021–2024)")
            c3.metric("Daily Return Kurtosis ⓘ", f"{mdf['log_return'].kurtosis():.2f}", delta="Heavy Tails (p < 0.0001)", help="Leptokurtic fat tails indicating frequent extreme market swings")
            c4.metric("Daily Return Skewness ⓘ", f"{mdf['log_return'].skew():.2f}", delta="Crash Asymmetry", help="Negative skewness reflecting faster downside market crashes")

    # =========================================================================
    # TAB 3: FORECASTING MODELS (Improvement 6)
    # =========================================================================
    with tab3:
        st.subheader("Statistical Forecast Accuracy ($N = 997$ Test Days)")

        # Central Comparison Table (Improvement 6)
        st.markdown("### 🏆 Core Comparison: Statistical Accuracy vs. Daily Economic Outcome")
        comparison_summary = [
            {"Model": "Random Forest", "RMSE (Statistical Error) ⓘ": "0.08497 (Winner)", "MAE ⓘ": "0.05151", "QLIKE ⓘ": "0.47287", "Daily Turnover ⓘ": "12.72", "Daily Net Sharpe ⓘ": "0.371"},
            {"Model": "XGBoost", "RMSE (Statistical Error) ⓘ": "0.08594", "MAE ⓘ": "0.05072 (Winner)", "QLIKE ⓘ": "0.48869", "Daily Turnover ⓘ": "17.31", "Daily Net Sharpe ⓘ": "0.416"},
            {"Model": "Hybrid GARCH+ML", "RMSE (Statistical Error) ⓘ": "0.08657", "MAE ⓘ": "0.05266", "QLIKE ⓘ": "0.47664", "Daily Turnover ⓘ": "17.27", "Daily Net Sharpe ⓘ": "0.321"},
            {"Model": "Historical Vol (20d)", "RMSE (Statistical Error) ⓘ": "0.10128", "MAE ⓘ": "0.06036", "QLIKE ⓘ": "0.56519", "Daily Turnover ⓘ": "8.29", "Daily Net Sharpe ⓘ": "0.393"},
            {"Model": "GARCH(1,1)", "RMSE (Statistical Error) ⓘ": "0.10497", "MAE ⓘ": "0.06320", "QLIKE ⓘ": "0.78001", "Daily Turnover ⓘ": "2.47 (Lowest)", "Daily Net Sharpe ⓘ": "0.490 (Winner)"}
        ]
        st.dataframe(pd.DataFrame(comparison_summary), use_container_width=True)
        
        st.info("💡 **Central Discovery:** Machine Learning models produce significantly more accurate volatility forecasts (lower RMSE/MAE/QLIKE), but GARCH achieves better daily economic performance because of its much lower turnover.")

        # Interactive Forecast Tracking Plot
        st.markdown("### 📈 Out-of-Sample Volatility Forecast Tracking")
        if not fdf.empty:
            fig_fc = go.Figure()
            fig_fc.add_trace(go.Scatter(x=fdf["Date"], y=fdf["Actual"], mode="lines", name="Target Realized Vol (5d)", line=dict(color="#0F172A", width=1.5, dash="dot")))
            if "Random Forest" in fdf.columns:
                fig_fc.add_trace(go.Scatter(x=fdf["Date"], y=fdf["Random Forest"], mode="lines", name="Random Forest (Winner RMSE)", line=dict(color="#059669", width=2.0)))
            if "XGBoost" in fdf.columns:
                fig_fc.add_trace(go.Scatter(x=fdf["Date"], y=fdf["XGBoost"], mode="lines", name="XGBoost (Winner MAE)", line=dict(color="#DC2626", width=2.0)))
            if "GARCH" in fdf.columns:
                fig_fc.add_trace(go.Scatter(x=fdf["Date"], y=fdf["GARCH"], mode="lines", name="GARCH(1,1) Benchmark", line=dict(color="#2563EB", width=2.0)))
            fig_fc.update_layout(template="plotly_white", hovermode="x unified", height=420)
            st.plotly_chart(fig_fc, use_container_width=True)

    # =========================================================================
    # TAB 4: PORTFOLIO SIMULATION (Improvements 3, 5, 7, 8)
    # =========================================================================
    with tab4:
        st.subheader("Dynamic Volatility-Targeting Simulation")

        # Research Baseline Box (Improvement 3)
        st.markdown("""
        <div class="baseline-box">
            <h4 style="margin:0 0 10px 0; color:#1E293B;">📌 Frozen Research Baseline Benchmark</h4>
            <div style="display:flex; justify-content:space-between; flex-wrap:wrap; font-size:0.95rem;">
                <span><b>Target Volatility:</b> 15%</span>
                <span><b>Transaction Friction:</b> 15 bps (10 bps fee + 5 bps slippage)</span>
                <span><b>Rebalancing Cadence:</b> Daily</span>
                <span><b>Forecast Smoothing:</b> None (Raw)</span>
                <span><b>Test Days:</b> 997 Out-of-Sample Days</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Forecast -> Portfolio Flow Diagram (Improvement 5)
        st.markdown("### ⚙️ How Forecasting Becomes a Financial Strategy")
        st.code("""
Volatility Forecast (sigma_hat)
        ↓
Compare with 15% Target Risk: Weight w_t = min(0.15 / sigma_hat, 1.5)
        ↓
Shift by 1 Day for Zero Lookahead Execution
        ↓
Gross Portfolio Return = w_{t-1} * Return_t
        ↓
Transaction Costs Deducted = |w_t - w_{t-1}| * 15 bps Total Friction
        ↓
Net Economic Utility (Net Sharpe & Net CAGR)
        """, language="text")

        st.markdown("---")

        # Run Live Simulation with user parameters
        sim_results = simulate_volatility_targeting(
            fdf,
            target_vol=target_vol_input,
            total_cost_bps=total_cost_input,
            cadence=cadence_days,
            ema_span=ema_smoothing
        )

        # Live Results Metric Cards
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            garch_s = sim_results["GARCH"]["sharpe_net"] if "GARCH" in sim_results else 0.0
            st.metric("GARCH Net Sharpe ⓘ", f"{garch_s:.3f}", delta=f"Turnover: {sim_results['GARCH']['turnover']:.2f}", help="Net Sharpe ratio after transaction costs")
        with c2:
            xgb_s = sim_results["XGBoost"]["sharpe_net"] if "XGBoost" in sim_results else 0.0
            st.metric("XGBoost Net Sharpe ⓘ", f"{xgb_s:.3f}", delta=f"Turnover: {sim_results['XGBoost']['turnover']:.2f}", help="Net Sharpe ratio after transaction costs")
        with c3:
            rf_s = sim_results["Random Forest"]["sharpe_net"] if "Random Forest" in sim_results else 0.0
            st.metric("Random Forest Net Sharpe ⓘ", f"{rf_s:.3f}", delta=f"Turnover: {sim_results['Random Forest']['turnover']:.2f}", help="Net Sharpe ratio after transaction costs")
        with c4:
            bh_s = sim_results["Buy & Hold"]["sharpe_net"]
            st.metric("Buy & Hold Sharpe ⓘ", f"{bh_s:.3f}", delta="Passive 100%", help="Passive unmanaged benchmark")

        # Live Equity Curve Plot
        fig_eq = go.Figure()
        colors = {
            "GARCH": "#2563EB",
            "XGBoost": "#DC2626",
            "Random Forest": "#059669",
            "Hybrid GARCH+ML": "#7C3AED",
            "Historical Volatility": "#D97706",
            "Buy & Hold": "#64748B"
        }
        for name, data in sim_results.items():
            fig_eq.add_trace(go.Scatter(
                x=data["dates"],
                y=data["cum_equity"],
                mode="lines",
                name=name,
                line=dict(color=colors.get(name, "#333"), width=2.2 if name in ["GARCH", "XGBoost"] else 1.5)
            ))
        fig_eq.update_layout(
            title=f"Cumulative Net Wealth ({target_vol_input*100:.0f}% Target Vol, {total_cost_input} bps Friction, {cadence_choice} Rebalancing)",
            xaxis_title="Date",
            yaxis_title="Portfolio Wealth ($1.00 Base)",
            template="plotly_white",
            hovermode="x unified",
            height=430
        )
        st.plotly_chart(fig_eq, use_container_width=True)

        # "Why Did The Result Change?" Explanation Box (Improvement 8)
        st.markdown("### ❓ Why Did the Result Change?")
        if cadence_choice != "Daily":
            st.success(f"""
            **Why did Machine Learning performance improve under {cadence_choice} rebalancing?**  
            Slowing down the rebalancing schedule from Daily to **{cadence_choice}** reduced XGBoost's annual turnover from **17.31 down to {sim_results['XGBoost']['turnover']:.2f}**, cutting transaction fee drag by more than half and lifting its Net Sharpe ratio to **{sim_results['XGBoost']['sharpe_net']:.3f}**!
            """)
        elif ema_smoothing > 1:
            st.success(f"""
            **Why did Forecast Smoothing improve Net Sharpe?**  
            Applying a **{ema_smoothing}-day EMA filter** smoothed day-to-day volatility jumps, reducing turnover from **12.72 down to {sim_results['Random Forest']['turnover']:.2f}** and preventing destructive fee burn.
            """)
        elif total_cost_input > 20:
            st.warning(f"""
            **Why is GARCH outperforming under {total_cost_input} bps friction?**  
            Under heavy transaction costs, high-turnover ML models suffer severe fee drag. GARCH rebalances only **{sim_results['GARCH']['turnover']:.2f} times per year**, paying almost zero friction.
            """)
        else:
            st.info("""
            **Research Baseline Conditions:** Under daily rebalancing and standard 15 bps friction, GARCH wins because its turnover is only **2.47** (cost drag of **-0.43%**) compared to XGBoost's turnover of **17.31** (cost drag of **-2.96%**).
            """)

        # Performance Table
        st.markdown("### 📋 Detailed Simulation Metrics Table")
        table_rows = []
        for name, data in sim_results.items():
            table_rows.append({
                "Strategy": name,
                "Net Sharpe ⓘ": f"{data['sharpe_net']:.3f}",
                "Net CAGR (%) ⓘ": f"{data['cagr_net']*100:.2f}%",
                "Gross CAGR (%) ⓘ": f"{data['cagr_gross']*100:.2f}%",
                "Annual Turnover ⓘ": f"{data['turnover']:.2f}",
                "Cost Drag (%) ⓘ": f"{data['cost_drag']*100:.2f}%",
                "Max Drawdown (%) ⓘ": f"{data['max_dd']*100:.2f}%"
            })
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True)

    # =========================================================================
    # TAB 5: FINAL VERDICT (Improvement 9)
    # =========================================================================
    with tab5:
        st.subheader("Final Research Verdict & Viva Summary")

        # 3 Verdict Cards
        v1, v2, v3 = st.columns(3)
        with v1:
            st.markdown("""
            <div class="verdict-card" style="border-top: 4px solid #059669;">
                <h4 style="color:#059669; margin:0;">Statistical Forecast Winner</h4>
                <h2 style="margin:10px 0; color:#0F172A;">Random Forest</h2>
                <p style="margin:0; font-size:1.1rem; color:#475569;"><b>RMSE = 0.08497</b></p>
                <p style="font-size:0.85rem; color:#64748B; margin-top:5px;">Lowest quadratic & QLIKE loss (p < 0.01)</p>
            </div>
            """, unsafe_allow_html=True)
        with v2:
            st.markdown("""
            <div class="verdict-card" style="border-top: 4px solid #2563EB;">
                <h4 style="color:#2563EB; margin:0;">Best Daily Portfolio Strategy</h4>
                <h2 style="margin:10px 0; color:#0F172A;">GARCH(1,1)</h2>
                <p style="margin:0; font-size:1.1rem; color:#475569;"><b>Net Sharpe = 0.490</b></p>
                <p style="font-size:0.85rem; color:#64748B; margin-top:5px;">Lowest turnover (2.47) & lowest cost drag (-0.43%)</p>
            </div>
            """, unsafe_allow_html=True)
        with v3:
            st.markdown("""
            <div class="verdict-card" style="border-top: 4px solid #7C3AED;">
                <h4 style="color:#7C3AED; margin:0;">Best Overall Implementation</h4>
                <h2 style="margin:10px 0; color:#0F172A;">XGBoost (Biweekly)</h2>
                <p style="margin:0; font-size:1.1rem; color:#475569;"><b>Net Sharpe = 0.558</b></p>
                <p style="font-size:0.85rem; color:#64748B; margin-top:5px;">Net CAGR = 17.00% under 10-day rebalancing</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # Master Defense Conclusion
        st.markdown("### 🎓 Master Defense Takeaway")
        st.markdown("""
        > **"Better forecasting accuracy does not automatically produce better investment performance. Portfolio implementation policies (rebalancing cadence and smoothing) and transaction costs determine whether forecasting improvements translate into real-world economic value."**
        """)

        st.markdown("""
        1. **Forecasting Space:** Machine learning captures non-linear feature interactions in returns, ranges, and volume, achieving double-digit error reductions over classical econometric models ($p < 0.01$).
        2. **Execution Space:** In daily volatility targeting, high forecast responsiveness causes hyperactive portfolio rebalancing (Annual Turnover: 12.7 to 17.3), generating 2.14% to 2.96% in annual transaction friction.
        3. **The Solution:** Adjusting implementation policies from daily to weekly/biweekly rebalancing cuts turnover by >50%, unlocking the true power of machine learning and lifting Net Sharpe to **0.558**.
        """)


if __name__ == "__main__":
    main()
