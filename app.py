"""
Quantitative Volatility Forecasting & Economic Utility Dashboard
================================================================
An institutional-grade research application structured into exactly 5 cohesive sections:
  1. OVERVIEW (Title, Research Question, Flowchart, Headline Winners, Core Takeaway)
  2. MARKET & DATA (NIFTY 50 Dynamics, Returns, Volatility Clustering, Why it matters)
  3. FORECASTING (5 Model Cards, Out-of-Sample Tracking, Loss Table, DM Significance Expander)
  4. PORTFOLIO SIMULATION (Frozen Baseline vs Interactive, Flow Diagram, 4 Controls,
                          Live Backtest, Turnover Paradox Breakdown, Exploratory Sensitivity & Cadence)
  5. FINAL VERDICT (Executive Viva Presentation: 3 Verdict Cards, Core Axioms, Conclusion & Limitations)
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
    page_title="Financial Volatility Forecasting | Quantitative Research Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Institutional CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.2rem;
    }
    .research-question-box {
        font-size: 1.15rem;
        font-weight: 500;
        color: #1E40AF;
        background-color: #EFF6FF;
        border-left: 4px solid #3B82F6;
        padding: 14px 20px;
        border-radius: 4px;
        margin: 15px 0 25px 0;
    }
    .baseline-card {
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
    .core-axiom {
        font-size: 1.25rem;
        font-weight: 700;
        color: #1E293B;
        text-align: center;
        background-color: #F1F5F9;
        padding: 12px;
        border-radius: 6px;
        margin: 15px 0;
        border: 1px dashed #94A3B8;
    }
    .badge-primary {
        background-color: #DBEAFE;
        color: #1E40AF;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-exploratory {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
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
    """
    Live simulator for volatility targeting across all models with interactive parameters.
    Preserves exact 1-day lagged execution w_{t-1} and proportional turnover friction.
    """
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
    bh_running_max = np.maximum.accumulate(bh_cum)
    bh_dd = (bh_cum - bh_running_max) / bh_running_max
    results["Buy & Hold"] = {
        "cum_equity": bh_cum,
        "drawdown": bh_dd,
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
            "drawdown": drawdowns,
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

    # Header & Subheader
    st.markdown('<div class="main-header">A Comparative & Hybrid Framework for Volatility Forecasting</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Statistical Accuracy versus Economic Utility in Financial Volatility Modeling</div>', unsafe_allow_html=True)

    # 4 INTERACTIVE CONTROLS IN SIDEBAR (Part 6)
    st.sidebar.header("🕹️ Interactive Simulation Controls")
    
    scenario = st.sidebar.selectbox(
        "Scenario Presets:",
        options=[
            "📌 Original Paper Baseline (Daily, 15 bps)",
            "🏆 Optimal Rebalancing (Biweekly, 15 bps)",
            "⚡ Zero-Friction Theoretical (Daily, 0 bps)",
            "⚠️ Heavy Friction Stress Test (Daily, 30 bps)",
            "⚙️ Custom Parameters"
        ],
        index=0,
        help="Quick configuration presets for testing alternative implementation scenarios."
    )

    if "Original Paper Baseline" in scenario:
        def_vol, def_cost, def_cadence, def_ema = 15, 15, "Daily", 1
    elif "Optimal Rebalancing" in scenario:
        def_vol, def_cost, def_cadence, def_ema = 15, 15, "Biweekly / 10-day", 1
    elif "Zero-Friction" in scenario:
        def_vol, def_cost, def_cadence, def_ema = 15, 0, "Daily", 1
    elif "Heavy Friction" in scenario:
        def_vol, def_cost, def_cadence, def_ema = 15, 30, "Daily", 1
    else:
        def_vol, def_cost, def_cadence, def_ema = 15, 15, "Daily", 1

    st.sidebar.markdown("---")

    target_vol_input = st.sidebar.slider(
        "Target Volatility ⓘ",
        min_value=5, max_value=25, value=def_vol, step=1, format="%d%%",
        help="Annualized target volatility level for position sizing (Default: 15%)."
    ) / 100.0

    total_cost_input = st.sidebar.slider(
        "Total Transaction Friction ⓘ",
        min_value=0, max_value=50, value=def_cost, step=1, format="%d bps",
        help="Total friction per unit position turnover (fee + slippage in basis points. 1 bps = 0.01%)."
    )

    cadence_options = ["Daily", "Weekly / 5-day", "Biweekly / 10-day"]
    cadence_choice = st.sidebar.selectbox(
        "Rebalancing Cadence ⓘ",
        options=cadence_options,
        index=cadence_options.index(def_cadence),
        help="How frequently portfolio weights are updated to match new forecasts."
    )
    cadence_map = {"Daily": 1, "Weekly / 5-day": 5, "Biweekly / 10-day": 10}
    cadence_days = cadence_map[cadence_choice]

    ema_smoothing = st.sidebar.slider(
        "Forecast Smoothing (EMA) ⓘ",
        min_value=1, max_value=10, value=def_ema, format="%d days",
        help="Exponential moving average filter applied to raw forecasts. 1d = Raw Forecast."
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📚 Key Terminology Guide")
    st.sidebar.markdown("""
    - **Volatility ⓘ:** Standard deviation of price returns measuring market turbulence.
    - **Basis Point (bps) ⓘ:** 1 bps = 0.01% = 0.0001.
    - **RMSE ⓘ:** Root Mean Squared Error (heavily penalizes large forecast outliers).
    - **MAE ⓘ:** Mean Absolute Error (average linear forecast discrepancy).
    - **QLIKE ⓘ:** Quasi-Likelihood asymmetric scale-invariant loss.
    - **Sharpe Ratio ⓘ:** Excess return generated per unit of realized volatility.
    - **CAGR ⓘ:** Compound Annual Growth Rate.
    - **Turnover ⓘ:** Frequency of total portfolio position changes per year.
    - **Drawdown ⓘ:** Peak-to-trough percentage decline in portfolio wealth.
    """)

    # 5 EXACT TABS (Part 2)
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "1. OVERVIEW",
        "2. MARKET & DATA",
        "3. FORECASTING",
        "4. PORTFOLIO SIMULATION",
        "5. FINAL VERDICT"
    ])

    # =========================================================================
    # TAB 1: OVERVIEW (Part 3)
    # =========================================================================
    with tab1:
        st.subheader("Executive Research Overview")

        st.markdown("""
        <div class="research-question-box">
            <b>Core Research Question:</b><br/>
            "Can machine-learning models improve volatility forecasts, and do those improvements translate into better portfolio performance after transaction costs?"
        </div>
        """, unsafe_allow_html=True)

        # Visual Workflow Diagram
        st.markdown("### 🔄 End-to-End Project Workflow")
        st.code("""
NIFTY 50 DATA (2010–2024, N = 3,679 Days)
        ↓
VOLATILITY (Forward 5-Day Realized Volatility Target)
        ↓
5 FORECASTING MODELS (Historical, GARCH, Random Forest, XGBoost, Hybrid)
        ↓
FORECAST ACCURACY (RMSE, MAE, QLIKE, Diebold-Mariano Tests)
        ↓
VOLATILITY-TARGETED PORTFOLIO (15% Dynamic Risk Target)
        ↓
TRANSACTION COSTS (15 bps Total Execution Friction)
        ↓
ECONOMIC PERFORMANCE (Net Sharpe, Net CAGR, Annual Turnover)
        ↓
FINAL CONCLUSION (Forecast Model ≠ Implementation Policy)
        """, language="text")

        st.markdown("---")

        # 3 Headline Result Cards
        st.markdown("### 🏆 Headline Empirical Results")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("""
            <div class="verdict-card" style="border-top: 4px solid #059669;">
                <span class="badge-primary">Primary Statistical Winner</span>
                <h3 style="margin:8px 0; color:#0F172A;">Random Forest</h3>
                <h4 style="margin:0; color:#059669;">RMSE = 0.08497</h4>
                <p style="font-size:0.85rem; color:#64748B; margin-top:6px;">16.1% Error Reduction vs Baseline (p < 0.01)</p>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown("""
            <div class="verdict-card" style="border-top: 4px solid #2563EB;">
                <span class="badge-primary">Primary Daily Economic Winner</span>
                <h3 style="margin:8px 0; color:#0F172A;">GARCH(1,1)</h3>
                <h4 style="margin:0; color:#2563EB;">Net Sharpe = 0.490</h4>
                <p style="font-size:0.85rem; color:#64748B; margin-top:6px;">Lowest Daily Turnover (2.47) & Low Cost Drag (-0.43%)</p>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown("""
            <div class="verdict-card" style="border-top: 4px solid #7C3AED;">
                <span class="badge-exploratory">Best Explored Implementation</span>
                <h3 style="margin:8px 0; color:#0F172A;">XGBoost (Biweekly)</h3>
                <h4 style="margin:0; color:#7C3AED;">Net Sharpe = 0.558</h4>
                <p style="font-size:0.85rem; color:#64748B; margin-top:6px;">Net CAGR = 17.00% under 10-day rebalancing cadence</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br/>", unsafe_allow_html=True)
        st.info("💡 **Core Academic Finding:** Machine learning improves statistical forecast accuracy, but portfolio implementation and transaction costs determine whether those improvements create economic value.")

    # =========================================================================
    # TAB 2: MARKET & DATA (Part 4)
    # =========================================================================
    with tab2:
        st.subheader("Market Benchmark & Empirical Stylized Facts")
        st.markdown("Evaluating 15 years of daily trading data for India's large-cap benchmark **NIFTY 50 Index (`^NSEI`)** from **2010 to 2024** ($N = 3,679$ trading days).")

        # Summary statistics cards
        c_m1, c_m2, c_m3, c_m4 = st.columns(4)
        c_m1.metric("Total Trading Days", "3,679 Days", help="Continuous monotonic trading sessions (2010–2024)")
        c_m2.metric("Return Kurtosis ⓘ", "8.941", delta="Heavy Tails (p < 0.0001)", help="Strong leptokurtosis indicating frequent extreme market returns")
        c_m3.metric("Return Skewness ⓘ", "-0.428", delta="Crash Asymmetry", help="Negative skewness reflecting faster downside market drops")
        c_m4.metric("Worst Historical Crash", "-13.90%", delta="March 23, 2020 (COVID)", help="Single-day market drop during global COVID dislocation")

        if not mdf.empty:
            fig_m = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, subplot_titles=["NIFTY 50 Historical Index Price (INR)", "Daily Continuous Log Returns"])
            fig_m.add_trace(go.Scatter(x=mdf["Date"], y=mdf["Close"], mode="lines", name="Close Price", line=dict(color="#2563EB")), row=1, col=1)
            fig_m.add_trace(go.Scatter(x=mdf["Date"], y=mdf["log_return"], mode="lines", name="Log Return", line=dict(color="#DC2626", width=0.8)), row=2, col=1)
            fig_m.update_layout(template="plotly_white", showlegend=False, height=460)
            st.plotly_chart(fig_m, use_container_width=True)

        st.markdown("""
        **Why does this matter?**  
        Financial returns contain heavy tails (Kurtosis = 8.941) and volatility clustering (quiet periods followed by sudden turbulent storm clusters). These stylized facts mean static risk models fail, making dynamic forward volatility forecasting essential for institutional risk management.
        """)

    # =========================================================================
    # TAB 3: FORECASTING (Part 5)
    # =========================================================================
    with tab3:
        st.subheader("Statistical Forecast Accuracy ($N = 997$ Out-of-Sample Days)")

        # 5 Simple Model Cards
        st.markdown("### 🤖 The 5 Forecasting Models")
        m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
        with m_col1:
            st.markdown("**Historical Vol**<br/><span style='font-size:0.85rem; color:#475569;'>Uses recent historical volatility as a simple baseline.</span>", unsafe_allow_html=True)
        with m_col2:
            st.markdown("**GARCH(1,1)**<br/><span style='font-size:0.85rem; color:#475569;'>Models how volatility changes and persists over time.</span>", unsafe_allow_html=True)
        with m_col3:
            st.markdown("**Random Forest**<br/><span style='font-size:0.85rem; color:#475569;'>Uses many decision trees to learn nonlinear relationships.</span>", unsafe_allow_html=True)
        with m_col4:
            st.markdown("**XGBoost**<br/><span style='font-size:0.85rem; color:#475569;'>Builds boosted trees sequentially to improve prediction errors.</span>", unsafe_allow_html=True)
        with m_col5:
            st.markdown("**Hybrid Model**<br/><span style='font-size:0.85rem; color:#475569;'>Combines GARCH information with machine-learning features.</span>", unsafe_allow_html=True)

        st.markdown("---")

        # Interactive Forecast Tracking Plot
        st.markdown("### 📈 Out-of-Sample Volatility Forecast Tracking")
        if not fdf.empty:
            fig_fc = go.Figure()
            fig_fc.add_trace(go.Scatter(x=fdf["Date"], y=fdf["Actual"], mode="lines", name="Realized Volatility (Target)", line=dict(color="#0F172A", width=1.5, dash="dot")))
            if "Random Forest" in fdf.columns:
                fig_fc.add_trace(go.Scatter(x=fdf["Date"], y=fdf["Random Forest"], mode="lines", name="Random Forest", line=dict(color="#059669", width=2.0)))
            if "XGBoost" in fdf.columns:
                fig_fc.add_trace(go.Scatter(x=fdf["Date"], y=fdf["XGBoost"], mode="lines", name="XGBoost", line=dict(color="#DC2626", width=2.0)))
            if "GARCH" in fdf.columns:
                fig_fc.add_trace(go.Scatter(x=fdf["Date"], y=fdf["GARCH"], mode="lines", name="GARCH(1,1)", line=dict(color="#2563EB", width=2.0)))
            fig_fc.update_layout(template="plotly_white", hovermode="x unified", height=420, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig_fc, use_container_width=True)

        # Statistical Loss Metrics Table
        st.markdown("### 📊 Primary Statistical Loss Metrics *(Lower is Better)*")
        stat_table_data = [
            {"Model": "Random Forest", "RMSE (Quadratic Loss) ⓘ": "0.08497 (Winner)", "MAE (Absolute Loss) ⓘ": "0.05151", "QLIKE (Scale-Invariant) ⓘ": "0.47287 (Winner)", "Relative RMSE": "-16.1%"},
            {"Model": "XGBoost", "RMSE (Quadratic Loss) ⓘ": "0.08594", "MAE (Absolute Loss) ⓘ": "0.05072 (Winner)", "QLIKE (Scale-Invariant) ⓘ": "0.48869", "Relative RMSE": "-15.1%"},
            {"Model": "Hybrid GARCH+ML", "RMSE (Quadratic Loss) ⓘ": "0.08657", "MAE (Absolute Loss) ⓘ": "0.05266", "QLIKE (Scale-Invariant) ⓘ": "0.47664", "Relative RMSE": "-14.5%"},
            {"Model": "Historical Vol (20d)", "RMSE (Quadratic Loss) ⓘ": "0.10128", "MAE (Absolute Loss) ⓘ": "0.06036", "QLIKE (Scale-Invariant) ⓘ": "0.56519", "Relative RMSE": "Benchmark"},
            {"Model": "GARCH(1,1)", "RMSE (Quadratic Loss) ⓘ": "0.10497", "MAE (Absolute Loss) ⓘ": "0.06320", "QLIKE (Scale-Invariant) ⓘ": "0.78001", "Relative RMSE": "+3.6%"}
        ]
        st.dataframe(pd.DataFrame(stat_table_data), use_container_width=True)

        st.info("💡 **Statistical Takeaway:** Model rankings depend on the loss function evaluated: **Random Forest** achieves the lowest RMSE and QLIKE, while **XGBoost** achieves the lowest MAE. All ML models achieve statistically significant error reductions over GARCH ($p < 0.01$).")

        # Diebold-Mariano Matrix inside Expander
        with st.expander("🔬 View Diebold-Mariano Statistical Significance Matrix ($p$-values)"):
            st.markdown("Pairwise Harvey-Leybourne-Newbold adjusted Diebold-Mariano test $p$-values across 997 out-of-sample test days ($h=5$):")
            if not dm_matrix_df.empty:
                st.dataframe(dm_matrix_df, use_container_width=True)
                st.caption("Values below 0.01 indicate statistically significant predictive difference at the 1% level.")

    # =========================================================================
    # TAB 4: PORTFOLIO SIMULATION (Parts 6, 7, 8)
    # =========================================================================
    with tab4:
        st.subheader("Dynamic Volatility-Targeting Portfolio Simulation")

        # Frozen Baseline Box (Part 6)
        st.markdown("""
        <div class="baseline-card">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                <span class="badge-primary">FROZEN RESEARCH BASELINE</span>
                <span style="color:#64748B; font-size:0.85rem;">Immutable Out-of-Sample Reference</span>
            </div>
            <div style="display:flex; justify-content:space-between; flex-wrap:wrap; font-size:0.95rem; color:#1E293B;">
                <span><b>Target Volatility:</b> 15%</span>
                <span><b>Transaction Friction:</b> 15 bps (10 bps fee + 5 bps slippage)</span>
                <span><b>Rebalancing Cadence:</b> Daily</span>
                <span><b>Forecast Smoothing:</b> None (Raw)</span>
                <span><b>Test Observations:</b> 997 Days</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Portfolio Flow Diagram (Part 6)
        st.markdown("### ⚙️ How Volatility Forecasts Become a Portfolio Strategy")
        st.code("""
VOLATILITY FORECAST (sigma_hat_t)
        ↓
COMPARE WITH TARGET RISK: Position Weight w_t = min(0.15 / sigma_hat_t, 1.5)
        ↓
1-DAY EXECUTION LAG: Position executed at w_{t-1} for tomorrow's return (Zero Lookahead)
        ↓
PORTFOLIO RETURN: Gross Return = w_{t-1} * Return_t
        ↓
TURNOVER: Absolute position change |w_t - w_{t-1}|
        ↓
TRANSACTION COST: Deduct (Turnover * 15 bps Friction)
        ↓
NET ECONOMIC PERFORMANCE: Net Sharpe & Net CAGR
        """, language="text")
        st.caption("If predicted volatility is high, the strategy reduces exposure. If predicted volatility is low, it increases exposure, subject to the 1.5x leverage cap.")

        st.markdown("---")

        # Run Live Simulation based on sidebar sliders
        sim_results = simulate_volatility_targeting(
            fdf,
            target_vol=target_vol_input,
            total_cost_bps=total_cost_input,
            cadence=cadence_days,
            ema_span=ema_smoothing
        )

        # Compact Live Metric Cards
        st.markdown(f"#### 📊 Live Strategy Performance *(Friction: {total_cost_input} bps | Cadence: {cadence_choice} | Target Vol: {target_vol_input*100:.0f}%)*")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            garch_s = sim_results["GARCH"]["sharpe_net"] if "GARCH" in sim_results else 0.0
            st.metric("GARCH Net Sharpe ⓘ", f"{garch_s:.3f}", delta=f"Turnover: {sim_results['GARCH']['turnover']:.2f}")
        with col2:
            xgb_s = sim_results["XGBoost"]["sharpe_net"] if "XGBoost" in sim_results else 0.0
            st.metric("XGBoost Net Sharpe ⓘ", f"{xgb_s:.3f}", delta=f"Turnover: {sim_results['XGBoost']['turnover']:.2f}")
        with col3:
            rf_s = sim_results["Random Forest"]["sharpe_net"] if "Random Forest" in sim_results else 0.0
            st.metric("Random Forest Net Sharpe ⓘ", f"{rf_s:.3f}", delta=f"Turnover: {sim_results['Random Forest']['turnover']:.2f}")
        with col4:
            bh_s = sim_results["Buy & Hold"]["sharpe_net"]
            st.metric("Buy & Hold Sharpe ⓘ", f"{bh_s:.3f}", delta="Passive 100%")

        # Interactive Model Filter
        all_models = list(sim_results.keys())
        selected_models = st.multiselect("🎛️ Filter Displayed Strategies:", options=all_models, default=all_models)

        # Interactive Cumulative Equity and Drawdown Charts
        colors = {
            "GARCH": "#2563EB",
            "XGBoost": "#DC2626",
            "Random Forest": "#059669",
            "Hybrid GARCH+ML": "#7C3AED",
            "Historical Volatility": "#D97706",
            "Buy & Hold": "#64748B"
        }

        fig_perf = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.07, row_heights=[0.7, 0.3], subplot_titles=["Cumulative Net Wealth ($1.00 Base)", "Underwater Drawdown (%)"])
        for m in selected_models:
            data = sim_results[m]
            fig_perf.add_trace(go.Scatter(
                x=data["dates"], y=data["cum_equity"], mode="lines", name=m,
                line=dict(color=colors.get(m, "#333"), width=2.2 if m in ["GARCH", "XGBoost"] else 1.5)
            ), row=1, col=1)
            fig_perf.add_trace(go.Scatter(
                x=data["dates"], y=data["drawdown"] * 100.0, mode="lines", name=m, showlegend=False,
                line=dict(color=colors.get(m, "#333"), width=1.3)
            ), row=2, col=1)
        fig_perf.update_layout(template="plotly_white", hovermode="x unified", height=520, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig_perf, use_container_width=True)

        # Ranked Strategy Performance Table with Badges
        st.markdown("### 📋 Ranked Performance Table")
        table_rows = []
        for name, data in sim_results.items():
            table_rows.append({
                "Strategy": name,
                "Net Sharpe ⓘ": data["sharpe_net"],
                "Net CAGR (%) ⓘ": data["cagr_net"] * 100.0,
                "Gross CAGR (%)": data["cagr_gross"] * 100.0,
                "Annual Turnover ⓘ": data["turnover"],
                "Cost Drag (%) ⓘ": data["cost_drag"] * 100.0,
                "Max Drawdown (%) ⓘ": data["max_dd"] * 100.0
            })
        df_display = pd.DataFrame(table_rows).sort_values("Net Sharpe ⓘ", ascending=False).reset_index(drop=True)
        rank_badges = ["🥇 1st", "🥈 2nd", "🥉 3rd", "4th", "5th", "6th"]
        df_display.insert(0, "Rank", rank_badges[:len(df_display)])
        
        df_formatted = df_display.copy()
        df_formatted["Net Sharpe ⓘ"] = df_formatted["Net Sharpe ⓘ"].map("{:.3f}".format)
        df_formatted["Net CAGR (%) ⓘ"] = df_formatted["Net CAGR (%) ⓘ"].map("{:+.2f}%".format)
        df_formatted["Gross CAGR (%)"] = df_formatted["Gross CAGR (%)"].map("{:+.2f}%".format)
        df_formatted["Annual Turnover ⓘ"] = df_formatted["Annual Turnover ⓘ"].map("{:.2f}".format)
        df_formatted["Cost Drag (%) ⓘ"] = df_formatted["Cost Drag (%) ⓘ"].map("{:.2f}%".format)
        df_formatted["Max Drawdown (%) ⓘ"] = df_formatted["Max Drawdown (%) ⓘ"].map("{:.2f}%".format)
        st.dataframe(df_formatted, use_container_width=True)

        # 1-Click CSV Export
        export_df = pd.DataFrame({"Date": fdf["Date"]})
        for m, d in sim_results.items():
            export_df[f"{m}_Equity"] = d["cum_equity"]
            export_df[f"{m}_Return"] = d["daily_returns"]
        csv_data = export_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="💾 Download Simulated Portfolio Returns (CSV)",
            data=csv_data,
            file_name="simulated_volatility_targeting_results.csv",
            mime="text/csv"
        )

        st.markdown("---")

        # DEDICATED TURNOVER / ECONOMIC UTILITY EXPLANATION (Part 7)
        st.markdown("### ❓ Why does forecast accuracy not equal portfolio performance?")
        st.code("""
FORECAST → POSITION → REBALANCING → TURNOVER → TRANSACTION COST → NET PERFORMANCE
        """, language="text")

        col_p1, col_p2 = st.columns(2)
        with col_p1:
            st.markdown("""
            **Primary Daily Implementation Comparison:**
            - **GARCH(1,1):** Annual Turnover = **2.47** | Net Sharpe = **0.490** | Cost Drag = **-0.43%**
            - **XGBoost:** Annual Turnover = **17.31** | Net Sharpe = **0.416** | Cost Drag = **-2.96%**
            """)
        with col_p2:
            st.info("""
            **The Core Disconnect Explained:**  
            XGBoost provides more accurate forecasts, but its daily portfolio implementation generates **7x higher turnover**. High-frequency weight adjustments cause transaction costs to consume **2.96% of annual wealth**, pulling its Net Sharpe below GARCH.
            """)

        st.markdown("---")

        # REBALANCING & SENSITIVITY EXTENSIONS (Part 8)
        st.markdown("### 🔬 Exploratory Implementation Extensions")
        st.caption("Investigating how implementation policy (rebalancing cadence and fee sensitivity) reconciles the gap between ML accuracy and net alpha.")

        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.markdown("#### 💰 Transaction Cost Grid Response (0 to 50 bps)")
            if not cost_df.empty:
                fig_cost = px.line(
                    cost_df,
                    x="Cost_bps",
                    y="Net_Sharpe",
                    color="Model",
                    markers=True,
                    labels={"Cost_bps": "Brokerage Fee (bps)", "Net_Sharpe": "Net Sharpe Ratio"}
                )
                fig_cost.update_layout(template="plotly_white", height=350)
                st.plotly_chart(fig_cost, use_container_width=True)
                st.caption("Break-even crossover occurs at ~0-2 bps fee. Above 2 bps, GARCH's low turnover gives it higher Net Sharpe.")

        with col_s2:
            st.markdown("#### ⏱️ Rebalancing Cadence Comparison")
            if not rebal_df.empty:
                fig_rebal = px.bar(
                    rebal_df,
                    x="Rebalancing",
                    y="Net_Sharpe",
                    color="Model",
                    barmode="group",
                    labels={"Rebalancing": "Rebalancing Cadence", "Net_Sharpe": "Net Sharpe Ratio"}
                )
                fig_rebal.update_layout(template="plotly_white", height=350)
                st.plotly_chart(fig_rebal, use_container_width=True)
                st.caption("Reducing rebalancing frequency lowers turnover, lifting XGBoost Net Sharpe from 0.416 (Daily) to 0.530 (Weekly) and 0.558 (Biweekly).")

    # =========================================================================
    # TAB 5: FINAL VERDICT (Part 9, 11, 13)
    # =========================================================================
    with tab5:
        st.subheader("Final Research Verdict & Academic Synthesis")

        # 3 Structured Verdict Sections (Part 9)
        col_v1, col_v2, col_v3 = st.columns(3)
        with col_v1:
            st.markdown("""
            <div class="verdict-card" style="border-top: 4px solid #059669;">
                <span class="badge-primary">SECTION 1: STATISTICAL WINNER</span>
                <h2 style="margin:10px 0; color:#0F172A;">Random Forest</h2>
                <h3 style="color:#059669; margin:0;">RMSE = 0.08497</h3>
                <p style="font-size:0.9rem; color:#64748B; margin-top:8px;">
                    <b>MAE:</b> 0.05151 | <b>QLIKE:</b> 0.47287<br/>
                    Statistically superior to GARCH (p = 0.0044)
                </p>
            </div>
            """, unsafe_allow_html=True)
        with col_v2:
            st.markdown("""
            <div class="verdict-card" style="border-top: 4px solid #2563EB;">
                <span class="badge-primary">SECTION 2: DAILY ECONOMIC WINNER</span>
                <h2 style="margin:10px 0; color:#0F172A;">GARCH(1,1)</h2>
                <h3 style="color:#2563EB; margin:0;">Net Sharpe = 0.490</h3>
                <p style="font-size:0.9rem; color:#64748B; margin-top:8px;">
                    <b>Net CAGR:</b> 14.85% | <b>Turnover:</b> 2.47<br/>
                    Cost Drag: -0.43% under daily 15 bps friction
                </p>
            </div>
            """, unsafe_allow_html=True)
        with col_v3:
            st.markdown("""
            <div class="verdict-card" style="border-top: 4px solid #7C3AED;">
                <span class="badge-exploratory">SECTION 3: BEST IMPLEMENTATION</span>
                <h2 style="margin:10px 0; color:#0F172A;">XGBoost (Biweekly)</h2>
                <h3 style="color:#7C3AED; margin:0;">Net Sharpe = 0.558</h3>
                <p style="font-size:0.9rem; color:#64748B; margin-top:8px;">
                    <b>Net CAGR:</b> 17.00% | <b>Turnover:</b> 4.66<br/>
                    10-Day rebalancing cadence cuts turnover by 73%
                </p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br/>", unsafe_allow_html=True)

        # Core Academic Axioms
        st.markdown('<div class="core-axiom">BETTER FORECASTING ≠ BETTER PORTFOLIO PERFORMANCE</div>', unsafe_allow_html=True)
        st.markdown('<div class="core-axiom">FORECAST MODEL ≠ PORTFOLIO IMPLEMENTATION POLICY</div>', unsafe_allow_html=True)

        # Final Academic Conclusion
        st.markdown("### 🎓 Academic Research Synthesis")
        st.markdown("""
        > **"Machine-learning models demonstrate superior statistical volatility forecasting accuracy in the evaluated out-of-sample period. However, high-frequency portfolio adjustments can create substantial turnover and transaction-cost drag. More conservative implementation policies, such as less frequent rebalancing or forecast smoothing, can improve the economic utility of machine-learning forecasts."**
        """)

        # Technical Methodology Expander (Part 11)
        with st.expander("📖 Technical Methodology Specification"):
            st.markdown("""
            - **Market & Sample:** NIFTY 50 Index (`^NSEI`), 2010–2024 ($N = 3,679$ daily sessions).
            - **Forecasting Target:** Forward 5-day realized volatility $RV_{t, t+5} = \sqrt{\sum_{i=1}^5 r_{t+i}^2} \times \sqrt{252/5}$.
            - **Validation Scheme:** Expanding-window walk-forward validation across 997 out-of-sample trading days.
            - **Portfolio Strategy:** 15% dynamic volatility targeting with weights $w_t = \min(\sigma_{target}/\hat{\sigma}_t, 1.5)$.
            - **Execution Integrity:** Strict 1-day execution lag ($w_{t-1} \cdot r_t$) to eliminate lookahead bias.
            - **Transaction Friction:** Proportional deduction $\Delta w_t \times 15\text{ bps}$ ($10\text{ bps}$ fee $+ 5\text{ bps}$ slippage).
            """)

        # Limitations Expander (Part 13)
        with st.expander("⚠️ Research Scope & Methodological Limitations"):
            st.markdown("""
            1. **Single Equity Benchmark:** Evaluation is conducted strictly on the NIFTY 50 index; cross-asset validation is required before generalising to global equities or fixed income.
            2. **Daily Sampling Frequency:** Evaluates daily OHLCV closing snapshots; intraday tick kernels or order book dynamics may alter signal-to-noise dynamics.
            3. **Linear Friction Assumptions:** Transaction costs are modeled as proportional fees and slippage; non-linear square-root market impact laws were not modeled.
            4. **Compliance Disclaimer:** All reported metrics are historical out-of-sample empirical research findings and do not represent guaranteed future investment performance.
            """)


if __name__ == "__main__":
    main()
