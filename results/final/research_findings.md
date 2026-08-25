# Consolidated Final Research Findings

**Project Title:** A Comparative and Hybrid Framework for Financial Volatility Forecasting: Statistical Accuracy versus Economic Utility  
**Sample Period:** 2010 – 2024 (NIFTY 50 Daily OHLCV Series, $N = 997$ Out-of-Sample Evaluation Days)

---

## 1. CONFIRMED BY EMPIRICAL DATA (Primary Research Hypotheses)

### 1.1 Non-Linear Machine Learning Significantly Outperforms Classical GARCH Statistically
- **Finding:** Random Forest, XGBoost, and the proposed Hybrid GARCH+ML model reduce out-of-sample forecast Root Mean Squared Error (RMSE) by **14.5% to 16.1%** relative to the 20-day Historical Volatility baseline.
- **Statistical Significance:** Diebold-Mariano tests (with Harvey-Leybourne-Newbold autocorrelation correction) reject the null hypothesis of equal predictive accuracy between GARCH and ML models at the **1% significance level** ($p = 0.0044$ for Random Forest, $p = 0.0025$ for XGBoost, $p = 0.0031$ for Hybrid).
- **Mincer-Zarnowitz Regression:** ML models achieve higher explanatory power ($R^2 = 33.5\%$ for Random Forest, $31.1\%$ for XGBoost) compared to standalone GARCH ($R^2 = 8.2\%$).

### 1.2 The Turnover-Friction Dilemma in Volatility Targeting
- **Finding:** In live dynamic volatility targeting (15% target vol), higher ML forecast responsiveness generates an annualized portfolio turnover of **12.72 to 17.31**, compared to just **2.47** for GARCH.
- **Transaction Cost Drag:** At standard institutional friction (10 bps fee + 5 bps slippage), ML strategies lose **2.14% to 2.96% in annualized net CAGR** solely to transaction costs.
- **Net Sharpe Leadership:** GARCH preserves its risk-adjusted performance, achieving the **highest Net Sharpe ratio (0.490)** and highest Net CAGR (14.85%), outperforming XGBoost (0.416 Net Sharpe) and Hybrid (0.321 Net Sharpe).

---

## 2. SUPPORTED BUT CONTEXT-DEPENDENT

### 2.1 Transaction Cost Sensitivity & Breakeven Crossover
- In a near-zero friction environment ($\le 5$ bps total fee + slippage), XGBoost achieves a Net Sharpe of **0.486**, closely matching GARCH (0.500).
- As transaction friction rises to $\ge 15$ bps, GARCH's low turnover advantage widens dramatically, with GARCH outperforming ML models by over **0.10 to 0.40 Sharpe units**.

### 2.2 Volatility Regime Asymmetry
- **Low Volatility Regimes (Tercile 1):** ML models exhibit strong error reductions (Relative RMSE $\approx 0.67$ to $0.69$).
- **High Volatility / Crash Regimes (Tercile 3):** Forecast errors increase across all models. Random Forest and Hybrid models maintain the lowest MAE (0.0798 and 0.0768) and lowest QLIKE loss.

---

## 3. EXPLORATORY EXTENSIONS

### 3.1 Rebalancing Frequency Mitigation
- Moving from Daily to **Weekly (5-day)** rebalancing reduces XGBoost turnover from **17.31 down to 7.11**, lifting its Net Sharpe from **0.416 up to 0.530** (surpassing Daily GARCH).
- **Biweekly (10-day)** rebalancing further reduces turnover to **4.66** and raises Net Sharpe to **0.558**.

### 3.2 Volatility Forecast Smoothing (EMA Filtering)
- Applying an Exponential Moving Average (EMA) smoothing span of 5 to 10 days to Random Forest forecasts reduces turnover from **12.72 down to 4.22**, improving Net Sharpe from **0.371 up to 0.436**.

---

## 4. FUTURE RESEARCH DIRECTIONS

1. **Explicit Turnover-Regularized ML Loss Functions**: Training machine learning models with a dual loss function penalizing both forecast error and predicted position turnover.
2. **Multi-Market External Validation**: Testing the same framework on global equity indices (S&P 500, FTSE 100) and emerging market peers.
3. **High-Frequency Intraday Realized Measures**: Utilizing 5-minute intraday realized variance / bipower variation targets to enhance signal-to-noise ratios.
