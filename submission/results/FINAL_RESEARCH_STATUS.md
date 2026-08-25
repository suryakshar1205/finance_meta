# Final Research Status and Academic Defense Summary

**Project Title:** A Comparative and Hybrid Framework for Financial Volatility Forecasting: Statistical Accuracy versus Economic Utility  
**Primary Dataset:** NIFTY 50 Index (`^NSEI`, National Stock Exchange of India, 2010–2024)  
**Out-of-Sample Window:** 997 Trading Days (2021–2024)  
**Auditor / Lead Researcher:** Quantitative Finance Research Division  
**Status:** Validated, Frozen, Claim-Audited, Publication-Grade

---

## A. What is Conclusively Established (Empirical Facts)

1. **Loss-Function-Dependent Statistical Forecasting Superiority:**
   - In statistical loss evaluation across 997 out-of-sample days, tree-based machine learning models outperform the classical econometric benchmark.
   - **Root Mean Squared Error (RMSE) Winner:** **Random Forest** ($\text{RMSE} = 0.08497$, a $16.1\%$ reduction versus the 20-day historical volatility baseline).
   - **Mean Absolute Error (MAE) Winner:** **XGBoost** ($\text{MAE} = 0.05072$, versus $0.06320$ for GARCH).
   - **Quasi-Likelihood (QLIKE) Winner:** **Random Forest** ($\text{QLIKE} = 0.47287$, versus $0.78001$ for GARCH).
   - Model rankings strictly depend on the loss function chosen.

2. **Statistical Significance under Autocorrelation-Corrected Diebold-Mariano Tests:**
   - Under the Harvey-Leybourne-Newbold (1997) multi-step Diebold-Mariano test, Random Forest ($DM = 2.851, p = 0.0044$), XGBoost ($DM = 3.035, p = 0.0025$), and Hybrid GARCH+ML ($DM = 2.962, p = 0.0031$) all reject the null hypothesis of equal forecast accuracy against GARCH(1,1) at the **1% significance level**.
   - GARCH(1,1) and the 20-day Historical Volatility benchmark are statistically indistinguishable ($DM = 0.351, p = 0.7253$).

3. **The Daily Volatility-Targeting Turnover Discrepancy:**
   - In dynamic volatility targeting (15% target risk), daily rebalancing of ML forecasts generates an annualized turnover of **12.72 (Random Forest)** and **17.31 (XGBoost)**, compared to just **2.47 for GARCH**.
   - Under standard institutional transaction costs (10 bps fee + 5 bps slippage), transaction friction consumes **2.14% to 2.96% in net CAGR** for ML models, while GARCH incurs a cost drag of only **0.43%**.

4. **GARCH Economic Resilience Under Daily Rebalancing:**
   - Because GARCH variance estimates mean-revert smoothly, GARCH achieves the **highest Net Sharpe ratio (0.490)** and highest Net CAGR ($14.85\%$) among all primary daily-rebalanced volatility-targeting strategies.

---

## B. What is Strongly Supported (Context-Dependent Findings)

1. **Rebalancing Cadence Optimization:**
   - Lengthening the rebalancing schedule from Daily to **Weekly (5-day)** cuts XGBoost turnover from $17.31$ down to $7.11$, raising its Net Sharpe from **0.416 up to 0.530** (surpassing Daily GARCH).
   - **Biweekly (10-day)** rebalancing further reduces XGBoost turnover to $4.66$ and increases Net Sharpe to **0.558** (Net CAGR $17.00\%$).
   - *Interpretation:* The results are consistent with the hypothesis that execution friction, rather than structural signal failure, explains the daily underperformance of responsive ML forecasts.

2. **Estimated Break-Even Transaction Cost:**
   - Under daily rebalancing, the estimated break-even transaction cost where XGBoost Net Sharpe falls below GARCH Net Sharpe is approximately **0 to 2 bps** of one-way transaction fee (5 to 7 bps total friction including 5 bps slippage). Above this threshold, GARCH's low turnover produces superior net returns.

3. **GARCH High QLIKE Investigation:**
   - GARCH exhibits an elevated QLIKE loss ($0.7800$) relative to ML ($0.473–0.489$). Because QLIKE penalizes variance under-predictions exponentially ($\sigma^2/h - \ln(\sigma^2/h) - 1$), GARCH's slow reaction to sudden volatility spikes is heavily penalized in statistical loss, yet this same smoothness prevents destructive portfolio churn in volatility targeting.

---

## C. What Remains Exploratory (Extensions & Modest Claims)

1. **Exponential Forecast Smoothing (EMA Filtering):**
   - Applying a 10-day EMA smoothing filter to Random Forest forecasts reduces annual turnover from **12.72 down to 4.22**, lifting Net Sharpe from **0.371 up to 0.436**.
   - *Status:* Exploratory research extension. Primary results retain raw forecasts to maintain clean separation between forecast models and implementation filters.

2. **Forecast Model vs. Portfolio Implementation Policy Distinction:**
   - A superior volatility forecast (e.g., Random Forest or XGBoost) is an input, whereas the trading strategy is determined by the **portfolio implementation policy** (rebalancing frequency, position inertia, smoothing).
   - *Status:* "Best volatility forecast" does not equal "Best trading strategy".

---

## D. What Should Be Future Research

1. **Explicit Turnover-Regularized Loss Functions:** Training machine learning models with joint loss functions: $\mathcal{L}(\hat{\sigma}) = \text{MSE}(\hat{\sigma}) + \gamma |\Delta w_t|$.
2. **Multi-Market & Cross-Asset Validation:** Testing on global equities (S&P 500, STOXX 50), commodities, and fixed income.
3. **High-Frequency Intraday Sampling:** Evaluating 5-minute realized kernels and jump-robust bipower variation measures.
4. **Multi-Asset Covariance Allocation:** Extending the framework from univariate volatility targeting to dynamic multi-asset covariance matrix optimization.

---

## E. Final Recommended Conclusion

The central empirical conclusion of this capstone is **not** that "Machine Learning is better" nor that "GARCH is better."

Instead, the study establishes that **statistical forecast accuracy and economic risk-management utility are distinct dimensions of model evaluation**:
1. Machine learning models effectively capture non-linear relationships in lagged market, return, volume, and volatility features, achieving statistically superior out-of-sample forecast accuracy ($p < 0.01$).
2. In live dynamic portfolio risk management, high forecast responsiveness induces severe trading turnover, rendering ML strategies vulnerable to transaction cost drag under unconstrained daily rebalancing.
3. Econometric GARCH models provide superior economic resilience under daily rebalancing due to structural parameter smoothness.
4. When implementation policies are adjusted (e.g., weekly rebalancing or forecast smoothing), quantitative managers can substantially mitigate turnover friction and translate ML predictive accuracy into net economic outperformance.
