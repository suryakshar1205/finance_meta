# Methodological Framework and Mathematical Specification

This document provides a comprehensive mathematical and algorithmic specification of the empirical research pipeline implemented in **`finance_meta`**.

---

## 7.1 Problem Definition
The primary research objective is to estimate and evaluate conditional financial volatility forecasts for the **NIFTY 50 Index (`^NSEI`)** across classical econometric, modern machine-learning, and hybrid paradigms, and to test whether statistical forecast improvements translate into economically meaningful risk-management utility after accounting for execution friction.

---

## 7.2 Volatility Forecasting Target
Given the information filtration $\mathcal{F}_t$ available at market close on trading day $t$, we seek to forecast the annualized forward realized volatility over the future $k$-day trading window $[t+1, t+k]$:

$$RV_{t, t+k} = \sqrt{\sum_{i=1}^k r_{t+i}^2} \times \sqrt{\frac{252}{k}}$$

where $r_{t+i} = \ln(P_{t+i} / P_{t+i-1})$ is the continuous daily log return.
- **Primary Forecast Horizon:** $k = 5$ trading days (1 weekly horizon).
- **Robustness Horizons:** $k \in \{1, 5, 10, 20\}$ trading days.

---

## 7.3 Historical Volatility Baseline
The baseline benchmark is the rolling sample standard deviation over a backward-looking window of $m = 20$ trading days:

$$\hat{\sigma}_{t}^{Hist} = \sqrt{252} \times \sqrt{\frac{1}{m-1} \sum_{j=0}^{m-1} \left(r_{t-j} - \bar{r}_t\right)^2}$$

---

## 7.4 Econometric Benchmark: GARCH(1,1)
The conditional mean and variance equations under Student's $t$ innovations are:

$$r_t = \mu + \epsilon_t, \quad \epsilon_t = \sigma_t z_t, \quad z_t \sim t(\nu)$$
$$\sigma_t^2 = \omega + \alpha \epsilon_{t-1}^2 + \beta \sigma_{t-1}^2$$

where $\omega > 0$, $\alpha \ge 0$, $\beta \ge 0$, and $\alpha + \beta < 1$.

The $k$-step forward cumulative annualized volatility forecast generated at time $t$ is:

$$\hat{\sigma}_{t, t+k}^{GARCH} = \sqrt{\frac{252}{k} \sum_{j=1}^k \mathbb{E}_t\left[\sigma_{t+j}^2\right]}$$

where multi-step expected variance follows the autoregressive recursion:
$$\mathbb{E}_t\left[\sigma_{t+j}^2\right] = \sigma_{uncond}^2 + (\alpha + \beta)^{j-1} \left(\sigma_{t+1}^2 - \sigma_{uncond}^2\right)$$

---

## 7.5 Random Forest Regression
Random Forest (Breiman, 2001) is an ensemble of $B = 200$ randomized, de-correlated decision trees:

$$\hat{\sigma}_{t, t+k}^{RF} = \frac{1}{B} \sum_{b=1}^B T_b\left(\mathbf{x}_t; \Theta_b\right)$$

- Trees are constructed on bootstrap samples using mean squared error splitting.
- Subsamples of features ($\sqrt{p}$) are considered at each node split to prevent tree correlation.

---

## 7.6 XGBoost (Gradient-Boosted Decision Trees)
XGBoost (Chen & Guestrin, 2016) builds an additive ensemble of $M = 200$ regression trees minimizing a regularized objective function:

$$\mathcal{L} = \sum_{i=1}^n l\left(y_i, \hat{y}_i\right) + \sum_{m=1}^M \left[\gamma T_m + \frac{1}{2} \lambda \sum_{j=1}^{T_m} w_{mj}^2\right]$$

- Learning rate shrinkage: $\eta = 0.03$.
- Subsample ratio: $0.80$; Column sample ratio: $0.80$.

---

## 7.7 Proposed Hybrid GARCH + Machine Learning Framework
The hybrid architecture extracts the recursive, in-sample GARCH conditional volatility estimate $\hat{\sigma}_{GARCH, t}$ and incorporates it as an explicit structural econometric predictor alongside the non-parametric feature matrix:

$$\hat{\sigma}_{t, t+k}^{Hybrid} = f_{XGBoost}\left(\hat{\sigma}_{GARCH, t}, \mathbf{x}_t^{features}\right)$$

This allows the gradient boosting tree ensemble to learn non-linear residual corrections and interaction effects between classical conditional heteroskedasticity and multi-horizon market features.

---

## 7.8 Feature Engineering (28 Leakage-Free Predictors)
All features are constructed strictly from information available at or prior to market close on day $t$:
1. **Lagged Log Returns:** $r_t, r_{t-1}, r_{t-2}, r_{t-4}, r_{t-9}, r_{t-19}$
2. **Rolling Historical Volatilities:** 5d, 10d, 20d, 30d, and 60d sample standard deviations ($\times \sqrt{252}$)
3. **Volatility Term Structure Ratios:** $\sigma_{5d}/\sigma_{20d}$, $\sigma_{20d}/\sigma_{60d}$
4. **Intraday Extreme-Value Range Estimator (Parkinson, 1980):**
   $$\sigma_{Parkinson, t}^2 = \frac{252}{4 \ln 2} \frac{1}{N} \sum_{i=0}^{N-1} \ln\left(\frac{High_{t-i}}{Low_{t-i}}\right)^2$$
5. **Intraday OHLC Estimator (Garman & Klass, 1980):**
   $$\sigma_{GK, t}^2 = \frac{252}{N} \sum_{i=0}^{N-1} \left[ 0.5 \ln\left(\frac{H_{t-i}}{L_{t-i}}\right)^2 - (2\ln 2 - 1) \ln\left(\frac{C_{t-i}}{O_{t-i}}\right)^2 \right]$$
6. **Price Momentum & Trend:** Distance to 20-day and 50-day moving averages, 5-day and 20-day returns.
7. **Volume Dynamics:** 1-day volume percentage change, 5/20 volume moving average ratio, 20-day volume volatility.

---

## 7.9 Walk-Forward Validation Engine
- **Evaluation Window:** 997 strictly out-of-sample trading days (January 2021 to December 2024).
- **Scheme:** Expanding historical window with an initial training size of $T_0 = 1,500$ days.
- **Refit Frequency:** Parameters (GARCH MLE coefficients, tree splits, and leaf weights) are re-estimated every 20 trading days (~monthly).
- **Daily Prediction:** Daily 1-step rolling forecasts are generated and stored out-of-sample.

---

## 7.10 Leakage Prevention Protocol
1. **Target Alignment:** The target $RV_{t, t+k}$ uses future returns $r_{t+1}, \dots, r_{t+k}$, while features $\mathbf{x}_t$ use past returns $r_t, r_{t-1}, \dots$.
2. **Temporal Ordering:** Zero future-data contamination is verified via lead-lag correlation assertions ($\text{Corr}(x_t, r_{t+1}) \approx 0$).
3. **No Global Pre-scaling:** Feature transformations are fitted strictly on past expanding training folds.

---

## 7.11 Forecast Evaluation Loss Metrics
1. **Root Mean Squared Error (RMSE):**
   $$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{t=1}^N \left(RV_t - \hat{\sigma}_t\right)^2}$$
2. **Mean Absolute Error (MAE):**
   $$\text{MAE} = \frac{1}{N} \sum_{t=1}^N |RV_t - \hat{\sigma}_t|$$
3. **Quasi-Likelihood Loss (QLIKE; Patton, 2011):**
   $$L_{QLIKE}\left(RV_t^2, \hat{\sigma}_t^2\right) = \frac{RV_t^2}{\hat{\sigma}_t^2} - \ln\left(\frac{RV_t^2}{\hat{\sigma}_t^2}\right) - 1$$

---

## 7.12 Diebold-Mariano Test with HLN Correction
To test the null hypothesis of equal predictive accuracy ($H_0: \mathbb{E}[d_t] = 0$ where $d_t = e_{1,t}^2 - e_{2,t}^2$):

$$DM = \frac{\bar{d}}{\sqrt{\hat{V}(\bar{d}) / N}}$$

where the spectral variance $\hat{V}(\bar{d})$ includes the **Harvey, Leybourne, and Newbold (1997)** multi-step finite-sample correction:

$$DM_{HLN} = DM \times \sqrt{\frac{N + 1 - 2h + h(h-1)/N}{N}}$$

---

## 7.13 Dynamic Volatility Targeting Strategy
To evaluate economic utility, forecasts dictate asset allocation under a 15% annualized target risk constraint:

$$w_t = \min\left(\max\left(\frac{\sigma_{target}}{\hat{\sigma}_t}, 0.0\right), 1.5\right)$$

To ensure zero lookahead bias in execution, position weights are shifted by 1 trading day: the portfolio holds weight $w_{t-1}$ from time $t-1$ to $t$.

---

## 7.14 Portfolio Turnover Calculation
Annualized portfolio turnover is computed as the annual average of absolute daily weight rebalancings:

$$\text{Turnover}_{annual} = 252 \times \frac{1}{N} \sum_{t=1}^N |w_t - w_{t-1}|$$

---

## 7.15 Transaction Cost and Slippage Modeling
Gross return:
$$R_{t}^{gross} = w_{t-1} R_t^{asset} + (1 - w_{t-1}) R_t^f$$

Transaction friction deduction:
$$\text{Cost}_t = |w_t - w_{t-1}| \times \left(c_{fee} + s_{slippage}\right)$$
$$\text{Baseline Friction:} \quad c_{fee} = 10\text{ bps (0.0010)}, \quad s_{slippage} = 5\text{ bps (0.0005)} \implies 15\text{ bps total}$$

Net return:
$$R_t^{net} = R_t^{gross} - \text{Cost}_t$$

---

## 7.16 Portfolio Performance Metrics
- **Compound Annual Growth Rate (CAGR):** $\text{CAGR} = \left(\prod (1 + R_t)\right)^{252/N} - 1$
- **Annualized Net Sharpe Ratio:** $\text{Sharpe} = \frac{\bar{R}^{net} - R^f}{\sigma(R^{net})} \times \sqrt{252}$
- **Annualized Net Sortino Ratio:** $\text{Sortino} = \frac{\bar{R}^{net} - R^f}{\sqrt{\frac{1}{N}\sum \min(R_t^{net} - R_t^f, 0)^2}} \times \sqrt{252}$
- **Maximum Drawdown:** $\text{MDD} = \min_t \left(\frac{W_t - \max_{\tau \le t} W_\tau}{\max_{\tau \le t} W_\tau}\right)$

---

## 7.17 Rebalancing Frequency Sensitivity
We evaluate performance across three discrete execution cadences:
1. **Daily (1-day):** Weights updated every trading session.
2. **Weekly (5-day):** Target weights sampled every 5 trading days and held constant.
3. **Biweekly (10-day):** Target weights sampled every 10 trading days and held constant.

---

## 7.18 Volatility Forecast Smoothing (Exploratory Extension)
To test whether turnover can be controlled without altering rebalancing frequency, an Exponential Moving Average (EMA) filter is applied directly to ML volatility forecasts:

$$\tilde{\sigma}_t = \alpha \hat{\sigma}_t + (1 - \alpha) \tilde{\sigma}_{t-1}, \quad \alpha = \frac{2}{\text{span} + 1}$$
with evaluation over $\text{span} \in \{1, 3, 5, 10\}$ trading days.
