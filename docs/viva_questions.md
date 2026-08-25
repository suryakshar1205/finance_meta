# Capstone Defense & Viva Voce Comprehensive Preparation Guide

This document contains 42 detailed defense questions and answers covering the theoretical, econometric, machine-learning, and economic dimensions of the capstone project:

**"A Comparative and Hybrid Framework for Financial Volatility Forecasting: Statistical Accuracy versus Economic Utility"**

---

## Section A: Problem & Motivation

### Q1: What is the primary research question of this capstone?
- **Short Answer:** Does superior out-of-sample volatility forecasting accuracy translate into superior economic performance after accounting for portfolio turnover, transaction costs, and rebalancing frequency?
- **Detailed Explanation:** While prior literature extensively evaluates volatility forecasting models on statistical loss metrics (e.g., RMSE, MAE, QLIKE), quantitative investment portfolios must translate forecasts into asset allocations. This study investigates whether the statistical gains of responsive machine-learning models survive realistic market execution friction.

### Q2: Why is volatility forecasting critical in quantitative finance?
- **Short Answer:** Volatility is the central risk metric governing derivative pricing, Value-at-Risk (VaR), portfolio margin requirements, and dynamic volatility targeting.
- **Detailed Explanation:** In modern risk management, under-estimating volatility leads to severe risk violations and catastrophic drawdowns, while over-estimating volatility leads to excessive cash holdings and drag on compounding capital.

### Q3: Why choose the NIFTY 50 Index for this empirical study?
- **Short Answer:** The NIFTY 50 is the benchmark index of the National Stock Exchange of India, one of the world's most active and liquid equity derivatives markets.
- **Detailed Explanation:** The NIFTY 50 provides a deep, liquid market with rich stylized facts (volatility clustering, fat tails, negative return-volatility asymmetry) over a 15-year macroeconomic span (2010–2024), offering a representative emerging/developed market testbed.

---

## Section B: Volatility Theory & Stylized Facts

### Q4: What is volatility clustering (Mandelbrot, 1963)?
- **Short Answer:** "Large changes tend to be followed by large changes, of either sign, and small changes tend to be followed by small changes."
- **Detailed Explanation:** Volatility is conditionally time-varying and autocorrelated. While raw asset returns exhibit zero linear autocorrelation (market efficiency), squared and absolute returns exhibit strong, persistent positive autocorrelation spanning 40+ trading days.

### Q5: What is leptokurtosis in financial returns?
- **Short Answer:** Return distributions have fatter tails and higher peaks than a standard Gaussian normal distribution.
- **Detailed Explanation:** The NIFTY 50 daily return distribution exhibits an excess kurtosis of $8.94$, decisively rejecting Gaussian normality ($p < 0.0001$). Extreme market shocks (such as -13.9% on March 23, 2020) occur far more frequently than predicted by a normal distribution.

### Q6: Why is forward realized volatility used as the target instead of squared returns?
- **Short Answer:** Daily squared returns ($r_t^2$) are an unbiased but extremely noisy proxy for latent conditional variance; multi-day realized volatility significantly improves signal-to-noise ratio.
- **Detailed Explanation:** Andersen and Bollerslev (1998) showed that daily squared return noise masks model forecast quality. Summing squared returns over a forward $k$-day window ($RV_{t, t+k} = \sqrt{\sum_{i=1}^k r_{t+i}^2 \times 252/k}$) produces a reliable, low-noise target proxy.

---

## Section C: Econometric GARCH Modeling

### Q7: What is a GARCH(1,1) model (Bollerslev, 1986)?
- **Short Answer:** A stationary autoregressive conditional heteroskedasticity model where conditional variance $\sigma_t^2$ is a linear combination of a constant ($\omega$), lagged squared innovation ($\epsilon_{t-1}^2$), and lagged conditional variance ($\sigma_{t-1}^2$).
- **Detailed Explanation:** $\sigma_t^2 = \omega + \alpha \epsilon_{t-1}^2 + \beta \sigma_{t-1}^2$. The model captures volatility clustering through the ARCH term ($\alpha$) and persistence through the GARCH term ($\beta$).

### Q8: What does persistence ($\lambda = \alpha + \beta$) mean in GARCH?
- **Short Answer:** It measures the rate at which volatility shocks decay back to the unconditional long-run variance.
- **Detailed Explanation:** For NIFTY 50, $\alpha = 0.0443$ and $\beta = 0.9434$, yielding $\lambda = 0.9877$. The implied volatility half-life is $\tau = \ln(0.5) / \ln(0.9877) \approx 55.9$ trading days, indicating strong long-memory persistence.

### Q9: Why use Student's $t$ innovations instead of Gaussian errors in GARCH?
- **Short Answer:** To explicitly model the heavy tails and conditional kurtosis observed in empirical equity returns.
- **Detailed Explanation:** Gaussian MLE underestimates the probability of extreme tail shocks. Fitting a Student's $t$ distribution estimates the degrees-of-freedom parameter ($\nu$), improving log-likelihood and standard error robustness.

### Q10: How do you verify that GARCH is well-specified?
- **Short Answer:** By running Ljung-Box $Q$ and Engle ARCH-LM tests on standardized residuals ($\hat{e}_t = \epsilon_t / \hat{\sigma}_t$).
- **Detailed Explanation:** If correctly specified, standardized residuals and squared standardized residuals should exhibit zero remaining autocorrelation or ARCH effects. In our sample, Ljung-Box on $\hat{e}_t^2$ yielded $p = 0.8191$ and ARCH-LM yielded $p = 0.8259$, confirming no remaining conditional heteroskedasticity.

---

## Section D: Machine Learning Models

### Q11: Why use Random Forest for volatility forecasting?
- **Short Answer:** Random Forest is a non-parametric ensemble of de-correlated bagged decision trees that captures non-linear feature interactions without overfitting.
- **Detailed Explanation:** By averaging predictions over 200 trees constructed on randomized bootstrap samples and random feature subsets, Random Forest reduces variance and produces robust out-of-sample volatility estimates.

### Q12: Why use XGBoost for volatility forecasting?
- **Short Answer:** XGBoost sequentially trains gradient-boosted regression trees that correct the residuals of prior trees with regularized shrinkage.
- **Detailed Explanation:** XGBoost employs second-order Taylor expansion of the loss function, column subsampling, and explicit $L_1/L_2$ leaf regularization, making it highly effective for capturing complex non-linear structures in financial features.

### Q13: What is the proposed Hybrid GARCH + ML framework?
- **Short Answer:** A model that feeds the in-sample GARCH conditional volatility estimate as an explicit structural feature into an XGBoost regression model.
- **Detailed Explanation:** Traditional ML ignores the theoretical autoregressive structure of conditional variance. By including $\hat{\sigma}_{GARCH, t}$ alongside 28 market features, the Hybrid model combines econometric domain structure with non-linear machine learning flexibility.

---

## Section E: Feature Engineering

### Q14: What is the Parkinson (1980) volatility estimator?
- **Short Answer:** An extreme-value volatility estimator that uses the session High and Low prices.
- **Detailed Explanation:** $\sigma_{Parkinson}^2 = \frac{252}{4 \ln 2} \frac{1}{N} \sum \ln(High_t / Low_t)^2$. It is approximately 5 times more efficient than close-to-close variance because it captures intraday price dispersion.

### Q15: What is the Garman-Klass (1980) volatility estimator?
- **Short Answer:** An extension of Parkinson that incorporates session Open, High, Low, and Close prices.
- **Detailed Explanation:** It accounts for opening jumps and closing trends, achieving up to 8 times greater statistical efficiency than close-to-close volatility estimators.

### Q16: Why are lagged returns and term-structure ratios included in the feature set?
- **Short Answer:** To capture mean-reversion, momentum, and shifts in the volatility term structure ($\sigma_{5d}/\sigma_{20d}$).
- **Detailed Explanation:** Term structure ratios signal whether the market is experiencing short-term volatility spikes relative to medium-term baselines, providing strong forward predictive power for machine learning trees.

---

## Section F: Validation & Leakage Prevention

### Q17: What is expanding-window walk-forward validation?
- **Short Answer:** An out-of-sample testing protocol where models are trained on all available past data up to time $t$, forecast day $t+1$, and periodic retraining occurs as time advances.
- **Detailed Explanation:** Unlike k-fold cross-validation which leaks future data, walk-forward validation strictly mirrors institutional live deployment: models re-estimate parameters every 20 days on expanding historical folds across 997 test days (2021–2024).

### Q18: How did you ensure zero lookahead bias in target and feature alignment?
- **Short Answer:** Features at time $t$ use data $\le t$; the target $RV_{t, t+k}$ uses future returns strictly over $[t+1, t+k]$; portfolio position weights $w_{t-1}$ are shifted by 1 day.
- **Detailed Explanation:** Lead-lag correlation assertions confirmed $\text{Corr}(x_t, r_{t+1}) \approx 0$. Execution weights calculated at close of day $t-1$ are held during session $t$, guaranteeing that no current or future information is consumed prior to execution.

---

## Section G: Statistical Loss Functions & Hypothesis Testing

### Q19: Why evaluate using RMSE, MAE, and QLIKE?
- **Short Answer:** Different loss functions penalize different forecast errors: RMSE penalizes large errors quadratically; MAE penalizes errors linearly; QLIKE is scale-invariant and robust to noise.
- **Detailed Explanation:** Patton (2011) proved that QLIKE and MSE are the only loss functions that guarantee consistent model ranking when evaluating against an imperfect realized volatility proxy.

### Q20: What is the Patton (2011) QLIKE loss function?
- **Short Answer:** $L_{QLIKE} = \frac{\sigma^2}{h} - \ln\left(\frac{\sigma^2}{h}\right) - 1$, where $\sigma^2$ is actual variance and $h$ is forecasted variance.
- **Detailed Explanation:** QLIKE measures the percentage error in variance. It penalizes under-predictions ($\sigma^2 \gg h$) exponentially, making it heavily sensitive to models that under-estimate high-volatility spikes.

### Q21: What is the Diebold-Mariano (1995) test?
- **Short Answer:** An econometric hypothesis test to determine whether the difference in loss between two competing forecasts is statistically significantly different from zero.
- **Detailed Explanation:** It tests $H_0: \mathbb{E}[d_t] = 0$ where $d_t = e_{1,t}^2 - e_{2,t}^2$. The test accounts for serial correlation in forecast errors using the Harvey-Leybourne-Newbold (1997) multi-step finite-sample correction.

---

## Section H: Portfolio Construction & Volatility Targeting

### Q22: What is dynamic volatility targeting?
- **Short Answer:** An asset allocation rule that scales exposure inversely with forecasted volatility to maintain constant annualized portfolio risk.
- **Detailed Explanation:** $w_t = \min(\sigma_{target}/\hat{\sigma}_t, w_{max})$. When forecasted volatility rises, leverage is reduced; when volatility falls, leverage is increased, generating superior risk-adjusted compounding.

### Q23: What target volatility and leverage constraints were used?
- **Short Answer:** Target volatility of 15.0% annualized, maximum leverage of 1.50 (150% equity exposure), and minimum leverage of 0.0 (no shorting).
- **Detailed Explanation:** The 15% target closely matches the historical unconditional volatility of the NIFTY 50 index (16.48%), providing a clean, un-levered baseline comparison against Buy & Hold.

---

## Section I: Turnover & Transaction Costs

### Q24: How is portfolio turnover calculated?
- **Short Answer:** $\text{Turnover}_{annual} = 252 \times \frac{1}{N} \sum |w_t - w_{t-1}|$.
- **Detailed Explanation:** It measures the total volume of portfolio rebalancing trades executed per year as a multiple of total portfolio equity.

### Q25: How are transaction costs and execution slippage modeled?
- **Short Answer:** Deducting $10\text{ bps}$ transaction fee + $5\text{ bps}$ slippage ($15\text{ bps}$ total friction) per unit of position turnover.
- **Detailed Explanation:** $\text{Cost}_t = |w_t - w_{t-1}| \times 0.0015$. This cost is subtracted directly from gross portfolio returns on every rebalancing session.

---

## Section J: Empirical Results & The Central Disconnect

### Q26: What are the primary statistical forecast accuracy results?
- **Short Answer:** Random Forest achieved lowest RMSE ($0.08497$) and QLIKE ($0.47287$); XGBoost achieved lowest MAE ($0.05072$); GARCH achieved RMSE ($0.10497$) and QLIKE ($0.78001$).
- **Detailed Explanation:** Random Forest, XGBoost, and Hybrid models reduced RMSE by 14.5% to 16.1% compared to historical baseline. Diebold-Mariano tests confirmed statistical superiority over GARCH at the 1% significance level ($p < 0.01$).

### Q27: What are the primary economic risk-management results under daily rebalancing?
- **Short Answer:** GARCH achieved the highest Net Sharpe (0.490) and Net CAGR (14.85%), outperforming XGBoost (Net Sharpe 0.416) and Random Forest (Net Sharpe 0.371).
- **Detailed Explanation:** GARCH generated an annual turnover of only $2.47$ (cost drag of $-0.43\%$), whereas XGBoost generated a turnover of $17.31$ (cost drag of $-2.96\%$) and Random Forest generated $12.72$ (cost drag of $-2.14\%$).

### Q28: Why does GARCH have high QLIKE loss yet the best daily economic performance?
- **Short Answer:** QLIKE exponentially penalizes GARCH's slow reaction to sudden volatility spikes; however, this same smoothness prevents destructive portfolio churn in volatility targeting.
- **Detailed Explanation:** GARCH's autoregressive persistence ($\beta = 0.943$) causes it to under-predict sudden crash spikes (58.7% under-prediction rate), producing high QLIKE loss. But in dynamic position allocation ($w_t = \sigma_{target}/\hat{\sigma}_t$), smooth forecasts prevent daily weight oscillations, minimizing transaction cost drag.

### Q29: Why does XGBoost's economic performance improve under weekly rebalancing?
- **Short Answer:** Lengthening rebalancing from daily to weekly cuts turnover by 59% (from 17.31 to 7.11), reducing cost drag from -2.96% to -1.25% and lifting Net Sharpe to 0.530.
- **Detailed Explanation:** The results are consistent with the hypothesis that execution friction, rather than poor signal quality, degrades daily ML performance. Weekly rebalancing allows the portfolio to harvest ML forecast accuracy while eliminating high-frequency noise trading.

### Q30: How does forecast smoothing affect Random Forest?
- **Short Answer:** A 10-day EMA smoothing filter cuts Random Forest turnover by 67% (from 12.72 to 4.22), increasing Net Sharpe from 0.371 to 0.436.
- **Detailed Explanation:** Smoothing acts as a low-pass filter on ML predictions, demonstrating the fundamental tradeoff between forecast responsiveness and portfolio stability.

### Q31: What is the estimated break-even transaction cost for XGBoost vs GARCH?
- **Short Answer:** Approximately 0 to 2 bps of one-way transaction fee (5 to 7 bps total friction including slippage).
- **Detailed Explanation:** In a friction-free market (0 bps fee), XGBoost Net Sharpe (0.486) closely matches GARCH (0.500). At realistic fees ($\ge 10$ bps), GARCH's low turnover provides decisive economic superiority under daily rebalancing.

---

## Section K: Methodological Nuance & Policy Distinction

### Q32: Can you say "Machine Learning is better than GARCH"?
- **Short Answer:** No. Machine learning is statistically superior in forecast loss metrics ($p < 0.01$), but economically inferior under naive daily rebalancing due to turnover friction.
- **Detailed Explanation:** A rigorous quantitative evaluation requires separating statistical accuracy from economic utility. Model superiority is context-dependent and loss-function dependent.

### Q33: Can you say "GARCH is better than Machine Learning"?
- **Short Answer:** No. GARCH fails to capture non-linear market feature interactions and produces higher forecast errors; its economic advantage is purely a byproduct of low turnover under daily rebalancing.
- **Detailed Explanation:** When ML models are implemented with weekly rebalancing or forecast smoothing, they surpass GARCH's Net Sharpe (XGBoost Weekly Sharpe 0.530 vs GARCH 0.490).

### Q34: What is the difference between a Forecast Model and a Portfolio Implementation Policy?
- **Short Answer:** The forecast model maps market data to predicted volatility; the implementation policy determines how that forecast is translated into trades (cadence, inertia, smoothing).
- **Detailed Explanation:** "Best volatility forecast" $\neq$ "Best trading strategy." A superior forecast can produce poor net returns if deployed under an inappropriate execution policy.

---

## Section L: Limitations, Contribution & Viva Defense

### Q35: What are the main limitations of this study?
- **Short Answer:** Single market benchmark (NIFTY 50), daily OHLCV sampling frequency, linear transaction cost assumptions, and univariate asset allocation.
- **Detailed Explanation:** The study intentionally establishes a clean, leakage-free benchmark on 997 out-of-sample days. Future research should evaluate multi-asset covariance matrices and intraday high-frequency tick data.

### Q36: Did you cherry-pick the test period?
- **Short Answer:** No. We used a strict, continuous 4-year chronological out-of-sample period (January 2021 to December 2024, 997 consecutive trading days).
- **Detailed Explanation:** The test window was fixed a priori before running walk-forward experiments, ensuring zero backtest overfitting.

### Q37: What is the central academic contribution of this capstone?
- **Short Answer:** Demonstrating that volatility model evaluation must extend beyond statistical forecast accuracy to incorporate the economic implementation of forecasts, particularly turnover, transaction costs, and rebalancing frequency.
- **Detailed Explanation:** The capstone rigorously proves why statistical superiority does not automatically yield economic alpha in quantitative finance, and provides actionable implementation policies (rebalancing cadence and smoothing) to bridge the gap.

### Q38: Why didn't you use Deep Learning (LSTM / Transformers)?
- **Short Answer:** Gradient-boosted decision trees (XGBoost) and Random Forests consistently match or outperform deep learning on tabular financial time series with limited sample size, while avoiding extreme compute overhead and overfitting.
- **Detailed Explanation:** Shwartz-Ziv and Armon (2022) demonstrated that tree-based models dominate deep architectures on tabular data. Tree ensembles provide deterministic, interpretable baselines with robust out-of-sample properties.

### Q39: How do you know there was no data leakage in the feature matrix?
- **Short Answer:** All 28 features at time $t$ use data $\le t$, confirmed via formal correlation assertions ($\text{Corr}(x_t, r_{t+1}) \approx 0$) and strictly chronological expanding-window splitting.
- **Detailed Explanation:** The feature pipeline code in `src/features.py` executes automated temporal boundary assertions before generating feature tables.

### Q40: What happens if transaction costs increase to 30 or 50 bps?
- **Short Answer:** GARCH's Net Sharpe remains resilient (0.471 at 30 bps, 0.451 at 50 bps), while ML Net Sharpe collapses (XGBoost drops to 0.275 at 30 bps, 0.134 at 50 bps).
- **Detailed Explanation:** Linear transaction costs scale directly with turnover. GARCH's low turnover of 2.47 insulates it from high friction, whereas high-turnover strategies experience severe compounding drag.

### Q41: Why did you test volatility regime terciles?
- **Short Answer:** To evaluate model robustness across Low ($T_1$), Medium ($T_2$), and High ($T_3$) volatility environments without cherry-picking arbitrary crisis dates.
- **Detailed Explanation:** In calm and moderate markets ($T_1, T_2$), ML models reduce RMSE by over 30%. In extreme volatility ($T_3$), the Hybrid and Random Forest models achieve the lowest MAE and lowest QLIKE loss.

### Q42: What is your primary advice to a quantitative portfolio manager based on these results?
- **Short Answer:** Never deploy raw, unconstrained daily ML volatility forecasts in dynamic risk targeting without turnover regularization, rebalancing inertia, or weekly execution schedules.
- **Detailed Explanation:** ML models provide real statistical forecasting alpha, but capturing that alpha in live production requires aligning forecast dynamics with transaction friction constraints.
