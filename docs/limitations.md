# Methodological Scope, Assumptions, and Limitations

This document explicitly defines the boundaries, structural assumptions, and academic limitations of the empirical research study.

---

## 1. Single Asset Benchmark Scope
- **Primary Market:** All empirical evaluations are conducted strictly on the **NIFTY 50 Index (`^NSEI`)**, representing the Indian large-cap equity market.
- **Cross-Asset Generalizability:** While findings illuminate structural dynamics between econometric and machine learning models, conclusions should not be assumed to hold unconditionally across global developed equity indices (e.g., S&P 500, FTSE 100), fixed income, commodities, or foreign exchange without independent empirical verification.

---

## 2. Daily Sampling Frequency vs. Intraday Dynamics
- **Sampling Frequency:** The analysis relies on daily OHLCV closing snapshots.
- **Intraday Information:** While extreme-value range estimators (Parkinson, 1980; Garman & Klass, 1980) incorporate intraday high-low-open price dynamics, tick-level order book dynamics and 5-minute high-frequency realized kernels were not utilized. High-frequency intraday realized measures could alter signal-to-noise ratios.

---

## 3. Finite Out-of-Sample Window
- **Evaluation Period:** Out-of-sample walk-forward testing covers **997 trading days (January 2021 to December 2024)**.
- **Regime Specificity:** Although this 4-year period includes post-COVID recovery, global monetary tightening (2022–2023), and all-time equity highs, the empirical findings reflect the specific macro-financial environment of this sample.

---

## 4. Linear Transaction Friction Modeling
- **Cost Assumption:** Transaction friction is modeled as a linear function of portfolio weight turnover ($|w_t - w_{t-1}| \times [c_{fee} + s_{slippage}]$) with a baseline of 10 bps fee + 5 bps slippage.
- **Market Impact:** Non-linear square-root market impact laws (Almgren et al., 2005) and order book liquidity constraints are not modeled, which is a reasonable approximation for large-cap index futures but would require refinement for large-scale institutional fund execution.

---

## 5. Univariate Portfolio Allocation Context
- **Risk Management Design:** Volatility targeting is evaluated on a single risky asset plus a cash equivalent ($w_t = \sigma_{target}/\hat{\sigma}_t$).
- **Multi-Asset Extensions:** The framework does not model dynamic covariance matrix optimization across multi-asset portfolios (e.g., DCC-GARCH vs. multi-output machine learning).

---

## 6. Model Specification and Parameter Search Space
- **GARCH Family:** Evaluated using the standard GARCH(1,1) specification with Student's $t$ innovations. Asymmetric GJR-GARCH, EGARCH, or FIGARCH long-memory models were not evaluated in the primary benchmark.
- **Machine Learning Hyperparameters:** Random Forest and XGBoost hyperparameters were set based on institutional best practices rather than heavily tuned on the test set to avoid snooping bias.

---

## 7. Statement on Research Integrity
These limitations represent standard empirical finance boundaries. Acknowledging them ensures that the research findings remain academically defensible, conservative, and methodologically sound.
