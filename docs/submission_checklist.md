# Capstone Submission Quality & Integrity Checklist

This checklist confirms that all academic, statistical, econometric, code, and packaging requirements are fully satisfied for the capstone submission:

**Project Title:** *A Comparative and Hybrid Framework for Financial Volatility Forecasting: Statistical Accuracy versus Economic Utility*

---

## 1. RESEARCH DESIGN & METHODOLOGY
- [x] Clear primary research question defined.
- [x] Explicit formal research hypotheses ($H_1$ to $H_5$) articulated.
- [x] Clear literature context and research gap established.
- [x] Primary benchmark market explicitly stated (NIFTY 50 Index, `^NSEI`).
- [x] Out-of-sample sample size documented ($N = 997$ trading days, 2021–2024).
- [x] Methodological limitations and boundaries explicitly stated.

---

## 2. DATA INTEGRITY & AUDIT
- [x] Dataset source and frequency documented (Daily OHLCV, 2010–2024, $N = 3,679$).
- [x] Raw data (`data/raw/`) strictly immutable and distinguished from processed data (`data/processed/`).
- [x] Monotonic datetime ordering verified.
- [x] Price consistency assertions checked ($High \ge \max(Open, Close)$, $Low \le \min(Open, Close)$, $Price > 0$).
- [x] Zero duplicate records verified.
- [x] Missing non-trading business days properly handled without artificial forward-filling.

---

## 3. MODELS & ESTIMATORS
- [x] Historical Volatility baseline implemented (20-day rolling sample standard deviation).
- [x] GARCH(1,1) econometric model implemented with Student's $t$ innovations.
- [x] GARCH standardized residual diagnostics verified (Ljung-Box $Q$ on $e_t^2$: $p=0.8191$, ARCH-LM: $p=0.8259$).
- [x] Random Forest regressor implemented (200 trees, regularized hyperparameters).
- [x] XGBoost gradient-boosted regressor implemented (200 rounds, $\eta=0.03$).
- [x] Proposed Hybrid GARCH + ML framework implemented.

---

## 4. VALIDATION & LEAKAGE PREVENTION
- [x] Chronological train/validation/test splits implemented without shuffling.
- [x] Expanding-window walk-forward validation engine implemented across 997 out-of-sample days.
- [x] Periodic 20-day model parameter refitting executed.
- [x] Target forward annualized realized volatility ($RV_{t, t+5}$) strictly calculated on $[t+1, t+5]$.
- [x] Feature matrix (28 predictors) uses data strictly $\le t$.
- [x] Zero lookahead bias verified via automated lead-lag assertions.

---

## 5. STATISTICAL EVALUATION
- [x] RMSE loss computed and reported.
- [x] MAE loss computed and reported.
- [x] Patton (2011) QLIKE robust loss computed and reported.
- [x] Diebold-Mariano hypothesis tests executed with Harvey-Leybourne-Newbold multi-step corrections.
- [x] Full $5 \times 5$ pairwise Diebold-Mariano matrix and heatmap generated.
- [x] Mincer-Zarnowitz OLS calibration regressions and bias statistics calculated.
- [x] Loss-function-dependent model rankings explicitly documented (RF wins on RMSE/QLIKE; XGBoost wins on MAE).

---

## 6. ECONOMIC BACKTESTING & TRANSACTION COSTS
- [x] Dynamic 15% volatility targeting strategy implemented with 0.0 to 1.5 leverage bounds.
- [x] Portfolio execution weights shifted by 1 day ($w_{t-1}$) for zero lookahead execution.
- [x] Portfolio annual turnover calculated and reported.
- [x] Linear transaction friction deducted (10 bps fee + 5 bps slippage per unit turnover).
- [x] Gross and Net CAGR, Annualized Volatility, Net Sharpe, Net Sortino, and Maximum Drawdown reported.
- [x] Transaction cost sensitivity evaluated across 0, 5, 10, 15, 20, 30, 50 bps.
- [x] Estimated break-even transaction cost reported ($\approx 0\text{--}2$ bps fee).

---

## 7. SENSITIVITY & EXPLORATORY EXTENSIONS
- [x] Volatility regime analysis conducted across Low, Medium, and High volatility terciles.
- [x] Rebalancing frequency sensitivity formalized across Daily, Weekly (5-day), and Biweekly (10-day) cadences.
- [x] Exploratory EMA forecast smoothing extension evaluated across 1, 3, 5, 10 day spans.
- [x] Clear conceptual distinction made between Forecast Model and Portfolio Implementation Policy.

---

## 8. DOCUMENTATION & REPRODUCIBILITY
- [x] Master `README.md` structured across 14 required sections.
- [x] Comprehensive `docs/methodology.md` created (Sections 7.1 to 7.18).
- [x] Detailed `docs/reproducibility.md` created with step-by-step instructions.
- [x] Explicit `docs/limitations.md` created.
- [x] 42-Question Viva defense guide `docs/viva_questions.md` created.
- [x] Academic research paper manuscript `paper/paper.md` structured across 18 formal sections.
- [x] Executive presentation deck `presentation/presentation_slides.md` structured across 14 slides.
- [x] Automated consistency validator `src/validate_results.py` executed and passed with zero discrepancies.
- [x] Final submission package `submission/` assembled without cache or temporary files.
