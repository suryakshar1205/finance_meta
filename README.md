# A Comparative and Hybrid Framework for Financial Volatility Forecasting: Statistical Accuracy versus Economic Utility

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Research Status](https://img.shields.io/badge/Research-Fully%20Reproducible%20Audit%20Passed-success.svg)]()
[![Market](https://img.shields.io/badge/Primary%20Market-NIFTY%2050%20Index-orange.svg)]()
[![License](https://img.shields.io/badge/License-Academic%20Research-lightgrey.svg)]()

An empirical quantitative finance research framework evaluating traditional econometric models (GARCH), modern machine learning models (Random Forest, XGBoost), and an integrated Hybrid GARCH + ML architecture for out-of-sample financial volatility forecasting and dynamic risk management on the **NIFTY 50** equity index.

---

> [!NOTE]
> **Research Compliance & Academic Integrity Disclaimer**  
> All empirical results, statistics, loss metrics, Diebold-Mariano $p$-values, and Sharpe ratios reported in this repository are derived strictly from live out-of-sample walk-forward experiments on historical market data ($N = 3,679$ daily observations, 2010–2024). No performance values or charts have been synthetically manufactured. Reported empirical results should not be interpreted as guaranteed future investment performance.

---

## 1. Central Research Question

> **Can machine-learning models improve out-of-sample financial volatility forecasts compared with traditional GARCH models, and do these statistical improvements translate into economically meaningful risk-management benefits after transaction costs and execution slippage?**

### Evaluated Model Suite:
1. **Historical Volatility** (Simple 20-Day Rolling Sample Benchmark)
2. **GARCH(1,1)** (Classical Econometric Benchmark with Student's $t$ innovations)
3. **Random Forest** (Non-linear Bagged Decision Tree Ensemble)
4. **XGBoost** (Gradient-Boosted Decision Trees)
5. **Hybrid GARCH + ML** (Proposed Structural-Econometric + ML Regressor)

---

## 2. Project Architecture & Directory Structure

```text
finance_meta/
├── README.md                           # Master research overview & documentation
├── requirements.txt                    # Pinned package dependencies
├── run_all_experiments.py              # Master reproducible end-to-end execution runner
├── build_notebooks.py                  # Jupyter notebook generation utility
│
├── config/
│   ├── config.yaml                     # Central declarative configuration
│   └── final_experiment.yaml           # Frozen production experiment configuration
│
├── data/
│   ├── raw/nifty50_daily_raw.csv       # Immutable raw NIFTY 50 OHLCV data (3,679 rows)
│   └── processed/nifty50_daily_processed.csv # Cleaned market series with returns & indicators
│
├── src/
│   ├── __init__.py                     # Package init
│   ├── data_loader.py                  # Yahoo Finance caching & data acquisition
│   ├── preprocessing.py                # Chronological sorting, duplicate & consistency checks
│   ├── volatility.py                   # Realized, Parkinson & Garman-Klass estimators
│   ├── features.py                     # Leakage-free feature engineering & integrity assertions
│   ├── garch.py                        # GARCH(1,1) model with Student's t & residual diagnostics
│   ├── ml_models.py                    # Random Forest & XGBoost regression pipelines
│   ├── hybrid.py                       # Integrated GARCH + ML hybrid architecture
│   ├── validation.py                   # Chronological splits & Walk-forward evaluation engine
│   ├── evaluation.py                   # MAE, RMSE, QLIKE, Diebold-Mariano matrix & MZ regressions
│   ├── backtesting.py                  # Volatility targeting, cost grid & rebalance frequency
│   ├── visualization.py                # Publication-grade Matplotlib visualizer (Agg backend)
│   ├── final_polish_analysis.py        # Final sensitivity & claim audit runner
│   └── validate_results.py             # Programmatic result consistency auditor
│
├── notebooks/                          # 12 Modular interactive Jupyter notebooks
│   ├── 01_data_collection.ipynb to 12_robustness.ipynb
│
├── results/
│   ├── audit_report.md                 # 21-Point methodological quality audit report
│   ├── baseline/                       # Frozen baseline artifacts & forecasts
│   ├── final_primary/                  # Frozen primary results before final polish
│   ├── final/                          # Frozen publication tables & final findings
│   │   ├── final_forecast_metrics.csv
│   │   ├── final_dm_tests.csv
│   │   ├── final_dm_comparison_matrix.csv
│   │   ├── final_portfolio_metrics.csv
│   │   ├── final_transaction_cost_sensitivity.csv
│   │   ├── final_turnover_analysis.csv
│   │   ├── rebalancing_frequency_sensitivity.csv
│   │   ├── final_calibration_analysis.csv
│   │   ├── final_model_parameters.csv
│   │   ├── final_regime_metrics.csv
│   │   ├── final_smoothed_ml_extension.csv
│   │   ├── claim_audit.csv
│   │   ├── final_configuration.yaml
│   │   └── FINAL_RESEARCH_STATUS.md
│   ├── figures/                        # 13 Publication figures (PNG, 300 DPI)
│   ├── tables/                         # CSV and JSON result summaries
│   └── forecasts/                      # Out-of-sample predictions (997 test days)
│
├── paper/
│   └── paper.md                        # Academic research paper manuscript (18 sections)
│
└── presentation/
    └── presentation_slides.md          # Capstone presentation slide deck (14 slides)
```

---

## 3. Quickstart & Reproducibility

### Prerequisites
Python 3.11+ installed.

### Setup Environment
```bash
pip install -r requirements.txt
```

### Reproduce Entire Research Study (Single Command)
```bash
python run_all_experiments.py
python src/refinement_experiments.py
python src/final_polish_analysis.py
python src/validate_results.py
```

---

## 4. Master Empirical Tables (Live Data Experiments)

### Master Table 1: Statistical Accuracy vs. Daily Risk Management ($N = 997$ Out-of-Sample Days)

| Model | RMSE | MAE | QLIKE | DM vs GARCH ($p$-val) | Daily Turnover | Net Sharpe | Net CAGR | Max DD |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Random Forest** | **0.08497** | 0.05151 | **0.47287** | **$p = 0.0044$ (***)** | 12.72 | 0.371 | 11.23% | -41.42% |
| **XGBoost** | 0.08594 | **0.05072** | 0.48869 | **$p = 0.0025$ (***)** | 17.31 | 0.416 | 12.48% | **-38.90%** |
| **Hybrid GARCH+ML** | 0.08657 | 0.05266 | 0.47664 | **$p = 0.0031$ (***)** | 17.27 | 0.321 | 9.80% | -41.56% |
| **Historical Vol** | 0.10128 | 0.06036 | 0.56519 | $p = 0.7253$ (Eqv) | 8.29 | 0.393 | 11.74% | -35.84% |
| **GARCH(1,1)** | 0.10497 | 0.06320 | 0.78001 | Benchmark | **2.47** | **0.490** | **14.85%** | -41.38% |

### Master Table 2: Portfolio Implementation Policy & Rebalancing Cadence

| Model | Implementation Policy | Annual Turnover | Gross CAGR | Net CAGR | Net Sharpe | Cost Drag |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **GARCH** | Daily (1-day) | 2.47 | 15.28% | 14.85% | 0.490 | -0.43% |
| **XGBoost** | Daily (1-day) | 17.31 | 15.44% | 12.48% | 0.416 | -2.96% |
| **XGBoost** | **Weekly (5-day)** | **7.11** | **17.32%** | **16.07%** | **0.530** | **-1.25%** |
| **XGBoost** | **Biweekly (10-day)** | **4.66** | **17.82%** | **17.00%** | **0.558** | **-0.82%** |
| **Random Forest** | Daily (1-day) | 12.72 | 13.37% | 11.23% | 0.371 | -2.14% |
| **Random Forest** | **EMA Smoothed (10d)** | **4.22** | **13.37%** | **13.20%** | **0.436** | **-0.17%** |

---

## 5. Core Research Contributions

1. **Statistical Accuracy Dimension:** Machine learning models capture non-linear relationships in lagged market returns, price ranges, and volume features, achieving statistically significant forecast error reductions over GARCH ($p < 0.01$ under Diebold-Mariano test). Model rankings depend on the loss function: **Random Forest** achieves lowest RMSE and QLIKE; **XGBoost** achieves lowest MAE.
2. **The Turnover-Friction Dilemma:** In real-world volatility-targeting portfolios, higher ML forecast responsiveness causes excessive daily rebalancing (Annual Turnover: 12.7 to 17.3), generating 2.14% to 2.96% in annual transaction friction under standard fees (10 bps fee + 5 bps slippage).
3. **GARCH Economic Resilience:** Because GARCH models mean-reverting conditional variance smoothly, it generates minimal turnover (2.47) and minimal friction (-0.43%), achieving the **highest Net Sharpe ratio (0.490)** among all daily strategies.
4. **Rebalancing Cadence Optimization:** Moving from daily to **weekly (5-day)** rebalancing cuts XGBoost turnover from 17.31 to 7.11, lifting its Net Sharpe ratio to **0.530** (surpassing daily GARCH).
5. **Central Contribution:** Demonstrating that volatility-model evaluation should extend beyond statistical forecast accuracy to incorporate the economic implementation of forecasts, particularly turnover, transaction costs, and rebalancing frequency.

---

## 6. Citations & Theoretical References

- Bollerslev, T. (1986). Generalized autoregressive conditional heteroskedasticity. *Journal of Econometrics*, 31(3), 307-327.
- Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5-32.
- Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *ACM SIGKDD*, 785-794.
- Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive accuracy. *Journal of Business & Economic Statistics*, 13(3), 253-263.
- Engle, R. F. (1982). Autoregressive conditional heteroscedasticity with estimates of the variance of United Kingdom inflation. *Econometrica*, 987-1007.
- Garman, M. B., & Klass, M. J. (1980). On the estimation of security price volatilities from historical data. *Journal of Business*, 67-78.
- Harvey, D., Leybourne, S., & Newbold, P. (1997). Testing the equality of prediction mean squared errors. *International Journal of Forecasting*, 13(2), 281-291.
- Mincer, J., & Zarnowitz, V. (1969). The evaluation of economic forecasts. In *Economic Forecasts and Expectations* (pp. 3-46). NBER.
- Parkinson, M. (1980). The extreme value method for estimating the variance of the rate of return. *Journal of Business*, 61-65.
- Patton, A. J. (2011). Volatility forecast comparison using imperfect volatility proxies. *Journal of Econometrics*, 160(1), 246-256.
