# Capstone Project Submission Overview & Examiner Guide

**Project Title:** A Comparative and Hybrid Framework for Financial Volatility Forecasting: Statistical Accuracy versus Economic Utility  
**Author / Division:** Quantitative Finance Research Division  
**Primary Market:** NIFTY 50 Index (`^NSEI`, National Stock Exchange of India, 2010–2024)  
**Sample Period:** 15 Years ($N = 3,679$ Total Observations, $N = 997$ Out-of-Sample Evaluation Days)  

---

## 1. Executive Summary & Central Research Contribution
This quantitative finance capstone provides a comprehensive, reproducible, and leakage-free empirical comparison of classical econometric (GARCH), modern machine learning (Random Forest, XGBoost), and hybrid volatility forecasting models on the NIFTY 50 equity benchmark.

### Core Empirical Findings:
1. **Statistical Forecasting Accuracy:** Machine learning models capture non-linear relationships in lagged market returns, price ranges, and volume features, achieving statistically significant forecast error reductions over GARCH ($p < 0.01$ under Diebold-Mariano tests). Model rankings depend on the loss function: **Random Forest** achieves lowest RMSE ($0.08497$) and lowest QLIKE ($0.47287$); **XGBoost** achieves lowest MAE ($0.05072$).
2. **Economic Risk Management (Daily Rebalancing):** In dynamic 15% volatility targeting, high ML forecast responsiveness induces severe portfolio turnover (12.7 to 17.3 per annum), creating an annual transaction cost drag of 2.14% to 2.96% under realistic frictions (10 bps fee + 5 bps slippage). Conversely, GARCH's structural parameter smoothness produces minimal turnover (2.47), preserving the **highest Net Sharpe ratio (0.490)** among all primary daily strategies.
3. **Portfolio Implementation Policy:** "Best volatility forecast" does not equal "Best trading strategy." Adjusting the implementation policy to **weekly rebalancing** cuts XGBoost turnover from 17.31 to 7.11, lifting its Net Sharpe ratio to **0.530** (surpassing daily GARCH).
4. **Central Contribution:** Demonstrating that volatility model evaluation must extend beyond statistical loss metrics to incorporate portfolio implementation policy, turnover friction, and execution costs.

---

## 2. Submission Directory Structure & File Guide

```text
submission/
├── README_SUBMISSION.md               # Master examiner overview (this file)
├── submission_checklist.md            # Comprehensive 8-category submission checklist
│
├── paper/
│   └── paper.md                       # Formal 18-section academic research manuscript
│
├── presentation/
│   └── presentation_slides.md         # 14-Slide executive presentation deck
│
├── source/                            # Clean, documented production source code
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── volatility.py
│   ├── features.py
│   ├── garch.py
│   ├── ml_models.py
│   ├── hybrid.py
│   ├── validation.py
│   ├── evaluation.py
│   ├── backtesting.py
│   ├── refinement_experiments.py
│   ├── final_polish_analysis.py
│   ├── visualization.py
│   └── validate_results.py
│
├── results/                           # Frozen empirical CSV tables
│   ├── final_forecast_metrics.csv
│   ├── final_dm_tests.csv
│   ├── final_dm_comparison_matrix.csv
│   ├── final_portfolio_metrics.csv
│   ├── final_transaction_cost_sensitivity.csv
│   ├── final_turnover_analysis.csv
│   ├── rebalancing_frequency_sensitivity.csv
│   ├── final_calibration_analysis.csv
│   ├── final_model_parameters.csv
│   ├── final_regime_metrics.csv
│   ├── final_smoothed_ml_extension.csv
│   ├── claim_audit.csv
│   ├── final_configuration.yaml
│   └── FINAL_RESEARCH_STATUS.md
│
└── figures/                           # 10 Publication figures (300 DPI)
    ├── 01_price_series.png
    ├── 02_realized_volatility.png
    ├── 03_acf_squared_returns.png
    ├── 04_forecast_comparison.png
    ├── 05_model_accuracy_comparison.png
    ├── 06_forecast_accuracy_vs_economic_utility.png
    ├── 07_equity_curves.png
    ├── 08_transaction_cost_sensitivity.png
    ├── 09_rebalancing_frequency_tradeoff.png
    └── 10_turnover_vs_net_sharpe.png
```

---

## 3. Master Empirical Results Summary

### Table 1: Statistical Forecast Accuracy vs. Daily Risk Management ($N = 997$ Test Days)

| Model | RMSE | MAE | QLIKE | DM vs GARCH ($p$-val) | Daily Turnover | Net Sharpe | Net CAGR | Max DD |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Random Forest** | **0.08497** | 0.05151 | **0.47287** | **$p = 0.0044$ (***)** | 12.72 | 0.371 | 11.23% | -41.42% |
| **XGBoost** | 0.08594 | **0.05072** | 0.48869 | **$p = 0.0025$ (***)** | 17.31 | 0.416 | 12.48% | **-38.90%** |
| **Hybrid GARCH+ML** | 0.08657 | 0.05266 | 0.47664 | **$p = 0.0031$ (***)** | 17.27 | 0.321 | 9.80% | -41.56% |
| **Historical Vol** | 0.10128 | 0.06036 | 0.56519 | $p = 0.7253$ (Eqv) | 8.29 | 0.393 | 11.74% | -35.84% |
| **GARCH(1,1)** | 0.10497 | 0.06320 | 0.78001 | Benchmark | **2.47** | **0.490** | **14.85%** | -41.38% |

### Table 2: Implementation Policy & Rebalancing Cadence

| Model | Implementation Policy | Annual Turnover | Gross CAGR | Net CAGR | Net Sharpe | Cost Drag |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **GARCH** | Daily (1-day) | 2.47 | 15.28% | 14.85% | 0.490 | -0.43% |
| **XGBoost** | Daily (1-day) | 17.31 | 15.44% | 12.48% | 0.416 | -2.96% |
| **XGBoost** | **Weekly (5-day)** | **7.11** | **17.32%** | **16.07%** | **0.530** | **-1.25%** |
| **XGBoost** | **Biweekly (10-day)** | **4.66** | **17.82%** | **17.00%** | **0.558** | **-0.82%** |
| **Random Forest** | Daily (1-day) | 12.72 | 13.37% | 11.23% | 0.371 | -2.14% |
| **Random Forest** | **EMA Smoothed (10d)** | **4.22** | **13.37%** | **13.20%** | **0.436** | **-0.17%** |

---

## 4. Replication Protocol
```bash
pip install -r requirements.txt
python run_all_experiments.py
python src/refinement_experiments.py
python src/final_polish_analysis.py
python src/validate_results.py
```
*Zero discrepancies verified across code, data, tables, figures, paper, presentation, and documentation.*
