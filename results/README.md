# Results Directory Structure & Artifact Lineage

This directory contains all empirical outputs, forecasts, metric tables, statistical significance tests, and publication-ready figures for the research study:

**"A Comparative and Hybrid Framework for Financial Volatility Forecasting: Statistical Accuracy versus Economic Utility"**

---

## Directory Organization

```text
results/
├── README.md                                  # Directory documentation & lineage index
├── audit_report.md                            # 21-Point methodological quality audit report
│
├── final_primary/                             # FROZEN READ-ONLY PRIMARY EXPERIMENTAL RESULTS
│   ├── final_forecast_metrics.csv
│   ├── final_dm_tests.csv
│   ├── final_dm_comparison_matrix.csv
│   ├── final_portfolio_metrics.csv
│   ├── final_transaction_cost_sensitivity.csv
│   ├── final_turnover_analysis.csv
│   ├── final_calibration_analysis.csv
│   ├── final_model_parameters.csv
│   ├── final_regime_metrics.csv
│   └── final_configuration.yaml
│
├── final/                                     # PRODUCTION FINAL PUBLICATION PACKAGE
│   ├── final_forecast_metrics.csv             # Statistical loss metrics (MAE, RMSE, QLIKE)
│   ├── final_dm_tests.csv                     # Harvey-Leybourne-Newbold DM tests vs GARCH
│   ├── final_dm_comparison_matrix.csv         # Full 5x5 pairwise DM statistic matrix
│   ├── final_portfolio_metrics.csv            # 15% Volatility-targeting backtest metrics
│   ├── final_transaction_cost_sensitivity.csv # 0 to 50 bps cost sensitivity grid
│   ├── final_turnover_analysis.csv            # Rebalance frequency performance metrics
│   ├── rebalancing_frequency_sensitivity.csv  # Formalized rebalance sensitivity with cost drag
│   ├── final_calibration_analysis.csv         # Mincer-Zarnowitz regressions & bias stats
│   ├── final_model_parameters.csv             # GARCH MLE parameters & residual diagnostics
│   ├── final_regime_metrics.csv               # Performance stratified across volatility terciles
│   ├── final_smoothed_ml_extension.csv        # Exploratory EMA forecast smoothing results
│   ├── claim_audit.csv                        # 12-Point verified numerical claim audit
│   ├── final_configuration.yaml               # Frozen experiment YAML configuration
│   └── FINAL_RESEARCH_STATUS.md               # Definitive research status & claim classifications
│
├── baseline/                                  # ORIGINAL EMPIRICAL BASELINE FREEZE
│   ├── config.yaml
│   ├── figures/
│   ├── tables/
│   └── forecasts/
│
├── figures/                                   # 13 PUBLICATION-GRADE FIGURES (300 DPI)
│   ├── 01_price_series.png
│   ├── 02_realized_volatility.png
│   ├── 03_acf_squared_returns.png
│   ├── 04_forecast_comparison.png
│   ├── 05_model_accuracy_comparison.png
│   ├── 06_forecast_accuracy_vs_economic_utility.png
│   ├── 07_equity_curves.png
│   ├── 08_transaction_cost_sensitivity.png
│   ├── 09_rebalancing_frequency_tradeoff.png
│   └── 10_turnover_vs_net_sharpe.png
│
├── tables/                                    # CSV EXPORTS FOR PAPER & REPLICATION
│   ├── forecast_metrics.csv
│   ├── portfolio_metrics.csv
│   ├── dm_comparison_matrix.csv
│   ├── calibration_analysis.csv
│   ├── transaction_cost_sensitivity.csv
│   ├── turnover_analysis.csv
│   ├── rebalancing_frequency_sensitivity.csv
│   └── data_summary.csv
│
└── forecasts/                                 # OUT-OF-SAMPLE WALK-FORWARD PREDICTIONS
    └── walk_forward_forecasts.csv             # 4,985 Out-of-sample forecast records (N = 997 days)
```

---

## Reproducibility Verification
All outputs in `results/final/` are generated deterministically and verified via:
```bash
python src/validate_results.py
```
Zero numerical discrepancies exist between exported CSVs, `paper/paper.md`, `presentation/presentation_slides.md`, and `README.md`.
