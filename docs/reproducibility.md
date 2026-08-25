# Reproducibility Guide & Verification Workflow

This document provides exact, step-by-step instructions to reproduce the entire quantitative research study and audit all empirical findings.

---

## System Requirements
- **Operating System:** Windows 10/11, macOS, or Linux
- **Python Version:** Python 3.11+
- **Hardware Requirements:** Minimum 8GB RAM, Quad-Core CPU

---

## Step-by-Step Replication Protocol

### Step 1: Environment Setup
Clone or enter the project repository and create a clean Python virtual environment:
```bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

### Step 2: Install Dependencies
Install all pinned dependencies from `requirements.txt`:
```bash
pip install -r requirements.txt
```

### Step 3: Verify Data and Configuration
Inspect `config/final_experiment.yaml` and verify that the data file `data/raw/nifty50_daily_raw.csv` is present.

### Step 4: Run the Master End-to-End Pipeline
Execute the full quantitative pipeline:
```bash
python run_all_experiments.py
```
*Outputs generated:*
- Data cleaning and preprocessing audit logs (`data/processed/nifty50_daily_processed.csv`)
- In-sample and out-of-sample walk-forward predictions across 997 test days (`results/forecasts/walk_forward_forecasts.csv`)
- Initial empirical tables and publication figures (`results/tables/`, `results/figures/`)

### Step 5: Run the Refinement and Sensitivity Suite
Execute advanced econometric diagnostics, Mincer-Zarnowitz regressions, transaction-cost grids, and regime breakdowns:
```bash
python src/refinement_experiments.py
```
*Outputs generated:*
- GARCH standardized residual diagnostic metrics (`results/final/final_model_parameters.csv`)
- Full 5x5 Diebold-Mariano comparison matrix (`results/final/final_dm_comparison_matrix.csv`)
- Mincer-Zarnowitz calibration regressions (`results/final/final_calibration_analysis.csv`)
- Multi-cost sensitivity analysis (`results/final/final_transaction_cost_sensitivity.csv`)
- Volatility regime tercile breakdowns (`results/final/final_regime_metrics.csv`)
- Exploratory EMA forecast smoothing results (`results/final/final_smoothed_ml_extension.csv`)

### Step 6: Run the Final Polish and Sensitivity Script
Formalize rebalancing frequency tradeoffs and generate the claim audit report:
```bash
python src/final_polish_analysis.py
```
*Outputs generated:*
- Rebalancing sensitivity table with cost drag (`results/final/rebalancing_frequency_sensitivity.csv`)
- 12-point verified claim audit (`results/final/claim_audit.csv`)
- Publication figures: `rebalancing_frequency_tradeoff.png` and `forecast_accuracy_vs_economic_utility.png`

### Step 7: Run the Programmatic Consistency Validator
Run the automated consistency assertion suite to verify zero discrepancies across paper, presentation, README, and CSV tables:
```bash
python src/validate_results.py
```
*Expected Output:*
```text
================================================================================
STARTING PROGRAMMATIC RESULTS CONSISTENCY AUDIT
================================================================================
[PASS] Found: results/final/final_forecast_metrics.csv
[PASS] Found: results/final/final_portfolio_metrics.csv
...
================================================================================
PROGRAMMATIC CONSISTENCY AUDIT PASSED: ZERO CRITICAL DISCREPANCIES
================================================================================
```

---

## Output Verification Directory
All final, publication-grade results are stored deterministically in:
- `results/final/`
- `submission/`
