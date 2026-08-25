"""
Jupyter Notebooks Generator for Quantitative Volatility Research
================================================================
Constructs the complete suite of 12 modular, interactive research notebooks.
Each notebook imports reusable methods from src/, illustrates mathematical
foundations, executes experiments, and visualizes empirical findings.
"""

import os
import nbformat as nbf


def make_notebook(cells):
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    return nb


def create_all_notebooks():
    os.makedirs("notebooks", exist_ok=True)

    # -------------------------------------------------------------
    # 01_data_collection.ipynb
    # -------------------------------------------------------------
    nb1_cells = [
        nbf.v4.new_markdown_cell("""# 01. Data Acquisition & Raw Inspection
**Project:** Comparative & Hybrid Volatility Forecasting Framework  
**Index:** NIFTY 50 (`^NSEI`)

### Research Objective
Acquire complete daily historical OHLCV series from primary market sources without lookahead or survivor bias, persist immutable raw data to `data/raw/`, and inspect raw structure.
"""),
        nbf.v4.new_code_cell("""import sys
sys.path.insert(0, "..")

from src.data_loader import load_config, fetch_market_data

# Load configuration
config = load_config("../config/config.yaml")
ticker_name = config["data"]["ticker"]
start_dt = config["data"]["start_date"]
end_dt = config["data"]["end_date"]

print("Project Configuration Loaded:")
print("  Ticker:", ticker_name)
print("  Date Range: %s to %s" % (start_dt, end_dt))
"""),
        nbf.v4.new_code_cell("""# Fetch / Load Raw Market Data
raw_df = fetch_market_data(
    ticker=ticker_name,
    start_date=start_dt,
    end_date=end_dt,
    save_path="../data/raw/nifty50_daily_raw.csv"
)

print("Total Observations Downloaded:", len(raw_df))
print("First 5 records:")
raw_df.head()
"""),
        nbf.v4.new_code_cell("""# Summary statistics of raw series
raw_df.info()
raw_df.describe()
""")
    ]
    with open("notebooks/01_data_collection.ipynb", "w", encoding="utf-8") as f:
        nbf.write(make_notebook(nb1_cells), f)

    # -------------------------------------------------------------
    # 02_data_cleaning.ipynb
    # -------------------------------------------------------------
    nb2_cells = [
        nbf.v4.new_markdown_cell("""# 02. Data Cleaning & Validation Audit
### Research Objective
Execute programmatic data integrity audits:
- Chronological monotonic sorting
- Duplicate detection & removal
- Missing-value inspection & forward imputation
- Price consistency assertions ($High \\ge Low$, $High \\ge \\max(Open, Close)$, $Price > 0$)
- Calendar trading gaps analysis
"""),
        nbf.v4.new_code_cell("""import sys
sys.path.insert(0, "..")
import json
import pandas as pd
from src.preprocessing import clean_market_data

raw_df = pd.read_csv("../data/raw/nifty50_daily_raw.csv", parse_dates=["Date"], index_col="Date")
clean_df, audit = clean_market_data(raw_df, save_path="../data/processed/nifty50_daily_processed.csv")

print("Data Cleaning & Validation Audit Summary:")
print(json.dumps(audit, indent=2))
"""),
        nbf.v4.new_code_cell("""# Verify cleaned dataset properties
print("Cleaned Shape:", clean_df.shape)
print("Null Values Count:")
print(clean_df.isna().sum())
clean_df.tail()
""")
    ]
    with open("notebooks/02_data_cleaning.ipynb", "w", encoding="utf-8") as f:
        nbf.write(make_notebook(nb2_cells), f)

    # -------------------------------------------------------------
    # 03_returns_volatility.ipynb
    # -------------------------------------------------------------
    nb3_cells = [
        nbf.v4.new_markdown_cell("""# 03. Return Dynamics & Rolling Volatility Estimators
### Mathematical Definitions
1. **Simple Return**: $R_t = \\frac{P_t}{P_{t-1}} - 1$
2. **Logarithmic Return**: $r_t = \\ln\\left(\\frac{P_t}{P_{t-1}}\\right) = \\ln(1 + R_t)$
3. **Annualized Rolling Volatility**: $\\sigma_{ann, w} = \\sqrt{252} \\times \\text{std}(r_{t-w+1:t})$
4. **Parkinson (1980) Range Volatility**:
   $$\\sigma_{Parkinson}^2 = \\frac{252}{4 \\ln 2} \\frac{1}{N} \\sum_{i=1}^N \\ln\\left(\\frac{High_i}{Low_i}\\right)^2$$
5. **Garman-Klass (1980) Microstructure Volatility**:
   $$\\sigma_{GK}^2 = \\frac{252}{N} \\sum_{i=1}^N \\left[ 0.5 \\ln\\left(\\frac{H_i}{L_i}\\right)^2 - (2\\ln 2 - 1) \\ln\\left(\\frac{C_i}{O_i}\\right)^2 \\right]$$
"""),
        nbf.v4.new_code_cell("""import sys
sys.path.insert(0, "..")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.volatility import (
    compute_returns,
    compute_rolling_volatility,
    compute_parkinson_volatility,
    compute_garman_klass_volatility
)

df = pd.read_csv("../data/processed/nifty50_daily_processed.csv", parse_dates=["Date"], index_col="Date")
df = compute_returns(df, price_col="Close")
df = compute_rolling_volatility(df, return_col="log_return", windows=[5, 10, 20, 30, 60])
df["vol_parkinson_20d"] = compute_parkinson_volatility(df, window=20)
df["vol_garman_klass_20d"] = compute_garman_klass_volatility(df, window=20)

df[["log_return", "vol_ann_20d", "vol_parkinson_20d", "vol_garman_klass_20d"]].tail()
"""),
        nbf.v4.new_code_cell("""# Compare Volatility Estimators
plt.figure(figsize=(11, 5))
plt.plot(df.index, df["vol_ann_20d"] * 100, label="Close-to-Close Rolling (20d)", lw=1.2)
plt.plot(df.index, df["vol_parkinson_20d"] * 100, label="Parkinson Range (20d)", lw=1.2, alpha=0.8)
plt.plot(df.index, df["vol_garman_klass_20d"] * 100, label="Garman-Klass (20d)", lw=1.2, alpha=0.8)
plt.title("NIFTY 50: Comparison of Realized Volatility Estimators")
plt.ylabel("Annualized Volatility (%)")
plt.xlabel("Date")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()
""")
    ]
    with open("notebooks/03_returns_volatility.ipynb", "w", encoding="utf-8") as f:
        nbf.write(make_notebook(nb3_cells), f)

    # -------------------------------------------------------------
    # 04_eda.ipynb
    # -------------------------------------------------------------
    nb4_cells = [
        nbf.v4.new_markdown_cell("""# 04. Exploratory Data Analysis & Stylized Facts of Volatility
### Stylized Facts of Financial Returns
1. **Fat Tails & Leptokurtosis**: Return distribution exhibits heavy tails and high excess kurtosis compared to standard Gaussian.
2. **Absence of Linear Autocorrelation**: Asset returns themselves show near-zero autocorrelation (efficient markets).
3. **Volatility Clustering & Long Memory (ARCH Effect)**: Squared returns $r_t^2$ and absolute returns $|r_t|$ display persistent, statistically significant autocorrelation across dozens of lags.
"""),
        nbf.v4.new_code_cell("""import sys
sys.path.insert(0, "..")
import pandas as pd
import matplotlib.pyplot as plt
from src.visualization import (
    plot_price_and_returns,
    plot_return_distribution,
    plot_volatility_clustering_acf,
    plot_rolling_volatilities
)

df = pd.read_csv("../data/processed/nifty50_daily_processed.csv", parse_dates=["Date"], index_col="Date")
from src.volatility import compute_returns, compute_rolling_volatility
df = compute_returns(df)
df = compute_rolling_volatility(df, windows=[5, 20, 60])

fig1 = plot_price_and_returns(df)
fig2 = plot_return_distribution(df["log_return"])
fig3 = plot_volatility_clustering_acf(df["log_return"])
fig4 = plot_rolling_volatilities(df)
""")
    ]
    with open("notebooks/04_eda.ipynb", "w", encoding="utf-8") as f:
        nbf.write(make_notebook(nb4_cells), f)

    # -------------------------------------------------------------
    # 05_baseline.ipynb
    # -------------------------------------------------------------
    nb5_cells = [
        nbf.v4.new_markdown_cell("""# 05. Benchmark 1: Historical Volatility Baseline
### Model Definition
The simple historical volatility baseline forecasts forward $k$-day realized volatility using the rolling sample standard deviation of past returns over a window $W$ (default: 20 trading days):
$$\\hat{\\sigma}_{t, t+k}^{Hist} = \\sqrt{252} \\times \\sqrt{\\frac{1}{W-1} \\sum_{i=0}^{W-1} (r_{t-i} - \\bar{r})^2}$$
"""),
        nbf.v4.new_code_cell("""import sys
sys.path.insert(0, "..")
import pandas as pd
from src.features import build_feature_dataset
from src.volatility import HistoricalVolatilityBaseline
from src.evaluation import compute_mae, compute_rmse, compute_qlike

df = pd.read_csv("../data/processed/nifty50_daily_processed.csv", parse_dates=["Date"], index_col="Date")
feat_df = build_feature_dataset(df, target_horizon=5)

model = HistoricalVolatilityBaseline(window=20)
preds = model.predict(feat_df)
actual = feat_df["target_rv_5d"]

valid = preds.notna() & actual.notna()
mae = compute_mae(actual[valid].values, preds[valid].values)
rmse = compute_rmse(actual[valid].values, preds[valid].values)
qlike = compute_qlike(actual[valid].values, preds[valid].values)

print("Historical Volatility Baseline (20d):")
print("  MAE:   %.5f" % mae)
print("  RMSE:  %.5f" % rmse)
print("  QLIKE: %.5f" % qlike)
""")
    ]
    with open("notebooks/05_baseline.ipynb", "w", encoding="utf-8") as f:
        nbf.write(make_notebook(nb5_cells), f)

    # -------------------------------------------------------------
    # 06_garch.ipynb
    # -------------------------------------------------------------
    nb6_cells = [
        nbf.v4.new_markdown_cell("""# 06. Benchmark 2: Econometric GARCH(1,1)
### Mathematical Formulation
$$\\sigma_t^2 = \\omega + \\alpha \\epsilon_{t-1}^2 + \\beta \\sigma_{t-1}^2$$
- $\\omega > 0, \\alpha \\ge 0, \\beta \\ge 0$
- **Volatility Persistence**: $\\lambda = \\alpha + \\beta$ (must be $< 1$ for covariance stationarity)
- **Half-Life of Shocks**: $t_{1/2} = \\frac{\\ln(0.5)}{\\ln(\\alpha + \\beta)}$
- **Unconditional Long-Term Variance**: $\\sigma_{uncond}^2 = \\frac{\\omega}{1 - \\alpha - \\beta}$
"""),
        nbf.v4.new_code_cell("""import sys
sys.path.insert(0, "..")
import pandas as pd
import json
from src.garch import GARCHModel

df = pd.read_csv("../data/processed/nifty50_daily_processed.csv", parse_dates=["Date"], index_col="Date")
from src.volatility import compute_returns
df = compute_returns(df)

train_slice = df.loc[:"2018-12-31"]["log_return"]
garch = GARCHModel(p=1, q=1, dist="StudentsT")
garch.fit(train_slice)

print("Fitted GARCH(1,1) In-Sample Summary:")
print(json.dumps(garch.params_summary, indent=2))
""")
    ]
    with open("notebooks/06_garch.ipynb", "w", encoding="utf-8") as f:
        nbf.write(make_notebook(nb6_cells), f)

    # -------------------------------------------------------------
    # 07_ml_models.ipynb
    # -------------------------------------------------------------
    nb7_cells = [
        nbf.v4.new_markdown_cell("""# 07. Machine Learning Models: Random Forest & XGBoost
### Implementation
- Supervised regression predicting strictly forward $k$-day realized volatility $RV_{t, t+k}$.
- Leakage-free features: return lags, historical volatilities, parkinson range, momentum, volume volatility.
- Hyperparameters tuned chronologically on past training/validation sets without lookahead bias.
"""),
        nbf.v4.new_code_cell("""import sys
sys.path.insert(0, "..")
import pandas as pd
from src.features import build_feature_dataset
from src.ml_models import RandomForestVolatilityModel, XGBoostVolatilityModel
from src.validation import chronological_split

df = pd.read_csv("../data/processed/nifty50_daily_processed.csv", parse_dates=["Date"], index_col="Date")
feat_df = build_feature_dataset(df, target_horizon=5)

target_col = "target_rv_5d"
feature_cols = [c for c in feat_df.columns if c not in [target_col, "Close", "log_return", "simple_return"]]

train_df, val_df, test_df = chronological_split(feat_df, train_end="2018-12-31", val_end="2020-12-31")

rf = RandomForestVolatilityModel(n_estimators=200, max_depth=6).fit(train_df[feature_cols], train_df[target_col])
xgb = XGBoostVolatilityModel(n_estimators=200, max_depth=4, learning_rate=0.03).fit(train_df[feature_cols], train_df[target_col])

print("Top 5 RF Features:")
print(rf.get_feature_importances().head(5))
print("")
print("Top 5 XGBoost Features:")
print(xgb.get_feature_importances().head(5))
""")
    ]
    with open("notebooks/07_ml_models.ipynb", "w", encoding="utf-8") as f:
        nbf.write(make_notebook(nb7_cells), f)

    # -------------------------------------------------------------
    # 08_hybrid_model.ipynb
    # -------------------------------------------------------------
    nb8_cells = [
        nbf.v4.new_markdown_cell("""# 08. Proposed Hybrid Framework: GARCH + Machine Learning
### Architecture
Combines econometric structural time-series information ($\hat{\sigma}_{GARCH, t}$) with non-linear feature representations:
$$\\hat{\\sigma}_{t, t+k}^{Hybrid} = f_{ML}\\left(\\hat{\\sigma}_{GARCH, t}, \\mathbf{x}_t^{returns}, \\mathbf{x}_t^{volatility}, \\mathbf{x}_t^{range}, \\mathbf{x}_t^{volume}\\right)$$
"""),
        nbf.v4.new_code_cell("""import sys
sys.path.insert(0, "..")
import pandas as pd
from src.features import build_feature_dataset
from src.hybrid import HybridGARCHMLModel
from src.validation import chronological_split

df = pd.read_csv("../data/processed/nifty50_daily_processed.csv", parse_dates=["Date"], index_col="Date")
feat_df = build_feature_dataset(df, target_horizon=5)

target_col = "target_rv_5d"
feature_cols = [c for c in feat_df.columns if c not in [target_col, "Close", "log_return", "simple_return"]]
train_df, val_df, test_df = chronological_split(feat_df, train_end="2018-12-31", val_end="2020-12-31")

hybrid = HybridGARCHMLModel(base_learner="xgboost")
hybrid.fit(train_df[feature_cols], train_df[target_col], train_df["log_return"])

print("Hybrid Model Fitted Successfully.")
print("Top Feature Importances in Hybrid Architecture:")
print(hybrid.get_feature_importances().head(8))
""")
    ]
    with open("notebooks/08_hybrid_model.ipynb", "w", encoding="utf-8") as f:
        nbf.write(make_notebook(nb8_cells), f)

    # -------------------------------------------------------------
    # 09_walk_forward.ipynb
    # -------------------------------------------------------------
    nb9_cells = [
        nbf.v4.new_markdown_cell("""# 09. Out-of-Sample Walk-Forward Validation
### Protocol
- Expanding/Rolling window cross-validation across 997+ trading days.
- Periodic institutional re-estimation every 20 days.
- Zero lookahead leakage: all forecasts evaluated on strictly out-of-sample forward observations.
"""),
        nbf.v4.new_code_cell("""import sys
sys.path.insert(0, "..")
import pandas as pd
from src.data_loader import load_config
from src.features import build_feature_dataset
from src.validation import run_walk_forward_evaluation

config = load_config("../config/config.yaml")
df = pd.read_csv("../data/processed/nifty50_daily_processed.csv", parse_dates=["Date"], index_col="Date")
feat_df = build_feature_dataset(df, target_horizon=5)

target_col = "target_rv_5d"
feature_cols = [c for c in feat_df.columns if c not in [target_col, "Close", "log_return", "simple_return"]]

forecasts_df = run_walk_forward_evaluation(feat_df, target_col=target_col, feature_cols=feature_cols, config=config)
print("Walk-forward forecast records:")
forecasts_df.head(10)
""")
    ]
    with open("notebooks/09_walk_forward.ipynb", "w", encoding="utf-8") as f:
        nbf.write(make_notebook(nb9_cells), f)

    # -------------------------------------------------------------
    # 10_evaluation.ipynb
    # -------------------------------------------------------------
    nb10_cells = [
        nbf.v4.new_markdown_cell("""# 10. Statistical Forecast Evaluation & Hypothesis Testing
### Statistical Methodology
1. **Loss Metrics**: MAE, RMSE, QLIKE
2. **Diebold-Mariano (1995) Test** with Harvey-Leybourne-Newbold (1997) autocorrelation correction.
"""),
        nbf.v4.new_code_cell("""import sys
sys.path.insert(0, "..")
import pandas as pd
from src.evaluation import compute_forecast_metrics_table, run_pairwise_dm_tests

forecasts_df = pd.read_csv("../results/forecasts/walk_forward_forecasts.csv", parse_dates=["Date"])

print("--- Out-of-Sample Forecast Accuracy Table ---")
metrics_table = compute_forecast_metrics_table(forecasts_df)
print(metrics_table.to_string(index=False))
print("")
print("--- Diebold-Mariano Pairwise Statistical Significance vs GARCH ---")
dm_tests = run_pairwise_dm_tests(forecasts_df, horizon=5, benchmark_model="GARCH")
print(dm_tests.to_string(index=False))
""")
    ]
    with open("notebooks/10_evaluation.ipynb", "w", encoding="utf-8") as f:
        nbf.write(make_notebook(nb10_cells), f)

    # -------------------------------------------------------------
    # 11_portfolio.ipynb
    # -------------------------------------------------------------
    nb11_cells = [
        nbf.v4.new_markdown_cell("""# 11. Economic Risk Management & Volatility Targeting Backtest
### Dynamic Asset Allocation
$$w_t = \\min\\left( \\max\\left( \\frac{\\sigma_{target}}{\\hat{\\sigma}_t}, w_{min} \\right), w_{max} \\right)$$
Includes transaction costs and execution slippage deductions.
"""),
        nbf.v4.new_code_cell("""import sys
sys.path.insert(0, "..")
import pandas as pd
from src.data_loader import load_config
from src.backtesting import compare_all_strategies

config = load_config("../config/config.yaml")
forecasts_df = pd.read_csv("../results/forecasts/walk_forward_forecasts.csv", parse_dates=["Date"])
prices = pd.read_csv("../data/processed/nifty50_daily_processed.csv", parse_dates=["Date"], index_col="Date")["Close"]

summary_df, daily_results = compare_all_strategies(forecasts_df, prices, config)
print("Institutional Portfolio & Risk Performance Summary:")
print(summary_df.to_string(index=False))
""")
    ]
    with open("notebooks/11_portfolio.ipynb", "w", encoding="utf-8") as f:
        nbf.write(make_notebook(nb11_cells), f)

    # -------------------------------------------------------------
    # 12_robustness.ipynb
    # -------------------------------------------------------------
    nb12_cells = [
        nbf.v4.new_markdown_cell("""# 12. Robustness Analysis: Multi-Horizon & Market Regimes
### Robustness Checks
1. Evaluation across $k \\in \\{1, 5, 10, 20\\}$ forward trading days.
2. Market Regimes: Low Volatility vs High Volatility / Crisis stress periods.
"""),
        nbf.v4.new_code_cell("""import sys
sys.path.insert(0, "..")
import pandas as pd

rob_df = pd.read_csv("../results/tables/robustness_horizons_metrics.csv")
reg_df = pd.read_csv("../results/tables/regime_forecast_metrics.csv")

print("--- Multi-Horizon Robustness Metrics ---")
print(rob_df.to_string(index=False))
print("")
print("--- Regime Performance Metrics ---")
print(reg_df.to_string(index=False))
""")
    ]
    with open("notebooks/12_robustness.ipynb", "w", encoding="utf-8") as f:
        nbf.write(make_notebook(nb12_cells), f)

    print("All 12 Jupyter Notebooks successfully generated in 'notebooks/'")


if __name__ == "__main__":
    create_all_notebooks()
