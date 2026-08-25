"""
Statistical Forecast Evaluation and Hypothesis Testing Module
=============================================================
Computes academic forecast accuracy metrics (MAE, RMSE, QLIKE) and executes
econometric Diebold-Mariano loss-differential tests with Harvey-Leybourne-Newbold corrections.
"""

import logging
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


def compute_mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean Absolute Error: mean(|actual - predicted|)"""
    return float(np.mean(np.abs(actual - predicted)))


def compute_rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Root Mean Squared Error: sqrt(mean((actual - predicted)^2))"""
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def compute_qlike(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Quasi-Likelihood (QLIKE) robust volatility loss function (Patton, 2011).

    Formula (in variance terms where h = predicted^2, sigma^2 = actual^2):
    QLIKE = mean( sigma^2 / h - ln(sigma^2 / h) - 1 )

    Parameters
    ----------
    actual : np.ndarray
        Actual realized volatility.
    predicted : np.ndarray
        Forecasted volatility.

    Returns
    -------
    float
        Mean QLIKE loss.
    """
    act_var = np.clip(actual ** 2, a_min=1e-8, a_max=None)
    pred_var = np.clip(predicted ** 2, a_min=1e-8, a_max=None)
    ratio = act_var / pred_var
    loss = ratio - np.log(ratio) - 1.0
    return float(np.mean(loss))


def compute_forecast_metrics_table(forecasts_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate unified comparison metrics table across all models evaluated on identical test dates.

    Parameters
    ----------
    forecasts_df : pd.DataFrame
        DataFrame with columns: Date, Actual, Predicted, Model.

    Returns
    -------
    pd.DataFrame
        Table with Model, MAE, RMSE, QLIKE, Relative_RMSE.
    """
    models = forecasts_df["Model"].unique()
    metrics = []

    for m in models:
        sub = forecasts_df[forecasts_df["Model"] == m].dropna(subset=["Actual", "Predicted"])
        act = sub["Actual"].values
        pred = sub["Predicted"].values

        mae = compute_mae(act, pred)
        rmse = compute_rmse(act, pred)
        qlike = compute_qlike(act, pred)

        metrics.append({
            "Model": m,
            "Observations": len(sub),
            "MAE": mae,
            "RMSE": rmse,
            "QLIKE": qlike
        })

    res = pd.DataFrame(metrics)
    
    # Add relative RMSE compared to Historical Volatility baseline
    if "Historical Volatility" in res["Model"].values:
        base_rmse = res.loc[res["Model"] == "Historical Volatility", "RMSE"].values[0]
        res["Relative_RMSE"] = res["RMSE"] / base_rmse
    else:
        res["Relative_RMSE"] = 1.0

    return res.sort_values(by="RMSE")


def diebold_mariano_test(
    e1: np.ndarray,
    e2: np.ndarray,
    h: int = 1,
    loss_type: str = "squared"
) -> Tuple[float, float, str]:
    """
    Diebold-Mariano (1995) test for statistical significance of predictive accuracy
    with Harvey-Leybourne-Newbold (1997) multi-step forecast correction.

    Null Hypothesis H0: E[d_t] = 0 (Both models have equal predictive accuracy).
    Alternative H1: Model 2 has superior accuracy (if test stat < 0 and significant)
                    or Model 1 has superior accuracy (if test stat > 0 and significant).

    Parameters
    ----------
    e1 : np.ndarray
        Forecast errors or loss series of Model 1 (Benchmark).
    e2 : np.ndarray
        Forecast errors or loss series of Model 2 (Competitor).
    h : int
        Forecast horizon (steps ahead).
    loss_type : str
        'squared' (MSE), 'absolute' (MAE), or 'qlike'.

    Returns
    -------
    Tuple[float, float, str]
        dm_stat, p_value, interpretation
    """
    T = len(e1)
    if T < 10:
        return 0.0, 1.0, "Insufficient sample size"

    if loss_type == "squared":
        d = (e1 ** 2) - (e2 ** 2)
    elif loss_type == "absolute":
        d = np.abs(e1) - np.abs(e2)
    else:
        d = e1 - e2

    d_mean = np.mean(d)

    # Auto-covariance estimation with Bartlett kernel up to lag h-1
    gamma_0 = np.var(d, ddof=1)
    autocov = 0.0
    for lag in range(1, h):
        weight = 1.0 - (lag / h)
        cov = np.cov(d[:-lag], d[lag:])[0, 1]
        autocov += 2.0 * weight * cov

    long_run_var = max(1e-12, (gamma_0 + autocov) / T)
    dm_stat = d_mean / np.sqrt(long_run_var)

    # Harvey-Leybourne-Newbold (HLN) small-sample modification factor
    hln_factor = np.sqrt((T + 1 - 2 * h + (h * (h - 1)) / T) / T)
    dm_stat_corrected = dm_stat * hln_factor

    # Two-sided Student's t distribution with T-1 degrees of freedom
    p_value = 2.0 * (1.0 - stats.t.cdf(abs(dm_stat_corrected), df=T - 1))

    if p_value < 0.01:
        sig = "Statistically significant at 1% level"
    elif p_value < 0.05:
        sig = "Statistically significant at 5% level"
    elif p_value < 0.10:
        sig = "Statistically significant at 10% level"
    else:
        sig = "No statistically significant difference (p >= 0.10)"

    return float(dm_stat_corrected), float(p_value), sig


def run_pairwise_dm_tests(
    forecasts_df: pd.DataFrame,
    horizon: int = 5,
    benchmark_model: str = "GARCH"
) -> pd.DataFrame:
    """
    Execute pairwise Diebold-Mariano hypothesis tests for all models against a specified benchmark.

    Parameters
    ----------
    forecasts_df : pd.DataFrame
        Walk-forward forecasts DataFrame.
    horizon : int
        Forecast horizon.
    benchmark_model : str
        Name of baseline/benchmark model.

    Returns
    -------
    pd.DataFrame
        Pairwise test summary table.
    """
    models = [m for m in forecasts_df["Model"].unique() if m != benchmark_model]
    
    # Pivot to get aligned actual and predictions per date
    pivot_df = forecasts_df.pivot(index="Date", columns="Model", values=["Actual", "Predicted"])
    actual_series = pivot_df["Actual"].iloc[:, 0].dropna()
    
    records = []
    
    if benchmark_model not in pivot_df["Predicted"].columns:
        logger.warning(f"Benchmark model '{benchmark_model}' not found in predictions.")
        return pd.DataFrame()

    bench_pred = pivot_df["Predicted"][benchmark_model].reindex(actual_series.index).dropna()
    valid_common_dates = bench_pred.index

    for m in models:
        if m not in pivot_df["Predicted"].columns:
            continue
        m_pred = pivot_df["Predicted"][m].reindex(valid_common_dates).dropna()
        common_idx = m_pred.index
        
        act = actual_series.loc[common_idx].values
        e_bench = act - bench_pred.loc[common_idx].values
        e_m = act - m_pred.values

        dm_stat, p_val, interp = diebold_mariano_test(e_bench, e_m, h=horizon, loss_type="squared")

        # Interpretation direction:
        # positive DM stat means e_bench^2 > e_m^2 -> Model m has LOWER squared error than benchmark
        superiority = f"{m} superior" if (dm_stat > 0 and p_val < 0.05) else (
            f"{benchmark_model} superior" if (dm_stat < 0 and p_val < 0.05) else "Equivalent"
        )

        records.append({
            "Benchmark": benchmark_model,
            "Competitor": m,
            "Loss": "MSE (Squared Error)",
            "DM_Statistic": dm_stat,
            "p_value": p_val,
            "Significance": interp,
            "Conclusion": superiority
        })

    return pd.DataFrame(records)


def generate_dm_comparison_matrix(
    forecasts_df: pd.DataFrame,
    horizon: int = 5,
    loss_type: str = "squared"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate full N x N pairwise Diebold-Mariano test statistic matrix and p-value matrix.

    A positive statistic in row i, column j indicates that Model j has smaller loss than Model i (Model j superior).
    """
    models = list(forecasts_df["Model"].unique())
    pivot_df = forecasts_df.pivot(index="Date", columns="Model", values=["Actual", "Predicted"])
    actual_series = pivot_df["Actual"].iloc[:, 0].dropna()

    dm_stat_matrix = pd.DataFrame(index=models, columns=models, dtype=float)
    dm_pval_matrix = pd.DataFrame(index=models, columns=models, dtype=float)

    for m1 in models:
        for m2 in models:
            if m1 == m2:
                dm_stat_matrix.loc[m1, m2] = 0.0
                dm_pval_matrix.loc[m1, m2] = 1.0
                continue

            pred1 = pivot_df["Predicted"][m1].reindex(actual_series.index).dropna()
            pred2 = pivot_df["Predicted"][m2].reindex(actual_series.index).dropna()
            common_idx = pred1.index.intersection(pred2.index)

            act = actual_series.loc[common_idx].values
            e1 = act - pred1.loc[common_idx].values
            e2 = act - pred2.loc[common_idx].values

            stat, pval, _ = diebold_mariano_test(e1, e2, h=horizon, loss_type=loss_type)
            dm_stat_matrix.loc[m1, m2] = stat
            dm_pval_matrix.loc[m1, m2] = pval

    return dm_stat_matrix, dm_pval_matrix


def compute_mincer_zarnowitz_calibration(forecasts_df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform Mincer-Zarnowitz (1969) OLS forecast calibration regressions:
        RV_t = alpha + beta * \hat{sigma}_t + e_t
    Under ideal unbiased forecast calibration, alpha = 0 and beta = 1.

    Returns
    -------
    pd.DataFrame
        Table with Model, Alpha (intercept), Beta (slope), R_squared, P_val_alpha, P_val_beta
    """
    import statsmodels.api as sm

    models = forecasts_df["Model"].unique()
    records = []

    for m in models:
        sub = forecasts_df[forecasts_df["Model"] == m].dropna(subset=["Actual", "Predicted"])
        y = sub["Actual"].values
        X = sm.add_constant(sub["Predicted"].values)

        ols = sm.OLS(y, X).fit()
        alpha = float(ols.params[0])
        beta = float(ols.params[1])
        r2 = float(ols.rsquared)
        p_alpha = float(ols.pvalues[0])
        p_beta = float(ols.pvalues[1])

        # Bias metrics
        mean_act = float(np.mean(y))
        mean_pred = float(np.mean(sub["Predicted"].values))
        mean_bias = mean_pred - mean_act  # Positive means overprediction on average
        underpred_pct = float(np.mean(sub["Predicted"].values < y) * 100.0)
        overpred_pct = float(np.mean(sub["Predicted"].values > y) * 100.0)

        records.append({
            "Model": m,
            "Mean_Actual": mean_act,
            "Mean_Forecast": mean_pred,
            "Mean_Bias": mean_bias,
            "Underpredict_Pct": underpred_pct,
            "Overpredict_Pct": overpred_pct,
            "MZ_Alpha": alpha,
            "MZ_Beta": beta,
            "MZ_R_Squared": r2,
            "MZ_Pval_Alpha": p_alpha,
            "MZ_Pval_Beta": p_beta
        })

    return pd.DataFrame(records)

