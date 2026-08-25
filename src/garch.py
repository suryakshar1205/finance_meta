"""
Econometric GARCH Modeling Module
=================================
Implements GARCH(1,1) estimation, diagnostics, parameter extraction,
conditional volatility series generation, and forward out-of-sample volatility forecasting.
"""

import logging
from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from arch import arch_model

logger = logging.getLogger(__name__)


class GARCHModel:
    """
    Econometric GARCH(1,1) volatility model wrapper with robust convergence handling.
    """
    def __init__(
        self,
        p: int = 1,
        q: int = 1,
        mean: str = "Constant",
        dist: str = "StudentsT",
        rescale: bool = True,
        annualization_factor: int = 252
    ):
        """
        Parameters
        ----------
        p : int
            Symmetric GARCH order (lagged variance).
        q : int
            ARCH order (lagged squared innovation).
        mean : str
            Mean model specification ('Constant' or 'Zero').
        dist : str
            Error distribution ('StudentsT' or 'Normal').
        rescale : bool
            Rescale returns by 100 for numerical optimizer stability.
        annualization_factor : int
            Trading days per year.
        """
        self.p = p
        self.q = q
        self.mean = mean
        self.dist = dist
        self.rescale = rescale
        self.annualization_factor = annualization_factor
        self.fitted_model = None
        self.params_summary: Dict[str, Any] = {}

    def fit(self, returns: pd.Series) -> "GARCHModel":
        """
        Fit GARCH(1,1) model using Maximum Likelihood Estimation strictly on historical returns.

        Parameters
        ----------
        returns : pd.Series
            Log return series.

        Returns
        -------
        GARCHModel
            Self fitted instance.
        """
        clean_ret = returns.dropna()
        scale_factor = 100.0 if self.rescale else 1.0
        scaled_ret = clean_ret * scale_factor

        try:
            am = arch_model(
                scaled_ret,
                p=self.p,
                q=self.q,
                mean=self.mean,
                dist=self.dist,
                rescale=False
            )
            res = am.fit(disp="off", show_warning=False)
            self.fitted_model = res

            # Extract model parameters
            params = res.params.to_dict()
            omega = params.get("omega", np.nan) / (scale_factor ** 2 if self.rescale else 1.0)
            alpha = params.get("alpha[1]", np.nan)
            beta = params.get("beta[1]", np.nan)
            persistence = alpha + beta

            # Half-life of volatility shocks in trading days: ln(0.5) / ln(persistence)
            half_life = np.log(0.5) / np.log(persistence) if 0 < persistence < 1 else np.nan

            self.params_summary = {
                "omega": omega,
                "alpha": alpha,
                "beta": beta,
                "persistence": persistence,
                "half_life_days": half_life,
                "aic": res.aic,
                "bic": res.bic,
                "log_likelihood": res.loglikelihood,
                "converged": res.convergence_flag == 0
            }
        except Exception as e:
            logger.warning(f"GARCH optimization error: {e}. Falling back to default parameters.")
            self.fitted_model = None
            self.params_summary = {
                "omega": 1e-5, "alpha": 0.08, "beta": 0.90, "persistence": 0.98,
                "half_life_days": 34.0, "converged": False
            }

        return self

    def get_conditional_volatility(self, returns: pd.Series) -> pd.Series:
        """
        Get in-sample annualized conditional volatility series: sigma_t * sqrt(252).
        """
        if self.fitted_model is None:
            # Fallback rolling volatility if model failed to converge
            return returns.rolling(20, min_periods=5).std(ddof=1) * np.sqrt(self.annualization_factor)

        cond_vol = self.fitted_model.conditional_volatility
        if self.rescale:
            cond_vol = cond_vol / 100.0

        # Annualize
        ann_cond_vol = cond_vol * np.sqrt(self.annualization_factor)
        ann_cond_vol.name = "garch_cond_vol_ann"
        return ann_cond_vol

    def forecast_horizon_volatility(
        self,
        horizon: int = 5,
        last_return: Optional[float] = None
    ) -> float:
        """
        Generate k-step forward cumulative annualized volatility forecast.

        Parameters
        ----------
        horizon : int
            Forecast horizon k in trading days.
        last_return : Optional[float]
            Most recent return observation.

        Returns
        -------
        float
            Annualized volatility forecast over the k-day forward horizon.
        """
        if self.fitted_model is None:
            # Safe default ~15% annualized if uninitialized
            return 0.15

        try:
            f = self.fitted_model.forecast(horizon=horizon, reindex=False)
            # Forecast variances across horizons [1..k]
            var_forecasts = f.variance.values[-1]  # Array of length horizon
            if self.rescale:
                var_forecasts = var_forecasts / (100.0 ** 2)

            # Cumulative average daily variance over horizon
            mean_daily_var = np.mean(var_forecasts)
            ann_vol_forecast = np.sqrt(mean_daily_var * self.annualization_factor)
            return float(ann_vol_forecast)
        except Exception as e:
            logger.warning(f"GARCH forecast generation error: {e}")
            return 0.15

    def run_residual_diagnostics(self, lags: int = 10) -> Dict[str, Any]:
        """
        Execute formal econometric diagnostic tests on fitted GARCH standardized residuals.
        Tests for:
          1. Remaining autocorrelation in standardized residuals e_t (Ljung-Box Q)
          2. Remaining ARCH effect in squared standardized residuals e_t^2 (Ljung-Box Q & ARCH-LM)
        """
        if self.fitted_model is None:
            return {"status": "Model not fitted"}

        from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch

        std_resid = self.fitted_model.std_resid.dropna()
        lb_raw = acorr_ljungbox(std_resid, lags=[lags], return_df=True)
        lb_sq = acorr_ljungbox(std_resid ** 2, lags=[lags], return_df=True)

        try:
            arch_test = het_arch(std_resid, nlags=lags)
            arch_lm_stat, arch_lm_pval = float(arch_test[0]), float(arch_test[1])
        except Exception:
            arch_lm_stat, arch_lm_pval = np.nan, np.nan

        diag = {
            "ljung_box_resid_stat": float(lb_raw["lb_stat"].values[0]),
            "ljung_box_resid_pval": float(lb_raw["lb_pvalue"].values[0]),
            "ljung_box_sq_resid_stat": float(lb_sq["lb_stat"].values[0]),
            "ljung_box_sq_resid_pval": float(lb_sq["lb_pvalue"].values[0]),
            "arch_lm_stat": arch_lm_stat,
            "arch_lm_pval": arch_lm_pval,
            "mean_std_resid": float(std_resid.mean()),
            "var_std_resid": float(std_resid.var(ddof=1)),
            "skew_std_resid": float(std_resid.skew()),
            "kurt_std_resid": float(std_resid.kurtosis())
        }
        return diag

