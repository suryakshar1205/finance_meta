"""
Hybrid GARCH + Machine Learning Framework Module
=================================================
Implements the integrated hybrid architecture fusing structural econometric
GARCH(1,1) conditional volatility signals with non-linear machine learning regressors.
"""

import logging
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

from src.garch import GARCHModel
from src.ml_models import XGBoostVolatilityModel, RandomForestVolatilityModel

logger = logging.getLogger(__name__)


class HybridGARCHMLModel:
    """
    Hybrid Volatility Forecasting Model:
    Combines econometric GARCH(1,1) conditional volatility estimates as structural
    features fed into non-linear gradient-boosted or ensemble tree regressors.
    """
    def __init__(
        self,
        base_learner: str = "xgboost",
        garch_p: int = 1,
        garch_q: int = 1,
        garch_dist: str = "StudentsT",
        residual_learning: bool = False,
        ml_params: Optional[Dict[str, Any]] = None,
        annualization_factor: int = 252
    ):
        self.base_learner_type = base_learner.lower()
        self.garch_p = garch_p
        self.garch_q = garch_q
        self.garch_dist = garch_dist
        self.residual_learning = residual_learning
        self.annualization_factor = annualization_factor
        self.ml_params = ml_params or {}

        # Sub-components
        self.garch_model = GARCHModel(
            p=self.garch_p,
            q=self.garch_q,
            dist=self.garch_dist,
            annualization_factor=self.annualization_factor
        )

        if self.base_learner_type == "xgboost":
            self.ml_model = XGBoostVolatilityModel(**self.ml_params)
        elif self.base_learner_type == "random_forest":
            self.ml_model = RandomForestVolatilityModel(**self.ml_params)
        else:
            raise ValueError(f"Unsupported base learner '{base_learner}'. Use 'xgboost' or 'random_forest'.")

        self.feature_names: List[str] = []
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series, returns: pd.Series) -> "HybridGARCHMLModel":
        """
        Fit GARCH on historical returns, extract conditional volatility feature,
        and train ML regressor on augmented feature matrix.

        Parameters
        ----------
        X : pd.DataFrame
            Historical feature matrix up to time t.
        y : pd.Series
            Target forward realized volatility.
        returns : pd.Series
            Log returns aligned with X.

        Returns
        -------
        HybridGARCHMLModel
            Fitted instance.
        """
        # 1. Fit GARCH model strictly on past returns
        self.garch_model.fit(returns)
        garch_cond_vol = self.garch_model.get_conditional_volatility(returns)

        # 2. Augment feature matrix with GARCH conditional volatility
        X_aug = X.copy()
        X_aug["garch_cond_vol"] = garch_cond_vol.reindex(X.index).ffill().bfill()

        # 3. Train ML Model
        if self.residual_learning:
            # Predict residual between actual RV and GARCH conditional vol
            target_residual = y - X_aug["garch_cond_vol"]
            self.ml_model.fit(X_aug, target_residual)
        else:
            # Direct prediction using augmented feature set
            self.ml_model.fit(X_aug, y)

        self.feature_names = list(X_aug.columns)
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame, current_returns: pd.Series) -> np.ndarray:
        """
        Generate out-of-sample forward volatility forecasts using the hybrid architecture.

        Parameters
        ----------
        X : pd.DataFrame
            Test/Walk-forward feature matrix.
        current_returns : pd.Series
            Return series up to current observation for GARCH conditional vol alignment.

        Returns
        -------
        np.ndarray
            Predicted forward realized volatility array.
        """
        if not self.is_fitted:
            raise ValueError("Hybrid model is not fitted yet.")

        garch_cond_vol = self.garch_model.get_conditional_volatility(current_returns)
        X_aug = X.copy()
        X_aug["garch_cond_vol"] = garch_cond_vol.reindex(X.index).ffill().bfill()

        if self.residual_learning:
            pred_residual = self.ml_model.predict(X_aug)
            preds = X_aug["garch_cond_vol"].values + pred_residual
        else:
            preds = self.ml_model.predict(X_aug)

        return np.clip(preds, a_min=1e-4, a_max=None)

    def get_feature_importances(self) -> pd.Series:
        """
        Extract feature importances from the ML component of the hybrid model.
        """
        return self.ml_model.get_feature_importances()
