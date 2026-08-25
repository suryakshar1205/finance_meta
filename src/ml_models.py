"""
Machine Learning Volatility Models Module
=========================================
Implements Random Forest and XGBoost regression architectures tailored for
leakage-free financial volatility prediction, including feature importance extraction.
"""

import logging
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

logger = logging.getLogger(__name__)


class RandomForestVolatilityModel:
    """
    Random Forest Regressor for forward realized volatility forecasting.
    """
    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 6,
        min_samples_split: int = 5,
        min_samples_leaf: int = 2,
        random_state: int = 42,
        n_jobs: int = -1
    ):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=n_jobs
        )
        self.feature_names: List[str] = []
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RandomForestVolatilityModel":
        """
        Fit Random Forest regressor on past training data.
        """
        self.feature_names = list(X.columns)
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate out-of-sample forward volatility predictions.
        """
        if not self.is_fitted:
            raise ValueError("RandomForest model is not fitted yet.")
        preds = self.model.predict(X)
        # Volatility is strictly positive
        return np.clip(preds, a_min=1e-4, a_max=None)

    def get_feature_importances(self) -> pd.Series:
        """
        Extract MDI feature importances.
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted.")
        return pd.Series(self.model.feature_importances_, index=self.feature_names).sort_values(ascending=False)


class XGBoostVolatilityModel:
    """
    Gradient Boosted Decision Trees (XGBoost) for forward realized volatility forecasting.
    """
    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 4,
        learning_rate: float = 0.03,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
        n_jobs: int = -1
    ):
        self.model = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state,
            n_jobs=n_jobs,
            verbosity=0
        )
        self.feature_names: List[str] = []
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "XGBoostVolatilityModel":
        """
        Fit XGBoost regressor on past training features and target.
        """
        self.feature_names = list(X.columns)
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate forward volatility predictions.
        """
        if not self.is_fitted:
            raise ValueError("XGBoost model is not fitted yet.")
        preds = self.model.predict(X)
        return np.clip(preds, a_min=1e-4, a_max=None)

    def get_feature_importances(self) -> pd.Series:
        """
        Extract XGBoost gain-based feature importances.
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted.")
        return pd.Series(self.model.feature_importances_, index=self.feature_names).sort_values(ascending=False)
