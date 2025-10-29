import numpy as np
import pandas as pd
from arch import arch_model
from typing import Optional, Dict, Any, Tuple
import warnings

class GARCHModel:
    """
    GARCH (Generalized Autoregressive Conditional Heteroskedasticity) model
    for volatility forecasting.

    Supports various GARCH variants including GARCH, EGARCH, and GJR-GARCH.
    """

    def __init__(self, vol: str = 'GARCH', p: int = 1, q: int = 1,
                 mean: str = 'Constant', dist: str = 'normal'):
        """
        Initialize GARCH model.

        Args:
            vol: Volatility model ('GARCH', 'EGARCH', 'GJRGARCH')
            p: Order of GARCH term
            q: Order of ARCH term
            mean: Mean model ('Constant', 'Zero', 'AR')
            dist: Error distribution ('normal', 't', 'ged')
        """
        self.vol = vol
        self.p = p
        self.q = q
        self.mean = mean
        self.dist = dist
        self.model = None
        self.fitted_model = None
        self.is_fitted = False

    def fit(self, returns: pd.Series, show_summary: bool = False) -> 'GARCHModel':
        """
        Fit GARCH model to return series.

        Args:
            returns: Return time series (usually log returns)
            show_summary: Whether to display model summary

        Returns:
            Self for method chaining
        """
        # Scale returns to percentage for better numerical stability
        scaled_returns = returns * 100

        self.model = arch_model(scaled_returns, vol=self.vol, p=self.p, q=self.q,
                               mean=self.mean, dist=self.dist)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.fitted_model = self.model.fit(disp='off')

        if show_summary:
            print(self.fitted_model.summary())

        self.is_fitted = True
        return self

    def forecast(self, horizon: int = 1, method: str = 'analytic') -> Dict[str, np.ndarray]:
        """
        Generate volatility forecasts.

        Args:
            horizon: Forecast horizon
            method: Forecasting method ('analytic', 'simulation', 'bootstrap')

        Returns:
            Dictionary with volatility forecasts
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before forecasting")

        forecast_result = self.fitted_model.forecast(horizon=horizon, method=method)

        return {
            'volatility': np.sqrt(forecast_result.variance.values[-1, :]),
            'variance': forecast_result.variance.values[-1, :],
            'mean': forecast_result.mean.values[-1, :] if hasattr(forecast_result, 'mean') else None
        }

    def conditional_volatility(self) -> pd.Series:
        """Get conditional volatility estimates."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        return np.sqrt(self.fitted_model.conditional_volatility / 100)

    def standardized_residuals(self) -> pd.Series:
        """Get standardized residuals."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        return self.fitted_model.std_resid

    def arch_lm_test(self, lags: int = 5) -> Dict[str, float]:
        """
        ARCH LM test for remaining ARCH effects in residuals.

        Args:
            lags: Number of lags for test

        Returns:
            Test statistic and p-value
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        try:
            from statsmodels.stats.diagnostic import het_arch
            resid = self.fitted_model.resid
            test_result = het_arch(resid, nlags=lags, ddof=0)
            return {
                'statistic': test_result[0],  # LM statistic
                'pvalue': test_result[1],     # p-value
                'reject_homoskedasticity': test_result[1] < 0.05
            }
        except (ImportError, Exception):
            # Fallback if het_arch is not available or fails
            return {
                'statistic': None,
                'pvalue': None,
                'reject_homoskedasticity': None
            }

    def ljung_box_test(self, lags: int = 10) -> Dict[str, float]:
        """
        Ljung-Box test for serial correlation in standardized residuals.

        Args:
            lags: Number of lags for test

        Returns:
            Test results
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        from statsmodels.stats.diagnostic import acorr_ljungbox
        std_resid = self.standardized_residuals()

        lb_result = acorr_ljungbox(std_resid, lags=lags, return_df=True)

        return {
            'statistic': lb_result['lb_stat'].iloc[-1],
            'pvalue': lb_result['lb_pvalue'].iloc[-1],
            'reject_independence': lb_result['lb_pvalue'].iloc[-1] < 0.05
        }

    def get_parameters(self) -> Dict[str, float]:
        """Get fitted model parameters."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        return dict(zip(self.fitted_model.params.index,
                       self.fitted_model.params.values))

    def aic(self) -> float:
        """Get Akaike Information Criterion."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        return self.fitted_model.aic

    def bic(self) -> float:
        """Get Bayesian Information Criterion."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        return self.fitted_model.bic

    def log_likelihood(self) -> float:
        """Get log likelihood."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        return self.fitted_model.llf