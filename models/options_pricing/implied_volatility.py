"""
Implied Volatility Calculator

This module provides methods to calculate implied volatility from market data,
which is essential for pricing options based on real market conditions.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar, brentq
from typing import Union, Optional
import warnings

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from .black_scholes import BlackScholesModel, BSParameters


class ImpliedVolatilityCalculator:
    """
    Calculator for implied volatility from historical stock price data.

    Provides methods to calculate:
    - Historical volatility from price series
    - Realized volatility with different time windows
    - GARCH-based volatility estimates
    """

    def __init__(self):
        """Initialize the implied volatility calculator."""
        self.min_vol = 0.01  # Minimum volatility (1%)
        self.max_vol = 5.0   # Maximum volatility (500%)

    def historical_volatility(self,
                            price_series: pd.Series,
                            window: int = 252,
                            annualize: bool = True) -> float:
        """
        Calculate historical volatility from price series.

        Parameters:
        -----------
        price_series : pd.Series
            Historical price data
        window : int
            Number of periods for volatility calculation (default: 252 trading days)
        annualize : bool
            Whether to annualize the volatility

        Returns:
        --------
        float : Historical volatility
        """
        if len(price_series) < 2:
            raise ValueError("Need at least 2 price points to calculate volatility")

        # Calculate returns
        returns = price_series.pct_change().dropna()

        if len(returns) < window:
            # Use all available data if window is larger than data
            vol = returns.std()
            if annualize:
                # Annualize based on available data frequency
                periods_per_year = min(252, len(returns))
                vol *= np.sqrt(periods_per_year)
        else:
            # Use rolling window
            rolling_vol = returns.rolling(window=window).std()
            vol = rolling_vol.iloc[-1]  # Latest volatility
            if annualize:
                vol *= np.sqrt(252)  # Assume daily data

        return max(self.min_vol, min(self.max_vol, vol))

    def realized_volatility(self,
                          returns: pd.Series,
                          window: int = 30) -> pd.Series:
        """
        Calculate rolling realized volatility.

        Parameters:
        -----------
        returns : pd.Series
            Return series
        window : int
            Rolling window size

        Returns:
        --------
        pd.Series : Rolling realized volatility
        """
        return returns.rolling(window=window).std() * np.sqrt(252)

    def ewm_volatility(self,
                      returns: pd.Series,
                      halflife: int = 30) -> pd.Series:
        """
        Calculate exponentially weighted moving average volatility.

        Parameters:
        -----------
        returns : pd.Series
            Return series
        halflife : int
            Half-life for exponential weighting

        Returns:
        --------
        pd.Series : EWMA volatility
        """
        return returns.ewm(halflife=halflife).std() * np.sqrt(252)

    def garch_volatility(self,
                        returns: pd.Series,
                        p: int = 1,
                        q: int = 1) -> float:
        """
        Simplified GARCH(p,q) volatility estimation.

        Parameters:
        -----------
        returns : pd.Series
            Return series
        p : int
            GARCH lag order
        q : int
            ARCH lag order

        Returns:
        --------
        float : GARCH volatility estimate
        """
        # Simplified GARCH - use exponentially weighted volatility as proxy
        # In practice, would use statsmodels.tsa.arch
        ewm_vol = self.ewm_volatility(returns, halflife=30)
        return ewm_vol.iloc[-1] if not ewm_vol.empty else self.historical_volatility(returns.cumsum())


class OptionImpliedVolatility:
    """
    Calculator for implied volatility from option prices.

    Uses numerical methods to solve for the volatility that makes
    the theoretical option price equal to the market price.
    """

    def __init__(self, tolerance: float = 1e-6, max_iterations: int = 100):
        """
        Initialize option implied volatility calculator.

        Parameters:
        -----------
        tolerance : float
            Convergence tolerance
        max_iterations : int
            Maximum iterations for numerical solver
        """
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.min_vol = 0.001  # 0.1%
        self.max_vol = 10.0   # 1000%

    def calculate_implied_volatility(self,
                                   market_price: float,
                                   S: float,
                                   K: float,
                                   T: float,
                                   r: float,
                                   option_type: str = 'call',
                                   initial_guess: float = 0.2) -> Optional[float]:
        """
        Calculate implied volatility from market option price.

        Parameters:
        -----------
        market_price : float
            Market price of the option
        S : float
            Current underlying price
        K : float
            Strike price
        T : float
            Time to expiration (years)
        r : float
            Risk-free rate
        option_type : str
            'call' or 'put'
        initial_guess : float
            Initial guess for volatility

        Returns:
        --------
        float : Implied volatility, or None if calculation fails
        """
        def objective_function(sigma):
            """Objective function: |theoretical_price - market_price|"""
            try:
                bs_params = BSParameters(S0=S, K=K, T=T, r=r, sigma=sigma)
                bs_model = BlackScholesModel(bs_params)

                if option_type.lower() == 'call':
                    theoretical_price = bs_model.call_price()
                else:
                    theoretical_price = bs_model.put_price()

                return abs(theoretical_price - market_price)
            except:
                return float('inf')

        try:
            # Use Brent's method for root finding
            result = minimize_scalar(
                objective_function,
                bounds=(self.min_vol, self.max_vol),
                method='bounded',
                options={'xatol': self.tolerance, 'maxiter': self.max_iterations}
            )

            if result.success and result.fun < self.tolerance:
                return result.x
            else:
                warnings.warn(f"Implied volatility calculation failed for {option_type} option")
                return None

        except Exception as e:
            warnings.warn(f"Error calculating implied volatility: {str(e)}")
            return None

    def calculate_vega(self,
                      S: float,
                      K: float,
                      T: float,
                      r: float,
                      sigma: float) -> float:
        """
        Calculate option vega (sensitivity to volatility).

        Parameters:
        -----------
        S, K, T, r, sigma : float
            Black-Scholes parameters

        Returns:
        --------
        float : Option vega
        """
        try:
            bs_params = BSParameters(S0=S, K=K, T=T, r=r, sigma=sigma)
            bs_model = BlackScholesModel(bs_params)
            greeks = bs_model.greeks('call')
            return greeks['vega']
        except:
            return 0.0


class VolatilityEstimator:
    """
    Comprehensive volatility estimation combining multiple methods.
    """

    def __init__(self):
        """Initialize volatility estimator."""
        self.hist_calc = ImpliedVolatilityCalculator()
        self.iv_calc = OptionImpliedVolatility()

    def estimate_volatility(self,
                          price_data: pd.Series,
                          method: str = 'historical',
                          **kwargs) -> float:
        """
        Estimate volatility using specified method.

        Parameters:
        -----------
        price_data : pd.Series
            Historical price data
        method : str
            'historical', 'ewma', 'garch'
        **kwargs : additional parameters for specific methods

        Returns:
        --------
        float : Volatility estimate
        """
        if method == 'historical':
            window = kwargs.get('window', 252)
            return self.hist_calc.historical_volatility(price_data, window)

        elif method == 'ewma':
            returns = price_data.pct_change().dropna()
            halflife = kwargs.get('halflife', 30)
            ewma_vol = self.hist_calc.ewm_volatility(returns, halflife)
            return ewma_vol.iloc[-1] if not ewma_vol.empty else 0.2

        elif method == 'garch':
            returns = price_data.pct_change().dropna()
            return self.hist_calc.garch_volatility(returns)

        else:
            raise ValueError(f"Unknown volatility method: {method}")

    def volatility_term_structure(self,
                                 price_data: pd.Series,
                                 windows: list = [30, 60, 90, 180, 252]) -> dict:
        """
        Calculate volatility term structure for different time horizons.

        Parameters:
        -----------
        price_data : pd.Series
            Historical price data
        windows : list
            List of window sizes (in days)

        Returns:
        --------
        dict : Volatility estimates for different windows
        """
        vol_structure = {}

        for window in windows:
            try:
                vol = self.hist_calc.historical_volatility(price_data, window)
                vol_structure[f'{window}d'] = vol
            except:
                vol_structure[f'{window}d'] = 0.2  # Default fallback

        return vol_structure

    def volatility_forecast(self,
                          price_data: pd.Series,
                          forecast_horizon: int = 30) -> dict:
        """
        Simple volatility forecasting using multiple methods.

        Parameters:
        -----------
        price_data : pd.Series
            Historical price data
        forecast_horizon : int
            Forecast horizon in days

        Returns:
        --------
        dict : Volatility forecasts from different methods
        """
        forecasts = {}

        # Historical average
        forecasts['historical'] = self.estimate_volatility(price_data, 'historical')

        # EWMA
        forecasts['ewma'] = self.estimate_volatility(price_data, 'ewma')

        # GARCH
        forecasts['garch'] = self.estimate_volatility(price_data, 'garch')

        # Simple ensemble average
        forecasts['ensemble'] = np.mean([
            forecasts['historical'],
            forecasts['ewma'],
            forecasts['garch']
        ])

        return forecasts