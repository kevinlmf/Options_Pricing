"""
Unified Forecast Interface

Connects time series forecasting models with option pricing models.
Provides end-to-end pipeline from price/volatility prediction to option valuation.
"""

import numpy as np
import pandas as pd
import torch
from typing import Dict, Any, Optional, Tuple, Union
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from time_series_forecasting.classical_models.garch import GARCHModel
from time_series_forecasting.deep_learning.rnn_models import LSTMForecaster, GRUForecaster, RNNTrainer
from models.black_scholes import BlackScholesModel, BSParameters
from models.heston import HestonModel, HestonParameters
from models.sabr import SABRModel, SABRParameters


class ForecastBasedOptionPricer:
    """
    Option pricing with time series forecasting for spot price and volatility.

    Workflow:
    1. Forecast future spot price using LSTM/GRU
    2. Forecast future volatility using GARCH
    3. Use forecasts as inputs to option pricing models
    4. Calculate option prices and Greeks
    """

    def __init__(self,
                 price_model: str = 'lstm',
                 volatility_model: str = 'garch',
                 option_model: str = 'black-scholes'):
        """
        Initialize the forecast-based option pricer.

        Args:
            price_model: Price forecasting model ('lstm', 'gru', 'rnn')
            volatility_model: Volatility forecasting model ('garch', 'egarch', 'gjr-garch')
            option_model: Option pricing model ('black-scholes', 'heston', 'sabr')
        """
        self.price_model_type = price_model
        self.volatility_model_type = volatility_model
        self.option_model_type = option_model

        # Models will be initialized when fit is called
        self.price_forecaster = None
        self.volatility_forecaster = None
        self.price_trainer = None

        self.is_fitted = False

    def fit_price_model(self,
                       price_history: Union[pd.Series, np.ndarray],
                       seq_length: int = 20,
                       hidden_size: int = 64,
                       num_layers: int = 2,
                       epochs: int = 100,
                       verbose: bool = True) -> Dict[str, Any]:
        """
        Fit price forecasting model to historical prices.

        Args:
            price_history: Historical price data
            seq_length: Length of input sequences
            hidden_size: Size of hidden layers
            num_layers: Number of RNN layers
            epochs: Training epochs
            verbose: Print training progress

        Returns:
            Training history
        """
        # Initialize the appropriate model
        if self.price_model_type == 'lstm':
            self.price_forecaster = LSTMForecaster(
                input_size=1,
                hidden_size=hidden_size,
                num_layers=num_layers,
                output_size=1,
                dropout=0.2
            )
        elif self.price_model_type == 'gru':
            self.price_forecaster = GRUForecaster(
                input_size=1,
                hidden_size=hidden_size,
                num_layers=num_layers,
                output_size=1,
                dropout=0.2
            )
        else:
            raise ValueError(f"Unknown price model: {self.price_model_type}")

        # Train the model
        self.price_trainer = RNNTrainer(self.price_forecaster, learning_rate=0.001)
        history = self.price_trainer.fit(
            train_data=price_history,
            seq_length=seq_length,
            forecast_horizon=1,
            epochs=epochs,
            batch_size=32,
            validation_split=0.2,
            verbose=verbose
        )

        return history

    def fit_volatility_model(self,
                            returns: pd.Series,
                            p: int = 1,
                            q: int = 1,
                            show_summary: bool = False) -> GARCHModel:
        """
        Fit GARCH model to returns for volatility forecasting.

        Args:
            returns: Return time series (log returns)
            p: GARCH order
            q: ARCH order
            show_summary: Display model summary

        Returns:
            Fitted GARCH model
        """
        vol_model = 'GARCH' if self.volatility_model_type == 'garch' else self.volatility_model_type.upper()

        self.volatility_forecaster = GARCHModel(vol=vol_model, p=p, q=q)
        self.volatility_forecaster.fit(returns, show_summary=show_summary)

        return self.volatility_forecaster

    def fit(self,
            price_history: Union[pd.Series, np.ndarray],
            seq_length: int = 20,
            price_model_params: Optional[Dict] = None,
            volatility_model_params: Optional[Dict] = None,
            verbose: bool = True) -> Dict[str, Any]:
        """
        Fit both price and volatility models.

        Args:
            price_history: Historical price data
            seq_length: Length of input sequences for price model
            price_model_params: Additional parameters for price model
            volatility_model_params: Additional parameters for volatility model
            verbose: Print training progress

        Returns:
            Dictionary with fitting results
        """
        if isinstance(price_history, np.ndarray):
            price_history = pd.Series(price_history)

        # Fit price model
        if verbose:
            print("Fitting price forecasting model...")
        price_params = price_model_params or {}
        price_history_fit = self.fit_price_model(price_history, seq_length=seq_length, verbose=verbose, **price_params)

        # Calculate returns for volatility model
        returns = np.log(price_history / price_history.shift(1)).dropna()

        # Fit volatility model
        if verbose:
            print("\nFitting volatility forecasting model...")
        vol_params = volatility_model_params or {}
        self.fit_volatility_model(returns, **vol_params)

        self.is_fitted = True

        return {
            'price_model_history': price_history_fit,
            'volatility_model': self.volatility_forecaster
        }

    def forecast(self,
                price_history: Union[pd.Series, np.ndarray],
                seq_length: int = 20,
                price_steps: int = 1,
                volatility_horizon: int = 1) -> Dict[str, Any]:
        """
        Generate forecasts for price and volatility.

        Args:
            price_history: Recent price history for forecasting
            seq_length: Length of input sequence
            price_steps: Number of steps ahead to forecast price
            volatility_horizon: Number of steps ahead to forecast volatility

        Returns:
            Dictionary with price and volatility forecasts
        """
        if not self.is_fitted:
            raise ValueError("Models must be fitted before forecasting")

        # Forecast price
        price_forecast = self.price_trainer.predict(
            data=price_history,
            seq_length=seq_length,
            steps=price_steps
        )

        # Forecast volatility
        vol_forecast = self.volatility_forecaster.forecast(horizon=volatility_horizon)

        return {
            'price_forecast': price_forecast,
            'volatility_forecast': vol_forecast['volatility'],
            'variance_forecast': vol_forecast['variance']
        }

    def price_option(self,
                    S0: float,
                    K: float,
                    T: float,
                    r: float,
                    sigma: float,
                    option_type: str = 'call',
                    q: float = 0.0,
                    model_params: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Price option using the specified option pricing model.

        Args:
            S0: Current spot price
            K: Strike price
            T: Time to maturity
            r: Risk-free rate
            sigma: Volatility
            option_type: 'call' or 'put'
            q: Dividend yield
            model_params: Additional model parameters (for Heston, SABR, etc.)

        Returns:
            Dictionary with option price and Greeks
        """
        if self.option_model_type == 'black-scholes':
            params = BSParameters(S0=S0, K=K, T=T, r=r, sigma=sigma, q=q)
            model = BlackScholesModel(params)

            price = model.call_price() if option_type.lower() == 'call' else model.put_price()
            greeks = model.greeks(option_type)

        elif self.option_model_type == 'heston':
            # Heston model requires additional parameters
            heston_params = model_params or {}
            params = HestonParameters(
                S0=S0, K=K, T=T, r=r, q=q,
                v0=sigma**2,  # Initial variance
                kappa=heston_params.get('kappa', 2.0),
                theta=heston_params.get('theta', sigma**2),
                sigma_v=heston_params.get('sigma_v', 0.3),
                rho=heston_params.get('rho', -0.7)
            )
            model = HestonModel(params)
            price = model.call_price() if option_type.lower() == 'call' else model.put_price()
            greeks = model.greeks(option_type)

        elif self.option_model_type == 'sabr':
            # SABR model parameters
            sabr_params = model_params or {}
            params = SABRParameters(
                S0=S0, K=K, T=T, r=r, q=q,
                alpha=sabr_params.get('alpha', sigma),
                beta=sabr_params.get('beta', 0.5),
                rho=sabr_params.get('rho', -0.3),
                nu=sabr_params.get('nu', 0.3)
            )
            model = SABRModel(params)
            price = model.call_price() if option_type.lower() == 'call' else model.put_price()
            greeks = model.greeks(option_type)

        else:
            raise ValueError(f"Unknown option model: {self.option_model_type}")

        return {
            'price': price,
            'greeks': greeks,
            'model': self.option_model_type,
            'parameters': {
                'S0': S0, 'K': K, 'T': T, 'r': r, 'sigma': sigma, 'q': q
            }
        }

    def forecast_and_price(self,
                          price_history: Union[pd.Series, np.ndarray],
                          K: float,
                          T: float,
                          r: float,
                          option_type: str = 'call',
                          q: float = 0.0,
                          seq_length: int = 20,
                          price_steps: int = 1,
                          volatility_horizon: int = 1,
                          model_params: Optional[Dict] = None) -> Dict[str, Any]:
        """
        End-to-end pipeline: forecast price and volatility, then price option.

        Args:
            price_history: Historical price data
            K: Strike price
            T: Time to maturity
            r: Risk-free rate
            option_type: 'call' or 'put'
            q: Dividend yield
            seq_length: Sequence length for price forecasting
            price_steps: Number of steps ahead to forecast price
            volatility_horizon: Number of steps ahead to forecast volatility
            model_params: Additional parameters for option pricing model

        Returns:
            Dictionary with forecasts and option valuation
        """
        if not self.is_fitted:
            raise ValueError("Models must be fitted before forecasting")

        # Generate forecasts
        forecasts = self.forecast(
            price_history=price_history,
            seq_length=seq_length,
            price_steps=price_steps,
            volatility_horizon=volatility_horizon
        )

        # Use forecasted values for option pricing
        S0_forecast = forecasts['price_forecast'][-1]  # Last forecasted price
        sigma_forecast = forecasts['volatility_forecast'][0]  # First forecasted volatility

        # Price option with forecasted parameters
        option_result = self.price_option(
            S0=S0_forecast,
            K=K,
            T=T,
            r=r,
            sigma=sigma_forecast,
            option_type=option_type,
            q=q,
            model_params=model_params
        )

        # Combine results
        return {
            'forecasts': forecasts,
            'option_valuation': option_result,
            'current_spot': price_history.iloc[-1] if isinstance(price_history, pd.Series) else price_history[-1],
            'forecasted_spot': S0_forecast,
            'forecasted_volatility': sigma_forecast
        }


def create_forecast_pricer(price_model: str = 'lstm',
                          volatility_model: str = 'garch',
                          option_model: str = 'black-scholes') -> ForecastBasedOptionPricer:
    """
    Factory function to create a ForecastBasedOptionPricer.

    Args:
        price_model: Price forecasting model ('lstm', 'gru')
        volatility_model: Volatility forecasting model ('garch', 'egarch', 'gjr-garch')
        option_model: Option pricing model ('black-scholes', 'heston', 'sabr')

    Returns:
        Configured ForecastBasedOptionPricer instance
    """
    return ForecastBasedOptionPricer(
        price_model=price_model,
        volatility_model=volatility_model,
        option_model=option_model
    )
