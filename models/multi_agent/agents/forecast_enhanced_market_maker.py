"""
Forecast-Enhanced Market Maker Agent

Extends the Market Maker agent with time series forecasting capabilities
for improved pricing and risk management decisions.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from models.multi_agent.agents.market_maker import MarketMaker, MarketMakerParameters
from models.multi_agent.agents.base_agent import MarketState
from time_series_forecasting.forecast_interface import ForecastBasedOptionPricer


@dataclass
class ForecastEnhancedParameters(MarketMakerParameters):
    """Extended parameters with forecasting capabilities"""

    # Forecasting parameters
    use_price_forecast: bool = True  # Use LSTM price forecasting
    use_volatility_forecast: bool = True  # Use GARCH volatility forecasting
    forecast_horizon: int = 5  # Days to forecast ahead
    forecast_weight: float = 0.3  # Weight given to forecasts (0-1)

    # Model configuration
    price_model: str = 'lstm'  # 'lstm', 'gru'
    volatility_model: str = 'garch'  # 'garch', 'egarch', 'gjr-garch'

    # Historical data requirements
    min_history_length: int = 100  # Minimum price history needed
    seq_length: int = 20  # Sequence length for LSTM


class ForecastEnhancedMarketMaker(MarketMaker):
    """
    Market Maker with Time Series Forecasting

    Enhances traditional market making with:
    1. LSTM/GRU for spot price forecasting
    2. GARCH for volatility forecasting
    3. Forecast-adjusted pricing and risk management
    4. Dynamic spread adjustment based on forecast uncertainty

    The agent uses forecasts to:
    - Anticipate price movements and adjust inventory targets
    - Improve volatility estimates for option pricing
    - Better manage risk by predicting market conditions
    - Optimize quote skew based on expected price direction
    """

    def __init__(self,
                 agent_id: str,
                 parameters: ForecastEnhancedParameters,
                 initial_cash: float = 2000000.0):
        """
        Initialize forecast-enhanced market maker.

        Parameters:
        -----------
        agent_id : str
            Unique identifier
        parameters : ForecastEnhancedParameters
            Behavior parameters including forecasting config
        initial_cash : float
            Initial capital
        """
        super().__init__(agent_id, parameters, initial_cash)
        self.forecast_params = parameters

        # Forecasting components
        self.forecast_pricer: Optional[ForecastBasedOptionPricer] = None
        self.price_history: List[float] = []
        self.forecast_cache: Dict[str, Dict] = {}  # Cache forecasts

        # Forecast tracking
        self.forecast_accuracy: List[float] = []
        self.is_forecast_ready = False

    def observe_market(self, market_state: MarketState) -> None:
        """
        Observe market and update forecasting models.

        Parameters:
        -----------
        market_state : MarketState
            Current market conditions
        """
        # Call parent observation
        super().observe_market(market_state)

        # Update price history
        self.price_history.append(market_state.underlying_price)

        # Initialize forecasting if enough history
        if len(self.price_history) >= self.forecast_params.min_history_length and not self.is_forecast_ready:
            self._initialize_forecasting()

        # Update forecasts periodically
        if self.is_forecast_ready and len(self.price_history) % 10 == 0:
            self._update_forecasts()

    def _initialize_forecasting(self) -> None:
        """Initialize and train forecasting models."""
        try:
            # Create forecast pricer
            self.forecast_pricer = ForecastBasedOptionPricer(
                price_model=self.forecast_params.price_model,
                volatility_model=self.forecast_params.volatility_model,
                option_model='black-scholes'
            )

            # Convert price history to pandas Series
            price_series = pd.Series(self.price_history)

            # Fit models
            self.forecast_pricer.fit(
                price_history=price_series,
                seq_length=self.forecast_params.seq_length,
                verbose=False
            )

            self.is_forecast_ready = True
            print(f"Agent {self.state.agent_id}: Forecasting models initialized")

        except Exception as e:
            print(f"Agent {self.state.agent_id}: Failed to initialize forecasting: {e}")
            self.is_forecast_ready = False

    def _update_forecasts(self) -> None:
        """Update forecasts based on latest data."""
        if not self.is_forecast_ready or self.forecast_pricer is None:
            return

        try:
            # Generate new forecasts
            price_series = pd.Series(self.price_history)

            forecasts = self.forecast_pricer.forecast(
                price_history=price_series,
                seq_length=self.forecast_params.seq_length,
                price_steps=self.forecast_params.forecast_horizon,
                volatility_horizon=self.forecast_params.forecast_horizon
            )

            # Cache forecasts
            self.forecast_cache = {
                'price_forecast': forecasts['price_forecast'],
                'volatility_forecast': forecasts['volatility_forecast'],
                'timestamp': len(self.price_history)
            }

        except Exception as e:
            print(f"Agent {self.state.agent_id}: Failed to update forecasts: {e}")

    def _generate_quotes(self, market_state: MarketState) -> Dict[Tuple[float, float], Tuple[float, float]]:
        """
        Generate forecast-enhanced bid-ask quotes.

        Parameters:
        -----------
        market_state : MarketState
            Current market state

        Returns:
        --------
        Dict: {(strike, expiry): (bid, ask)} quotes with forecast adjustments
        """
        quotes = {}

        for (strike, expiry), theoretical_price in market_state.theoretical_prices.items():
            # Calculate base spread
            spread = self._calculate_spread(strike, expiry, market_state)

            # Calculate inventory adjustment
            inventory_adjustment = self._calculate_inventory_adjustment(strike, expiry, market_state)

            # Calculate risk adjustment
            risk_adjustment = self._calculate_risk_adjustment(strike, expiry, market_state)

            # Add forecast adjustment
            forecast_adjustment = self._calculate_forecast_adjustment(strike, expiry, market_state)

            # Final quotes with forecast enhancement
            mid_price = (theoretical_price + inventory_adjustment +
                        risk_adjustment + forecast_adjustment)
            half_spread = spread / 2

            bid = max(0.01, mid_price - half_spread)
            ask = mid_price + half_spread

            quotes[(strike, expiry)] = (bid, ask)

        return quotes

    def _calculate_forecast_adjustment(self, strike: float, expiry: float,
                                      market_state: MarketState) -> float:
        """
        Calculate price adjustment based on forecasts.

        Parameters:
        -----------
        strike : float
            Option strike price
        expiry : float
            Option expiry time
        market_state : MarketState
            Current market state

        Returns:
        --------
        float: Forecast-based price adjustment
        """
        if not self.is_forecast_ready or not self.forecast_cache:
            return 0.0

        try:
            # Get forecasts
            price_forecast = self.forecast_cache.get('price_forecast', None)
            vol_forecast = self.forecast_cache.get('volatility_forecast', None)

            if price_forecast is None or vol_forecast is None:
                return 0.0

            # Expected price change
            current_price = market_state.underlying_price
            expected_price = np.mean(price_forecast)  # Average forecasted price
            price_change_pct = (expected_price - current_price) / current_price

            # Expected volatility change
            current_vol = market_state.underlying_volatility
            expected_vol = np.mean(vol_forecast)
            vol_change = expected_vol - current_vol

            # Calculate moneyness
            moneyness = strike / current_price

            # Forecast-based adjustment
            # If expecting price to rise and option is OTM call, increase price
            # If expecting higher volatility, increase all option prices

            price_sensitivity = 1.0 if moneyness > 1.0 else -1.0  # Call OTM vs ITM
            price_adjustment = price_change_pct * price_sensitivity * 0.5

            vol_adjustment = vol_change * 2.0  # Higher vol = higher option value

            # Combine adjustments with forecast weight
            total_adjustment = (price_adjustment + vol_adjustment) * self.forecast_params.forecast_weight

            # Scale by theoretical price
            theoretical_price = market_state.theoretical_prices.get((strike, expiry), 1.0)

            return total_adjustment * theoretical_price

        except Exception as e:
            print(f"Forecast adjustment error: {e}")
            return 0.0

    def _adjust_inventory_target_with_forecast(self, market_state: MarketState) -> None:
        """
        Adjust inventory targets based on price forecasts.

        If forecasting price increase, target long inventory.
        If forecasting price decrease, target short inventory.
        """
        if not self.is_forecast_ready or not self.forecast_cache:
            return

        try:
            price_forecast = self.forecast_cache.get('price_forecast', None)
            if price_forecast is None:
                return

            # Expected price direction
            current_price = market_state.underlying_price
            expected_price = np.mean(price_forecast)
            expected_return = (expected_price - current_price) / current_price

            # Adjust inventory targets based on expected return
            # Positive return -> want long delta
            # Negative return -> want short delta
            for (strike, expiry) in market_state.theoretical_prices.keys():
                instrument = f"CALL_{strike}_{expiry}"

                # Simple rule: target inventory in direction of forecast
                target = expected_return * 10  # Scale factor
                self.inventory_target[instrument] = target

        except Exception as e:
            print(f"Inventory adjustment error: {e}")

    def get_forecast_stats(self) -> Dict:
        """
        Get forecasting performance statistics.

        Returns:
        --------
        Dict: Forecasting metrics
        """
        stats = {
            'is_forecast_ready': self.is_forecast_ready,
            'price_history_length': len(self.price_history),
            'has_cached_forecast': bool(self.forecast_cache),
            'forecast_accuracy': np.mean(self.forecast_accuracy) if self.forecast_accuracy else None,
        }

        if self.forecast_cache:
            stats.update({
                'latest_price_forecast': self.forecast_cache.get('price_forecast', None),
                'latest_vol_forecast': self.forecast_cache.get('volatility_forecast', None),
                'forecast_age': len(self.price_history) - self.forecast_cache.get('timestamp', 0)
            })

        return stats

    def get_performance_summary(self) -> Dict:
        """
        Get combined performance summary including forecasting.

        Returns:
        --------
        Dict: Complete performance metrics
        """
        base_stats = self.get_market_making_stats()
        forecast_stats = self.get_forecast_stats()

        return {**base_stats, 'forecasting': forecast_stats}
