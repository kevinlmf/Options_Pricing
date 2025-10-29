"""
Time Series Driven Multi-Agent System for Option Pricing

This module implements agents that use time series forecasting to make trading decisions.
The market price emerges from the interaction of these agents.

Architecture:
1. Each agent has its own time series models (LSTM for price, GARCH for volatility)
2. Agents make decisions based on their forecasts
3. Market environment facilitates trading and price discovery
4. Final option prices emerge from agent interactions
5. Compare with traditional Black-Scholes pricing
"""

import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from time_series_forecasting.deep_learning.rnn_models import LSTMForecaster, RNNTrainer
from time_series_forecasting.classical_models.garch import GARCHModel


class AgentRole(Enum):
    """Agent roles in the market"""
    MARKET_MAKER = "market_maker"
    INFORMED_TRADER = "informed_trader"
    ARBITRAGEUR = "arbitrageur"
    NOISE_TRADER = "noise_trader"


@dataclass
class AgentForecast:
    """Forecast data for an agent"""
    price_forecast: np.ndarray
    volatility_forecast: np.ndarray
    confidence: float  # Confidence in the forecast (0-1)
    timestamp: int


@dataclass
class Order:
    """Order placed by an agent"""
    agent_id: str
    option_key: Tuple[float, float]  # (strike, expiry)
    side: str  # 'buy' or 'sell'
    quantity: float
    limit_price: float
    timestamp: int


@dataclass
class Trade:
    """Executed trade"""
    buyer_id: str
    seller_id: str
    option_key: Tuple[float, float]
    quantity: float
    price: float
    timestamp: int


class TimeSeriesDrivenAgent:
    """
    Base class for agents that use time series forecasting for decision making.

    Each agent:
    1. Maintains price history
    2. Trains LSTM for price prediction
    3. Trains GARCH for volatility prediction
    4. Makes trading decisions based on forecasts
    """

    def __init__(self,
                 agent_id: str,
                 role: AgentRole,
                 initial_cash: float = 1000000.0,
                 seq_length: int = 20,
                 forecast_horizon: int = 5):
        """
        Initialize time series driven agent.

        Parameters:
        -----------
        agent_id : str
            Unique identifier
        role : AgentRole
            Agent's role in the market
        initial_cash : float
            Initial capital
        seq_length : int
            Sequence length for LSTM
        forecast_horizon : int
            How many steps to forecast ahead
        """
        self.agent_id = agent_id
        self.role = role
        self.cash = initial_cash
        self.initial_cash = initial_cash
        self.seq_length = seq_length
        self.forecast_horizon = forecast_horizon

        # Portfolio
        self.positions: Dict[Tuple[float, float], float] = {}  # {(K,T): quantity}

        # Price history
        self.price_history: List[float] = []

        # Models
        self.price_model: Optional[LSTMForecaster] = None
        self.vol_model: Optional[GARCHModel] = None
        self.price_trainer: Optional[RNNTrainer] = None

        # Forecasts
        self.current_forecast: Optional[AgentForecast] = None

        # Trading history
        self.trades: List[Trade] = []
        self.pnl_history: List[float] = []

        # State
        self.is_model_ready = False
        self.min_history_length = 100

    def observe_market(self, spot_price: float) -> None:
        """
        Observe current market spot price.

        Parameters:
        -----------
        spot_price : float
            Current underlying price
        """
        self.price_history.append(spot_price)

        # Initialize models when enough history
        if len(self.price_history) >= self.min_history_length and not self.is_model_ready:
            self._initialize_models()

        # Update forecasts periodically
        if self.is_model_ready and len(self.price_history) % 10 == 0:
            self._update_forecasts()

    def _initialize_models(self) -> None:
        """Initialize and train time series models."""
        try:
            # Initialize LSTM for price prediction
            self.price_model = LSTMForecaster(
                input_size=1,
                hidden_size=64,
                num_layers=2,
                output_size=1,
                dropout=0.2
            )

            # Train price model
            self.price_trainer = RNNTrainer(self.price_model, learning_rate=0.001)
            price_series = pd.Series(self.price_history)

            self.price_trainer.fit(
                train_data=price_series,
                seq_length=self.seq_length,
                forecast_horizon=1,
                epochs=50,
                batch_size=32,
                validation_split=0.2,
                verbose=False
            )

            # Initialize GARCH for volatility
            returns = np.log(price_series / price_series.shift(1)).dropna()
            self.vol_model = GARCHModel(vol='GARCH', p=1, q=1)
            self.vol_model.fit(returns, show_summary=False)

            self.is_model_ready = True
            print(f"Agent {self.agent_id}: Models initialized")

        except Exception as e:
            print(f"Agent {self.agent_id}: Model initialization failed - {e}")
            self.is_model_ready = False

    def _update_forecasts(self) -> None:
        """Update forecasts based on latest data."""
        if not self.is_model_ready:
            return

        try:
            # Price forecast
            price_series = pd.Series(self.price_history)
            price_pred = self.price_trainer.predict(
                data=price_series,
                seq_length=self.seq_length,
                steps=self.forecast_horizon
            )

            # Volatility forecast
            returns = np.log(price_series / price_series.shift(1)).dropna()
            vol_result = self.vol_model.forecast(horizon=self.forecast_horizon)
            vol_pred = vol_result['volatility']

            # Calculate confidence based on recent forecast accuracy
            confidence = self._calculate_confidence()

            self.current_forecast = AgentForecast(
                price_forecast=price_pred,
                volatility_forecast=vol_pred,
                confidence=confidence,
                timestamp=len(self.price_history)
            )

        except Exception as e:
            print(f"Agent {self.agent_id}: Forecast update failed - {e}")

    def _calculate_confidence(self) -> float:
        """Calculate confidence in forecasts based on recent performance."""
        # Simple heuristic: higher confidence if recent forecasts were accurate
        # In practice, track actual vs predicted and compute accuracy
        return np.random.uniform(0.6, 0.9)

    def get_forecast(self) -> Optional[AgentForecast]:
        """Get current forecast."""
        return self.current_forecast

    def calculate_pnl(self, current_prices: Dict[Tuple[float, float], float]) -> float:
        """
        Calculate current P&L.

        Parameters:
        -----------
        current_prices : Dict
            Current option prices {(K,T): price}

        Returns:
        --------
        float: Current P&L
        """
        position_value = sum(
            qty * current_prices.get(key, 0.0)
            for key, qty in self.positions.items()
        )

        total_value = self.cash + position_value
        pnl = total_value - self.initial_cash

        self.pnl_history.append(pnl)
        return pnl

    def make_decision(self, market_state: Dict) -> List[Order]:
        """
        Make trading decision based on forecasts.
        To be implemented by subclasses.

        Parameters:
        -----------
        market_state : Dict
            Current market state including order book, prices, etc.

        Returns:
        --------
        List[Order]: Orders to place
        """
        raise NotImplementedError("Subclasses must implement make_decision")


class TSMarketMaker(TimeSeriesDrivenAgent):
    """
    Market Maker driven by time series forecasts.

    Behavior:
    - Provides liquidity by quoting bid-ask spreads
    - Uses price forecast to skew quotes (expect rise -> higher quotes)
    - Uses volatility forecast to adjust spread width
    - Manages inventory risk
    """

    def __init__(self, agent_id: str, initial_cash: float = 2000000.0,
                 spread_multiplier: float = 1.0):
        super().__init__(agent_id, AgentRole.MARKET_MAKER, initial_cash)
        self.spread_multiplier = spread_multiplier
        self.inventory_limit = 100
        self.base_spread = 0.02

    def make_decision(self, market_state: Dict) -> List[Order]:
        """Generate bid-ask quotes based on forecasts."""
        orders = []

        if not self.is_model_ready or self.current_forecast is None:
            return orders

        # Get forecast
        forecast = self.current_forecast
        current_spot = self.price_history[-1]
        expected_spot = np.mean(forecast.price_forecast)
        expected_vol = np.mean(forecast.volatility_forecast)

        # Price direction signal
        price_signal = (expected_spot - current_spot) / current_spot

        # Generate quotes for each option
        for option_key in market_state.get('available_options', []):
            strike, expiry = option_key

            # Calculate theoretical value using forecast
            theo_value = self._calculate_theoretical_value(
                strike, expiry, expected_spot, expected_vol, market_state
            )

            # Adjust for inventory
            current_position = self.positions.get(option_key, 0.0)
            inventory_adjustment = -current_position * 0.01  # Skew away from large positions

            # Adjust for price forecast
            forecast_adjustment = price_signal * theo_value * 0.3 * forecast.confidence

            # Calculate spread based on volatility
            spread = self.base_spread * (1 + expected_vol * self.spread_multiplier)

            # Final bid and ask
            mid = theo_value + inventory_adjustment + forecast_adjustment
            bid_price = max(0.01, mid - spread/2)
            ask_price = mid + spread/2

            # Create orders
            if abs(current_position) < self.inventory_limit:
                # Willing to buy
                orders.append(Order(
                    agent_id=self.agent_id,
                    option_key=option_key,
                    side='buy',
                    quantity=10.0,
                    limit_price=bid_price,
                    timestamp=len(self.price_history)
                ))

                # Willing to sell
                orders.append(Order(
                    agent_id=self.agent_id,
                    option_key=option_key,
                    side='sell',
                    quantity=10.0,
                    limit_price=ask_price,
                    timestamp=len(self.price_history)
                ))

        return orders

    def _calculate_theoretical_value(self, strike: float, expiry: float,
                                    spot: float, vol: float,
                                    market_state: Dict) -> float:
        """Calculate theoretical option value using simple Black-Scholes."""
        from scipy.stats import norm

        r = market_state.get('risk_free_rate', 0.05)
        T = expiry
        S = spot
        K = strike
        sigma = vol

        if T <= 0:
            return max(S - K, 0)

        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)

        call_price = S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
        return max(call_price, 0.01)


class TSInformedTrader(TimeSeriesDrivenAgent):
    """
    Informed Trader driven by time series forecasts.

    Behavior:
    - Takes directional positions based on price forecasts
    - Buys calls if expecting price increase
    - Buys puts if expecting price decrease
    - Position sizing based on confidence
    """

    def __init__(self, agent_id: str, initial_cash: float = 1000000.0,
                 conviction_threshold: float = 0.02):
        super().__init__(agent_id, AgentRole.INFORMED_TRADER, initial_cash)
        self.conviction_threshold = conviction_threshold
        self.max_position_size = 50

    def make_decision(self, market_state: Dict) -> List[Order]:
        """Make directional trades based on forecasts."""
        orders = []

        if not self.is_model_ready or self.current_forecast is None:
            return orders

        forecast = self.current_forecast
        current_spot = self.price_history[-1]
        expected_spot = np.mean(forecast.price_forecast)

        # Expected return
        expected_return = (expected_spot - current_spot) / current_spot

        # Only trade if strong conviction
        if abs(expected_return) < self.conviction_threshold:
            return orders

        # Position size based on conviction and confidence
        position_size = min(
            self.max_position_size,
            abs(expected_return) * 100 * forecast.confidence
        )

        # Find suitable options
        best_option = self._find_best_option(market_state, expected_return > 0)

        if best_option:
            option_key, market_price = best_option

            if expected_return > 0:
                # Bullish - buy calls
                # Willing to pay up to theoretical value + premium
                limit_price = market_price * 1.05

                orders.append(Order(
                    agent_id=self.agent_id,
                    option_key=option_key,
                    side='buy',
                    quantity=position_size,
                    limit_price=limit_price,
                    timestamp=len(self.price_history)
                ))
            else:
                # Bearish - sell calls or buy puts (simplified: sell calls)
                limit_price = market_price * 0.95

                orders.append(Order(
                    agent_id=self.agent_id,
                    option_key=option_key,
                    side='sell',
                    quantity=position_size,
                    limit_price=limit_price,
                    timestamp=len(self.price_history)
                ))

        return orders

    def _find_best_option(self, market_state: Dict, is_bullish: bool
                         ) -> Optional[Tuple[Tuple[float, float], float]]:
        """Find the best option to trade given market view."""
        available_options = market_state.get('available_options', [])
        order_book = market_state.get('order_book', {})

        if not available_options:
            return None

        current_spot = self.price_history[-1]

        # For bullish: slightly OTM calls
        # For bearish: slightly ITM or ATM calls to sell
        best_option = None
        best_score = -float('inf')

        for option_key in available_options:
            strike, expiry = option_key
            moneyness = strike / current_spot

            if is_bullish:
                # Want OTM calls (moneyness 1.0-1.1)
                score = -abs(moneyness - 1.05)
            else:
                # Want ATM/ITM calls to sell (moneyness 0.95-1.0)
                score = -abs(moneyness - 0.975)

            if score > best_score and option_key in order_book:
                best_score = score
                # Get mid price from order book
                book = order_book[option_key]
                if book['bids'] and book['asks']:
                    mid_price = (book['bids'][0][0] + book['asks'][0][0]) / 2
                    best_option = (option_key, mid_price)

        return best_option


class TSArbitrageur(TimeSeriesDrivenAgent):
    """
    Arbitrageur driven by time series forecasts.

    Behavior:
    - Compares market prices with forecast-based theoretical values
    - Buys underpriced options
    - Sells overpriced options
    - Hedges with delta-neutral strategies
    """

    def __init__(self, agent_id: str, initial_cash: float = 1500000.0,
                 arbitrage_threshold: float = 0.03):
        super().__init__(agent_id, AgentRole.ARBITRAGEUR, initial_cash)
        self.arbitrage_threshold = arbitrage_threshold
        self.max_position = 30

    def make_decision(self, market_state: Dict) -> List[Order]:
        """Find and exploit mispricings."""
        orders = []

        if not self.is_model_ready or self.current_forecast is None:
            return orders

        forecast = self.current_forecast
        expected_spot = np.mean(forecast.price_forecast)
        expected_vol = np.mean(forecast.volatility_forecast)

        order_book = market_state.get('order_book', {})

        for option_key in market_state.get('available_options', []):
            strike, expiry = option_key

            # Calculate theoretical value
            theo_value = self._calculate_theoretical_value(
                strike, expiry, expected_spot, expected_vol, market_state
            )

            # Get market prices
            if option_key not in order_book:
                continue

            book = order_book[option_key]

            # Check for arbitrage on ask side (market selling too cheap)
            if book['asks']:
                best_ask = book['asks'][0][0]
                if theo_value > best_ask * (1 + self.arbitrage_threshold):
                    # Buy underpriced
                    orders.append(Order(
                        agent_id=self.agent_id,
                        option_key=option_key,
                        side='buy',
                        quantity=self.max_position,
                        limit_price=best_ask * 1.01,
                        timestamp=len(self.price_history)
                    ))

            # Check for arbitrage on bid side (market buying too expensive)
            if book['bids']:
                best_bid = book['bids'][0][0]
                if theo_value < best_bid * (1 - self.arbitrage_threshold):
                    # Sell overpriced
                    orders.append(Order(
                        agent_id=self.agent_id,
                        option_key=option_key,
                        side='sell',
                        quantity=self.max_position,
                        limit_price=best_bid * 0.99,
                        timestamp=len(self.price_history)
                    ))

        return orders

    def _calculate_theoretical_value(self, strike: float, expiry: float,
                                    spot: float, vol: float,
                                    market_state: Dict) -> float:
        """Calculate theoretical value (same as market maker)."""
        from scipy.stats import norm

        r = market_state.get('risk_free_rate', 0.05)
        T = expiry
        S = spot
        K = strike
        sigma = vol

        if T <= 0:
            return max(S - K, 0)

        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)

        call_price = S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
        return max(call_price, 0.01)
