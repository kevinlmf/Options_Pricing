"""
Noise Trader Agent

Implements noise traders that create demand/supply imbalances through
behavioral biases and non-fundamental trading patterns. These agents
contribute to pricing deviations by creating persistent market pressures
that rational arbitrageurs cannot fully offset.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
from enum import Enum

from .base_agent import BaseAgent, AgentType, MarketState


class NoiseTraderBehavior(Enum):
    """Types of noise trader behaviors"""
    MOMENTUM = "momentum"  # Follow price trends
    MEAN_REVERSION = "mean_reversion"  # Bet against trends
    OVERCONFIDENT = "overconfident"  # Trade too frequently with high conviction
    HERDING = "herding"  # Follow other traders
    RANDOM = "random"  # Purely random trading


@dataclass
class NoiseTraderParameters:
    """Parameters controlling noise trader behavior"""

    # Behavior type
    behavior_type: NoiseTraderBehavior = NoiseTraderBehavior.MOMENTUM

    # Trading frequency and size
    base_trade_frequency: float = 0.1  # Base probability of trading per period
    trade_size_mean: float = 10.0  # Average trade size
    trade_size_std: float = 5.0  # Standard deviation of trade size

    # Behavioral parameters
    overconfidence_factor: float = 1.5  # How overconfident (affects trade size)
    momentum_lookback: int = 5  # Periods to look back for momentum
    herding_sensitivity: float = 0.3  # Sensitivity to others' actions

    # Bias parameters
    volatility_bias: float = 0.2  # Tend to trade more in high volatility
    moneyness_bias: float = 0.1  # Preference for certain moneyness levels
    time_decay_bias: float = 0.15  # Preference based on time to expiry

    # Risk parameters (noise traders are typically less risk-aware)
    position_limit: float = 500.0  # Maximum position size
    loss_tolerance: float = 0.3  # Higher loss tolerance than rational agents


class NoiseTrader(BaseAgent):
    """
    Noise Trader Agent

    Represents irrational market participants who trade based on:
    - Behavioral biases (overconfidence, herding, etc.)
    - Non-fundamental signals (momentum, random patterns)
    - Emotional responses to market conditions

    These agents create persistent demand/supply imbalances that
    contribute to option pricing deviations from theoretical values.
    """

    def __init__(self, agent_id: str, parameters: NoiseTraderParameters, initial_cash: float = 500000.0):
        """
        Initialize noise trader.

        Parameters:
        -----------
        agent_id : str
            Unique identifier
        parameters : NoiseTraderParameters
            Behavior parameters
        initial_cash : float
            Initial capital (typically lower than institutional agents)
        """
        super().__init__(agent_id, AgentType.NOISE_TRADER, initial_cash)
        self.params = parameters

        # Noise trader specific state
        self.price_history: Dict[Tuple[float, float], List[float]] = {}  # Track price history
        self.sentiment: float = 0.0  # Current market sentiment (-1 to 1)
        self.confidence_level: float = 0.5  # Current confidence level (0 to 1)
        self.last_trades: List[Dict] = []  # Recent trading history

        # Behavioral state
        self.momentum_signal: Dict[Tuple[float, float], float] = {}
        self.herding_signal: float = 0.0
        self.recent_pnl: List[float] = []

    def observe_market(self, market_state: MarketState) -> None:
        """
        Observe market and update behavioral state.

        Parameters:
        -----------
        market_state : MarketState
            Current market conditions
        """
        self.state.last_action_time = market_state.timestamp

        # Update price history
        self._update_price_history(market_state)

        # Update behavioral signals
        self._update_momentum_signals(market_state)
        self._update_sentiment(market_state)
        self._update_confidence(market_state)
        self._update_herding_signal(market_state)

        # Update unrealized P&L
        self.state.unrealized_pnl = self.calculate_unrealized_pnl(market_state)
        self.recent_pnl.append(self.state.total_pnl)
        if len(self.recent_pnl) > 10:
            self.recent_pnl.pop(0)

    def make_decision(self, market_state: MarketState) -> Dict[str, Union[str, float, Dict]]:
        """
        Make trading decision based on noise trader behavior.

        Parameters:
        -----------
        market_state : MarketState
            Current market conditions

        Returns:
        --------
        Dict: Trading decision
        """
        # Determine if should trade this period
        if not self._should_trade(market_state):
            return {
                'action': 'hold',
                'reason': 'no_trading_signal'
            }

        # Check basic position limits
        if not self.check_risk_limits():
            return {
                'action': 'hold',
                'reason': 'risk_limit_exceeded'
            }

        # Select instrument and direction based on behavior
        trade_decision = self._generate_trade_decision(market_state)

        if trade_decision is None:
            return {
                'action': 'hold',
                'reason': 'no_suitable_instruments'
            }

        return trade_decision

    def _should_trade(self, market_state: MarketState) -> bool:
        """
        Determine if noise trader should trade this period.

        Parameters:
        -----------
        market_state : MarketState
            Current market state

        Returns:
        --------
        bool: Whether to trade
        """
        base_probability = self.params.base_trade_frequency

        # Adjust based on volatility (noise traders trade more in volatile markets)
        volatility_mult = 1 + self.params.volatility_bias * market_state.underlying_volatility

        # Adjust based on recent P&L (overconfidence after wins, more trading after losses)
        if self.recent_pnl:
            recent_change = self.recent_pnl[-1] - (self.recent_pnl[-2] if len(self.recent_pnl) > 1 else 0)
            if recent_change > 0:  # Recent gain -> overconfidence
                pnl_mult = 1 + self.params.overconfidence_factor * 0.1
            else:  # Recent loss -> revenge trading
                pnl_mult = 1 + abs(recent_change) / 10000  # Scale by loss magnitude
        else:
            pnl_mult = 1.0

        # Final trading probability
        trade_probability = base_probability * volatility_mult * pnl_mult

        return np.random.random() < min(trade_probability, 0.8)  # Cap at 80%

    def _generate_trade_decision(self, market_state: MarketState) -> Optional[Dict]:
        """
        Generate specific trade decision based on behavior type.

        Parameters:
        -----------
        market_state : MarketState
            Current market state

        Returns:
        --------
        Optional[Dict]: Trade decision or None
        """
        # Select target instrument
        target_option = self._select_target_option(market_state)
        if target_option is None:
            return None

        strike, expiry = target_option

        # Determine trade direction and size based on behavior
        direction, size = self._determine_trade_direction_and_size(strike, expiry, market_state)

        if size <= 0:
            return None

        # Get current market price
        market_price = market_state.option_prices.get((strike, expiry))
        if market_price is None:
            return None

        # Add noise trader price impact (they tend to trade at worse prices)
        if direction == 'buy':
            execution_price = market_price * (1 + np.random.uniform(0.001, 0.005))  # Buy high
        else:
            execution_price = market_price * (1 - np.random.uniform(0.001, 0.005))  # Sell low

        return {
            'action': direction,
            'instrument': f"CALL_{strike}_{expiry}",
            'quantity': size,
            'price': execution_price,
            'trader_type': 'noise',
            'behavior_type': self.params.behavior_type.value,
            'confidence': self.confidence_level
        }

    def _select_target_option(self, market_state: MarketState) -> Optional[Tuple[float, float]]:
        """
        Select option to trade based on noise trader preferences.

        Parameters:
        -----------
        market_state : MarketState
            Current market state

        Returns:
        --------
        Optional[Tuple[float, float]]: (strike, expiry) or None
        """
        available_options = list(market_state.theoretical_prices.keys())
        if not available_options:
            return None

        # Apply behavioral biases to selection
        option_scores = {}

        for strike, expiry in available_options:
            score = 1.0

            # Moneyness bias (prefer certain strikes relative to spot)
            moneyness = np.log(market_state.underlying_price / strike)
            if self.params.moneyness_bias > 0:
                # Prefer slightly OTM options (classic retail bias)
                optimal_moneyness = 0.05  # Slightly OTM
                score *= np.exp(-abs(moneyness - optimal_moneyness) / 0.1)

            # Time bias (prefer shorter-dated options for excitement)
            if self.params.time_decay_bias > 0:
                score *= np.exp(-expiry * 2)  # Prefer shorter expiry

            # Volatility bias (more attracted to high IV options)
            iv = market_state.implied_volatilities.get((strike, expiry), 0.2)
            score *= (1 + self.params.volatility_bias * iv)

            option_scores[(strike, expiry)] = score

        # Select based on weighted probabilities
        options, scores = zip(*option_scores.items())
        probabilities = np.array(scores)
        probabilities /= probabilities.sum()

        selected_idx = np.random.choice(len(options), p=probabilities)
        return options[selected_idx]

    def _determine_trade_direction_and_size(self, strike: float, expiry: float,
                                          market_state: MarketState) -> Tuple[str, float]:
        """
        Determine trade direction and size based on behavior type.

        Parameters:
        -----------
        strike : float
            Option strike
        expiry : float
            Option expiry
        market_state : MarketState
            Current market state

        Returns:
        --------
        Tuple[str, float]: (direction, size)
        """
        if self.params.behavior_type == NoiseTraderBehavior.MOMENTUM:
            direction, size = self._momentum_trade(strike, expiry, market_state)

        elif self.params.behavior_type == NoiseTraderBehavior.MEAN_REVERSION:
            direction, size = self._mean_reversion_trade(strike, expiry, market_state)

        elif self.params.behavior_type == NoiseTraderBehavior.OVERCONFIDENT:
            direction, size = self._overconfident_trade(strike, expiry, market_state)

        elif self.params.behavior_type == NoiseTraderBehavior.HERDING:
            direction, size = self._herding_trade(strike, expiry, market_state)

        else:  # RANDOM
            direction, size = self._random_trade(strike, expiry, market_state)

        # Apply size constraints
        max_size = min(self.params.position_limit,
                      self.state.cash_position / market_state.option_prices.get((strike, expiry), 1.0))
        size = min(size, max_size)

        return direction, max(0, size)

    def _momentum_trade(self, strike: float, expiry: float, market_state: MarketState) -> Tuple[str, float]:
        """Momentum-based trading logic."""
        momentum = self.momentum_signal.get((strike, expiry), 0.0)

        if momentum > 0.02:  # Positive momentum -> buy
            direction = 'buy'
            size = self.params.trade_size_mean * (1 + momentum)
        elif momentum < -0.02:  # Negative momentum -> sell
            direction = 'sell'
            size = self.params.trade_size_mean * (1 + abs(momentum))
        else:
            direction = 'buy' if np.random.random() > 0.5 else 'sell'
            size = self.params.trade_size_mean * 0.5

        return direction, size

    def _mean_reversion_trade(self, strike: float, expiry: float, market_state: MarketState) -> Tuple[str, float]:
        """Mean reversion trading logic."""
        momentum = self.momentum_signal.get((strike, expiry), 0.0)

        # Trade against momentum
        if momentum > 0.02:  # Recent rise -> expect fall -> sell
            direction = 'sell'
            size = self.params.trade_size_mean * (1 + momentum)
        elif momentum < -0.02:  # Recent fall -> expect rise -> buy
            direction = 'buy'
            size = self.params.trade_size_mean * (1 + abs(momentum))
        else:
            direction = 'buy' if np.random.random() > 0.5 else 'sell'
            size = self.params.trade_size_mean * 0.5

        return direction, size

    def _overconfident_trade(self, strike: float, expiry: float, market_state: MarketState) -> Tuple[str, float]:
        """Overconfident trading logic."""
        direction = 'buy' if np.random.random() > 0.5 else 'sell'

        # Overconfident traders trade larger sizes
        base_size = self.params.trade_size_mean * self.params.overconfidence_factor

        # Size increases with confidence
        size = base_size * (1 + self.confidence_level)

        return direction, size

    def _herding_trade(self, strike: float, expiry: float, market_state: MarketState) -> Tuple[str, float]:
        """Herding-based trading logic."""
        # Follow the herd
        if self.herding_signal > 0.1:
            direction = 'buy'
        elif self.herding_signal < -0.1:
            direction = 'sell'
        else:
            direction = 'buy' if np.random.random() > 0.5 else 'sell'

        # Size based on herd strength
        size = self.params.trade_size_mean * (1 + abs(self.herding_signal))

        return direction, size

    def _random_trade(self, strike: float, expiry: float, market_state: MarketState) -> Tuple[str, float]:
        """Random trading logic."""
        direction = 'buy' if np.random.random() > 0.5 else 'sell'
        size = max(1, np.random.normal(self.params.trade_size_mean, self.params.trade_size_std))

        return direction, size

    def _update_price_history(self, market_state: MarketState) -> None:
        """Update price history for momentum calculations."""
        for (strike, expiry), price in market_state.option_prices.items():
            if (strike, expiry) not in self.price_history:
                self.price_history[(strike, expiry)] = []

            self.price_history[(strike, expiry)].append(price)

            # Keep only recent history
            max_history = self.params.momentum_lookback * 2
            if len(self.price_history[(strike, expiry)]) > max_history:
                self.price_history[(strike, expiry)].pop(0)

    def _update_momentum_signals(self, market_state: MarketState) -> None:
        """Update momentum signals for each option."""
        for (strike, expiry), prices in self.price_history.items():
            if len(prices) >= self.params.momentum_lookback:
                # Calculate momentum as recent price change
                recent_prices = prices[-self.params.momentum_lookback:]
                momentum = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
                self.momentum_signal[(strike, expiry)] = momentum

    def _update_sentiment(self, market_state: MarketState) -> None:
        """Update overall market sentiment."""
        # Simple sentiment based on underlying price movement and volatility
        current_sentiment = 0.0

        # Positive sentiment if underlying is rising
        if hasattr(self, '_last_underlying_price'):
            price_change = (market_state.underlying_price - self._last_underlying_price) / self._last_underlying_price
            current_sentiment += price_change * 2  # Amplify price movements

        # Negative sentiment if volatility is high (fear)
        vol_impact = -(market_state.underlying_volatility - 0.2) * 2
        current_sentiment += vol_impact

        # Update sentiment with momentum
        self.sentiment = 0.7 * self.sentiment + 0.3 * np.clip(current_sentiment, -1, 1)
        self._last_underlying_price = market_state.underlying_price

    def _update_confidence(self, market_state: MarketState) -> None:
        """Update confidence level based on recent performance."""
        if len(self.recent_pnl) >= 2:
            recent_performance = self.recent_pnl[-1] - self.recent_pnl[-2]
            if recent_performance > 0:
                # Increase confidence after gains
                self.confidence_level = min(1.0, self.confidence_level + 0.1)
            else:
                # Decrease confidence after losses
                self.confidence_level = max(0.1, self.confidence_level - 0.05)
        else:
            self.confidence_level = 0.5  # Neutral confidence

    def _update_herding_signal(self, market_state: MarketState) -> None:
        """Update herding signal based on market order flow."""
        # Simplified herding signal based on aggregate order flow
        total_flow = sum(market_state.order_flow.values())
        self.herding_signal = 0.8 * self.herding_signal + 0.2 * np.clip(total_flow / 1000, -1, 1)

    def get_noise_trader_stats(self) -> Dict:
        """
        Get noise trader specific performance statistics.

        Returns:
        --------
        Dict: Noise trader performance metrics
        """
        base_stats = self.get_performance_summary()

        noise_stats = {
            'behavior_type': self.params.behavior_type.value,
            'current_sentiment': self.sentiment,
            'current_confidence': self.confidence_level,
            'herding_signal': self.herding_signal,
            'trades_made': len(self.last_trades),
            'avg_trade_size': np.mean([trade.get('quantity', 0) for trade in self.last_trades]) if self.last_trades else 0,
            'momentum_signals_count': len(self.momentum_signal)
        }

        return {**base_stats, **noise_stats}