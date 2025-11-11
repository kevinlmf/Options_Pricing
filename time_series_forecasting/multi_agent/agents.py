"""
Multi-Agent Components for Structural Forecasting

Each agent type encodes behavioral assumptions that generate observable patterns.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class AgentState:
    """State variables for an agent"""
    inventory: float = 0.0
    cash: float = 100000.0
    last_action: Optional[str] = None
    action_history: List[str] = None

    def __post_init__(self):
        if self.action_history is None:
            self.action_history = []


class BaseAgent(ABC):
    """Base class for all agent types"""

    def __init__(self, risk_aversion: float = 1.0, max_validation_history: int = 200):
        self.risk_aversion = risk_aversion
        self.state = AgentState()
        self.max_validation_history = max_validation_history
        self.validation_history: List[Dict[str, float]] = []

    @abstractmethod
    def decide_action(self, market_state: Dict) -> Dict:
        """
        Decide action based on current market state.

        Returns:
        --------
        action : Dict
            {'type': 'buy'/'sell'/'hold', 'quantity': float, 'price': float}
        """
        pass

    @abstractmethod
    def infer_parameter(self, action_history: List[Dict], **kwargs) -> float:
        """
        Infer market parameter from action history.

        This is the key structural insight: actions → parameters
        
        Parameters:
        -----------
        action_history : List[Dict]
            History of agent actions
        **kwargs : dict
            Additional parameters (e.g., historical_returns for drift calculation)
        """
        pass

    def record_validation(self, record: Dict[str, float]):
        """Store validation feedback for learning/analysis."""
        if len(self.validation_history) >= self.max_validation_history:
            self.validation_history.pop(0)
        self.validation_history.append(record)

    def update_from_validation(self, *args, **kwargs):
        """Optional hook for adaptive parameter updates."""
        return


class MarketMaker(BaseAgent):
    """
    Market Maker Agent

    Behavioral Assumption:
    - Provides liquidity by quoting bid-ask spreads
    - Wider spreads in high volatility environments

    Structural Inference:
    - spread_width → implied_volatility

    Model: σ_implied = f(spread_width, inventory_risk)
    """

    def __init__(self, risk_aversion: float = 1.0, target_spread: float = 0.01):
        super().__init__(risk_aversion)
        self.target_spread = target_spread
        self.base_vol = 0.15
        self.vol_coefficient_mean = 1.5
        self.vol_coefficient_var = 3.0
        self.max_vol_caps = (0.5, 1.0, 2.0)
        self.volatility_learning_rate = 0.1

    def decide_action(self, market_state: Dict) -> Dict:
        """
        Decide bid-ask spread based on volatility perception.

        Parameters:
        -----------
        market_state : Dict
            {'price': float, 'recent_returns': np.ndarray, 'volume': float}
        """
        current_price = market_state['price']
        recent_returns = market_state.get('recent_returns', np.array([]))

        if len(recent_returns) > 0:
            # Estimate volatility from recent returns
            realized_vol = np.std(recent_returns)

            # Adjust spread based on volatility and inventory risk
            # IMPROVED: Reduced coefficient from 10 to 2 to avoid extreme spreads
            # Spread should be proportional to volatility, but not too sensitive
            inventory_penalty = abs(self.state.inventory) * 0.001
            spread = self.target_spread + realized_vol * 2 + inventory_penalty
        else:
            spread = self.target_spread

        # Quote bid-ask
        bid = current_price * (1 - spread/2)
        ask = current_price * (1 + spread/2)

        action = {
            'type': 'quote',
            'bid': bid,
            'ask': ask,
            'spread': spread,
            'mid': current_price
        }

        self.state.action_history.append(action)
        return action

    def infer_parameter(self, action_history: List[Dict]) -> float:
        """
        Infer implied volatility from spread history.

        Structural Model:
        σ_implied = α × E[spread] + β × Var[spread]

        Intuition: Market makers widen spreads when they perceive risk.
        """
        if len(action_history) == 0:
            return 0.15  # Default volatility (15%, more reasonable)

        spreads = [action['spread'] for action in action_history if 'spread' in action]

        if len(spreads) < 2:
            return 0.15  # Default volatility (15%, more reasonable)

        mean_spread = np.mean(spreads)
        var_spread = np.var(spreads)

        # Structural mapping: spread dynamics → volatility
        # FINAL CALIBRATION: Coefficients tuned to match actual market volatility
        # With spread coefficient = 2 in decide_action, typical spread ≈ 0.02-0.03
        # Using coefficients (1.5, 3) to produce predictions in 15-25% range
        implied_vol = self.base_vol + mean_spread * self.vol_coefficient_mean + var_spread * self.vol_coefficient_var

        # Adaptive clipping: more reasonable upper bound
        # Normal market: 50%, Volatile: 100%, Extreme: 200%
        if mean_spread < 0.01:
            max_vol = self.max_vol_caps[0]  # 50% for normal markets
        elif mean_spread < 0.05:
            max_vol = self.max_vol_caps[1]  # 100% for volatile markets
        else:
            max_vol = self.max_vol_caps[2]  # 200% for extreme markets
        
        return np.clip(implied_vol, 0.05, max_vol)

    def update_from_validation(self,
                               predicted_vol: float,
                               realized_vol: float,
                               learning_rate: Optional[float] = None):
        """Adapt volatility mapping based on validation feedback."""
        if learning_rate is None:
            learning_rate = self.volatility_learning_rate

        if not np.isfinite(predicted_vol) or not np.isfinite(realized_vol) or realized_vol <= 0:
            return

        error = realized_vol - predicted_vol
        adjustment = learning_rate * error
        abs_error = abs(error)

        if abs_error < 0.01:
            adjustment *= 0.5
        elif abs_error > 0.05:
            adjustment *= min(2.0, 1.0 + abs_error)

        # Update structural coefficients conservatively
        self.base_vol = float(np.clip(self.base_vol + 0.5 * adjustment, 0.05, 1.5))
        self.vol_coefficient_mean = float(np.clip(self.vol_coefficient_mean + 2.0 * adjustment, 0.1, 10.0))
        self.vol_coefficient_var = float(np.clip(self.vol_coefficient_var + 5.0 * adjustment, 0.5, 25.0))

        self.record_validation({
            'predicted_vol': float(predicted_vol),
            'realized_vol': float(realized_vol),
            'error': float(error)
        })


class Arbitrageur(BaseAgent):
    """
    Arbitrageur Agent

    Behavioral Assumption:
    - Exploits price discrepancies
    - More active when drift/momentum is strong

    Structural Inference:
    - trading_intensity → implied_drift

    Model: μ_implied = f(trade_frequency, trade_direction, historical_drift, momentum)
    
    IMPROVED: Now considers multiple factors for better drift prediction
    """

    def __init__(self, risk_aversion: float = 0.5, threshold: float = 0.02):
        super().__init__(risk_aversion)
        self.threshold = threshold  # Minimum profit threshold
        self.trade_count = 0
        self.historical_prices = []  # Store prices for drift calculation
        self.historical_weight_primary = 0.8
        self.historical_weight_fallback = 0.6
        self.momentum_blend = 0.3
        self.learning_rate = 0.05

    def decide_action(self, market_state: Dict) -> Dict:
        """
        Decide whether to arbitrage based on momentum signals.

        Parameters:
        -----------
        market_state : Dict
            {'price': float, 'recent_returns': np.ndarray, 'trend': float}
        """
        current_price = market_state['price']
        recent_returns = market_state.get('recent_returns', np.array([]))

        if len(recent_returns) > 0:
            # Simple momentum: recent average return
            momentum = np.mean(recent_returns[-5:])  # Last 5 periods

            # Trade if momentum exceeds threshold
            if momentum > self.threshold:
                action_type = 'buy'
                quantity = min(10, abs(momentum) * 100)  # Scale with momentum
                self.trade_count += 1
            elif momentum < -self.threshold:
                action_type = 'sell'
                quantity = min(10, abs(momentum) * 100)
                self.trade_count += 1
            else:
                action_type = 'hold'
                quantity = 0
        else:
            action_type = 'hold'
            quantity = 0

        action = {
            'type': action_type,
            'quantity': quantity,
            'price': current_price,
            'momentum': momentum if len(recent_returns) > 0 else 0.0
        }

        self.state.action_history.append(action)
        return action

    def infer_parameter(self, action_history: List[Dict], historical_returns: Optional[np.ndarray] = None) -> float:
        """
        Infer implied drift from arbitrage activity.
        
        IMPROVED: Enhanced drift prediction with multiple factors:
        1. Trade frequency and direction (structural signal)
        2. Historical drift (empirical baseline) - PRIORITY if available
        3. Momentum strength (price momentum)
        
        Structural Model:
        μ_implied = α × (trade_freq × net_direction) + β × historical_drift + γ × momentum
        
        Parameters:
        -----------
        action_history : List[Dict]
            History of agent actions
        historical_returns : Optional[np.ndarray]
            Historical returns for direct drift calculation (more accurate)
        
        Intuition: 
        - Arbitrageurs are more active when they detect exploitable trends
        - Historical drift provides empirical baseline (most reliable)
        - Momentum captures recent price trends
        """
        if len(action_history) == 0:
            return 0.0

        # Count trades
        trades = [a for a in action_history if a['type'] in ['buy', 'sell']]
        trade_frequency = len(trades) / max(len(action_history), 1)

        # Net direction
        buys = sum(1 for a in trades if a['type'] == 'buy')
        sells = sum(1 for a in trades if a['type'] == 'sell')
        net_direction = (buys - sells) / max(len(trades), 1) if len(trades) > 0 else 0

        # Calculate momentum and historical drift
        momentum = 0.0
        historical_drift = 0.0
        
        # PRIORITY: Use historical_returns if provided (more accurate)
        if historical_returns is not None and len(historical_returns) > 0:
            # Use full historical data for drift calculation
            historical_drift = np.mean(historical_returns) * 252  # Annualized
            
            # Short-term momentum (last 20 periods)
            if len(historical_returns) >= 20:
                momentum = np.mean(historical_returns[-20:]) * 252
            else:
                momentum = historical_drift
        
        # FALLBACK: Calculate from action history if no historical_returns
        elif len(action_history) > 1:
            # Extract prices from action history
            prices = [a.get('price', 0) for a in action_history if 'price' in a and a.get('price', 0) > 0]
            
            if len(prices) > 1:
                # Calculate returns
                price_array = np.array(prices)
                returns = np.diff(np.log(price_array))
                
                if len(returns) > 0:
                    # Short-term momentum (last 5 periods, annualized)
                    if len(returns) >= 5:
                        momentum = np.mean(returns[-5:]) * 252
                    else:
                        momentum = np.mean(returns) * 252
                    
                    # Historical drift (full period, annualized)
                    historical_drift = np.mean(returns) * 252
                    
                    # Also use momentum from action if available
                    momentums = [a.get('momentum', 0) for a in action_history if 'momentum' in a]
                    if len(momentums) > 0:
                        avg_momentum = np.mean(momentums)
                        if abs(avg_momentum) > 1e-6:
                            # Convert daily momentum to annual
                            momentum_from_actions = avg_momentum * 252
                            # Blend with price-based momentum
                            momentum = 0.7 * momentum + 0.3 * momentum_from_actions
        
        # IMPROVED: Multi-factor drift prediction
        # Factor 1: Structural signal (increased coefficient from 0.5 to 2.0)
        structural_signal = trade_frequency * net_direction * 2.0
        
        # Factor 2: Historical drift (PRIORITY if available)
        # If we have historical drift from full data, use it as primary signal
        if abs(historical_drift) > 1e-6:
            # Historical drift is more reliable, weight it higher
            if historical_returns is not None:
                weight = self.historical_weight_primary
            else:
                weight = self.historical_weight_fallback
            implied_drift = weight * historical_drift + (1 - weight) * structural_signal
        else:
            # Only structural signal available
            implied_drift = structural_signal
        
        # Factor 3: Momentum adjustment (if strong momentum detected and consistent with historical)
        # Only adjust if momentum is strong AND in same direction as historical drift
        # This prevents negative momentum from overriding positive historical drift
        if abs(momentum) > 0.05:  # Strong momentum (> 5% annualized)
            # Check if momentum is consistent with historical drift direction
            if (historical_drift > 0 and momentum > 0) or (historical_drift < 0 and momentum < 0):
                # Momentum confirms historical drift, blend them
                blend = self.momentum_blend
                implied_drift = (1 - blend) * implied_drift + blend * momentum
            elif abs(momentum) > abs(historical_drift) * 2:
                # Momentum is very strong and opposite, might indicate regime change
                # But still trust historical drift more (70% historical, 30% momentum)
                blend = self.momentum_blend
                implied_drift = (1 - blend) * implied_drift + blend * momentum * 0.5  # Dampened momentum
            # Otherwise, ignore inconsistent momentum
        
        return np.clip(implied_drift, -0.5, 0.5)

    def update_from_validation(self,
                               predicted_drift: float,
                               realized_drift: float,
                               learning_rate: Optional[float] = None):
        """Adapt drift inference parameters based on validation feedback."""
        if learning_rate is None:
            learning_rate = self.learning_rate

        if not np.isfinite(predicted_drift) or not np.isfinite(realized_drift):
            return

        error = realized_drift - predicted_drift
        abs_error = abs(error)

        rate_adjust = learning_rate
        if abs_error < 0.01:
            rate_adjust *= 0.5
        elif abs_error > 0.05:
            rate_adjust *= min(2.0, 1.0 + abs_error * 2)

        # Lower threshold when realized drift exceeds predictions (encourage activity)
        self.threshold = float(np.clip(self.threshold - rate_adjust * error, 0.005, 0.15))

        # Adjust weights between historical and structural signals
        self.historical_weight_primary = float(np.clip(
            self.historical_weight_primary - rate_adjust * error * 0.1, 0.5, 0.95))
        self.historical_weight_fallback = float(np.clip(
            self.historical_weight_fallback - rate_adjust * error * 0.1, 0.4, 0.9))

        # Allow more responsiveness to momentum when persistent gaps remain
        self.momentum_blend = float(np.clip(
            self.momentum_blend + rate_adjust * error * 0.1, 0.1, 0.6))

        self.record_validation({
            'predicted_drift': float(predicted_drift),
            'realized_drift': float(realized_drift),
            'error': float(error)
        })


class NoiseTrader(BaseAgent):
    """
    Noise Trader Agent

    Behavioral Assumption:
    - Trades based on sentiment, herding, noise
    - Behavior clusters in certain regimes (high vol, crashes)

    Structural Inference:
    - order_clustering → regime_detection

    Model: regime = f(clustering_coefficient, order_imbalance)
    """

    def __init__(self, risk_aversion: float = 2.0, herding_factor: float = 0.3):
        super().__init__(risk_aversion)
        self.herding_factor = herding_factor
        self.sentiment = 0.0  # [-1, 1]

    def decide_action(self, market_state: Dict) -> Dict:
        """
        Trade based on sentiment and herding.

        Parameters:
        -----------
        market_state : Dict
            {'price': float, 'recent_returns': np.ndarray, 'volume': float}
        """
        current_price = market_state['price']
        recent_returns = market_state.get('recent_returns', np.array([]))

        if len(recent_returns) > 0:
            # Update sentiment based on recent returns (momentum chasing)
            recent_change = recent_returns[-1] if len(recent_returns) > 0 else 0
            self.sentiment = 0.7 * self.sentiment + 0.3 * np.sign(recent_change)

            # Add herding: amplify sentiment with noise
            noise = np.random.randn() * 0.2
            effective_sentiment = self.sentiment * (1 + self.herding_factor) + noise

            # Random trading with sentiment bias
            if effective_sentiment > 0.3:
                action_type = 'buy'
                quantity = np.random.uniform(1, 5)
            elif effective_sentiment < -0.3:
                action_type = 'sell'
                quantity = np.random.uniform(1, 5)
            else:
                action_type = 'hold'
                quantity = 0
        else:
            action_type = 'hold'
            quantity = 0

        action = {
            'type': action_type,
            'quantity': quantity,
            'price': current_price,
            'sentiment': self.sentiment
        }

        self.state.action_history.append(action)
        return action

    def infer_parameter(self, action_history: List[Dict]) -> str:
        """
        Infer market regime from noise trader clustering.

        Structural Model:
        regime = 'high_vol' if clustering_high else 'normal'

        Intuition: Noise traders cluster their activity during stress periods.
        """
        if len(action_history) < 10:
            return 'normal'

        # Measure order clustering (consecutive similar actions)
        recent_actions = action_history[-20:]
        action_types = [a['type'] for a in recent_actions]

        # Count consecutive runs
        max_run = 1
        current_run = 1
        for i in range(1, len(action_types)):
            if action_types[i] == action_types[i-1] and action_types[i] != 'hold':
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 1

        # High clustering → stress regime
        clustering_threshold = 5
        if max_run >= clustering_threshold:
            return 'high_vol'
        else:
            return 'normal'


def simulate_agent_interaction(agents: List[BaseAgent],
                               market_data: np.ndarray,
                               n_periods: int = 100) -> Dict[str, List]:
    """
    Simulate multi-agent interaction over market data.

    Parameters:
    -----------
    agents : List[BaseAgent]
        List of agent instances
    market_data : np.ndarray
        Historical price data
    n_periods : int
        Number of periods to simulate

    Returns:
    --------
    simulation_results : Dict
        {'actions': List[Dict], 'parameters': Dict}
    """
    all_actions = {type(agent).__name__: [] for agent in agents}

    # Compute returns
    returns = np.diff(np.log(market_data))

    for t in range(min(n_periods, len(returns))):
        # Construct market state
        market_state = {
            'price': market_data[t],
            'recent_returns': returns[max(0, t-20):t+1],
            'volume': 1000.0  # Placeholder
        }

        # Each agent decides
        for agent in agents:
            action = agent.decide_action(market_state)
            all_actions[type(agent).__name__].append(action)

    # Infer parameters from agent behaviors
    inferred_params = {}
    for agent in agents:
        agent_type = type(agent).__name__
        # Pass historical returns to Arbitrageur for better drift calculation
        if agent_type == 'Arbitrageur':
            param = agent.infer_parameter(agent.state.action_history, historical_returns=returns)
        else:
            param = agent.infer_parameter(agent.state.action_history)
        inferred_params[agent_type] = param

    return {
        'actions': all_actions,
        'parameters': inferred_params
    }
