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

    def __init__(self, risk_aversion: float = 1.0):
        self.risk_aversion = risk_aversion
        self.state = AgentState()

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
    def infer_parameter(self, action_history: List[Dict]) -> float:
        """
        Infer market parameter from action history.

        This is the key structural insight: actions → parameters
        """
        pass


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
            inventory_penalty = abs(self.state.inventory) * 0.001
            spread = self.target_spread + realized_vol * 10 + inventory_penalty
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
            return 0.5  # Default volatility

        spreads = [action['spread'] for action in action_history if 'spread' in action]

        if len(spreads) < 2:
            return 0.5

        mean_spread = np.mean(spreads)
        var_spread = np.var(spreads)

        # Structural mapping: spread dynamics → volatility
        # Calibrated heuristic (in practice, estimated from historical data)
        implied_vol = 0.2 + mean_spread * 20 + var_spread * 50

        return np.clip(implied_vol, 0.1, 2.0)


class Arbitrageur(BaseAgent):
    """
    Arbitrageur Agent

    Behavioral Assumption:
    - Exploits price discrepancies
    - More active when drift/momentum is strong

    Structural Inference:
    - trading_intensity → implied_drift

    Model: μ_implied = f(trade_frequency, trade_direction)
    """

    def __init__(self, risk_aversion: float = 0.5, threshold: float = 0.02):
        super().__init__(risk_aversion)
        self.threshold = threshold  # Minimum profit threshold
        self.trade_count = 0

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

    def infer_parameter(self, action_history: List[Dict]) -> float:
        """
        Infer implied drift from arbitrage activity.

        Structural Model:
        μ_implied = α × (trade_freq) × sign(net_direction)

        Intuition: Arbitrageurs are more active when they detect exploitable trends.
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

        # Structural mapping: activity → drift
        # High frequency + directional bias → strong drift signal
        implied_drift = trade_frequency * net_direction * 0.5

        return np.clip(implied_drift, -0.5, 0.5)


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
        param = agent.infer_parameter(agent.state.action_history)
        inferred_params[agent_type] = param

    return {
        'actions': all_actions,
        'parameters': inferred_params
    }
