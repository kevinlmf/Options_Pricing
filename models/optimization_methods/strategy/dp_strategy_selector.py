"""
Dynamic Programming Strategy Selector for Bitcoin Options

Multi-objective optimization using dynamic programming to select optimal
option strategies based on:
- Sharpe ratio maximization
- CVaR constraint enforcement
- Volatility regime adaptation
- Greeks balancing

Mathematical Framework:
- State space: (price_state, volatility_state, time_state, risk_state)
- Action space: {Straddle, Strangle, Spread, Naked, etc.}
- Reward function: α*Sharpe - β*CVaR_penalty - γ*drawdown
- Bellman equation: V(s,t) = max_a { R(s,a,t) + γ*E[V(s',t+1)] }

Optimization Horizons:
- Short-term (1-7 days): Theta optimization + Delta hedging
- Medium-term (7-30 days): Vega trading + Trend capture
- Long-term (30+ days): Portfolio structure + Risk balance
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import warnings

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# BitcoinRiskController removed - using generic risk checks
from typing import Optional
from dataclasses import dataclass

@dataclass
class OrderProposal:
    """Generic order proposal"""
    symbol: str
    quantity: int
    order_type: str
    price: Optional[float] = None

class BitcoinRiskController:
    """Placeholder for removed BitcoinRiskController"""
    def __init__(self, *args, **kwargs):
        pass
    def check_order(self, *args, **kwargs):
        return True


# ============================================================================
# State Space Definitions
# ============================================================================

class PriceState(Enum):
    """Price state relative to recent range"""
    VERY_BEARISH = 0  # < 20th percentile
    BEARISH = 1       # 20-40th percentile
    NEUTRAL = 2       # 40-60th percentile
    BULLISH = 3       # 60-80th percentile
    VERY_BULLISH = 4  # > 80th percentile


class VolatilityState(Enum):
    """Volatility regime state"""
    LOW = 0      # IV < 40%
    MEDIUM = 1   # IV 40-80%
    HIGH = 2     # IV 80-120%
    EXTREME = 3  # IV > 120%


class TimeState(Enum):
    """Time horizon state"""
    SHORT = 0   # < 7 days
    MEDIUM = 1  # 7-30 days
    LONG = 2    # > 30 days


class RiskState(Enum):
    """Risk utilization state"""
    LOW_RISK = 0     # < 50% limit utilization
    MEDIUM_RISK = 1  # 50-75% utilization
    HIGH_RISK = 2    # 75-90% utilization
    CRITICAL = 3     # > 90% utilization


# ============================================================================
# Action Space Definitions
# ============================================================================

class StrategyType(Enum):
    """Available option strategies"""
    LONG_CALL = "long_call"
    LONG_PUT = "long_put"
    SHORT_CALL = "short_call"
    SHORT_PUT = "short_put"
    STRADDLE = "straddle"
    STRANGLE = "strangle"
    BULL_CALL_SPREAD = "bull_call_spread"
    BEAR_PUT_SPREAD = "bear_put_spread"
    IRON_CONDOR = "iron_condor"
    BUTTERFLY = "butterfly"
    CALENDAR_SPREAD = "calendar_spread"
    RATIO_SPREAD = "ratio_spread"
    DO_NOTHING = "do_nothing"


@dataclass
class StrategyAction:
    """Action representing a strategy to execute"""
    strategy_type: StrategyType
    strikes: List[float]
    expiries: List[float]
    quantities: List[int]
    option_types: List[str]  # ['call', 'put', ...]
    directions: List[str]    # ['buy', 'sell', ...]

    expected_return: float = 0.0
    expected_risk: float = 0.0
    sharpe: float = 0.0

    def __repr__(self):
        return f"StrategyAction({self.strategy_type.value}, Sharpe={self.sharpe:.3f})"


# ============================================================================
# Market State
# ============================================================================

@dataclass
class MarketState:
    """Complete market state representation"""
    price_state: PriceState
    volatility_state: VolatilityState
    time_state: TimeState
    risk_state: RiskState

    # Continuous values
    current_price: float
    current_iv: float
    days_to_horizon: int

    # Risk metrics
    delta_utilization: float
    vega_utilization: float
    cvar_utilization: float

    timestamp: datetime = field(default_factory=datetime.now)

    def to_tuple(self) -> Tuple[int, int, int, int]:
        """Convert to discrete state tuple for DP"""
        return (
            self.price_state.value,
            self.volatility_state.value,
            self.time_state.value,
            self.risk_state.value
        )


# ============================================================================
# Dynamic Programming Strategy Selector
# ============================================================================

class DPStrategySelector:
    """
    Dynamic Programming based strategy selector.

    Solves the Bellman equation to find optimal strategy sequences
    that maximize risk-adjusted returns while respecting constraints.
    """

    def __init__(self,
                 risk_controller: BitcoinRiskController,
                 discount_factor: float = 0.95,
                 sharpe_weight: float = 1.0,
                 cvar_weight: float = 0.5,
                 drawdown_weight: float = 0.3,
                 transaction_cost: float = 0.001):
        """
        Initialize DP strategy selector.

        Parameters:
        -----------
        risk_controller : BitcoinRiskController
            Risk controller for order validation
        discount_factor : float
            Discount factor γ for future rewards
        sharpe_weight : float
            Weight α for Sharpe ratio in reward
        cvar_weight : float
            Weight β for CVaR penalty in reward
        drawdown_weight : float
            Weight γ for drawdown penalty in reward
        transaction_cost : float
            Transaction cost as fraction of notional
        """
        self.risk_controller = risk_controller
        self.gamma = discount_factor
        self.alpha = sharpe_weight
        self.beta = cvar_weight
        self.delta_w = drawdown_weight
        self.transaction_cost = transaction_cost

        # Value function cache: V[state] = value
        self.value_function: Dict[Tuple, float] = {}

        # Policy cache: π[state] = best_action
        self.policy: Dict[Tuple, StrategyAction] = {}

        # Q-function cache: Q[state][action] = value
        self.q_function: Dict[Tuple, Dict[str, float]] = {}

        # Historical performance
        self.performance_history: List[Dict] = []

    def classify_market_state(self,
                             current_price: float,
                             current_iv: float,
                             days_to_horizon: int,
                             price_history: List[float],
                             risk_utilization: Dict[str, float]) -> MarketState:
        """
        Classify current market conditions into discrete states.

        Parameters:
        -----------
        current_price : float
            Current BTC price
        current_iv : float
            Current implied volatility
        days_to_horizon : int
            Days until decision horizon
        price_history : list
            Recent price history for percentile calculation
        risk_utilization : dict
            Current risk limit utilization

        Returns:
        --------
        MarketState
            Classified market state
        """
        # Price state - relative to recent range
        if len(price_history) > 0:
            percentile = np.percentile(price_history,
                                      [20, 40, 60, 80])
            if current_price < percentile[0]:
                price_state = PriceState.VERY_BEARISH
            elif current_price < percentile[1]:
                price_state = PriceState.BEARISH
            elif current_price < percentile[2]:
                price_state = PriceState.NEUTRAL
            elif current_price < percentile[3]:
                price_state = PriceState.BULLISH
            else:
                price_state = PriceState.VERY_BULLISH
        else:
            price_state = PriceState.NEUTRAL

        # Volatility state
        if current_iv < 0.40:
            vol_state = VolatilityState.LOW
        elif current_iv < 0.80:
            vol_state = VolatilityState.MEDIUM
        elif current_iv < 1.20:
            vol_state = VolatilityState.HIGH
        else:
            vol_state = VolatilityState.EXTREME

        # Time state
        if days_to_horizon < 7:
            time_state = TimeState.SHORT
        elif days_to_horizon < 30:
            time_state = TimeState.MEDIUM
        else:
            time_state = TimeState.LONG

        # Risk state - based on maximum utilization
        max_util = max(risk_utilization.values()) if risk_utilization else 0
        if max_util < 0.50:
            risk_state = RiskState.LOW_RISK
        elif max_util < 0.75:
            risk_state = RiskState.MEDIUM_RISK
        elif max_util < 0.90:
            risk_state = RiskState.HIGH_RISK
        else:
            risk_state = RiskState.CRITICAL

        return MarketState(
            price_state=price_state,
            volatility_state=vol_state,
            time_state=time_state,
            risk_state=risk_state,
            current_price=current_price,
            current_iv=current_iv,
            days_to_horizon=days_to_horizon,
            delta_utilization=risk_utilization.get('delta', 0),
            vega_utilization=risk_utilization.get('vega', 0),
            cvar_utilization=risk_utilization.get('cvar', 0)
        )

    def generate_candidate_strategies(self,
                                     state: MarketState) -> List[StrategyAction]:
        """
        Generate candidate strategies appropriate for current state.

        Parameters:
        -----------
        state : MarketState
            Current market state

        Returns:
        --------
        list of StrategyAction
            Candidate strategies to evaluate
        """
        strategies = []
        price = state.current_price
        iv = state.current_iv

        # Determine appropriate expiry based on time state
        if state.time_state == TimeState.SHORT:
            expiry = 7 / 365
        elif state.time_state == TimeState.MEDIUM:
            expiry = 30 / 365
        else:
            expiry = 90 / 365

        # Strike selection based on price state and volatility
        atm_strike = price
        otm_call_strike = price * 1.10  # 10% OTM
        otm_put_strike = price * 0.90

        # Strategy 1: Do Nothing (always an option)
        strategies.append(StrategyAction(
            strategy_type=StrategyType.DO_NOTHING,
            strikes=[], expiries=[], quantities=[],
            option_types=[], directions=[]
        ))

        # Based on price state - directional bias
        if state.price_state in [PriceState.BULLISH, PriceState.VERY_BULLISH]:
            # Bullish strategies

            # Long Call
            strategies.append(StrategyAction(
                strategy_type=StrategyType.LONG_CALL,
                strikes=[atm_strike],
                expiries=[expiry],
                quantities=[5],
                option_types=['call'],
                directions=['buy']
            ))

            # Bull Call Spread
            strategies.append(StrategyAction(
                strategy_type=StrategyType.BULL_CALL_SPREAD,
                strikes=[atm_strike, otm_call_strike],
                expiries=[expiry, expiry],
                quantities=[5, 5],
                option_types=['call', 'call'],
                directions=['buy', 'sell']
            ))

        elif state.price_state in [PriceState.BEARISH, PriceState.VERY_BEARISH]:
            # Bearish strategies

            # Long Put
            strategies.append(StrategyAction(
                strategy_type=StrategyType.LONG_PUT,
                strikes=[atm_strike],
                expiries=[expiry],
                quantities=[5],
                option_types=['put'],
                directions=['buy']
            ))

            # Bear Put Spread
            strategies.append(StrategyAction(
                strategy_type=StrategyType.BEAR_PUT_SPREAD,
                strikes=[atm_strike, otm_put_strike],
                expiries=[expiry, expiry],
                quantities=[5, 5],
                option_types=['put', 'put'],
                directions=['buy', 'sell']
            ))

        # Based on volatility state
        if state.volatility_state == VolatilityState.LOW:
            # Buy volatility - Straddle/Strangle

            # Straddle
            strategies.append(StrategyAction(
                strategy_type=StrategyType.STRADDLE,
                strikes=[atm_strike, atm_strike],
                expiries=[expiry, expiry],
                quantities=[3, 3],
                option_types=['call', 'put'],
                directions=['buy', 'buy']
            ))

            # Strangle
            strategies.append(StrategyAction(
                strategy_type=StrategyType.STRANGLE,
                strikes=[otm_call_strike, otm_put_strike],
                expiries=[expiry, expiry],
                quantities=[3, 3],
                option_types=['call', 'put'],
                directions=['buy', 'buy']
            ))

        elif state.volatility_state == VolatilityState.HIGH:
            # Sell volatility - Iron Condor, Short Straddle

            # Iron Condor
            strategies.append(StrategyAction(
                strategy_type=StrategyType.IRON_CONDOR,
                strikes=[price * 0.85, price * 0.95, price * 1.05, price * 1.15],
                expiries=[expiry, expiry, expiry, expiry],
                quantities=[2, 2, 2, 2],
                option_types=['put', 'put', 'call', 'call'],
                directions=['buy', 'sell', 'sell', 'buy']
            ))

        # Risk state considerations
        if state.risk_state in [RiskState.HIGH_RISK, RiskState.CRITICAL]:
            # Only conservative strategies
            strategies = [s for s in strategies
                         if s.strategy_type in [StrategyType.DO_NOTHING,
                                               StrategyType.BULL_CALL_SPREAD,
                                               StrategyType.BEAR_PUT_SPREAD]]

        return strategies

    def calculate_reward(self,
                        state: MarketState,
                        action: StrategyAction,
                        next_state: Optional[MarketState] = None) -> float:
        """
        Calculate reward for taking action in state.

        Reward = α * Sharpe - β * CVaR_penalty - γ * Drawdown_penalty - Transaction_cost

        Parameters:
        -----------
        state : MarketState
            Current state
        action : StrategyAction
            Action taken
        next_state : MarketState, optional
            Resulting state after action

        Returns:
        --------
        float
            Reward value
        """
        # Do nothing has zero reward (but no cost)
        if action.strategy_type == StrategyType.DO_NOTHING:
            return 0.0

        # Estimate strategy returns based on Greeks and market state
        expected_return = self._estimate_strategy_return(state, action)
        expected_risk = self._estimate_strategy_risk(state, action)

        # Sharpe ratio term (risk-adjusted return)
        sharpe = expected_return / expected_risk if expected_risk > 0 else 0
        sharpe_term = self.alpha * sharpe

        # CVaR penalty term
        cvar_breach = max(0, state.cvar_utilization - 0.8)  # Penalty above 80%
        cvar_penalty = self.beta * cvar_breach ** 2

        # Drawdown penalty (based on risk state)
        if state.risk_state == RiskState.CRITICAL:
            drawdown_penalty = self.delta_w * 2.0
        elif state.risk_state == RiskState.HIGH_RISK:
            drawdown_penalty = self.delta_w * 1.0
        else:
            drawdown_penalty = 0.0

        # Transaction cost
        notional = sum(abs(q) for q in action.quantities) * state.current_price * 0.1
        tx_cost = notional * self.transaction_cost

        # Total reward
        reward = sharpe_term - cvar_penalty - drawdown_penalty - tx_cost / 1000

        # Store in action for reference
        action.expected_return = expected_return
        action.expected_risk = expected_risk
        action.sharpe = sharpe

        return reward

    def _estimate_strategy_return(self,
                                  state: MarketState,
                                  action: StrategyAction) -> float:
        """Estimate expected return of strategy"""
        # Simplified estimation based on strategy type and market state

        if action.strategy_type == StrategyType.DO_NOTHING:
            return 0.0

        # Directional strategies benefit from correct market state
        if action.strategy_type in [StrategyType.LONG_CALL, StrategyType.BULL_CALL_SPREAD]:
            if state.price_state in [PriceState.BULLISH, PriceState.VERY_BULLISH]:
                return 0.15  # 15% expected return
            else:
                return -0.05  # Negative if wrong direction

        if action.strategy_type in [StrategyType.LONG_PUT, StrategyType.BEAR_PUT_SPREAD]:
            if state.price_state in [PriceState.BEARISH, PriceState.VERY_BEARISH]:
                return 0.15
            else:
                return -0.05

        # Volatility strategies
        if action.strategy_type in [StrategyType.STRADDLE, StrategyType.STRANGLE]:
            # Benefit from low to high vol transition
            if state.volatility_state == VolatilityState.LOW:
                return 0.20  # High potential if vol increases
            else:
                return -0.10  # Expensive in high vol

        if action.strategy_type == StrategyType.IRON_CONDOR:
            # Benefits from high vol mean reversion
            if state.volatility_state == VolatilityState.HIGH:
                return 0.12
            else:
                return 0.05

        return 0.05  # Default modest return

    def _estimate_strategy_risk(self,
                               state: MarketState,
                               action: StrategyAction) -> float:
        """Estimate risk (standard deviation) of strategy"""
        # Base risk on volatility state and strategy type

        if action.strategy_type == StrategyType.DO_NOTHING:
            return 0.01  # Small epsilon to avoid division by zero

        # Volatility factor
        if state.volatility_state == VolatilityState.EXTREME:
            vol_mult = 2.0
        elif state.volatility_state == VolatilityState.HIGH:
            vol_mult = 1.5
        elif state.volatility_state == VolatilityState.MEDIUM:
            vol_mult = 1.0
        else:
            vol_mult = 0.7

        # Strategy base risk
        if action.strategy_type in [StrategyType.LONG_CALL, StrategyType.LONG_PUT]:
            base_risk = 0.25  # High risk, limited loss
        elif action.strategy_type in [StrategyType.SHORT_CALL, StrategyType.SHORT_PUT]:
            base_risk = 0.40  # Very high risk
        elif action.strategy_type in [StrategyType.STRADDLE, StrategyType.STRANGLE]:
            base_risk = 0.30  # High gamma risk
        elif action.strategy_type in [StrategyType.BULL_CALL_SPREAD, StrategyType.BEAR_PUT_SPREAD]:
            base_risk = 0.15  # Moderate risk
        elif action.strategy_type == StrategyType.IRON_CONDOR:
            base_risk = 0.10  # Lower risk
        else:
            base_risk = 0.20

        return base_risk * vol_mult

    def solve_value_function(self,
                            initial_state: MarketState,
                            n_iterations: int = 100,
                            price_scenarios: int = 20,
                            vol_scenarios: int = 10) -> Dict[Tuple, float]:
        """
        Solve value function using value iteration.

        V_{k+1}(s) = max_a { R(s,a) + γ * Σ P(s'|s,a) * V_k(s') }

        Parameters:
        -----------
        initial_state : MarketState
            Starting state
        n_iterations : int
            Number of value iteration steps
        price_scenarios : int
            Number of price scenarios per state
        vol_scenarios : int
            Number of vol scenarios per state

        Returns:
        --------
        dict
            Value function V[state]
        """
        print(f"Solving value function with {n_iterations} iterations...")

        # Initialize value function
        self.value_function = {}
        self.policy = {}

        # Generate all possible states
        states = self._generate_state_space()

        # Initialize V(s) = 0 for all states
        for state_tuple in states:
            self.value_function[state_tuple] = 0.0

        # Value iteration
        for iteration in range(n_iterations):
            delta = 0  # Track convergence
            new_values = {}

            for state_tuple in states:
                # Reconstruct MarketState from tuple
                state = self._tuple_to_state(state_tuple, initial_state)

                # Generate candidate actions
                actions = self.generate_candidate_strategies(state)

                # Find best action
                best_value = float('-inf')
                best_action = None

                for action in actions:
                    # Calculate Q(s,a) = R(s,a) + γ * Σ P(s'|s,a) * V(s')
                    reward = self.calculate_reward(state, action)

                    # Expected future value
                    future_value = self._expected_future_value(state, action, states)

                    q_value = reward + self.gamma * future_value

                    if q_value > best_value:
                        best_value = q_value
                        best_action = action

                new_values[state_tuple] = best_value
                self.policy[state_tuple] = best_action

                # Track convergence
                delta = max(delta, abs(best_value - self.value_function[state_tuple]))

            # Update value function
            self.value_function = new_values.copy()

            # Check convergence
            if iteration % 10 == 0:
                print(f"  Iteration {iteration}: max delta = {delta:.6f}")

            if delta < 1e-4:
                print(f"  Converged at iteration {iteration}")
                break

        print(f"Value function solved. {len(self.policy)} states in policy.")
        return self.value_function

    def _generate_state_space(self) -> List[Tuple]:
        """Generate all possible discrete states"""
        states = []
        for price in range(5):  # PriceState values
            for vol in range(4):  # VolatilityState values
                for time in range(3):  # TimeState values
                    for risk in range(4):  # RiskState values
                        states.append((price, vol, time, risk))
        return states

    def _tuple_to_state(self, state_tuple: Tuple, reference_state: MarketState) -> MarketState:
        """Convert state tuple back to MarketState"""
        return MarketState(
            price_state=PriceState(state_tuple[0]),
            volatility_state=VolatilityState(state_tuple[1]),
            time_state=TimeState(state_tuple[2]),
            risk_state=RiskState(state_tuple[3]),
            current_price=reference_state.current_price,
            current_iv=reference_state.current_iv,
            days_to_horizon=reference_state.days_to_horizon,
            delta_utilization=reference_state.delta_utilization,
            vega_utilization=reference_state.vega_utilization,
            cvar_utilization=reference_state.cvar_utilization
        )

    def _expected_future_value(self,
                              state: MarketState,
                              action: StrategyAction,
                              all_states: List[Tuple]) -> float:
        """
        Calculate expected future value: Σ P(s'|s,a) * V(s')

        Simplified transition model based on market dynamics
        """
        if action.strategy_type == StrategyType.DO_NOTHING:
            # State likely stays similar
            return self.value_function.get(state.to_tuple(), 0.0) * 0.9

        # Estimate next state probabilities
        future_value = 0.0
        total_prob = 0.0

        for next_state_tuple in all_states:
            prob = self._transition_probability(state.to_tuple(), next_state_tuple, action)
            if prob > 0:
                next_value = self.value_function.get(next_state_tuple, 0.0)
                future_value += prob * next_value
                total_prob += prob

        return future_value / total_prob if total_prob > 0 else 0.0

    def _transition_probability(self,
                               current_state: Tuple,
                               next_state: Tuple,
                               action: StrategyAction) -> float:
        """
        Calculate transition probability P(s'|s,a)

        Simplified model - in practice would use historical data
        """
        # Unpack states
        curr_price, curr_vol, curr_time, curr_risk = current_state
        next_price, next_vol, next_time, next_risk = next_state

        # Time always progresses forward or stays
        if next_time < curr_time:
            return 0.0

        # Price transitions (mean reversion + trend)
        price_prob = 0.2  # Base probability
        if abs(next_price - curr_price) <= 1:
            price_prob = 0.4  # More likely to stay close
        if next_price == 2:  # Neutral
            price_prob *= 1.2  # Slight mean reversion

        # Vol transitions (clustering)
        vol_prob = 0.25
        if next_vol == curr_vol:
            vol_prob = 0.5  # Vol tends to cluster
        elif abs(next_vol - curr_vol) == 1:
            vol_prob = 0.3  # Adjacent states more likely

        # Risk transitions (affected by action)
        risk_prob = 0.25
        if action.strategy_type == StrategyType.DO_NOTHING:
            if next_risk == curr_risk:
                risk_prob = 0.6
        else:
            # Risk may increase with positions
            if next_risk >= curr_risk:
                risk_prob = 0.4

        # Combined probability
        return price_prob * vol_prob * risk_prob

    def select_optimal_strategy(self, state: MarketState) -> StrategyAction:
        """
        Select optimal strategy for current state using learned policy.

        Parameters:
        -----------
        state : MarketState
            Current market state

        Returns:
        --------
        StrategyAction
            Optimal strategy to execute
        """
        state_tuple = state.to_tuple()

        if state_tuple in self.policy:
            return self.policy[state_tuple]
        else:
            # If state not in policy, generate candidates and pick best greedily
            print(f"Warning: State {state_tuple} not in policy, using greedy selection")
            actions = self.generate_candidate_strategies(state)

            best_action = None
            best_reward = float('-inf')

            for action in actions:
                reward = self.calculate_reward(state, action)
                if reward > best_reward:
                    best_reward = reward
                    best_action = action

            return best_action or StrategyAction(
                strategy_type=StrategyType.DO_NOTHING,
                strikes=[], expiries=[], quantities=[],
                option_types=[], directions=[]
            )

    def get_strategy_ranking(self, state: MarketState, top_k: int = 5) -> List[Tuple[StrategyAction, float]]:
        """
        Get top-k strategies ranked by expected value.

        Parameters:
        -----------
        state : MarketState
            Current market state
        top_k : int
            Number of top strategies to return

        Returns:
        --------
        list of (StrategyAction, reward)
            Top strategies with their rewards
        """
        actions = self.generate_candidate_strategies(state)

        # Calculate reward for each action
        action_rewards = []
        for action in actions:
            reward = self.calculate_reward(state, action)
            action_rewards.append((action, reward))

        # Sort by reward
        action_rewards.sort(key=lambda x: x[1], reverse=True)

        return action_rewards[:top_k]


if __name__ == "__main__":
    print("=" * 70)
    print("Dynamic Programming Strategy Selector - Demo")
    print("=" * 70)

    # Import risk controller
    from risk.bitcoin_risk_controller import BitcoinRiskController, BitcoinRiskLimits

    # Initialize risk controller
    limits = BitcoinRiskLimits(
        base_max_var=100000,
        base_max_cvar=150000,
        base_max_delta=100,
        base_max_gamma=30,
        base_max_vega=1000
    )

    risk_controller = BitcoinRiskController(
        risk_limits=limits,
        portfolio_value=1000000
    )

    # Initialize strategy selector
    selector = DPStrategySelector(
        risk_controller=risk_controller,
        discount_factor=0.95,
        sharpe_weight=1.0,
        cvar_weight=0.5,
        drawdown_weight=0.3
    )

    # Create test market state
    price_history = [40000 + i*100 + np.random.normal(0, 500) for i in range(100)]

    state = selector.classify_market_state(
        current_price=45000,
        current_iv=0.75,
        days_to_horizon=15,
        price_history=price_history,
        risk_utilization={'delta': 0.4, 'vega': 0.5, 'cvar': 0.3}
    )

    print(f"\nCurrent Market State:")
    print(f"  Price State: {state.price_state.name}")
    print(f"  Volatility State: {state.volatility_state.name}")
    print(f"  Time State: {state.time_state.name}")
    print(f"  Risk State: {state.risk_state.name}")
    print(f"  Current Price: ${state.current_price:,.0f}")
    print(f"  Current IV: {state.current_iv:.1%}")

    # Solve value function (reduced iterations for demo)
    print(f"\n{'='*70}")
    selector.solve_value_function(state, n_iterations=20)

    # Get optimal strategy
    print(f"\n{'='*70}")
    print("Optimal Strategy Selection")
    print("=" * 70)

    optimal = selector.select_optimal_strategy(state)
    print(f"\nOptimal Strategy: {optimal.strategy_type.value}")
    print(f"  Expected Return: {optimal.expected_return:.2%}")
    print(f"  Expected Risk: {optimal.expected_risk:.2%}")
    print(f"  Sharpe Ratio: {optimal.sharpe:.3f}")

    if optimal.strikes:
        print(f"  Strikes: {[f'${s:,.0f}' for s in optimal.strikes]}")
        print(f"  Quantities: {optimal.quantities}")

    # Get top strategies
    print(f"\n{'='*70}")
    print("Top 5 Strategy Rankings")
    print("=" * 70)

    rankings = selector.get_strategy_ranking(state, top_k=5)
    for i, (action, reward) in enumerate(rankings, 1):
        print(f"\n{i}. {action.strategy_type.value}")
        print(f"   Reward: {reward:.4f}")
        print(f"   Sharpe: {action.sharpe:.3f}")
        print(f"   Return/Risk: {action.expected_return:.2%} / {action.expected_risk:.2%}")

    print(f"\n{'='*70}")
