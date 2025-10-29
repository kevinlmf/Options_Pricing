"""
Base Agent Classes for Multi-Agent Option Pricing

This module defines the fundamental building blocks for all market agents
in the option pricing deviation framework.
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
from enum import Enum


class AgentType(Enum):
    """Types of market agents"""
    MARKET_MAKER = "market_maker"
    ARBITRAGEUR = "arbitrageur"
    NOISE_TRADER = "noise_trader"
    INSTITUTIONAL = "institutional"


@dataclass
class AgentState:
    """Current state of an agent"""
    agent_id: str
    agent_type: AgentType
    cash_position: float = 0.0
    inventory: Dict[str, float] = field(default_factory=dict)  # {instrument: quantity}
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    risk_exposure: float = 0.0
    last_action_time: float = 0.0

    @property
    def total_pnl(self) -> float:
        """Total profit and loss"""
        return self.realized_pnl + self.unrealized_pnl

    @property
    def net_worth(self) -> float:
        """Net worth including cash and positions"""
        return self.cash_position + self.unrealized_pnl


@dataclass
class MarketState:
    """Current state of the option market"""
    timestamp: float
    underlying_price: float
    underlying_volatility: float
    risk_free_rate: float

    # Option market data
    option_prices: Dict[Tuple[float, float], float] = field(default_factory=dict)  # {(strike, expiry): price}
    theoretical_prices: Dict[Tuple[float, float], float] = field(default_factory=dict)
    implied_volatilities: Dict[Tuple[float, float], float] = field(default_factory=dict)

    # Market microstructure
    bid_ask_spreads: Dict[Tuple[float, float], Tuple[float, float]] = field(default_factory=dict)  # {(K,T): (bid, ask)}
    order_flow: Dict[Tuple[float, float], float] = field(default_factory=dict)  # Net order flow
    inventory_levels: Dict[str, float] = field(default_factory=dict)  # Aggregate inventory by agent type

    @property
    def pricing_deviations(self) -> Dict[Tuple[float, float], float]:
        """Calculate pricing deviations from theoretical values"""
        deviations = {}
        for key in self.option_prices:
            if key in self.theoretical_prices:
                deviations[key] = self.option_prices[key] - self.theoretical_prices[key]
        return deviations


class BaseAgent(ABC):
    """
    Abstract base class for all market agents.

    Defines the common interface and behavior that all agents must implement.
    Each agent can observe market state and make trading decisions.
    """

    def __init__(self, agent_id: str, agent_type: AgentType, initial_cash: float = 1000000.0):
        """
        Initialize base agent.

        Parameters:
        -----------
        agent_id : str
            Unique identifier for the agent
        agent_type : AgentType
            Type of agent (market maker, arbitrageur, etc.)
        initial_cash : float
            Initial cash position
        """
        self.state = AgentState(
            agent_id=agent_id,
            agent_type=agent_type,
            cash_position=initial_cash
        )

        # Performance tracking
        self.performance_history: List[Dict] = []

        # Risk management
        self.max_position_size: float = initial_cash * 0.1  # 10% of initial capital
        self.stop_loss_threshold: float = -initial_cash * 0.05  # 5% stop loss

    @abstractmethod
    def observe_market(self, market_state: MarketState) -> None:
        """
        Observe current market state and update internal state.

        Parameters:
        -----------
        market_state : MarketState
            Current market conditions
        """
        pass

    @abstractmethod
    def make_decision(self, market_state: MarketState) -> Dict[str, Union[str, float, Dict]]:
        """
        Make trading decision based on current market state.

        Parameters:
        -----------
        market_state : MarketState
            Current market conditions

        Returns:
        --------
        Dict containing:
        - 'action': 'buy', 'sell', 'hold', 'quote'
        - 'instrument': instrument identifier
        - 'quantity': size of order
        - 'price': limit price (if applicable)
        - 'metadata': additional information
        """
        pass

    def update_position(self, instrument: str, quantity: float, price: float) -> None:
        """
        Update agent's position after a trade.

        Parameters:
        -----------
        instrument : str
            Instrument identifier
        quantity : float
            Quantity traded (positive for buy, negative for sell)
        price : float
            Execution price
        """
        # Update inventory
        current_qty = self.state.inventory.get(instrument, 0.0)
        self.state.inventory[instrument] = current_qty + quantity

        # Update cash position
        cash_flow = -quantity * price  # Cash out for buy, cash in for sell
        self.state.cash_position += cash_flow

        # Update realized P&L
        if quantity * current_qty < 0:  # Position reduction
            # Calculate realized P&L for the closed portion
            reduction_qty = min(abs(quantity), abs(current_qty))
            # Simplified P&L calculation - in reality would need cost basis tracking
            realized_change = reduction_qty * price * np.sign(quantity)
            self.state.realized_pnl += realized_change

    def calculate_unrealized_pnl(self, market_state: MarketState) -> float:
        """
        Calculate unrealized P&L based on current market prices.

        Parameters:
        -----------
        market_state : MarketState
            Current market state with prices

        Returns:
        --------
        float: Unrealized P&L
        """
        unrealized_pnl = 0.0

        for instrument, quantity in self.state.inventory.items():
            if quantity != 0:
                # Try to find current market price
                current_price = self._get_market_price(instrument, market_state)
                if current_price is not None:
                    # Simplified unrealized P&L - assumes cost basis at current price
                    # In reality, would track actual cost basis
                    unrealized_pnl += quantity * current_price

        return unrealized_pnl

    def _get_market_price(self, instrument: str, market_state: MarketState) -> Optional[float]:
        """
        Get current market price for an instrument.

        Parameters:
        -----------
        instrument : str
            Instrument identifier
        market_state : MarketState
            Current market state

        Returns:
        --------
        Optional[float]: Current price if available
        """
        # Parse instrument identifier (assuming format like "CALL_100_0.25")
        try:
            parts = instrument.split('_')
            if len(parts) >= 3:
                option_type = parts[0]
                strike = float(parts[1])
                expiry = float(parts[2])

                key = (strike, expiry)
                return market_state.option_prices.get(key)
        except:
            pass

        return None

    def check_risk_limits(self) -> bool:
        """
        Check if agent is within risk limits.

        Returns:
        --------
        bool: True if within limits, False otherwise
        """
        # Check stop loss
        if self.state.total_pnl < self.stop_loss_threshold:
            return False

        # Check position size limits
        total_exposure = sum(abs(qty) for qty in self.state.inventory.values())
        if total_exposure > self.max_position_size:
            return False

        return True

    def record_performance(self, market_state: MarketState) -> None:
        """
        Record current performance metrics.

        Parameters:
        -----------
        market_state : MarketState
            Current market state
        """
        # Update unrealized P&L
        self.state.unrealized_pnl = self.calculate_unrealized_pnl(market_state)

        # Record performance snapshot
        performance_snapshot = {
            'timestamp': market_state.timestamp,
            'cash_position': self.state.cash_position,
            'unrealized_pnl': self.state.unrealized_pnl,
            'realized_pnl': self.state.realized_pnl,
            'total_pnl': self.state.total_pnl,
            'net_worth': self.state.net_worth,
            'inventory_count': len([qty for qty in self.state.inventory.values() if qty != 0]),
            'risk_exposure': self.state.risk_exposure
        }

        self.performance_history.append(performance_snapshot)

    def get_performance_summary(self) -> Dict:
        """
        Get summary of agent performance.

        Returns:
        --------
        Dict: Performance summary statistics
        """
        if not self.performance_history:
            return {}

        df = pd.DataFrame(self.performance_history)

        return {
            'total_return': self.state.total_pnl,
            'return_pct': self.state.total_pnl / (1000000.0 - self.state.total_pnl) * 100,  # Assuming 1M initial
            'max_pnl': df['total_pnl'].max(),
            'min_pnl': df['total_pnl'].min(),
            'volatility': df['total_pnl'].std(),
            'sharpe_ratio': df['total_pnl'].mean() / (df['total_pnl'].std() + 1e-8),
            'max_drawdown': (df['total_pnl'].cummax() - df['total_pnl']).max(),
            'trades_count': len(self.performance_history),
            'current_positions': len([qty for qty in self.state.inventory.values() if qty != 0])
        }