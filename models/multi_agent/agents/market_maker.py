"""
Market Maker Agent

Implements a market maker that provides liquidity by quoting bid-ask spreads
while managing inventory risk. This agent's behavior contributes to pricing
deviations through inventory-driven quote adjustments.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

from .base_agent import BaseAgent, AgentType, MarketState


@dataclass
class MarketMakerParameters:
    """Parameters controlling market maker behavior"""

    # Inventory management
    max_inventory: float = 100.0  # Maximum inventory per option
    inventory_risk_aversion: float = 0.01  # How much inventory affects quotes

    # Spread management
    base_spread: float = 0.02  # Base bid-ask spread (2%)
    volatility_adjustment: float = 0.5  # Spread increases with volatility

    # Risk management
    position_limit: float = 1000.0  # Maximum total position size
    loss_limit: float = -50000.0  # Maximum loss tolerance

    # Pricing adjustments
    adverse_selection_protection: float = 0.001  # Protection against informed traders
    liquidity_provision_incentive: float = 0.0005  # Incentive for providing liquidity


class MarketMaker(BaseAgent):
    """
    Market Maker Agent

    Provides liquidity by continuously quoting bid-ask spreads across different
    options. Adjusts quotes based on:
    - Current inventory levels (inventory risk)
    - Market volatility and uncertainty
    - Order flow information
    - Risk exposure

    This creates realistic pricing deviations as inventory imbalances
    cause the market maker to skew quotes away from theoretical values.
    """

    def __init__(self, agent_id: str, parameters: MarketMakerParameters, initial_cash: float = 2000000.0):
        """
        Initialize market maker.

        Parameters:
        -----------
        agent_id : str
            Unique identifier
        parameters : MarketMakerParameters
            Behavior parameters
        initial_cash : float
            Initial capital
        """
        super().__init__(agent_id, AgentType.MARKET_MAKER, initial_cash)
        self.params = parameters

        # Market maker specific state
        self.current_quotes: Dict[Tuple[float, float], Tuple[float, float]] = {}  # {(K,T): (bid, ask)}
        self.inventory_target: Dict[str, float] = {}  # Target inventory levels
        self.last_order_flow: Dict[Tuple[float, float], float] = {}  # Recent order flow

        # Performance tracking
        self.quotes_provided = 0
        self.trades_executed = 0
        self.inventory_risk_history: List[float] = []

    def observe_market(self, market_state: MarketState) -> None:
        """
        Observe market and update internal state.

        Parameters:
        -----------
        market_state : MarketState
            Current market conditions
        """
        self.state.last_action_time = market_state.timestamp

        # Update inventory risk assessment
        self._assess_inventory_risk(market_state)

        # Track order flow
        self._update_order_flow(market_state)

        # Update unrealized P&L
        self.state.unrealized_pnl = self.calculate_unrealized_pnl(market_state)

    def make_decision(self, market_state: MarketState) -> Dict[str, Union[str, float, Dict]]:
        """
        Make market making decision - provide quotes for options.

        Parameters:
        -----------
        market_state : MarketState
            Current market conditions

        Returns:
        --------
        Dict: Market making decision with quotes
        """
        # Check risk limits
        if not self.check_risk_limits():
            return {
                'action': 'hold',
                'reason': 'risk_limit_exceeded',
                'quotes': {}
            }

        # Generate quotes for all available options
        new_quotes = self._generate_quotes(market_state)
        self.current_quotes = new_quotes
        self.quotes_provided += 1

        return {
            'action': 'quote',
            'quotes': new_quotes,
            'agent_id': self.state.agent_id,
            'timestamp': market_state.timestamp
        }

    def _generate_quotes(self, market_state: MarketState) -> Dict[Tuple[float, float], Tuple[float, float]]:
        """
        Generate bid-ask quotes for all options.

        Parameters:
        -----------
        market_state : MarketState
            Current market state

        Returns:
        --------
        Dict: {(strike, expiry): (bid, ask)} quotes
        """
        quotes = {}

        for (strike, expiry), theoretical_price in market_state.theoretical_prices.items():
            # Calculate base spread
            spread = self._calculate_spread(strike, expiry, market_state)

            # Calculate inventory adjustment
            inventory_adjustment = self._calculate_inventory_adjustment(strike, expiry, market_state)

            # Calculate risk adjustment
            risk_adjustment = self._calculate_risk_adjustment(strike, expiry, market_state)

            # Final quotes
            mid_price = theoretical_price + inventory_adjustment + risk_adjustment
            half_spread = spread / 2

            bid = max(0.01, mid_price - half_spread)  # Minimum price of $0.01
            ask = mid_price + half_spread

            quotes[(strike, expiry)] = (bid, ask)

        return quotes

    def _calculate_spread(self, strike: float, expiry: float, market_state: MarketState) -> float:
        """
        Calculate bid-ask spread based on market conditions.

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
        float: Bid-ask spread
        """
        # Base spread
        spread = self.params.base_spread

        # Volatility adjustment
        volatility_mult = 1 + self.params.volatility_adjustment * market_state.underlying_volatility
        spread *= volatility_mult

        # Time to expiry adjustment (shorter expiry = wider spread)
        time_mult = 1 + max(0, (0.1 - expiry) * 2)  # Wider spreads for options < 0.1 years
        spread *= time_mult

        # Moneyness adjustment (far OTM = wider spread)
        moneyness = abs(np.log(market_state.underlying_price / strike))
        moneyness_mult = 1 + moneyness * 0.5
        spread *= moneyness_mult

        return spread

    def _calculate_inventory_adjustment(self, strike: float, expiry: float, market_state: MarketState) -> float:
        """
        Calculate inventory-driven price adjustment.

        This is the key mechanism that creates pricing deviations from theoretical values.

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
        float: Price adjustment due to inventory position
        """
        instrument = f"CALL_{strike}_{expiry}"
        current_inventory = self.state.inventory.get(instrument, 0.0)

        # Inventory pressure: positive inventory -> lower quotes to sell
        #                   negative inventory -> higher quotes to buy
        inventory_pressure = -current_inventory * self.params.inventory_risk_aversion

        # Scale by theoretical price
        theoretical_price = market_state.theoretical_prices.get((strike, expiry), 1.0)
        inventory_adjustment = inventory_pressure * theoretical_price

        return inventory_adjustment

    def _calculate_risk_adjustment(self, strike: float, expiry: float, market_state: MarketState) -> float:
        """
        Calculate risk-based price adjustment.

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
        float: Risk-based price adjustment
        """
        # Adverse selection protection
        adverse_selection = self.params.adverse_selection_protection

        # Order flow adjustment
        recent_flow = self.last_order_flow.get((strike, expiry), 0.0)
        flow_adjustment = recent_flow * 0.001  # Small adjustment based on recent flow

        # Total risk adjustment
        return adverse_selection + flow_adjustment

    def _assess_inventory_risk(self, market_state: MarketState) -> None:
        """
        Assess current inventory risk exposure.

        Parameters:
        -----------
        market_state : MarketState
            Current market state
        """
        total_risk = 0.0

        for instrument, quantity in self.state.inventory.items():
            if quantity != 0:
                # Calculate dollar risk (delta-equivalent exposure)
                price = self._get_market_price(instrument, market_state) or 1.0
                position_value = abs(quantity * price)

                # Add gamma risk for options (simplified)
                gamma_risk = position_value * 0.1  # Assume 10% gamma risk
                total_risk += position_value + gamma_risk

        self.state.risk_exposure = total_risk
        self.inventory_risk_history.append(total_risk)

    def _update_order_flow(self, market_state: MarketState) -> None:
        """
        Update order flow tracking.

        Parameters:
        -----------
        market_state : MarketState
            Current market state
        """
        # Update with current order flow from market state
        self.last_order_flow = market_state.order_flow.copy()

    def execute_trade(self, strike: float, expiry: float, quantity: float, price: float, is_buyer: bool) -> None:
        """
        Execute a trade with a counterparty.

        Parameters:
        -----------
        strike : float
            Option strike
        expiry : float
            Option expiry
        quantity : float
            Trade quantity
        price : float
            Trade price
        is_buyer : bool
            Whether market maker is buying (True) or selling (False)
        """
        instrument = f"CALL_{strike}_{expiry}"

        # Determine actual quantity based on role
        actual_quantity = quantity if is_buyer else -quantity

        # Update position
        self.update_position(instrument, actual_quantity, price)

        # Track trade
        self.trades_executed += 1

        # Update performance
        self.record_performance_snapshot(strike, expiry, price)

    def record_performance_snapshot(self, strike: float, expiry: float, price: float) -> None:
        """Record performance after a trade."""
        # This would be called after each trade to track performance
        pass

    def get_market_making_stats(self) -> Dict:
        """
        Get market making specific performance statistics.

        Returns:
        --------
        Dict: Market making performance metrics
        """
        base_stats = self.get_performance_summary()

        mm_stats = {
            'quotes_provided': self.quotes_provided,
            'trades_executed': self.trades_executed,
            'avg_inventory_risk': np.mean(self.inventory_risk_history) if self.inventory_risk_history else 0,
            'max_inventory_risk': max(self.inventory_risk_history) if self.inventory_risk_history else 0,
            'current_active_quotes': len(self.current_quotes),
            'inventory_positions': len([qty for qty in self.state.inventory.values() if abs(qty) > 0.01])
        }

        return {**base_stats, **mm_stats}