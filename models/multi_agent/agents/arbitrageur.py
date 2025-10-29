"""
Arbitrageur Agent

Implements an arbitrageur that identifies and exploits pricing deviations
from theoretical values, subject to capital constraints and transaction costs.
This agent helps reduce but cannot completely eliminate pricing deviations.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

from .base_agent import BaseAgent, AgentType, MarketState


@dataclass
class ArbitrageParameters:
    """Parameters controlling arbitrageur behavior"""

    # Detection thresholds
    min_deviation_threshold: float = 0.01  # Minimum profitable deviation (1%)
    min_profit_threshold: float = 100.0  # Minimum expected profit per trade

    # Capacity constraints
    max_capital_per_trade: float = 100000.0  # Maximum capital per arbitrage trade
    max_total_exposure: float = 1000000.0  # Maximum total exposure across all trades

    # Risk management
    max_holding_period: float = 0.1  # Maximum holding period (0.1 years = ~1 month)
    stop_loss_threshold: float = 0.05  # Stop loss at 5% of position

    # Transaction costs
    transaction_cost_rate: float = 0.002  # 0.2% transaction costs
    borrowing_cost_rate: float = 0.03  # 3% annual borrowing cost

    # Execution parameters
    execution_delay: float = 0.001  # Execution delay (reaction time)
    market_impact: float = 0.001  # Market impact per dollar traded


class Arbitrageur(BaseAgent):
    """
    Arbitrageur Agent

    Identifies pricing deviations from theoretical values and executes
    arbitrage trades to exploit them. The agent's behavior creates a
    natural limit to how large pricing deviations can become, but
    capacity constraints prevent complete elimination of deviations.

    Key features:
    - Detects profitable deviations above minimum threshold
    - Executes trades subject to capital constraints
    - Manages risk through position limits and stop losses
    - Accounts for transaction costs and market impact
    """

    def __init__(self, agent_id: str, parameters: ArbitrageParameters, initial_cash: float = 5000000.0):
        """
        Initialize arbitrageur.

        Parameters:
        -----------
        agent_id : str
            Unique identifier
        parameters : ArbitrageParameters
            Behavior parameters
        initial_cash : float
            Initial capital (typically higher than other agents)
        """
        super().__init__(agent_id, AgentType.ARBITRAGEUR, initial_cash)
        self.params = parameters

        # Arbitrageur specific state
        self.active_arbitrages: List[Dict] = []  # Current arbitrage positions
        self.completed_arbitrages: List[Dict] = []  # Historical arbitrage trades
        self.deviation_history: List[Dict] = []  # Track deviations over time

        # Performance tracking
        self.arbitrages_attempted = 0
        self.arbitrages_successful = 0
        self.total_arbitrage_profit = 0.0

    def observe_market(self, market_state: MarketState) -> None:
        """
        Observe market and identify arbitrage opportunities.

        Parameters:
        -----------
        market_state : MarketState
            Current market conditions
        """
        self.state.last_action_time = market_state.timestamp

        # Identify current pricing deviations
        deviations = self._identify_deviations(market_state)
        self._record_deviations(deviations, market_state.timestamp)

        # Monitor existing arbitrage positions
        self._monitor_existing_positions(market_state)

        # Update unrealized P&L
        self.state.unrealized_pnl = self.calculate_unrealized_pnl(market_state)

    def make_decision(self, market_state: MarketState) -> Dict[str, Union[str, float, Dict]]:
        """
        Make arbitrage decision based on identified opportunities.

        Parameters:
        -----------
        market_state : MarketState
            Current market conditions

        Returns:
        --------
        Dict: Arbitrage decision
        """
        # Check risk limits
        if not self.check_risk_limits():
            return {
                'action': 'hold',
                'reason': 'risk_limit_exceeded'
            }

        # Find best arbitrage opportunity
        best_opportunity = self._find_best_opportunity(market_state)

        if best_opportunity is None:
            return {
                'action': 'hold',
                'reason': 'no_profitable_opportunities'
            }

        # Execute arbitrage trade
        return self._execute_arbitrage(best_opportunity, market_state)

    def _identify_deviations(self, market_state: MarketState) -> Dict[Tuple[float, float], Dict]:
        """
        Identify pricing deviations from theoretical values.

        Parameters:
        -----------
        market_state : MarketState
            Current market state

        Returns:
        --------
        Dict: Deviations with analysis
        """
        deviations = {}

        for (strike, expiry) in market_state.theoretical_prices:
            theoretical_price = market_state.theoretical_prices[(strike, expiry)]
            market_price = market_state.option_prices.get((strike, expiry))

            if market_price is not None:
                # Calculate absolute and relative deviation
                abs_deviation = market_price - theoretical_price
                rel_deviation = abs_deviation / theoretical_price

                # Check if deviation exceeds threshold
                if abs(rel_deviation) >= self.params.min_deviation_threshold:
                    # Estimate potential profit after costs
                    expected_profit = self._estimate_profit(abs_deviation, theoretical_price)

                    deviations[(strike, expiry)] = {
                        'theoretical_price': theoretical_price,
                        'market_price': market_price,
                        'abs_deviation': abs_deviation,
                        'rel_deviation': rel_deviation,
                        'expected_profit': expected_profit,
                        'direction': 'buy' if abs_deviation < 0 else 'sell'  # Buy underpriced, sell overpriced
                    }

        return deviations

    def _estimate_profit(self, deviation: float, theoretical_price: float) -> float:
        """
        Estimate profit from arbitraging a deviation.

        Parameters:
        -----------
        deviation : float
            Price deviation
        theoretical_price : float
            Theoretical fair value

        Returns:
        --------
        float: Expected profit per unit
        """
        # Gross profit is the deviation
        gross_profit = abs(deviation)

        # Subtract transaction costs
        transaction_cost = theoretical_price * self.params.transaction_cost_rate * 2  # Buy and sell

        # Subtract estimated borrowing costs
        holding_cost = theoretical_price * self.params.borrowing_cost_rate * self.params.max_holding_period

        # Net profit
        net_profit = gross_profit - transaction_cost - holding_cost

        return net_profit

    def _find_best_opportunity(self, market_state: MarketState) -> Optional[Dict]:
        """
        Find the most profitable arbitrage opportunity.

        Parameters:
        -----------
        market_state : MarketState
            Current market state

        Returns:
        --------
        Optional[Dict]: Best opportunity or None
        """
        deviations = self._identify_deviations(market_state)

        if not deviations:
            return None

        # Filter by profitability
        profitable_ops = {
            key: value for key, value in deviations.items()
            if value['expected_profit'] >= self.params.min_profit_threshold
        }

        if not profitable_ops:
            return None

        # Find opportunity with highest expected profit
        best_key = max(profitable_ops.keys(), key=lambda k: profitable_ops[k]['expected_profit'])
        best_opportunity = profitable_ops[best_key]
        best_opportunity['strike'] = best_key[0]
        best_opportunity['expiry'] = best_key[1]

        return best_opportunity

    def _execute_arbitrage(self, opportunity: Dict, market_state: MarketState) -> Dict[str, Union[str, float, Dict]]:
        """
        Execute arbitrage trade.

        Parameters:
        -----------
        opportunity : Dict
            Arbitrage opportunity details
        market_state : MarketState
            Current market state

        Returns:
        --------
        Dict: Trade execution details
        """
        strike = opportunity['strike']
        expiry = opportunity['expiry']
        direction = opportunity['direction']

        # Calculate position size based on capital constraints
        position_size = self._calculate_position_size(opportunity)

        if position_size <= 0:
            return {
                'action': 'hold',
                'reason': 'insufficient_capital'
            }

        # Create arbitrage record
        arbitrage_record = {
            'arbitrage_id': f"{self.state.agent_id}_{len(self.active_arbitrages)}",
            'strike': strike,
            'expiry': expiry,
            'direction': direction,
            'entry_time': market_state.timestamp,
            'entry_price': opportunity['market_price'],
            'theoretical_price': opportunity['theoretical_price'],
            'position_size': position_size,
            'expected_profit': opportunity['expected_profit'] * position_size,
            'status': 'active'
        }

        # Add to active arbitrages
        self.active_arbitrages.append(arbitrage_record)

        # Update position
        instrument = f"CALL_{strike}_{expiry}"
        quantity = position_size if direction == 'buy' else -position_size
        self.update_position(instrument, quantity, opportunity['market_price'])

        # Track attempt
        self.arbitrages_attempted += 1

        return {
            'action': 'arbitrage',
            'direction': direction,
            'instrument': instrument,
            'quantity': abs(quantity),
            'price': opportunity['market_price'],
            'expected_profit': arbitrage_record['expected_profit'],
            'arbitrage_id': arbitrage_record['arbitrage_id']
        }

    def _calculate_position_size(self, opportunity: Dict) -> float:
        """
        Calculate optimal position size for arbitrage.

        Parameters:
        -----------
        opportunity : Dict
            Arbitrage opportunity

        Returns:
        --------
        float: Position size (number of contracts)
        """
        # Maximum size based on capital per trade
        max_capital_size = self.params.max_capital_per_trade / opportunity['theoretical_price']

        # Maximum size based on total exposure limit
        current_exposure = sum(abs(qty) * price for qty, price in
                             [(self.state.inventory.get(f"CALL_{k[0]}_{k[1]}", 0), v)
                              for k, v in opportunity.items() if isinstance(k, tuple)])

        remaining_capacity = (self.params.max_total_exposure - current_exposure) / opportunity['theoretical_price']
        max_exposure_size = max(0, remaining_capacity)

        # Take the minimum of constraints
        position_size = min(max_capital_size, max_exposure_size)

        return max(0, int(position_size))  # Round down to integer contracts

    def _monitor_existing_positions(self, market_state: MarketState) -> None:
        """
        Monitor existing arbitrage positions for closure.

        Parameters:
        -----------
        market_state : MarketState
            Current market state
        """
        positions_to_close = []

        for i, arbitrage in enumerate(self.active_arbitrages):
            # Check if position should be closed
            should_close, reason = self._should_close_position(arbitrage, market_state)

            if should_close:
                # Close the position
                self._close_arbitrage_position(arbitrage, market_state, reason)
                positions_to_close.append(i)

        # Remove closed positions (in reverse order to maintain indices)
        for i in reversed(positions_to_close):
            self.active_arbitrages.pop(i)

    def _should_close_position(self, arbitrage: Dict, market_state: MarketState) -> Tuple[bool, str]:
        """
        Determine if an arbitrage position should be closed.

        Parameters:
        -----------
        arbitrage : Dict
            Arbitrage position details
        market_state : MarketState
            Current market state

        Returns:
        --------
        Tuple[bool, str]: (should_close, reason)
        """
        # Check time limit
        holding_period = market_state.timestamp - arbitrage['entry_time']
        if holding_period >= self.params.max_holding_period:
            return True, 'time_limit'

        # Check if deviation has been corrected
        current_market_price = market_state.option_prices.get((arbitrage['strike'], arbitrage['expiry']))
        current_theoretical_price = market_state.theoretical_prices.get((arbitrage['strike'], arbitrage['expiry']))

        if current_market_price is not None and current_theoretical_price is not None:
            current_deviation = abs(current_market_price - current_theoretical_price)
            relative_deviation = current_deviation / current_theoretical_price

            # Close if deviation has been sufficiently reduced
            if relative_deviation < self.params.min_deviation_threshold * 0.5:
                return True, 'deviation_corrected'

        # Check stop loss
        if current_market_price is not None:
            if arbitrage['direction'] == 'buy':
                loss = arbitrage['entry_price'] - current_market_price
            else:
                loss = current_market_price - arbitrage['entry_price']

            loss_pct = loss / arbitrage['entry_price']
            if loss_pct >= self.params.stop_loss_threshold:
                return True, 'stop_loss'

        return False, 'hold'

    def _close_arbitrage_position(self, arbitrage: Dict, market_state: MarketState, reason: str) -> None:
        """
        Close an arbitrage position.

        Parameters:
        -----------
        arbitrage : Dict
            Arbitrage position to close
        market_state : MarketState
            Current market state
        reason : str
            Reason for closing
        """
        strike = arbitrage['strike']
        expiry = arbitrage['expiry']
        instrument = f"CALL_{strike}_{expiry}"

        # Get current market price
        current_price = market_state.option_prices.get((strike, expiry), arbitrage['entry_price'])

        # Calculate realized profit
        if arbitrage['direction'] == 'buy':
            profit_per_unit = current_price - arbitrage['entry_price']
        else:
            profit_per_unit = arbitrage['entry_price'] - current_price

        total_profit = profit_per_unit * arbitrage['position_size']

        # Update position (reverse the original trade)
        close_quantity = -arbitrage['position_size'] if arbitrage['direction'] == 'buy' else arbitrage['position_size']
        self.update_position(instrument, close_quantity, current_price)

        # Update performance tracking
        self.total_arbitrage_profit += total_profit
        if total_profit > 0:
            self.arbitrages_successful += 1

        # Record completed arbitrage
        arbitrage_record = arbitrage.copy()
        arbitrage_record.update({
            'exit_time': market_state.timestamp,
            'exit_price': current_price,
            'realized_profit': total_profit,
            'close_reason': reason,
            'status': 'completed'
        })

        self.completed_arbitrages.append(arbitrage_record)

    def _record_deviations(self, deviations: Dict, timestamp: float) -> None:
        """Record deviation data for analysis."""
        for key, deviation in deviations.items():
            record = {
                'timestamp': timestamp,
                'strike': key[0],
                'expiry': key[1],
                **deviation
            }
            self.deviation_history.append(record)

    def get_arbitrage_stats(self) -> Dict:
        """
        Get arbitrage-specific performance statistics.

        Returns:
        --------
        Dict: Arbitrage performance metrics
        """
        base_stats = self.get_performance_summary()

        arbitrage_stats = {
            'arbitrages_attempted': self.arbitrages_attempted,
            'arbitrages_successful': self.arbitrages_successful,
            'success_rate': self.arbitrages_successful / max(1, self.arbitrages_attempted) * 100,
            'total_arbitrage_profit': self.total_arbitrage_profit,
            'active_arbitrages': len(self.active_arbitrages),
            'completed_arbitrages': len(self.completed_arbitrages),
            'avg_profit_per_arbitrage': self.total_arbitrage_profit / max(1, len(self.completed_arbitrages))
        }

        return {**base_stats, **arbitrage_stats}