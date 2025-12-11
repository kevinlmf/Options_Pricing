"""
Trade Executor - Layer 7
========================

Portfolio construction and trade execution system.

This module handles the execution of portfolio optimization decisions,
including order generation, transaction cost optimization, and execution
algorithms.

Key Features:
- Order generation from optimization results
- Transaction cost modeling and optimization
- Execution algorithms (market orders, limit orders, VWAP)
- Portfolio rebalancing
- Settlement processing
- Performance attribution
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    VWAP = "vwap"  # Volume Weighted Average Price
    TWAP = "twap"  # Time Weighted Average Price


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


@dataclass
class ExecutionOrder:
    """Individual trade execution order"""
    asset: str
    side: OrderSide
    quantity: float
    order_type: OrderType
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    execution_timeframe: timedelta = timedelta(minutes=30)
    priority: int = 1  # 1 = highest priority
    notes: str = ""


@dataclass
class ExecutionResult:
    """Results from order execution"""
    order: ExecutionOrder
    executed_quantity: float
    executed_price: float
    execution_timestamp: datetime
    transaction_costs: float
    slippage: float
    status: str  # 'filled', 'partial', 'cancelled', 'expired'
    execution_duration: timedelta
    market_conditions: Dict[str, float] = field(default_factory=dict)


class TradeExecutor:
    """
    Portfolio construction and trade execution engine.

    This class takes optimization results from Layer 5 and executes
    the required trades while optimizing for transaction costs and
    market impact.
    """

    def __init__(self,
                 base_transaction_cost: float = 0.001,  # 10bps base cost
                 market_impact_model: str = "square_root",
                 max_market_impact: float = 0.005):  # 50bps max impact

        self.base_transaction_cost = base_transaction_cost
        self.market_impact_model = market_impact_model
        self.max_market_impact = max_market_impact

    def generate_execution_orders(self,
                                optimal_weights: Dict[str, float],
                                current_positions: Dict[str, float],
                                portfolio_value: float,
                                trading_signals: Dict,
                                risk_breaches: Optional[List] = None) -> List[ExecutionOrder]:
        """
        Generate execution orders from optimization results.

        Parameters:
        -----------
        optimal_weights : Dict[str, float]
            Target portfolio weights from optimizer
        current_positions : Dict[str, float]
            Current portfolio positions
        portfolio_value : float
            Total portfolio value
        trading_signals : Dict
            Trading signals from Layer 4
        risk_breaches : Optional[List]
            Risk breaches requiring urgent action

        Returns:
        --------
        List[ExecutionOrder] : List of execution orders
        """
        orders = []

        # Calculate required position changes
        position_changes = {}
        for asset in set(optimal_weights.keys()) | set(current_positions.keys()):
            target_weight = optimal_weights.get(asset, 0.0)
            current_weight = current_positions.get(asset, 0.0)
            weight_change = target_weight - current_weight

            if abs(weight_change) > 1e-6:  # Only trade meaningful changes
                position_changes[asset] = weight_change

        # Sort by priority and size
        sorted_changes = sorted(position_changes.items(),
                              key=lambda x: abs(x[1]), reverse=True)

        for asset, weight_change in sorted_changes:
            # Determine order side
            if weight_change > 0:
                side = OrderSide.BUY
            else:
                side = OrderSide.SELL

            # Calculate dollar amount
            dollar_amount = abs(weight_change) * portfolio_value

            # Determine order type based on urgency and signal strength
            order_type, priority, timeframe = self._determine_order_parameters(
                asset, trading_signals, risk_breaches, dollar_amount
            )

            # Create execution order
            order = ExecutionOrder(
                asset=asset,
                side=side,
                quantity=dollar_amount,
                order_type=order_type,
                execution_timeframe=timeframe,
                priority=priority,
                notes=f"Rebalancing to target weight {optimal_weights.get(asset, 0.0):.2%}"
            )

            orders.append(order)

        return orders

    def _determine_order_parameters(self, asset: str, trading_signals: Dict,
                                  risk_breaches: Optional[List],
                                  dollar_amount: float) -> Tuple[OrderType, int, timedelta]:
        """
        Determine order type, priority, and timeframe based on signals and urgency.
        """
        # Default parameters
        order_type = OrderType.VWAP
        priority = 3  # Normal priority
        timeframe = timedelta(hours=1)

        # Check for urgent risk breaches
        if risk_breaches:
            critical_breaches = [b for b in risk_breaches
                               if hasattr(b, 'severity') and
                               str(b.severity).upper() in ['CRITICAL', 'EMERGENCY']]
            if critical_breaches:
                order_type = OrderType.MARKET
                priority = 1  # Highest priority
                timeframe = timedelta(minutes=5)
                return order_type, priority, timeframe

        # Check trading signal strength
        if asset in trading_signals:
            signal = trading_signals[asset]
            if hasattr(signal, 'confidence'):
                confidence = signal.confidence
            elif isinstance(signal, dict) and 'confidence' in signal:
                confidence = signal.get('confidence', 0.5)
            else:
                confidence = 0.5

            if confidence > 0.8:
                # Strong signal - more aggressive execution
                order_type = OrderType.TWAP
                priority = 2
                timeframe = timedelta(minutes=30)
            elif confidence < 0.3:
                # Weak signal - more conservative
                order_type = OrderType.LIMIT
                priority = 4
                timeframe = timedelta(hours=2)

        # Adjust for trade size
        if dollar_amount > 100000:  # Large trade
            if order_type == OrderType.VWAP:
                timeframe = timedelta(hours=2)
        elif dollar_amount < 10000:  # Small trade
            order_type = OrderType.MARKET  # Execute immediately

        return order_type, priority, timeframe

    def execute_orders(self,
                      orders: List[ExecutionOrder],
                      market_data: pd.DataFrame,
                      current_prices: Dict[str, float],
                      available_cash: float) -> List[ExecutionResult]:
        """
        Execute a list of orders with transaction cost optimization.

        Parameters:
        -----------
        orders : List[ExecutionOrder]
            Orders to execute
        market_data : pd.DataFrame
            Recent market data for execution simulation
        current_prices : Dict[str, float]
            Current market prices
        available_cash : float
            Available cash for buying

        Returns:
        --------
        List[ExecutionResult] : Execution results
        """
        results = []

        # Sort orders by priority
        sorted_orders = sorted(orders, key=lambda x: x.priority)

        remaining_cash = available_cash

        for order in sorted_orders:
            # Check if we have enough cash for buys
            if order.side == OrderSide.BUY and remaining_cash < order.quantity:
                # Partial execution or skip
                executable_amount = min(order.quantity, remaining_cash)
                if executable_amount < 0.01 * order.quantity:  # Less than 1% executable
                    continue
                order.quantity = executable_amount

            # Execute the order
            result = self._execute_single_order(order, market_data, current_prices)

            if result.status == 'filled':
                # Update remaining cash
                if order.side == OrderSide.BUY:
                    remaining_cash -= result.executed_quantity * result.executed_price + result.transaction_costs
                else:
                    remaining_cash += result.executed_quantity * result.executed_price - result.transaction_costs

            results.append(result)

        return results

    def _execute_single_order(self,
                            order: ExecutionOrder,
                            market_data: pd.DataFrame,
                            current_prices: Dict[str, float]) -> ExecutionResult:
        """
        Execute a single order with realistic market simulation.
        """
        asset = order.asset
        current_price = current_prices.get(asset, 100.0)  # Default price

        # Simulate execution based on order type
        if order.order_type == OrderType.MARKET:
            # Immediate execution at current price with slippage
            slippage = self._calculate_market_impact(order.quantity, current_price, market_data)
            executed_price = current_price * (1 + slippage if order.side == OrderSide.SELL else 1 - slippage)
            executed_quantity = order.quantity / executed_price  # Convert dollars to shares
            execution_time = timedelta(seconds=np.random.uniform(1, 60))  # 1-60 seconds

        elif order.order_type == OrderType.LIMIT:
            # Limit order - may not fill immediately
            limit_price = order.limit_price or current_price * (1.005 if order.side == OrderSide.SELL else 0.995)
            # Simplified: assume fills if within 0.5% of current price
            if ((order.side == OrderSide.BUY and limit_price >= current_price * 0.995) or
                (order.side == OrderSide.SELL and limit_price <= current_price * 1.005)):
                executed_price = limit_price
                executed_quantity = order.quantity / executed_price
                execution_time = timedelta(minutes=np.random.uniform(1, 30))
            else:
                # No fill
                return ExecutionResult(
                    order=order,
                    executed_quantity=0.0,
                    executed_price=0.0,
                    execution_timestamp=datetime.now(),
                    transaction_costs=0.0,
                    slippage=0.0,
                    status='cancelled',
                    execution_duration=timedelta(0),
                    market_conditions={'reason': 'limit_price_not_reached'}
                )

        elif order.order_type in [OrderType.VWAP, OrderType.TWAP]:
            # Algorithmic execution over timeframe
            executed_price = self._simulate_algorithmic_execution(
                order, market_data, current_price
            )
            executed_quantity = order.quantity / executed_price
            execution_time = order.execution_timeframe

        else:
            # Default to market order
            executed_price = current_price
            executed_quantity = order.quantity / executed_price
            execution_time = timedelta(seconds=1)

        # Calculate transaction costs
        transaction_costs = self._calculate_transaction_costs(
            executed_quantity, executed_price, order.order_type
        )

        # Calculate slippage
        slippage = (executed_price - current_price) / current_price
        if order.side == OrderSide.SELL:
            slippage = -slippage  # Negative slippage for sells means better execution

        # Market conditions
        market_conditions = {
            'volatility': market_data['Close'].pct_change().std() if 'Close' in market_data.columns else 0.02,
            'volume': market_data.get('Volume', pd.Series([1000000])).iloc[-1] if 'Volume' in market_data.columns else 1000000,
            'spread': 0.0005  # Assume 5bps spread
        }

        return ExecutionResult(
            order=order,
            executed_quantity=executed_quantity,
            executed_price=executed_price,
            execution_timestamp=datetime.now(),
            transaction_costs=transaction_costs,
            slippage=slippage,
            status='filled',
            execution_duration=execution_time,
            market_conditions=market_conditions
        )

    def _calculate_market_impact(self, dollar_amount: float,
                               current_price: float,
                               market_data: pd.DataFrame) -> float:
        """Calculate expected market impact for large orders"""
        # Simplified market impact model
        avg_daily_volume = market_data.get('Volume', pd.Series([1000000])).mean()
        participation_rate = dollar_amount / (avg_daily_volume * current_price)

        if self.market_impact_model == "square_root":
            # Square root market impact model
            impact = 0.1 * np.sqrt(participation_rate)  # 10% impact coefficient
        else:
            # Linear model
            impact = 0.05 * participation_rate

        return min(impact, self.max_market_impact)

    def _simulate_algorithmic_execution(self,
                                      order: ExecutionOrder,
                                      market_data: pd.DataFrame,
                                      current_price: float) -> float:
        """Simulate VWAP/TWAP execution over timeframe"""
        # Simplified: assume execution at slight discount/premium
        if order.order_type == OrderType.VWAP:
            # VWAP typically gets better prices
            adjustment = np.random.normal(-0.001, 0.002)  # Small discount
        else:  # TWAP
            adjustment = np.random.normal(0.000, 0.003)  # Neutral to slight premium

        return current_price * (1 + adjustment)

    def _calculate_transaction_costs(self,
                                   quantity: float,
                                   price: float,
                                   order_type: OrderType) -> float:
        """Calculate total transaction costs"""
        dollar_amount = quantity * price

        # Base commission
        commission = dollar_amount * self.base_transaction_cost

        # Additional costs based on order type
        if order_type == OrderType.MARKET:
            additional_cost = dollar_amount * 0.0005  # 5bps for immediacy
        elif order_type in [OrderType.VWAP, OrderType.TWAP]:
            additional_cost = dollar_amount * 0.0002  # 2bps for algorithms
        else:
            additional_cost = 0.0

        # Market data fees, clearing, etc.
        other_fees = dollar_amount * 0.0001  # 1bps

        return commission + additional_cost + other_fees

    def generate_execution_report(self, results: List[ExecutionResult]) -> str:
        """Generate execution summary report"""
        total_executed = sum(r.executed_quantity * r.executed_price for r in results)
        total_costs = sum(r.transaction_costs for r in results)
        total_slippage = sum(abs(r.slippage) for r in results if r.status == 'filled')
        filled_orders = sum(1 for r in results if r.status == 'filled')

        report = ".2f"".2f"f"""
=== Trade Execution Report ===
Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Execution Summary:
- Orders Processed: {len(results)}
- Orders Filled: {filled_orders}
- Total Value Executed: ${total_executed:,.2f}
- Total Transaction Costs: ${total_costs:,.2f}
- Average Slippage: {total_slippage/filled_orders:.2% if filled_orders > 0 else 0:.2%}
- Cost as % of Volume: {total_costs/total_executed:.2% if total_executed > 0 else 0:.2%}

Order Details:
"""

        for result in results:
            if result.status == 'filled':
                direction = "BOUGHT" if result.order.side == OrderSide.BUY else "SOLD"
                report += f"- {direction} {result.order.asset}: "
                report += ".2f"
                report += f" (${result.transaction_costs:.2f} costs, "
                report += ".2f"

        report += "\n" + "="*50 + "\n"
        return report

