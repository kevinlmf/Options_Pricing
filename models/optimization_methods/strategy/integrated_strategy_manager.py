"""
Integrated Strategy Manager

Unifies traditional quantitative strategies with options pricing and risk control:

Flow:
1. Traditional strategies generate trading signals
2. Signals are converted to option strategies using pricing models
3. DP optimizer selects best option structure
4. Risk controller validates trades
5. Approved trades are executed

This creates a complete pipeline from signal → strategy → pricing → risk → execution
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from .traditional_strategies import (
    StrategyEnsemble,
    TradingSignal,
    SignalType
)
from .dp_strategy_selector import (
    DPStrategySelector,
    StrategyType,
    StrategyAction,
    MarketState
)
from risk.bitcoin_risk_controller import (
    BitcoinRiskController,
    OrderProposal,
    RiskCheckResult
)
from ...options_pricing.black_scholes import BlackScholesModel, BSParameters


@dataclass
class IntegratedStrategyOutput:
    """Complete output from integrated strategy system"""
    # Input signals
    traditional_signals: Dict[str, TradingSignal]

    # Option strategy selection
    option_strategy: StrategyAction
    strategy_alternatives: List[Tuple[StrategyAction, float]]  # (strategy, reward)

    # Risk validation
    orders: List[OrderProposal]
    risk_checks: List[RiskCheckResult]
    approved_orders: List[OrderProposal]
    rejected_orders: List[OrderProposal]

    # Execution summary
    total_notional: float
    total_delta: float
    total_vega: float
    expected_pnl: float
    max_loss: float

    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class SignalToOptionConverter:
    """
    Converts traditional trading signals to option strategies

    Maps directional/volatility signals to appropriate option structures
    """

    @staticmethod
    def signal_to_option_strategy(signal: TradingSignal,
                                  current_price: float,
                                  current_iv: float,
                                  time_horizon_days: int = 30) -> StrategyType:
        """
        Convert trading signal to option strategy type

        Signal Mapping:
        - STRONG_BUY (bullish) → Bull Call Spread / Long Call
        - BUY (bullish) → Long Call
        - STRONG_SELL (bearish) → Bear Put Spread / Long Put
        - SELL (bearish) → Long Put
        - buy_volatility → Long Straddle / Strangle
        - sell_volatility → Iron Condor / Short Straddle
        - hedge_long/short → Delta hedging with options
        """

        direction = signal.direction.lower()
        confidence = signal.confidence

        # Volatility strategies
        if 'volatility' in direction:
            if direction == 'buy_volatility':
                return StrategyType.STRADDLE if confidence > 0.7 else StrategyType.STRANGLE
            elif direction == 'sell_volatility':
                return StrategyType.IRON_CONDOR if confidence > 0.7 else StrategyType.BUTTERFLY
            else:
                return StrategyType.DO_NOTHING

        # Directional strategies
        elif direction == 'bullish':
            if signal.signal_type == SignalType.STRONG_BUY and confidence > 0.8:
                return StrategyType.BULL_CALL_SPREAD
            elif signal.signal_type in [SignalType.BUY, SignalType.STRONG_BUY]:
                return StrategyType.LONG_CALL
            else:
                return StrategyType.DO_NOTHING

        elif direction == 'bearish':
            if signal.signal_type == SignalType.STRONG_SELL and confidence > 0.8:
                return StrategyType.BEAR_PUT_SPREAD
            elif signal.signal_type in [SignalType.SELL, SignalType.STRONG_SELL]:
                return StrategyType.LONG_PUT
            else:
                return StrategyType.DO_NOTHING

        # Hedging strategies
        elif 'hedge' in direction:
            if direction == 'hedge_long':
                return StrategyType.LONG_CALL
            elif direction == 'hedge_short':
                return StrategyType.LONG_PUT
            else:
                return StrategyType.DO_NOTHING

        else:
            return StrategyType.DO_NOTHING

    @staticmethod
    def create_strategy_action(strategy_type: StrategyType,
                              current_price: float,
                              current_iv: float,
                              time_horizon_days: int,
                              confidence: float,
                              notional_size: float = 50000) -> StrategyAction:
        """
        Create concrete strategy action with strikes, quantities, etc.

        Parameters:
        -----------
        strategy_type : StrategyType
            Type of option strategy
        current_price : float
            Current underlying price
        current_iv : float
            Current implied volatility
        time_horizon_days : int
            Days until expiration
        confidence : float
            Signal confidence (affects position sizing)
        notional_size : float
            Base notional size in dollars
        """
        expiry = time_horizon_days / 365.0
        risk_free_rate = 0.05

        # Adjust quantity based on confidence
        base_quantity = int(notional_size / current_price * confidence * 0.1)
        base_quantity = max(1, min(base_quantity, 20))  # Cap at 1-20 contracts

        if strategy_type == StrategyType.LONG_CALL:
            # Long call slightly OTM
            strike = current_price * 1.05
            return StrategyAction(
                strategy_type=strategy_type,
                strikes=[strike],
                expiries=[expiry],
                quantities=[base_quantity],
                option_types=['call'],
                directions=['buy']
            )

        elif strategy_type == StrategyType.LONG_PUT:
            # Long put slightly OTM
            strike = current_price * 0.95
            return StrategyAction(
                strategy_type=strategy_type,
                strikes=[strike],
                expiries=[expiry],
                quantities=[base_quantity],
                option_types=['put'],
                directions=['buy']
            )

        elif strategy_type == StrategyType.BULL_CALL_SPREAD:
            # Buy lower strike, sell higher strike
            lower_strike = current_price * 1.02
            upper_strike = current_price * 1.10
            return StrategyAction(
                strategy_type=strategy_type,
                strikes=[lower_strike, upper_strike],
                expiries=[expiry, expiry],
                quantities=[base_quantity, base_quantity],
                option_types=['call', 'call'],
                directions=['buy', 'sell']
            )

        elif strategy_type == StrategyType.BEAR_PUT_SPREAD:
            # Buy higher strike, sell lower strike
            upper_strike = current_price * 0.98
            lower_strike = current_price * 0.90
            return StrategyAction(
                strategy_type=strategy_type,
                strikes=[upper_strike, lower_strike],
                expiries=[expiry, expiry],
                quantities=[base_quantity, base_quantity],
                option_types=['put', 'put'],
                directions=['buy', 'sell']
            )

        elif strategy_type == StrategyType.STRADDLE:
            # Buy ATM call and put
            strike = current_price
            qty = max(1, base_quantity // 2)
            return StrategyAction(
                strategy_type=strategy_type,
                strikes=[strike, strike],
                expiries=[expiry, expiry],
                quantities=[qty, qty],
                option_types=['call', 'put'],
                directions=['buy', 'buy']
            )

        elif strategy_type == StrategyType.STRANGLE:
            # Buy OTM call and put
            call_strike = current_price * 1.05
            put_strike = current_price * 0.95
            qty = max(1, base_quantity // 2)
            return StrategyAction(
                strategy_type=strategy_type,
                strikes=[call_strike, put_strike],
                expiries=[expiry, expiry],
                quantities=[qty, qty],
                option_types=['call', 'put'],
                directions=['buy', 'buy']
            )

        elif strategy_type == StrategyType.IRON_CONDOR:
            # Sell OTM call spread and put spread
            qty = max(1, base_quantity // 2)
            strikes = [
                current_price * 0.90,  # Long put (far OTM)
                current_price * 0.95,  # Short put
                current_price * 1.05,  # Short call
                current_price * 1.10   # Long call (far OTM)
            ]
            return StrategyAction(
                strategy_type=strategy_type,
                strikes=strikes,
                expiries=[expiry] * 4,
                quantities=[qty, qty, qty, qty],
                option_types=['put', 'put', 'call', 'call'],
                directions=['buy', 'sell', 'sell', 'buy']
            )

        elif strategy_type == StrategyType.BUTTERFLY:
            # ATM butterfly
            qty = max(1, base_quantity // 3)
            lower = current_price * 0.95
            middle = current_price
            upper = current_price * 1.05
            return StrategyAction(
                strategy_type=strategy_type,
                strikes=[lower, middle, upper],
                expiries=[expiry, expiry, expiry],
                quantities=[qty, qty*2, qty],
                option_types=['call', 'call', 'call'],
                directions=['buy', 'sell', 'buy']
            )

        else:  # DO_NOTHING
            return StrategyAction(
                strategy_type=StrategyType.DO_NOTHING,
                strikes=[],
                expiries=[],
                quantities=[],
                option_types=[],
                directions=[]
            )


class IntegratedStrategyManager:
    """
    Main integration class that combines:
    1. Traditional strategies (signal generation)
    2. Options pricing (strategy construction)
    3. DP optimization (strategy selection)
    4. Risk management (validation & execution)
    """

    def __init__(self,
                 risk_controller: BitcoinRiskController,
                 enable_dp_optimization: bool = True,
                 enable_risk_control: bool = True):
        """
        Initialize integrated strategy manager

        Parameters:
        -----------
        risk_controller : BitcoinRiskController
            Risk controller for order validation
        enable_dp_optimization : bool
            Enable DP-based strategy optimization
        enable_risk_control : bool
            Enable risk control validation
        """
        # Components
        self.traditional_strategies = StrategyEnsemble()
        self.risk_controller = risk_controller
        self.signal_converter = SignalToOptionConverter()

        # Optional DP optimizer
        self.enable_dp = enable_dp_optimization
        if enable_dp_optimization:
            self.dp_selector = DPStrategySelector(
                risk_controller=risk_controller,
                discount_factor=0.95,
                sharpe_weight=1.0,
                cvar_weight=0.5,
                drawdown_weight=0.3
            )
        else:
            self.dp_selector = None

        self.enable_risk_control = enable_risk_control

        # Statistics
        self.execution_history: List[IntegratedStrategyOutput] = []
        self.total_signals_generated = 0
        self.total_strategies_created = 0
        self.total_orders_submitted = 0
        self.total_orders_approved = 0
        self.total_orders_rejected = 0

    def execute_integrated_strategy(self,
                                    prices: List[float],
                                    current_price: float,
                                    current_iv: float,
                                    current_delta: float = 0.0,
                                    time_horizon_days: int = 30) -> IntegratedStrategyOutput:
        """
        Execute complete integrated strategy pipeline

        Flow:
        1. Generate traditional strategy signals
        2. Convert signals to option strategies
        3. (Optional) Use DP to optimize selection
        4. Create orders with options pricing
        5. Risk-check all orders
        6. Return execution summary

        Parameters:
        -----------
        prices : list of float
            Historical price data
        current_price : float
            Current underlying price
        current_iv : float
            Current implied volatility
        current_delta : float
            Current portfolio delta
        time_horizon_days : int
            Trading horizon in days

        Returns:
        --------
        IntegratedStrategyOutput
            Complete execution summary
        """
        print("\n" + "="*80)
        print("INTEGRATED STRATEGY EXECUTION")
        print("="*80)

        # Step 1: Generate traditional strategy signals
        print("\n[Step 1/5] Generating Traditional Strategy Signals...")
        signals = self.traditional_strategies.generate_ensemble_signal(
            prices=prices,
            current_price=current_price,
            current_iv=current_iv,
            current_delta=current_delta
        )
        self.total_signals_generated += 1

        print(self.traditional_strategies.get_signal_summary(signals))

        # Step 2: Convert ensemble signal to option strategy
        print("\n[Step 2/5] Converting Signal to Option Strategy...")
        ensemble_signal = signals['ensemble']

        option_strategy_type = self.signal_converter.signal_to_option_strategy(
            signal=ensemble_signal,
            current_price=current_price,
            current_iv=current_iv,
            time_horizon_days=time_horizon_days
        )

        option_strategy = self.signal_converter.create_strategy_action(
            strategy_type=option_strategy_type,
            current_price=current_price,
            current_iv=current_iv,
            time_horizon_days=time_horizon_days,
            confidence=ensemble_signal.confidence
        )

        print(f"  Selected Option Strategy: {option_strategy.strategy_type.value}")
        print(f"  Number of Legs: {len(option_strategy.strikes)}")
        self.total_strategies_created += 1

        # Step 3: (Optional) DP Optimization
        strategy_alternatives = []
        if self.enable_dp and option_strategy.strategy_type != StrategyType.DO_NOTHING:
            print("\n[Step 3/5] DP Strategy Optimization...")
            # This would normally involve solving the full DP
            # For now, we just use the direct conversion
            print("  Using direct signal-to-strategy conversion")
        else:
            print("\n[Step 3/5] DP Optimization: DISABLED")

        # Step 4: Create orders from strategy
        print("\n[Step 4/5] Creating Orders with Options Pricing...")
        orders = self._create_orders_from_strategy(
            option_strategy,
            current_price,
            current_iv
        )

        print(f"  Created {len(orders)} orders")
        for i, order in enumerate(orders, 1):
            print(f"    Order {i}: {order.direction.upper()} {order.quantity} "
                  f"{order.option_type.upper()} @ ${order.strike:,.0f}")

        self.total_orders_submitted += len(orders)

        # Step 5: Risk control validation
        print("\n[Step 5/5] Risk Control Validation...")
        risk_checks = []
        approved_orders = []
        rejected_orders = []

        if self.enable_risk_control and len(orders) > 0:
            for order in orders:
                result = self.risk_controller.check_order(order)
                risk_checks.append(result)

                if result.approved:
                    approved_orders.append(order)
                    self.total_orders_approved += 1
                    print(f"  ✓ Order APPROVED: {order.option_type} {order.strike}")
                else:
                    rejected_orders.append(order)
                    self.total_orders_rejected += 1
                    print(f"  ✗ Order REJECTED: {order.option_type} {order.strike}")
                    print(f"    Reason: {result.reasons[0] if result.reasons else 'Unknown'}")
        else:
            print("  Risk control: DISABLED - All orders auto-approved")
            approved_orders = orders
            self.total_orders_approved += len(orders)

        # Calculate execution summary
        total_notional = sum(self._calculate_order_notional(o, current_price) for o in approved_orders)
        total_delta = sum(risk_checks[i].proposed_metrics.get('order_delta', 0)
                         for i, o in enumerate(orders) if o in approved_orders) if risk_checks else 0
        total_vega = sum(risk_checks[i].proposed_metrics.get('order_vega', 0)
                        for i, o in enumerate(orders) if o in approved_orders) if risk_checks else 0

        # Create output
        output = IntegratedStrategyOutput(
            traditional_signals=signals,
            option_strategy=option_strategy,
            strategy_alternatives=strategy_alternatives,
            orders=orders,
            risk_checks=risk_checks,
            approved_orders=approved_orders,
            rejected_orders=rejected_orders,
            total_notional=total_notional,
            total_delta=total_delta,
            total_vega=total_vega,
            expected_pnl=0.0,  # Would calculate from pricing model
            max_loss=abs(total_notional) * 0.5  # Conservative estimate
        )

        self.execution_history.append(output)

        # Print summary
        print("\n" + "="*80)
        print("EXECUTION SUMMARY")
        print("="*80)
        print(f"  Strategy: {option_strategy.strategy_type.value}")
        print(f"  Total Orders: {len(orders)}")
        print(f"  Approved: {len(approved_orders)} ✓")
        print(f"  Rejected: {len(rejected_orders)} ✗")
        print(f"  Total Notional: ${total_notional:,.0f}")
        print(f"  Net Delta: {total_delta:.2f}")
        print(f"  Net Vega: {total_vega:.2f}")
        print("="*80)

        return output

    def _create_orders_from_strategy(self,
                                    strategy: StrategyAction,
                                    current_price: float,
                                    current_iv: float) -> List[OrderProposal]:
        """Create order proposals from strategy action"""
        if strategy.strategy_type == StrategyType.DO_NOTHING:
            return []

        orders = []
        for i in range(len(strategy.strikes)):
            order = OrderProposal(
                symbol='BTC',
                option_type=strategy.option_types[i],
                strike=strategy.strikes[i],
                expiry=strategy.expiries[i],
                quantity=strategy.quantities[i],
                direction=strategy.directions[i],
                underlying_price=current_price,
                volatility=current_iv,
                risk_free_rate=0.05
            )
            orders.append(order)

        return orders

    def _calculate_order_notional(self, order: OrderProposal, current_price: float) -> float:
        """Calculate notional value of order"""
        params = BSParameters(
            S0=order.underlying_price,
            K=order.strike,
            T=order.expiry,
            r=order.risk_free_rate,
            sigma=order.volatility
        )
        model = BlackScholesModel(params)

        if order.option_type.lower() == 'call':
            price = model.call_price()
        else:
            price = model.put_price()

        multiplier = order.quantity * (1 if order.direction == 'buy' else -1)
        return price * multiplier

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return {
            'total_signals': self.total_signals_generated,
            'total_strategies': self.total_strategies_created,
            'total_orders': self.total_orders_submitted,
            'approved_orders': self.total_orders_approved,
            'rejected_orders': self.total_orders_rejected,
            'approval_rate': (self.total_orders_approved / self.total_orders_submitted * 100
                            if self.total_orders_submitted > 0 else 0),
            'total_executions': len(self.execution_history)
        }


if __name__ == "__main__":
    from risk.bitcoin_risk_controller import BitcoinRiskLimits

    print("Integrated Strategy Manager Demo\n")

    # Setup risk controller
    limits = BitcoinRiskLimits(
        base_max_var=200000,
        base_max_cvar=300000,
        base_max_delta=150,
        base_max_gamma=50,
        base_max_vega=5000,
        base_max_theta=5000,
        dynamic_adjustment=True
    )

    risk_controller = BitcoinRiskController(
        risk_limits=limits,
        portfolio_value=1000000
    )

    # Initialize portfolio state
    np.random.seed(42)
    prices = [45000]
    for _ in range(100):
        prices.append(prices[-1] * (1 + np.random.normal(0.001, 0.03)))

    returns = np.diff(prices) / np.array(prices[:-1])

    risk_controller.update_portfolio_state(
        positions=[],
        greeks={'delta': 30, 'gamma': 10, 'vega': 300, 'theta': -400, 'rho': 20},
        returns=returns.tolist(),
        current_iv=0.75
    )

    # Create integrated manager
    manager = IntegratedStrategyManager(
        risk_controller=risk_controller,
        enable_dp_optimization=False,  # Can enable for full DP
        enable_risk_control=True
    )

    # Execute strategy
    result = manager.execute_integrated_strategy(
        prices=prices,
        current_price=prices[-1],
        current_iv=0.75,
        current_delta=30.0,
        time_horizon_days=30
    )

    # Performance summary
    print("\n")
    print("="*80)
    print("PERFORMANCE SUMMARY")
    print("="*80)
    perf = manager.get_performance_summary()
    for key, value in perf.items():
        if 'rate' in key:
            print(f"  {key}: {value:.1f}%")
        else:
            print(f"  {key}: {value}")
    print("="*80)
