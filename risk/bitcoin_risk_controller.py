"""
Bitcoin Options Risk Controller

Specialized risk management for Bitcoin options with:
- Dynamic risk limits based on volatility regime
- VaR/CVaR/Delta threshold enforcement
- Pre-trade risk checks that reject orders exceeding limits
- Volatility-adjusted position sizing

Design Philosophy:
- Bitcoin's high volatility requires stricter tail risk management
- CVaR is more important than VaR due to fat tails
- Greeks limits should adapt to volatility regime
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import warnings

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from risk.var_models import HistoricalVaR, ParametricVaR, MonteCarloVaR
from risk.cvar_models import ExpectedShortfall
from models.options_pricing.black_scholes import BlackScholesModel, BSParameters


class VolatilityRegime(Enum):
    """Volatility regime classification"""
    LOW = "low"      # IV < 40%
    MEDIUM = "medium"  # IV 40-80%
    HIGH = "high"    # IV > 80%
    EXTREME = "extreme"  # IV > 120%


class RiskCheckStatus(Enum):
    """Status of risk check"""
    APPROVED = "approved"
    REJECTED = "rejected"
    WARNING = "warning"


@dataclass
class BitcoinRiskLimits:
    """
    Risk limits for Bitcoin options portfolio.
    Dynamically adjusted based on volatility regime.
    """
    # Base limits (at medium volatility)
    base_max_var: float = 100000  # $100k
    base_max_cvar: float = 150000  # $150k (more important for BTC)
    base_max_delta: float = 100
    base_max_gamma: float = 30
    base_max_vega: float = 1000
    base_max_theta: float = 2000

    # Position limits
    max_position_concentration: float = 0.20  # 20% max per position
    max_leverage: float = 2.0  # Conservative for crypto

    # Confidence levels
    var_confidence_level: float = 0.05  # 95% confidence
    cvar_confidence_level: float = 0.05

    # Volatility adjustment factors
    volatility_regime: VolatilityRegime = VolatilityRegime.MEDIUM
    current_iv: float = 0.60  # 60% implied volatility

    # Dynamic adjustment enabled
    dynamic_adjustment: bool = True

    def get_adjusted_limits(self) -> Dict[str, float]:
        """
        Get risk limits adjusted for current volatility regime.

        In high volatility: tighten limits
        In low volatility: relax limits
        """
        if not self.dynamic_adjustment:
            return {
                'max_var': self.base_max_var,
                'max_cvar': self.base_max_cvar,
                'max_delta': self.base_max_delta,
                'max_gamma': self.base_max_gamma,
                'max_vega': self.base_max_vega,
                'max_theta': self.base_max_theta
            }

        # Adjustment factors based on regime
        if self.volatility_regime == VolatilityRegime.LOW:
            var_mult = 1.3
            delta_mult = 1.5
            gamma_mult = 1.4
        elif self.volatility_regime == VolatilityRegime.MEDIUM:
            var_mult = 1.0
            delta_mult = 1.0
            gamma_mult = 1.0
        elif self.volatility_regime == VolatilityRegime.HIGH:
            var_mult = 0.7
            delta_mult = 0.6
            gamma_mult = 0.5
        else:  # EXTREME
            var_mult = 0.4
            delta_mult = 0.3
            gamma_mult = 0.3

        return {
            'max_var': self.base_max_var * var_mult,
            'max_cvar': self.base_max_cvar * var_mult,  # CVaR same adjustment as VaR
            'max_delta': self.base_max_delta * delta_mult,
            'max_gamma': self.base_max_gamma * gamma_mult,
            'max_vega': self.base_max_vega,  # Vega limit stays constant
            'max_theta': self.base_max_theta
        }

    def update_volatility_regime(self, current_iv: float):
        """Update volatility regime based on current implied volatility"""
        self.current_iv = current_iv

        if current_iv < 0.40:
            self.volatility_regime = VolatilityRegime.LOW
        elif current_iv < 0.80:
            self.volatility_regime = VolatilityRegime.MEDIUM
        elif current_iv < 1.20:
            self.volatility_regime = VolatilityRegime.HIGH
        else:
            self.volatility_regime = VolatilityRegime.EXTREME


@dataclass
class OrderProposal:
    """Proposed order for risk checking"""
    symbol: str
    option_type: str  # 'call' or 'put'
    strike: float
    expiry: float  # Time to expiry in years
    quantity: int
    direction: str  # 'buy' or 'sell'
    underlying_price: float
    volatility: float
    risk_free_rate: float = 0.05
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class RiskCheckResult:
    """Result of pre-trade risk check"""
    status: RiskCheckStatus
    approved: bool
    reasons: List[str]
    risk_metrics: Dict[str, float]
    proposed_metrics: Dict[str, float]
    limit_utilization: Dict[str, float]
    volatility_regime: str
    timestamp: datetime = field(default_factory=datetime.now)


class BitcoinRiskController:
    """
    Risk controller specialized for Bitcoin options.

    Key Features:
    - Dynamic risk limits based on volatility
    - Strict CVaR enforcement (tail risk critical for BTC)
    - Pre-trade checks reject violating orders
    - Real-time Greeks monitoring
    """

    def __init__(self,
                 risk_limits: BitcoinRiskLimits,
                 portfolio_value: float = 1000000,
                 var_method: str = 'historical',
                 cvar_method: str = 'historical'):
        """
        Initialize Bitcoin risk controller.

        Parameters:
        -----------
        risk_limits : BitcoinRiskLimits
            Risk limit configuration
        portfolio_value : float
            Current portfolio value
        var_method : str
            VaR calculation method
        cvar_method : str
            CVaR calculation method
        """
        self.risk_limits = risk_limits
        self.portfolio_value = portfolio_value
        self.var_method = var_method
        self.cvar_method = cvar_method

        # Initialize risk calculators
        confidence_var = risk_limits.var_confidence_level
        confidence_cvar = risk_limits.cvar_confidence_level

        if var_method == 'historical':
            self.var_calculator = HistoricalVaR(confidence_level=confidence_var)
        elif var_method == 'parametric':
            self.var_calculator = ParametricVaR(confidence_level=confidence_var)
        elif var_method == 'monte_carlo':
            self.var_calculator = MonteCarloVaR(confidence_level=confidence_var, n_simulations=10000)
        else:
            raise ValueError(f"Unknown VaR method: {var_method}")

        self.cvar_calculator = ExpectedShortfall(confidence_level=confidence_cvar, method=cvar_method)

        # Portfolio state
        self.current_positions: List[Dict] = []
        self.current_greeks: Dict[str, float] = {
            'delta': 0.0,
            'gamma': 0.0,
            'vega': 0.0,
            'theta': 0.0,
            'rho': 0.0
        }
        self.historical_returns: List[float] = []

        # Risk metrics cache
        self.current_var: Optional[float] = None
        self.current_cvar: Optional[float] = None

        # Risk check history
        self.check_history: List[RiskCheckResult] = []

        # Statistics
        self.total_checks = 0
        self.total_rejections = 0
        self.rejection_reasons: Dict[str, int] = {}

    def update_portfolio_state(self,
                              positions: List[Dict],
                              greeks: Dict[str, float],
                              returns: List[float],
                              current_iv: Optional[float] = None):
        """
        Update current portfolio state.

        Parameters:
        -----------
        positions : list of dict
            Current positions
        greeks : dict
            Current portfolio Greeks
        returns : list
            Historical returns for risk calculation
        current_iv : float, optional
            Current implied volatility for regime detection
        """
        self.current_positions = positions.copy()
        self.current_greeks = greeks.copy()
        self.historical_returns = returns.copy()

        # Update volatility regime if IV provided
        if current_iv is not None:
            self.risk_limits.update_volatility_regime(current_iv)

        # Recalculate VaR/CVaR
        if len(self.historical_returns) > 0:
            returns_array = np.array(self.historical_returns)
            self.current_var = self.var_calculator.calculate_var(returns_array, self.portfolio_value)
            self.current_cvar = self.cvar_calculator.calculate_cvar(returns_array, self.portfolio_value)

    def check_order(self, order: OrderProposal) -> RiskCheckResult:
        """
        Perform comprehensive pre-trade risk check.
        REJECTS orders that would breach risk limits.

        Parameters:
        -----------
        order : OrderProposal
            Proposed order

        Returns:
        --------
        RiskCheckResult
            Detailed risk check result with approval/rejection
        """
        self.total_checks += 1
        reasons = []
        status = RiskCheckStatus.APPROVED

        # Get current adjusted limits
        adjusted_limits = self.risk_limits.get_adjusted_limits()

        # Calculate order Greeks
        order_greeks = self._calculate_order_greeks(order)
        order_value = self._calculate_order_value(order)

        # Calculate proposed portfolio state
        proposed_greeks = {
            key: self.current_greeks[key] + order_greeks[key]
            for key in self.current_greeks.keys()
        }

        # Calculate limit utilization
        utilization = {}

        # === RISK CHECK 1: Delta Limit ===
        proposed_delta_abs = abs(proposed_greeks['delta'])
        if adjusted_limits['max_delta'] is not None:
            utilization['delta'] = proposed_delta_abs / adjusted_limits['max_delta']

            if proposed_delta_abs > adjusted_limits['max_delta']:
                reason = (f"❌ DELTA LIMIT EXCEEDED: {proposed_delta_abs:.2f} > "
                         f"{adjusted_limits['max_delta']:.2f} "
                         f"(regime: {self.risk_limits.volatility_regime.value})")
                reasons.append(reason)
                status = RiskCheckStatus.REJECTED
                self._record_rejection("delta")

        # === RISK CHECK 2: Gamma Limit ===
        proposed_gamma_abs = abs(proposed_greeks['gamma'])
        if adjusted_limits['max_gamma'] is not None:
            utilization['gamma'] = proposed_gamma_abs / adjusted_limits['max_gamma']

            if proposed_gamma_abs > adjusted_limits['max_gamma']:
                reason = (f"❌ GAMMA LIMIT EXCEEDED: {proposed_gamma_abs:.2f} > "
                         f"{adjusted_limits['max_gamma']:.2f}")
                reasons.append(reason)
                status = RiskCheckStatus.REJECTED
                self._record_rejection("gamma")

        # === RISK CHECK 3: Vega Limit ===
        proposed_vega_abs = abs(proposed_greeks['vega'])
        if adjusted_limits['max_vega'] is not None:
            utilization['vega'] = proposed_vega_abs / adjusted_limits['max_vega']

            if proposed_vega_abs > adjusted_limits['max_vega']:
                reason = f"❌ VEGA LIMIT EXCEEDED: {proposed_vega_abs:.2f} > {adjusted_limits['max_vega']:.2f}"
                reasons.append(reason)
                status = RiskCheckStatus.REJECTED
                self._record_rejection("vega")

        # === RISK CHECK 4: Theta Limit ===
        proposed_theta_abs = abs(proposed_greeks['theta'])
        if adjusted_limits['max_theta'] is not None:
            utilization['theta'] = proposed_theta_abs / adjusted_limits['max_theta']

            if proposed_theta_abs > adjusted_limits['max_theta']:
                reason = f"❌ THETA LIMIT EXCEEDED: ${proposed_theta_abs:.2f}/day > ${adjusted_limits['max_theta']:.2f}/day"
                reasons.append(reason)
                status = RiskCheckStatus.REJECTED
                self._record_rejection("theta")

        # === RISK CHECK 5: VaR Limit ===
        if adjusted_limits['max_var'] is not None and self.current_var is not None:
            # Estimate proposed VaR using delta approximation
            delta_change_pct = abs(order_greeks['delta'] / (self.portfolio_value / order.underlying_price))
            proposed_var = self.current_var * (1 + delta_change_pct)

            utilization['var'] = proposed_var / adjusted_limits['max_var']

            if proposed_var > adjusted_limits['max_var']:
                reason = (f"❌ VaR LIMIT EXCEEDED: ${proposed_var:,.0f} > "
                         f"${adjusted_limits['max_var']:,.0f} (95% confidence)")
                reasons.append(reason)
                status = RiskCheckStatus.REJECTED
                self._record_rejection("var")

        # === RISK CHECK 6: CVaR Limit (CRITICAL FOR BITCOIN) ===
        if adjusted_limits['max_cvar'] is not None and self.current_cvar is not None:
            # Estimate proposed CVaR
            delta_change_pct = abs(order_greeks['delta'] / (self.portfolio_value / order.underlying_price))
            proposed_cvar = self.current_cvar * (1 + delta_change_pct)

            utilization['cvar'] = proposed_cvar / adjusted_limits['max_cvar']

            if proposed_cvar > adjusted_limits['max_cvar']:
                reason = (f"❌ CVaR LIMIT EXCEEDED: ${proposed_cvar:,.0f} > "
                         f"${adjusted_limits['max_cvar']:,.0f} (Expected Shortfall)")
                reasons.append(reason)
                status = RiskCheckStatus.REJECTED
                self._record_rejection("cvar")

        # === RISK CHECK 7: Position Concentration ===
        if self.risk_limits.max_position_concentration is not None and self.portfolio_value > 0:
            concentration = abs(order_value) / self.portfolio_value
            utilization['concentration'] = concentration / self.risk_limits.max_position_concentration

            if concentration > self.risk_limits.max_position_concentration:
                reason = (f"❌ POSITION CONCENTRATION EXCEEDED: {concentration:.1%} > "
                         f"{self.risk_limits.max_position_concentration:.1%}")
                reasons.append(reason)
                status = RiskCheckStatus.REJECTED
                self._record_rejection("concentration")

        # === RISK CHECK 8: High Volatility Warning ===
        if self.risk_limits.volatility_regime in [VolatilityRegime.HIGH, VolatilityRegime.EXTREME]:
            if status == RiskCheckStatus.APPROVED:
                status = RiskCheckStatus.WARNING
            reasons.append(f"⚠️  WARNING: {self.risk_limits.volatility_regime.value.upper()} volatility regime "
                          f"(IV={self.risk_limits.current_iv:.1%})")

        # Success message if approved
        if status == RiskCheckStatus.APPROVED:
            reasons.append("✓ All risk checks passed")

        # Create result
        result = RiskCheckResult(
            status=status,
            approved=(status == RiskCheckStatus.APPROVED or status == RiskCheckStatus.WARNING),
            reasons=reasons,
            risk_metrics={
                'current_delta': self.current_greeks['delta'],
                'current_gamma': self.current_greeks['gamma'],
                'current_vega': self.current_greeks['vega'],
                'current_theta': self.current_greeks['theta'],
                'current_var': self.current_var or 0,
                'current_cvar': self.current_cvar or 0,
            },
            proposed_metrics={
                'proposed_delta': proposed_greeks['delta'],
                'proposed_gamma': proposed_greeks['gamma'],
                'proposed_vega': proposed_greeks['vega'],
                'proposed_theta': proposed_greeks['theta'],
                'order_delta': order_greeks['delta'],
                'order_gamma': order_greeks['gamma'],
                'order_vega': order_greeks['vega'],
                'order_theta': order_greeks['theta'],
                'order_value': order_value,
            },
            limit_utilization=utilization,
            volatility_regime=self.risk_limits.volatility_regime.value,
            timestamp=datetime.now()
        )

        self.check_history.append(result)

        if status == RiskCheckStatus.REJECTED:
            self.total_rejections += 1

        return result

    def get_max_order_size(self,
                          order_template: OrderProposal,
                          risk_metric: str = 'delta') -> int:
        """
        Calculate maximum order size within risk limits.

        Parameters:
        -----------
        order_template : OrderProposal
            Template order with quantity=1
        risk_metric : str
            Constraining metric ('delta', 'gamma', 'vega', 'var', 'cvar', 'all')

        Returns:
        --------
        int
            Maximum allowed quantity
        """
        adjusted_limits = self.risk_limits.get_adjusted_limits()

        # Calculate single contract Greeks
        test_order = order_template
        test_order.quantity = 1
        unit_greeks = self._calculate_order_greeks(test_order)

        max_quantities = []

        # Check each constraint
        if risk_metric in ['delta', 'all'] and adjusted_limits['max_delta'] is not None:
            available = adjusted_limits['max_delta'] - abs(self.current_greeks['delta'])
            if abs(unit_greeks['delta']) > 1e-6:
                max_qty = int(available / abs(unit_greeks['delta']))
                max_quantities.append(max(0, max_qty))

        if risk_metric in ['gamma', 'all'] and adjusted_limits['max_gamma'] is not None:
            available = adjusted_limits['max_gamma'] - abs(self.current_greeks['gamma'])
            if abs(unit_greeks['gamma']) > 1e-6:
                max_qty = int(available / abs(unit_greeks['gamma']))
                max_quantities.append(max(0, max_qty))

        if risk_metric in ['vega', 'all'] and adjusted_limits['max_vega'] is not None:
            available = adjusted_limits['max_vega'] - abs(self.current_greeks['vega'])
            if abs(unit_greeks['vega']) > 1e-6:
                max_qty = int(available / abs(unit_greeks['vega']))
                max_quantities.append(max(0, max_qty))

        if risk_metric in ['concentration', 'all']:
            unit_value = abs(self._calculate_order_value(test_order))
            max_position_value = self.portfolio_value * self.risk_limits.max_position_concentration
            if unit_value > 1e-6:
                max_qty = int(max_position_value / unit_value)
                max_quantities.append(max(0, max_qty))

        return min(max_quantities) if max_quantities else 0

    def _calculate_order_greeks(self, order: OrderProposal) -> Dict[str, float]:
        """Calculate Greeks for proposed order"""
        params = BSParameters(
            S0=order.underlying_price,
            K=order.strike,
            T=order.expiry,
            r=order.risk_free_rate,
            sigma=order.volatility
        )
        model = BlackScholesModel(params)

        # Get Greeks based on option type
        if order.option_type.lower() == 'call':
            delta = model.call_delta()
            theta = model.call_theta()
            rho = model.call_rho()
        else:
            delta = model.put_delta()
            theta = model.put_theta()
            rho = model.put_rho()

        # Gamma and Vega are the same for calls and puts
        gamma = model.gamma()
        vega = model.vega()

        # Adjust for quantity and direction
        multiplier = order.quantity * (1 if order.direction == 'buy' else -1)

        return {
            'delta': delta * multiplier,
            'gamma': gamma * multiplier,
            'vega': vega * multiplier,
            'theta': theta * multiplier,
            'rho': rho * multiplier
        }

    def _calculate_order_value(self, order: OrderProposal) -> float:
        """Calculate theoretical value of order"""
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

    def _record_rejection(self, reason: str):
        """Record rejection reason for statistics"""
        if reason not in self.rejection_reasons:
            self.rejection_reasons[reason] = 0
        self.rejection_reasons[reason] += 1

    def generate_risk_report(self) -> Dict[str, Any]:
        """Generate comprehensive risk report"""
        adjusted_limits = self.risk_limits.get_adjusted_limits()

        report = {
            'timestamp': datetime.now().isoformat(),
            'portfolio_value': self.portfolio_value,
            'volatility_regime': self.risk_limits.volatility_regime.value,
            'current_iv': self.risk_limits.current_iv,
            'current_greeks': self.current_greeks.copy(),
            'current_var': self.current_var,
            'current_cvar': self.current_cvar,
            'adjusted_limits': adjusted_limits,
            'base_limits': {
                'max_var': self.risk_limits.base_max_var,
                'max_cvar': self.risk_limits.base_max_cvar,
                'max_delta': self.risk_limits.base_max_delta,
                'max_gamma': self.risk_limits.base_max_gamma,
                'max_vega': self.risk_limits.base_max_vega,
            },
            'utilization': {},
            'statistics': {
                'total_checks': self.total_checks,
                'total_rejections': self.total_rejections,
                'rejection_rate': self.total_rejections / self.total_checks if self.total_checks > 0 else 0,
                'rejection_reasons': self.rejection_reasons.copy()
            }
        }

        # Calculate utilization
        if adjusted_limits['max_delta'] is not None:
            report['utilization']['delta'] = abs(self.current_greeks['delta']) / adjusted_limits['max_delta']

        if adjusted_limits['max_gamma'] is not None:
            report['utilization']['gamma'] = abs(self.current_greeks['gamma']) / adjusted_limits['max_gamma']

        if adjusted_limits['max_vega'] is not None:
            report['utilization']['vega'] = abs(self.current_greeks['vega']) / adjusted_limits['max_vega']

        if self.current_var and adjusted_limits['max_var']:
            report['utilization']['var'] = self.current_var / adjusted_limits['max_var']

        if self.current_cvar and adjusted_limits['max_cvar']:
            report['utilization']['cvar'] = self.current_cvar / adjusted_limits['max_cvar']

        return report

    def get_risk_status_summary(self) -> str:
        """Get human-readable risk status summary"""
        report = self.generate_risk_report()

        lines = []
        lines.append("=" * 70)
        lines.append("BITCOIN OPTIONS RISK CONTROLLER - STATUS REPORT")
        lines.append("=" * 70)
        lines.append(f"Portfolio Value: ${report['portfolio_value']:,.2f}")
        lines.append(f"Volatility Regime: {report['volatility_regime'].upper()} (IV={report['current_iv']:.1%})")
        lines.append("")
        lines.append("Current Greeks:")
        lines.append(f"  Delta: {report['current_greeks']['delta']:>10.2f}")
        lines.append(f"  Gamma: {report['current_greeks']['gamma']:>10.2f}")
        lines.append(f"  Vega:  {report['current_greeks']['vega']:>10.2f}")
        lines.append(f"  Theta: {report['current_greeks']['theta']:>10.2f}")
        lines.append("")
        lines.append("Risk Metrics:")
        if report['current_var']:
            lines.append(f"  VaR (95%):  ${report['current_var']:>10,.0f}")
        if report['current_cvar']:
            lines.append(f"  CVaR (95%): ${report['current_cvar']:>10,.0f}")
        lines.append("")
        lines.append("Risk Limit Utilization:")
        for metric, util in report['utilization'].items():
            bar_length = int(util * 20)
            bar = "█" * bar_length + "░" * (20 - bar_length)
            status = "🔴" if util > 0.9 else "🟡" if util > 0.7 else "🟢"
            lines.append(f"  {metric.upper():<8} {status} [{bar}] {util:>6.1%}")
        lines.append("")
        lines.append(f"Total Checks: {report['statistics']['total_checks']}")
        lines.append(f"Total Rejections: {report['statistics']['total_rejections']} "
                    f"({report['statistics']['rejection_rate']:.1%})")
        lines.append("=" * 70)

        return "\n".join(lines)


if __name__ == "__main__":
    print("Bitcoin Options Risk Controller - Demo\n")

    # Initialize with Bitcoin-specific limits
    limits = BitcoinRiskLimits(
        base_max_var=100000,
        base_max_cvar=150000,
        base_max_delta=100,
        base_max_gamma=30,
        base_max_vega=1000,
        base_max_theta=2000,
        max_position_concentration=0.20,
        dynamic_adjustment=True
    )

    controller = BitcoinRiskController(
        risk_limits=limits,
        portfolio_value=1000000,
        var_method='historical',
        cvar_method='historical'
    )

    # Simulate portfolio state
    current_greeks = {
        'delta': 60.0,
        'gamma': 15.0,
        'vega': 600.0,
        'theta': -800.0,
        'rho': 50.0
    }

    # Generate realistic Bitcoin returns (high volatility)
    np.random.seed(42)
    returns = np.random.normal(0.002, 0.05, 250)  # 50% annualized vol

    controller.update_portfolio_state(
        positions=[],
        greeks=current_greeks,
        returns=returns.tolist(),
        current_iv=0.75  # 75% IV - high but not extreme for BTC
    )

    print(controller.get_risk_status_summary())
    print("\n")

    # Test Order 1: Moderate position (should approve)
    print("=" * 70)
    print("TEST 1: Moderate Call Purchase")
    print("=" * 70)

    order1 = OrderProposal(
        symbol='BTC',
        option_type='call',
        strike=45000,
        expiry=0.25,
        quantity=5,
        direction='buy',
        underlying_price=45000,
        volatility=0.75,
        risk_free_rate=0.05
    )

    result1 = controller.check_order(order1)
    print(f"\nStatus: {result1.status.value.upper()}")
    print(f"Approved: {'✓ YES' if result1.approved else '✗ NO'}")
    print(f"\nReasons:")
    for reason in result1.reasons:
        print(f"  {reason}")
    print(f"\nOrder Impact:")
    print(f"  Delta: {result1.risk_metrics['current_delta']:.2f} → {result1.proposed_metrics['proposed_delta']:.2f}")
    print(f"  Gamma: {result1.risk_metrics['current_gamma']:.2f} → {result1.proposed_metrics['proposed_gamma']:.2f}")

    # Test Order 2: Large position (should reject)
    print("\n" + "=" * 70)
    print("TEST 2: Large Position (Should Reject)")
    print("=" * 70)

    order2 = OrderProposal(
        symbol='BTC',
        option_type='call',
        strike=45000,
        expiry=0.25,
        quantity=50,  # Very large
        direction='buy',
        underlying_price=45000,
        volatility=0.75,
        risk_free_rate=0.05
    )

    result2 = controller.check_order(order2)
    print(f"\nStatus: {result2.status.value.upper()}")
    print(f"Approved: {'✓ YES' if result2.approved else '✗ NO'}")
    print(f"\nReasons:")
    for reason in result2.reasons:
        print(f"  {reason}")

    # Test Order 3: During extreme volatility
    print("\n" + "=" * 70)
    print("TEST 3: Order During EXTREME Volatility Regime")
    print("=" * 70)

    controller.update_portfolio_state(
        positions=[],
        greeks=current_greeks,
        returns=returns.tolist(),
        current_iv=1.50  # 150% IV - extreme for BTC
    )

    order3 = OrderProposal(
        symbol='BTC',
        option_type='put',
        strike=40000,
        expiry=0.083,  # 1 month
        quantity=3,
        direction='buy',
        underlying_price=45000,
        volatility=1.50,
        risk_free_rate=0.05
    )

    result3 = controller.check_order(order3)
    print(f"\nStatus: {result3.status.value.upper()}")
    print(f"Approved: {'✓ YES' if result3.approved else '✗ NO'}")
    print(f"\nAdjusted Limits (EXTREME regime):")
    adj_limits = limits.get_adjusted_limits()
    print(f"  Max Delta: {adj_limits['max_delta']:.0f} (down from {limits.base_max_delta:.0f})")
    print(f"  Max Gamma: {adj_limits['max_gamma']:.0f} (down from {limits.base_max_gamma:.0f})")
    print(f"\nReasons:")
    for reason in result3.reasons:
        print(f"  {reason}")

    # Max order size calculation
    print("\n" + "=" * 70)
    print("MAXIMUM ORDER SIZE CALCULATION")
    print("=" * 70)

    template = OrderProposal(
        symbol='BTC',
        option_type='call',
        strike=45000,
        expiry=0.25,
        quantity=1,
        direction='buy',
        underlying_price=45000,
        volatility=1.50,
        risk_free_rate=0.05
    )

    max_size = controller.get_max_order_size(template, risk_metric='all')
    print(f"\nMaximum quantity under EXTREME volatility: {max_size} contracts")

    print("\n" + "=" * 70)
