"""
PnL Attribution Analysis
=======================

Comprehensive P&L breakdown for volatility arbitrage strategies.

This module provides detailed attribution of trading profits and losses:

Gamma P&L: 0.5 * gamma * (dS)^2 - convexity profits from price movements
Theta P&L: theta * dt - time decay profits
Vega P&L: vega * dσ - volatility change profits/losses
Delta P&L: delta * dS - directional exposure profits/losses
Trading Costs: commissions, slippage, market impact

The dual convergence framework enables superior PnL attribution
by providing more accurate Greeks and volatility forecasts.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum


class PnLComponent(Enum):
    GAMMA_PNL = "gamma_pnl"
    THETA_PNL = "theta_pnl"
    VEGA_PNL = "vega_pnl"
    DELTA_PNL = "delta_pnl"
    TRADING_COSTS = "trading_costs"
    MARKET_IMPACT = "market_impact"
    DIVIDEND_PNL = "dividend_pnl"
    CARRY_PNL = "carry_pnl"


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics for volatility strategies"""
    total_return: float = 0.0
    annualized_return: float = 0.0
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    calmar_ratio: float = 0.0
    sortino_ratio: float = 0.0

    # Strategy-specific metrics
    gamma_contribution: float = 0.0
    theta_contribution: float = 0.0
    vega_contribution: float = 0.0
    trading_cost_ratio: float = 0.0
    vol_edge_capture: float = 0.0


@dataclass
class PnLAttribution:
    """Detailed P&L attribution for a trading period"""
    timestamp: datetime
    period_start: datetime
    period_end: datetime

    # Core P&L components
    gamma_pnl: float = 0.0
    theta_pnl: float = 0.0
    vega_pnl: float = 0.0
    delta_pnl: float = 0.0

    # Costs and other
    trading_costs: float = 0.0
    market_impact: float = 0.0
    dividend_pnl: float = 0.0
    carry_pnl: float = 0.0

    # Totals
    total_pnl: float = 0.0
    unrealized_pnl: float = 0.0

    # Greeks at period end
    ending_delta: float = 0.0
    ending_gamma: float = 0.0
    ending_theta: float = 0.0
    ending_vega: float = 0.0

    # Market conditions
    avg_volatility: float = 0.0
    price_range: float = 0.0
    volume_traded: float = 0.0

    # Attribution details
    pnl_breakdown: Dict[str, float] = field(default_factory=dict)
    trade_details: List[Dict[str, Any]] = field(default_factory=list)


class PnLAttributionEngine:
    """
    Advanced P&L attribution engine for volatility arbitrage strategies.

    This engine provides the detailed profit/loss breakdown that quant traders
    need to understand what's working and what's not in their strategies.

    Core Attribution:
    - Gamma P&L: Convexity profits from price movements
    - Theta P&L: Time decay profits
    - Vega P&L: Volatility change profits
    - Delta P&L: Directional exposure
    - Trading Costs: Commissions and market impact
    """

    def __init__(self, attribution_frequency: str = 'daily'):
        self.attribution_frequency = attribution_frequency
        self.attribution_history: List[PnLAttribution] = []
        self.performance_metrics = PerformanceMetrics()

    def calculate_pnl_attribution(self,
                                position_data: Dict[str, Any],
                                price_history: pd.DataFrame,
                                trade_log: List[Dict[str, Any]],
                                period_start: datetime,
                                period_end: datetime) -> PnLAttribution:
        """
        Calculate detailed P&L attribution for a trading period.

        Parameters:
        -----------
        position_data : Dict[str, Any]
            Current position information (Greeks, sizes, etc.)
        price_history : pd.DataFrame
            Price movements during the period
        trade_log : List[Dict[str, Any]]
            All trades executed during the period
        period_start, period_end : datetime
            Attribution period

        Returns:
        --------
        PnLAttribution : Complete attribution analysis
        """

        # Calculate Gamma P&L
        gamma_pnl = self._calculate_gamma_pnl(position_data, price_history)

        # Calculate Theta P&L
        theta_pnl = self._calculate_theta_pnl(position_data, period_start, period_end)

        # Calculate Vega P&L
        vega_pnl = self._calculate_vega_pnl(position_data, price_history)

        # Calculate Delta P&L
        delta_pnl = self._calculate_delta_pnl(position_data, price_history)

        # Calculate Trading Costs
        trading_costs = self._calculate_trading_costs(trade_log)

        # Calculate Market Impact
        market_impact = self._calculate_market_impact(trade_log, price_history)

        # Other components
        dividend_pnl = 0.0  # Simplified
        carry_pnl = 0.0     # Simplified

        # Total P&L
        total_pnl = (gamma_pnl + theta_pnl + vega_pnl + delta_pnl -
                    trading_costs - market_impact + dividend_pnl + carry_pnl)

        # Unrealized P&L (mark-to-market)
        unrealized_pnl = self._calculate_unrealized_pnl(position_data, price_history)

        # Market conditions
        avg_volatility = price_history['Close'].pct_change().std() * np.sqrt(252)
        price_range = (price_history['High'].max() - price_history['Low'].min()) / price_history['Close'].iloc[0]
        volume_traded = price_history['Volume'].sum()

        # Detailed breakdown
        pnl_breakdown = {
            'gamma_pnl': gamma_pnl,
            'theta_pnl': theta_pnl,
            'vega_pnl': vega_pnl,
            'delta_pnl': delta_pnl,
            'trading_costs': -trading_costs,
            'market_impact': -market_impact,
            'dividend_pnl': dividend_pnl,
            'carry_pnl': carry_pnl,
            'total_pnl': total_pnl
        }

        attribution = PnLAttribution(
            timestamp=datetime.now(),
            period_start=period_start,
            period_end=period_end,
            gamma_pnl=gamma_pnl,
            theta_pnl=theta_pnl,
            vega_pnl=vega_pnl,
            delta_pnl=delta_pnl,
            trading_costs=trading_costs,
            market_impact=market_impact,
            dividend_pnl=dividend_pnl,
            carry_pnl=carry_pnl,
            total_pnl=total_pnl,
            unrealized_pnl=unrealized_pnl,
            ending_delta=position_data.get('total_delta', 0.0),
            ending_gamma=position_data.get('total_gamma', 0.0),
            ending_theta=position_data.get('total_theta', 0.0),
            ending_vega=position_data.get('total_vega', 0.0),
            avg_volatility=avg_volatility,
            price_range=price_range,
            volume_traded=volume_traded,
            pnl_breakdown=pnl_breakdown,
            trade_details=trade_log
        )

        self.attribution_history.append(attribution)
        self._update_performance_metrics(attribution)

        return attribution

    def generate_attribution_report(self,
                                  attribution: PnLAttribution,
                                  include_charts: bool = False) -> str:
        """Generate detailed P&L attribution report"""

        report = f"""
🔍 P&L Attribution Report
Period: {attribution.period_start.strftime('%Y-%m-%d')} to {attribution.period_end.strftime('%Y-%m-%d')}

📊 Core P&L Components:
   Gamma P&L: ${attribution.gamma_pnl:,.2f} ({attribution.gamma_pnl/attribution.total_pnl*100:.1f}%)
   Theta P&L: ${attribution.theta_pnl:,.2f} ({attribution.theta_pnl/attribution.total_pnl*100:.1f}%)
   Vega P&L:  ${attribution.vega_pnl:,.2f} ({attribution.vega_pnl/attribution.total_pnl*100:.1f}%)
   Delta P&L: ${attribution.delta_pnl:,.2f} ({attribution.delta_pnl/attribution.total_pnl*100:.1f}%)

💰 Costs & Other:
   Trading Costs: ${attribution.trading_costs:,.2f}
   Market Impact: ${attribution.market_impact:,.2f}
   Dividend P&L: ${attribution.dividend_pnl:,.2f}
   Carry P&L:    ${attribution.carry_pnl:,.2f}

📈 Total P&L: ${attribution.total_pnl:,.2f}
   Unrealized:  ${attribution.unrealized_pnl:,.2f}

🎯 Position Greeks (End of Period):
   Delta: {attribution.ending_delta:.4f}
   Gamma: {attribution.ending_gamma:.4f}
   Theta: {attribution.ending_theta:.4f}
   Vega:  {attribution.ending_vega:.4f}

🌊 Market Conditions:
   Avg Volatility: {attribution.avg_volatility:.1%}
   Price Range: {attribution.price_range:.1%}
   Volume Traded: {attribution.volume_traded:,.0f}
"""

        # Performance insights
        report += "\n💡 Performance Insights:\n"

        if attribution.gamma_pnl > 0:
            report += "   ✅ Gamma scalping profitable - good convexity capture\n"
        else:
            report += "   ⚠️  Gamma scalping losses - poor convexity capture\n"

        if attribution.theta_pnl > 0:
            report += "   ✅ Time decay working in your favor\n"
        else:
            report += "   ⚠️  Time decay working against you\n"

        if attribution.vega_pnl > 0:
            report += "   ✅ Volatility edge captured\n"
        else:
            report += "   ⚠️  Volatility edge not captured\n"

        cost_ratio = (attribution.trading_costs + attribution.market_impact) / abs(attribution.total_pnl) if attribution.total_pnl != 0 else 0
        if cost_ratio < 0.1:
            report += "   ✅ Trading costs well controlled\n"
        else:
            report += "   ⚠️  Trading costs too high\n"

        return report

    def get_performance_summary(self) -> str:
        """Generate performance summary report"""

        if not self.attribution_history:
            return "No attribution data available"

        total_pnl = sum(attr.total_pnl for attr in self.attribution_history)
        total_trading_costs = sum(attr.trading_costs for attr in self.attribution_history)

        # Calculate key ratios
        gamma_total = sum(attr.gamma_pnl for attr in self.attribution_history)
        theta_total = sum(attr.theta_pnl for attr in self.attribution_history)
        vega_total = sum(attr.vega_pnl for attr in self.attribution_history)

        report = f"""
📊 Performance Summary - {len(self.attribution_history)} Periods

💰 P&L Breakdown:
   Total P&L:     ${total_pnl:,.2f}
   Gamma P&L:     ${gamma_total:,.2f} ({gamma_total/total_pnl*100:.1f}%)
   Theta P&L:     ${theta_total:,.2f} ({theta_total/total_pnl*100:.1f}%)
   Vega P&L:      ${vega_total:,.2f} ({vega_total/total_pnl*100:.1f}%)
   Trading Costs: ${total_trading_costs:,.2f} ({total_trading_costs/total_pnl*100:.1f}%)

🎯 Strategy Effectiveness:
   Sharpe Ratio: {self.performance_metrics.sharpe_ratio:.2f}
   Win Rate: {self.performance_metrics.win_rate:.1%}
   Profit Factor: {self.performance_metrics.profit_factor:.2f}
   Max Drawdown: {self.performance_metrics.max_drawdown:.1%}

🔬 Strategy Attribution:
   Gamma Contribution: {self.performance_metrics.gamma_contribution:.1%}
   Theta Contribution: {self.performance_metrics.theta_contribution:.1%}
   Vega Contribution: {self.performance_metrics.vega_contribution:.1%}
   Trading Cost Ratio: {self.performance_metrics.trading_cost_ratio:.1%}
"""

        return report

    def _calculate_gamma_pnl(self, position_data: Dict[str, Any], price_history: pd.DataFrame) -> float:
        """Calculate gamma P&L from convexity"""

        total_gamma = position_data.get('total_gamma', 0.0)

        if total_gamma == 0 or price_history.empty:
            return 0.0

        # Calculate price changes
        price_changes = price_history['Close'].diff().dropna()

        # Gamma P&L = 0.5 * gamma * (dS)^2 for each price change
        gamma_pnl = 0.0
        for dS in price_changes:
            gamma_pnl += 0.5 * total_gamma * (dS ** 2)

        return gamma_pnl

    def _calculate_theta_pnl(self, position_data: Dict[str, Any],
                           period_start: datetime, period_end: datetime) -> float:
        """Calculate theta P&L from time decay"""

        total_theta = position_data.get('total_theta', 0.0)
        time_elapsed = (period_end - period_start).total_seconds() / (365.25 * 24 * 3600)  # Years

        # Theta P&L = theta * dt
        # Note: Theta is usually negative, so positive theta_pnl means time worked in your favor
        theta_pnl = total_theta * time_elapsed

        return theta_pnl

    def _calculate_vega_pnl(self, position_data: Dict[str, Any], price_history: pd.DataFrame) -> float:
        """Calculate vega P&L from volatility changes"""

        total_vega = position_data.get('total_vega', 0.0)

        if total_vega == 0 or price_history.empty:
            return 0.0

        # Calculate realized volatility changes
        returns = price_history['Close'].pct_change().dropna()
        realized_vol = returns.rolling(30).std() * np.sqrt(252)  # 30-day rolling vol

        vol_changes = realized_vol.diff().dropna()

        # Vega P&L = vega * dσ for each volatility change
        vega_pnl = total_vega * vol_changes.sum()

        return vega_pnl

    def _calculate_delta_pnl(self, position_data: Dict[str, Any], price_history: pd.DataFrame) -> float:
        """Calculate delta P&L from directional exposure"""

        # This is the P&L from unhedged delta exposure
        # In a properly hedged strategy, this should be close to zero

        total_delta = position_data.get('total_delta', 0.0)

        if total_delta == 0 or price_history.empty:
            return 0.0

        # Delta P&L = delta * dS
        price_change = price_history['Close'].iloc[-1] - price_history['Close'].iloc[0]
        delta_pnl = total_delta * price_change * 100  # 100 shares per contract

        return delta_pnl

    def _calculate_trading_costs(self, trade_log: List[Dict[str, Any]]) -> float:
        """Calculate total trading costs"""

        total_costs = 0.0

        for trade in trade_log:
            # Commission (simplified)
            notional = abs(trade.get('quantity', 0) * trade.get('price', 0))
            commission = max(notional * 0.0005, 1.0)  # 5bps or $1 minimum

            # Slippage (simplified)
            slippage = notional * 0.0002  # 2bps

            total_costs += commission + slippage

        return total_costs

    def _calculate_market_impact(self, trade_log: List[Dict[str, Any]], price_history: pd.DataFrame) -> float:
        """Calculate market impact costs"""

        if not trade_log or price_history.empty:
            return 0.0

        # Simplified market impact calculation
        avg_daily_volume = price_history['Volume'].mean()
        total_trade_size = sum(abs(trade.get('quantity', 0)) for trade in trade_log)

        # Market impact = 0.1 * sqrt(participation_rate)
        participation_rate = total_trade_size / avg_daily_volume
        market_impact = 0.1 * np.sqrt(participation_rate) * price_history['Close'].iloc[-1] * total_trade_size

        return market_impact

    def _calculate_unrealized_pnl(self, position_data: Dict[str, Any], price_history: pd.DataFrame) -> float:
        """Calculate unrealized P&L (mark-to-market)"""

        if price_history.empty:
            return 0.0

        current_price = price_history['Close'].iloc[-1]

        # Simplified unrealized P&L calculation
        # In practice, this would re-price all options at current market conditions
        unrealized_pnl = 0.0

        for position_name, position in position_data.items():
            if isinstance(position, dict) and 'unrealized_pnl' in position:
                unrealized_pnl += position['unrealized_pnl']

        return unrealized_pnl

    def _update_performance_metrics(self, attribution: PnLAttribution):
        """Update rolling performance metrics"""

        # Simple update - in practice, you'd maintain more sophisticated tracking
        pnl_values = [attr.total_pnl for attr in self.attribution_history[-30:]]  # Last 30 periods

        if pnl_values:
            self.performance_metrics.total_return = sum(pnl_values)
            self.performance_metrics.annualized_return = self.performance_metrics.total_return * 12 / len(pnl_values)
            self.performance_metrics.volatility = np.std(pnl_values)
            self.performance_metrics.sharpe_ratio = (self.performance_metrics.annualized_return /
                                                   self.performance_metrics.volatility) if self.performance_metrics.volatility > 0 else 0

            # Update component contributions
            gamma_total = sum(attr.gamma_pnl for attr in self.attribution_history[-30:])
            theta_total = sum(attr.theta_pnl for attr in self.attribution_history[-30:])
            vega_total = sum(attr.vega_pnl for attr in self.attribution_history[-30:])
            costs_total = sum(attr.trading_costs + attr.market_impact for attr in self.attribution_history[-30:])

            total_abs = abs(self.performance_metrics.total_return)
            if total_abs > 0:
                self.performance_metrics.gamma_contribution = gamma_total / total_abs
                self.performance_metrics.theta_contribution = theta_total / total_abs
                self.performance_metrics.vega_contribution = vega_total / total_abs
                self.performance_metrics.trading_cost_ratio = costs_total / total_abs

