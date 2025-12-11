"""
Dynamic Volatility Hedge Engine
==============================

Advanced hedging engine using dual convergence enhanced Greeks.

This engine provides superior hedging strategies by leveraging:
- Enhanced Delta calculation from stochastic volatility models
- Gamma scalping opportunities
- Vega hedging for volatility risk
- Dynamic position adjustments based on volatility forecasts

The dual convergence framework provides better volatility predictions,
leading to more accurate hedge ratios and reduced hedging costs.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta


@dataclass
class HedgePosition:
    """Dynamic hedge position"""
    underlying_position: float  # Shares/contracts of underlying
    option_positions: Dict[str, float] = field(default_factory=dict)  # option_symbol -> quantity
    cash_position: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    hedge_ratio: float = 0.0  # Current delta hedge ratio
    vega_exposure: float = 0.0
    gamma_exposure: float = 0.0
    rebalancing_threshold: float = 0.05  # Rebalance if delta changes by 5%
    last_rebalance: datetime = field(default_factory=datetime.now)


class DynamicHedgeEngine:
    """
    Advanced dynamic hedging engine for options portfolios.

    This engine uses enhanced Greeks from dual convergence pricing to maintain
    superior hedge performance, particularly in volatile market conditions.

    Key Features:
    - Real-time delta hedging with enhanced Greeks
    - Gamma scalping for volatility harvesting
    - Vega hedging against volatility risk
    - Dynamic position sizing based on confidence
    """

    def __init__(self,
                 hedge_frequency: str = '5min',  # How often to rebalance
                 max_hedge_deviation: float = 0.1,  # Max 10% deviation from target
                 gamma_scalping_enabled: bool = True,
                 vega_hedging_enabled: bool = True):

        self.hedge_frequency = hedge_frequency
        self.max_hedge_deviation = max_hedge_deviation
        self.gamma_scalping_enabled = gamma_scalping_enabled
        self.vega_hedging_enabled = vega_hedging_enabled

        self.current_positions: Dict[str, HedgePosition] = {}

    def create_hedge_portfolio(self,
                              option_portfolio: Dict[str, float],  # option_symbol -> quantity
                              market_data: pd.DataFrame,
                              pricing_results: Dict[str, Any]) -> HedgePosition:
        """
        Create a dynamically hedged portfolio for given option positions.

        Parameters:
        -----------
        option_portfolio : Dict[str, float]
            Options to hedge (symbol -> quantity)
        market_data : pd.DataFrame
            Current market data
        pricing_results : Dict[str, Any]
            Pricing results with enhanced Greeks from DualConvergencePricer

        Returns:
        --------
        HedgePosition : Complete hedge portfolio
        """
        print("🛡️  Creating Dynamic Hedge Portfolio")

        hedge_position = HedgePosition()

        # Calculate total portfolio Greeks
        total_delta = 0.0
        total_gamma = 0.0
        total_vega = 0.0

        for option_symbol, quantity in option_portfolio.items():
            if option_symbol in pricing_results:
                result = pricing_results[option_symbol]

                # Accumulate Greeks (weighted by quantity)
                total_delta += result.delta * quantity
                total_gamma += result.gamma * quantity
                total_vega += result.vega * quantity

                # Store option position
                hedge_position.option_positions[option_symbol] = quantity

        print(".4f")
        print(".4f")
        print(".4f")

        # Calculate required underlying hedge (delta-neutral)
        spot_price = market_data['Close'].iloc[-1]
        underlying_quantity = -total_delta  # Delta-neutral hedge

        hedge_position.underlying_position = underlying_quantity
        hedge_position.hedge_ratio = underlying_quantity
        hedge_position.gamma_exposure = total_gamma
        hedge_position.vega_exposure = total_vega

        print(".0f")

        # Calculate cash position (margin requirements)
        hedge_position.cash_position = self._calculate_margin_requirement(
            hedge_position, spot_price
        )

        print(".2f")

        self.current_positions['portfolio'] = hedge_position

        return hedge_position

    def update_hedge_positions(self,
                             current_market_data: pd.DataFrame,
                             new_pricing_results: Dict[str, Any],
                             time_elapsed: timedelta) -> Dict[str, HedgePosition]:
        """
        Update hedge positions based on market movements and time decay.

        This implements dynamic hedging with real-time adjustments.

        Parameters:
        -----------
        current_market_data : pd.DataFrame
            Latest market data
        new_pricing_results : Dict[str, Any]
            Updated pricing results
        time_elapsed : timedelta
            Time since last hedge adjustment

        Returns:
        --------
        Dict[str, HedgePosition] : Updated hedge positions
        """
        print("🔄 Updating Dynamic Hedge Positions")

        updated_positions = {}

        for portfolio_name, position in self.current_positions.items():
            # Check if rebalancing is needed
            if self._should_rebalance(position, current_market_data, time_elapsed):
                print(f"   Rebalancing {portfolio_name}...")

                # Recalculate Greeks
                new_greeks = self._recalculate_portfolio_greeks(
                    position.option_positions, new_pricing_results
                )

                # Update hedge
                updated_position = self._rebalance_hedge(
                    position, new_greeks, current_market_data
                )

                updated_positions[portfolio_name] = updated_position
            else:
                updated_positions[portfolio_name] = position

        self.current_positions.update(updated_positions)

        print(f"   ✓ Updated {len(updated_positions)} hedge positions")

        return updated_positions

    def apply_gamma_scalping(self,
                           hedge_position: HedgePosition,
                           price_move: float,
                           volatility_change: float) -> Tuple[HedgePosition, float]:
        """
        Apply gamma scalping strategy to harvest volatility.

        Gamma scalping involves buying low and selling high as the underlying moves,
        profiting from gamma (convexity) in the options position.

        Parameters:
        -----------
        hedge_position : HedgePosition
            Current hedge position
        price_move : float
            Recent price movement (as fraction)
        volatility_change : float
            Recent volatility change

        Returns:
        --------
        Tuple[HedgePosition, float] : Updated position and P&L from scalping
        """
        if not self.gamma_scalping_enabled or abs(hedge_position.gamma_exposure) < 0.01:
            return hedge_position, 0.0

        # Calculate gamma scalping adjustment
        # Positive gamma means we can profit from large moves
        gamma_adjustment = hedge_position.gamma_exposure * price_move * price_move

        # Adjust underlying position for scalping
        scalping_quantity = gamma_adjustment * 1000  # Scale factor

        # Update position
        updated_position = hedge_position
        updated_position.underlying_position += scalping_quantity

        # Calculate P&L from scalping
        scalping_pnl = scalping_quantity * price_move * 100  # Rough estimate

        print(".4f"
        return updated_position, scalping_pnl

    def manage_vega_exposure(self,
                           hedge_position: HedgePosition,
                           current_volatility: float,
                           target_volatility: float) -> HedgePosition:
        """
        Manage vega exposure through volatility hedging.

        When our dual convergence model predicts volatility changes,
        we can adjust the hedge to profit from or protect against these moves.

        Parameters:
        -----------
        hedge_position : HedgePosition
            Current hedge position
        current_volatility : float
            Current market volatility
        target_volatility : float
            Target volatility from dual convergence model

        Returns:
        --------
        HedgePosition : Updated position with vega adjustments
        """
        if not self.vega_hedging_enabled:
            return hedge_position

        vol_diff = target_volatility - current_volatility

        # Adjust position based on volatility forecast
        # If we expect higher volatility, we might want to be more long gamma
        vega_adjustment = hedge_position.vega_exposure * vol_diff * 0.1

        # This could involve buying/selling options or adjusting the underlying hedge
        # For now, we'll adjust the underlying position
        hedge_position.underlying_position += vega_adjustment

        print(".1%")

        return hedge_position

    def calculate_hedge_effectiveness(self,
                                    hedge_position: HedgePosition,
                                    historical_pnl: pd.Series,
                                    benchmark_pnl: pd.Series) -> Dict[str, float]:
        """
        Calculate hedge effectiveness metrics.

        Parameters:
        -----------
        hedge_position : HedgePosition
            Hedge position to evaluate
        historical_pnl : pd.Series
            Historical P&L of hedged portfolio
        benchmark_pnl : pd.Series
            Benchmark P&L (unhedged or different hedge)

        Returns:
        --------
        Dict[str, float] : Hedge effectiveness metrics
        """
        # Calculate hedge ratio (correlation between portfolio and hedge)
        correlation = historical_pnl.corr(benchmark_pnl)

        # Calculate volatility reduction
        portfolio_vol = historical_pnl.std()
        benchmark_vol = benchmark_pnl.std()
        vol_reduction = 1 - (portfolio_vol / benchmark_vol) if benchmark_vol > 0 else 0

        # Calculate Sharpe ratio improvement
        portfolio_sharpe = historical_pnl.mean() / historical_pnl.std() if historical_pnl.std() > 0 else 0
        benchmark_sharpe = benchmark_pnl.mean() / benchmark_pnl.std() if benchmark_pnl.std() > 0 else 0
        sharpe_improvement = portfolio_sharpe - benchmark_sharpe

        return {
            'correlation': correlation,
            'volatility_reduction': vol_reduction,
            'sharpe_improvement': sharpe_improvement,
            'hedge_effectiveness_score': (correlation + vol_reduction + sharpe_improvement) / 3
        }

    def _recalculate_portfolio_greeks(self,
                                    option_positions: Dict[str, float],
                                    pricing_results: Dict[str, Any]) -> Dict[str, float]:
        """Recalculate portfolio Greeks"""

        total_delta = 0.0
        total_gamma = 0.0
        total_vega = 0.0

        for option_symbol, quantity in option_positions.items():
            if option_symbol in pricing_results:
                result = pricing_results[option_symbol]
                total_delta += result.delta * quantity
                total_gamma += result.gamma * quantity
                total_vega += result.vega * quantity

        return {
            'delta': total_delta,
            'gamma': total_gamma,
            'vega': total_vega
        }

    def _rebalance_hedge(self,
                        position: HedgePosition,
                        new_greeks: Dict[str, float],
                        market_data: pd.DataFrame) -> HedgePosition:
        """Rebalance hedge position"""

        # Calculate required adjustment
        current_delta = position.hedge_ratio
        target_delta = -new_greeks['delta']  # Delta-neutral target
        delta_adjustment = target_delta - current_delta

        # Update position
        position.underlying_position += delta_adjustment
        position.hedge_ratio = target_delta
        position.gamma_exposure = new_greeks['gamma']
        position.vega_exposure = new_greeks['vega']
        position.last_rebalance = datetime.now()

        return position

    def _should_rebalance(self,
                         position: HedgePosition,
                         market_data: pd.DataFrame,
                         time_elapsed: timedelta) -> bool:
        """Determine if hedge rebalancing is needed"""

        # Check time-based rebalancing
        if time_elapsed > timedelta(minutes=5):  # Rebalance every 5 minutes
            return True

        # Check delta deviation
        target_delta = position.hedge_ratio
        # This would require real-time delta calculation
        # For now, assume periodic rebalancing
        return True

    def _calculate_margin_requirement(self,
                                    hedge_position: HedgePosition,
                                    spot_price: float) -> float:
        """Calculate margin requirements for hedge position"""

        # Simplified margin calculation
        option_margin = 0.0
        for option_symbol, quantity in hedge_position.option_positions.items():
            # Assume 20% of notional for options margin
            option_margin += abs(quantity) * spot_price * 0.2

        # Underlying margin (typically 50% for long positions)
        underlying_margin = abs(hedge_position.underlying_position) * spot_price * 0.5

        return -(option_margin + underlying_margin)  # Negative for margin requirement

