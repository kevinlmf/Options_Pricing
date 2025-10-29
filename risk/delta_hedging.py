"""
Delta Hedging and Dynamic Risk Management Module

Implements delta-neutral hedging strategies for option portfolios
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HedgeFrequency(Enum):
    """Hedging frequency options"""
    CONTINUOUS = "continuous"  # Theoretical continuous hedging
    DAILY = "daily"
    HOURLY = "hourly"
    TICK = "tick"  # Hedge on every price change


@dataclass
class Position:
    """Option or underlying position"""
    instrument_id: str
    position_type: str  # 'option', 'underlying'
    quantity: float
    entry_price: float
    current_price: float
    delta: float = 0.0
    gamma: float = 0.0
    vega: float = 0.0
    theta: float = 0.0


@dataclass
class HedgingTransaction:
    """Record of a hedging transaction"""
    timestamp: pd.Timestamp
    underlying_quantity: float  # Positive = buy, negative = sell
    underlying_price: float
    transaction_cost: float
    portfolio_delta_before: float
    portfolio_delta_after: float
    reason: str


class DeltaHedger:
    """
    Delta Hedging Engine

    Maintains delta-neutral positions through dynamic hedging
    """

    def __init__(self,
                 transaction_cost: float = 0.0005,  # 5 bps
                 hedge_threshold: float = 0.1,      # Rehedge if |delta| > 0.1
                 hedge_frequency: HedgeFrequency = HedgeFrequency.DAILY):
        """
        Initialize Delta Hedger

        Parameters:
        -----------
        transaction_cost : float
            Transaction cost as fraction of trade value
        hedge_threshold : float
            Delta threshold to trigger rehedging
        hedge_frequency : HedgeFrequency
            How often to check/rehedge
        """
        self.transaction_cost = transaction_cost
        self.hedge_threshold = hedge_threshold
        self.hedge_frequency = hedge_frequency

        self.option_positions: List[Position] = []
        self.underlying_position: float = 0.0  # Net underlying position
        self.hedge_history: List[HedgingTransaction] = []

        self.cumulative_hedge_cost = 0.0
        self.total_pnl = 0.0

    def add_option_position(self, position: Position):
        """Add an option position to the portfolio"""
        self.option_positions.append(position)
        logger.info(f"Added position: {position.instrument_id}, Delta: {position.delta:.4f}")

    def get_portfolio_delta(self) -> float:
        """
        Calculate total portfolio delta

        Returns:
        --------
        float : Net portfolio delta
        """
        option_delta = sum(pos.quantity * pos.delta for pos in self.option_positions)
        total_delta = option_delta + self.underlying_position

        return total_delta

    def get_portfolio_greeks(self) -> Dict[str, float]:
        """
        Calculate all portfolio Greeks

        Returns:
        --------
        dict : Portfolio Greeks
        """
        greeks = {
            'delta': 0.0,
            'gamma': 0.0,
            'vega': 0.0,
            'theta': 0.0
        }

        for pos in self.option_positions:
            greeks['delta'] += pos.quantity * pos.delta
            greeks['gamma'] += pos.quantity * pos.gamma
            greeks['vega'] += pos.quantity * pos.vega
            greeks['theta'] += pos.quantity * pos.theta

        # Add underlying delta
        greeks['delta'] += self.underlying_position

        return greeks

    def calculate_hedge_quantity(self) -> float:
        """
        Calculate required hedge quantity to achieve delta neutrality

        Returns:
        --------
        float : Quantity of underlying to trade (positive = buy, negative = sell)
        """
        current_delta = self.get_portfolio_delta()

        # To neutralize delta, we need to offset it with underlying
        # Underlying has delta = 1.0 per unit
        hedge_quantity = -current_delta

        return hedge_quantity

    def execute_hedge(self,
                     timestamp: pd.Timestamp,
                     underlying_price: float,
                     force: bool = False) -> Optional[HedgingTransaction]:
        """
        Execute delta hedge if needed

        Parameters:
        -----------
        timestamp : pd.Timestamp
            Current timestamp
        underlying_price : float
            Current underlying price
        force : bool
            Force hedging regardless of threshold

        Returns:
        --------
        HedgingTransaction or None : Hedge transaction if executed
        """
        portfolio_delta_before = self.get_portfolio_delta()

        # Check if hedging is needed
        if not force and abs(portfolio_delta_before) < self.hedge_threshold:
            return None

        # Calculate hedge quantity
        hedge_quantity = self.calculate_hedge_quantity()

        if abs(hedge_quantity) < 1e-6:
            return None  # No hedging needed

        # Calculate transaction cost
        trade_value = abs(hedge_quantity * underlying_price)
        cost = trade_value * self.transaction_cost

        # Execute hedge
        self.underlying_position += hedge_quantity
        self.cumulative_hedge_cost += cost

        portfolio_delta_after = self.get_portfolio_delta()

        # Record transaction
        transaction = HedgingTransaction(
            timestamp=timestamp,
            underlying_quantity=hedge_quantity,
            underlying_price=underlying_price,
            transaction_cost=cost,
            portfolio_delta_before=portfolio_delta_before,
            portfolio_delta_after=portfolio_delta_after,
            reason="delta_rebalance"
        )

        self.hedge_history.append(transaction)

        logger.info(
            f"Hedge executed: {hedge_quantity:.4f} @ {underlying_price:.2f}, "
            f"Delta: {portfolio_delta_before:.4f} -> {portfolio_delta_after:.4f}, "
            f"Cost: ${cost:.2f}"
        )

        return transaction

    def update_positions(self,
                        new_prices: Dict[str, float],
                        new_greeks: Dict[str, Dict[str, float]]):
        """
        Update all position prices and Greeks

        Parameters:
        -----------
        new_prices : dict
            Dictionary mapping instrument_id to new price
        new_greeks : dict
            Dictionary mapping instrument_id to Greeks dict
        """
        for pos in self.option_positions:
            if pos.instrument_id in new_prices:
                pos.current_price = new_prices[pos.instrument_id]

            if pos.instrument_id in new_greeks:
                greeks = new_greeks[pos.instrument_id]
                pos.delta = greeks.get('delta', pos.delta)
                pos.gamma = greeks.get('gamma', pos.gamma)
                pos.vega = greeks.get('vega', pos.vega)
                pos.theta = greeks.get('theta', pos.theta)

    def calculate_pnl(self, underlying_price: float) -> Dict[str, float]:
        """
        Calculate portfolio P&L

        Parameters:
        -----------
        underlying_price : float
            Current underlying price

        Returns:
        --------
        dict : P&L breakdown
        """
        # Option P&L
        option_pnl = sum(
            pos.quantity * (pos.current_price - pos.entry_price)
            for pos in self.option_positions
        )

        # Underlying P&L (mark-to-market)
        # Need to track entry prices of underlying positions
        underlying_pnl = 0.0  # Simplified - would need entry price tracking

        # Total P&L
        gross_pnl = option_pnl + underlying_pnl
        net_pnl = gross_pnl - self.cumulative_hedge_cost

        return {
            'option_pnl': option_pnl,
            'underlying_pnl': underlying_pnl,
            'gross_pnl': gross_pnl,
            'hedge_costs': self.cumulative_hedge_cost,
            'net_pnl': net_pnl
        }

    def get_hedging_statistics(self) -> Dict:
        """
        Get statistics about hedging performance

        Returns:
        --------
        dict : Hedging statistics
        """
        if not self.hedge_history:
            return {
                'num_hedges': 0,
                'total_cost': 0.0,
                'avg_cost_per_hedge': 0.0,
                'total_volume_traded': 0.0
            }

        num_hedges = len(self.hedge_history)
        total_cost = sum(h.transaction_cost for h in self.hedge_history)
        total_volume = sum(abs(h.underlying_quantity * h.underlying_price)
                          for h in self.hedge_history)

        return {
            'num_hedges': num_hedges,
            'total_cost': total_cost,
            'avg_cost_per_hedge': total_cost / num_hedges,
            'total_volume_traded': total_volume,
            'hedge_history': self.hedge_history
        }

    def get_hedge_history_df(self) -> pd.DataFrame:
        """
        Get hedge history as DataFrame

        Returns:
        --------
        pd.DataFrame : Hedge transaction history
        """
        if not self.hedge_history:
            return pd.DataFrame()

        data = []
        for h in self.hedge_history:
            data.append({
                'timestamp': h.timestamp,
                'quantity': h.underlying_quantity,
                'price': h.underlying_price,
                'cost': h.transaction_cost,
                'delta_before': h.portfolio_delta_before,
                'delta_after': h.portfolio_delta_after,
                'reason': h.reason
            })

        return pd.DataFrame(data)


class GammaTradingStrategy:
    """
    Gamma trading strategy - profits from realized volatility

    Maintains delta-neutral position and rebalances to capture gamma profits
    """

    def __init__(self,
                 delta_hedger: DeltaHedger,
                 rebalance_threshold: float = 0.1):
        """
        Initialize Gamma Trading Strategy

        Parameters:
        -----------
        delta_hedger : DeltaHedger
            Delta hedging engine
        rebalance_threshold : float
            Delta threshold to trigger rebalancing
        """
        self.hedger = delta_hedger
        self.rebalance_threshold = rebalance_threshold

        self.gamma_pnl_history = []

    def execute_gamma_scalp(self,
                           timestamp: pd.Timestamp,
                           underlying_price: float,
                           price_change: float) -> Dict:
        """
        Execute gamma scalping trade

        Parameters:
        -----------
        timestamp : pd.Timestamp
            Current timestamp
        underlying_price : float
            Current underlying price
        price_change : float
            Change in underlying price

        Returns:
        --------
        dict : Gamma scalping result
        """
        # Get current gamma
        greeks = self.hedger.get_portfolio_greeks()
        gamma = greeks['gamma']

        # Theoretical gamma P&L = 0.5 * Gamma * (ΔS)²
        gamma_pnl_theoretical = 0.5 * gamma * (price_change ** 2)

        # Execute delta hedge
        hedge_transaction = self.hedger.execute_hedge(
            timestamp, underlying_price, force=False
        )

        # Actual P&L from hedging
        actual_pnl = 0.0
        if hedge_transaction:
            actual_pnl = -(hedge_transaction.underlying_quantity * price_change)

        result = {
            'timestamp': timestamp,
            'gamma': gamma,
            'price_change': price_change,
            'theoretical_gamma_pnl': gamma_pnl_theoretical,
            'actual_pnl': actual_pnl,
            'hedge_cost': hedge_transaction.transaction_cost if hedge_transaction else 0.0
        }

        self.gamma_pnl_history.append(result)

        return result

    def get_gamma_pnl_df(self) -> pd.DataFrame:
        """Get gamma P&L history as DataFrame"""
        return pd.DataFrame(self.gamma_pnl_history)


class VolatilityTrader:
    """
    Volatility trading - profit from differences between implied and realized vol
    """

    def __init__(self, delta_hedger: DeltaHedger):
        """
        Initialize Volatility Trader

        Parameters:
        -----------
        delta_hedger : DeltaHedger
            Delta hedging engine
        """
        self.hedger = delta_hedger
        self.pnl_history = []

    def calculate_vol_pnl(self,
                         realized_vol: float,
                         implied_vol: float,
                         vega: float,
                         time_period: float) -> float:
        """
        Calculate P&L from volatility difference

        Approximation: P&L ≈ Vega * (σ_realized - σ_implied) * √T

        Parameters:
        -----------
        realized_vol : float
            Realized volatility over period
        implied_vol : float
            Implied volatility at entry
        vega : float
            Portfolio vega
        time_period : float
            Time period (years)

        Returns:
        --------
        float : Estimated volatility P&L
        """
        vol_diff = realized_vol - implied_vol
        vol_pnl = vega * vol_diff * np.sqrt(time_period)

        return vol_pnl

    def record_period_pnl(self,
                         timestamp: pd.Timestamp,
                         realized_vol: float,
                         implied_vol: float):
        """
        Record volatility P&L for a period
        """
        greeks = self.hedger.get_portfolio_greeks()
        vega = greeks['vega']

        vol_pnl = self.calculate_vol_pnl(
            realized_vol, implied_vol, vega, 1/252  # Daily
        )

        self.pnl_history.append({
            'timestamp': timestamp,
            'realized_vol': realized_vol,
            'implied_vol': implied_vol,
            'vega': vega,
            'vol_pnl': vol_pnl
        })

    def get_vol_pnl_df(self) -> pd.DataFrame:
        """Get volatility P&L history as DataFrame"""
        return pd.DataFrame(self.pnl_history)
