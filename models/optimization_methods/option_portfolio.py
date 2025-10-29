"""
Option Portfolio Management

This module provides comprehensive option portfolio construction and management,
including various option strategies and portfolio-level risk analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import warnings

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from ..options_pricing.black_scholes import BlackScholesModel, BSParameters
from ..options_pricing.implied_volatility import VolatilityEstimator


class OptionType(Enum):
    """Option type enumeration."""
    CALL = "call"
    PUT = "put"


class PositionType(Enum):
    """Position type enumeration."""
    LONG = 1
    SHORT = -1


@dataclass
class OptionPosition:
    """
    Represents a single option position.

    Attributes:
    -----------
    symbol : str
        Underlying symbol
    option_type : OptionType
        Call or Put
    position_type : PositionType
        Long or Short
    strike : float
        Strike price
    expiry : float
        Time to expiry (years)
    quantity : int
        Number of contracts
    premium : float
        Option premium paid/received
    underlying_price : float
        Current underlying price
    volatility : float
        Implied volatility
    risk_free_rate : float
        Risk-free rate
    """
    symbol: str
    option_type: OptionType
    position_type: PositionType
    strike: float
    expiry: float
    quantity: int
    premium: float
    underlying_price: float
    volatility: float
    risk_free_rate: float = 0.05

    def __post_init__(self):
        """Initialize Black-Scholes model for this position."""
        self.bs_params = BSParameters(
            S0=self.underlying_price,
            K=self.strike,
            T=self.expiry,
            r=self.risk_free_rate,
            sigma=self.volatility
        )
        self.bs_model = BlackScholesModel(self.bs_params)

    @property
    def current_value(self) -> float:
        """Current theoretical value of the option."""
        if self.option_type == OptionType.CALL:
            price = self.bs_model.call_price()
        else:
            price = self.bs_model.put_price()
        return price * self.quantity * self.position_type.value

    @property
    def pnl(self) -> float:
        """Current P&L of the position."""
        current_val = self.current_value
        initial_val = self.premium * self.quantity * self.position_type.value
        return current_val - initial_val

    @property
    def greeks(self) -> Dict[str, float]:
        """Calculate position-level Greeks."""
        option_greeks = self.bs_model.greeks(self.option_type.value)
        position_greeks = {}

        for greek, value in option_greeks.items():
            position_greeks[greek] = value * self.quantity * self.position_type.value

        return position_greeks


class OptionStrategy:
    """
    Base class for option strategies.
    """

    def __init__(self, name: str, description: str):
        """
        Initialize option strategy.

        Parameters:
        -----------
        name : str
            Strategy name
        description : str
            Strategy description
        """
        self.name = name
        self.description = description
        self.positions: List[OptionPosition] = []

    def add_position(self, position: OptionPosition):
        """Add a position to the strategy."""
        self.positions.append(position)

    def remove_position(self, index: int):
        """Remove a position from the strategy."""
        if 0 <= index < len(self.positions):
            self.positions.pop(index)

    @property
    def total_value(self) -> float:
        """Total current value of all positions."""
        return sum(pos.current_value for pos in self.positions)

    @property
    def total_pnl(self) -> float:
        """Total P&L of all positions."""
        return sum(pos.pnl for pos in self.positions)

    @property
    def portfolio_greeks(self) -> Dict[str, float]:
        """Calculate portfolio-level Greeks."""
        portfolio_greeks = {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}

        for position in self.positions:
            pos_greeks = position.greeks
            for greek in portfolio_greeks:
                portfolio_greeks[greek] += pos_greeks.get(greek, 0)

        return portfolio_greeks

    def payoff_at_expiry(self, spot_prices: np.ndarray) -> np.ndarray:
        """
        Calculate strategy payoff at expiry for different spot prices.

        Parameters:
        -----------
        spot_prices : np.ndarray
            Array of possible spot prices at expiry

        Returns:
        --------
        np.ndarray : Payoff values
        """
        total_payoff = np.zeros_like(spot_prices)

        for position in self.positions:
            if position.option_type == OptionType.CALL:
                option_payoff = np.maximum(spot_prices - position.strike, 0)
            else:  # PUT
                option_payoff = np.maximum(position.strike - spot_prices, 0)

            # Account for position direction and quantity
            position_payoff = (option_payoff * position.quantity * position.position_type.value -
                             position.premium * position.quantity * position.position_type.value)
            total_payoff += position_payoff

        return total_payoff


class StrategyBuilder:
    """
    Builder class for constructing common option strategies.
    """

    def __init__(self, vol_estimator: Optional[VolatilityEstimator] = None):
        """
        Initialize strategy builder.

        Parameters:
        -----------
        vol_estimator : VolatilityEstimator, optional
            Volatility estimator for calculating option prices
        """
        self.vol_estimator = vol_estimator or VolatilityEstimator()

    def _create_option_position(self,
                               symbol: str,
                               underlying_price: float,
                               strike: float,
                               expiry: float,
                               option_type: OptionType,
                               position_type: PositionType,
                               quantity: int,
                               volatility: Optional[float] = None,
                               risk_free_rate: float = 0.05) -> OptionPosition:
        """Create an option position with calculated premium."""

        if volatility is None:
            # Use a default volatility if not provided
            volatility = 0.2

        # Calculate theoretical premium
        bs_params = BSParameters(
            S0=underlying_price,
            K=strike,
            T=expiry,
            r=risk_free_rate,
            sigma=volatility
        )
        bs_model = BlackScholesModel(bs_params)

        if option_type == OptionType.CALL:
            premium = bs_model.call_price()
        else:
            premium = bs_model.put_price()

        return OptionPosition(
            symbol=symbol,
            option_type=option_type,
            position_type=position_type,
            strike=strike,
            expiry=expiry,
            quantity=quantity,
            premium=premium,
            underlying_price=underlying_price,
            volatility=volatility,
            risk_free_rate=risk_free_rate
        )

    def protective_put(self,
                      symbol: str,
                      underlying_price: float,
                      strike: float,
                      expiry: float,
                      quantity: int = 1,
                      volatility: Optional[float] = None) -> OptionStrategy:
        """
        Create a protective put strategy.

        Strategy: Long stock + Long put
        """
        strategy = OptionStrategy(
            "Protective Put",
            "Long stock position protected by long put option"
        )

        put_position = self._create_option_position(
            symbol, underlying_price, strike, expiry,
            OptionType.PUT, PositionType.LONG, quantity, volatility
        )

        strategy.add_position(put_position)
        return strategy

    def covered_call(self,
                    symbol: str,
                    underlying_price: float,
                    strike: float,
                    expiry: float,
                    quantity: int = 1,
                    volatility: Optional[float] = None) -> OptionStrategy:
        """
        Create a covered call strategy.

        Strategy: Long stock + Short call
        """
        strategy = OptionStrategy(
            "Covered Call",
            "Long stock position with short call for income generation"
        )

        call_position = self._create_option_position(
            symbol, underlying_price, strike, expiry,
            OptionType.CALL, PositionType.SHORT, quantity, volatility
        )

        strategy.add_position(call_position)
        return strategy

    def long_straddle(self,
                     symbol: str,
                     underlying_price: float,
                     strike: float,
                     expiry: float,
                     quantity: int = 1,
                     volatility: Optional[float] = None) -> OptionStrategy:
        """
        Create a long straddle strategy.

        Strategy: Long call + Long put (same strike)
        """
        strategy = OptionStrategy(
            "Long Straddle",
            "Long call and put at same strike - profits from volatility"
        )

        call_position = self._create_option_position(
            symbol, underlying_price, strike, expiry,
            OptionType.CALL, PositionType.LONG, quantity, volatility
        )

        put_position = self._create_option_position(
            symbol, underlying_price, strike, expiry,
            OptionType.PUT, PositionType.LONG, quantity, volatility
        )

        strategy.add_position(call_position)
        strategy.add_position(put_position)
        return strategy

    def bull_call_spread(self,
                        symbol: str,
                        underlying_price: float,
                        lower_strike: float,
                        upper_strike: float,
                        expiry: float,
                        quantity: int = 1,
                        volatility: Optional[float] = None) -> OptionStrategy:
        """
        Create a bull call spread strategy.

        Strategy: Long call (lower strike) + Short call (higher strike)
        """
        strategy = OptionStrategy(
            "Bull Call Spread",
            "Limited upside profit strategy - long low strike call, short high strike call"
        )

        long_call = self._create_option_position(
            symbol, underlying_price, lower_strike, expiry,
            OptionType.CALL, PositionType.LONG, quantity, volatility
        )

        short_call = self._create_option_position(
            symbol, underlying_price, upper_strike, expiry,
            OptionType.CALL, PositionType.SHORT, quantity, volatility
        )

        strategy.add_position(long_call)
        strategy.add_position(short_call)
        return strategy

    def bear_put_spread(self,
                       symbol: str,
                       underlying_price: float,
                       lower_strike: float,
                       upper_strike: float,
                       expiry: float,
                       quantity: int = 1,
                       volatility: Optional[float] = None) -> OptionStrategy:
        """
        Create a bear put spread strategy.

        Strategy: Long put (higher strike) + Short put (lower strike)
        """
        strategy = OptionStrategy(
            "Bear Put Spread",
            "Limited downside profit strategy - long high strike put, short low strike put"
        )

        long_put = self._create_option_position(
            symbol, underlying_price, upper_strike, expiry,
            OptionType.PUT, PositionType.LONG, quantity, volatility
        )

        short_put = self._create_option_position(
            symbol, underlying_price, lower_strike, expiry,
            OptionType.PUT, PositionType.SHORT, quantity, volatility
        )

        strategy.add_position(long_put)
        strategy.add_position(short_put)
        return strategy

    def iron_condor(self,
                   symbol: str,
                   underlying_price: float,
                   strikes: Tuple[float, float, float, float],  # put_low, put_high, call_low, call_high
                   expiry: float,
                   quantity: int = 1,
                   volatility: Optional[float] = None) -> OptionStrategy:
        """
        Create an iron condor strategy.

        Strategy: Short put spread + Short call spread
        """
        put_low, put_high, call_low, call_high = strikes

        strategy = OptionStrategy(
            "Iron Condor",
            "Market neutral strategy - profits from low volatility"
        )

        # Short put spread
        short_put_high = self._create_option_position(
            symbol, underlying_price, put_high, expiry,
            OptionType.PUT, PositionType.SHORT, quantity, volatility
        )
        long_put_low = self._create_option_position(
            symbol, underlying_price, put_low, expiry,
            OptionType.PUT, PositionType.LONG, quantity, volatility
        )

        # Short call spread
        short_call_low = self._create_option_position(
            symbol, underlying_price, call_low, expiry,
            OptionType.CALL, PositionType.SHORT, quantity, volatility
        )
        long_call_high = self._create_option_position(
            symbol, underlying_price, call_high, expiry,
            OptionType.CALL, PositionType.LONG, quantity, volatility
        )

        strategy.add_position(short_put_high)
        strategy.add_position(long_put_low)
        strategy.add_position(short_call_low)
        strategy.add_position(long_call_high)

        return strategy


class OptionPortfolio:
    """
    Comprehensive option portfolio management class.
    """

    def __init__(self, name: str = "Options Portfolio"):
        """
        Initialize option portfolio.

        Parameters:
        -----------
        name : str
            Portfolio name
        """
        self.name = name
        self.strategies: List[OptionStrategy] = []
        self.individual_positions: List[OptionPosition] = []

    def add_strategy(self, strategy: OptionStrategy):
        """Add a strategy to the portfolio."""
        self.strategies.append(strategy)

    def add_position(self, position: OptionPosition):
        """Add an individual position to the portfolio."""
        self.individual_positions.append(position)

    def remove_strategy(self, index: int):
        """Remove a strategy from the portfolio."""
        if 0 <= index < len(self.strategies):
            self.strategies.pop(index)

    def remove_position(self, index: int):
        """Remove an individual position from the portfolio."""
        if 0 <= index < len(self.individual_positions):
            self.individual_positions.pop(index)

    @property
    def all_positions(self) -> List[OptionPosition]:
        """Get all positions (from strategies and individual positions)."""
        all_positions = []

        # Add positions from strategies
        for strategy in self.strategies:
            all_positions.extend(strategy.positions)

        # Add individual positions
        all_positions.extend(self.individual_positions)

        return all_positions

    @property
    def total_value(self) -> float:
        """Total current value of the portfolio."""
        return sum(pos.current_value for pos in self.all_positions)

    @property
    def total_pnl(self) -> float:
        """Total P&L of the portfolio."""
        return sum(pos.pnl for pos in self.all_positions)

    @property
    def portfolio_greeks(self) -> Dict[str, float]:
        """Calculate portfolio-level Greeks."""
        portfolio_greeks = {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}

        for position in self.all_positions:
            pos_greeks = position.greeks
            for greek in portfolio_greeks:
                portfolio_greeks[greek] += pos_greeks.get(greek, 0)

        return portfolio_greeks

    def get_summary(self) -> pd.DataFrame:
        """Get portfolio summary as DataFrame."""
        positions = self.all_positions

        if not positions:
            return pd.DataFrame()

        summary_data = []
        for i, pos in enumerate(positions):
            greeks = pos.greeks
            summary_data.append({
                'Position': i + 1,
                'Symbol': pos.symbol,
                'Type': pos.option_type.value.upper(),
                'Position': 'Long' if pos.position_type == PositionType.LONG else 'Short',
                'Strike': pos.strike,
                'Expiry': pos.expiry,
                'Quantity': pos.quantity,
                'Premium': pos.premium,
                'Current_Value': pos.current_value,
                'PnL': pos.pnl,
                'Delta': greeks.get('delta', 0),
                'Gamma': greeks.get('gamma', 0),
                'Theta': greeks.get('theta', 0),
                'Vega': greeks.get('vega', 0),
                'Rho': greeks.get('rho', 0)
            })

        return pd.DataFrame(summary_data)

    def exposure_by_underlying(self) -> Dict[str, Dict]:
        """Calculate exposure by underlying symbol."""
        exposures = {}

        for position in self.all_positions:
            symbol = position.symbol

            if symbol not in exposures:
                exposures[symbol] = {
                    'total_value': 0,
                    'total_pnl': 0,
                    'delta': 0,
                    'gamma': 0,
                    'theta': 0,
                    'vega': 0,
                    'rho': 0,
                    'positions': 0
                }

            exposures[symbol]['total_value'] += position.current_value
            exposures[symbol]['total_pnl'] += position.pnl
            exposures[symbol]['positions'] += 1

            # Add Greeks
            pos_greeks = position.greeks
            for greek in ['delta', 'gamma', 'theta', 'vega', 'rho']:
                exposures[symbol][greek] += pos_greeks.get(greek, 0)

        return exposures