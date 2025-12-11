"""
Trading Environment Module
==========================

SDE-Controlled State Transitions

Key Design Principle:
    State transition probability is determined by market SDE and volatility dynamics,
    NOT by agent actions. Agent actions only affect portfolio, not market state transitions.
"""

from .trading_environment import TradingEnvironment, MarketState, PortfolioState

__all__ = [
    'TradingEnvironment',
    'MarketState',
    'PortfolioState'
]



