"""
Multi-Agent Option Pricing Framework

This module implements a multi-agent system to explain option pricing deviations
from theoretical models (like Black-Scholes) through market microstructure and
agent behavior.

Key Components:
- BaseAgent: Abstract base class for all market agents
- MarketMaker: Manages inventory risk and provides liquidity
- Arbitrageur: Exploits pricing deviations subject to capital constraints
- NoiseTrader: Generates demand/supply imbalances through behavioral patterns
- AgentInteraction: Coordinates agent interactions and price formation
"""

from .base_agent import BaseAgent, AgentState, MarketState
from .market_maker import MarketMaker, MarketMakerParameters
from .arbitrageur import Arbitrageur, ArbitrageParameters
from .noise_trader import NoiseTrader, NoiseTraderParameters, NoiseTraderBehavior
from .agent_interaction import AgentInteractionEngine, MarketEquilibrium

__all__ = [
    'BaseAgent', 'AgentState', 'MarketState',
    'MarketMaker', 'MarketMakerParameters',
    'Arbitrageur', 'ArbitrageParameters',
    'NoiseTrader', 'NoiseTraderParameters', 'NoiseTraderBehavior',
    'AgentInteractionEngine', 'MarketEquilibrium'
]