"""
Multi-Agent Structural Model for Parameter Forecasting

Structural approach: Simulate agent behaviors to derive market parameters
vs. Reduced-form: Direct statistical modeling (LSTM/GARCH)
"""

from .agent_forecaster import MultiAgentForecaster
from .agents import MarketMaker, Arbitrageur, NoiseTrader

__all__ = [
    'MultiAgentForecaster',
    'MarketMaker',
    'Arbitrageur',
    'NoiseTrader'
]
