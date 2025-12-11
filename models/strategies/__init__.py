"""
Strategies Layer - Layer 8.5
============================

Advanced trading strategies that convert dual convergence volatility predictions
into executable trades with full risk management.

Key Strategies:
- Volatility Arbitrage: Buy cheap vol, sell expensive vol
- Gamma Scalping: Dynamic hedging for volatility harvesting
- PnL Attribution: Detailed performance breakdown
- Risk Management: Position sizing and stop-loss rules

This layer represents the final monetization of our dual convergence framework.
"""

from .volatility_arbitrage import (
    VolatilityArbitrageStrategy,
    VolatilityTradeSignal,
    OptionPosition,
    PortfolioPosition,
    PnLComponents
)

__all__ = [
    'VolatilityArbitrageStrategy',
    'VolatilityTradeSignal',
    'OptionPosition',
    'PortfolioPosition',
    'PnLComponents'
]
