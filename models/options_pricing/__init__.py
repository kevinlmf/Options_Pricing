"""
Options Pricing Layer - Layer 3.5
=================================

Advanced options pricing engine that integrates dual convergence volatility forecasts
with stochastic volatility models (Heston/SABR) and Monte Carlo simulation.

Key Features:
- Dual convergence σ(t) as input to Heston/SABR models
- Monte Carlo pricing using existing Rust engine
- Implied volatility surface calibration
- Volatility arbitrage signals (skew mispricing)
- Enhanced Greeks calculation
- Dynamic volatility hedging strategies

This layer provides the 'pricing lens' for our dual convergence framework,
complementing the factor allocation approach for complete options trading capability.
"""

from .dual_convergence_pricer import DualConvergencePricer, PricingEngine, OptionContract
from .volatility_arbitrage import VolatilityArbitrageDetector, ArbitrageSignal
from .dynamic_hedge_engine import DynamicHedgeEngine, HedgePosition

__all__ = [
    'DualConvergencePricer',
    'PricingEngine',
    'OptionContract',
    'VolatilityArbitrageDetector',
    'ArbitrageSignal',
    'DynamicHedgeEngine',
    'HedgePosition'
]

