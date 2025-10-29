"""
Options Pricing Models and Optimization Methods

Reorganized structure:
- options_pricing/: All pricing models (Black-Scholes, Heston, SABR, etc.)
- optimization_methods/: Portfolio optimization, strategy selection

Backward compatibility maintained for imports.
"""

# Import from subpackages for backward compatibility
from .options_pricing import (
    BaseModel,
    ModelParameters,
    BlackScholesModel,
    BSParameters,
    HestonModel,
    HestonParameters,
    SABRModel,
    BinomialTreeModel,
    LocalVolatilityModel,
    ModelCalibrator,
    ImpliedVolatilityCalculator
)

from .optimization_methods import (
    OptionPortfolio,
    PortfolioOptimizer,
    PortfolioGreeks,
    OptimizationResult,
    DPStrategySelector
)

__all__ = [
    # Pricing models
    'BaseModel',
    'ModelParameters',
    'BlackScholesModel',
    'BSParameters',
    'HestonModel',
    'HestonParameters',
    'SABRModel',
    'BinomialTreeModel',
    'LocalVolatilityModel',
    'ModelCalibrator',
    'ImpliedVolatilityCalculator',

    # Optimization methods
    'OptionPortfolio',
    'PortfolioOptimizer',
    'PortfolioGreeks',
    'OptimizationResult',
    'DPStrategySelector'
]
