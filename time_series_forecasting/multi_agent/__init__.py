"""
Multi-Agent Structural Model for Parameter Forecasting

Structural approach: Simulate agent behaviors to derive market parameters
vs. Reduced-form: Direct statistical modeling (LSTM/GARCH)

New: Rust-accelerated Monte Carlo validation for statistical soundness
"""

from .agent_forecaster import MultiAgentForecaster
from .agents import MarketMaker, Arbitrageur, NoiseTrader

# Validated forecaster with Rust Monte Carlo
try:
    from .validated_forecaster import (
        ValidatedMultiAgentForecaster,
        create_validated_forecaster
    )
    from .adaptive_coordinator import (
        AdaptiveAgentCoordinator,
        TrainingStepResult
    )
    __all__ = [
        'MultiAgentForecaster',
        'ValidatedMultiAgentForecaster',
        'create_validated_forecaster',
        'MarketMaker',
        'Arbitrageur',
        'NoiseTrader',
        'AdaptiveAgentCoordinator',
        'TrainingStepResult'
    ]
except ImportError:
    # Rust module not available
    __all__ = [
        'MultiAgentForecaster',
        'MarketMaker',
        'Arbitrageur',
        'NoiseTrader'
    ]
