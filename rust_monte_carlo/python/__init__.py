"""
Monte Carlo Validator - High-Performance Rust Backend for Multi-Agent Validation

This module provides ultra-fast Monte Carlo simulation capabilities
for validating multi-agent option pricing forecasts.
"""

from .monte_carlo_validator import (
    MonteCarloValidator,
    quick_validate,
    batch_validate,
    ValidationResult
)

__all__ = [
    'MonteCarloValidator',
    'quick_validate',
    'batch_validate',
    'ValidationResult',
]

__version__ = '0.1.0'
