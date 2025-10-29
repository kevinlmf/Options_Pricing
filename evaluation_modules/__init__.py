"""
Model Evaluation and Testing Framework

This package implements comprehensive evaluation tools for derivatives and risk models,
following the theoretical framework: Pure Math → Applied Math → Model Validation

Provides tools for:
- Model backtesting and validation
- Performance metrics and benchmarking
- Statistical tests for model adequacy
- Cross-validation and out-of-sample testing
- Model comparison and selection
"""

from .model_validation import ModelValidator, BacktestFramework, ValidationMetrics
from .performance_metrics import PerformanceAnalyzer, BenchmarkComparison, RiskAdjustedMetrics
from .statistical_tests import StatisticalTestSuite, GoodnessOfFitTests, ModelAdequacyTests

__all__ = [
    'ModelValidator',
    'BacktestFramework',
    'ValidationMetrics',
    'PerformanceAnalyzer',
    'BenchmarkComparison',
    'RiskAdjustedMetrics',
    'StatisticalTestSuite',
    'GoodnessOfFitTests',
    'ModelAdequacyTests'
]