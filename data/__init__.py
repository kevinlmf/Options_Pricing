"""
Data Management Package

This package implements data handling utilities for derivatives and risk management,
following the theoretical framework: Pure Math → Applied Math → Data Processing

Provides tools for:
- Market data collection and cleaning
- Financial time series preprocessing
- Data validation and quality checks
- Statistical data transformations
- Sample data generation for testing
"""

from .market_data import MarketDataProvider, YahooDataProvider, MockDataProvider
from .data_preprocessing import DataPreprocessor, TimeSeriesProcessor, FinancialDataValidator
from .sample_generators import SampleDataGenerator, CorrelatedReturnsGenerator, VolatilityModelGenerator

__all__ = [
    'MarketDataProvider',
    'YahooDataProvider',
    'MockDataProvider',
    'DataPreprocessor',
    'TimeSeriesProcessor',
    'FinancialDataValidator',
    'SampleDataGenerator',
    'CorrelatedReturnsGenerator',
    'VolatilityModelGenerator'
]