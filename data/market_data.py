"""
Market Data Providers

Implementation of market data collection and management utilities.
Follows the mathematical framework: Pure Math → Applied Math → Data Processing

Mathematical Foundation:
- Pure Math: Time series analysis, statistical properties
- Applied Math: Data interpolation, filtering, stochastic modeling
- Data Processing: API integration, data cleaning, validation
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Union, Optional, Tuple, Any
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
import warnings
import requests
from io import StringIO


class MarketDataProvider(ABC):
    """Abstract base class for market data providers"""

    @abstractmethod
    def get_price_data(self, symbol: str, start_date: str, end_date: str,
                      frequency: str = 'daily') -> pd.DataFrame:
        """
        Get historical price data for a symbol.

        Args:
            symbol: Ticker symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            frequency: Data frequency ('daily', 'weekly', 'monthly')

        Returns:
            DataFrame with OHLCV data
        """
        pass

    @abstractmethod
    def get_multiple_symbols(self, symbols: List[str], start_date: str, end_date: str,
                           price_column: str = 'Close') -> pd.DataFrame:
        """
        Get price data for multiple symbols.

        Args:
            symbols: List of ticker symbols
            start_date: Start date
            end_date: End date
            price_column: Price column to extract

        Returns:
            DataFrame with symbols as columns
        """
        pass

    def calculate_returns(self, prices: pd.DataFrame, method: str = 'simple') -> pd.DataFrame:
        """
        Calculate returns from price data.

        Args:
            prices: Price DataFrame
            method: Return calculation method ('simple', 'log')

        Returns:
            Returns DataFrame
        """
        if method == 'simple':
            returns = prices.pct_change()
        elif method == 'log':
            returns = np.log(prices / prices.shift(1))
        else:
            raise ValueError("Method must be 'simple' or 'log'")

        return returns.dropna()

    def validate_data_quality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate data quality and identify issues.

        Args:
            data: Price or returns data

        Returns:
            Data quality report
        """
        report = {
            'n_observations': len(data),
            'n_columns': len(data.columns),
            'date_range': {
                'start': data.index[0] if len(data) > 0 else None,
                'end': data.index[-1] if len(data) > 0 else None
            },
            'missing_data': {},
            'outliers': {},
            'data_gaps': [],
            'suspicious_values': {}
        }

        # Check missing data
        for col in data.columns:
            missing_count = data[col].isna().sum()
            missing_pct = missing_count / len(data) * 100
            report['missing_data'][col] = {
                'count': missing_count,
                'percentage': missing_pct
            }

        # Check for outliers (values > 5 standard deviations)
        for col in data.select_dtypes(include=[np.number]).columns:
            col_data = data[col].dropna()
            if len(col_data) > 0:
                mean_val = col_data.mean()
                std_val = col_data.std()
                outliers = col_data[np.abs(col_data - mean_val) > 5 * std_val]
                report['outliers'][col] = {
                    'count': len(outliers),
                    'values': outliers.tolist()
                }

        # Check for data gaps (missing dates)
        if isinstance(data.index, pd.DatetimeIndex):
            expected_dates = pd.date_range(start=data.index[0], end=data.index[-1], freq='D')
            actual_dates = set(data.index.date)
            expected_dates_set = set(expected_dates.date)
            missing_dates = expected_dates_set - actual_dates

            # Filter out weekends for daily data
            missing_weekdays = [d for d in missing_dates if d.weekday() < 5]
            report['data_gaps'] = sorted(missing_weekdays)

        # Check for suspicious values (zeros, negative prices, extreme values)
        for col in data.select_dtypes(include=[np.number]).columns:
            col_data = data[col].dropna()
            suspicious = {
                'zeros': (col_data == 0).sum(),
                'negatives': (col_data < 0).sum(),
                'infinite': np.isinf(col_data).sum()
            }
            report['suspicious_values'][col] = suspicious

        return report


class YahooDataProvider(MarketDataProvider):
    """
    Yahoo Finance data provider using direct API calls.
    Note: This is a simplified implementation for demonstration.
    """

    def __init__(self, cache_data: bool = True):
        """
        Initialize Yahoo data provider.

        Args:
            cache_data: Whether to cache downloaded data
        """
        self.cache_data = cache_data
        self.cache = {}

    def get_price_data(self, symbol: str, start_date: str, end_date: str,
                      frequency: str = 'daily') -> pd.DataFrame:
        """Get historical price data from Yahoo Finance"""

        # Create cache key
        cache_key = f"{symbol}_{start_date}_{end_date}_{frequency}"

        if self.cache_data and cache_key in self.cache:
            return self.cache[cache_key].copy()

        try:
            # Convert dates to timestamps
            start_ts = int(pd.Timestamp(start_date).timestamp())
            end_ts = int(pd.Timestamp(end_date).timestamp())

            # Construct Yahoo Finance URL
            interval_map = {'daily': '1d', 'weekly': '1wk', 'monthly': '1mo'}
            interval = interval_map.get(frequency, '1d')

            url = f"https://query1.finance.yahoo.com/v7/finance/download/{symbol}"
            params = {
                'period1': start_ts,
                'period2': end_ts,
                'interval': interval,
                'events': 'history',
                'includeAdjustedClose': 'true'
            }

            # Make request with headers to avoid blocking
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }

            response = requests.get(url, params=params, headers=headers)
            response.raise_for_status()

            # Parse CSV data
            data = pd.read_csv(StringIO(response.text))
            data['Date'] = pd.to_datetime(data['Date'])
            data.set_index('Date', inplace=True)
            data.sort_index(inplace=True)

            # Rename columns for consistency
            if 'Adj Close' in data.columns:
                data['AdjClose'] = data['Adj Close']
                data.drop('Adj Close', axis=1, inplace=True)

            # Cache the data
            if self.cache_data:
                self.cache[cache_key] = data.copy()

            return data

        except Exception as e:
            warnings.warn(f"Failed to fetch data for {symbol}: {e}")
            # Return empty DataFrame with expected structure
            return pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Volume', 'AdjClose'])

    def get_multiple_symbols(self, symbols: List[str], start_date: str, end_date: str,
                           price_column: str = 'Close') -> pd.DataFrame:
        """Get price data for multiple symbols"""

        all_data = {}

        for symbol in symbols:
            try:
                data = self.get_price_data(symbol, start_date, end_date)
                if not data.empty and price_column in data.columns:
                    all_data[symbol] = data[price_column]
                else:
                    print(f"Warning: No data or missing column for {symbol}")
            except Exception as e:
                print(f"Error fetching data for {symbol}: {e}")

        if not all_data:
            return pd.DataFrame()

        # Combine all data
        combined_data = pd.DataFrame(all_data)

        # Forward fill missing data and then drop any remaining NaN rows
        combined_data = combined_data.fillna(method='ffill').dropna()

        return combined_data

    def get_option_data(self, symbol: str, expiration_date: Optional[str] = None) -> pd.DataFrame:
        """
        Get option data for a symbol (simplified implementation).
        In practice, this would require more sophisticated option data APIs.
        """
        warnings.warn("Option data fetching not fully implemented - returning sample data")

        # Return sample option data structure
        sample_data = pd.DataFrame({
            'Strike': [90, 95, 100, 105, 110],
            'Call_Price': [12.5, 8.2, 4.8, 2.1, 0.8],
            'Put_Price': [0.3, 1.1, 2.8, 6.1, 10.8],
            'Call_IV': [0.22, 0.20, 0.19, 0.21, 0.24],
            'Put_IV': [0.24, 0.21, 0.19, 0.20, 0.22],
            'Call_Volume': [150, 230, 180, 90, 45],
            'Put_Volume': [45, 80, 160, 220, 180]
        })

        return sample_data


class MockDataProvider(MarketDataProvider):
    """
    Mock data provider for testing and demonstration purposes.
    Generates realistic synthetic financial data.
    """

    def __init__(self, random_seed: Optional[int] = None):
        """
        Initialize mock data provider.

        Args:
            random_seed: Random seed for reproducible data
        """
        self.random_seed = random_seed
        if random_seed is not None:
            np.random.seed(random_seed)

    def get_price_data(self, symbol: str, start_date: str, end_date: str,
                      frequency: str = 'daily') -> pd.DataFrame:
        """Generate synthetic price data"""

        # Create date range
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        n_days = len(date_range)

        if n_days == 0:
            return pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Volume', 'AdjClose'])

        # Symbol-specific parameters
        symbol_params = self._get_symbol_parameters(symbol)

        # Generate returns using GBM
        dt = 1/252  # Daily
        returns = np.random.normal(
            symbol_params['mu'] * dt,
            symbol_params['sigma'] * np.sqrt(dt),
            n_days
        )

        # Add some volatility clustering (GARCH-like effects)
        volatility = np.ones(n_days) * symbol_params['sigma']
        for i in range(1, n_days):
            volatility[i] = 0.95 * volatility[i-1] + 0.05 * symbol_params['sigma'] + 0.03 * abs(returns[i-1])
            returns[i] = np.random.normal(symbol_params['mu'] * dt, volatility[i] * np.sqrt(dt))

        # Generate price levels
        initial_price = symbol_params['initial_price']
        prices = initial_price * np.exp(np.cumsum(returns))

        # Generate OHLC data
        data = []
        for i, price in enumerate(prices):
            if i == 0:
                open_price = initial_price
            else:
                open_price = data[-1]['Close']

            close_price = price

            # Generate high and low with some randomness
            high_low_range = abs(close_price - open_price) * np.random.uniform(0.5, 2.0)
            high = max(open_price, close_price) + high_low_range * np.random.uniform(0, 0.5)
            low = min(open_price, close_price) - high_low_range * np.random.uniform(0, 0.5)

            # Ensure high >= max(open, close) and low <= min(open, close)
            high = max(high, open_price, close_price)
            low = min(low, open_price, close_price)

            # Generate volume (log-normal distribution)
            base_volume = symbol_params['avg_volume']
            volume_multiplier = np.random.lognormal(0, 0.5)
            volume = int(base_volume * volume_multiplier)

            data.append({
                'Open': round(open_price, 2),
                'High': round(high, 2),
                'Low': round(low, 2),
                'Close': round(close_price, 2),
                'Volume': volume,
                'AdjClose': round(close_price, 2)  # Assume no adjustments
            })

        # Create DataFrame
        df = pd.DataFrame(data, index=date_range)

        # Apply frequency filter if not daily
        if frequency == 'weekly':
            df = df.resample('W').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum',
                'AdjClose': 'last'
            })
        elif frequency == 'monthly':
            df = df.resample('M').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum',
                'AdjClose': 'last'
            })

        return df

    def get_multiple_symbols(self, symbols: List[str], start_date: str, end_date: str,
                           price_column: str = 'Close') -> pd.DataFrame:
        """Generate synthetic data for multiple symbols with realistic correlations"""

        # Get individual price series
        all_data = {}
        returns_data = {}

        for symbol in symbols:
            price_data = self.get_price_data(symbol, start_date, end_date)
            if not price_data.empty:
                all_data[symbol] = price_data[price_column]
                returns_data[symbol] = price_data[price_column].pct_change().dropna()

        if not all_data:
            return pd.DataFrame()

        # Apply correlation structure to returns
        if len(symbols) > 1:
            returns_df = pd.DataFrame(returns_data)

            # Generate correlation matrix based on symbol types
            corr_matrix = self._generate_correlation_matrix(symbols)

            # Apply correlation using Cholesky decomposition
            if len(returns_df) > 1:
                try:
                    L = np.linalg.cholesky(corr_matrix)
                    uncorrelated_returns = np.random.standard_normal((len(returns_df), len(symbols)))
                    correlated_returns = np.dot(uncorrelated_returns, L.T)

                    # Replace returns with correlated versions (maintaining original volatility)
                    for i, symbol in enumerate(symbols):
                        original_std = returns_df[symbol].std()
                        original_mean = returns_df[symbol].mean()
                        new_returns = correlated_returns[:, i] * original_std + original_mean

                        # Reconstruct prices
                        initial_price = all_data[symbol].iloc[0]
                        new_prices = initial_price * np.exp(np.cumsum(np.concatenate([[0], new_returns])))
                        all_data[symbol] = pd.Series(new_prices, index=all_data[symbol].index)
                except np.linalg.LinAlgError:
                    # If correlation matrix is not positive definite, use original data
                    pass

        combined_data = pd.DataFrame(all_data)
        return combined_data.fillna(method='ffill').dropna()

    def _get_symbol_parameters(self, symbol: str) -> Dict[str, float]:
        """Get realistic parameters for different symbol types"""

        # Default parameters
        params = {
            'mu': 0.08,  # 8% annual return
            'sigma': 0.20,  # 20% annual volatility
            'initial_price': 100.0,
            'avg_volume': 1000000
        }

        # Symbol-specific adjustments
        symbol_upper = symbol.upper()

        if any(tech in symbol_upper for tech in ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']):
            # Tech stocks: higher volatility, higher returns
            params.update({'mu': 0.12, 'sigma': 0.30, 'initial_price': 150.0, 'avg_volume': 2000000})

        elif any(fin in symbol_upper for fin in ['JPM', 'BAC', 'WFC', 'GS', 'MS']):
            # Financial stocks: moderate volatility
            params.update({'mu': 0.06, 'sigma': 0.25, 'initial_price': 80.0, 'avg_volume': 1500000})

        elif any(util in symbol_upper for util in ['JNJ', 'PG', 'KO', 'PEP']):
            # Utilities/Consumer staples: lower volatility, steady returns
            params.update({'mu': 0.05, 'sigma': 0.15, 'initial_price': 120.0, 'avg_volume': 800000})

        elif 'SPY' in symbol_upper or 'QQQ' in symbol_upper:
            # ETFs: market-like behavior
            params.update({'mu': 0.07, 'sigma': 0.18, 'initial_price': 300.0, 'avg_volume': 5000000})

        elif 'BTC' in symbol_upper or 'ETH' in symbol_upper:
            # Crypto: very high volatility
            params.update({'mu': 0.15, 'sigma': 0.80, 'initial_price': 50000.0, 'avg_volume': 500000})

        return params

    def _generate_correlation_matrix(self, symbols: List[str]) -> np.ndarray:
        """Generate realistic correlation matrix for symbols"""
        n = len(symbols)

        # Start with base correlation
        base_corr = 0.3
        corr_matrix = np.full((n, n), base_corr)
        np.fill_diagonal(corr_matrix, 1.0)

        # Adjust correlations based on sector similarity
        for i in range(n):
            for j in range(i+1, n):
                symbol_i = symbols[i].upper()
                symbol_j = symbols[j].upper()

                # Higher correlation for same sector
                if self._same_sector(symbol_i, symbol_j):
                    corr_matrix[i, j] = corr_matrix[j, i] = 0.7
                elif self._related_sectors(symbol_i, symbol_j):
                    corr_matrix[i, j] = corr_matrix[j, i] = 0.5
                else:
                    corr_matrix[i, j] = corr_matrix[j, i] = np.random.uniform(0.1, 0.4)

        # Ensure positive definite
        eigenvals, eigenvecs = np.linalg.eigh(corr_matrix)
        eigenvals = np.maximum(eigenvals, 0.01)  # Ensure positive eigenvalues
        corr_matrix = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T

        # Rescale to correlation matrix
        d = np.sqrt(np.diag(corr_matrix))
        corr_matrix = corr_matrix / np.outer(d, d)

        return corr_matrix

    def _same_sector(self, symbol1: str, symbol2: str) -> bool:
        """Check if symbols are from the same sector"""
        tech_stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
        financial_stocks = ['JPM', 'BAC', 'WFC', 'GS', 'MS']
        consumer_stocks = ['JNJ', 'PG', 'KO', 'PEP']

        sectors = [tech_stocks, financial_stocks, consumer_stocks]

        for sector in sectors:
            if symbol1 in sector and symbol2 in sector:
                return True

        return False

    def _related_sectors(self, symbol1: str, symbol2: str) -> bool:
        """Check if symbols are from related sectors"""
        # Tech and growth stocks tend to be correlated
        growth_related = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'QQQ']

        # Financial and economic sensitive stocks
        economy_sensitive = ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'SPY']

        related_groups = [growth_related, economy_sensitive]

        for group in related_groups:
            if symbol1 in group and symbol2 in group:
                return True

        return False


if __name__ == "__main__":
    # Demonstration of market data providers
    print("Market Data Providers Demonstration")
    print("=" * 50)

    # Test Mock Data Provider
    print("\n1. Mock Data Provider Test:")
    print("-" * 30)

    mock_provider = MockDataProvider(random_seed=42)

    # Single symbol test
    apple_data = mock_provider.get_price_data('AAPL', '2023-01-01', '2023-12-31')
    print(f"AAPL data shape: {apple_data.shape}")
    print(f"Date range: {apple_data.index[0]} to {apple_data.index[-1]}")
    print(f"Price range: ${apple_data['Close'].min():.2f} - ${apple_data['Close'].max():.2f}")
    print(f"Average volume: {apple_data['Volume'].mean():,.0f}")

    # Multiple symbols test
    symbols = ['AAPL', 'MSFT', 'SPY', 'JPM', 'JNJ']
    multi_data = mock_provider.get_multiple_symbols(symbols, '2023-01-01', '2023-06-30', 'Close')
    print(f"\nMultiple symbols data shape: {multi_data.shape}")
    print("Correlation matrix:")
    returns = mock_provider.calculate_returns(multi_data)
    correlation_matrix = returns.corr()
    print(correlation_matrix.round(3))

    # Data quality validation
    print("\n2. Data Quality Validation:")
    print("-" * 30)

    quality_report = mock_provider.validate_data_quality(multi_data)
    print(f"Observations: {quality_report['n_observations']}")
    print(f"Columns: {quality_report['n_columns']}")
    print(f"Date range: {quality_report['date_range']['start']} to {quality_report['date_range']['end']}")

    print("\nMissing data:")
    for col, info in quality_report['missing_data'].items():
        print(f"  {col}: {info['count']} missing ({info['percentage']:.1f}%)")

    print("\nOutliers detected:")
    for col, info in quality_report['outliers'].items():
        if info['count'] > 0:
            print(f"  {col}: {info['count']} outliers")

    # Return analysis
    print("\n3. Return Analysis:")
    print("-" * 20)

    print("Simple returns statistics:")
    simple_returns = mock_provider.calculate_returns(multi_data, method='simple')
    print(simple_returns.describe().round(4))

    print("\nLog returns statistics:")
    log_returns = mock_provider.calculate_returns(multi_data, method='log')
    print(log_returns.describe().round(4))

    # Test Yahoo Finance provider (with fallback to mock if unavailable)
    print("\n4. Yahoo Finance Provider Test:")
    print("-" * 35)

    try:
        yahoo_provider = YahooDataProvider()

        # Try to get real data (may fail due to network/API issues)
        try:
            spy_data = yahoo_provider.get_price_data('SPY', '2023-01-01', '2023-01-31')
            if not spy_data.empty:
                print(f"SPY data successfully retrieved: {spy_data.shape}")
                print(f"Columns: {list(spy_data.columns)}")
                print(f"Latest close: ${spy_data['Close'].iloc[-1]:.2f}")
            else:
                print("No data retrieved from Yahoo Finance")
        except Exception as e:
            print(f"Yahoo Finance data retrieval failed: {e}")
            print("This is normal in environments without internet access")

        # Test with multiple symbols (will likely use mock data as fallback)
        test_symbols = ['SPY', 'QQQ']
        try:
            multi_yahoo = yahoo_provider.get_multiple_symbols(test_symbols, '2023-01-01', '2023-01-31')
            if not multi_yahoo.empty:
                print(f"Multiple symbols data: {multi_yahoo.shape}")
        except:
            print("Multiple symbols test failed - using mock data as intended fallback")

    except Exception as e:
        print(f"Yahoo provider initialization failed: {e}")

    print("\n5. Sample Option Data:")
    print("-" * 25)

    # Get sample option data (from Yahoo provider's mock implementation)
    try:
        yahoo_provider = YahooDataProvider()
        option_data = yahoo_provider.get_option_data('AAPL')
        print("Sample option chain:")
        print(option_data.round(2))

        # Calculate put-call parity check
        S = 100  # Current stock price
        r = 0.05  # Risk-free rate
        T = 0.25  # 3 months to expiry

        print(f"\nPut-Call Parity Check (S=${S}, r={r:.1%}, T={T:.2f}y):")
        for _, row in option_data.iterrows():
            K = row['Strike']
            C = row['Call_Price']
            P = row['Put_Price']

            # Put-call parity: C - P = S - K*e^(-r*T)
            theoretical_diff = S - K * np.exp(-r * T)
            actual_diff = C - P

            print(f"Strike {K}: Theoretical={theoretical_diff:.2f}, Actual={actual_diff:.2f}")

    except Exception as e:
        print(f"Option data test failed: {e}")

    print(f"\nMarket data providers demonstration completed successfully!")
    print(f"Generated {len(multi_data)} days of data for {len(symbols)} symbols")