"""
Data Preprocessing Utilities

Implementation of data preprocessing and validation tools for financial data.
Follows the mathematical framework: Pure Math → Applied Math → Data Processing

Mathematical Foundation:
- Pure Math: Statistical transformations, outlier detection theory
- Applied Math: Time series analysis, filtering, interpolation
- Data Processing: Missing data handling, normalization, validation
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Union, Optional, Tuple, Any, Callable
from scipy import stats
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler, RobustScaler
import warnings


class DataPreprocessor:
    """
    General data preprocessing utilities for financial data.
    """

    def __init__(self):
        self.scalers = {}
        self.preprocessing_history = []

    def handle_missing_data(self, data: pd.DataFrame, method: str = 'forward_fill',
                          limit: Optional[int] = None) -> pd.DataFrame:
        """
        Handle missing data in financial time series.

        Args:
            data: Input DataFrame
            method: Method for handling missing data
                   ('forward_fill', 'backward_fill', 'interpolate', 'drop', 'mean_fill')
            limit: Maximum number of consecutive NaN values to fill

        Returns:
            DataFrame with missing data handled
        """
        data_clean = data.copy()

        if method == 'forward_fill':
            data_clean = data_clean.fillna(method='ffill', limit=limit)

        elif method == 'backward_fill':
            data_clean = data_clean.fillna(method='bfill', limit=limit)

        elif method == 'interpolate':
            # Use time-based interpolation for financial data
            data_clean = data_clean.interpolate(method='time', limit=limit)

        elif method == 'linear_interpolate':
            data_clean = data_clean.interpolate(method='linear', limit=limit)

        elif method == 'drop':
            data_clean = data_clean.dropna()

        elif method == 'mean_fill':
            for col in data_clean.select_dtypes(include=[np.number]).columns:
                mean_val = data_clean[col].mean()
                data_clean[col] = data_clean[col].fillna(mean_val)

        elif method == 'median_fill':
            for col in data_clean.select_dtypes(include=[np.number]).columns:
                median_val = data_clean[col].median()
                data_clean[col] = data_clean[col].fillna(median_val)

        else:
            raise ValueError(f"Unknown method: {method}")

        # Record preprocessing step
        self.preprocessing_history.append({
            'step': 'handle_missing_data',
            'method': method,
            'limit': limit,
            'na_before': data.isna().sum().sum(),
            'na_after': data_clean.isna().sum().sum()
        })

        return data_clean

    def detect_outliers(self, data: pd.DataFrame, method: str = 'iqr',
                       threshold: float = 1.5) -> pd.DataFrame:
        """
        Detect outliers in financial data.

        Args:
            data: Input DataFrame
            method: Outlier detection method ('iqr', 'zscore', 'modified_zscore', 'isolation_forest')
            threshold: Threshold parameter for outlier detection

        Returns:
            Boolean DataFrame indicating outliers
        """
        outliers = pd.DataFrame(False, index=data.index, columns=data.columns)

        for col in data.select_dtypes(include=[np.number]).columns:
            col_data = data[col].dropna()

            if len(col_data) == 0:
                continue

            if method == 'iqr':
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                outliers[col] = (data[col] < lower_bound) | (data[col] > upper_bound)

            elif method == 'zscore':
                z_scores = np.abs(stats.zscore(col_data, nan_policy='omit'))
                outlier_indices = col_data.index[z_scores > threshold]
                outliers.loc[outlier_indices, col] = True

            elif method == 'modified_zscore':
                median = np.median(col_data)
                mad = np.median(np.abs(col_data - median))
                modified_z_scores = 0.6745 * (col_data - median) / mad
                outlier_indices = col_data.index[np.abs(modified_z_scores) > threshold]
                outliers.loc[outlier_indices, col] = True

            elif method == 'rolling_zscore':
                # Rolling Z-score for time series data
                window = min(30, len(col_data) // 4)
                if window > 5:
                    rolling_mean = data[col].rolling(window=window).mean()
                    rolling_std = data[col].rolling(window=window).std()
                    rolling_zscore = np.abs((data[col] - rolling_mean) / rolling_std)
                    outliers[col] = rolling_zscore > threshold

            else:
                raise ValueError(f"Unknown outlier detection method: {method}")

        return outliers

    def winsorize_data(self, data: pd.DataFrame, lower_percentile: float = 0.01,
                      upper_percentile: float = 0.99) -> pd.DataFrame:
        """
        Winsorize data to handle extreme outliers.

        Args:
            data: Input DataFrame
            lower_percentile: Lower percentile for winsorization
            upper_percentile: Upper percentile for winsorization

        Returns:
            Winsorized DataFrame
        """
        data_winsorized = data.copy()

        for col in data.select_dtypes(include=[np.number]).columns:
            col_data = data[col].dropna()

            if len(col_data) == 0:
                continue

            lower_bound = col_data.quantile(lower_percentile)
            upper_bound = col_data.quantile(upper_percentile)

            data_winsorized[col] = data_winsorized[col].clip(lower=lower_bound, upper=upper_bound)

        self.preprocessing_history.append({
            'step': 'winsorize_data',
            'lower_percentile': lower_percentile,
            'upper_percentile': upper_percentile
        })

        return data_winsorized

    def normalize_data(self, data: pd.DataFrame, method: str = 'standard',
                      feature_range: Tuple[float, float] = (0, 1)) -> pd.DataFrame:
        """
        Normalize/standardize financial data.

        Args:
            data: Input DataFrame
            method: Normalization method ('standard', 'robust', 'minmax', 'unit_vector')
            feature_range: Range for MinMax scaling

        Returns:
            Normalized DataFrame
        """
        data_normalized = data.copy()
        numeric_columns = data.select_dtypes(include=[np.number]).columns

        if method == 'standard':
            scaler = StandardScaler()
            data_normalized[numeric_columns] = scaler.fit_transform(data[numeric_columns])
            self.scalers['standard'] = scaler

        elif method == 'robust':
            scaler = RobustScaler()
            data_normalized[numeric_columns] = scaler.fit_transform(data[numeric_columns])
            self.scalers['robust'] = scaler

        elif method == 'minmax':
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler(feature_range=feature_range)
            data_normalized[numeric_columns] = scaler.fit_transform(data[numeric_columns])
            self.scalers['minmax'] = scaler

        elif method == 'unit_vector':
            # Normalize each row to unit vector
            for col in numeric_columns:
                col_data = data_normalized[col]
                norm = np.sqrt((col_data ** 2).sum())
                if norm > 0:
                    data_normalized[col] = col_data / norm

        else:
            raise ValueError(f"Unknown normalization method: {method}")

        self.preprocessing_history.append({
            'step': 'normalize_data',
            'method': method,
            'feature_range': feature_range if method == 'minmax' else None
        })

        return data_normalized

    def smooth_data(self, data: pd.DataFrame, method: str = 'rolling_mean',
                   window: int = 5, **kwargs) -> pd.DataFrame:
        """
        Smooth financial time series data.

        Args:
            data: Input DataFrame
            method: Smoothing method ('rolling_mean', 'exponential', 'savgol')
            window: Window size for smoothing
            **kwargs: Additional parameters for specific methods

        Returns:
            Smoothed DataFrame
        """
        data_smoothed = data.copy()

        for col in data.select_dtypes(include=[np.number]).columns:
            col_data = data[col]

            if method == 'rolling_mean':
                data_smoothed[col] = col_data.rolling(window=window, min_periods=1).mean()

            elif method == 'exponential':
                alpha = kwargs.get('alpha', 2 / (window + 1))
                data_smoothed[col] = col_data.ewm(alpha=alpha).mean()

            elif method == 'savgol':
                polyorder = kwargs.get('polyorder', min(3, window - 1))
                if len(col_data.dropna()) >= window:
                    smoothed_values = savgol_filter(col_data.dropna(), window, polyorder)
                    data_smoothed.loc[col_data.dropna().index, col] = smoothed_values

            else:
                raise ValueError(f"Unknown smoothing method: {method}")

        self.preprocessing_history.append({
            'step': 'smooth_data',
            'method': method,
            'window': window,
            'kwargs': kwargs
        })

        return data_smoothed

    def clean_data(self, data: pd.DataFrame, missing_data_method: str = 'forward_fill',
                   outlier_method: str = 'iqr', outlier_threshold: float = 1.5,
                   winsorize: bool = True, winsorize_range: Tuple[float, float] = (0.05, 0.95)) -> pd.DataFrame:
        """
        Complete data cleaning pipeline.

        This is a convenience method that combines multiple preprocessing steps.

        Args:
            data: Input DataFrame
            missing_data_method: Method for handling missing data
            outlier_method: Method for outlier detection
            outlier_threshold: Threshold for outlier detection
            winsorize: Whether to apply winsorization
            winsorize_range: Winsorization percentile range

        Returns:
            Cleaned DataFrame
        """
        cleaned_data = data.copy()

        # Step 1: Handle missing data
        cleaned_data = self.handle_missing_data(cleaned_data, method=missing_data_method)

        # Step 2: Detect and handle outliers through winsorization
        if winsorize:
            cleaned_data = self.winsorize_data(cleaned_data,
                                             lower_percentile=winsorize_range[0],
                                             upper_percentile=winsorize_range[1])

        return cleaned_data


class TimeSeriesProcessor:
    """
    Specialized processor for financial time series data.
    """

    @staticmethod
    def calculate_returns(prices: pd.DataFrame, method: str = 'simple',
                         periods: int = 1) -> pd.DataFrame:
        """
        Calculate returns from price data.

        Args:
            prices: Price DataFrame
            method: Return calculation method ('simple', 'log', 'percent')
            periods: Number of periods for return calculation

        Returns:
            Returns DataFrame
        """
        if method == 'simple':
            returns = prices.pct_change(periods=periods)
        elif method == 'log':
            returns = np.log(prices / prices.shift(periods))
        elif method == 'percent':
            returns = (prices / prices.shift(periods) - 1) * 100
        else:
            raise ValueError(f"Unknown return method: {method}")

        return returns.dropna()

    @staticmethod
    def calculate_volatility(returns: pd.DataFrame, window: int = 30,
                           method: str = 'rolling') -> pd.DataFrame:
        """
        Calculate rolling volatility measures.

        Args:
            returns: Returns DataFrame
            window: Window size for volatility calculation
            method: Volatility method ('rolling', 'ewm', 'garch')

        Returns:
            Volatility DataFrame
        """
        if method == 'rolling':
            volatility = returns.rolling(window=window).std() * np.sqrt(252)  # Annualized

        elif method == 'ewm':
            # Exponentially weighted moving volatility
            volatility = returns.ewm(span=window).std() * np.sqrt(252)

        elif method == 'realized':
            # Realized volatility using high-frequency concepts
            volatility = np.sqrt(returns.rolling(window=window).apply(lambda x: np.sum(x**2)) * 252)

        else:
            raise ValueError(f"Unknown volatility method: {method}")

        return volatility

    @staticmethod
    def calculate_correlations(returns: pd.DataFrame, window: int = 60,
                             method: str = 'rolling') -> pd.DataFrame:
        """
        Calculate rolling correlation matrices.

        Args:
            returns: Returns DataFrame
            window: Window size for correlation calculation
            method: Correlation method ('rolling', 'ewm', 'dtw')

        Returns:
            Panel of correlation matrices
        """
        if method == 'rolling':
            # Rolling correlation - return as dict of correlation matrices
            correlations = {}
            for i in range(window, len(returns)):
                window_data = returns.iloc[i-window:i]
                correlations[returns.index[i]] = window_data.corr()

        elif method == 'ewm':
            # Exponentially weighted correlation
            correlations = {}
            for i in range(window, len(returns)):
                window_data = returns.iloc[:i]
                ewm_cov = window_data.ewm(span=window).cov().iloc[-len(returns.columns):, :]
                ewm_std = window_data.ewm(span=window).std().iloc[-1]

                # Convert covariance to correlation
                corr_matrix = ewm_cov.div(ewm_std, axis=0).div(ewm_std, axis=1)
                correlations[returns.index[i]] = corr_matrix

        else:
            raise ValueError(f"Unknown correlation method: {method}")

        return correlations

    @staticmethod
    def detect_regime_changes(returns: pd.DataFrame, method: str = 'variance_change',
                            window: int = 60) -> pd.DataFrame:
        """
        Detect regime changes in financial time series.

        Args:
            returns: Returns DataFrame
            method: Detection method ('variance_change', 'mean_change', 'correlation_change')
            window: Window size for change detection

        Returns:
            DataFrame indicating regime change points
        """
        change_points = pd.DataFrame(False, index=returns.index, columns=returns.columns)

        for col in returns.columns:
            col_returns = returns[col].dropna()

            if method == 'variance_change':
                # Detect significant changes in variance
                rolling_var = col_returns.rolling(window=window).var()
                var_changes = np.abs(rolling_var.pct_change()) > 0.5  # 50% variance change
                change_points.loc[var_changes.index[var_changes], col] = True

            elif method == 'mean_change':
                # Detect significant changes in mean
                rolling_mean = col_returns.rolling(window=window).mean()
                rolling_std = col_returns.rolling(window=window).std()
                z_scores = np.abs((rolling_mean - rolling_mean.shift(1)) / rolling_std)
                mean_changes = z_scores > 2  # 2 standard deviation change
                change_points.loc[mean_changes.index[mean_changes], col] = True

            elif method == 'cusum':
                # CUSUM test for structural breaks
                mean_return = col_returns.mean()
                std_return = col_returns.std()
                cumsum = np.cumsum(col_returns - mean_return) / std_return

                # Simple threshold-based detection
                threshold = 3 * np.sqrt(window)
                breaks = np.abs(cumsum) > threshold
                change_points.loc[col_returns.index[breaks], col] = True

        return change_points


class FinancialDataValidator:
    """
    Validator for financial data integrity and consistency.
    """

    @staticmethod
    def validate_price_data(prices: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate price data for common issues.

        Args:
            prices: Price DataFrame (OHLCV format expected)

        Returns:
            Validation report
        """
        report = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'statistics': {}
        }

        # Check for required columns in OHLCV data
        expected_columns = ['Open', 'High', 'Low', 'Close']
        available_columns = [col for col in expected_columns if col in prices.columns]

        if len(available_columns) < len(expected_columns):
            missing = set(expected_columns) - set(available_columns)
            report['warnings'].append(f"Missing expected columns: {missing}")

        # For available OHLC columns, validate relationships
        if all(col in prices.columns for col in ['Open', 'High', 'Low', 'Close']):
            # High should be >= max(Open, Close)
            high_violations = (prices['High'] < np.maximum(prices['Open'], prices['Close']))
            if high_violations.any():
                n_violations = high_violations.sum()
                report['errors'].append(f"High price violations: {n_violations} instances")
                report['valid'] = False

            # Low should be <= min(Open, Close)
            low_violations = (prices['Low'] > np.minimum(prices['Open'], prices['Close']))
            if low_violations.any():
                n_violations = low_violations.sum()
                report['errors'].append(f"Low price violations: {n_violations} instances")
                report['valid'] = False

        # Check for negative prices
        numeric_columns = prices.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            negative_prices = (prices[col] <= 0).sum()
            if negative_prices > 0:
                report['errors'].append(f"Negative/zero prices in {col}: {negative_prices} instances")
                report['valid'] = False

        # Check for extreme price movements (> 50% daily change)
        if 'Close' in prices.columns:
            returns = prices['Close'].pct_change().dropna()
            extreme_returns = (np.abs(returns) > 0.5).sum()
            if extreme_returns > 0:
                report['warnings'].append(f"Extreme daily returns (>50%): {extreme_returns} instances")

        # Check for duplicate timestamps
        if prices.index.duplicated().any():
            duplicates = prices.index.duplicated().sum()
            report['errors'].append(f"Duplicate timestamps: {duplicates} instances")
            report['valid'] = False

        # Calculate basic statistics
        report['statistics'] = {
            'n_observations': len(prices),
            'date_range': {
                'start': prices.index[0] if len(prices) > 0 else None,
                'end': prices.index[-1] if len(prices) > 0 else None
            },
            'missing_data_pct': prices.isna().sum().sum() / (len(prices) * len(prices.columns)) * 100,
            'price_statistics': prices.describe().to_dict() if len(prices) > 0 else {}
        }

        return report

    @staticmethod
    def validate_returns_data(returns: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate returns data for statistical properties.

        Args:
            returns: Returns DataFrame

        Returns:
            Validation report
        """
        report = {
            'valid': True,
            'warnings': [],
            'statistical_tests': {},
            'outliers': {}
        }

        for col in returns.select_dtypes(include=[np.number]).columns:
            col_returns = returns[col].dropna()

            if len(col_returns) == 0:
                continue

            # Test for normality (Jarque-Bera test)
            try:
                jb_stat, jb_pvalue = stats.jarque_bera(col_returns)
                report['statistical_tests'][f'{col}_jarque_bera'] = {
                    'statistic': jb_stat,
                    'pvalue': jb_pvalue,
                    'is_normal': jb_pvalue > 0.05
                }

                if jb_pvalue <= 0.05:
                    report['warnings'].append(f"{col}: Returns not normally distributed (JB p-value: {jb_pvalue:.4f})")

            except Exception:
                pass

            # Test for stationarity (ADF test)
            try:
                from statsmodels.tsa.stattools import adfuller
                adf_result = adfuller(col_returns)
                report['statistical_tests'][f'{col}_adf'] = {
                    'statistic': adf_result[0],
                    'pvalue': adf_result[1],
                    'is_stationary': adf_result[1] <= 0.05
                }

                if adf_result[1] > 0.05:
                    report['warnings'].append(f"{col}: Returns may not be stationary (ADF p-value: {adf_result[1]:.4f})")

            except ImportError:
                pass  # statsmodels not available
            except Exception:
                pass

            # Detect extreme outliers
            q99 = col_returns.quantile(0.99)
            q01 = col_returns.quantile(0.01)
            extreme_outliers = ((col_returns > q99) | (col_returns < q01)).sum()
            report['outliers'][col] = {
                'extreme_count': extreme_outliers,
                'extreme_pct': extreme_outliers / len(col_returns) * 100,
                'q01': q01,
                'q99': q99
            }

            if extreme_outliers / len(col_returns) > 0.05:  # More than 5% extreme outliers
                report['warnings'].append(f"{col}: High number of extreme outliers ({extreme_outliers/len(col_returns)*100:.1f}%)")

        return report

    @staticmethod
    def check_data_consistency(data_dict: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Check consistency across multiple related datasets.

        Args:
            data_dict: Dictionary of dataset name -> DataFrame

        Returns:
            Consistency report
        """
        report = {
            'consistent': True,
            'issues': [],
            'date_alignment': {},
            'correlation_checks': {}
        }

        datasets = list(data_dict.keys())
        if len(datasets) < 2:
            return report

        # Check date alignment
        date_ranges = {}
        for name, data in data_dict.items():
            if hasattr(data, 'index') and isinstance(data.index, pd.DatetimeIndex):
                date_ranges[name] = {
                    'start': data.index[0],
                    'end': data.index[-1],
                    'freq': pd.infer_freq(data.index),
                    'length': len(data)
                }

        report['date_alignment'] = date_ranges

        # Check for significant date misalignments
        start_dates = [info['start'] for info in date_ranges.values() if info['start'] is not None]
        end_dates = [info['end'] for info in date_ranges.values() if info['end'] is not None]

        if len(set(start_dates)) > 1:
            report['issues'].append("Datasets have different start dates")
            report['consistent'] = False

        if len(set(end_dates)) > 1:
            report['issues'].append("Datasets have different end dates")

        # Check correlations between datasets (if numeric)
        numeric_datasets = {name: data for name, data in data_dict.items()
                           if hasattr(data, 'select_dtypes')}

        if len(numeric_datasets) >= 2:
            dataset_names = list(numeric_datasets.keys())
            for i in range(len(dataset_names)):
                for j in range(i + 1, len(dataset_names)):
                    name1, name2 = dataset_names[i], dataset_names[j]
                    data1, data2 = numeric_datasets[name1], numeric_datasets[name2]

                    # Try to find common columns or calculate overall correlation
                    common_index = data1.index.intersection(data2.index)
                    if len(common_index) > 10:  # Need sufficient overlap
                        try:
                            # Calculate correlation between first numeric columns
                            col1 = data1.select_dtypes(include=[np.number]).columns[0]
                            col2 = data2.select_dtypes(include=[np.number]).columns[0]

                            series1 = data1.loc[common_index, col1]
                            series2 = data2.loc[common_index, col2]

                            correlation = series1.corr(series2)
                            report['correlation_checks'][f'{name1}_vs_{name2}'] = correlation

                        except (IndexError, KeyError):
                            pass

        return report


if __name__ == "__main__":
    # Demonstration of data preprocessing utilities
    print("Data Preprocessing Utilities Demonstration")
    print("=" * 60)

    # Generate sample data with various issues
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    n_assets = 3

    # Create sample price data with intentional issues
    sample_data = {}
    for i in range(n_assets):
        prices = 100 * np.exp(np.cumsum(np.random.normal(0.0005, 0.02, len(dates))))

        # Introduce some issues
        if i == 0:
            # Add some missing data
            missing_indices = np.random.choice(len(prices), size=10, replace=False)
            prices[missing_indices] = np.nan

        if i == 1:
            # Add some outliers
            outlier_indices = np.random.choice(len(prices), size=5, replace=False)
            prices[outlier_indices] *= np.random.choice([0.5, 2.0], size=5)

        sample_data[f'Asset_{i+1}'] = prices

    # Create DataFrame
    price_df = pd.DataFrame(sample_data, index=dates)

    print("\n1. Original Data Summary:")
    print("-" * 30)
    print(f"Shape: {price_df.shape}")
    print(f"Missing values per column:")
    print(price_df.isna().sum())

    # Initialize preprocessor
    preprocessor = DataPreprocessor()

    print("\n2. Missing Data Handling:")
    print("-" * 30)

    # Test different missing data methods
    cleaned_forward = preprocessor.handle_missing_data(price_df, method='forward_fill')
    print(f"Forward fill - Missing values after: {cleaned_forward.isna().sum().sum()}")

    cleaned_interp = preprocessor.handle_missing_data(price_df, method='interpolate')
    print(f"Interpolation - Missing values after: {cleaned_interp.isna().sum().sum()}")

    print("\nPreprocessing history:")
    for step in preprocessor.preprocessing_history:
        print(f"  {step}")

    print("\n3. Outlier Detection:")
    print("-" * 25)

    # Detect outliers using different methods
    outliers_iqr = preprocessor.detect_outliers(cleaned_interp, method='iqr', threshold=2.0)
    outliers_zscore = preprocessor.detect_outliers(cleaned_interp, method='zscore', threshold=3.0)

    print(f"IQR method outliers per column:")
    print(outliers_iqr.sum())
    print(f"\nZ-score method outliers per column:")
    print(outliers_zscore.sum())

    # Winsorize data
    winsorized_data = preprocessor.winsorize_data(cleaned_interp, lower_percentile=0.05, upper_percentile=0.95)
    print(f"\nWinsorization completed (5%-95% range)")

    print("\n4. Data Normalization:")
    print("-" * 25)

    # Test different normalization methods
    normalized_standard = preprocessor.normalize_data(winsorized_data, method='standard')
    print(f"Standard normalization - Mean: {normalized_standard.mean().round(3).tolist()}")
    print(f"Standard normalization - Std: {normalized_standard.std().round(3).tolist()}")

    normalized_robust = preprocessor.normalize_data(winsorized_data, method='robust')
    print(f"Robust normalization - Median: {normalized_robust.median().round(3).tolist()}")

    print("\n5. Data Smoothing:")
    print("-" * 20)

    # Apply smoothing
    smoothed_rolling = preprocessor.smooth_data(winsorized_data, method='rolling_mean', window=7)
    smoothed_exp = preprocessor.smooth_data(winsorized_data, method='exponential', window=10)

    print(f"Rolling mean smoothing applied (window=7)")
    print(f"Exponential smoothing applied (window=10)")

    # Compare volatility before and after smoothing
    original_vol = winsorized_data.std()
    smoothed_vol = smoothed_rolling.std()
    print(f"\nVolatility reduction:")
    for col in original_vol.index:
        reduction = (1 - smoothed_vol[col] / original_vol[col]) * 100
        print(f"  {col}: {reduction:.1f}% reduction")

    print("\n6. Time Series Processing:")
    print("-" * 30)

    # Calculate returns
    ts_processor = TimeSeriesProcessor()

    simple_returns = ts_processor.calculate_returns(winsorized_data, method='simple')
    log_returns = ts_processor.calculate_returns(winsorized_data, method='log')

    print(f"Returns calculated:")
    print(f"  Simple returns shape: {simple_returns.shape}")
    print(f"  Log returns shape: {log_returns.shape}")

    print(f"\nReturn statistics (annualized):")
    annual_returns = simple_returns.mean() * 252
    annual_vol = simple_returns.std() * np.sqrt(252)
    for col in simple_returns.columns:
        print(f"  {col}: Return={annual_returns[col]:.2%}, Vol={annual_vol[col]:.2%}")

    # Calculate rolling volatility
    rolling_vol = ts_processor.calculate_volatility(simple_returns, window=30, method='rolling')
    print(f"\nRolling volatility calculated (30-day window)")

    # Calculate correlations
    rolling_corr = ts_processor.calculate_correlations(simple_returns, window=60, method='rolling')
    print(f"Rolling correlations calculated: {len(rolling_corr)} time points")

    # Detect regime changes
    regime_changes = ts_processor.detect_regime_changes(simple_returns, method='variance_change', window=30)
    print(f"\nRegime changes detected:")
    print(regime_changes.sum())

    print("\n7. Data Validation:")
    print("-" * 20)

    # Create mock OHLCV data for validation
    ohlcv_data = pd.DataFrame({
        'Open': winsorized_data['Asset_1'],
        'High': winsorized_data['Asset_1'] * 1.02,
        'Low': winsorized_data['Asset_1'] * 0.98,
        'Close': winsorized_data['Asset_1'] * np.random.uniform(0.99, 1.01, len(winsorized_data)),
        'Volume': np.random.lognormal(15, 0.5, len(winsorized_data))
    })

    # Add some intentional OHLC violations for testing
    ohlcv_data.loc[ohlcv_data.index[10], 'High'] = ohlcv_data.loc[ohlcv_data.index[10], 'Low'] - 1
    ohlcv_data.loc[ohlcv_data.index[20], 'Close'] = -5  # Negative price

    validator = FinancialDataValidator()
    price_validation = validator.validate_price_data(ohlcv_data)

    print(f"Price data validation:")
    print(f"  Valid: {price_validation['valid']}")
    print(f"  Errors: {len(price_validation['errors'])}")
    for error in price_validation['errors']:
        print(f"    - {error}")
    print(f"  Warnings: {len(price_validation['warnings'])}")
    for warning in price_validation['warnings']:
        print(f"    - {warning}")

    # Validate returns data
    returns_validation = validator.validate_returns_data(simple_returns)
    print(f"\nReturns data validation:")
    print(f"  Warnings: {len(returns_validation['warnings'])}")
    for warning in returns_validation['warnings']:
        print(f"    - {warning}")

    # Check data consistency
    datasets = {
        'prices': winsorized_data,
        'returns': simple_returns,
        'volatility': rolling_vol.iloc[30:]  # Skip NaN values
    }

    consistency_report = validator.check_data_consistency(datasets)
    print(f"\nData consistency check:")
    print(f"  Consistent: {consistency_report['consistent']}")
    if consistency_report['issues']:
        print(f"  Issues:")
        for issue in consistency_report['issues']:
            print(f"    - {issue}")

    print(f"\nData preprocessing demonstration completed successfully!")
    print(f"Processed {len(price_df)} days of data for {len(price_df.columns)} assets")
    print(f"Applied {len(preprocessor.preprocessing_history)} preprocessing steps")