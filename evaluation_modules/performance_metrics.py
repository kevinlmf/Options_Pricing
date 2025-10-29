"""
Performance Metrics and Analysis Module

Implements comprehensive performance evaluation tools for derivatives and risk models,
including risk-adjusted returns, benchmark comparisons, and attribution analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from abc import ABC, abstractmethod
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings


class PerformanceAnalyzer:
    """
    Comprehensive performance analysis for financial models and strategies.

    Calculates various performance metrics including returns, volatility,
    Sharpe ratio, maximum drawdown, and other risk-adjusted measures.
    """

    def __init__(self, returns: Union[np.ndarray, pd.Series],
                 risk_free_rate: float = 0.02):
        """
        Initialize performance analyzer.

        Parameters:
        -----------
        returns : array-like
            Time series of returns (decimal format, e.g., 0.01 for 1%)
        risk_free_rate : float
            Annual risk-free rate (default: 2%)
        """
        self.returns = np.asarray(returns) if not isinstance(returns, pd.Series) else returns
        self.risk_free_rate = risk_free_rate

        # Annualization factors
        self.trading_days = 252
        self.trading_weeks = 52
        self.trading_months = 12

    def total_return(self) -> float:
        """Calculate total cumulative return."""
        return (1 + self.returns).prod() - 1

    def annualized_return(self, frequency: str = 'daily') -> float:
        """
        Calculate annualized return.

        Parameters:
        -----------
        frequency : str
            Frequency of returns ('daily', 'weekly', 'monthly')
        """
        total_ret = self.total_return()
        n_periods = len(self.returns)

        freq_map = {
            'daily': self.trading_days,
            'weekly': self.trading_weeks,
            'monthly': self.trading_months
        }

        periods_per_year = freq_map.get(frequency, self.trading_days)
        return (1 + total_ret) ** (periods_per_year / n_periods) - 1

    def volatility(self, frequency: str = 'daily', annualized: bool = True) -> float:
        """
        Calculate volatility (standard deviation of returns).

        Parameters:
        -----------
        frequency : str
            Frequency of returns
        annualized : bool
            Whether to annualize the volatility
        """
        vol = np.std(self.returns, ddof=1)

        if annualized:
            freq_map = {
                'daily': self.trading_days,
                'weekly': self.trading_weeks,
                'monthly': self.trading_months
            }
            periods_per_year = freq_map.get(frequency, self.trading_days)
            vol *= np.sqrt(periods_per_year)

        return vol

    def sharpe_ratio(self, frequency: str = 'daily') -> float:
        """Calculate Sharpe ratio (risk-adjusted return)."""
        excess_return = self.annualized_return(frequency) - self.risk_free_rate
        vol = self.volatility(frequency, annualized=True)

        return excess_return / vol if vol != 0 else 0.0

    def sortino_ratio(self, frequency: str = 'daily') -> float:
        """Calculate Sortino ratio (downside risk-adjusted return)."""
        excess_return = self.annualized_return(frequency) - self.risk_free_rate
        downside_returns = self.returns[self.returns < 0]

        if len(downside_returns) == 0:
            return np.inf

        downside_vol = np.std(downside_returns, ddof=1)
        freq_map = {
            'daily': self.trading_days,
            'weekly': self.trading_weeks,
            'monthly': self.trading_months
        }
        periods_per_year = freq_map.get(frequency, self.trading_days)
        downside_vol *= np.sqrt(periods_per_year)

        return excess_return / downside_vol if downside_vol != 0 else np.inf

    def maximum_drawdown(self) -> Dict[str, float]:
        """
        Calculate maximum drawdown and related metrics.

        Returns:
        --------
        dict : Dictionary containing max drawdown, peak, trough, and duration
        """
        cumulative = (1 + self.returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max

        max_dd = drawdown.min()
        max_dd_idx = drawdown.idxmin() if isinstance(drawdown, pd.Series) else np.argmin(drawdown)

        # Find peak before max drawdown
        if isinstance(drawdown, pd.Series):
            peak_idx = running_max.loc[:max_dd_idx].idxmax()
            peak_value = running_max.loc[peak_idx]
            trough_value = cumulative.loc[max_dd_idx]
        else:
            peak_idx = np.argmax(running_max[:max_dd_idx + 1])
            peak_value = running_max[peak_idx]
            trough_value = cumulative[max_dd_idx]

        return {
            'max_drawdown': max_dd,
            'peak_value': peak_value,
            'trough_value': trough_value,
            'peak_date': peak_idx,
            'trough_date': max_dd_idx,
            'duration': max_dd_idx - peak_idx if isinstance(max_dd_idx, int) else
                       (max_dd_idx - peak_idx).days if hasattr(max_dd_idx, 'days') else None
        }

    def var_estimate(self, confidence_level: float = 0.05) -> float:
        """Estimate Value at Risk from return distribution."""
        return np.percentile(self.returns, confidence_level * 100)

    def cvar_estimate(self, confidence_level: float = 0.05) -> float:
        """Estimate Conditional Value at Risk (Expected Shortfall)."""
        var = self.var_estimate(confidence_level)
        return self.returns[self.returns <= var].mean()

    def calmar_ratio(self, frequency: str = 'daily') -> float:
        """Calculate Calmar ratio (annualized return / max drawdown)."""
        ann_return = self.annualized_return(frequency)
        max_dd = abs(self.maximum_drawdown()['max_drawdown'])

        return ann_return / max_dd if max_dd != 0 else np.inf

    def performance_summary(self, frequency: str = 'daily') -> Dict[str, float]:
        """Generate comprehensive performance summary."""
        max_dd_info = self.maximum_drawdown()

        return {
            'Total Return': self.total_return(),
            'Annualized Return': self.annualized_return(frequency),
            'Volatility': self.volatility(frequency),
            'Sharpe Ratio': self.sharpe_ratio(frequency),
            'Sortino Ratio': self.sortino_ratio(frequency),
            'Maximum Drawdown': max_dd_info['max_drawdown'],
            'Calmar Ratio': self.calmar_ratio(frequency),
            'VaR (5%)': self.var_estimate(),
            'CVaR (5%)': self.cvar_estimate(),
            'Skewness': stats.skew(self.returns),
            'Kurtosis': stats.kurtosis(self.returns),
            'Best Period': np.max(self.returns),
            'Worst Period': np.min(self.returns),
            'Hit Rate': np.mean(self.returns > 0),
            'Average Win': np.mean(self.returns[self.returns > 0]) if np.any(self.returns > 0) else 0,
            'Average Loss': np.mean(self.returns[self.returns < 0]) if np.any(self.returns < 0) else 0
        }


class BenchmarkComparison:
    """
    Tools for comparing strategy performance against benchmarks.

    Includes relative performance metrics, tracking error, information ratio,
    and attribution analysis.
    """

    def __init__(self, strategy_returns: Union[np.ndarray, pd.Series],
                 benchmark_returns: Union[np.ndarray, pd.Series]):
        """
        Initialize benchmark comparison.

        Parameters:
        -----------
        strategy_returns : array-like
            Strategy returns
        benchmark_returns : array-like
            Benchmark returns
        """
        self.strategy_returns = np.asarray(strategy_returns)
        self.benchmark_returns = np.asarray(benchmark_returns)

        if len(self.strategy_returns) != len(self.benchmark_returns):
            raise ValueError("Strategy and benchmark returns must have same length")

        self.excess_returns = self.strategy_returns - self.benchmark_returns

    def alpha_beta(self) -> Tuple[float, float]:
        """
        Calculate Jensen's alpha and beta using linear regression.

        Returns:
        --------
        tuple : (alpha, beta)
        """
        # Add constant term for intercept
        X = np.column_stack([np.ones(len(self.benchmark_returns)), self.benchmark_returns])

        try:
            # Using normal equation: (X'X)^(-1)X'y
            coeffs = np.linalg.inv(X.T @ X) @ X.T @ self.strategy_returns
            alpha, beta = coeffs[0], coeffs[1]
        except np.linalg.LinAlgError:
            # Fallback to pseudoinverse
            coeffs = np.linalg.pinv(X) @ self.strategy_returns
            alpha, beta = coeffs[0], coeffs[1]

        return alpha, beta

    def tracking_error(self, annualized: bool = True) -> float:
        """
        Calculate tracking error (volatility of excess returns).

        Parameters:
        -----------
        annualized : bool
            Whether to annualize tracking error
        """
        te = np.std(self.excess_returns, ddof=1)

        if annualized:
            te *= np.sqrt(252)  # Assuming daily data

        return te

    def information_ratio(self) -> float:
        """Calculate information ratio (excess return / tracking error)."""
        mean_excess = np.mean(self.excess_returns)
        te = self.tracking_error(annualized=False)

        # Annualize
        ann_excess = mean_excess * 252
        ann_te = te * np.sqrt(252)

        return ann_excess / ann_te if ann_te != 0 else 0.0

    def up_down_capture(self) -> Dict[str, float]:
        """
        Calculate upside and downside capture ratios.

        Returns:
        --------
        dict : Dictionary with upside_capture and downside_capture
        """
        # Identify up and down periods
        up_periods = self.benchmark_returns > 0
        down_periods = self.benchmark_returns < 0

        if not np.any(up_periods) or not np.any(down_periods):
            warnings.warn("Insufficient up/down periods for capture analysis")
            return {'upside_capture': np.nan, 'downside_capture': np.nan}

        # Calculate capture ratios
        strategy_up = np.mean(self.strategy_returns[up_periods])
        benchmark_up = np.mean(self.benchmark_returns[up_periods])
        upside_capture = strategy_up / benchmark_up if benchmark_up != 0 else np.nan

        strategy_down = np.mean(self.strategy_returns[down_periods])
        benchmark_down = np.mean(self.benchmark_returns[down_periods])
        downside_capture = strategy_down / benchmark_down if benchmark_down != 0 else np.nan

        return {
            'upside_capture': upside_capture,
            'downside_capture': downside_capture
        }

    def rolling_correlation(self, window: int = 60) -> np.ndarray:
        """
        Calculate rolling correlation between strategy and benchmark.

        Parameters:
        -----------
        window : int
            Rolling window size
        """
        if len(self.strategy_returns) < window:
            raise ValueError(f"Insufficient data for window size {window}")

        correlations = []
        for i in range(window - 1, len(self.strategy_returns)):
            start_idx = i - window + 1
            corr = np.corrcoef(
                self.strategy_returns[start_idx:i+1],
                self.benchmark_returns[start_idx:i+1]
            )[0, 1]
            correlations.append(corr)

        return np.array(correlations)

    def comparison_summary(self) -> Dict[str, float]:
        """Generate comprehensive benchmark comparison summary."""
        alpha, beta = self.alpha_beta()
        capture_ratios = self.up_down_capture()

        return {
            'Alpha': alpha,
            'Beta': beta,
            'Correlation': np.corrcoef(self.strategy_returns, self.benchmark_returns)[0, 1],
            'Tracking Error': self.tracking_error(),
            'Information Ratio': self.information_ratio(),
            'Upside Capture': capture_ratios['upside_capture'],
            'Downside Capture': capture_ratios['downside_capture'],
            'Excess Return (Ann.)': np.mean(self.excess_returns) * 252,
            'Excess Volatility': np.std(self.strategy_returns, ddof=1) * np.sqrt(252) -
                               np.std(self.benchmark_returns, ddof=1) * np.sqrt(252)
        }


class RiskAdjustedMetrics:
    """
    Advanced risk-adjusted performance metrics.

    Implements sophisticated measures like Treynor ratio, M2 measure,
    and multi-factor attribution models.
    """

    def __init__(self, returns: Union[np.ndarray, pd.Series],
                 market_returns: Optional[Union[np.ndarray, pd.Series]] = None,
                 risk_free_rate: float = 0.02):
        """
        Initialize risk-adjusted metrics calculator.

        Parameters:
        -----------
        returns : array-like
            Portfolio/strategy returns
        market_returns : array-like, optional
            Market benchmark returns
        risk_free_rate : float
            Risk-free rate
        """
        self.returns = np.asarray(returns)
        self.market_returns = np.asarray(market_returns) if market_returns is not None else None
        self.risk_free_rate = risk_free_rate

    def treynor_ratio(self) -> float:
        """
        Calculate Treynor ratio (excess return per unit of systematic risk).

        Requires market returns to calculate beta.
        """
        if self.market_returns is None:
            raise ValueError("Market returns required for Treynor ratio")

        # Calculate beta
        covariance = np.cov(self.returns, self.market_returns)[0, 1]
        market_variance = np.var(self.market_returns, ddof=1)
        beta = covariance / market_variance if market_variance != 0 else 0

        # Annual excess return
        ann_return = np.mean(self.returns) * 252
        excess_return = ann_return - self.risk_free_rate

        return excess_return / beta if beta != 0 else np.inf

    def modigliani_ratio(self) -> float:
        """
        Calculate M² (Modigliani-Modigliani) measure.

        Risk-adjusted return scaled to market volatility.
        """
        if self.market_returns is None:
            raise ValueError("Market returns required for M² ratio")

        # Calculate Sharpe ratios
        portfolio_sharpe = self._sharpe_ratio(self.returns)
        market_sharpe = self._sharpe_ratio(self.market_returns)

        # Market volatility
        market_vol = np.std(self.market_returns, ddof=1) * np.sqrt(252)

        # M² = (Portfolio Sharpe - Market Sharpe) * Market Vol
        return (portfolio_sharpe - market_sharpe) * market_vol

    def _sharpe_ratio(self, returns: np.ndarray) -> float:
        """Helper method to calculate Sharpe ratio."""
        excess_return = np.mean(returns) * 252 - self.risk_free_rate
        volatility = np.std(returns, ddof=1) * np.sqrt(252)
        return excess_return / volatility if volatility != 0 else 0

    def omega_ratio(self, threshold: float = 0.0) -> float:
        """
        Calculate Omega ratio.

        Ratio of probability-weighted gains to losses relative to threshold.

        Parameters:
        -----------
        threshold : float
            Threshold return (default: 0)
        """
        excess_returns = self.returns - threshold / 252  # Daily threshold

        gains = excess_returns[excess_returns > 0]
        losses = excess_returns[excess_returns <= 0]

        gain_sum = np.sum(gains) if len(gains) > 0 else 0
        loss_sum = abs(np.sum(losses)) if len(losses) > 0 else 1e-10  # Avoid division by zero

        return gain_sum / loss_sum

    def kappa_ratio(self, n: int = 3) -> float:
        """
        Calculate Kappa ratio (generalized Sortino ratio).

        Parameters:
        -----------
        n : int
            Moment order (default: 3 for skewness-adjusted)
        """
        negative_returns = self.returns[self.returns < 0]

        if len(negative_returns) == 0:
            return np.inf

        # Calculate n-th lower partial moment
        lpm_n = np.mean(np.power(np.abs(negative_returns), n))

        # Annualized excess return
        ann_return = np.mean(self.returns) * 252 - self.risk_free_rate

        return ann_return / (np.power(lpm_n, 1/n) * np.sqrt(252)) if lpm_n > 0 else np.inf

    def sterling_ratio(self, lookback_periods: int = 36) -> float:
        """
        Calculate Sterling ratio (return / average drawdown).

        Parameters:
        -----------
        lookback_periods : int
            Number of periods for drawdown calculation
        """
        if len(self.returns) < lookback_periods:
            lookback_periods = len(self.returns)

        # Calculate drawdowns over rolling windows
        cumulative = (1 + self.returns).cumprod()
        drawdowns = []

        for i in range(lookback_periods - 1, len(cumulative)):
            window = cumulative[i - lookback_periods + 1:i + 1]
            running_max = np.maximum.accumulate(window)
            dd = (window - running_max) / running_max
            max_dd = np.min(dd)
            if max_dd < 0:
                drawdowns.append(abs(max_dd))

        if not drawdowns:
            return np.inf

        avg_drawdown = np.mean(drawdowns)
        ann_return = np.mean(self.returns) * 252

        return ann_return / avg_drawdown if avg_drawdown != 0 else np.inf

    def comprehensive_metrics(self) -> Dict[str, float]:
        """Generate comprehensive risk-adjusted metrics summary."""
        metrics = {
            'Sharpe Ratio': self._sharpe_ratio(self.returns),
            'Omega Ratio': self.omega_ratio(),
            'Kappa Ratio (3)': self.kappa_ratio(3),
            'Sterling Ratio': self.sterling_ratio()
        }

        if self.market_returns is not None:
            metrics.update({
                'Treynor Ratio': self.treynor_ratio(),
                'M² Measure': self.modigliani_ratio()
            })

        return metrics


# Model-specific evaluation metrics
class ModelAccuracyMetrics:
    """
    Specialized metrics for evaluating model prediction accuracy.

    Includes pricing error analysis, hedging effectiveness,
    and calibration quality measures.
    """

    @staticmethod
    def pricing_errors(predicted_prices: np.ndarray,
                      actual_prices: np.ndarray) -> Dict[str, float]:
        """
        Calculate comprehensive pricing error metrics.

        Parameters:
        -----------
        predicted_prices : array-like
            Model-predicted prices
        actual_prices : array-like
            Actual market prices
        """
        errors = predicted_prices - actual_prices
        relative_errors = errors / actual_prices

        return {
            'MAE': mean_absolute_error(actual_prices, predicted_prices),
            'RMSE': np.sqrt(mean_squared_error(actual_prices, predicted_prices)),
            'MAPE': np.mean(np.abs(relative_errors)) * 100,
            'Max Error': np.max(np.abs(errors)),
            'Mean Bias': np.mean(errors),
            'Error Std': np.std(errors, ddof=1),
            'R²': 1 - np.sum(errors**2) / np.sum((actual_prices - np.mean(actual_prices))**2)
        }

    @staticmethod
    def hedging_effectiveness(hedged_pnl: np.ndarray,
                            unhedged_pnl: np.ndarray) -> Dict[str, float]:
        """
        Evaluate hedging effectiveness.

        Parameters:
        -----------
        hedged_pnl : array-like
            P&L of hedged portfolio
        unhedged_pnl : array-like
            P&L of unhedged portfolio
        """
        hedged_var = np.var(hedged_pnl, ddof=1)
        unhedged_var = np.var(unhedged_pnl, ddof=1)

        variance_reduction = (unhedged_var - hedged_var) / unhedged_var if unhedged_var != 0 else 0

        return {
            'Variance Reduction': variance_reduction,
            'Hedged Volatility': np.std(hedged_pnl, ddof=1),
            'Unhedged Volatility': np.std(unhedged_pnl, ddof=1),
            'Hedge Ratio': 1 - hedged_var / unhedged_var if unhedged_var != 0 else 1,
            'Hedged Sharpe': np.mean(hedged_pnl) / np.std(hedged_pnl, ddof=1) if np.std(hedged_pnl) != 0 else 0
        }