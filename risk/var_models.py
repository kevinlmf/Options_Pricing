"""
Value at Risk (VaR) Models

Implementation of various VaR calculation methods.
Follows the mathematical framework: Pure Math → Applied Math → Risk Management

Mathematical Foundation:
- Pure Math: Quantile functions, tail distributions, probability theory
- Applied Math: Statistical estimation, Monte Carlo methods, portfolio theory
- Risk Management: VaR calculation, backtesting, model validation
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Union, Optional, Tuple
from abc import ABC, abstractmethod
from scipy import stats
from scipy.optimize import minimize_scalar
import warnings


class VarCalculator(ABC):
    """Abstract base class for VaR calculators"""

    def __init__(self, confidence_level: float = 0.05):
        """
        Initialize VaR calculator.

        Args:
            confidence_level: VaR confidence level (e.g., 0.05 for 95% VaR)
        """
        if not 0 < confidence_level < 1:
            raise ValueError("Confidence level must be between 0 and 1")
        self.confidence_level = confidence_level
        self.alpha = confidence_level

    @abstractmethod
    def calculate_var(self, returns: Union[np.ndarray, pd.Series],
                     portfolio_value: float = 1.0) -> float:
        """Calculate VaR for given returns"""
        pass

    @abstractmethod
    def calculate_component_var(self, returns: np.ndarray, weights: np.ndarray,
                              portfolio_value: float = 1.0) -> np.ndarray:
        """Calculate component VaR contributions"""
        pass

    def backtest_var(self, returns: Union[np.ndarray, pd.Series],
                    var_estimates: Union[np.ndarray, pd.Series]) -> Dict[str, float]:
        """
        Backtest VaR model using Kupiec test and other metrics.

        Args:
            returns: Actual portfolio returns
            var_estimates: VaR estimates for each period

        Returns:
            Dictionary with backtesting statistics
        """
        returns = np.array(returns)
        var_estimates = np.array(var_estimates)

        if len(returns) != len(var_estimates):
            raise ValueError("Returns and VaR estimates must have same length")

        # Count violations (returns worse than VaR)
        violations = returns < -var_estimates
        n_violations = np.sum(violations)
        n_observations = len(returns)
        violation_rate = n_violations / n_observations

        # Kupiec test for unconditional coverage
        expected_violations = self.alpha * n_observations
        if n_violations == 0:
            kupiec_stat = -2 * np.log((1 - self.alpha) ** n_observations)
        elif n_violations == n_observations:
            kupiec_stat = -2 * np.log(self.alpha ** n_observations)
        else:
            kupiec_stat = -2 * (
                expected_violations * np.log(self.alpha) +
                (n_observations - expected_violations) * np.log(1 - self.alpha) -
                n_violations * np.log(violation_rate) -
                (n_observations - n_violations) * np.log(1 - violation_rate)
            )

        # P-value (chi-squared with 1 degree of freedom)
        kupiec_pvalue = 1 - stats.chi2.cdf(kupiec_stat, 1)

        # Additional metrics
        if n_violations > 0:
            avg_violation_size = np.mean(returns[violations] + var_estimates[violations])
            max_violation_size = np.min(returns[violations] + var_estimates[violations])
        else:
            avg_violation_size = 0
            max_violation_size = 0

        return {
            'violation_rate': violation_rate,
            'expected_violation_rate': self.alpha,
            'n_violations': n_violations,
            'n_observations': n_observations,
            'kupiec_statistic': kupiec_stat,
            'kupiec_pvalue': kupiec_pvalue,
            'test_passed': kupiec_pvalue > 0.05,  # 95% confidence
            'avg_violation_size': avg_violation_size,
            'max_violation_size': max_violation_size
        }


class HistoricalVaR(VarCalculator):
    """
    Historical Simulation VaR

    Uses historical returns to estimate VaR via empirical quantiles.
    Non-parametric approach that preserves actual distribution characteristics.
    """

    def __init__(self, confidence_level: float = 0.05, window_size: Optional[int] = None):
        """
        Initialize Historical VaR calculator.

        Args:
            confidence_level: VaR confidence level
            window_size: Rolling window size (None for expanding window)
        """
        super().__init__(confidence_level)
        self.window_size = window_size

    def calculate_var(self, returns: Union[np.ndarray, pd.Series],
                     portfolio_value: float = 1.0) -> float:
        """Calculate Historical VaR"""
        returns = np.array(returns)

        if len(returns) == 0:
            raise ValueError("Returns array cannot be empty")

        # Use rolling window if specified
        if self.window_size is not None and len(returns) > self.window_size:
            returns = returns[-self.window_size:]

        # Calculate empirical quantile
        var_quantile = np.percentile(returns, self.alpha * 100)

        return -var_quantile * portfolio_value

    def calculate_component_var(self, returns: np.ndarray, weights: np.ndarray,
                              portfolio_value: float = 1.0) -> np.ndarray:
        """Calculate component VaR using historical simulation"""
        if returns.ndim != 2:
            raise ValueError("Returns must be 2D array (assets x time)")

        n_assets = returns.shape[0]
        if len(weights) != n_assets:
            raise ValueError("Weights length must match number of assets")

        # Portfolio returns
        portfolio_returns = np.dot(weights, returns)

        # Use rolling window if specified
        if self.window_size is not None and returns.shape[1] > self.window_size:
            returns = returns[:, -self.window_size:]
            portfolio_returns = portfolio_returns[-self.window_size:]

        # Find VaR scenario
        var_quantile = np.percentile(portfolio_returns, self.alpha * 100)
        var_scenario_idx = np.argmin(np.abs(portfolio_returns - var_quantile))

        # Component contributions at VaR scenario
        scenario_returns = returns[:, var_scenario_idx]
        component_contributions = weights * scenario_returns * portfolio_value

        return -component_contributions

    def rolling_var(self, returns: Union[np.ndarray, pd.Series],
                   window_size: int, portfolio_value: float = 1.0) -> np.ndarray:
        """Calculate rolling Historical VaR"""
        returns = np.array(returns)
        n_obs = len(returns)

        if window_size >= n_obs:
            raise ValueError("Window size must be less than number of observations")

        var_estimates = np.full(n_obs, np.nan)

        for i in range(window_size, n_obs):
            window_returns = returns[i-window_size:i]
            var_estimates[i] = self.calculate_var(window_returns, portfolio_value)

        return var_estimates


class ParametricVaR(VarCalculator):
    """
    Parametric VaR (Variance-Covariance Method)

    Assumes returns follow a normal distribution and uses analytical formulas.
    Fast computation but may underestimate tail risk for non-normal distributions.
    """

    def __init__(self, confidence_level: float = 0.05, distribution: str = 'normal'):
        """
        Initialize Parametric VaR calculator.

        Args:
            confidence_level: VaR confidence level
            distribution: Distribution assumption ('normal', 't', 'skew_t')
        """
        super().__init__(confidence_level)
        self.distribution = distribution.lower()

        if self.distribution not in ['normal', 't', 'skew_t']:
            raise ValueError("Distribution must be 'normal', 't', or 'skew_t'")

    def calculate_var(self, returns: Union[np.ndarray, pd.Series],
                     portfolio_value: float = 1.0) -> float:
        """Calculate Parametric VaR"""
        returns = np.array(returns)

        if len(returns) == 0:
            raise ValueError("Returns array cannot be empty")

        # Estimate distribution parameters
        mean_return = np.mean(returns)
        std_return = np.std(returns, ddof=1)

        # Calculate VaR based on distribution assumption
        if self.distribution == 'normal':
            var_quantile = stats.norm.ppf(self.alpha, mean_return, std_return)

        elif self.distribution == 't':
            # Fit t-distribution
            df, loc, scale = stats.t.fit(returns)
            var_quantile = stats.t.ppf(self.alpha, df, loc, scale)

        elif self.distribution == 'skew_t':
            # Fit skewed t-distribution
            try:
                from scipy.stats import skewt
                params = skewt.fit(returns)
                var_quantile = skewt.ppf(self.alpha, *params)
            except ImportError:
                warnings.warn("Skewed t-distribution not available, using t-distribution")
                df, loc, scale = stats.t.fit(returns)
                var_quantile = stats.t.ppf(self.alpha, df, loc, scale)

        return -var_quantile * portfolio_value

    def calculate_component_var(self, returns: np.ndarray, weights: np.ndarray,
                              portfolio_value: float = 1.0) -> np.ndarray:
        """Calculate component VaR using delta-normal approach"""
        if returns.ndim != 2:
            raise ValueError("Returns must be 2D array (assets x time)")

        n_assets = returns.shape[0]
        if len(weights) != n_assets:
            raise ValueError("Weights length must match number of assets")

        # Calculate covariance matrix
        cov_matrix = np.cov(returns)

        # Portfolio variance
        portfolio_var = np.dot(weights, np.dot(cov_matrix, weights))
        portfolio_std = np.sqrt(portfolio_var)

        # VaR multiplier based on distribution
        if self.distribution == 'normal':
            var_multiplier = stats.norm.ppf(1 - self.alpha)
        elif self.distribution == 't':
            # Use average degrees of freedom across assets
            avg_df = np.mean([stats.t.fit(returns[i, :])[0] for i in range(n_assets)])
            var_multiplier = stats.t.ppf(1 - self.alpha, avg_df)
        else:
            var_multiplier = stats.norm.ppf(1 - self.alpha)  # Fallback

        # Component VaR = (weight * marginal VaR)
        marginal_var = np.dot(cov_matrix, weights) / portfolio_std * var_multiplier
        component_var = weights * marginal_var * portfolio_value

        return component_var

    def calculate_marginal_var(self, returns: np.ndarray, weights: np.ndarray,
                             portfolio_value: float = 1.0) -> np.ndarray:
        """Calculate marginal VaR contributions"""
        if returns.ndim != 2:
            raise ValueError("Returns must be 2D array (assets x time)")

        cov_matrix = np.cov(returns)
        portfolio_var = np.dot(weights, np.dot(cov_matrix, weights))
        portfolio_std = np.sqrt(portfolio_var)

        if self.distribution == 'normal':
            var_multiplier = stats.norm.ppf(1 - self.alpha)
        else:
            var_multiplier = stats.norm.ppf(1 - self.alpha)  # Simplified

        marginal_var = np.dot(cov_matrix, weights) / portfolio_std * var_multiplier * portfolio_value

        return marginal_var


class MonteCarloVaR(VarCalculator):
    """
    Monte Carlo VaR

    Uses Monte Carlo simulation to generate portfolio return scenarios
    and calculates VaR from the simulated distribution.
    """

    def __init__(self, confidence_level: float = 0.05, n_simulations: int = 10000,
                 random_seed: Optional[int] = None):
        """
        Initialize Monte Carlo VaR calculator.

        Args:
            confidence_level: VaR confidence level
            n_simulations: Number of Monte Carlo simulations
            random_seed: Random seed for reproducibility
        """
        super().__init__(confidence_level)
        self.n_simulations = n_simulations
        self.random_seed = random_seed

    def calculate_var(self, returns: Union[np.ndarray, pd.Series],
                     portfolio_value: float = 1.0) -> float:
        """
        Calculate Monte Carlo VaR using bootstrap resampling
        """
        returns = np.array(returns)

        if len(returns) == 0:
            raise ValueError("Returns array cannot be empty")

        if self.random_seed is not None:
            np.random.seed(self.random_seed)

        # Bootstrap resampling
        simulated_returns = np.random.choice(returns, size=self.n_simulations, replace=True)

        # Calculate VaR from simulated distribution
        var_quantile = np.percentile(simulated_returns, self.alpha * 100)

        return -var_quantile * portfolio_value

    def calculate_component_var(self, returns: np.ndarray, weights: np.ndarray,
                              portfolio_value: float = 1.0) -> np.ndarray:
        """Calculate component VaR using Monte Carlo simulation"""
        if returns.ndim != 2:
            raise ValueError("Returns must be 2D array (assets x time)")

        n_assets, n_periods = returns.shape
        if len(weights) != n_assets:
            raise ValueError("Weights length must match number of assets")

        if self.random_seed is not None:
            np.random.seed(self.random_seed)

        # Generate scenarios by resampling historical periods
        scenario_indices = np.random.choice(n_periods, size=self.n_simulations, replace=True)
        simulated_returns = returns[:, scenario_indices]

        # Calculate portfolio returns for each scenario
        portfolio_returns = np.dot(weights, simulated_returns)

        # Find VaR scenario
        var_quantile = np.percentile(portfolio_returns, self.alpha * 100)
        var_scenario_idx = np.argmin(np.abs(portfolio_returns - var_quantile))

        # Component contributions at VaR scenario
        scenario_returns = simulated_returns[:, var_scenario_idx]
        component_contributions = weights * scenario_returns * portfolio_value

        return -component_contributions

    def simulate_portfolio_returns(self, returns: np.ndarray, weights: np.ndarray,
                                 method: str = 'bootstrap') -> np.ndarray:
        """
        Simulate portfolio returns using different methods.

        Args:
            returns: Asset returns matrix (assets x time)
            weights: Portfolio weights
            method: Simulation method ('bootstrap', 'parametric', 'cholesky')

        Returns:
            Simulated portfolio returns
        """
        if returns.ndim != 2:
            raise ValueError("Returns must be 2D array (assets x time)")

        n_assets = returns.shape[0]
        if len(weights) != n_assets:
            raise ValueError("Weights length must match number of assets")

        if self.random_seed is not None:
            np.random.seed(self.random_seed)

        if method == 'bootstrap':
            # Bootstrap resampling
            scenario_indices = np.random.choice(returns.shape[1], size=self.n_simulations, replace=True)
            simulated_asset_returns = returns[:, scenario_indices]

        elif method == 'parametric':
            # Parametric simulation assuming multivariate normal
            mean_returns = np.mean(returns, axis=1)
            cov_matrix = np.cov(returns)
            simulated_asset_returns = np.random.multivariate_normal(
                mean_returns, cov_matrix, size=self.n_simulations
            ).T

        elif method == 'cholesky':
            # Cholesky decomposition for correlation simulation
            mean_returns = np.mean(returns, axis=1)
            cov_matrix = np.cov(returns)

            try:
                L = np.linalg.cholesky(cov_matrix)
                independent_shocks = np.random.standard_normal((n_assets, self.n_simulations))
                correlated_shocks = np.dot(L, independent_shocks)
                simulated_asset_returns = mean_returns[:, np.newaxis] + correlated_shocks
            except np.linalg.LinAlgError:
                # Fall back to eigendecomposition if Cholesky fails
                eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)
                eigenvals = np.maximum(eigenvals, 1e-8)  # Ensure positive definite
                sqrt_eigenvals = np.sqrt(eigenvals)
                independent_shocks = np.random.standard_normal((n_assets, self.n_simulations))
                correlated_shocks = np.dot(eigenvecs, sqrt_eigenvals[:, np.newaxis] * independent_shocks)
                simulated_asset_returns = mean_returns[:, np.newaxis] + correlated_shocks

        else:
            raise ValueError("Method must be 'bootstrap', 'parametric', or 'cholesky'")

        # Calculate portfolio returns
        portfolio_returns = np.dot(weights, simulated_asset_returns)

        return portfolio_returns


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)

    # Generate sample return data
    n_days = 1000
    returns = np.random.normal(0.001, 0.02, n_days)  # Daily returns

    portfolio_value = 1000000  # $1M portfolio

    print("VaR Model Comparison")
    print("=" * 50)

    # Historical VaR
    hist_var = HistoricalVaR(confidence_level=0.05)
    hist_var_value = hist_var.calculate_var(returns, portfolio_value)
    print(f"Historical VaR (95%): ${hist_var_value:,.2f}")

    # Parametric VaR
    param_var = ParametricVaR(confidence_level=0.05)
    param_var_value = param_var.calculate_var(returns, portfolio_value)
    print(f"Parametric VaR (95%): ${param_var_value:,.2f}")

    # Monte Carlo VaR
    mc_var = MonteCarloVaR(confidence_level=0.05, n_simulations=10000, random_seed=42)
    mc_var_value = mc_var.calculate_var(returns, portfolio_value)
    print(f"Monte Carlo VaR (95%): ${mc_var_value:,.2f}")

    # Backtesting example
    print("\nBacktesting Results:")
    print("-" * 20)

    # Generate out-of-sample returns for backtesting
    test_returns = np.random.normal(0.001, 0.02, 100)
    var_estimates = np.full(100, hist_var_value / portfolio_value)  # Convert back to returns

    backtest_results = hist_var.backtest_var(test_returns, var_estimates)
    print(f"Violation Rate: {backtest_results['violation_rate']:.3f}")
    print(f"Expected Rate: {backtest_results['expected_violation_rate']:.3f}")
    print(f"Kupiec Test P-value: {backtest_results['kupiec_pvalue']:.3f}")
    print(f"Test Passed: {backtest_results['test_passed']}")

    # Multi-asset example
    print("\nMulti-Asset Portfolio VaR:")
    print("-" * 30)

    # Generate 3-asset returns
    n_assets = 3
    asset_returns = np.random.multivariate_normal(
        mean=[0.001, 0.0008, 0.0012],
        cov=[[0.0004, 0.0001, 0.0002],
             [0.0001, 0.0009, 0.0001],
             [0.0002, 0.0001, 0.0016]],
        size=n_days
    ).T

    weights = np.array([0.4, 0.35, 0.25])

    # Component VaR
    component_var = param_var.calculate_component_var(asset_returns, weights, portfolio_value)
    print(f"Component VaR: {component_var}")
    print(f"Total VaR: ${np.sum(component_var):,.2f}")


class VaRModel:
    """
    Unified VaR model interface that combines all VaR calculation methods.

    This class provides a simple interface for calculating VaR using different
    methodologies while maintaining the mathematical rigor of individual calculators.
    """

    def __init__(self):
        """Initialize VaR model with all calculation methods."""
        self.historical = HistoricalVaR()
        self.parametric = ParametricVaR()
        self.monte_carlo = MonteCarloVaR()

    def historical_var(self, returns: np.ndarray, confidence_level: float = 0.05) -> float:
        """Calculate historical VaR."""
        return self.historical.calculate_var(returns, confidence_level)

    def parametric_var(self, returns: np.ndarray, confidence_level: float = 0.05) -> float:
        """Calculate parametric VaR."""
        return self.parametric.calculate_var(returns, confidence_level)

    def monte_carlo_var(self, returns: np.ndarray, confidence_level: float = 0.05, n_simulations: int = 10000) -> float:
        """Calculate Monte Carlo VaR."""
        # Create a new MonteCarloVaR instance with the specified parameters
        mc_var = MonteCarloVaR(confidence_level=confidence_level, n_simulations=n_simulations)
        return mc_var.calculate_var(returns)