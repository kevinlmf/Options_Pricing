"""
Conditional Value at Risk (CVaR) Models

Implementation of CVaR (Expected Shortfall) calculation methods.
Follows the mathematical framework: Pure Math → Applied Math → Risk Management

Mathematical Foundation:
- Pure Math: Conditional expectations, tail distributions, optimization theory
- Applied Math: Linear programming, coherent risk measures, tail statistics
- Risk Management: CVaR calculation, coherent risk properties, portfolio optimization
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Union, Optional, Tuple
from abc import ABC, abstractmethod
from scipy import stats, optimize
import warnings


class CVarCalculator(ABC):
    """Abstract base class for CVaR calculators"""

    def __init__(self, confidence_level: float = 0.05):
        """
        Initialize CVaR calculator.

        Args:
            confidence_level: CVaR confidence level (e.g., 0.05 for 95% CVaR)
        """
        if not 0 < confidence_level < 1:
            raise ValueError("Confidence level must be between 0 and 1")
        self.confidence_level = confidence_level
        self.alpha = confidence_level

    @abstractmethod
    def calculate_cvar(self, returns: Union[np.ndarray, pd.Series],
                      portfolio_value: float = 1.0) -> float:
        """Calculate CVaR for given returns"""
        pass

    @abstractmethod
    def calculate_var_cvar(self, returns: Union[np.ndarray, pd.Series],
                          portfolio_value: float = 1.0) -> Tuple[float, float]:
        """Calculate both VaR and CVaR simultaneously"""
        pass

    def verify_coherence_properties(self, returns_scenarios: List[np.ndarray],
                                  weights_scenarios: List[np.ndarray]) -> Dict[str, bool]:
        """
        Verify that CVaR satisfies coherent risk measure properties.

        Tests:
        1. Monotonicity: X ≤ Y ⟹ ρ(X) ≥ ρ(Y)
        2. Translation Invariance: ρ(X + c) = ρ(X) - c
        3. Positive Homogeneity: ρ(λX) = λρ(X) for λ > 0
        4. Sub-additivity: ρ(X + Y) ≤ ρ(X) + ρ(Y)
        """
        results = {}

        # Test translation invariance
        returns = returns_scenarios[0]
        constant = 0.01
        cvar_original = self.calculate_cvar(returns)
        cvar_shifted = self.calculate_cvar(returns + constant)
        results['translation_invariance'] = abs(cvar_shifted - (cvar_original - constant)) < 1e-6

        # Test positive homogeneity
        lambda_val = 2.5
        cvar_scaled = self.calculate_cvar(lambda_val * returns)
        results['positive_homogeneity'] = abs(cvar_scaled - lambda_val * cvar_original) < 1e-6

        # Test sub-additivity (approximately, using different return series)
        if len(returns_scenarios) >= 2:
            returns_y = returns_scenarios[1]
            cvar_x = self.calculate_cvar(returns)
            cvar_y = self.calculate_cvar(returns_y)
            cvar_sum = self.calculate_cvar(returns + returns_y)
            results['sub_additivity'] = cvar_sum <= (cvar_x + cvar_y + 1e-6)  # Small tolerance
        else:
            results['sub_additivity'] = None

        # Test monotonicity
        returns_worse = returns - 0.005  # Worse returns
        cvar_worse = self.calculate_cvar(returns_worse)
        results['monotonicity'] = cvar_worse >= cvar_original - 1e-6

        return results


class ExpectedShortfall(CVarCalculator):
    """
    Expected Shortfall (ES) / Conditional VaR

    Calculates the expected loss given that loss exceeds VaR.
    This is the most common CVaR implementation.
    """

    def __init__(self, confidence_level: float = 0.05, method: str = 'historical'):
        """
        Initialize Expected Shortfall calculator.

        Args:
            confidence_level: CVaR confidence level
            method: Calculation method ('historical', 'parametric', 'mixed')
        """
        super().__init__(confidence_level)
        self.method = method.lower()

        if self.method not in ['historical', 'parametric', 'mixed']:
            raise ValueError("Method must be 'historical', 'parametric', or 'mixed'")

    def calculate_cvar(self, returns: Union[np.ndarray, pd.Series],
                      portfolio_value: float = 1.0) -> float:
        """Calculate CVaR using specified method"""
        returns = np.array(returns)

        if len(returns) == 0:
            raise ValueError("Returns array cannot be empty")

        if self.method == 'historical':
            return self._historical_cvar(returns, portfolio_value)
        elif self.method == 'parametric':
            return self._parametric_cvar(returns, portfolio_value)
        else:  # mixed
            return self._mixed_cvar(returns, portfolio_value)

    def calculate_var_cvar(self, returns: Union[np.ndarray, pd.Series],
                          portfolio_value: float = 1.0) -> Tuple[float, float]:
        """Calculate both VaR and CVaR"""
        returns = np.array(returns)

        if self.method == 'historical':
            var_quantile = np.percentile(returns, self.alpha * 100)
            tail_losses = returns[returns <= var_quantile]
            cvar_quantile = np.mean(tail_losses) if len(tail_losses) > 0 else var_quantile

        elif self.method == 'parametric':
            mean_return = np.mean(returns)
            std_return = np.std(returns, ddof=1)

            # Assume normal distribution
            var_quantile = stats.norm.ppf(self.alpha, mean_return, std_return)

            # CVaR for normal distribution: μ - σ * φ(Φ⁻¹(α)) / α
            phi_inv_alpha = stats.norm.ppf(self.alpha)
            phi_phi_inv_alpha = stats.norm.pdf(phi_inv_alpha)
            cvar_quantile = mean_return - std_return * phi_phi_inv_alpha / self.alpha

        else:  # mixed
            # Use parametric if sample size is small, historical otherwise
            if len(returns) < 100:
                return self._parametric_var_cvar(returns, portfolio_value)
            else:
                return self._historical_var_cvar(returns, portfolio_value)

        var_value = -var_quantile * portfolio_value
        cvar_value = -cvar_quantile * portfolio_value

        return var_value, cvar_value

    def _historical_cvar(self, returns: np.ndarray, portfolio_value: float) -> float:
        """Historical CVaR calculation"""
        var_quantile = np.percentile(returns, self.alpha * 100)
        tail_losses = returns[returns <= var_quantile]

        if len(tail_losses) == 0:
            # No observations in tail - use VaR as approximation
            return -var_quantile * portfolio_value

        cvar_quantile = np.mean(tail_losses)
        return -cvar_quantile * portfolio_value

    def _parametric_cvar(self, returns: np.ndarray, portfolio_value: float) -> float:
        """Parametric CVaR assuming normal distribution"""
        mean_return = np.mean(returns)
        std_return = np.std(returns, ddof=1)

        # CVaR for normal distribution
        phi_inv_alpha = stats.norm.ppf(self.alpha)
        phi_phi_inv_alpha = stats.norm.pdf(phi_inv_alpha)
        cvar_quantile = mean_return - std_return * phi_phi_inv_alpha / self.alpha

        return -cvar_quantile * portfolio_value

    def _mixed_cvar(self, returns: np.ndarray, portfolio_value: float) -> float:
        """Mixed approach: parametric for small samples, historical for large samples"""
        if len(returns) < 100:
            return self._parametric_cvar(returns, portfolio_value)
        else:
            return self._historical_cvar(returns, portfolio_value)

    def _historical_var_cvar(self, returns: np.ndarray, portfolio_value: float) -> Tuple[float, float]:
        """Historical VaR and CVaR calculation"""
        var_quantile = np.percentile(returns, self.alpha * 100)
        tail_losses = returns[returns <= var_quantile]
        cvar_quantile = np.mean(tail_losses) if len(tail_losses) > 0 else var_quantile

        return -var_quantile * portfolio_value, -cvar_quantile * portfolio_value

    def _parametric_var_cvar(self, returns: np.ndarray, portfolio_value: float) -> Tuple[float, float]:
        """Parametric VaR and CVaR calculation"""
        mean_return = np.mean(returns)
        std_return = np.std(returns, ddof=1)

        var_quantile = stats.norm.ppf(self.alpha, mean_return, std_return)

        phi_inv_alpha = stats.norm.ppf(self.alpha)
        phi_phi_inv_alpha = stats.norm.pdf(phi_inv_alpha)
        cvar_quantile = mean_return - std_return * phi_phi_inv_alpha / self.alpha

        return -var_quantile * portfolio_value, -cvar_quantile * portfolio_value

    def calculate_component_cvar(self, returns: np.ndarray, weights: np.ndarray,
                                portfolio_value: float = 1.0) -> np.ndarray:
        """
        Calculate component CVaR contributions using Euler allocation principle.

        For coherent risk measures, component contributions sum to total risk.
        """
        if returns.ndim != 2:
            raise ValueError("Returns must be 2D array (assets x time)")

        n_assets = returns.shape[0]
        if len(weights) != n_assets:
            raise ValueError("Weights length must match number of assets")

        # Portfolio returns
        portfolio_returns = np.dot(weights, returns)

        # Calculate portfolio CVaR
        portfolio_cvar = self.calculate_cvar(portfolio_returns, portfolio_value)

        # Calculate marginal CVaR contributions
        epsilon = 0.0001  # Small perturbation for finite difference
        marginal_cvar = np.zeros(n_assets)

        for i in range(n_assets):
            # Perturb weight slightly
            perturbed_weights = weights.copy()
            perturbed_weights[i] += epsilon

            # Renormalize weights
            perturbed_weights = perturbed_weights / np.sum(perturbed_weights) * np.sum(weights)

            # Calculate perturbed portfolio CVaR
            perturbed_returns = np.dot(perturbed_weights, returns)
            perturbed_cvar = self.calculate_cvar(perturbed_returns, portfolio_value)

            # Marginal CVaR
            marginal_cvar[i] = (perturbed_cvar - portfolio_cvar) / epsilon

        # Component CVaR = weight * marginal CVaR
        component_cvar = weights * marginal_cvar

        # Ensure components sum to total (up to numerical precision)
        total_component = np.sum(component_cvar)
        if abs(total_component - portfolio_cvar) > 1e-2:
            # Adjust for numerical errors
            adjustment_factor = portfolio_cvar / total_component
            component_cvar *= adjustment_factor

        return component_cvar

    def rolling_cvar(self, returns: Union[np.ndarray, pd.Series],
                    window_size: int, portfolio_value: float = 1.0) -> np.ndarray:
        """Calculate rolling CVaR estimates"""
        returns = np.array(returns)
        n_obs = len(returns)

        if window_size >= n_obs:
            raise ValueError("Window size must be less than number of observations")

        cvar_estimates = np.full(n_obs, np.nan)

        for i in range(window_size, n_obs):
            window_returns = returns[i-window_size:i]
            cvar_estimates[i] = self.calculate_cvar(window_returns, portfolio_value)

        return cvar_estimates

    def spectral_risk_measure(self, returns: Union[np.ndarray, pd.Series],
                             phi_function, portfolio_value: float = 1.0) -> float:
        """
        Calculate spectral risk measure with custom risk aversion function φ(u).

        CVaR is a special case where φ(u) = 1/α for u ≤ α, 0 otherwise.

        Args:
            returns: Return series
            phi_function: Risk aversion function φ(u) where u ∈ [0,1]
            portfolio_value: Portfolio value

        Returns:
            Spectral risk measure value
        """
        returns = np.array(returns)
        sorted_returns = np.sort(returns)
        n = len(returns)

        # Calculate spectral risk measure
        risk_measure = 0
        for i, return_val in enumerate(sorted_returns):
            u = (i + 0.5) / n  # Midpoint of interval
            weight = phi_function(u) / n
            risk_measure += weight * return_val

        return -risk_measure * portfolio_value


class OptimizationBasedCVaR(CVarCalculator):
    """
    CVaR calculation using linear programming optimization.

    This approach is useful for portfolio optimization with CVaR constraints
    and provides the exact CVaR formulation used in optimization.
    """

    def __init__(self, confidence_level: float = 0.05):
        super().__init__(confidence_level)

    def calculate_cvar(self, returns: Union[np.ndarray, pd.Series],
                      portfolio_value: float = 1.0) -> float:
        """Calculate CVaR using linear programming formulation"""
        returns = np.array(returns)
        n_scenarios = len(returns)

        if n_scenarios == 0:
            raise ValueError("Returns array cannot be empty")

        # CVaR optimization problem:
        # min_t,u t + (1/α) * (1/n) * Σ u_i
        # s.t. u_i ≥ -r_i - t, u_i ≥ 0

        def objective(vars):
            t = vars[0]
            u = vars[1:]
            return t + (1/self.alpha) * np.mean(u)

        def constraint_u_positive(vars):
            return vars[1:]  # u_i ≥ 0

        def constraint_u_loss(vars):
            t = vars[0]
            u = vars[1:]
            return u + returns + t  # u_i ≥ -r_i - t

        # Initial guess
        x0 = np.zeros(n_scenarios + 1)

        # Constraints
        constraints = [
            {'type': 'ineq', 'fun': constraint_u_positive},
            {'type': 'ineq', 'fun': constraint_u_loss}
        ]

        # Solve optimization
        result = optimize.minimize(objective, x0, method='SLSQP', constraints=constraints)

        if not result.success:
            warnings.warn("CVaR optimization did not converge, using fallback method")
            # Fallback to expected shortfall
            es = ExpectedShortfall(self.confidence_level)
            return es.calculate_cvar(returns, portfolio_value)

        cvar_value = result.fun
        return cvar_value * portfolio_value

    def calculate_var_cvar(self, returns: Union[np.ndarray, pd.Series],
                          portfolio_value: float = 1.0) -> Tuple[float, float]:
        """Calculate both VaR and CVaR using optimization"""
        returns = np.array(returns)

        # VaR is simply the quantile
        var_quantile = np.percentile(returns, self.alpha * 100)
        var_value = -var_quantile * portfolio_value

        # CVaR from optimization
        cvar_value = self.calculate_cvar(returns, portfolio_value)

        return var_value, cvar_value

    def calculate_component_cvar(self, returns: np.ndarray, weights: np.ndarray,
                                portfolio_value: float = 1.0) -> np.ndarray:
        """Calculate component CVaR using optimization-based approach"""
        # This is complex for the optimization approach
        # Fall back to Expected Shortfall method
        es = ExpectedShortfall(self.confidence_level)
        return es.calculate_component_cvar(returns, weights, portfolio_value)


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)

    # Generate sample return data with fat tails
    n_days = 1000
    normal_returns = np.random.normal(0.001, 0.015, n_days)
    fat_tail_returns = np.random.standard_t(5, n_days) * 0.02 + 0.001
    returns = 0.7 * normal_returns + 0.3 * fat_tail_returns

    portfolio_value = 1000000  # $1M portfolio

    print("CVaR Model Comparison")
    print("=" * 50)

    # Expected Shortfall - Historical
    es_hist = ExpectedShortfall(confidence_level=0.05, method='historical')
    cvar_hist = es_hist.calculate_cvar(returns, portfolio_value)
    var_hist, cvar_hist_both = es_hist.calculate_var_cvar(returns, portfolio_value)

    print(f"Historical CVaR (95%): ${cvar_hist:,.2f}")
    print(f"Historical VaR (95%): ${var_hist:,.2f}")
    print(f"CVaR/VaR Ratio: {cvar_hist/var_hist:.3f}")

    # Expected Shortfall - Parametric
    es_param = ExpectedShortfall(confidence_level=0.05, method='parametric')
    cvar_param = es_param.calculate_cvar(returns, portfolio_value)
    print(f"Parametric CVaR (95%): ${cvar_param:,.2f}")

    # Optimization-based CVaR
    opt_cvar = OptimizationBasedCVaR(confidence_level=0.05)
    cvar_opt = opt_cvar.calculate_cvar(returns, portfolio_value)
    print(f"Optimization CVaR (95%): ${cvar_opt:,.2f}")

    # Test coherence properties
    print("\nCoherence Properties Test:")
    print("-" * 30)

    returns_list = [returns, np.random.normal(0.0005, 0.018, n_days)]
    coherence_results = es_hist.verify_coherence_properties(returns_list, [])

    for prop, result in coherence_results.items():
        if result is not None:
            print(f"{prop}: {'✓ Pass' if result else '✗ Fail'}")

    # Multi-asset CVaR
    print("\nMulti-Asset Portfolio CVaR:")
    print("-" * 30)

    # Generate 3-asset returns with correlation
    n_assets = 3
    correlation_matrix = np.array([[1.0, 0.3, 0.2],
                                  [0.3, 1.0, 0.4],
                                  [0.2, 0.4, 1.0]])

    # Generate correlated returns
    independent_returns = np.random.standard_t(5, (n_assets, n_days))
    L = np.linalg.cholesky(correlation_matrix)
    correlated_returns = np.dot(L, independent_returns)

    # Add different means and volatilities
    mean_returns = np.array([0.001, 0.0008, 0.0012])
    vol_returns = np.array([0.02, 0.025, 0.03])

    asset_returns = mean_returns[:, np.newaxis] + vol_returns[:, np.newaxis] * correlated_returns

    weights = np.array([0.4, 0.35, 0.25])

    # Component CVaR
    component_cvar = es_hist.calculate_component_cvar(asset_returns, weights, portfolio_value)
    print(f"Component CVaR:")
    for i, comp_cvar in enumerate(component_cvar):
        print(f"  Asset {i+1}: ${comp_cvar:,.2f}")
    print(f"Total CVaR: ${np.sum(component_cvar):,.2f}")

    # Spectral risk measure example
    print("\nSpectral Risk Measure:")
    print("-" * 20)

    # Define exponential risk aversion function
    def exponential_phi(u, gamma=2):
        """Exponential risk aversion: more weight on tail losses"""
        return gamma * np.exp(gamma * u) / (np.exp(gamma) - 1)

    spectral_risk = es_hist.spectral_risk_measure(returns, exponential_phi, portfolio_value)
    print(f"Spectral Risk (γ=2): ${spectral_risk:,.2f}")

    # Rolling CVaR
    print("\nRolling CVaR Analysis:")
    print("-" * 20)

    rolling_cvar = es_hist.rolling_cvar(returns, window_size=250, portfolio_value=portfolio_value)
    valid_cvar = rolling_cvar[~np.isnan(rolling_cvar)]

    print(f"Rolling CVaR - Mean: ${np.mean(valid_cvar):,.2f}")
    print(f"Rolling CVaR - Std: ${np.std(valid_cvar):,.2f}")
    print(f"Rolling CVaR - Max: ${np.max(valid_cvar):,.2f}")


class CVaRModel:
    """
    Unified CVaR model interface that combines all CVaR calculation methods.

    This class provides a simple interface for calculating CVaR using different
    methodologies while maintaining mathematical rigor.
    """

    def __init__(self):
        """Initialize CVaR model with all calculation methods."""
        self.expected_shortfall_calc = ExpectedShortfall()
        self.optimization_based_calc = OptimizationBasedCVaR()

    def expected_shortfall(self, returns: np.ndarray, confidence_level: float = 0.05) -> float:
        """Calculate CVaR using expected shortfall method."""
        return self.expected_shortfall_calc.calculate_cvar(returns, confidence_level)

    def optimization_cvar(self, returns: np.ndarray, confidence_level: float = 0.05) -> float:
        """Calculate CVaR using optimization-based method."""
        return self.optimization_based_calc.calculate_cvar(returns, confidence_level)