"""
Risk Measures Framework

Implementation of coherent risk measures and risk measure validation.
Follows the mathematical framework: Pure Math → Applied Math → Risk Management

Mathematical Foundation:
- Pure Math: Functional analysis, convex analysis, measure theory
- Applied Math: Optimization theory, coherent risk measure axioms
- Risk Management: Risk measure validation, model comparison, regulatory compliance
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Callable, Tuple, Any
from abc import ABC, abstractmethod
from scipy import stats, optimize
import warnings
from .var_models import VarCalculator, HistoricalVaR, ParametricVaR, MonteCarloVaR
from .cvar_models import ExpectedShortfall


class RiskMeasure(ABC):
    """Abstract base class for risk measures"""

    @abstractmethod
    def calculate(self, returns: np.ndarray) -> float:
        """Calculate risk measure value"""
        pass

    @abstractmethod
    def is_coherent(self) -> bool:
        """Check if risk measure is coherent"""
        pass


class CoherentRiskMeasures:
    """
    Implementation of various coherent risk measures and their properties.

    A risk measure ρ is coherent if it satisfies:
    1. Monotonicity: X ≤ Y ⟹ ρ(X) ≥ ρ(Y)
    2. Translation Invariance: ρ(X + c) = ρ(X) - c
    3. Positive Homogeneity: ρ(λX) = λρ(X) for λ > 0
    4. Sub-additivity: ρ(X + Y) ≤ ρ(X) + ρ(Y)
    """

    @staticmethod
    def expected_shortfall(returns: np.ndarray, alpha: float) -> float:
        """
        Expected Shortfall (CVaR) - coherent risk measure

        Args:
            returns: Return series
            alpha: Confidence level (e.g., 0.05 for 95% ES)

        Returns:
            Expected Shortfall value
        """
        es_calc = ExpectedShortfall(alpha, method='historical')
        return es_calc.calculate_cvar(returns)

    @staticmethod
    def worst_case_risk(returns: np.ndarray) -> float:
        """
        Worst Case Risk - coherent (extreme case of ES with α→0)

        Returns:
            Worst possible loss
        """
        return -np.min(returns)

    @staticmethod
    def spectral_risk_measure(returns: np.ndarray, phi_function: Callable[[float], float]) -> float:
        """
        Spectral Risk Measure - coherent if φ is non-decreasing and integrates to 1

        Args:
            returns: Return series
            phi_function: Risk aversion function φ(u) where u ∈ [0,1]

        Returns:
            Spectral risk measure value
        """
        sorted_returns = np.sort(returns)
        n = len(returns)

        risk_measure = 0
        for i, return_val in enumerate(sorted_returns):
            u = (i + 0.5) / n  # Midpoint of interval
            weight = phi_function(u) / n
            risk_measure += weight * return_val

        return -risk_measure

    @staticmethod
    def distortion_risk_measure(returns: np.ndarray, g_function: Callable[[float], float]) -> float:
        """
        Distortion Risk Measure - coherent if g is concave

        Args:
            returns: Return series
            g_function: Distortion function g: [0,1] → [0,1]

        Returns:
            Distortion risk measure value
        """
        sorted_returns = np.sort(returns)
        n = len(returns)

        # Calculate distorted probabilities
        risk_measure = 0
        for i in range(n):
            p_lower = i / n
            p_upper = (i + 1) / n

            # Distorted probability weight
            if i == 0:
                weight = g_function(p_upper)
            elif i == n - 1:
                weight = 1 - g_function(p_lower)
            else:
                weight = g_function(p_upper) - g_function(p_lower)

            risk_measure += weight * sorted_returns[i]

        return -risk_measure

    @staticmethod
    def entropic_risk_measure(returns: np.ndarray, theta: float) -> float:
        """
        Entropic Risk Measure - coherent

        Args:
            returns: Return series
            theta: Risk aversion parameter (θ > 0)

        Returns:
            Entropic risk measure value
        """
        if theta <= 0:
            raise ValueError("Risk aversion parameter must be positive")

        # Avoid numerical overflow
        max_return = np.max(-returns * theta)
        exp_values = np.exp(-returns * theta - max_return)

        return (np.log(np.mean(exp_values)) + max_return) / theta

    @staticmethod
    def optimized_certainty_equivalent(returns: np.ndarray,
                                     utility_function: Callable[[float], float]) -> float:
        """
        Optimized Certainty Equivalent - coherent if utility is concave

        Args:
            returns: Return series
            utility_function: Concave utility function

        Returns:
            Negative of certainty equivalent
        """
        # Expected utility
        expected_utility = np.mean([utility_function(r) for r in returns])

        # Find certainty equivalent (inverse utility)
        def objective(ce):
            return (utility_function(ce) - expected_utility) ** 2

        # Search for certainty equivalent
        result = optimize.minimize_scalar(objective, bounds=(-1, 1), method='bounded')
        certainty_equivalent = result.x

        return -certainty_equivalent


class RiskMeasureValidator:
    """
    Validator for risk measure properties and backtesting.
    """

    def __init__(self, risk_measure_func: Callable[[np.ndarray], float]):
        """
        Initialize risk measure validator.

        Args:
            risk_measure_func: Risk measure calculation function
        """
        self.risk_measure_func = risk_measure_func

    def test_coherence_properties(self, test_returns: List[np.ndarray],
                                tolerance: float = 1e-6) -> Dict[str, bool]:
        """
        Test if risk measure satisfies coherence properties.

        Args:
            test_returns: List of return series for testing
            tolerance: Numerical tolerance for tests

        Returns:
            Dictionary with test results for each property
        """
        if len(test_returns) < 2:
            raise ValueError("Need at least 2 return series for coherence testing")

        X = test_returns[0]
        Y = test_returns[1]

        results = {}

        # Test 1: Translation Invariance
        c = 0.01
        rho_X = self.risk_measure_func(X)
        rho_X_c = self.risk_measure_func(X + c)
        results['translation_invariance'] = abs(rho_X_c - (rho_X - c)) < tolerance

        # Test 2: Positive Homogeneity
        lambda_val = 2.5
        rho_lambda_X = self.risk_measure_func(lambda_val * X)
        results['positive_homogeneity'] = abs(rho_lambda_X - lambda_val * rho_X) < tolerance

        # Test 3: Sub-additivity
        rho_Y = self.risk_measure_func(Y)
        rho_X_Y = self.risk_measure_func(X + Y)
        results['sub_additivity'] = rho_X_Y <= (rho_X + rho_Y + tolerance)

        # Test 4: Monotonicity
        epsilon = 0.001
        X_worse = X - epsilon  # Worse returns
        rho_X_worse = self.risk_measure_func(X_worse)
        results['monotonicity'] = rho_X_worse >= (rho_X - tolerance)

        return results

    def backtest_risk_measure(self, historical_returns: np.ndarray,
                            risk_estimates: np.ndarray,
                            confidence_level: float = 0.05,
                            window_size: int = 250) -> Dict[str, Any]:
        """
        Backtest risk measure estimates against actual losses.

        Args:
            historical_returns: Historical return series
            risk_estimates: Risk measure estimates for each period
            confidence_level: Expected exceedance probability
            window_size: Rolling window size for estimation

        Returns:
            Backtesting statistics
        """
        if len(historical_returns) != len(risk_estimates):
            raise ValueError("Returns and risk estimates must have same length")

        # Find exceedances (actual losses exceeding risk estimates)
        losses = -historical_returns
        exceedances = losses > risk_estimates

        n_exceedances = np.sum(exceedances)
        n_observations = len(exceedances)
        exceedance_rate = n_exceedances / n_observations

        # Expected exceedance rate
        expected_rate = confidence_level

        # Statistical tests
        results = {
            'n_observations': n_observations,
            'n_exceedances': n_exceedances,
            'exceedance_rate': exceedance_rate,
            'expected_rate': expected_rate,
        }

        # Kupiec unconditional coverage test
        if n_exceedances == 0:
            kupiec_stat = -2 * np.log((1 - expected_rate) ** n_observations)
        elif n_exceedances == n_observations:
            kupiec_stat = -2 * np.log(expected_rate ** n_observations)
        else:
            # Log-likelihood ratio
            ll_unrestricted = (n_exceedances * np.log(exceedance_rate) +
                             (n_observations - n_exceedances) * np.log(1 - exceedance_rate))
            ll_restricted = (n_exceedances * np.log(expected_rate) +
                           (n_observations - n_exceedances) * np.log(1 - expected_rate))
            kupiec_stat = -2 * (ll_restricted - ll_unrestricted)

        kupiec_pvalue = 1 - stats.chi2.cdf(kupiec_stat, 1)
        results['kupiec_statistic'] = kupiec_stat
        results['kupiec_pvalue'] = kupiec_pvalue
        results['kupiec_test_passed'] = kupiec_pvalue > 0.05

        # Christoffersen independence test (clustering of exceedances)
        if n_exceedances > 1:
            # Count transitions
            n00 = np.sum((~exceedances[:-1]) & (~exceedances[1:]))  # No exc -> No exc
            n01 = np.sum((~exceedances[:-1]) & exceedances[1:])     # No exc -> Exc
            n10 = np.sum(exceedances[:-1] & (~exceedances[1:]))     # Exc -> No exc
            n11 = np.sum(exceedances[:-1] & exceedances[1:])        # Exc -> Exc

            # Independence test
            if n01 > 0 and n10 > 0 and n00 > 0 and n11 > 0:
                pi_01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0
                pi_11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0
                pi = (n01 + n11) / (n00 + n01 + n10 + n11)

                if pi_01 > 0 and pi_11 > 0 and pi > 0 and (1-pi) > 0:
                    ll_unrestricted = (n00 * np.log(1 - pi_01) + n01 * np.log(pi_01) +
                                     n10 * np.log(1 - pi_11) + n11 * np.log(pi_11))
                    ll_restricted = ((n00 + n10) * np.log(1 - pi) +
                                   (n01 + n11) * np.log(pi))

                    independence_stat = -2 * (ll_restricted - ll_unrestricted)
                    independence_pvalue = 1 - stats.chi2.cdf(independence_stat, 1)
                else:
                    independence_stat = 0
                    independence_pvalue = 1
            else:
                independence_stat = 0
                independence_pvalue = 1

            results['independence_statistic'] = independence_stat
            results['independence_pvalue'] = independence_pvalue
            results['independence_test_passed'] = independence_pvalue > 0.05
        else:
            results['independence_statistic'] = None
            results['independence_pvalue'] = None
            results['independence_test_passed'] = None

        # Combined test (Christoffersen)
        if results['independence_test_passed'] is not None:
            combined_stat = kupiec_stat + independence_stat
            combined_pvalue = 1 - stats.chi2.cdf(combined_stat, 2)
            results['combined_statistic'] = combined_stat
            results['combined_pvalue'] = combined_pvalue
            results['combined_test_passed'] = combined_pvalue > 0.05

        # Loss function evaluation
        if n_exceedances > 0:
            # Average exceedance size
            exceedance_losses = losses[exceedances]
            exceedance_estimates = risk_estimates[exceedances]
            avg_exceedance_size = np.mean(exceedance_losses - exceedance_estimates)
            max_exceedance_size = np.max(exceedance_losses - exceedance_estimates)
        else:
            avg_exceedance_size = 0
            max_exceedance_size = 0

        results['avg_exceedance_size'] = avg_exceedance_size
        results['max_exceedance_size'] = max_exceedance_size

        # Regulatory traffic light system
        if exceedance_rate <= expected_rate + 0.002:  # Green zone
            traffic_light = 'Green'
        elif exceedance_rate <= expected_rate + 0.01:  # Yellow zone
            traffic_light = 'Yellow'
        else:  # Red zone
            traffic_light = 'Red'

        results['traffic_light'] = traffic_light

        return results

    def compare_risk_measures(self, returns_data: np.ndarray,
                            risk_measures: Dict[str, Callable]) -> Dict[str, Any]:
        """
        Compare different risk measures on the same data.

        Args:
            returns_data: Return series for comparison
            risk_measures: Dictionary of risk measure name -> function

        Returns:
            Comparison results
        """
        results = {}
        risk_values = {}

        # Calculate risk measures
        for name, func in risk_measures.items():
            try:
                risk_values[name] = func(returns_data)
            except Exception as e:
                print(f"Warning: Error calculating {name}: {e}")
                risk_values[name] = np.nan

        results['risk_values'] = risk_values

        # Rank risk measures by conservatism
        valid_measures = {k: v for k, v in risk_values.items() if not np.isnan(v)}
        if valid_measures:
            sorted_measures = sorted(valid_measures.items(), key=lambda x: x[1], reverse=True)
            results['conservatism_ranking'] = [name for name, _ in sorted_measures]

        # Test coherence properties for each measure
        coherence_results = {}
        test_data = [returns_data]

        # Create additional test series
        if len(returns_data) > 100:
            test_data.append(returns_data[:len(returns_data)//2])
            test_data.append(np.random.permutation(returns_data))

        for name, func in risk_measures.items():
            validator = RiskMeasureValidator(func)
            try:
                coherence_results[name] = validator.test_coherence_properties(test_data)
            except Exception as e:
                coherence_results[name] = {'error': str(e)}

        results['coherence_tests'] = coherence_results

        # Statistical properties comparison
        stats_comparison = {}
        for name, risk_val in valid_measures.items():
            # Bootstrap confidence interval
            bootstrap_values = []
            for _ in range(1000):
                bootstrap_sample = np.random.choice(returns_data, size=len(returns_data), replace=True)
                try:
                    bootstrap_values.append(risk_measures[name](bootstrap_sample))
                except:
                    continue

            if bootstrap_values:
                stats_comparison[name] = {
                    'mean': np.mean(bootstrap_values),
                    'std': np.std(bootstrap_values),
                    'ci_lower': np.percentile(bootstrap_values, 2.5),
                    'ci_upper': np.percentile(bootstrap_values, 97.5),
                }

        results['statistical_comparison'] = stats_comparison

        return results

    def model_validation_report(self, returns: np.ndarray,
                              risk_estimates: np.ndarray,
                              confidence_level: float = 0.05) -> Dict[str, Any]:
        """
        Generate comprehensive model validation report.

        Args:
            returns: Historical returns
            risk_estimates: Risk measure estimates
            confidence_level: Expected confidence level

        Returns:
            Comprehensive validation report
        """
        report = {
            'summary': {
                'n_observations': len(returns),
                'confidence_level': confidence_level,
                'mean_return': np.mean(returns),
                'volatility': np.std(returns),
                'mean_risk_estimate': np.mean(risk_estimates),
            }
        }

        # Backtesting results
        backtest_results = self.backtest_risk_measure(returns, risk_estimates, confidence_level)
        report['backtesting'] = backtest_results

        # Risk measure stability over time
        window_size = min(250, len(returns) // 4)
        if window_size > 50:
            rolling_estimates = []
            for i in range(window_size, len(returns)):
                window_data = returns[i-window_size:i]
                rolling_estimates.append(self.risk_measure_func(window_data))

            report['stability'] = {
                'rolling_mean': np.mean(rolling_estimates),
                'rolling_std': np.std(rolling_estimates),
                'rolling_min': np.min(rolling_estimates),
                'rolling_max': np.max(rolling_estimates),
                'coefficient_of_variation': np.std(rolling_estimates) / np.mean(rolling_estimates) if np.mean(rolling_estimates) != 0 else np.inf
            }

        # Performance metrics
        losses = -returns
        exceedances = losses > risk_estimates

        if np.sum(exceedances) > 0:
            report['performance'] = {
                'hit_rate': np.mean(exceedances),
                'false_alarm_rate': np.mean(~exceedances & (risk_estimates > np.percentile(losses, 95))),
                'mean_exceedance': np.mean(losses[exceedances] - risk_estimates[exceedances]),
                'predictive_ability': 1 - np.mean((losses - risk_estimates) ** 2) / np.var(losses)
            }

        return report


if __name__ == "__main__":
    # Example usage and demonstration
    np.random.seed(42)

    # Generate sample data with different characteristics
    n_periods = 1000

    # Normal returns
    normal_returns = np.random.normal(0.001, 0.02, n_periods)

    # Fat-tailed returns (t-distribution)
    t_returns = np.random.standard_t(5, n_periods) * 0.02 + 0.001

    # Mixed returns with occasional extreme losses
    mixed_returns = np.random.normal(0.001, 0.015, n_periods)
    extreme_indices = np.random.choice(n_periods, size=20, replace=False)
    mixed_returns[extreme_indices] -= np.random.exponential(0.05, 20)

    print("Risk Measures Analysis")
    print("=" * 50)

    # Test coherent risk measures
    alpha = 0.05
    test_data = mixed_returns

    print(f"\nCoherent Risk Measures (α = {alpha}):")
    print("-" * 40)

    # Expected Shortfall
    es_value = CoherentRiskMeasures.expected_shortfall(test_data, alpha)
    print(f"Expected Shortfall: {es_value:.4f}")

    # Worst Case Risk
    wcr_value = CoherentRiskMeasures.worst_case_risk(test_data)
    print(f"Worst Case Risk: {wcr_value:.4f}")

    # Entropic Risk Measure
    entropic_value = CoherentRiskMeasures.entropic_risk_measure(test_data, theta=1.0)
    print(f"Entropic Risk (θ=1): {entropic_value:.4f}")

    # Spectral Risk Measure with exponential weighting
    def exponential_phi(u):
        gamma = 2.0
        return gamma * np.exp(gamma * u) / (np.exp(gamma) - 1)

    spectral_value = CoherentRiskMeasures.spectral_risk_measure(test_data, exponential_phi)
    print(f"Spectral Risk (exponential): {spectral_value:.4f}")

    # Risk measure comparison
    print(f"\nRisk Measure Comparison:")
    print("-" * 30)

    # Define risk measures to compare
    risk_measures = {
        'Historical_VaR': lambda x: HistoricalVaR(alpha).calculate_var(x),
        'Parametric_VaR': lambda x: ParametricVaR(alpha).calculate_var(x),
        'Expected_Shortfall': lambda x: CoherentRiskMeasures.expected_shortfall(x, alpha),
        'Entropic_Risk': lambda x: CoherentRiskMeasures.entropic_risk_measure(x, 1.0),
    }

    # Compare risk measures
    validator = RiskMeasureValidator(lambda x: CoherentRiskMeasures.expected_shortfall(x, alpha))
    comparison = validator.compare_risk_measures(test_data, risk_measures)

    print("Risk Values:")
    for name, value in comparison['risk_values'].items():
        print(f"  {name}: {value:.4f}")

    print(f"\nConservatism Ranking (most to least conservative):")
    for i, name in enumerate(comparison['conservatism_ranking'], 1):
        print(f"  {i}. {name}")

    # Coherence properties testing
    print(f"\nCoherence Properties Test:")
    print("-" * 30)

    for measure_name, tests in comparison['coherence_tests'].items():
        if 'error' not in tests:
            print(f"{measure_name}:")
            for prop, passed in tests.items():
                status = "✓" if passed else "✗"
                print(f"  {prop}: {status}")
        print()

    # Backtesting example
    print(f"Backtesting Analysis:")
    print("-" * 20)

    # Generate out-of-sample data for backtesting
    oos_returns = np.random.standard_t(5, 100) * 0.02 + 0.001

    # Calculate risk estimates using Expected Shortfall
    es_calculator = ExpectedShortfall(alpha, method='historical')
    risk_estimates = []

    window_size = 250
    for i in range(len(oos_returns)):
        if i < window_size:
            # Use all available data if insufficient history
            estimation_data = np.concatenate([test_data[-(window_size-i):], oos_returns[:i]]) if i > 0 else test_data[-window_size:]
        else:
            # Use rolling window
            estimation_data = oos_returns[i-window_size:i]

        risk_estimates.append(es_calculator.calculate_cvar(estimation_data))

    risk_estimates = np.array(risk_estimates)

    # Perform backtesting
    backtest_results = validator.backtest_risk_measure(oos_returns, risk_estimates, alpha)

    print(f"Exceedance Rate: {backtest_results['exceedance_rate']:.3f} (Expected: {backtest_results['expected_rate']:.3f})")
    print(f"Kupiec Test: {'Pass' if backtest_results['kupiec_test_passed'] else 'Fail'} (p-value: {backtest_results['kupiec_pvalue']:.3f})")

    if backtest_results['independence_test_passed'] is not None:
        print(f"Independence Test: {'Pass' if backtest_results['independence_test_passed'] else 'Fail'}")
        print(f"Combined Test: {'Pass' if backtest_results['combined_test_passed'] else 'Fail'}")

    print(f"Traffic Light: {backtest_results['traffic_light']}")
    print(f"Average Exceedance Size: {backtest_results['avg_exceedance_size']:.4f}")

    # Model validation report
    print(f"\nModel Validation Report Summary:")
    print("-" * 35)

    validation_report = validator.model_validation_report(oos_returns, risk_estimates, alpha)

    print(f"Observations: {validation_report['summary']['n_observations']}")
    print(f"Mean Risk Estimate: {validation_report['summary']['mean_risk_estimate']:.4f}")

    if 'stability' in validation_report:
        cv = validation_report['stability']['coefficient_of_variation']
        print(f"Model Stability (CV): {cv:.3f}")

    if 'performance' in validation_report:
        pred_ability = validation_report['performance']['predictive_ability']
        print(f"Predictive Ability: {pred_ability:.3f}")

    print(f"\nRisk measures framework successfully demonstrated!")
    print(f"Analyzed {n_periods} periods with comprehensive coherence and backtesting validation.")