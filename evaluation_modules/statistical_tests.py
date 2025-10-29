"""
Statistical Tests Module

Implements comprehensive statistical testing framework for model validation,
including goodness-of-fit tests, model adequacy tests, and diagnostic procedures.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from scipy import stats
from scipy.stats import jarque_bera, shapiro, anderson, kstest, chi2_contingency
from scipy.optimize import minimize
import warnings
from abc import ABC, abstractmethod


class StatisticalTestSuite:
    """
    Comprehensive suite of statistical tests for model validation.

    Provides a unified interface for running multiple statistical tests
    and interpreting results in the context of financial model validation.
    """

    def __init__(self, significance_level: float = 0.05):
        """
        Initialize statistical test suite.

        Parameters:
        -----------
        significance_level : float
            Default significance level for hypothesis tests (default: 5%)
        """
        self.alpha = significance_level
        self.results_history = []

    def normality_tests(self, data: np.ndarray,
                       tests: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
        """
        Run comprehensive normality tests.

        Parameters:
        -----------
        data : array-like
            Data to test for normality
        tests : list, optional
            Specific tests to run. If None, runs all available tests.

        Returns:
        --------
        dict : Test results with statistics, p-values, and conclusions
        """
        if tests is None:
            tests = ['jarque_bera', 'shapiro', 'anderson', 'kolmogorov_smirnov']

        results = {}
        data = np.asarray(data).flatten()

        # Remove any invalid values
        data = data[np.isfinite(data)]

        if len(data) < 3:
            warnings.warn("Insufficient data for normality tests")
            return {}

        # Jarque-Bera test
        if 'jarque_bera' in tests:
            try:
                jb_stat, jb_pval = jarque_bera(data)
                results['jarque_bera'] = {
                    'statistic': jb_stat,
                    'p_value': jb_pval,
                    'reject_null': jb_pval < self.alpha,
                    'conclusion': 'Non-normal' if jb_pval < self.alpha else 'Normal',
                    'description': 'Tests for normality based on skewness and kurtosis'
                }
            except Exception as e:
                results['jarque_bera'] = {'error': str(e)}

        # Shapiro-Wilk test (for smaller samples)
        if 'shapiro' in tests and len(data) <= 5000:
            try:
                sw_stat, sw_pval = shapiro(data)
                results['shapiro'] = {
                    'statistic': sw_stat,
                    'p_value': sw_pval,
                    'reject_null': sw_pval < self.alpha,
                    'conclusion': 'Non-normal' if sw_pval < self.alpha else 'Normal',
                    'description': 'Shapiro-Wilk test for normality'
                }
            except Exception as e:
                results['shapiro'] = {'error': str(e)}

        # Anderson-Darling test
        if 'anderson' in tests:
            try:
                ad_result = anderson(data, dist='norm')
                # Use 5% critical value (index 2 for 5% significance level)
                critical_val = ad_result.critical_values[2] if len(ad_result.critical_values) > 2 else ad_result.critical_values[-1]
                results['anderson'] = {
                    'statistic': ad_result.statistic,
                    'critical_value': critical_val,
                    'reject_null': ad_result.statistic > critical_val,
                    'conclusion': 'Non-normal' if ad_result.statistic > critical_val else 'Normal',
                    'description': 'Anderson-Darling test for normality'
                }
            except Exception as e:
                results['anderson'] = {'error': str(e)}

        # Kolmogorov-Smirnov test against normal distribution
        if 'kolmogorov_smirnov' in tests:
            try:
                # Standardize data
                standardized = (data - np.mean(data)) / np.std(data, ddof=1)
                ks_stat, ks_pval = kstest(standardized, 'norm')
                results['kolmogorov_smirnov'] = {
                    'statistic': ks_stat,
                    'p_value': ks_pval,
                    'reject_null': ks_pval < self.alpha,
                    'conclusion': 'Non-normal' if ks_pval < self.alpha else 'Normal',
                    'description': 'Kolmogorov-Smirnov test against normal distribution'
                }
            except Exception as e:
                results['kolmogorov_smirnov'] = {'error': str(e)}

        self.results_history.append(('normality_tests', results))
        return results

    def independence_tests(self, data: np.ndarray,
                          max_lags: int = 10) -> Dict[str, Dict[str, Any]]:
        """
        Test for independence/serial correlation in data.

        Parameters:
        -----------
        data : array-like
            Time series data
        max_lags : int
            Maximum number of lags to test

        Returns:
        --------
        dict : Independence test results
        """
        data = np.asarray(data).flatten()
        data = data[np.isfinite(data)]

        if len(data) < max_lags + 10:
            warnings.warn("Insufficient data for independence tests")
            return {}

        results = {}

        # Ljung-Box test for serial correlation
        try:
            from statsmodels.stats.diagnostic import acorr_ljungbox
            lb_results = acorr_ljungbox(data, lags=max_lags, return_df=True)

            # Take the result at maximum lag
            lb_stat = lb_results['lb_stat'].iloc[-1]
            lb_pval = lb_results['lb_pvalue'].iloc[-1]

            results['ljung_box'] = {
                'statistic': lb_stat,
                'p_value': lb_pval,
                'reject_null': lb_pval < self.alpha,
                'conclusion': 'Serially correlated' if lb_pval < self.alpha else 'Independent',
                'description': f'Ljung-Box test for serial correlation up to {max_lags} lags',
                'lags_tested': max_lags
            }
        except ImportError:
            # Fallback implementation
            results['ljung_box'] = self._manual_ljung_box(data, max_lags)
        except Exception as e:
            results['ljung_box'] = {'error': str(e)}

        # Runs test for randomness
        try:
            runs_result = self._runs_test(data)
            results['runs_test'] = runs_result
        except Exception as e:
            results['runs_test'] = {'error': str(e)}

        self.results_history.append(('independence_tests', results))
        return results

    def _manual_ljung_box(self, data: np.ndarray, max_lags: int) -> Dict[str, Any]:
        """Manual implementation of Ljung-Box test."""
        n = len(data)
        data_mean = np.mean(data)

        # Calculate autocorrelations
        autocorrs = []
        for lag in range(1, max_lags + 1):
            if lag >= n:
                break

            numerator = np.sum((data[:-lag] - data_mean) * (data[lag:] - data_mean))
            denominator = np.sum((data - data_mean) ** 2)
            autocorr = numerator / denominator if denominator != 0 else 0
            autocorrs.append(autocorr)

        # Calculate Ljung-Box statistic
        lb_stat = 0
        for i, rho in enumerate(autocorrs, 1):
            lb_stat += (rho ** 2) / (n - i)

        lb_stat *= n * (n + 2)

        # Chi-square distribution with max_lags degrees of freedom
        p_value = 1 - stats.chi2.cdf(lb_stat, len(autocorrs))

        return {
            'statistic': lb_stat,
            'p_value': p_value,
            'reject_null': p_value < self.alpha,
            'conclusion': 'Serially correlated' if p_value < self.alpha else 'Independent',
            'description': f'Manual Ljung-Box test for {len(autocorrs)} lags'
        }

    def _runs_test(self, data: np.ndarray) -> Dict[str, Any]:
        """
        Runs test for randomness.

        Tests whether the sequence of values above/below median is random.
        """
        median = np.median(data)
        runs, n1, n2 = 0, 0, 0

        # Convert to sequence of above/below median
        sequence = data > median
        n1 = np.sum(sequence)  # Number of values above median
        n2 = len(data) - n1    # Number of values below median

        # Count runs
        if len(sequence) > 0:
            runs = 1
            for i in range(1, len(sequence)):
                if sequence[i] != sequence[i-1]:
                    runs += 1

        # Expected runs and variance under null hypothesis
        if n1 + n2 < 2:
            return {'error': 'Insufficient variation in data'}

        expected_runs = (2 * n1 * n2) / (n1 + n2) + 1
        var_runs = (2 * n1 * n2 * (2 * n1 * n2 - n1 - n2)) / ((n1 + n2) ** 2 * (n1 + n2 - 1))

        if var_runs <= 0:
            return {'error': 'Invalid variance calculation'}

        # Z-score
        z_score = (runs - expected_runs) / np.sqrt(var_runs)
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

        return {
            'statistic': z_score,
            'p_value': p_value,
            'runs_observed': runs,
            'runs_expected': expected_runs,
            'reject_null': p_value < self.alpha,
            'conclusion': 'Non-random' if p_value < self.alpha else 'Random',
            'description': 'Runs test for randomness'
        }

    def homoscedasticity_tests(self, residuals: np.ndarray,
                              fitted_values: Optional[np.ndarray] = None) -> Dict[str, Dict[str, Any]]:
        """
        Test for homoscedasticity (constant variance).

        Parameters:
        -----------
        residuals : array-like
            Model residuals
        fitted_values : array-like, optional
            Fitted values from model

        Returns:
        --------
        dict : Homoscedasticity test results
        """
        residuals = np.asarray(residuals).flatten()
        residuals = residuals[np.isfinite(residuals)]

        if len(residuals) < 10:
            warnings.warn("Insufficient data for homoscedasticity tests")
            return {}

        results = {}

        # Breusch-Pagan test (simplified version)
        if fitted_values is not None:
            fitted_values = np.asarray(fitted_values).flatten()
            fitted_values = fitted_values[np.isfinite(fitted_values)]

            if len(fitted_values) == len(residuals):
                try:
                    bp_result = self._breusch_pagan_test(residuals, fitted_values)
                    results['breusch_pagan'] = bp_result
                except Exception as e:
                    results['breusch_pagan'] = {'error': str(e)}

        # Goldfeld-Quandt test
        try:
            gq_result = self._goldfeld_quandt_test(residuals)
            results['goldfeld_quandt'] = gq_result
        except Exception as e:
            results['goldfeld_quandt'] = {'error': str(e)}

        self.results_history.append(('homoscedasticity_tests', results))
        return results

    def _breusch_pagan_test(self, residuals: np.ndarray,
                           fitted_values: np.ndarray) -> Dict[str, Any]:
        """Simplified Breusch-Pagan test for heteroscedasticity."""
        n = len(residuals)
        squared_residuals = residuals ** 2

        # Regress squared residuals on fitted values
        # Using simple correlation as proxy for regression
        correlation = np.corrcoef(squared_residuals, fitted_values)[0, 1]

        # LM statistic approximation
        lm_stat = n * (correlation ** 2)
        p_value = 1 - stats.chi2.cdf(lm_stat, 1)

        return {
            'statistic': lm_stat,
            'p_value': p_value,
            'reject_null': p_value < self.alpha,
            'conclusion': 'Heteroscedastic' if p_value < self.alpha else 'Homoscedastic',
            'description': 'Breusch-Pagan test for heteroscedasticity'
        }

    def _goldfeld_quandt_test(self, residuals: np.ndarray) -> Dict[str, Any]:
        """Goldfeld-Quandt test for heteroscedasticity."""
        n = len(residuals)
        if n < 20:
            return {'error': 'Insufficient data for Goldfeld-Quandt test'}

        # Sort residuals and split into groups
        sorted_residuals = np.sort(residuals ** 2)
        split_point = n // 3

        group1 = sorted_residuals[:split_point]
        group2 = sorted_residuals[-split_point:]

        # F-test for equal variances
        var1 = np.var(group1, ddof=1) if len(group1) > 1 else 0
        var2 = np.var(group2, ddof=1) if len(group2) > 1 else 0

        if var1 <= 0 or var2 <= 0:
            return {'error': 'Invalid variance calculation'}

        f_stat = var2 / var1
        df1, df2 = len(group2) - 1, len(group1) - 1
        p_value = 2 * min(stats.f.cdf(f_stat, df1, df2), 1 - stats.f.cdf(f_stat, df1, df2))

        return {
            'statistic': f_stat,
            'p_value': p_value,
            'reject_null': p_value < self.alpha,
            'conclusion': 'Heteroscedastic' if p_value < self.alpha else 'Homoscedastic',
            'description': 'Goldfeld-Quandt test for heteroscedasticity'
        }


class GoodnessOfFitTests:
    """
    Specialized goodness-of-fit tests for financial distributions.

    Includes tests for common financial distributions like normal,
    Student's t, skewed distributions, etc.
    """

    def __init__(self, significance_level: float = 0.05):
        """
        Initialize goodness-of-fit test suite.

        Parameters:
        -----------
        significance_level : float
            Significance level for tests
        """
        self.alpha = significance_level

    def distribution_fit_test(self, data: np.ndarray,
                             distribution: str = 'norm') -> Dict[str, Any]:
        """
        Test goodness-of-fit for specified distribution.

        Parameters:
        -----------
        data : array-like
            Data to test
        distribution : str
            Distribution to test against ('norm', 't', 'skewnorm', 'laplace')

        Returns:
        --------
        dict : Fit test results including parameters and statistics
        """
        data = np.asarray(data).flatten()
        data = data[np.isfinite(data)]

        if len(data) < 10:
            return {'error': 'Insufficient data for distribution fitting'}

        try:
            # Get distribution object
            if distribution == 'norm':
                dist = stats.norm
                params = stats.norm.fit(data)
            elif distribution == 't':
                dist = stats.t
                params = stats.t.fit(data)
            elif distribution == 'skewnorm':
                dist = stats.skewnorm
                params = stats.skewnorm.fit(data)
            elif distribution == 'laplace':
                dist = stats.laplace
                params = stats.laplace.fit(data)
            else:
                return {'error': f'Unknown distribution: {distribution}'}

            # Kolmogorov-Smirnov test
            ks_stat, ks_pval = kstest(data, lambda x: dist.cdf(x, *params))

            # Anderson-Darling test (if available)
            ad_stat, ad_pval = None, None
            try:
                if distribution == 'norm':
                    ad_result = anderson(data, dist='norm')
                    ad_stat = ad_result.statistic
                    # Approximate p-value
                    ad_pval = self._anderson_darling_pvalue(ad_stat)
            except:
                pass

            # Calculate log-likelihood and AIC/BIC
            log_likelihood = np.sum(dist.logpdf(data, *params))
            k = len(params)  # Number of parameters
            n = len(data)
            aic = 2 * k - 2 * log_likelihood
            bic = k * np.log(n) - 2 * log_likelihood

            return {
                'distribution': distribution,
                'parameters': params,
                'log_likelihood': log_likelihood,
                'aic': aic,
                'bic': bic,
                'ks_statistic': ks_stat,
                'ks_p_value': ks_pval,
                'ks_reject_null': ks_pval < self.alpha,
                'ad_statistic': ad_stat,
                'ad_p_value': ad_pval,
                'ad_reject_null': ad_pval < self.alpha if ad_pval is not None else None,
                'conclusion': 'Poor fit' if ks_pval < self.alpha else 'Good fit'
            }

        except Exception as e:
            return {'error': str(e)}

    def _anderson_darling_pvalue(self, statistic: float) -> float:
        """Approximate p-value for Anderson-Darling statistic."""
        # Approximation for normal distribution
        if statistic < 0.2:
            return 1 - np.exp(-13.436 + 101.14 * statistic - 223.73 * statistic**2)
        elif statistic < 0.34:
            return 1 - np.exp(-8.318 + 42.796 * statistic - 59.938 * statistic**2)
        elif statistic < 0.6:
            return np.exp(0.9177 - 4.279 * statistic - 1.38 * statistic**2)
        else:
            return np.exp(1.2937 - 5.709 * statistic + 0.0186 * statistic**2)

    def compare_distributions(self, data: np.ndarray,
                             distributions: List[str] = None) -> Dict[str, Dict[str, Any]]:
        """
        Compare multiple distribution fits and rank them.

        Parameters:
        -----------
        data : array-like
            Data to fit
        distributions : list, optional
            List of distributions to compare

        Returns:
        --------
        dict : Comparison results with rankings
        """
        if distributions is None:
            distributions = ['norm', 't', 'skewnorm', 'laplace']

        results = {}
        for dist in distributions:
            results[dist] = self.distribution_fit_test(data, dist)

        # Rank by AIC (lower is better)
        valid_results = {k: v for k, v in results.items() if 'aic' in v}
        if valid_results:
            sorted_by_aic = sorted(valid_results.items(), key=lambda x: x[1]['aic'])
            for i, (dist, result) in enumerate(sorted_by_aic):
                results[dist]['aic_rank'] = i + 1

        # Rank by BIC (lower is better)
        valid_results = {k: v for k, v in results.items() if 'bic' in v}
        if valid_results:
            sorted_by_bic = sorted(valid_results.items(), key=lambda x: x[1]['bic'])
            for i, (dist, result) in enumerate(sorted_by_bic):
                results[dist]['bic_rank'] = i + 1

        return results


class ModelAdequacyTests:
    """
    Specialized tests for evaluating model adequacy in financial contexts.

    Includes backtesting procedures, model stability tests,
    and parameter stability analysis.
    """

    def __init__(self, significance_level: float = 0.05):
        """
        Initialize model adequacy test suite.

        Parameters:
        -----------
        significance_level : float
            Significance level for tests
        """
        self.alpha = significance_level

    def kupiec_test(self, violations: np.ndarray,
                   total_observations: int,
                   expected_violation_rate: float) -> Dict[str, Any]:
        """
        Kupiec POF (Proportion of Failures) test for VaR backtesting.

        Parameters:
        -----------
        violations : array-like
            Binary array indicating violations (1) or no violations (0)
        total_observations : int
            Total number of observations
        expected_violation_rate : float
            Expected violation rate (e.g., 0.05 for 5% VaR)

        Returns:
        --------
        dict : Kupiec test results
        """
        violations = np.asarray(violations).flatten()
        n_violations = np.sum(violations)
        n_total = total_observations

        if n_total == 0:
            return {'error': 'No observations provided'}

        actual_rate = n_violations / n_total

        # Likelihood ratio statistic
        if n_violations == 0 or n_violations == n_total:
            lr_stat = 0  # Boundary case
        else:
            try:
                likelihood_unrestricted = (
                    (actual_rate ** n_violations) *
                    ((1 - actual_rate) ** (n_total - n_violations))
                )
                likelihood_restricted = (
                    (expected_violation_rate ** n_violations) *
                    ((1 - expected_violation_rate) ** (n_total - n_violations))
                )

                if likelihood_restricted > 0:
                    lr_stat = -2 * np.log(likelihood_restricted / likelihood_unrestricted)
                else:
                    lr_stat = np.inf
            except (ZeroDivisionError, ValueError):
                lr_stat = np.inf

        # Chi-square test with 1 degree of freedom
        p_value = 1 - stats.chi2.cdf(lr_stat, 1) if lr_stat < np.inf else 0

        return {
            'statistic': lr_stat,
            'p_value': p_value,
            'reject_null': p_value < self.alpha,
            'conclusion': 'Model inadequate' if p_value < self.alpha else 'Model adequate',
            'violations_observed': n_violations,
            'violations_expected': expected_violation_rate * n_total,
            'actual_violation_rate': actual_rate,
            'expected_violation_rate': expected_violation_rate,
            'description': 'Kupiec POF test for VaR model adequacy'
        }

    def christoffersen_test(self, violations: np.ndarray) -> Dict[str, Any]:
        """
        Christoffersen conditional coverage test.

        Tests for independence of violations in addition to correct coverage.

        Parameters:
        -----------
        violations : array-like
            Binary sequence of violations

        Returns:
        --------
        dict : Christoffersen test results
        """
        violations = np.asarray(violations).astype(int).flatten()
        n = len(violations)

        if n < 10:
            return {'error': 'Insufficient data for Christoffersen test'}

        # Count transitions
        n00 = n01 = n10 = n11 = 0

        for i in range(n - 1):
            if violations[i] == 0 and violations[i + 1] == 0:
                n00 += 1
            elif violations[i] == 0 and violations[i + 1] == 1:
                n01 += 1
            elif violations[i] == 1 and violations[i + 1] == 0:
                n10 += 1
            elif violations[i] == 1 and violations[i + 1] == 1:
                n11 += 1

        # Calculate probabilities
        n0 = n00 + n01
        n1 = n10 + n11

        if n0 == 0 or n1 == 0:
            return {'error': 'Insufficient variation in violations'}

        # Independence test
        try:
            pi_01 = n01 / n0 if n0 > 0 else 0
            pi_11 = n11 / n1 if n1 > 0 else 0
            pi = (n01 + n11) / (n - 1)

            # Likelihood ratio for independence
            if pi_01 > 0 and pi_11 > 0 and pi > 0 and (1 - pi) > 0:
                lr_ind = -2 * (
                    n00 * np.log(1 - pi) + n01 * np.log(pi) +
                    n10 * np.log(1 - pi) + n11 * np.log(pi) -
                    n00 * np.log(1 - pi_01) - n01 * np.log(pi_01) -
                    n10 * np.log(1 - pi_11) - n11 * np.log(pi_11)
                )
            else:
                lr_ind = 0

            p_value_ind = 1 - stats.chi2.cdf(lr_ind, 1) if lr_ind > 0 else 1

        except (ValueError, ZeroDivisionError):
            lr_ind = 0
            p_value_ind = 1

        return {
            'independence_statistic': lr_ind,
            'independence_p_value': p_value_ind,
            'reject_independence': p_value_ind < self.alpha,
            'conclusion': 'Violations not independent' if p_value_ind < self.alpha else 'Violations independent',
            'transition_counts': {'n00': n00, 'n01': n01, 'n10': n10, 'n11': n11},
            'description': 'Christoffersen test for violation independence'
        }

    def parameter_stability_test(self, parameters: np.ndarray,
                                window_size: int = 50) -> Dict[str, Any]:
        """
        Test for parameter stability over time using rolling windows.

        Parameters:
        -----------
        parameters : array-like
            Time series of parameter estimates
        window_size : int
            Size of rolling window for stability analysis

        Returns:
        --------
        dict : Parameter stability test results
        """
        parameters = np.asarray(parameters).flatten()
        n = len(parameters)

        if n < 2 * window_size:
            return {'error': f'Insufficient data for window size {window_size}'}

        # Calculate rolling statistics
        rolling_means = []
        rolling_stds = []

        for i in range(window_size, n - window_size + 1):
            window = parameters[i - window_size:i + window_size]
            rolling_means.append(np.mean(window))
            rolling_stds.append(np.std(window, ddof=1))

        rolling_means = np.array(rolling_means)
        rolling_stds = np.array(rolling_stds)

        # Test for trend in rolling means (Mann-Kendall test approximation)
        n_windows = len(rolling_means)
        if n_windows < 3:
            return {'error': 'Insufficient windows for stability test'}

        # Simple trend test using linear regression
        x = np.arange(n_windows)
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, rolling_means)

        # Variance stability test
        variance_changes = np.diff(rolling_stds)
        variance_stability_stat = np.mean(np.abs(variance_changes)) / np.mean(rolling_stds)

        return {
            'trend_slope': slope,
            'trend_p_value': p_value,
            'trend_significant': p_value < self.alpha,
            'variance_stability_ratio': variance_stability_stat,
            'mean_parameter': np.mean(parameters),
            'parameter_std': np.std(parameters, ddof=1),
            'coefficient_of_variation': np.std(parameters, ddof=1) / np.mean(parameters) if np.mean(parameters) != 0 else np.inf,
            'conclusion': 'Parameters unstable' if p_value < self.alpha else 'Parameters stable',
            'description': 'Parameter stability analysis using rolling windows'
        }

    def model_comparison_test(self, errors1: np.ndarray,
                             errors2: np.ndarray) -> Dict[str, Any]:
        """
        Compare two models using Diebold-Mariano test.

        Parameters:
        -----------
        errors1 : array-like
            Forecast errors from model 1
        errors2 : array-like
            Forecast errors from model 2

        Returns:
        --------
        dict : Model comparison test results
        """
        errors1 = np.asarray(errors1).flatten()
        errors2 = np.asarray(errors2).flatten()

        if len(errors1) != len(errors2):
            return {'error': 'Error series must have same length'}

        if len(errors1) < 10:
            return {'error': 'Insufficient data for model comparison'}

        # Loss differential
        loss_diff = errors1**2 - errors2**2
        mean_diff = np.mean(loss_diff)
        std_diff = np.std(loss_diff, ddof=1)

        if std_diff == 0:
            return {'error': 'No variation in loss differential'}

        # Diebold-Mariano statistic
        dm_stat = mean_diff / (std_diff / np.sqrt(len(loss_diff)))
        p_value = 2 * (1 - stats.t.cdf(abs(dm_stat), len(loss_diff) - 1))

        return {
            'dm_statistic': dm_stat,
            'p_value': p_value,
            'reject_null': p_value < self.alpha,
            'conclusion': 'Models significantly different' if p_value < self.alpha else 'No significant difference',
            'mean_loss_diff': mean_diff,
            'preferred_model': 'Model 1' if mean_diff < 0 else 'Model 2',
            'description': 'Diebold-Mariano test for model comparison'
        }