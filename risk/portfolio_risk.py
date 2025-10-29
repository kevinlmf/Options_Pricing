"""
Portfolio Risk Management

Implementation of portfolio-level risk metrics and attribution methods.
Follows the mathematical framework: Pure Math → Applied Math → Risk Management

Mathematical Foundation:
- Pure Math: Linear algebra, multivariate statistics, optimization theory
- Applied Math: Portfolio theory, factor models, risk decomposition
- Risk Management: Risk attribution, concentration metrics, diversification measures
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Tuple, Any
from scipy import linalg, stats
from scipy.optimize import minimize
import warnings
from .var_models import VarCalculator, HistoricalVaR, ParametricVaR
from .cvar_models import ExpectedShortfall


class PortfolioRiskMetrics:
    """
    Comprehensive portfolio risk analysis including traditional and modern risk metrics.
    """

    def __init__(self, returns: np.ndarray, weights: np.ndarray,
                 asset_names: Optional[List[str]] = None,
                 benchmark_returns: Optional[np.ndarray] = None):
        """
        Initialize portfolio risk metrics calculator.

        Args:
            returns: Asset returns matrix (assets x time) or (time x assets)
            weights: Portfolio weights
            asset_names: Asset names for reporting
            benchmark_returns: Benchmark returns for relative metrics
        """
        # Ensure returns are in (assets x time) format
        if returns.ndim == 1:
            returns = returns.reshape(1, -1)
        elif returns.shape[0] > returns.shape[1]:
            returns = returns.T  # Transpose if time x assets

        self.returns = returns
        self.n_assets, self.n_periods = returns.shape

        if len(weights) != self.n_assets:
            raise ValueError("Weights length must match number of assets")

        self.weights = np.array(weights)
        self.asset_names = asset_names or [f"Asset_{i+1}" for i in range(self.n_assets)]
        self.benchmark_returns = benchmark_returns

        # Calculate portfolio returns
        self.portfolio_returns = np.dot(self.weights, self.returns)

        # Precompute common statistics
        self._compute_basic_statistics()

    def _compute_basic_statistics(self):
        """Precompute basic portfolio statistics"""
        self.mean_returns = np.mean(self.returns, axis=1)
        self.portfolio_mean = np.dot(self.weights, self.mean_returns)
        self.cov_matrix = np.cov(self.returns)
        self.portfolio_variance = np.dot(self.weights, np.dot(self.cov_matrix, self.weights))
        self.portfolio_volatility = np.sqrt(self.portfolio_variance)

    def get_basic_metrics(self) -> Dict[str, float]:
        """Calculate basic portfolio risk metrics"""
        portfolio_returns = self.portfolio_returns

        metrics = {
            'expected_return': self.portfolio_mean,
            'volatility': self.portfolio_volatility,
            'variance': self.portfolio_variance,
            'skewness': stats.skew(portfolio_returns),
            'kurtosis': stats.kurtosis(portfolio_returns),
            'min_return': np.min(portfolio_returns),
            'max_return': np.max(portfolio_returns),
        }

        # Add percentiles
        percentiles = [1, 5, 10, 25, 75, 90, 95, 99]
        for p in percentiles:
            metrics[f'percentile_{p}'] = np.percentile(portfolio_returns, p)

        return metrics

    def get_downside_metrics(self, target_return: float = 0.0) -> Dict[str, float]:
        """Calculate downside risk metrics"""
        portfolio_returns = self.portfolio_returns
        downside_returns = portfolio_returns[portfolio_returns < target_return]

        if len(downside_returns) == 0:
            downside_deviation = 0
            downside_variance = 0
        else:
            downside_deviations = target_return - downside_returns
            downside_variance = np.mean(downside_deviations ** 2)
            downside_deviation = np.sqrt(downside_variance)

        # Sortino ratio
        excess_return = self.portfolio_mean - target_return
        sortino_ratio = excess_return / downside_deviation if downside_deviation > 0 else np.inf

        return {
            'downside_deviation': downside_deviation,
            'downside_variance': downside_variance,
            'sortino_ratio': sortino_ratio,
            'downside_frequency': len(downside_returns) / len(portfolio_returns),
        }

    def get_var_cvar_metrics(self, confidence_levels: List[float] = [0.01, 0.05, 0.10],
                           methods: List[str] = ['historical', 'parametric']) -> Dict[str, Dict[str, float]]:
        """Calculate VaR and CVaR metrics at multiple confidence levels"""
        metrics = {}

        for confidence_level in confidence_levels:
            level_key = f"{int((1-confidence_level)*100)}%"
            metrics[level_key] = {}

            for method in methods:
                if method == 'historical':
                    var_calc = HistoricalVaR(confidence_level)
                    cvar_calc = ExpectedShortfall(confidence_level, 'historical')
                elif method == 'parametric':
                    var_calc = ParametricVaR(confidence_level)
                    cvar_calc = ExpectedShortfall(confidence_level, 'parametric')
                else:
                    continue

                var_value = var_calc.calculate_var(self.portfolio_returns)
                cvar_value = cvar_calc.calculate_cvar(self.portfolio_returns)

                metrics[level_key][f'var_{method}'] = var_value
                metrics[level_key][f'cvar_{method}'] = cvar_value
                metrics[level_key][f'cvar_var_ratio_{method}'] = cvar_value / var_value if var_value != 0 else 1

        return metrics

    def get_concentration_metrics(self) -> Dict[str, float]:
        """Calculate portfolio concentration metrics"""
        # Herfindahl-Hirschman Index (HHI)
        hhi = np.sum(self.weights ** 2)

        # Effective number of assets
        effective_assets = 1 / hhi if hhi > 0 else 0

        # Maximum weight
        max_weight = np.max(np.abs(self.weights))

        # Concentration ratio (top N assets)
        sorted_weights = np.sort(np.abs(self.weights))[::-1]
        cr3 = np.sum(sorted_weights[:min(3, len(sorted_weights))])  # Top 3
        cr5 = np.sum(sorted_weights[:min(5, len(sorted_weights))])  # Top 5

        # Gini coefficient for weight distribution
        def gini_coefficient(weights):
            abs_weights = np.abs(weights)
            abs_weights = abs_weights / np.sum(abs_weights)  # Normalize
            sorted_weights = np.sort(abs_weights)
            n = len(sorted_weights)
            index = np.arange(1, n + 1)
            return (np.sum((2 * index - n - 1) * sorted_weights)) / (n * np.sum(sorted_weights))

        gini = gini_coefficient(self.weights)

        return {
            'herfindahl_index': hhi,
            'effective_num_assets': effective_assets,
            'max_weight': max_weight,
            'concentration_ratio_3': cr3,
            'concentration_ratio_5': cr5,
            'gini_coefficient': gini,
        }

    def get_diversification_metrics(self) -> Dict[str, float]:
        """Calculate diversification metrics"""
        # Diversification ratio: weighted average volatility / portfolio volatility
        asset_volatilities = np.sqrt(np.diag(self.cov_matrix))
        weighted_avg_vol = np.dot(np.abs(self.weights), asset_volatilities)
        diversification_ratio = weighted_avg_vol / self.portfolio_volatility

        # Maximum diversification ratio (theoretical maximum)
        min_var_weights = self._calculate_minimum_variance_weights()
        min_var_portfolio_vol = np.sqrt(np.dot(min_var_weights, np.dot(self.cov_matrix, min_var_weights)))
        min_var_weighted_avg_vol = np.dot(min_var_weights, asset_volatilities)
        max_diversification_ratio = min_var_weighted_avg_vol / min_var_portfolio_vol

        # Diversification efficiency
        diversification_efficiency = diversification_ratio / max_diversification_ratio

        return {
            'diversification_ratio': diversification_ratio,
            'max_diversification_ratio': max_diversification_ratio,
            'diversification_efficiency': diversification_efficiency,
        }

    def _calculate_minimum_variance_weights(self) -> np.ndarray:
        """Calculate minimum variance portfolio weights"""
        try:
            inv_cov = linalg.inv(self.cov_matrix)
            ones = np.ones((self.n_assets, 1))
            min_var_weights = np.dot(inv_cov, ones) / np.dot(ones.T, np.dot(inv_cov, ones))
            return min_var_weights.flatten()
        except linalg.LinAlgError:
            # Fallback: equal weights if covariance matrix is singular
            return np.ones(self.n_assets) / self.n_assets

    def get_factor_exposure_metrics(self, factor_returns: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Calculate factor exposure and attribution metrics.

        Args:
            factor_returns: Factor returns matrix (factors x time)
        """
        if factor_returns is None:
            # Use first principal components as factors
            U, s, Vt = linalg.svd(self.returns, full_matrices=False)
            n_factors = min(5, self.n_assets)
            factor_returns = Vt[:n_factors, :]
            factor_names = [f'PC_{i+1}' for i in range(n_factors)]
        else:
            if factor_returns.ndim == 1:
                factor_returns = factor_returns.reshape(1, -1)
            elif factor_returns.shape[0] > factor_returns.shape[1]:
                factor_returns = factor_returns.T

            n_factors = factor_returns.shape[0]
            factor_names = [f'Factor_{i+1}' for i in range(n_factors)]

        # Portfolio factor exposures (beta)
        portfolio_factor_exposures = np.zeros(n_factors)
        factor_variances = np.var(factor_returns, axis=1)

        for i in range(n_factors):
            covariance = np.cov(self.portfolio_returns, factor_returns[i, :])[0, 1]
            portfolio_factor_exposures[i] = covariance / factor_variances[i]

        # Asset factor exposures
        asset_factor_exposures = np.zeros((self.n_assets, n_factors))
        for i in range(self.n_assets):
            for j in range(n_factors):
                covariance = np.cov(self.returns[i, :], factor_returns[j, :])[0, 1]
                asset_factor_exposures[i, j] = covariance / factor_variances[j]

        # Factor contribution to portfolio risk
        factor_contributions = np.zeros(n_factors)
        for i in range(n_factors):
            factor_portfolio_exposure = np.dot(self.weights, asset_factor_exposures[:, i])
            factor_contributions[i] = (factor_portfolio_exposure ** 2 * factor_variances[i]) / self.portfolio_variance

        return {
            'portfolio_factor_exposures': dict(zip(factor_names, portfolio_factor_exposures)),
            'asset_factor_exposures': {
                self.asset_names[i]: dict(zip(factor_names, asset_factor_exposures[i, :]))
                for i in range(self.n_assets)
            },
            'factor_risk_contributions': dict(zip(factor_names, factor_contributions)),
        }

    def stress_test(self, stress_scenarios: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
        """
        Perform stress testing on the portfolio.

        Args:
            stress_scenarios: Dictionary of scenario names and return vectors

        Returns:
            Stress test results for each scenario
        """
        results = {}

        for scenario_name, scenario_returns in stress_scenarios.items():
            if len(scenario_returns) != self.n_assets:
                raise ValueError(f"Scenario {scenario_name} must have returns for all {self.n_assets} assets")

            # Portfolio return under stress
            portfolio_stress_return = np.dot(self.weights, scenario_returns)

            # Asset contributions to stress
            asset_contributions = self.weights * scenario_returns

            results[scenario_name] = {
                'portfolio_return': portfolio_stress_return,
                'asset_contributions': dict(zip(self.asset_names, asset_contributions)),
                'worst_contributor': self.asset_names[np.argmin(asset_contributions)],
                'best_contributor': self.asset_names[np.argmax(asset_contributions)],
            }

        return results

    def generate_comprehensive_report(self, portfolio_value: float = 1.0,
                                    factor_returns: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Generate comprehensive portfolio risk report"""
        report = {
            'portfolio_summary': {
                'num_assets': self.n_assets,
                'num_periods': self.n_periods,
                'portfolio_value': portfolio_value,
                'weights': dict(zip(self.asset_names, self.weights)),
            },
            'basic_metrics': self.get_basic_metrics(),
            'downside_metrics': self.get_downside_metrics(),
            'var_cvar_metrics': self.get_var_cvar_metrics(),
            'concentration_metrics': self.get_concentration_metrics(),
            'diversification_metrics': self.get_diversification_metrics(),
        }

        # Add factor analysis if factor returns provided
        if factor_returns is not None:
            report['factor_metrics'] = self.get_factor_exposure_metrics(factor_returns)

        # Scale monetary metrics by portfolio value
        for metric_group in ['var_cvar_metrics']:
            if metric_group in report:
                for level in report[metric_group]:
                    for key in report[metric_group][level]:
                        if 'var' in key or 'cvar' in key:
                            if 'ratio' not in key:
                                report[metric_group][level][key] *= portfolio_value

        return report


class RiskAttribution:
    """
    Risk attribution and decomposition methods for portfolios.
    """

    def __init__(self, returns: np.ndarray, weights: np.ndarray,
                 asset_names: Optional[List[str]] = None):
        """
        Initialize risk attribution calculator.

        Args:
            returns: Asset returns matrix (assets x time)
            weights: Portfolio weights
            asset_names: Asset names
        """
        if returns.ndim == 1:
            returns = returns.reshape(1, -1)
        elif returns.shape[0] > returns.shape[1]:
            returns = returns.T

        self.returns = returns
        self.weights = np.array(weights)
        self.n_assets = len(weights)
        self.asset_names = asset_names or [f"Asset_{i+1}" for i in range(self.n_assets)]

        self.cov_matrix = np.cov(returns)
        self.portfolio_variance = np.dot(weights, np.dot(self.cov_matrix, weights))

    def marginal_contribution_to_risk(self) -> Dict[str, float]:
        """Calculate marginal contribution to risk (MCTR)"""
        # MCTR = ∂σ_p/∂w_i = (Σw)_i / σ_p
        portfolio_vol = np.sqrt(self.portfolio_variance)
        cov_times_weights = np.dot(self.cov_matrix, self.weights)
        mctr = cov_times_weights / portfolio_vol

        return dict(zip(self.asset_names, mctr))

    def component_contribution_to_risk(self) -> Dict[str, float]:
        """Calculate component contribution to risk (CCTR)"""
        # CCTR = w_i * MCTR_i
        mctr = self.marginal_contribution_to_risk()
        cctr = {asset: self.weights[i] * mctr[asset]
                for i, asset in enumerate(self.asset_names)}

        return cctr

    def percentage_contribution_to_risk(self) -> Dict[str, float]:
        """Calculate percentage contribution to risk (PCTR)"""
        # PCTR = CCTR_i / σ_p
        cctr = self.component_contribution_to_risk()
        portfolio_vol = np.sqrt(self.portfolio_variance)
        pctr = {asset: cctr[asset] / portfolio_vol
                for asset in self.asset_names}

        return pctr

    def risk_budgeting_analysis(self, target_risk_budgets: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Analyze risk budgets vs actual risk contributions.

        Args:
            target_risk_budgets: Target risk budget percentages for each asset

        Returns:
            Risk budgeting analysis results
        """
        pctr = self.percentage_contribution_to_risk()

        if target_risk_budgets is None:
            # Equal risk budgeting
            target_risk_budgets = {asset: 1.0 / self.n_assets for asset in self.asset_names}

        # Ensure target budgets sum to 1
        total_target = sum(target_risk_budgets.values())
        if abs(total_target - 1.0) > 1e-6:
            target_risk_budgets = {asset: budget / total_target
                                 for asset, budget in target_risk_budgets.items()}

        # Calculate deviations
        deviations = {asset: pctr[asset] - target_risk_budgets.get(asset, 0)
                     for asset in self.asset_names}

        # Risk budget efficiency
        sum_squared_deviations = sum(dev ** 2 for dev in deviations.values())
        risk_budget_efficiency = 1 - sum_squared_deviations

        return {
            'actual_risk_contributions': pctr,
            'target_risk_budgets': target_risk_budgets,
            'deviations': deviations,
            'sum_squared_deviations': sum_squared_deviations,
            'risk_budget_efficiency': risk_budget_efficiency,
        }

    def sector_risk_attribution(self, sector_mapping: Dict[str, str]) -> Dict[str, Dict[str, float]]:
        """
        Attribute risk to sectors.

        Args:
            sector_mapping: Mapping of asset names to sector names

        Returns:
            Sector-level risk attribution
        """
        # Get individual asset contributions
        pctr = self.percentage_contribution_to_risk()
        cctr = self.component_contribution_to_risk()

        # Aggregate by sector
        sectors = set(sector_mapping.values())
        sector_pctr = {}
        sector_cctr = {}
        sector_weights = {}

        for sector in sectors:
            sector_assets = [asset for asset, sec in sector_mapping.items() if sec == sector]
            sector_pctr[sector] = sum(pctr.get(asset, 0) for asset in sector_assets)
            sector_cctr[sector] = sum(cctr.get(asset, 0) for asset in sector_assets)
            sector_weights[sector] = sum(self.weights[self.asset_names.index(asset)]
                                       for asset in sector_assets if asset in self.asset_names)

        return {
            'sector_percentage_contributions': sector_pctr,
            'sector_component_contributions': sector_cctr,
            'sector_weights': sector_weights,
        }

    def calculate_tracking_error_attribution(self, benchmark_returns: np.ndarray,
                                           benchmark_weights: np.ndarray) -> Dict[str, Any]:
        """
        Calculate tracking error attribution vs benchmark.

        Args:
            benchmark_returns: Benchmark asset returns
            benchmark_weights: Benchmark weights

        Returns:
            Tracking error attribution results
        """
        if benchmark_returns.shape != self.returns.shape:
            raise ValueError("Benchmark returns must have same shape as portfolio returns")

        if len(benchmark_weights) != self.n_assets:
            raise ValueError("Benchmark weights must match number of assets")

        # Active weights
        active_weights = self.weights - benchmark_weights

        # Active returns (portfolio - benchmark)
        portfolio_returns = np.dot(self.weights, self.returns)
        benchmark_portfolio_returns = np.dot(benchmark_weights, self.returns)
        active_returns = portfolio_returns - benchmark_portfolio_returns

        # Tracking error (volatility of active returns)
        tracking_error = np.std(active_returns, ddof=1)

        # Asset-level attribution to tracking error
        active_return_contributions = {}
        for i, asset in enumerate(self.asset_names):
            # Contribution = active_weight * (asset_return - benchmark_return)
            asset_active_returns = active_weights[i] * self.returns[i, :]
            active_return_contributions[asset] = np.mean(asset_active_returns)

        return {
            'tracking_error': tracking_error,
            'active_weights': dict(zip(self.asset_names, active_weights)),
            'active_return_contributions': active_return_contributions,
            'total_active_return': np.mean(active_returns),
        }


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)

    # Generate sample multi-asset portfolio data
    n_assets = 5
    n_periods = 1000
    asset_names = ['US Equities', 'EU Equities', 'Bonds', 'Commodities', 'REITs']

    # Generate correlated returns
    correlation_matrix = np.array([
        [1.0, 0.7, -0.2, 0.3, 0.5],
        [0.7, 1.0, -0.1, 0.2, 0.4],
        [-0.2, -0.1, 1.0, 0.1, 0.2],
        [0.3, 0.2, 0.1, 1.0, 0.3],
        [0.5, 0.4, 0.2, 0.3, 1.0]
    ])

    # Generate factor structure
    L = linalg.cholesky(correlation_matrix)
    independent_shocks = np.random.standard_normal((n_assets, n_periods))
    correlated_shocks = np.dot(L, independent_shocks)

    # Add means and volatilities
    mean_returns = np.array([0.0008, 0.0006, 0.0003, 0.0005, 0.0007])  # Daily
    volatilities = np.array([0.015, 0.018, 0.008, 0.025, 0.020])  # Daily

    returns = mean_returns[:, np.newaxis] + volatilities[:, np.newaxis] * correlated_shocks

    # Portfolio weights
    weights = np.array([0.30, 0.25, 0.20, 0.15, 0.10])
    portfolio_value = 10000000  # $10M

    print("Portfolio Risk Analysis")
    print("=" * 60)

    # Initialize portfolio risk metrics
    risk_metrics = PortfolioRiskMetrics(returns, weights, asset_names)

    # Basic metrics
    basic = risk_metrics.get_basic_metrics()
    print(f"\nBasic Metrics:")
    print(f"Expected Return (annual): {basic['expected_return'] * 252:.2%}")
    print(f"Volatility (annual): {basic['volatility'] * np.sqrt(252):.2%}")
    print(f"Skewness: {basic['skewness']:.3f}")
    print(f"Kurtosis: {basic['kurtosis']:.3f}")

    # VaR/CVaR metrics
    var_cvar = risk_metrics.get_var_cvar_metrics()
    print(f"\nVaR/CVaR Metrics (Portfolio Value: ${portfolio_value:,}):")
    for level, metrics in var_cvar.items():
        print(f"  {level} Confidence:")
        for method in ['historical', 'parametric']:
            var_val = metrics[f'var_{method}'] * portfolio_value
            cvar_val = metrics[f'cvar_{method}'] * portfolio_value
            ratio = metrics[f'cvar_var_ratio_{method}']
            print(f"    {method.title()}: VaR=${var_val:,.0f}, CVaR=${cvar_val:,.0f}, Ratio={ratio:.2f}")

    # Concentration metrics
    concentration = risk_metrics.get_concentration_metrics()
    print(f"\nConcentration Metrics:")
    print(f"Herfindahl Index: {concentration['herfindahl_index']:.3f}")
    print(f"Effective # Assets: {concentration['effective_num_assets']:.1f}")
    print(f"Max Weight: {concentration['max_weight']:.1%}")

    # Diversification metrics
    diversification = risk_metrics.get_diversification_metrics()
    print(f"\nDiversification Metrics:")
    print(f"Diversification Ratio: {diversification['diversification_ratio']:.3f}")
    print(f"Diversification Efficiency: {diversification['diversification_efficiency']:.1%}")

    # Risk Attribution
    print(f"\nRisk Attribution Analysis:")
    print("-" * 40)

    risk_attr = RiskAttribution(returns, weights, asset_names)

    # Component contributions
    cctr = risk_attr.component_contribution_to_risk()
    pctr = risk_attr.percentage_contribution_to_risk()

    print(f"Risk Contributions:")
    for asset in asset_names:
        weight = weights[asset_names.index(asset)]
        print(f"  {asset:12s}: Weight={weight:6.1%}, Risk Contrib={pctr[asset]:6.1%}")

    # Risk budgeting analysis
    target_budgets = {
        'US Equities': 0.35, 'EU Equities': 0.30, 'Bonds': 0.15,
        'Commodities': 0.15, 'REITs': 0.05
    }

    budget_analysis = risk_attr.risk_budgeting_analysis(target_budgets)
    print(f"\nRisk Budget Analysis:")
    print(f"Risk Budget Efficiency: {budget_analysis['risk_budget_efficiency']:.1%}")
    print(f"Deviations from Target:")
    for asset in asset_names:
        deviation = budget_analysis['deviations'][asset]
        print(f"  {asset:12s}: {deviation:+6.1%}")

    # Sector attribution
    sector_mapping = {
        'US Equities': 'Equities', 'EU Equities': 'Equities',
        'Bonds': 'Fixed Income', 'Commodities': 'Alternatives', 'REITs': 'Real Estate'
    }

    sector_attr = risk_attr.sector_risk_attribution(sector_mapping)
    print(f"\nSector Risk Attribution:")
    for sector, risk_contrib in sector_attr['sector_percentage_contributions'].items():
        weight = sector_attr['sector_weights'][sector]
        print(f"  {sector:12s}: Weight={weight:6.1%}, Risk Contrib={risk_contrib:6.1%}")

    # Stress testing
    print(f"\nStress Test Scenarios:")
    print("-" * 25)

    stress_scenarios = {
        'Market Crash': np.array([-0.20, -0.25, 0.05, -0.15, -0.30]),  # Equity crash, bonds up
        'Interest Rate Shock': np.array([-0.05, -0.08, -0.15, 0.02, -0.10]),  # Rate rise
        'Inflation Spike': np.array([-0.10, -0.12, -0.20, 0.15, 0.05]),  # Commodities up
    }

    stress_results = risk_metrics.stress_test(stress_scenarios)
    for scenario, results in stress_results.items():
        portfolio_return = results['portfolio_return'] * portfolio_value
        worst_asset = results['worst_contributor']
        print(f"  {scenario}: Portfolio Impact=${portfolio_return:,.0f}, Worst: {worst_asset}")

    print(f"\nComprehensive Risk Report Generated Successfully!")
    print(f"Portfolio analyzed across {n_periods} periods with {n_assets} assets")


# Alias for backward compatibility
PortfolioRisk = PortfolioRiskMetrics