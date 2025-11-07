"""
Portfolio Optimizer with CVaR and Greek Constraints

This module adds to the existing strategy layer a portfolio optimizer that:
1. Computes portfolio Greeks (Δ_p, Γ_p, Vega_p, Θ_p)
2. Optimizes allocation with CVaR constraints
3. Enforces Greek limits (e.g., delta-neutral)
4. Maximizes expected utility (Sharpe / min CVaR)

Integrates seamlessly with existing DP/RL strategy selectors.
"""

import numpy as np
from scipy.optimize import minimize, LinearConstraint, Bounds
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class PortfolioGreeks:
    """Aggregated portfolio Greeks"""
    delta: float = 0.0
    gamma: float = 0.0
    vega: float = 0.0
    theta: float = 0.0
    rho: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            'delta': self.delta,
            'gamma': self.gamma,
            'vega': self.vega,
            'theta': self.theta,
            'rho': self.rho
        }


@dataclass
class OptimizationResult:
    """Portfolio optimization result"""
    positions: Dict[str, float]  # {option_id: quantity}
    portfolio_greeks: PortfolioGreeks
    expected_return: float
    expected_sharpe: float
    expected_cvar: float
    var_95: float
    converged: bool
    objective_value: float


class PortfolioOptimizer:
    """
    Portfolio optimizer with CVaR and Greek constraints.

    Objective:
    ----------
    maximize: α × Expected_Return - β × CVaR_95 - γ × |Δ_p|

    subject to:
        |Δ_p| ≤ Δ_max
        |Vega_p| ≤ Vega_max
        |Γ_p| ≤ Γ_max
        Σ |position_i| ≤ Budget
    """

    def __init__(self,
                 risk_aversion: float = 1.0,
                 delta_neutrality_weight: float = 0.5,
                 confidence_level: float = 0.95,
                 max_delta: float = 1000.0,
                 max_vega: float = 5000.0,
                 max_gamma: float = 100.0,
                 max_theta: Optional[float] = None):
        """
        Initialize portfolio optimizer.

        Parameters:
        -----------
        risk_aversion : float
            Weight on CVaR term (higher = more risk-averse)
        delta_neutrality_weight : float
            Weight on delta-neutrality objective
        confidence_level : float
            Confidence level for CVaR (e.g., 0.95)
        max_delta, max_vega, max_gamma : float
            Greek constraints
        max_theta : Optional[float]
            Theta constraint (None = no constraint)
        """
        self.risk_aversion = risk_aversion
        self.delta_neutrality_weight = delta_neutrality_weight
        self.confidence_level = confidence_level
        self.max_delta = max_delta
        self.max_vega = max_vega
        self.max_gamma = max_gamma
        self.max_theta = max_theta

    def optimize_portfolio(self,
                          option_data: Dict[str, Dict],
                          forecasted_returns: Dict[str, float],
                          return_covariance: Optional[np.ndarray] = None,
                          max_position_size: float = 100.0,
                          n_scenarios: int = 1000) -> OptimizationResult:
        """
        Optimize portfolio allocation.

        Parameters:
        -----------
        option_data : Dict[str, Dict]
            {option_id: {'price': float, 'greeks': {'delta': ..., 'vega': ...}}}
        forecasted_returns : Dict[str, float]
            Expected returns for each option
        return_covariance : Optional[np.ndarray]
            Covariance matrix of returns (estimated if None)
        max_position_size : float
            Maximum position size per option
        n_scenarios : int
            Number of Monte Carlo scenarios for CVaR

        Returns:
        --------
        OptimizationResult
            Optimal portfolio allocation
        """
        option_ids = list(option_data.keys())
        n_options = len(option_ids)

        if n_options == 0:
            return self._empty_result()

        # Extract data
        prices = np.array([option_data[id]['price'] for id in option_ids])
        deltas = np.array([option_data[id]['greeks']['delta'] for id in option_ids])
        vegas = np.array([option_data[id]['greeks']['vega'] for id in option_ids])
        gammas = np.array([option_data[id]['greeks']['gamma'] for id in option_ids])
        thetas = np.array([option_data[id]['greeks'].get('theta', 0.0) for id in option_ids])

        mu = np.array([forecasted_returns.get(id, 0.0) for id in option_ids])

        # Estimate covariance if not provided
        if return_covariance is None:
            # Simple diagonal covariance (independent assets)
            vol_estimates = np.abs(vegas) * 0.01  # Rough vol estimate
            return_covariance = np.diag(vol_estimates ** 2)

        # Generate return scenarios for CVaR
        return_scenarios = self._generate_scenarios(mu, return_covariance, n_scenarios)

        # Define objective function
        def objective(weights):
            # Expected return
            expected_return = np.dot(mu, weights)

            # CVaR calculation
            portfolio_returns = return_scenarios @ weights
            var_cutoff = np.percentile(portfolio_returns, (1 - self.confidence_level) * 100)
            tail_losses = portfolio_returns[portfolio_returns <= var_cutoff]
            cvar = -np.mean(tail_losses) if len(tail_losses) > 0 else 0.0

            # Portfolio delta (penalize deviation from zero)
            portfolio_delta = np.dot(deltas, weights)
            delta_penalty = self.delta_neutrality_weight * (portfolio_delta ** 2)

            # Objective: maximize return - risk_aversion × CVaR - delta_penalty
            obj = -(expected_return - self.risk_aversion * cvar - delta_penalty)

            return obj

        # Bounds on position sizes
        bounds = Bounds(
            lb=np.full(n_options, -max_position_size),
            ub=np.full(n_options, max_position_size)
        )

        # Constraints
        constraints = []

        # Delta constraint: |Σ Δ_i × w_i| ≤ Δ_max
        constraints.append({
            'type': 'ineq',
            'fun': lambda w: self.max_delta - abs(np.dot(deltas, w))
        })

        # Vega constraint: |Σ Vega_i × w_i| ≤ Vega_max
        constraints.append({
            'type': 'ineq',
            'fun': lambda w: self.max_vega - abs(np.dot(vegas, w))
        })

        # Gamma constraint: |Σ Γ_i × w_i| ≤ Γ_max
        constraints.append({
            'type': 'ineq',
            'fun': lambda w: self.max_gamma - abs(np.dot(gammas, w))
        })

        # Theta constraint: |Σ Θ_i × w_i| ≤ Θ_max (if specified)
        if self.max_theta is not None:
            constraints.append({
                'type': 'ineq',
                'fun': lambda w: self.max_theta - abs(np.dot(thetas, w))
            })

        # Initial guess: zero positions
        w0 = np.zeros(n_options)

        # Optimize
        result = minimize(
            objective,
            w0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 500, 'ftol': 1e-8}
        )

        # Extract solution
        optimal_weights = result.x
        positions = {option_ids[i]: optimal_weights[i]
                    for i in range(n_options)
                    if abs(optimal_weights[i]) > 1e-6}

        # Compute portfolio Greeks
        portfolio_greeks = PortfolioGreeks(
            delta=np.dot(deltas, optimal_weights),
            gamma=np.dot(gammas, optimal_weights),
            vega=np.dot(vegas, optimal_weights),
            theta=np.dot(thetas, optimal_weights)
        )

        # Compute metrics
        expected_return = np.dot(mu, optimal_weights)
        portfolio_returns = return_scenarios @ optimal_weights
        expected_vol = np.std(portfolio_returns)
        expected_sharpe = expected_return / expected_vol if expected_vol > 0 else 0.0

        var_95 = -np.percentile(portfolio_returns, 5)
        tail_losses = portfolio_returns[portfolio_returns <= -var_95]
        expected_cvar = -np.mean(tail_losses) if len(tail_losses) > 0 else 0.0

        return OptimizationResult(
            positions=positions,
            portfolio_greeks=portfolio_greeks,
            expected_return=expected_return,
            expected_sharpe=expected_sharpe,
            expected_cvar=expected_cvar,
            var_95=var_95,
            converged=result.success,
            objective_value=-result.fun
        )

    def compute_portfolio_greeks(self,
                                positions: Dict[str, float],
                                option_data: Dict[str, Dict]) -> PortfolioGreeks:
        """
        Compute aggregated portfolio Greeks.

        Parameters:
        -----------
        positions : Dict[str, float]
            Current positions {option_id: quantity}
        option_data : Dict[str, Dict]
            Option data with greeks

        Returns:
        --------
        PortfolioGreeks
            Aggregated Greeks
        """
        total_delta = 0.0
        total_gamma = 0.0
        total_vega = 0.0
        total_theta = 0.0
        total_rho = 0.0

        for option_id, quantity in positions.items():
            if option_id in option_data:
                greeks = option_data[option_id]['greeks']
                total_delta += quantity * greeks.get('delta', 0.0)
                total_gamma += quantity * greeks.get('gamma', 0.0)
                total_vega += quantity * greeks.get('vega', 0.0)
                total_theta += quantity * greeks.get('theta', 0.0)
                total_rho += quantity * greeks.get('rho', 0.0)

        return PortfolioGreeks(
            delta=total_delta,
            gamma=total_gamma,
            vega=total_vega,
            theta=total_theta,
            rho=total_rho
        )

    def hedge_to_delta_neutral(self,
                               current_positions: Dict[str, float],
                               option_data: Dict[str, Dict],
                               hedging_instruments: List[str]) -> Dict[str, float]:
        """
        Find hedging positions to make portfolio delta-neutral.

        Parameters:
        -----------
        current_positions : Dict[str, float]
            Current portfolio positions
        option_data : Dict[str, Dict]
            All available options with Greeks
        hedging_instruments : List[str]
            Option IDs available for hedging

        Returns:
        --------
        Dict[str, float]
            Hedge positions to add
        """
        # Current portfolio delta
        current_greeks = self.compute_portfolio_greeks(current_positions, option_data)
        target_delta = -current_greeks.delta  # Need to offset

        # Extract deltas of hedging instruments
        deltas = np.array([option_data[id]['greeks']['delta']
                          for id in hedging_instruments])

        if len(hedging_instruments) == 0:
            logger.warning("No hedging instruments available")
            return {}

        # Simple least squares solution
        if len(hedging_instruments) == 1:
            # Single instrument: exact hedge
            hedge_qty = target_delta / deltas[0] if deltas[0] != 0 else 0.0
            return {hedging_instruments[0]: hedge_qty}
        else:
            # Multiple instruments: minimize ||hedge||²
            A = deltas.reshape(1, -1)
            b = np.array([target_delta])
            hedge_weights = np.linalg.lstsq(A.T, b, rcond=None)[0]

            hedge_positions = {hedging_instruments[i]: hedge_weights[i]
                              for i in range(len(hedging_instruments))
                              if abs(hedge_weights[i]) > 1e-6}

            return hedge_positions

    def _generate_scenarios(self,
                           mu: np.ndarray,
                           cov: np.ndarray,
                           n_scenarios: int) -> np.ndarray:
        """Generate return scenarios for CVaR calculation"""
        n_assets = len(mu)
        cov_stable = cov + np.eye(n_assets) * 1e-8  # Add small diagonal for stability

        scenarios = np.random.multivariate_normal(
            mean=mu,
            cov=cov_stable,
            size=n_scenarios
        )

        return scenarios

    def _empty_result(self) -> OptimizationResult:
        """Return empty result"""
        return OptimizationResult(
            positions={},
            portfolio_greeks=PortfolioGreeks(),
            expected_return=0.0,
            expected_sharpe=0.0,
            expected_cvar=0.0,
            var_95=0.0,
            converged=True,
            objective_value=0.0
        )


# ============================================================================
# HELPER: Integrate with Existing Strategy Selectors
# ============================================================================

class IntegratedStrategyOptimizer:
    """
    Wrapper that integrates portfolio optimizer with existing DP/RL strategies.

    Flow:
    1. DP/RL selector proposes strategy (e.g., "Straddle")
    2. Portfolio optimizer determines optimal sizing with Greek constraints
    3. Risk controller validates
    """

    def __init__(self,
                 strategy_selector,  # Existing DP/RL selector
                 portfolio_optimizer: PortfolioOptimizer):
        self.strategy_selector = strategy_selector
        self.portfolio_optimizer = portfolio_optimizer

    def select_and_optimize(self,
                           market_state: Dict,
                           available_options: Dict[str, Dict],
                           forecasted_returns: Dict[str, float]) -> Tuple[str, Dict[str, float]]:
        """
        Select strategy and optimize allocation.

        Parameters:
        -----------
        market_state : Dict
            Current market state (price, vol, regime, etc.)
        available_options : Dict[str, Dict]
            Available options with prices and Greeks
        forecasted_returns : Dict[str, float]
            Forecasted returns

        Returns:
        --------
        strategy_name : str
            Selected strategy name
        positions : Dict[str, float]
            Optimal positions
        """
        # Step 1: Strategy selector picks strategy type
        strategy_name = self.strategy_selector.select_strategy(market_state)

        # Step 2: Filter options relevant to this strategy
        # (In real implementation, you'd map strategy to option combos)
        relevant_options = available_options  # Simplified

        # Step 3: Portfolio optimizer finds optimal allocation
        opt_result = self.portfolio_optimizer.optimize_portfolio(
            option_data=relevant_options,
            forecasted_returns=forecasted_returns
        )

        logger.info(f"Selected strategy: {strategy_name}")
        logger.info(f"Portfolio Greeks: {opt_result.portfolio_greeks.to_dict()}")
        logger.info(f"Expected Sharpe: {opt_result.expected_sharpe:.3f}")
        logger.info(f"Expected CVaR: ${opt_result.expected_cvar:,.0f}")

        return strategy_name, opt_result.positions
