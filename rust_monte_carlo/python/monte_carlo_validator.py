"""
Python interface for Rust Monte Carlo validator
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

try:
    # Try relative import first (if installed as package)
    try:
        from ..monte_carlo_rust import (
            PyMonteCarloEngine,
            quick_validate as _quick_validate,
            batch_validate as _batch_validate
        )
    except ImportError:
        # Try absolute import (if installed via pip)
        from monte_carlo_rust import (
            PyMonteCarloEngine,
            quick_validate as _quick_validate,
            batch_validate as _batch_validate
        )
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False
    print("⚠️  Warning: Rust module not available. Using pure Python fallback.")


@dataclass
class ValidationResult:
    """Result of Monte Carlo validation"""
    agent_id: str
    mean_error: float
    std_error: float
    confidence_interval: Tuple[float, float]
    is_valid: bool
    p_value: float
    statistics: Dict[str, float]


class MonteCarloValidator:
    """
    High-performance Monte Carlo validator for multi-agent forecasts

    Uses Rust backend for maximum performance with parallel execution.
    """

    def __init__(
        self,
        n_simulations: int = 10_000,
        n_steps: int = 100,
        dt: float = 1/252,
        initial_price: float = 100.0,
        drift: float = 0.0,
        volatility: float = 0.2,
    ):
        """
        Initialize Monte Carlo validator

        Args:
            n_simulations: Number of Monte Carlo paths to simulate
            n_steps: Number of time steps per path
            dt: Time increment (default: 1/252 for daily)
            initial_price: Starting price for simulations
            drift: Expected return (annualized)
            volatility: Volatility (annualized)
        """
        self.n_simulations = n_simulations
        self.n_steps = n_steps
        self.dt = dt
        self.initial_price = initial_price
        self.drift = drift
        self.volatility = volatility

        if RUST_AVAILABLE:
            self.engine = PyMonteCarloEngine(
                n_simulations=n_simulations,
                n_steps=n_steps,
                dt=dt,
                initial_price=initial_price,
                drift=drift,
                volatility=volatility,
            )
        else:
            self.engine = None

    def validate_agent_prediction(
        self,
        agent_id: str,
        predicted_mean: float,
        predicted_std: float,
        confidence: float = 0.95,
        confidence_level: float = 0.95,
    ) -> ValidationResult:
        """
        Validate a single agent's prediction

        Args:
            agent_id: Identifier for the agent
            predicted_mean: Agent's predicted mean price
            predicted_std: Agent's predicted standard deviation
            confidence: Agent's confidence in prediction
            confidence_level: Statistical confidence level for validation

        Returns:
            ValidationResult object with detailed validation metrics
        """
        if not RUST_AVAILABLE:
            return self._validate_python(agent_id, predicted_mean, predicted_std)

        result_dict = self.engine.validate_prediction(
            agent_id=agent_id,
            predicted_mean=predicted_mean,
            predicted_std=predicted_std,
            confidence=confidence,
            confidence_level=confidence_level,
        )

        return ValidationResult(
            agent_id=result_dict['agent_id'],
            mean_error=result_dict['mean_error'],
            std_error=result_dict['std_error'],
            confidence_interval=tuple(result_dict['confidence_interval']),
            is_valid=result_dict['is_valid'],
            p_value=result_dict['p_value'],
            statistics=result_dict['statistics'],
        )

    def validate_multiple_agents(
        self,
        predictions: List[Dict[str, float]],
        confidence_level: float = 0.95,
    ) -> List[ValidationResult]:
        """
        Validate multiple agents in parallel (Rust backend only)

        Args:
            predictions: List of dicts with keys: agent_id, predicted_mean, predicted_std
            confidence_level: Statistical confidence level

        Returns:
            List of ValidationResult objects
        """
        results = []
        for pred in predictions:
            result = self.validate_agent_prediction(
                agent_id=pred.get('agent_id', 'unknown'),
                predicted_mean=pred['predicted_mean'],
                predicted_std=pred['predicted_std'],
                confidence=pred.get('confidence', 0.95),
                confidence_level=confidence_level,
            )
            results.append(result)
        return results

    def run_scenario_analysis(
        self,
        volatility_scenarios: List[float]
    ) -> List[Dict[str, float]]:
        """
        Run Monte Carlo with different volatility scenarios

        Args:
            volatility_scenarios: List of volatility values to test

        Returns:
            List of dicts with scenario results
        """
        if not RUST_AVAILABLE:
            raise NotImplementedError("Scenario analysis requires Rust backend")

        return self.engine.scenario_analysis(volatility_scenarios)

    def calculate_option_greeks(
        self,
        option_type: str,
        strike: float
    ) -> Dict[str, float]:
        """
        Calculate option Greeks using Monte Carlo

        Args:
            option_type: 'call' or 'put'
            strike: Strike price

        Returns:
            Dict with delta, gamma, vega, theta
        """
        if not RUST_AVAILABLE:
            raise NotImplementedError("Greeks calculation requires Rust backend")

        return self.engine.calculate_greeks(option_type, strike)

    def run_simulations(self) -> np.ndarray:
        """
        Run Monte Carlo simulations and return all paths

        Returns:
            Array of shape (n_simulations, n_steps+1)
        """
        if not RUST_AVAILABLE:
            return self._simulate_python()

        paths = self.engine.run_simulations()
        return np.array(paths)

    # Fallback implementations
    def _simulate_python(self) -> np.ndarray:
        """Pure Python fallback (slow)"""
        np.random.seed(42)
        paths = np.zeros((self.n_simulations, self.n_steps + 1))
        paths[:, 0] = self.initial_price

        for i in range(self.n_simulations):
            for t in range(1, self.n_steps + 1):
                z = np.random.normal(0, 1)
                drift_term = self.drift * self.dt
                diff_term = self.volatility * np.sqrt(self.dt) * z
                paths[i, t] = paths[i, t-1] * np.exp(drift_term + diff_term)

        return paths

    def _validate_python(self, agent_id, predicted_mean, predicted_std):
        """Pure Python validation fallback"""
        paths = self._simulate_python()
        final_prices = paths[:, -1]

        mean = np.mean(final_prices)
        std = np.std(final_prices)

        mean_error = abs(mean - predicted_mean)
        std_error = abs(std - predicted_std)

        ci_lower = np.percentile(final_prices, 2.5)
        ci_upper = np.percentile(final_prices, 97.5)

        is_valid = ci_lower <= predicted_mean <= ci_upper

        return ValidationResult(
            agent_id=agent_id,
            mean_error=mean_error,
            std_error=std_error,
            confidence_interval=(ci_lower, ci_upper),
            is_valid=is_valid,
            p_value=0.5,  # Simplified
            statistics={
                'mean': mean,
                'std': std,
                'min': float(np.min(final_prices)),
                'max': float(np.max(final_prices)),
            }
        )


# Module-level convenience functions
def quick_validate(
    predicted_mean: float,
    predicted_std: float,
    n_simulations: int = 10_000,
    initial_price: float = 100.0,
    drift: float = 0.0,
    volatility: float = 0.2,
) -> bool:
    """
    Quick validation check (Rust backend)

    Returns:
        True if prediction is valid, False otherwise
    """
    if not RUST_AVAILABLE:
        validator = MonteCarloValidator(n_simulations=n_simulations)
        result = validator.validate_agent_prediction("quick", predicted_mean, predicted_std)
        return result.is_valid

    return _quick_validate(
        n_simulations=n_simulations,
        predicted_mean=predicted_mean,
        predicted_std=predicted_std,
        initial_price=initial_price,
        drift=drift,
        volatility=volatility,
    )


def batch_validate(
    predictions: List[Tuple[float, float]],
    n_simulations: int = 10_000,
    initial_price: float = 100.0,
    drift: float = 0.0,
    volatility: float = 0.2,
) -> List[bool]:
    """
    Validate multiple predictions in parallel (Rust backend)

    Args:
        predictions: List of (mean, std) tuples

    Returns:
        List of boolean validation results
    """
    if not RUST_AVAILABLE:
        validator = MonteCarloValidator(n_simulations=n_simulations)
        results = []
        for i, (mean, std) in enumerate(predictions):
            result = validator.validate_agent_prediction(f"agent_{i}", mean, std)
            results.append(result.is_valid)
        return results

    return _batch_validate(
        n_simulations=n_simulations,
        predictions=predictions,
        initial_price=initial_price,
        drift=drift,
        volatility=volatility,
    )
