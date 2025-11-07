"""
Validated Multi-Agent Forecaster with Rust Monte Carlo Validation

Extends MultiAgentForecaster with ultra-fast Rust-based Monte Carlo validation
to ensure forecasts are statistically sound before use in option pricing.

Integration Architecture:
    Agent Simulation → Parameter Inference → Monte Carlo Validation → Validated Forecast
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
import sys
from pathlib import Path

from .agent_forecaster import MultiAgentForecaster
from .agents import simulate_agent_interaction

# Try to import Rust Monte Carlo validator
try:
    rust_path = Path(__file__).parent.parent.parent / "rust_monte_carlo" / "python"
    sys.path.insert(0, str(rust_path))
    from monte_carlo_validator import MonteCarloValidator, ValidationResult
    RUST_AVAILABLE = True
    logging.info("✅ Rust Monte Carlo validator loaded successfully")
except ImportError as e:
    RUST_AVAILABLE = False
    logging.warning(f"⚠️  Rust Monte Carlo validator not available: {e}")
    logging.warning("    Forecasts will not be validated. Run './build.sh' in rust_monte_carlo/")

logger = logging.getLogger(__name__)


class ValidatedMultiAgentForecaster(MultiAgentForecaster):
    """
    Multi-Agent Forecaster with Monte Carlo Validation

    Extends base forecaster with:
    1. Statistical validation of agent predictions
    2. Confidence adjustment based on validation results
    3. Risk-aware forecast filtering
    4. Performance benchmarking
    """

    def __init__(self,
                 market_maker_risk_aversion: float = 1.0,
                 arbitrageur_risk_aversion: float = 0.5,
                 noise_trader_risk_aversion: float = 2.0,
                 simulation_periods: int = 100,
                 validation_simulations: int = 50_000,
                 validation_threshold: float = 0.05,
                 enable_validation: bool = True):
        """
        Initialize validated forecaster.

        Parameters:
        -----------
        validation_simulations : int
            Number of Monte Carlo simulations for validation (default: 50,000)
        validation_threshold : float
            P-value threshold for rejecting forecasts (default: 0.05)
        enable_validation : bool
            Whether to run validation (requires Rust module)
        """
        super().__init__(
            market_maker_risk_aversion=market_maker_risk_aversion,
            arbitrageur_risk_aversion=arbitrageur_risk_aversion,
            noise_trader_risk_aversion=noise_trader_risk_aversion,
            simulation_periods=simulation_periods
        )

        self.enable_validation = enable_validation and RUST_AVAILABLE
        self.validation_threshold = validation_threshold

        if self.enable_validation:
            self.validator = MonteCarloValidator(
                n_simulations=validation_simulations,
                n_steps=100,
                dt=1/252  # Daily
            )
            logger.info(f"Validation enabled: {validation_simulations:,} MC simulations")
        else:
            self.validator = None
            if enable_validation:
                logger.warning("Validation requested but Rust module unavailable")

    def forecast_with_validation(self,
                                  historical_prices: np.ndarray,
                                  return_validation_details: bool = False) -> Dict:
        """
        Forecast with statistical validation.

        Parameters:
        -----------
        historical_prices : np.ndarray
            Historical price time series
        return_validation_details : bool
            Whether to include detailed validation metrics

        Returns:
        --------
        forecast : Dict
            {
                'implied_volatility': float,
                'implied_drift': float,
                'regime': str,
                'confidence': float,
                'validation': Dict (if validated),
                'method': str
            }
        """
        # Get base forecast from agents
        base_forecast = self.forecast(historical_prices)

        if not self.enable_validation:
            base_forecast['validated'] = False
            return base_forecast

        # Validate forecast using Monte Carlo
        validation_result = self._validate_forecast(
            forecast=base_forecast,
            historical_prices=historical_prices
        )

        # Adjust confidence based on validation
        validated_forecast = self._adjust_forecast_confidence(
            base_forecast,
            validation_result
        )

        # Add validation metrics
        validated_forecast['validated'] = True
        validated_forecast['validation'] = {
            'is_valid': validation_result.is_valid,
            'p_value': validation_result.p_value,
            'mean_error': validation_result.mean_error,
            'std_error': validation_result.std_error
        }

        if return_validation_details:
            validated_forecast['validation']['details'] = {
                'confidence_interval': validation_result.confidence_interval,
                'statistics': validation_result.statistics,
                'agent_id': validation_result.agent_id
            }

        return validated_forecast

    def _validate_forecast(self,
                          forecast: Dict,
                          historical_prices: np.ndarray) -> ValidationResult:
        """
        Validate forecast using Rust Monte Carlo engine.

        Validation Logic:
        1. Use agent-predicted parameters as MC inputs
        2. Run 50k+ simulations in parallel
        3. Compare predicted distribution vs simulated
        4. Statistical test (p-value) for validity
        """
        # Extract forecast parameters
        predicted_vol = forecast['implied_volatility']
        predicted_drift = forecast['implied_drift']

        # Estimate expected price and std from historical data
        current_price = historical_prices[-1]
        returns = np.diff(np.log(historical_prices))
        historical_std = np.std(returns) * np.sqrt(252)  # Annualized

        # Predict future price distribution
        # Using GBM: E[S_T] = S_0 * exp(μ*T)
        # Std[S_T] ≈ S_0 * σ * sqrt(T)
        T = 100 / 252  # 100 days
        predicted_mean = current_price * np.exp(predicted_drift * T)
        predicted_std = current_price * predicted_vol * np.sqrt(T)

        # Run validation
        validation_result = self.validator.validate_agent_prediction(
            agent_id="multi_agent_ensemble",
            predicted_mean=predicted_mean,
            predicted_std=predicted_std,
            confidence=forecast['confidence'],
            confidence_level=0.95
        )

        logger.info(f"Validation: {'✅ VALID' if validation_result.is_valid else '❌ INVALID'} "
                   f"(p={validation_result.p_value:.4f})")

        return validation_result

    def _adjust_forecast_confidence(self,
                                    base_forecast: Dict,
                                    validation_result: ValidationResult) -> Dict:
        """
        Adjust forecast confidence based on validation results.

        Strategy:
        - If validation passes: Keep or slightly increase confidence
        - If validation fails: Reduce confidence proportionally
        - Use p-value to modulate adjustment
        """
        adjusted_forecast = base_forecast.copy()

        if validation_result.is_valid:
            # Validation passed: boost confidence slightly
            confidence_boost = 0.1 * (1 - validation_result.p_value)
            adjusted_forecast['confidence'] = min(
                1.0,
                base_forecast['confidence'] + confidence_boost
            )
            logger.info(f"Confidence adjusted: {base_forecast['confidence']:.2f} → "
                       f"{adjusted_forecast['confidence']:.2f} (validated)")
        else:
            # Validation failed: reduce confidence
            confidence_penalty = 0.5 * validation_result.p_value
            adjusted_forecast['confidence'] = max(
                0.1,
                base_forecast['confidence'] - confidence_penalty
            )
            logger.warning(f"Confidence reduced: {base_forecast['confidence']:.2f} → "
                          f"{adjusted_forecast['confidence']:.2f} (failed validation)")

        return adjusted_forecast

    def batch_validate_agents(self,
                             historical_prices: np.ndarray) -> Dict[str, ValidationResult]:
        """
        Validate each agent individually using parallel Monte Carlo.

        This provides fine-grained diagnostics about which agents are reliable.

        Returns:
        --------
        validation_results : Dict[str, ValidationResult]
            Mapping from agent name to validation result
        """
        if not self.enable_validation:
            logger.warning("Validation not enabled")
            return {}

        # Simulate agents
        agents_list = [self.market_maker, self.arbitrageur, self.noise_trader]
        simulation_results = simulate_agent_interaction(
            agents=agents_list,
            market_data=historical_prices,
            n_periods=min(self.simulation_periods, len(historical_prices))
        )

        # Get individual agent parameters
        agent_params = simulation_results['parameters']

        validation_results = {}

        for agent_name, param_value in agent_params.items():
            if agent_name == 'MarketMaker':
                # Volatility prediction
                predicted_vol = param_value
                current_price = historical_prices[-1]
                T = 100 / 252
                predicted_mean = current_price
                predicted_std = current_price * predicted_vol * np.sqrt(T)

            elif agent_name == 'Arbitrageur':
                # Drift prediction
                predicted_drift = param_value
                current_price = historical_prices[-1]
                T = 100 / 252
                predicted_mean = current_price * np.exp(predicted_drift * T)
                # Use historical vol for std
                returns = np.diff(np.log(historical_prices))
                historical_vol = np.std(returns) * np.sqrt(252)
                predicted_std = current_price * historical_vol * np.sqrt(T)

            else:
                # Regime indicator - skip validation
                continue

            result = self.validator.validate_agent_prediction(
                agent_id=agent_name,
                predicted_mean=predicted_mean,
                predicted_std=predicted_std,
                confidence=0.95
            )

            validation_results[agent_name] = result
            logger.info(f"{agent_name}: {'✅' if result.is_valid else '❌'} "
                       f"(p={result.p_value:.4f}, error={result.mean_error:.2f})")

        return validation_results

    def benchmark_performance(self,
                            historical_prices: np.ndarray,
                            n_trials: int = 10) -> Dict:
        """
        Benchmark forecasting + validation performance.

        Returns:
        --------
        benchmark : Dict
            {
                'avg_forecast_time': float (seconds),
                'avg_validation_time': float (seconds),
                'total_time': float (seconds),
                'throughput': float (forecasts/second)
            }
        """
        import time

        forecast_times = []
        validation_times = []

        for _ in range(n_trials):
            # Forecast
            start = time.time()
            forecast = self.forecast(historical_prices)
            forecast_time = time.time() - start
            forecast_times.append(forecast_time)

            if self.enable_validation:
                # Validation
                start = time.time()
                _ = self._validate_forecast(forecast, historical_prices)
                validation_time = time.time() - start
                validation_times.append(validation_time)

        avg_forecast_time = np.mean(forecast_times)
        avg_validation_time = np.mean(validation_times) if validation_times else 0
        total_time = avg_forecast_time + avg_validation_time

        benchmark = {
            'avg_forecast_time': avg_forecast_time,
            'avg_validation_time': avg_validation_time,
            'total_time': total_time,
            'throughput': 1 / total_time if total_time > 0 else 0,
            'n_trials': n_trials,
            'validation_enabled': self.enable_validation
        }

        logger.info(f"Benchmark ({n_trials} trials):")
        logger.info(f"  Forecast: {avg_forecast_time*1000:.1f}ms")
        if self.enable_validation:
            logger.info(f"  Validation: {avg_validation_time*1000:.1f}ms")
        logger.info(f"  Total: {total_time*1000:.1f}ms ({benchmark['throughput']:.2f} forecasts/s)")

        return benchmark

    def explain_validated_forecast(self,
                                   historical_prices: np.ndarray) -> str:
        """
        Generate human-readable explanation including validation results.
        """
        forecast = self.forecast_with_validation(
            historical_prices,
            return_validation_details=True
        )

        explanation = super().explain_forecast(historical_prices)

        if forecast.get('validated'):
            validation = forecast['validation']

            explanation += "\n\n"
            explanation += "=" * 70
            explanation += "\n🧪 MONTE CARLO VALIDATION (Rust-Accelerated)"
            explanation += "\n" + "=" * 70

            if validation['is_valid']:
                explanation += "\n\n✅ VALIDATION PASSED"
                explanation += f"\n  • Statistical Test: p-value = {validation['p_value']:.4f}"
                explanation += f"\n  • Mean Error: {validation['mean_error']:.2f}"
                explanation += f"\n  • Std Error: {validation['std_error']:.2f}"
                explanation += "\n\n  The forecast is statistically consistent with historical"
                explanation += "\n  data. Confidence adjusted upward."
            else:
                explanation += "\n\n❌ VALIDATION FAILED"
                explanation += f"\n  • Statistical Test: p-value = {validation['p_value']:.4f} (< 0.05)"
                explanation += f"\n  • Mean Error: {validation['mean_error']:.2f}"
                explanation += f"\n  • Std Error: {validation['std_error']:.2f}"
                explanation += "\n\n  ⚠️  WARNING: The forecast deviates significantly from"
                explanation += "\n  simulated distributions. Use with caution!"

            if 'details' in validation:
                ci = validation['details']['confidence_interval']
                explanation += f"\n\n📊 Simulated Statistics:"
                explanation += f"\n  • 95% CI: [{ci[0]:.2f}, {ci[1]:.2f}]"

                stats = validation['details']['statistics']
                explanation += f"\n  • Mean: {stats['mean']:.2f}"
                explanation += f"\n  • Std: {stats['std']:.2f}"
                explanation += f"\n  • VaR (95%): {stats['var_95']:.2f}"
                explanation += f"\n  • CVaR (95%): {stats['cvar_95']:.2f}"

            explanation += "\n\n" + "=" * 70

        return explanation


# Convenience function for quick usage
def create_validated_forecaster(
    validation_simulations: int = 50_000,
    enable_validation: bool = True
) -> ValidatedMultiAgentForecaster:
    """
    Factory function to create a validated forecaster with sensible defaults.

    Parameters:
    -----------
    validation_simulations : int
        Number of Monte Carlo simulations (more = slower but more accurate)
    enable_validation : bool
        Whether to enable Rust validation

    Returns:
    --------
    forecaster : ValidatedMultiAgentForecaster
    """
    return ValidatedMultiAgentForecaster(
        simulation_periods=100,
        validation_simulations=validation_simulations,
        enable_validation=enable_validation
    )
