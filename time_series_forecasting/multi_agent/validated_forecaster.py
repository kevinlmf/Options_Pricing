"""
Validated Multi-Agent Forecaster with Rust Monte Carlo Validation

Extends MultiAgentForecaster with ultra-fast Rust-based Monte Carlo validation
to ensure forecasts are statistically sound before use in option pricing.

Integration Architecture:
    Agent Simulation → Parameter Inference → Monte Carlo Validation → Validated Forecast
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
import sys
from pathlib import Path
import time
from collections import deque

from .agent_forecaster import MultiAgentForecaster
from .agents import simulate_agent_interaction
from models.options_pricing.black_scholes import BlackScholesModel, BSParameters

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
                 enable_validation: bool = True,
                 auto_adapt_agents: bool = True,
                 adaptive_learning_rate: float = 0.05,
                 max_validation_records: int = 200,
                 risk_free_rate: float = 0.05):
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
        self.auto_adapt_agents = auto_adapt_agents
        self.adaptive_learning_rate = adaptive_learning_rate
        self.validation_history = deque(maxlen=max_validation_records)
        self.validation_simulations = validation_simulations
        self.validation_n_steps = 100
        self.validation_dt = 1 / 252
        self.risk_free_rate = risk_free_rate

        if self.enable_validation:
            self.validator = self._create_validator(
                initial_price=100.0,
                drift=0.0,
                volatility=0.2
            )
            logger.info(f"Validation enabled: {validation_simulations:,} MC simulations (Rust backend)")
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
        validation_result, validation_context = self._validate_forecast(
            forecast=base_forecast,
            historical_prices=historical_prices
        )

        self._record_validation(base_forecast, validation_result, validation_context)

        if self.auto_adapt_agents:
            self._update_agents_from_validation(
                base_forecast,
                validation_result,
                validation_context
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
            'std_error': validation_result.std_error,
            'predicted_mean': validation_context.get('predicted_mean'),
            'predicted_std': validation_context.get('predicted_std'),
            'realized_mean': validation_context.get('actual_mean'),
            'realized_std': validation_context.get('actual_std'),
            'mean_ci': validation_context.get('mean_ci'),
            'mean_within_ci': validation_context.get('mean_within_ci'),
            'vol_mc_mean': validation_context.get('vol_mc_mean'),
            'vol_mc_ci': validation_context.get('vol_mc_ci'),
            'vol_within_ci': validation_context.get('vol_within_ci'),
            'option_abs_error_mean': validation_context.get('option_abs_error_mean'),
            'option_within_ci_ratio': validation_context.get('option_within_ci_ratio'),
            'option_surface': validation_context.get('option_surface'),
            'var_pred': validation_context.get('var_pred'),
            'var_rn': validation_context.get('var_rn'),
            'cvar_pred': validation_context.get('cvar_pred'),
            'cvar_rn': validation_context.get('cvar_rn'),
            'var_diff': validation_context.get('var_diff'),
            'cvar_diff': validation_context.get('cvar_diff')
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
        predicted_mean = current_price * np.exp((predicted_drift + 0.5 * predicted_vol**2) * T)
        predicted_std = current_price * predicted_vol * np.sqrt(T)

        # Run validation
        validator = self._create_validator(
            initial_price=current_price,
            drift=predicted_drift,
            volatility=max(predicted_vol, 1e-6)
        )

        validation_result = validator.validate_agent_prediction(
            agent_id="multi_agent_ensemble",
            predicted_mean=predicted_mean,
            predicted_std=predicted_std,
            confidence=forecast['confidence'],
            confidence_level=0.95
        )

        logger.info(f"Validation: {'✅ VALID' if validation_result.is_valid else '❌ INVALID'} "
                   f"(p={validation_result.p_value:.4f})")

        predicted_paths = self._get_simulation_paths(validator)
        volatility_metrics = self._compute_volatility_metrics(
            paths=predicted_paths,
            current_price=current_price,
            predicted_vol=predicted_vol,
            horizon=T
        )

        rn_validator = self._create_validator(
            initial_price=current_price,
            drift=self.risk_free_rate,
            volatility=max(predicted_vol, 1e-6)
        )
        rn_paths = self._get_simulation_paths(rn_validator)

        option_metrics = self._compute_option_metrics(
            predicted_paths=predicted_paths,
            rn_paths=rn_paths,
            current_price=current_price,
            predicted_vol=predicted_vol,
            predicted_drift=predicted_drift,
            horizon=T
        )

        tail_metrics = self._compute_tail_risk_metrics(
            predicted_paths=predicted_paths,
            rn_paths=rn_paths,
            current_price=current_price,
            horizon=T
        )

        validation_context = {
            'predicted_mean': predicted_mean,
            'predicted_std': predicted_std,
            'actual_mean': validation_result.statistics.get('mean') if validation_result.statistics else None,
            'actual_std': validation_result.statistics.get('std') if validation_result.statistics else None,
            'mean_ci': validation_result.confidence_interval,
            'mean_within_ci': (
                validation_result.confidence_interval[0] <= predicted_mean <= validation_result.confidence_interval[1]
                if validation_result.confidence_interval and all(np.isfinite(x) for x in validation_result.confidence_interval)
                else None
            ),
            'current_price': current_price,
            'horizon': T,
            **volatility_metrics,
            **option_metrics,
            **tail_metrics
        }

        return validation_result, validation_context

    def _compute_volatility_metrics(self,
                                    paths: Optional[np.ndarray],
                                    current_price: float,
                                    predicted_vol: float,
                                    horizon: float) -> Dict[str, float]:
        metrics = {
            'vol_mc_mean': None,
            'vol_mc_ci': (None, None),
            'vol_within_ci': None
        }

        if paths is None or len(paths) == 0:
            return metrics

        paths = np.asarray(paths)
        if paths.ndim != 2:
            return metrics

        final_prices = paths[:, -1]
        simulated_returns = np.log(final_prices / current_price)
        if len(simulated_returns) == 0:
            return metrics

        simulated_return_std = np.std(simulated_returns)
        sim_vol_mean = float(simulated_return_std / np.sqrt(horizon))
        sim_vol_se = sim_vol_mean / np.sqrt(len(simulated_returns))
        ci = (
            sim_vol_mean - 1.96 * sim_vol_se,
            sim_vol_mean + 1.96 * sim_vol_se
        )

        within_ci = None
        if np.isfinite(predicted_vol) and all(np.isfinite(ci_val) for ci_val in ci):
            within_ci = ci[0] <= predicted_vol <= ci[1]

        metrics.update({
            'vol_mc_mean': sim_vol_mean,
            'vol_mc_ci': ci,
            'vol_within_ci': within_ci
        })

        return metrics

    def _compute_option_metrics(self,
                                predicted_paths: Optional[np.ndarray],
                                rn_paths: Optional[np.ndarray],
                                current_price: float,
                                predicted_vol: float,
                                predicted_drift: float,
                                horizon: float) -> Dict[str, Any]:
        metrics: Dict[str, Any] = {
            'option_abs_error_mean': None,
            'option_within_ci_ratio': None,
            'option_surface': []
        }

        if predicted_paths is None:
            return metrics

        strike_ratios = [0.9, 1.0, 1.1]
        maturities = [max(horizon / 2, self.validation_dt), horizon]

        option_records = []
        abs_errors = []
        within_ci_count = 0
        total_count = 0

        for ratio in strike_ratios:
            strike = current_price * ratio
            for mat in maturities:
                price_mc, ci_mc = self._price_from_paths(
                    paths=predicted_paths,
                    maturity=mat,
                    strike=strike,
                    discount_rate=self.risk_free_rate
                )

                price_rn, _ = self._price_from_paths(
                    paths=rn_paths,
                    maturity=mat,
                    strike=strike,
                    discount_rate=self.risk_free_rate
                )

                price_bs = self._black_scholes_price(
                    S0=current_price,
                    K=strike,
                    T=mat,
                    r=self.risk_free_rate,
                    sigma=max(predicted_vol, 1e-6)
                )

                within_ci = None
                if (price_bs is not None and
                        ci_mc[0] is not None and
                        ci_mc[1] is not None):
                    within_ci = ci_mc[0] <= price_bs <= ci_mc[1]
                    if within_ci:
                        within_ci_count += 1

                record = {
                    'strike_ratio': ratio,
                    'maturity': mat,
                    'price_mc': price_mc,
                    'price_rn': price_rn,
                    'price_bs': price_bs,
                    'confidence_interval': ci_mc,
                    'within_ci': within_ci
                }
                option_records.append(record)

                if price_mc is not None and price_bs is not None:
                    abs_error = abs(price_bs - price_mc)
                    abs_errors.append(abs_error)
                    total_count += 1

        if abs_errors:
            metrics['option_abs_error_mean'] = float(np.mean(abs_errors))
        if total_count > 0:
            metrics['option_within_ci_ratio'] = within_ci_count / total_count

        metrics['option_surface'] = option_records
        return metrics

    def _compute_tail_risk_metrics(self,
                                   predicted_paths: Optional[np.ndarray],
                                   rn_paths: Optional[np.ndarray],
                                   current_price: float,
                                   horizon: float,
                                   alpha: float = 0.95) -> Dict[str, Any]:
        metrics: Dict[str, Any] = {
            'var_pred': None,
            'var_rn': None,
            'cvar_pred': None,
            'cvar_rn': None,
            'var_diff': None,
            'cvar_diff': None
        }

        pred_returns = self._compute_returns_from_paths(predicted_paths, current_price, horizon)
        rn_returns = self._compute_returns_from_paths(rn_paths, current_price, horizon)

        if pred_returns is None or rn_returns is None:
            return metrics

        var_pred = float(np.percentile(pred_returns, (1 - alpha) * 100))
        var_rn = float(np.percentile(rn_returns, (1 - alpha) * 100))

        cvar_pred = float(np.mean(pred_returns[pred_returns <= var_pred])) if np.any(pred_returns <= var_pred) else var_pred
        cvar_rn = float(np.mean(rn_returns[rn_returns <= var_rn])) if np.any(rn_returns <= var_rn) else var_rn

        metrics.update({
            'var_pred': var_pred,
            'var_rn': var_rn,
            'cvar_pred': cvar_pred,
            'cvar_rn': cvar_rn,
            'var_diff': abs(var_pred - var_rn),
            'cvar_diff': abs(cvar_pred - cvar_rn)
        })

        return metrics

    def _price_from_paths(self,
                          paths: Optional[np.ndarray],
                          maturity: float,
                          strike: float,
                          discount_rate: float) -> Tuple[Optional[float], Tuple[Optional[float], Optional[float]]]:
        if paths is None or len(paths) == 0:
            return None, (None, None)

        paths = np.asarray(paths)
        if paths.ndim != 2:
            return None, (None, None)

        step_index = min(
            paths.shape[1] - 1,
            max(1, int(round(maturity / self.validation_dt)))
        )
        terminal_prices = paths[:, step_index]
        discounted = np.exp(-discount_rate * maturity)
        payoffs = np.maximum(terminal_prices - strike, 0.0)
        price = discounted * np.mean(payoffs)
        payoff_std = np.std(payoffs)
        se = discounted * payoff_std / np.sqrt(len(payoffs))
        ci = (price - 1.96 * se, price + 1.96 * se)
        return float(price), ci

    def _compute_returns_from_paths(self,
                                    paths: Optional[np.ndarray],
                                    current_price: float,
                                    horizon: float) -> Optional[np.ndarray]:
        if paths is None or len(paths) == 0:
            return None

        paths = np.asarray(paths)
        if paths.ndim != 2:
            return None

        step_index = min(
            paths.shape[1] - 1,
            max(1, int(round(horizon / self.validation_dt)))
        )
        terminal_prices = paths[:, step_index]
        returns = terminal_prices / current_price - 1.0
        return returns

    def _black_scholes_price(self,
                             S0: float,
                             K: float,
                             T: float,
                             r: float,
                             sigma: float) -> Optional[float]:
        try:
            params = BSParameters(
                S0=S0,
                K=K,
                T=T,
                r=r,
                sigma=sigma
            )
            model = BlackScholesModel(params)
            return model.call_price()
        except Exception as exc:
            logger.warning(f"Black-Scholes pricing failed: {exc}")
            return None

    def _get_simulation_paths(self, validator: Optional[MonteCarloValidator]) -> Optional[np.ndarray]:
        if validator is None:
            return None
        try:
            return np.asarray(validator.run_simulations())
        except Exception as exc:
            logger.warning(f"Monte Carlo simulations unavailable: {exc}")
            return None

    def _record_validation(self,
                           forecast: Dict,
                           validation_result: ValidationResult,
                           context: Dict[str, float]):
        """Persist validation outcomes for learning and diagnostics."""
        record = {
            'timestamp': time.time(),
            'implied_volatility': forecast.get('implied_volatility'),
            'implied_drift': forecast.get('implied_drift'),
            'predicted_mean': context.get('predicted_mean'),
            'predicted_std': context.get('predicted_std'),
            'realized_mean': context.get('actual_mean'),
            'realized_std': context.get('actual_std'),
            'p_value': validation_result.p_value,
            'is_valid': validation_result.is_valid,
            'mean_ci': context.get('mean_ci'),
            'mean_within_ci': context.get('mean_within_ci'),
            'vol_mc_mean': context.get('vol_mc_mean'),
            'vol_mc_ci': context.get('vol_mc_ci'),
            'vol_within_ci': context.get('vol_within_ci'),
            'option_abs_error_mean': context.get('option_abs_error_mean'),
            'option_within_ci_ratio': context.get('option_within_ci_ratio'),
            'var_pred': context.get('var_pred'),
            'var_rn': context.get('var_rn'),
            'cvar_pred': context.get('cvar_pred'),
            'cvar_rn': context.get('cvar_rn'),
            'var_diff': context.get('var_diff'),
            'cvar_diff': context.get('cvar_diff')
        }
        self.validation_history.append(record)

    def _update_agents_from_validation(self,
                                       forecast: Dict,
                                       validation_result: ValidationResult,
                                       context: Dict[str, float]):
        """Adjust agent parameters based on validation feedback."""
        if not context:
            return

        current_price = context.get('current_price')
        horizon = context.get('horizon')
        predicted_vol = forecast.get('implied_volatility')
        predicted_drift = forecast.get('implied_drift')
        realized_std = context.get('actual_std')
        realized_mean = context.get('actual_mean')

        if current_price and horizon and realized_std:
            denom = current_price * np.sqrt(horizon)
            if denom > 0:
                realized_vol = realized_std / denom
                self.market_maker.update_from_validation(
                    predicted_vol,
                    realized_vol,
                    learning_rate=self.adaptive_learning_rate
                )

        if current_price and horizon and realized_mean and realized_mean > 0:
            realized_drift = np.log(realized_mean / current_price) / horizon
            self.arbitrageur.update_from_validation(
                predicted_drift,
                realized_drift,
                learning_rate=self.adaptive_learning_rate
            )

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
                agent_drift = 0.0
                agent_volatility = max(predicted_vol, 1e-6)

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
                agent_drift = predicted_drift
                agent_volatility = max(historical_vol, 1e-6)

            else:
                # Regime indicator - skip validation
                continue

            validator = self._create_validator(
                initial_price=current_price,
                drift=agent_drift,
                volatility=agent_volatility
            )

            result = validator.validate_agent_prediction(
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
                _ , _ = self._validate_forecast(forecast, historical_prices)
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

    def _create_validator(self,
                          initial_price: float,
                          drift: float,
                          volatility: float) -> Optional[MonteCarloValidator]:
        """Factory to create a Monte Carlo validator with dynamic parameters."""
        if not self.enable_validation:
            return None

        return MonteCarloValidator(
            n_simulations=self.validation_simulations,
            n_steps=self.validation_n_steps,
            dt=self.validation_dt,
            initial_price=initial_price,
            drift=drift,
            volatility=max(volatility, 1e-6)
        )

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
    enable_validation: bool = True,
    auto_adapt_agents: bool = True,
    adaptive_learning_rate: float = 0.05,
    max_validation_records: int = 200,
    risk_free_rate: float = 0.05,
    option_strike_ratio: float = 1.0
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
        enable_validation=enable_validation,
        auto_adapt_agents=auto_adapt_agents,
        adaptive_learning_rate=adaptive_learning_rate,
        max_validation_records=max_validation_records,
        risk_free_rate=risk_free_rate
    )
