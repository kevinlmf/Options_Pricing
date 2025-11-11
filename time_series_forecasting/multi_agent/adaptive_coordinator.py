"""
Adaptive coordination loop for multi-agent forecaster.

Implements a lightweight reinforcement-learning style trainer that
iteratively runs the validated forecaster, reads the Monte Carlo feedback,
and adjusts the agent learning dynamics (learning rate, reward tracking).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np


@dataclass
class TrainingStepResult:
    iteration: int
    reward: float
    mean_error: float
    std_error: float
    p_value: float
    learning_rate: float
    is_valid: bool
    mean_within_ci: Optional[bool]
    volatility_error: float
    volatility_within_ci: Optional[bool]
    option_error: float
    option_within_ci: Optional[float]
    tail_var_diff: float
    tail_cvar_diff: float
    forecast: Dict


class AdaptiveAgentCoordinator:
    """
    Simple reinforcement-style trainer for validated multi-agent forecasts.

    The coordinator treats the Monte Carlo validation feedback as a reward
    signal and adapts the agents' learning rate plus keeps a running value
    estimate of performance.
    """

    def __init__(self,
                 forecaster,
                 reward_penalty: float = 2.0,
                 reward_smoothing: float = 0.2,
                 learning_rate_bounds: tuple = (0.01, 0.1),
                 reward_scale: float = 0.05,
                 volatility_error_weight: float = 0.1,
                 option_error_weight: float = 0.05,
                 tail_risk_weight: float = 0.05):
        self.forecaster = forecaster
        self.reward_penalty = reward_penalty
        self.reward_smoothing = reward_smoothing
        self.learning_rate_bounds = learning_rate_bounds
        self.reward_scale = reward_scale
        self.volatility_error_weight = volatility_error_weight
        self.option_error_weight = option_error_weight
        self.tail_risk_weight = tail_risk_weight
        self.value_estimate = 0.0
        self.history: List[TrainingStepResult] = []

    def step(self,
             historical_prices,
             return_forecast: bool = False) -> TrainingStepResult:
        """
        Execute a single training step:
            1. Run validated forecast.
            2. Compute reward from validation metrics.
            3. Update adaptive learning rate.
        """
        forecast = self.forecaster.forecast_with_validation(
            historical_prices,
            return_validation_details=True
        )

        validation = forecast.get('validation', {})
        mean_error = validation.get('mean_error', np.nan)
        std_error = validation.get('std_error', np.nan)
        p_value = validation.get('p_value', np.nan)
        is_valid = validation.get('is_valid', False)
        vol_mc_mean = validation.get('vol_mc_mean')
        vol_mc_ci = validation.get('vol_mc_ci', (None, None))
        vol_within_ci = validation.get('vol_within_ci')
        predicted_vol = forecast.get('implied_volatility', np.nan)
        mean_within_ci = validation.get('mean_within_ci')
        option_error = validation.get('option_abs_error_mean')
        option_within_ci = validation.get('option_within_ci_ratio')
        var_diff = validation.get('var_diff')
        cvar_diff = validation.get('cvar_diff')

        if np.isnan(mean_error):
            reward = -self.reward_penalty
        else:
            reward = -abs(mean_error)

        threshold = getattr(self.forecaster, 'validation_threshold', 0.05)
        if np.isfinite(p_value):
            penalty_scale = max(0.0, (threshold - p_value) / max(threshold, 1e-6))
            reward -= self.reward_penalty * penalty_scale
        else:
            reward -= self.reward_penalty

        if vol_mc_mean is not None and np.isfinite(vol_mc_mean) and np.isfinite(predicted_vol):
            vol_error = abs(vol_mc_mean - predicted_vol)
            reward -= self.volatility_error_weight * vol_error
        else:
            vol_error = np.nan

        if option_error is not None and np.isfinite(option_error):
            reward -= self.option_error_weight * option_error

        tail_penalty = 0.0
        if var_diff is not None and np.isfinite(var_diff):
            tail_penalty += abs(var_diff)
        if cvar_diff is not None and np.isfinite(cvar_diff):
            tail_penalty += abs(cvar_diff)
        reward -= self.tail_risk_weight * tail_penalty

        self.value_estimate = (
            (1 - self.reward_smoothing) * self.value_estimate +
            self.reward_smoothing * reward
        )

        current_lr = self.forecaster.adaptive_learning_rate
        if reward < 0:
            scale_factor = 1 + self.reward_scale * min(abs(reward), 10.0)
            scale_factor = min(scale_factor, 2.0)
        else:
            scale_factor = max(1 - self.reward_scale * reward, 0.7)

        new_learning_rate = float(np.clip(
            current_lr * scale_factor,
            self.learning_rate_bounds[0],
            self.learning_rate_bounds[1]
        ))
        self.forecaster.adaptive_learning_rate = new_learning_rate

        result = TrainingStepResult(
            iteration=len(self.history),
            reward=reward,
            mean_error=mean_error if np.isfinite(mean_error) else np.nan,
            std_error=std_error if np.isfinite(std_error) else np.nan,
            p_value=p_value if np.isfinite(p_value) else np.nan,
            learning_rate=new_learning_rate,
            is_valid=is_valid,
            mean_within_ci=mean_within_ci,
            volatility_error=vol_error if np.isfinite(vol_error) else np.nan,
            volatility_within_ci=vol_within_ci,
            option_error=option_error if option_error is not None and np.isfinite(option_error) else np.nan,
            option_within_ci=option_within_ci,
            tail_var_diff=var_diff if var_diff is not None and np.isfinite(var_diff) else np.nan,
            tail_cvar_diff=cvar_diff if cvar_diff is not None and np.isfinite(cvar_diff) else np.nan,
            forecast=forecast if return_forecast else {}
        )

        self.history.append(result)
        return result

    def train(self,
              historical_prices,
              iterations: int = 10,
              return_forecasts: bool = False) -> List[TrainingStepResult]:
        """Run multiple training iterations."""
        results = []
        for _ in range(iterations):
            result = self.step(
                historical_prices,
                return_forecast=return_forecasts
            )
            results.append(result)
        return results

    def summary(self) -> Dict[str, float]:
        """Provide quick stats of the training progress."""
        if not self.history:
            return {}

        mean_errors = [r.mean_error for r in self.history if np.isfinite(r.mean_error)]
        rewards = [r.reward for r in self.history]
        validity_rate = np.mean([1.0 if r.is_valid else 0.0 for r in self.history])

        vol_errors = [r.volatility_error for r in self.history if np.isfinite(r.volatility_error)]
        vol_ci_rate = np.mean([
            1.0 if r.volatility_within_ci else 0.0
            for r in self.history
            if r.volatility_within_ci is not None
        ]) if any(r.volatility_within_ci is not None for r in self.history) else np.nan
        mean_ci_rate = np.mean([
            1.0 if r.mean_within_ci else 0.0
            for r in self.history
            if r.mean_within_ci is not None
        ]) if any(r.mean_within_ci is not None for r in self.history) else np.nan
        option_errors = [r.option_error for r in self.history if np.isfinite(r.option_error)]
        option_ci_rate = np.mean([
            r.option_within_ci
            for r in self.history
            if r.option_within_ci is not None
        ]) if any(r.option_within_ci is not None for r in self.history) else np.nan
        tail_var_diffs = [r.tail_var_diff for r in self.history if np.isfinite(r.tail_var_diff)]
        tail_cvar_diffs = [r.tail_cvar_diff for r in self.history if np.isfinite(r.tail_cvar_diff)]

        return {
            'episodes': len(self.history),
            'avg_mean_error': float(np.mean(mean_errors)) if mean_errors else np.nan,
            'best_mean_error': float(np.min(mean_errors)) if mean_errors else np.nan,
            'avg_reward': float(np.mean(rewards)),
            'latest_learning_rate': self.history[-1].learning_rate,
            'validation_success_rate': float(validity_rate),
            'value_estimate': float(self.value_estimate),
            'avg_volatility_error': float(np.mean(vol_errors)) if vol_errors else np.nan,
            'volatility_within_ci_rate': float(vol_ci_rate) if np.isfinite(vol_ci_rate) else np.nan,
            'mean_within_ci_rate': float(mean_ci_rate) if np.isfinite(mean_ci_rate) else np.nan,
            'avg_option_error': float(np.mean(option_errors)) if option_errors else np.nan,
            'option_within_ci_rate': float(option_ci_rate) if np.isfinite(option_ci_rate) else np.nan,
            'avg_tail_var_diff': float(np.mean(tail_var_diffs)) if tail_var_diffs else np.nan,
            'avg_tail_cvar_diff': float(np.mean(tail_cvar_diffs)) if tail_cvar_diffs else np.nan
        }

