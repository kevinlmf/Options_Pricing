#!/usr/bin/env python3
"""
Adaptive Training Demo for the Validated Multi-Agent Forecaster
===============================================================

Demonstrates how the reinforcement-style coordinator iteratively
runs the validated forecaster, uses Monte Carlo validation as the
reward signal, and adapts the agent learning rate to shrink
forecast vs simulation error.
"""

import sys
import time
from pathlib import Path

import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from time_series_forecasting.multi_agent import (
    create_validated_forecaster,
    AdaptiveAgentCoordinator
)
from monitoring.reporting import generate_training_report


def generate_synthetic_prices(
    n_days: int = 252,
    initial_price: float = 100.0,
    annual_drift: float = 0.10,
    annual_vol: float = 0.20,
    seed: int = 123
) -> np.ndarray:
    """Utility to produce a GBM-like synthetic price path."""
    np.random.seed(seed)
    dt = 1 / 252
    returns = np.random.normal(
        loc=annual_drift * dt,
        scale=annual_vol * np.sqrt(dt),
        size=n_days
    )
    prices = initial_price * np.exp(np.cumsum(returns))
    return prices


def main():
    print("=" * 80)
    print("🤖 ADAPTIVE MULTI-AGENT TRAINING DEMO")
    print("=" * 80)

    prices = generate_synthetic_prices()
    print(f"Generated synthetic series with {len(prices)} observations.")
    print(f"  Start: {prices[0]:.2f}, End: {prices[-1]:.2f}, Return: {(prices[-1]/prices[0]-1):+.2%}")
    print("-" * 80)

    # Create validated forecaster with adaptive learning enabled
    forecaster = create_validated_forecaster(
        validation_simulations=20_000,
        auto_adapt_agents=True,
        adaptive_learning_rate=0.05,
        risk_free_rate=0.0
    )

    coordinator = AdaptiveAgentCoordinator(
        forecaster=forecaster,
        reward_penalty=2.0,
        reward_smoothing=0.2,
        learning_rate_bounds=(0.01, 0.1),
        reward_scale=0.05,
        volatility_error_weight=0.1
    )

    iterations = 20
    print(f"Training for {iterations} iterations ...")
    print("-" * 80)

    for i in range(iterations):
        start = time.time()
        result = coordinator.step(prices)
        elapsed = time.time() - start

        print(f"[Iteration {i+1:02d}] "
              f"reward={result.reward:+.3f}  "
              f"mean_error={result.mean_error:.2f}  "
              f"p_value={result.p_value:.4f}  "
              f"learning_rate={result.learning_rate:.3f}  "
              f"{'✅' if result.is_valid else '❌'}  "
              f"({elapsed*1000:.1f} ms)")

    print("-" * 80)
    summary = coordinator.summary()
    if summary:
        print("📈 Training Summary:")
        print(f"  Episodes:               {summary['episodes']}")
        print(f"  Avg mean error:         {summary['avg_mean_error']:.2f}")
        print(f"  Best mean error:        {summary['best_mean_error']:.2f}")
        print(f"  Avg reward:             {summary['avg_reward']:.3f}")
        print(f"  Validation success rate:{summary['validation_success_rate']:.1%}")
        print(f"  Avg volatility error:   {summary['avg_volatility_error']:.4f}")
        vol_ci = summary['volatility_within_ci_rate']
        if np.isfinite(vol_ci):
            print(f"  Volatility within CI:   {vol_ci:.1%}")
        mean_ci = summary.get('mean_within_ci_rate')
        if mean_ci is not None and np.isfinite(mean_ci):
            print(f"  Mean within CI:         {mean_ci:.1%}")
        option_err = summary.get('avg_option_error')
        if option_err is not None and np.isfinite(option_err):
            print(f"  Avg option error:       {option_err:.4f}")
        option_ci = summary.get('option_within_ci_rate')
        if option_ci is not None and np.isfinite(option_ci):
            print(f"  Option price within CI: {option_ci:.1%}")
        tail_var = summary.get('avg_tail_var_diff')
        tail_cvar = summary.get('avg_tail_cvar_diff')
        if tail_var is not None and np.isfinite(tail_var):
            print(f"  Avg VaR diff:           {tail_var:.4f}")
        if tail_cvar is not None and np.isfinite(tail_cvar):
            print(f"  Avg CVaR diff:          {tail_cvar:.4f}")
        print(f"  Final learning rate:    {summary['latest_learning_rate']:.3f}")
    else:
        print("No training history available.")

    reports_dir = project_root / "reports"
    reports_dir.mkdir(exist_ok=True)
    report_path = reports_dir / "adaptive_training_report.json"
    generate_training_report(coordinator.history, summary or {}, report_path)
    print(f"\n📄 Training report saved to: {report_path}")

    print("=" * 80)
    print("Adaptive training demo complete.")
    print("=" * 80)


if __name__ == "__main__":
    main()

