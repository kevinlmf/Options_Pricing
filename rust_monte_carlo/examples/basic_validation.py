#!/usr/bin/env python3
"""
Basic Example: Validate Multi-Agent Predictions

This example shows how to use the Rust-accelerated Monte Carlo validator
to validate predictions from multiple agents in your options pricing system.
"""

import sys
sys.path.insert(0, '..')

from python.monte_carlo_validator import MonteCarloValidator, quick_validate, batch_validate
import time

print("="*80)
print("🚀 Rust Monte Carlo Validator - Basic Example")
print("="*80)

# Example 1: Quick validation check
print("\n📊 Example 1: Quick Validation")
print("-" * 40)

start = time.time()
is_valid = quick_validate(
    predicted_mean=105.0,
    predicted_std=8.5,
    n_simulations=50_000,
    initial_price=100.0,
    drift=0.05,
    volatility=0.2
)
elapsed = time.time() - start

print(f"Prediction: mean={105.0:.2f}, std={8.5:.2f}")
print(f"Result: {'✅ VALID' if is_valid else '❌ INVALID'}")
print(f"Time: {elapsed:.3f}s for 50,000 simulations")

# Example 2: Detailed validation
print("\n📊 Example 2: Detailed Validation")
print("-" * 40)

validator = MonteCarloValidator(
    n_simulations=100_000,
    n_steps=100,
    dt=1/252,  # Daily steps
    initial_price=100.0,
    drift=0.05,
    volatility=0.2
)

start = time.time()
result = validator.validate_agent_prediction(
    agent_id="momentum_agent",
    predicted_mean=105.0,
    predicted_std=8.5,
    confidence=0.95
)
elapsed = time.time() - start

print(f"Agent: {result.agent_id}")
print(f"Valid: {'✅ YES' if result.is_valid else '❌ NO'}")
print(f"Mean Error: {result.mean_error:.2f}")
print(f"Std Error: {result.std_error:.2f}")
print(f"Confidence Interval: [{result.confidence_interval[0]:.2f}, {result.confidence_interval[1]:.2f}]")
print(f"P-value: {result.p_value:.4f}")
print(f"\nSimulated Statistics:")
print(f"  Mean: {result.statistics['mean']:.2f}")
print(f"  Std: {result.statistics['std']:.2f}")
print(f"  VaR (95%): {result.statistics['var_95']:.2f}")
print(f"  CVaR (95%): {result.statistics['cvar_95']:.2f}")
print(f"\nTime: {elapsed:.3f}s for 100,000 simulations")

# Example 3: Batch validation of multiple agents
print("\n📊 Example 3: Multi-Agent Batch Validation")
print("-" * 40)

# Simulate predictions from 5 different agents
agent_predictions = [
    (105.0, 8.5),   # Momentum agent
    (102.0, 7.8),   # Mean reversion agent
    (110.0, 12.0),  # Volatility agent
    (150.0, 5.0),   # Bad agent (too optimistic)
    (98.0, 6.5),    # Conservative agent
]

start = time.time()
results = batch_validate(
    predictions=agent_predictions,
    n_simulations=50_000,
    initial_price=100.0,
    drift=0.05,
    volatility=0.2
)
elapsed = time.time() - start

print(f"Validated {len(agent_predictions)} agents in {elapsed:.3f}s")
print(f"Average time per agent: {elapsed/len(agent_predictions):.3f}s")
print("\nResults:")
for i, (pred, valid) in enumerate(zip(agent_predictions, results)):
    status = "✅ VALID" if valid else "❌ INVALID"
    print(f"  Agent {i+1} (mean={pred[0]:.1f}, std={pred[1]:.1f}): {status}")

# Example 4: Scenario Analysis
print("\n📊 Example 4: Volatility Scenario Analysis")
print("-" * 40)

try:
    volatility_scenarios = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4]

    start = time.time()
    scenarios = validator.run_scenario_analysis(volatility_scenarios)
    elapsed = time.time() - start

    print(f"Ran {len(volatility_scenarios)} scenarios in {elapsed:.3f}s\n")
    print("Vol   Mean    Std     VaR95   CVaR95")
    print("-" * 45)
    for scenario in scenarios:
        print(f"{scenario['volatility']:.2f}  "
              f"{scenario['mean']:6.2f}  "
              f"{scenario['std']:6.2f}  "
              f"{scenario['var_95']:6.2f}  "
              f"{scenario['cvar_95']:6.2f}")
except NotImplementedError:
    print("⚠️  Scenario analysis requires Rust backend")

# Example 5: Option Greeks
print("\n📊 Example 5: Option Greeks Calculation")
print("-" * 40)

try:
    greeks = validator.calculate_option_greeks(
        option_type="call",
        strike=100.0
    )

    print("Call Option Greeks (Strike=100):")
    print(f"  Delta: {greeks['delta']:.4f}")
    print(f"  Gamma: {greeks['gamma']:.4f}")
    print(f"  Vega:  {greeks['vega']:.4f}")
    print(f"  Theta: {greeks['theta']:.4f}")
except NotImplementedError:
    print("⚠️  Greeks calculation requires Rust backend")

print("\n" + "="*80)
print("✅ All examples completed!")
print("="*80)
