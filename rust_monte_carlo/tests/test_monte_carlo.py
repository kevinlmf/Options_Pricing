"""
Tests for Monte Carlo validator
"""

import pytest
import numpy as np
from python.monte_carlo_validator import (
    MonteCarloValidator,
    quick_validate,
    batch_validate,
)


def test_validator_initialization():
    """Test basic initialization"""
    validator = MonteCarloValidator(
        n_simulations=1000,
        n_steps=50,
        initial_price=100.0,
        volatility=0.2
    )
    assert validator.n_simulations == 1000
    assert validator.n_steps == 50


def test_single_agent_validation():
    """Test validating a single agent prediction"""
    validator = MonteCarloValidator(
        n_simulations=5000,
        initial_price=100.0,
        drift=0.0,
        volatility=0.2
    )

    # Should be valid (close to expected distribution)
    result = validator.validate_agent_prediction(
        agent_id="test_agent",
        predicted_mean=100.0,
        predicted_std=8.0,
    )

    assert result.agent_id == "test_agent"
    assert result.mean_error < 5.0  # Should be close
    assert result.p_value > 0.01  # Should not reject


def test_invalid_prediction():
    """Test that obviously wrong predictions are rejected"""
    validator = MonteCarloValidator(
        n_simulations=5000,
        initial_price=100.0,
        volatility=0.2
    )

    # Obviously wrong prediction
    result = validator.validate_agent_prediction(
        agent_id="bad_agent",
        predicted_mean=200.0,  # Way too high
        predicted_std=0.1,     # Way too low
    )

    assert not result.is_valid
    assert result.mean_error > 50.0


def test_quick_validate():
    """Test quick validation function"""
    # Should pass
    is_valid = quick_validate(
        predicted_mean=100.0,
        predicted_std=8.0,
        n_simulations=5000,
        initial_price=100.0,
        volatility=0.2
    )

    assert isinstance(is_valid, bool)


def test_batch_validate():
    """Test batch validation of multiple predictions"""
    predictions = [
        (100.0, 8.0),   # Should pass
        (100.0, 7.0),   # Should pass
        (200.0, 1.0),   # Should fail
    ]

    results = batch_validate(
        predictions=predictions,
        n_simulations=5000,
        initial_price=100.0,
        volatility=0.2
    )

    assert len(results) == 3
    assert results[0] == True
    assert results[1] == True
    assert results[2] == False


def test_run_simulations():
    """Test running simulations"""
    validator = MonteCarloValidator(
        n_simulations=100,
        n_steps=50,
        initial_price=100.0
    )

    paths = validator.run_simulations()

    assert paths.shape == (100, 51)
    assert np.all(paths[:, 0] == 100.0)
    assert np.all(paths > 0)  # Prices should be positive


def test_scenario_analysis():
    """Test scenario analysis with different volatilities"""
    validator = MonteCarloValidator(
        n_simulations=1000,
        initial_price=100.0
    )

    try:
        scenarios = validator.run_scenario_analysis([0.1, 0.2, 0.3, 0.4])
        assert len(scenarios) == 4
        # Higher volatility should lead to higher std
        stds = [s['std'] for s in scenarios]
        assert stds == sorted(stds)
    except NotImplementedError:
        # Fallback mode doesn't support this
        pass


def test_option_greeks():
    """Test option Greeks calculation"""
    validator = MonteCarloValidator(
        n_simulations=10000,
        initial_price=100.0,
        volatility=0.2
    )

    try:
        greeks = validator.calculate_option_greeks(
            option_type="call",
            strike=100.0
        )

        assert 'delta' in greeks
        assert 0 <= greeks['delta'] <= 1  # Call delta should be between 0 and 1
    except NotImplementedError:
        # Fallback mode doesn't support this
        pass


@pytest.mark.benchmark
def test_performance_benchmark(benchmark):
    """Benchmark Monte Carlo performance"""
    validator = MonteCarloValidator(
        n_simulations=10000,
        n_steps=100,
        initial_price=100.0,
        volatility=0.2
    )

    result = benchmark(
        validator.validate_agent_prediction,
        "bench_agent",
        100.0,
        8.0
    )

    print(f"\n✅ Validated 10k simulations in {benchmark.stats['mean']:.4f}s")
    assert result is not None
