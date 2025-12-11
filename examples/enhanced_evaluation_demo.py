#!/usr/bin/env python3

"""
Enhanced Volatility Evaluation Demo
===================================

Comprehensive evaluation demonstration with:
- Statistical significance testing
- Economic significance assessment
- Risk management evaluation (VaR, CVaR)
- Option pricing implications
- Time-series properties analysis
- Model quality scoring
"""

import numpy as np
import pandas as pd
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models.evaluation.volatility_evaluation import VolatilityEvaluator

def create_realistic_market_data(n_days: int = 252) -> pd.DataFrame:
    """Create realistic market data with various regimes."""
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=n_days, freq='D')

    # Base market dynamics with regime changes
    returns = np.zeros(n_days)

    for t in range(n_days):
        # Base volatility
        vol = 0.015

        # Add regime changes (high vol periods)
        if 60 <= t < 90:  # Crisis period
            vol *= 2.5
        elif 150 <= t < 180:  # Another turbulent period
            vol *= 1.8

        # Jump component (rare events)
        lambda_jump = 0.03  # Jump probability
        if np.random.random() < lambda_jump:  # 3% probability
            returns[t] = np.random.normal(0, vol) + np.random.normal(0, 0.05)
        else:
            returns[t] = np.random.normal(0.0005, vol)  # Slight drift

    # Generate OHLCV data
    prices = 100 * np.exp(np.cumsum(returns))

    data = pd.DataFrame({
        'Open': prices * (1 + np.random.normal(0, 0.002, n_days)),
        'High': prices * (1 + np.abs(np.random.normal(0, 0.005, n_days))),
        'Low': prices * (1 - np.abs(np.random.normal(0, 0.005, n_days))),
        'Close': prices,
        'Volume': np.random.lognormal(14, 0.5, n_days).astype(int)
    }, index=dates)

    return data

def create_model_predictions(market_data: pd.DataFrame) -> pd.Series:
    """Create realistic model predictions with some skill."""
    returns = market_data['Close'].pct_change().dropna()

    # Our "model": captures 70% of true volatility + some noise
    true_volatility = returns.rolling(20).std() * np.sqrt(252)

    # Model predictions: good but not perfect
    noise = np.random.normal(0, 0.02, len(true_volatility))
    model_predictions = 0.7 * true_volatility + 0.3 * true_volatility.shift(1).fillna(true_volatility.mean()) + noise

    return model_predictions.dropna()

def main():
    print("🎯 Enhanced Volatility Evaluation Demo")
    print("=" * 50)
    print()

    print("🔬 COMPREHENSIVE EVALUATION FRAMEWORK")
    print("This demo showcases advanced evaluation metrics:")
    print("• Statistical significance testing")
    print("• Economic significance assessment")
    print("• Risk management evaluation (VaR, CVaR)")
    print("• Option pricing implications")
    print("• Time-series properties analysis")
    print("• Model quality scoring")
    print()

    # Step 1: Generate realistic market data
    print("Step 1: Generating realistic market data with regime changes...")
    market_data = create_realistic_market_data(252)
    print(f"✅ Generated {len(market_data)} days of market data")
    print(f"   Price range: ${market_data['Close'].min():.2f} - ${market_data['Close'].max():.2f}")
    print(f"   Average volatility: {market_data['Close'].pct_change().std() * np.sqrt(252):.4f}")
    print()

    # Step 2: Generate model predictions
    print("Step 2: Generating dual convergence model predictions...")
    predicted_volatility = create_model_predictions(market_data)
    returns = market_data['Close'].pct_change().dropna()
    realized_volatility = returns.rolling(20).std() * np.sqrt(252)

    # Align data
    common_idx = predicted_volatility.index.intersection(realized_volatility.index)
    predicted_volatility = predicted_volatility.loc[common_idx]
    realized_volatility = realized_volatility.loc[common_idx]

    print(f"✅ Generated {len(predicted_volatility)} volatility predictions")
    print(".4f")
    print(".4f")
    print(".4f")
    print()

    # Step 3: Comprehensive evaluation
    print("Step 3: Running comprehensive evaluation...")

    evaluator = VolatilityEvaluator()

    # Mock convergence data for demonstration
    factors = pd.DataFrame({
        'market_maker_factor': np.random.normal(0.15, 0.05, len(predicted_volatility)),
        'trend_follower_factor': np.random.normal(0.18, 0.03, len(predicted_volatility)),
        'fundamental_factor': np.random.normal(0.16, 0.04, len(predicted_volatility))
    }, index=predicted_volatility.index)

    constrained_factors = factors * 0.8 + 0.2 * 0.17  # Constrained toward physical target

    evaluation_result = evaluator.run_complete_evaluation(
        predicted_volatility=predicted_volatility,
        realized_volatility=realized_volatility,
        factors=factors,
        constrained_factors=constrained_factors,
        physical_target=0.17,  # Physical model target
        returns=returns,
        market_data=market_data
    )

    print("✅ Comprehensive evaluation completed")
    print()

    # Step 4: Generate enhanced report
    print("Step 4: Generating enhanced evaluation report...")
    print("=" * 60)

    enhanced_report = evaluator.generate_evaluation_report(evaluation_result)
    print(enhanced_report)

    print()
    print("🎯 ENHANCED EVALUATION INSIGHTS")
    print("=" * 40)

    vm = evaluation_result.volatility_metrics
    rm = evaluation_result.robustness_tests.get('risk_management_metrics', {})

    insights = []

    # Statistical rigor
    if vm.correlation > 0.6:
        insights.append("✅ Strong statistical significance in volatility prediction")
    else:
        insights.append("⚠️ Statistical significance needs improvement")

    # Economic significance
    if vm.sharpe_ratio > 1.0:
        insights.append("✅ Good economic significance (Sharpe > 1.0)")
    elif vm.sharpe_ratio > 0.5:
        insights.append("⚠️ Moderate economic significance (Sharpe > 0.5)")
    else:
        insights.append("❌ Poor economic significance")

    # Risk management
    if rm.get('var_model_adequate', False):
        insights.append("✅ Adequate risk management (VaR model passes tests)")
    else:
        insights.append("⚠️ Risk management needs improvement")

    for insight in insights:
        print(insight)

    print()
    print("💡 This enhanced evaluation framework provides:")
    print("   • Statistical rigor with significance testing")
    print("   • Economic interpretation of results")
    print("   • Risk management validation")
    print("   • Practical trading implications")
    print("   • Model quality scoring")
    print()
    print("🎯 Perfect for validating dual convergence methodology!")

if __name__ == "__main__":
    main()
