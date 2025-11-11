#!/usr/bin/env python3
"""
Multi-Agent vs Traditional Methods: When to Use What?

This demo shows:
1. Scenario 1: Stable Market → Traditional methods perform better
2. Scenario 2: Regime Change → Multi-Agent performs better
3. Scenario 3: Validation ensures we use the right method
"""

import sys
import os
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from time_series_forecasting.multi_agent import create_validated_forecaster
from time_series_forecasting.classical_models.garch import GARCHModel
from models.options_pricing.black_scholes import BlackScholesModel, BSParameters

print("=" * 80)
print("MULTI-AGENT vs TRADITIONAL METHODS: WHEN TO USE WHAT?")
print("=" * 80)


def generate_stable_market(n_days=252):
    """Generate stable market data (low volatility, consistent trend)"""
    np.random.seed(42)
    initial_price = 100.0
    drift = 0.08 / 252  # 8% annual drift
    volatility = 0.15 / np.sqrt(252)  # 15% annual vol (stable)
    
    returns = np.random.normal(drift, volatility, n_days)
    prices = initial_price * np.exp(np.cumsum(returns))
    return prices


def generate_regime_change_market(n_days=252):
    """Generate market with regime change (volatility spike)"""
    np.random.seed(123)
    initial_price = 100.0
    
    # First half: stable regime
    stable_drift = 0.05 / 252
    stable_vol = 0.12 / np.sqrt(252)
    
    # Second half: volatile regime (regime change)
    volatile_drift = -0.02 / 252
    volatile_vol = 0.35 / np.sqrt(252)
    
    mid_point = n_days // 2
    
    returns = np.zeros(n_days)
    returns[:mid_point] = np.random.normal(stable_drift, stable_vol, mid_point)
    returns[mid_point:] = np.random.normal(volatile_drift, volatile_vol, n_days - mid_point)
    
    prices = initial_price * np.exp(np.cumsum(returns))
    return prices


def traditional_forecast(prices):
    """Traditional time-series forecast (GARCH)"""
    returns = np.diff(np.log(prices))
    
    # Simple GARCH-like volatility estimate
    historical_vol = np.std(returns) * np.sqrt(252)
    historical_drift = np.mean(returns) * 252
    
    return {
        'volatility': historical_vol,
        'drift': historical_drift,
        'method': 'Traditional (GARCH)',
        'confidence': 0.8  # High confidence in stable markets
    }


def multi_agent_forecast(prices, enable_validation=True):
    """Multi-agent forecast with validation"""
    forecaster = create_validated_forecaster(
        validation_simulations=50_000,
        enable_validation=enable_validation
    )
    
    forecast = forecaster.forecast_with_validation(
        historical_prices=prices,
        return_validation_details=True
    )
    
    return {
        'volatility': forecast['implied_volatility'],
        'drift': forecast['implied_drift'],
        'method': 'Multi-Agent',
        'confidence': forecast['confidence'],
        'validated': forecast.get('validated', False),
        'validation_passed': forecast.get('validation', {}).get('is_valid', False),
        'p_value': forecast.get('validation', {}).get('p_value', 1.0)
    }


def price_option(S0, K, T, r, sigma):
    """Price an option using Black-Scholes"""
    params = BSParameters(S0=S0, K=K, T=T, r=r, sigma=sigma)
    model = BlackScholesModel(params)
    return model.call_price()


def scenario_1_stable_market():
    """Scenario 1: Stable Market - Traditional methods perform better"""
    print("\n" + "=" * 80)
    print("SCENARIO 1: STABLE MARKET")
    print("=" * 80)
    print("\n📊 Market Characteristics:")
    print("  • Low volatility (15% annual)")
    print("  • Consistent trend (8% annual drift)")
    print("  • No regime changes")
    print("  • Historical patterns are reliable")
    
    prices = generate_stable_market()
    current_price = prices[-1]
    
    print(f"\n  Current Price: ${current_price:.2f}")
    print(f"  Price Range: ${prices.min():.2f} - ${prices.max():.2f}")
    
    # Traditional forecast
    print("\n" + "-" * 80)
    print("1. TRADITIONAL FORECAST (GARCH)")
    print("-" * 80)
    trad_forecast = traditional_forecast(prices)
    print(f"  Volatility: {trad_forecast['volatility']:.2%}")
    print(f"  Drift:      {trad_forecast['drift']:+.2%}")
    print(f"  Confidence: {trad_forecast['confidence']:.1%}")
    
    # Multi-agent forecast
    print("\n" + "-" * 80)
    print("2. MULTI-AGENT FORECAST")
    print("-" * 80)
    ma_forecast = multi_agent_forecast(prices)
    print(f"  Volatility: {ma_forecast['volatility']:.2%}")
    print(f"  Drift:      {ma_forecast['drift']:+.2%}")
    print(f"  Confidence: {ma_forecast['confidence']:.1%}")
    if ma_forecast.get('validated'):
        status = "✅ PASSED" if ma_forecast['validation_passed'] else "❌ FAILED"
        print(f"  Validation: {status} (p={ma_forecast['p_value']:.4f})")
    
    # Compare option pricing
    print("\n" + "-" * 80)
    print("3. OPTION PRICING COMPARISON")
    print("-" * 80)
    K = current_price * 1.05  # 5% OTM
    T = 30 / 365  # 30 days
    r = 0.05
    
    trad_price = price_option(current_price, K, T, r, trad_forecast['volatility'])
    ma_price = price_option(current_price, K, T, r, ma_forecast['volatility'])
    
    print(f"  Strike: ${K:.2f}, Time to Expiry: {T*365:.0f} days")
    print(f"  Traditional Price: ${trad_price:.2f} (vol={trad_forecast['volatility']:.2%})")
    print(f"  Multi-Agent Price: ${ma_price:.2f} (vol={ma_forecast['volatility']:.2%})")
    print(f"  Difference: ${abs(trad_price - ma_price):.2f} ({abs(trad_price - ma_price)/trad_price*100:.1f}%)")
    
    # Conclusion
    print("\n" + "-" * 80)
    print("💡 CONCLUSION FOR SCENARIO 1")
    print("-" * 80)
    if abs(trad_price - ma_price) / trad_price < 0.1:  # Within 10%
        print("  ✅ Both methods perform similarly in stable markets")
        print("  → Traditional methods are simpler and faster")
        print("  → Use Traditional for stable, well-behaved markets")
    else:
        print("  ⚠️  Methods diverge - check validation results")
        if ma_forecast.get('validation_passed'):
            print("  → Multi-Agent validated - can use either method")
        else:
            print("  → Multi-Agent failed validation - prefer Traditional")


def scenario_2_regime_change():
    """Scenario 2: Regime Change - Multi-Agent performs better"""
    print("\n\n" + "=" * 80)
    print("SCENARIO 2: REGIME CHANGE MARKET")
    print("=" * 80)
    print("\n📊 Market Characteristics:")
    print("  • Volatility spike mid-period (12% → 35%)")
    print("  • Trend reversal (5% → -2% drift)")
    print("  • Structural break in market dynamics")
    print("  • Historical patterns become unreliable")
    
    prices = generate_regime_change_market()
    current_price = prices[-1]
    
    print(f"\n  Current Price: ${current_price:.2f}")
    print(f"  Price Range: ${prices.min():.2f} - ${prices.max():.2f}")
    print(f"  Note: Regime change occurred at day {len(prices)//2}")
    
    # Traditional forecast (uses all history, gets confused)
    print("\n" + "-" * 80)
    print("1. TRADITIONAL FORECAST (GARCH)")
    print("-" * 80)
    trad_forecast = traditional_forecast(prices)
    print(f"  Volatility: {trad_forecast['volatility']:.2%}")
    print(f"  Drift:      {trad_forecast['drift']:+.2%}")
    print(f"  Confidence: {trad_forecast['confidence']:.1%}")
    print(f"  ⚠️  Problem: Averages across regimes, misses structural change")
    
    # Multi-agent forecast (adapts to regime)
    print("\n" + "-" * 80)
    print("2. MULTI-AGENT FORECAST")
    print("-" * 80)
    ma_forecast = multi_agent_forecast(prices)
    print(f"  Volatility: {ma_forecast['volatility']:.2%}")
    print(f"  Drift:      {ma_forecast['drift']:+.2%}")
    print(f"  Confidence: {ma_forecast['confidence']:.1%}")
    if ma_forecast.get('validated'):
        status = "✅ PASSED" if ma_forecast['validation_passed'] else "❌ FAILED"
        print(f"  Validation: {status} (p={ma_forecast['p_value']:.4f})")
    print(f"  ✅ Advantage: Agents adapt to regime changes")
    
    # Compare option pricing
    print("\n" + "-" * 80)
    print("3. OPTION PRICING COMPARISON")
    print("-" * 80)
    K = current_price * 1.05  # 5% OTM
    T = 30 / 365  # 30 days
    r = 0.05
    
    trad_price = price_option(current_price, K, T, r, trad_forecast['volatility'])
    ma_price = price_option(current_price, K, T, r, ma_forecast['volatility'])
    
    print(f"  Strike: ${K:.2f}, Time to Expiry: {T*365:.0f} days")
    print(f"  Traditional Price: ${trad_price:.2f} (vol={trad_forecast['volatility']:.2%})")
    print(f"  Multi-Agent Price: ${ma_price:.2f} (vol={ma_forecast['volatility']:.2%})")
    print(f"  Difference: ${abs(trad_price - ma_price):.2f} ({abs(trad_price - ma_price)/trad_price*100:.1f}%)")
    
    # Conclusion
    print("\n" + "-" * 80)
    print("💡 CONCLUSION FOR SCENARIO 2")
    print("-" * 80)
    if ma_forecast.get('validation_passed'):
        print("  ✅ Multi-Agent validated and captures regime change")
        print("  → Use Multi-Agent for markets with regime changes")
        print("  → Multi-Agent provides better risk assessment")
    else:
        print("  ⚠️  Multi-Agent failed validation")
        print("  → System should fallback to Traditional")
        print("  → This demonstrates validation protecting against unreliable forecasts")


def scenario_3_validation_decision():
    """Scenario 3: How validation decides which method to use"""
    print("\n\n" + "=" * 80)
    print("SCENARIO 3: VALIDATION DECISION MECHANISM")
    print("=" * 80)
    print("\n📊 This scenario shows how Monte Carlo validation decides")
    print("   whether to use Multi-Agent or fallback to Traditional methods")
    
    # Use regime change market (more interesting)
    prices = generate_regime_change_market()
    current_price = prices[-1]
    
    print(f"\n  Current Price: ${current_price:.2f}")
    
    # Get both forecasts
    trad_forecast = traditional_forecast(prices)
    ma_forecast = multi_agent_forecast(prices, enable_validation=True)
    
    print("\n" + "-" * 80)
    print("VALIDATION PROCESS")
    print("-" * 80)
    print("1. Multi-Agent generates forecast")
    print(f"   → Volatility: {ma_forecast['volatility']:.2%}")
    print(f"   → Drift: {ma_forecast['drift']:+.2%}")
    
    if ma_forecast.get('validated'):
        print("\n2. Monte Carlo Validation (50k simulations)")
        print(f"   → P-value: {ma_forecast['p_value']:.4f}")
        
        if ma_forecast['validation_passed']:
            print("   → ✅ Validation PASSED (p > 0.05)")
            print("   → Decision: USE Multi-Agent forecast")
            selected_method = "Multi-Agent"
            selected_vol = ma_forecast['volatility']
        else:
            print("   → ❌ Validation FAILED (p < 0.05)")
            print("   → Decision: FALLBACK to Traditional methods")
            selected_method = "Traditional"
            selected_vol = trad_forecast['volatility']
    else:
        print("\n2. Validation not available (Rust module not built)")
        print("   → Decision: FALLBACK to Traditional methods (safer)")
        selected_method = "Traditional"
        selected_vol = trad_forecast['volatility']
    
    print("\n" + "-" * 80)
    print("FINAL DECISION")
    print("-" * 80)
    print(f"  Selected Method: {selected_method}")
    print(f"  Selected Volatility: {selected_vol:.2%}")
    
    # Price option with selected method
    K = current_price * 1.05
    T = 30 / 365
    r = 0.05
    final_price = price_option(current_price, K, T, r, selected_vol)
    
    print(f"  Option Price (using {selected_method}): ${final_price:.2f}")
    
    print("\n" + "-" * 80)
    print("💡 KEY INSIGHT")
    print("-" * 80)
    print("  Validation ensures we only use reliable forecasts")
    print("  • If Multi-Agent validated → Use it (better for regime changes)")
    print("  • If Multi-Agent failed → Fallback to Traditional (safer)")
    print("  • This protects against unreliable predictions")


def main():
    """Run all scenarios"""
    scenario_1_stable_market()
    scenario_2_regime_change()
    scenario_3_validation_decision()
    
    print("\n\n" + "=" * 80)
    print("SUMMARY: WHEN TO USE WHAT?")
    print("=" * 80)
    print("\n📊 Traditional Methods (GARCH, LSTM):")
    print("  ✅ Best for: Stable markets, consistent patterns")
    print("  ✅ Advantages: Fast, simple, reliable in normal conditions")
    print("  ❌ Weakness: Misses regime changes, structural breaks")
    
    print("\n🤖 Multi-Agent Methods:")
    print("  ✅ Best for: Regime changes, structural breaks, interpretability")
    print("  ✅ Advantages: Adapts to market structure, interpretable")
    print("  ❌ Weakness: More complex, requires validation")
    
    print("\n🧪 Monte Carlo Validation:")
    print("  ✅ Ensures forecasts are statistically sound")
    print("  ✅ Automatically selects best method")
    print("  ✅ Protects against unreliable predictions")
    
    print("\n💡 Recommendation:")
    print("  Use validated Multi-Agent system:")
    print("  • If validation passes → Use Multi-Agent (better for regime changes)")
    print("  • If validation fails → Auto-fallback to Traditional (safer)")
    print("  • Best of both worlds: Adaptability + Safety")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()



