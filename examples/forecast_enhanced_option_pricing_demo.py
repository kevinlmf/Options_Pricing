"""
Forecast-Enhanced Option Pricing Demo

End-to-end example demonstrating:
1. Time series forecasting (LSTM + GARCH)
2. Option pricing with forecasted parameters
3. Comparison with traditional methods
4. Performance evaluation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from time_series_forecasting.forecast_interface import ForecastBasedOptionPricer
from models.black_scholes import BlackScholesModel, BSParameters


def generate_synthetic_price_data(n_days: int = 500, S0: float = 100.0,
                                  mu: float = 0.0001, sigma: float = 0.02,
                                  seed: int = 42) -> pd.Series:
    """
    Generate synthetic stock price data using geometric Brownian motion.

    Parameters:
    -----------
    n_days : int
        Number of days of data
    S0 : float
        Initial stock price
    mu : float
        Drift (daily)
    sigma : float
        Volatility (daily)
    seed : int
        Random seed

    Returns:
    --------
    pd.Series: Synthetic price data
    """
    np.random.seed(seed)

    # Generate returns
    returns = np.random.normal(mu, sigma, n_days)

    # Convert to prices
    prices = [S0]
    for r in returns:
        prices.append(prices[-1] * np.exp(r))

    # Create time series
    dates = pd.date_range(start='2023-01-01', periods=n_days+1, freq='D')
    price_series = pd.Series(prices, index=dates)

    return price_series


def demo_basic_forecasting():
    """Demonstrate basic price and volatility forecasting."""
    print("=" * 80)
    print("DEMO 1: Basic Time Series Forecasting")
    print("=" * 80)

    # Generate synthetic data
    print("\n1. Generating synthetic price data...")
    price_data = generate_synthetic_price_data(n_days=500, S0=100.0)
    print(f"   Generated {len(price_data)} days of price data")
    print(f"   Starting price: ${price_data.iloc[0]:.2f}")
    print(f"   Ending price: ${price_data.iloc[-1]:.2f}")

    # Split into train and test
    train_size = int(len(price_data) * 0.8)
    train_data = price_data[:train_size]
    test_data = price_data[train_size:]

    # Create forecast pricer
    print("\n2. Initializing forecast-based option pricer...")
    pricer = ForecastBasedOptionPricer(
        price_model='lstm',
        volatility_model='garch',
        option_model='black-scholes'
    )

    # Fit models
    print("\n3. Training forecasting models...")
    print("   This may take a few minutes...")
    fit_results = pricer.fit(
        price_history=train_data,
        seq_length=20,
        verbose=True
    )
    print("\n   Training complete!")

    # Generate forecasts
    print("\n4. Generating forecasts...")
    forecasts = pricer.forecast(
        price_history=train_data,
        seq_length=20,
        price_steps=5,
        volatility_horizon=5
    )

    print(f"\n   Price Forecasts (next 5 days):")
    for i, price in enumerate(forecasts['price_forecast'], 1):
        print(f"     Day {i}: ${price:.2f}")

    print(f"\n   Volatility Forecasts (next 5 days):")
    for i, vol in enumerate(forecasts['volatility_forecast'], 1):
        print(f"     Day {i}: {vol*100:.2f}%")

    return pricer, train_data, test_data


def demo_option_pricing_with_forecasts(pricer, price_data):
    """Demonstrate option pricing using forecasted parameters."""
    print("\n" + "=" * 80)
    print("DEMO 2: Option Pricing with Forecasted Parameters")
    print("=" * 80)

    # Option parameters
    K = 105.0  # Strike price
    T = 0.25   # 3 months to maturity
    r = 0.05   # Risk-free rate
    q = 0.02   # Dividend yield

    print(f"\n1. Option Parameters:")
    print(f"   Strike Price (K): ${K:.2f}")
    print(f"   Time to Maturity (T): {T*365:.0f} days")
    print(f"   Risk-free Rate (r): {r*100:.2f}%")
    print(f"   Dividend Yield (q): {q*100:.2f}%")

    # Price option with forecasts
    print("\n2. Pricing option with forecast-enhanced approach...")
    result = pricer.forecast_and_price(
        price_history=price_data,
        K=K,
        T=T,
        r=r,
        option_type='call',
        q=q,
        seq_length=20
    )

    print(f"\n3. Results:")
    print(f"   Current Spot Price: ${result['current_spot']:.2f}")
    print(f"   Forecasted Spot Price: ${result['forecasted_spot']:.2f}")
    print(f"   Forecasted Volatility: {result['forecasted_volatility']*100:.2f}%")
    print(f"\n   Option Price (Call): ${result['option_valuation']['price']:.4f}")

    print(f"\n4. Greeks:")
    greeks = result['option_valuation']['greeks']
    print(f"   Delta: {greeks['delta']:.4f}")
    print(f"   Gamma: {greeks['gamma']:.6f}")
    print(f"   Vega: {greeks['vega']:.4f}")
    print(f"   Theta: {greeks['theta']:.4f}")
    print(f"   Rho: {greeks['rho']:.4f}")

    return result


def demo_comparison_with_traditional():
    """Compare forecast-based pricing with traditional methods."""
    print("\n" + "=" * 80)
    print("DEMO 3: Comparison with Traditional Black-Scholes")
    print("=" * 80)

    # Generate data
    price_data = generate_synthetic_price_data(n_days=500, S0=100.0)
    train_data = price_data[:400]

    # Forecast-based approach
    print("\n1. Forecast-Based Approach")
    print("   Training models...")
    forecast_pricer = ForecastBasedOptionPricer(
        price_model='lstm',
        volatility_model='garch',
        option_model='black-scholes'
    )
    forecast_pricer.fit(price_history=train_data, seq_length=20, verbose=False)

    # Price with forecasts
    K, T, r, q = 105.0, 0.25, 0.05, 0.02
    forecast_result = forecast_pricer.forecast_and_price(
        price_history=train_data,
        K=K, T=T, r=r, q=q,
        option_type='call'
    )

    print(f"   Forecasted Spot: ${forecast_result['forecasted_spot']:.2f}")
    print(f"   Forecasted Vol: {forecast_result['forecasted_volatility']*100:.2f}%")
    print(f"   Option Price: ${forecast_result['option_valuation']['price']:.4f}")

    # Traditional Black-Scholes
    print("\n2. Traditional Black-Scholes Approach")
    current_price = train_data.iloc[-1]
    returns = np.log(train_data / train_data.shift(1)).dropna()
    historical_vol = returns.std() * np.sqrt(252)  # Annualized

    bs_params = BSParameters(S0=current_price, K=K, T=T, r=r, sigma=historical_vol, q=q)
    bs_model = BlackScholesModel(bs_params)
    bs_price = bs_model.call_price()

    print(f"   Current Spot: ${current_price:.2f}")
    print(f"   Historical Vol: {historical_vol*100:.2f}%")
    print(f"   Option Price: ${bs_price:.4f}")

    # Comparison
    print("\n3. Comparison")
    price_diff = forecast_result['option_valuation']['price'] - bs_price
    print(f"   Price Difference: ${price_diff:.4f} ({price_diff/bs_price*100:.2f}%)")

    spot_diff = forecast_result['forecasted_spot'] - current_price
    print(f"   Spot Price Change: ${spot_diff:.2f} ({spot_diff/current_price*100:.2f}%)")

    vol_diff = (forecast_result['forecasted_volatility'] - historical_vol) * 100
    print(f"   Volatility Change: {vol_diff:.2f} percentage points")

    return forecast_result, bs_price


def demo_sensitivity_analysis(pricer, price_data):
    """Demonstrate sensitivity to different strikes and maturities."""
    print("\n" + "=" * 80)
    print("DEMO 4: Sensitivity Analysis")
    print("=" * 80)

    current_price = price_data.iloc[-1]
    strikes = [current_price * k for k in [0.9, 0.95, 1.0, 1.05, 1.1]]
    maturities = [30/365, 60/365, 90/365, 180/365, 365/365]

    print(f"\n1. Pricing options with different strikes (T=90 days):")
    print(f"   {'Strike':<10} {'Moneyness':<12} {'Price':<12} {'Delta':<12}")
    print(f"   {'-'*50}")

    r, q = 0.05, 0.02
    T = 90/365

    for strike in strikes:
        result = pricer.forecast_and_price(
            price_history=price_data,
            K=strike,
            T=T,
            r=r,
            q=q,
            option_type='call'
        )
        price = result['option_valuation']['price']
        delta = result['option_valuation']['greeks']['delta']
        moneyness = strike / current_price

        print(f"   ${strike:<9.2f} {moneyness:<12.2f} ${price:<11.4f} {delta:<12.4f}")

    print(f"\n2. Pricing options with different maturities (ATM):")
    print(f"   {'Maturity (days)':<18} {'Price':<12} {'Theta':<12}")
    print(f"   {'-'*50}")

    strike = current_price  # ATM

    for maturity in maturities:
        result = pricer.forecast_and_price(
            price_history=price_data,
            K=strike,
            T=maturity,
            r=r,
            q=q,
            option_type='call'
        )
        price = result['option_valuation']['price']
        theta = result['option_valuation']['greeks']['theta']
        days = int(maturity * 365)

        print(f"   {days:<18} ${price:<11.4f} {theta:<12.4f}")


def main():
    """Run all demos."""
    print("\n")
    print("=" * 80)
    print("FORECAST-ENHANCED OPTION PRICING DEMONSTRATION")
    print("=" * 80)
    print("\nThis demo showcases the integration of time series forecasting")
    print("into options pricing using LSTM for price prediction and GARCH")
    print("for volatility forecasting.\n")

    try:
        # Demo 1: Basic forecasting
        pricer, train_data, test_data = demo_basic_forecasting()

        # Demo 2: Option pricing with forecasts
        demo_option_pricing_with_forecasts(pricer, train_data)

        # Demo 3: Comparison
        demo_comparison_with_traditional()

        # Demo 4: Sensitivity analysis
        demo_sensitivity_analysis(pricer, train_data)

        print("\n" + "=" * 80)
        print("DEMONSTRATION COMPLETE")
        print("=" * 80)
        print("\nKey Takeaways:")
        print("1. Time series forecasting can enhance option pricing")
        print("2. LSTM effectively captures price dynamics")
        print("3. GARCH models provide accurate volatility forecasts")
        print("4. Forecast-based approach adapts to market conditions")
        print("\n")

    except Exception as e:
        print(f"\nError during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
