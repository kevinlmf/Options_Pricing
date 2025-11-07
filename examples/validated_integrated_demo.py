#!/usr/bin/env python3
"""
Validated Integrated Portfolio Optimization Demo

This demo shows the complete pipeline with Monte Carlo validation:
1. Generate market data
2. Run multi-agent forecasting
3. Validate with Monte Carlo (50k simulations)
4. If validation passes → Use multi-agent forecast
5. If validation fails → Fall back to traditional time-series methods
6. Continue with pricing, optimization, risk control, and monitoring
"""

import sys
import os
import numpy as np
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import components
from time_series_forecasting.deep_learning.rnn_models import LSTMForecaster
from time_series_forecasting.classical_models.garch import GARCHModel
from time_series_forecasting.multi_agent import create_validated_forecaster
from models.options_pricing.black_scholes import BlackScholesModel, BSParameters
from models.optimization_methods.strategy.portfolio_optimizer import PortfolioOptimizer, PortfolioGreeks
from risk.portfolio_risk import PortfolioRiskMetrics
from evaluation_modules.realtime_monitor import RealTimeMonitor

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def generate_synthetic_data(n_periods: int = 200) -> np.ndarray:
    """Generate synthetic Bitcoin price data"""
    np.random.seed(42)
    prices = [50000.0]

    for _ in range(n_periods - 1):
        ret = np.random.randn() * 0.02  # 2% daily vol
        prices.append(prices[-1] * (1 + ret))

    return np.array(prices)


def main():
    print("="*80)
    print(" VALIDATED INTEGRATED PORTFOLIO OPTIMIZATION DEMO")
    print("="*80)
    print("\nThis demo shows the complete pipeline with Monte Carlo validation:")
    print("  1. Market Data")
    print("  2. Multi-Agent Forecasting")
    print("  3. Monte Carlo Validation (50k simulations)")
    print("  4. Decision: Use Multi-Agent if validated, else fallback to traditional")
    print("  5. Options Pricing")
    print("  6. Portfolio Optimization (CVaR + Greek Constraints)")
    print("  7. Risk Control")
    print("  8. Real-time Monitoring")
    print("="*80)

    # ========================================================================
    # LAYER 1: MARKET DATA
    # ========================================================================
    print("\n[LAYER 1: MARKET DATA]")
    print("-" * 80)

    prices = generate_synthetic_data(n_periods=150)
    current_price = prices[-1]
    historical_prices = prices[:-1]

    print(f"✓ Loaded {len(prices)} periods of data")
    print(f"  Current Price: ${current_price:,.2f}")
    print(f"  Price Range: ${prices.min():,.2f} - ${prices.max():,.2f}")

    # ========================================================================
    # LAYER 2: MULTI-AGENT FORECASTING WITH VALIDATION
    # ========================================================================
    print("\n[LAYER 2: MULTI-AGENT FORECASTING WITH VALIDATION]")
    print("-" * 80)
    print("🤖 Running multi-agent structural forecasting...")
    print("🧪 Validating with Monte Carlo (50k simulations)...")
    print()

    # Create validated forecaster
    forecaster = create_validated_forecaster(
        validation_simulations=50_000,
        enable_validation=True
    )

    # Get validated forecast
    forecast = forecaster.forecast_with_validation(
        historical_prices=historical_prices,
        return_validation_details=True
    )

    # Extract parameters
    implied_volatility = forecast['implied_volatility']
    implied_drift = forecast['implied_drift']
    confidence = forecast['confidence']
    
    # Check validation result
    validation_passed = forecast.get('validated', False) and forecast.get('validation', {}).get('is_valid', False)
    validation_p_value = forecast.get('validation', {}).get('p_value', 1.0)

    print(f"✓ Multi-Agent Forecast Complete")
    print(f"  Implied Volatility: {implied_volatility:.2%}")
    print(f"  Implied Drift:      {implied_drift:+.2%}")
    print(f"  Confidence:         {confidence:.1%}")
    print()
    print(f"✓ Monte Carlo Validation Complete")
    print(f"  Validation Status: {'✅ PASSED' if validation_passed else '❌ FAILED'}")
    print(f"  P-value:            {validation_p_value:.4f}")
    
    if validation_passed:
        print(f"\n  → Multi-Agent forecast is statistically sound")
        print(f"  → Using multi-agent parameters for pricing")
        use_multi_agent = True
        forecast_method = "Multi-Agent (Validated)"
    else:
        print(f"\n  ⚠️  Multi-Agent forecast deviates from simulations")
        print(f"  → Falling back to traditional time-series methods")
        use_multi_agent = False
        forecast_method = "Traditional Time-Series (Fallback)"
        
        # Fallback to traditional methods
        returns = np.diff(np.log(historical_prices))
        implied_volatility = np.std(returns) * np.sqrt(252)  # Annualized
        implied_drift = np.mean(returns) * 252  # Annualized
        confidence = 0.5  # Lower confidence for fallback

    print(f"\n📊 Selected Forecast Method: {forecast_method}")
    print(f"  Final Volatility: {implied_volatility:.2%}")
    print(f"  Final Drift:      {implied_drift:+.2%}")
    print(f"  Confidence:       {confidence:.1%}")

    # ========================================================================
    # LAYER 3: OPTIONS PRICING
    # ========================================================================
    print("\n[LAYER 3: OPTIONS PRICING]")
    print("-" * 80)

    # Generate option contracts
    strikes = np.linspace(current_price * 0.9, current_price * 1.1, 8)
    expiries = [7/365, 14/365, 30/365, 60/365]
    
    option_data = {}
    option_id = 0
    
    for strike in strikes:
        for expiry in expiries:
            for opt_type in ['call', 'put']:
                option_id += 1
                
                bs_params = BSParameters(
                    S0=current_price,
                    K=strike,
                    T=expiry,
                    r=0.05,
                    sigma=implied_volatility
                )
                bs_model = BlackScholesModel(bs_params)
                
                price = bs_model.call_price() if opt_type == 'call' else bs_model.put_price()
                greeks = bs_model.greeks(opt_type)
                
                option_data[f"OPT_{option_id}"] = {
                    'type': opt_type,
                    'strike': strike,
                    'expiry': expiry,
                    'price': price,
                    'greeks': {
                        'delta': greeks['delta'],
                        'gamma': greeks['gamma'],
                        'vega': greeks['vega'],
                        'theta': greeks['theta']
                    }
                }

    print(f"✓ Priced {len(option_data)} options")
    print(f"  Using volatility: {implied_volatility:.2%} ({forecast_method})")
    print(f"  Price range: ${min(o['price'] for o in option_data.values()):.2f} - ${max(o['price'] for o in option_data.values()):.2f}")

    # ========================================================================
    # LAYER 4: PORTFOLIO OPTIMIZATION
    # ========================================================================
    print("\n[LAYER 4: PORTFOLIO OPTIMIZATION]")
    print("-" * 80)

    # Forecast returns (adjusted by confidence)
    forecasted_returns = {}
    for opt_id, opt_data in option_data.items():
        moneyness = current_price / opt_data['strike']
        time_factor = 1.0 - opt_data['expiry']
        
        if opt_data['type'] == 'call':
            base_ret = 0.15 + max(0, (moneyness - 1)) * 0.5 + time_factor * 0.05
        else:
            base_ret = 0.10 + max(0, (1 - moneyness)) * 0.5 + time_factor * 0.05
        
        # Adjust by confidence
        forecasted_returns[opt_id] = base_ret * confidence

    # Calculate average expiry to set appropriate Theta limit
    avg_expiry = np.mean([opt_data['expiry'] for opt_data in option_data.values()])
    # Theta is annualized, so adjust limit based on expiry
    # For short-term options (7-30 days), Theta can be large
    max_theta_for_optimizer = 50000.0 if avg_expiry < 0.1 else 20000.0
    
    optimizer = PortfolioOptimizer(
        risk_aversion=0.1,
        delta_neutrality_weight=0.0,  # Don't force delta-neutral, just respect limits
        confidence_level=0.95,
        max_delta=400.0,  # Slightly tighter to ensure we stay within limits
        max_vega=2500.0,  # Slightly tighter
        max_gamma=40.0,   # Slightly tighter
        max_theta=max_theta_for_optimizer  # Add Theta constraint to optimizer
    )

    n_opts = len(option_data)
    return_covariance = np.eye(n_opts) * 0.04

    opt_result = optimizer.optimize_portfolio(
        option_data=option_data,
        forecasted_returns=forecasted_returns,
        return_covariance=return_covariance,
        max_position_size=50.0,
        n_scenarios=500
    )

    print(f"✓ Portfolio Optimization Complete")
    print(f"  Converged: {opt_result.converged}")
    print(f"  Positions: {len(opt_result.positions)}")
    print(f"\n  Portfolio Greeks:")
    print(f"    Δ_p: {opt_result.portfolio_greeks.delta:,.2f}")
    print(f"    Γ_p: {opt_result.portfolio_greeks.gamma:,.2f}")
    print(f"    ν_p: {opt_result.portfolio_greeks.vega:,.2f}")
    print(f"    Θ_p: {opt_result.portfolio_greeks.theta:,.2f}")

    # ========================================================================
    # LAYER 5-7: RISK CONTROL, EXECUTION, MONITORING
    # ========================================================================
    print("\n[LAYER 5-7: RISK CONTROL, EXECUTION, MONITORING]")
    print("-" * 80)

    # Risk limits (must match optimizer constraints)
    max_delta = 500.0  # Slightly higher than optimizer (400) to allow for small violations
    max_gamma = 50.0   # Slightly higher than optimizer (40)
    max_vega = 3000.0  # Slightly higher than optimizer (2500)
    # Theta is annualized, so for a portfolio of options, large values are expected
    # Use the same limit as optimizer
    max_theta = max_theta_for_optimizer
    
    # Check portfolio risk by validating Greeks
    portfolio_greeks = opt_result.portfolio_greeks
    risk_approved = (
        abs(portfolio_greeks.delta) <= max_delta and
        abs(portfolio_greeks.gamma) <= max_gamma and
        abs(portfolio_greeks.vega) <= max_vega and
        abs(portfolio_greeks.theta) <= max_theta
    )

    print(f"✓ Risk Control Check: {'✅ PASSED' if risk_approved else '❌ FAILED'}")
    if not risk_approved:
        print(f"  Delta: {portfolio_greeks.delta:.2f} (limit: {max_delta})")
        print(f"  Gamma: {portfolio_greeks.gamma:.2f} (limit: {max_gamma})")
        print(f"  Vega:  {portfolio_greeks.vega:.2f} (limit: {max_vega})")
        print(f"  Theta: {portfolio_greeks.theta:.2f} (limit: {max_theta})")

    # ========================================================================
    # LAYER 6: PORTFOLIO CONSTRUCTION & TRADE EXECUTION
    # ========================================================================
    print("\n[LAYER 6: PORTFOLIO CONSTRUCTION & TRADE EXECUTION]")
    print("-" * 80)
    
    # Always construct portfolio to demonstrate the process
    # In production, you might skip if risk check fails
    if not risk_approved:
        print("⚠️  Risk check failed - constructing portfolio anyway for demonstration")
        print("   In production, you would adjust positions or risk limits first")
        print()
    
    # Construct actual option portfolio from optimized positions
    from models.optimization_methods.option_portfolio import (
        OptionPortfolio, OptionPosition, OptionType, PositionType
    )
    
    portfolio = OptionPortfolio(name="Validated Multi-Agent Portfolio")
    positions_dict = {opt_id: pos for opt_id, pos in opt_result.positions.items() if pos != 0}
    
    print(f"✓ Constructing Portfolio with {len(positions_dict)} positions...")
    
    total_cost = 0.0
    executed_trades = []
    
    for opt_id, quantity in positions_dict.items():
        opt_data = option_data[opt_id]
        
        # Determine position type (long if quantity > 0, short if < 0)
        pos_type = PositionType.LONG if quantity > 0 else PositionType.SHORT
        opt_type = OptionType.CALL if opt_data['type'] == 'call' else OptionType.PUT
        
        # Create option position
        position = OptionPosition(
            symbol="STOCK",
            option_type=opt_type,
            position_type=pos_type,
            strike=opt_data['strike'],
            expiry=opt_data['expiry'],
            quantity=abs(int(quantity)),
            premium=opt_data['price'],
            underlying_price=current_price,
            volatility=implied_volatility,
            risk_free_rate=0.05
        )
        
        portfolio.add_position(position)
        
        # Calculate trade cost (premium * quantity * 100 per contract)
        trade_cost = opt_data['price'] * abs(quantity) * 100
        total_cost += trade_cost if pos_type == PositionType.LONG else -trade_cost
        
        executed_trades.append({
            'option_id': opt_id,
            'type': opt_data['type'],
            'strike': opt_data['strike'],
            'expiry_days': int(opt_data['expiry'] * 365),
            'quantity': int(quantity),
            'position': 'LONG' if quantity > 0 else 'SHORT',
            'premium': opt_data['price'],
            'cost': trade_cost
        })
    
    print(f"\n✓ Portfolio Constructed:")
    print(f"  Total Positions: {len(executed_trades)}")
    print(f"  Total Cost: ${total_cost:,.2f}")
    
    print(f"\n📋 Executed Trades:")
    for i, trade in enumerate(executed_trades[:10], 1):  # Show first 10
        print(f"  {i}. {trade['position']} {abs(trade['quantity'])} "
              f"{trade['type'].upper()} @ ${trade['strike']:.2f} "
              f"(Exp: {trade['expiry_days']}d, Cost: ${trade['cost']:,.2f})")
    if len(executed_trades) > 10:
        print(f"  ... and {len(executed_trades) - 10} more positions")
    
    # Portfolio statistics
    portfolio_value = portfolio.total_value
    portfolio_pnl = portfolio.total_pnl
    portfolio_greeks_full = portfolio.portfolio_greeks
    
    print(f"\n📊 Portfolio Statistics:")
    print(f"  Current Value: ${portfolio_value:,.2f}")
    if abs(total_cost) > 1e-6:
        print(f"  Total P&L: ${portfolio_pnl:,.2f} ({portfolio_pnl/total_cost*100:+.2f}%)")
    else:
        print(f"  Total P&L: ${portfolio_pnl:,.2f}")
    print(f"  Portfolio Greeks:")
    print(f"    Δ: {portfolio_greeks_full['delta']:,.2f}")
    print(f"    Γ: {portfolio_greeks_full['gamma']:,.2f}")
    print(f"    ν: {portfolio_greeks_full['vega']:,.2f}")
    print(f"    Θ: {portfolio_greeks_full['theta']:,.2f}")
    
    if not risk_approved:
        print(f"\n⚠️  WARNING: Portfolio constructed despite risk check failure")
        print(f"   In production, this would be rejected or positions adjusted")

    # ========================================================================
    # LAYER 7: REAL-TIME MONITORING
    # ========================================================================
    print("\n[LAYER 7: REAL-TIME MONITORING]")
    print("-" * 80)
    
    # Initialize monitor
    monitor = RealTimeMonitor(initial_capital=1000000.0)
    
    # Update monitor with actual portfolio positions
    positions_for_monitor = {}
    greeks_for_monitor = {}
    
    for opt_id, quantity in positions_dict.items():
        if quantity != 0:
            positions_for_monitor[opt_id] = quantity
            greeks_for_monitor[opt_id] = option_data[opt_id]['greeks']
    
    if positions_for_monitor:
        monitor.update(positions_for_monitor, option_data, greeks_for_monitor)
        
        print(f"✓ Real-time Monitoring Initialized")
        metrics = monitor.get_metrics_summary()
        if metrics:
            portfolio_info = metrics.get('portfolio', {})
            print(f"  Portfolio Value: ${portfolio_info.get('value', 0):,.2f}")
            print(f"  Total P&L: ${portfolio_info.get('total_pnl', 0):,.2f}")
            
            # Additional metrics
            if 'risk_metrics' in metrics:
                risk = metrics['risk_metrics']
                print(f"  VaR (95%): ${risk.get('var_95', 0):,.2f}")
                print(f"  CVaR (95%): ${risk.get('cvar_95', 0):,.2f}")
        else:
            print(f"  Portfolio Value: ${monitor.portfolio_value:,.2f}")
            print(f"  Total P&L: ${monitor.portfolio_value - monitor.initial_capital:,.2f}")
    else:
        print(f"✓ Real-time Monitoring Initialized (No positions)")

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "="*80)
    print("✅ DEMO COMPLETE")
    print("="*80)
    print(f"\n📊 Summary:")
    print(f"  Forecast Method: {forecast_method}")
    print(f"  Validation: {'✅ PASSED' if validation_passed else '❌ FAILED'} (p={validation_p_value:.4f})")
    print(f"  Options Priced: {len(option_data)}")
    if 'executed_trades' in locals() and len(executed_trades) > 0:
        print(f"  Portfolio Positions: {len(executed_trades)}")
        print(f"  Portfolio Value: ${portfolio_value:,.2f}")
        print(f"  Total P&L: ${portfolio_pnl:,.2f}")
        if not risk_approved:
            print(f"  ⚠️  Risk Status: FAILED (portfolio constructed for demo)")
    else:
        print(f"  Portfolio Positions: 0")
    print(f"  Risk Check: {'✅ PASSED' if risk_approved else '❌ FAILED'}")
    print(f"\n💡 Key Insight:")
    if validation_passed:
        print(f"  Multi-Agent forecast validated → Used for pricing and optimization")
        print(f"  Portfolio constructed and trades executed successfully")
        print(f"  This demonstrates the power of structural forecasting with validation")
    else:
        print(f"  Multi-Agent forecast failed validation → Fallback to traditional methods")
        print(f"  This shows the importance of validation in preventing unreliable forecasts")
    print("\n" + "="*80)


if __name__ == "__main__":
    main()

