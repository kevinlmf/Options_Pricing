"""
Integrated Portfolio Optimization Demo

Complete pipeline demonstration:

[Market Data]
    ↓
[Forecasting Layer] → σ_t+1, μ_t+1
    ↓
[Pricing Engine] → V_i(S, σ, t, r)
    ↓
Compute Greeks: Δ_i, Γ_i, Vega_i, Θ_i
    ↓
Aggregate Greeks → (Δ_p, Γ_p, Vega_p, Θ_p)
    ↓
Portfolio Optimization:
    - CVaR(returns)
    - Constraints on Greeks
    - Objective: Sharpe / VaR / Expected Utility
    ↓
Risk Control
    ↓
Execution
    ↓
Monitoring (Real-time feedback)
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
from time_series_forecasting.multi_agent import MultiAgentForecaster
from time_series_forecasting.forecast_comparator import ForecastComparator
from models.options_pricing.black_scholes import BlackScholesModel, BSParameters
from models.optimization_methods.strategy.portfolio_optimizer import PortfolioOptimizer, PortfolioGreeks
from risk.bitcoin_risk_controller import BitcoinRiskController, BitcoinRiskLimits, OrderProposal
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
    print(" INTEGRATED PORTFOLIO OPTIMIZATION DEMO")
    print("="*80)
    print("\nThis demo shows the complete pipeline:")
    print("  1. Market Data")
    print("  2. Forecasting (σ, μ)")
    print("  3. Options Pricing")
    print("  4. Greeks Computation & Aggregation")
    print("  5. Portfolio Optimization (CVaR + Greek Constraints)")
    print("  6. Risk Control")
    print("  7. Execution")
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
    # LAYER 2: FORECASTING → σ, μ (COMPARISON: Reduced-form vs Structural)
    # ========================================================================
    print("\n[LAYER 2: FORECASTING COMPARISON]")
    print("-" * 80)
    print("📊 Comparing two approaches:")
    print("  • Reduced-form (Time-Series): Statistical fitting")
    print("  • Structural (Multi-Agent):   Behavioral simulation")
    print()

    # 2A: Reduced-form forecast (traditional time-series)
    returns = np.diff(np.log(historical_prices))
    rf_vol = np.std(returns) * np.sqrt(252)  # Annualized
    rf_drift = np.mean(returns) * 252  # Annualized

    # Mock reduced-form forecaster
    class ReducedFormForecaster:
        def forecast(self, prices):
            return {'volatility': rf_vol, 'drift': rf_drift}

    rf_forecaster = ReducedFormForecaster()

    # 2B: Structural forecast (multi-agent simulation)
    ma_forecaster = MultiAgentForecaster(simulation_periods=min(100, len(historical_prices)))

    # 2C: Run comparison
    comparator = ForecastComparator()
    comparison = comparator.compare_forecasts(
        historical_prices=historical_prices,
        reduced_form_forecaster=rf_forecaster,
        structural_forecaster=ma_forecaster
    )

    # Display comparison
    print(comparator.generate_comparison_report(comparison))

    # Use structural forecast for downstream (more robust for regime changes)
    realized_vol = comparison['structural']['volatility']
    expected_return = comparison['structural']['drift']

    print(f"\n✓ Using Structural Forecast for downstream:")
    print(f"  σ (volatility): {realized_vol:.2%} per year")
    print(f"  μ (drift):      {expected_return:.2%} per year")
    print(f"  Regime:         {comparison['structural']['regime']}")
    print(f"  Confidence:     {comparison['structural']['confidence']:.1%}")

    # ========================================================================
    # LAYER 3: PRICING ENGINE → V_i(S, σ, t, r)
    # ========================================================================
    print("\n[LAYER 3: PRICING ENGINE]")
    print("-" * 80)

    r = 0.05  # Risk-free rate

    # Price multiple options
    strikes = [current_price * 0.95, current_price, current_price * 1.05]
    expiries = [30/365, 60/365]  # 1-month, 2-month

    option_data = {}

    for strike in strikes:
        for expiry in expiries:
            for option_type in ['call', 'put']:
                # Create model parameters
                params = BSParameters(
                    S0=current_price,
                    K=strike,
                    T=expiry,
                    r=r,
                    sigma=realized_vol,
                    q=0.0
                )

                # Instantiate model
                bs_model = BlackScholesModel(params)

                # Price option
                if option_type == 'call':
                    price = bs_model.call_price()
                else:
                    price = bs_model.put_price()

                # Compute Greeks
                greeks = {
                    'delta': bs_model.delta(option_type),
                    'gamma': bs_model.gamma(),
                    'vega': bs_model.vega(),
                    'theta': bs_model.theta(option_type),
                    'rho': bs_model.rho(option_type)
                }

                option_id = f"{option_type}_{strike:.0f}_{int(expiry*365)}d"
                option_data[option_id] = {
                    'price': price,
                    'strike': strike,
                    'expiry': expiry,
                    'type': option_type,
                    'greeks': greeks
                }

    print(f"✓ Priced {len(option_data)} options")
    print(f"  Example: {list(option_data.keys())[0]}")
    sample = list(option_data.values())[0]
    print(f"    Price:  ${sample['price']:.2f}")
    print(f"    Delta:  {sample['greeks']['delta']:.4f}")
    print(f"    Vega:   {sample['greeks']['vega']:.2f}")

    # ========================================================================
    # LAYER 4: PORTFOLIO OPTIMIZATION (CVaR + Greeks)
    # ========================================================================
    print("\n[LAYER 4: PORTFOLIO OPTIMIZATION]")
    print("-" * 80)

    # Forecast returns for each option (demo with attractive returns to showcase optimizer)
    forecasted_returns = {}
    for opt_id, opt_data in option_data.items():
        # Heuristic based on moneyness and time value
        moneyness = current_price / opt_data['strike']
        time_factor = 1.0 - opt_data['expiry']  # Favor shorter-dated

        if opt_data['type'] == 'call':
            # Calls benefit from upward movement - enhanced for demo
            exp_ret = 0.15 + max(0, (moneyness - 1)) * 0.5 + time_factor * 0.05
        else:
            # Puts benefit from downward movement - enhanced for demo
            exp_ret = 0.10 + max(0, (1 - moneyness)) * 0.5 + time_factor * 0.05

        forecasted_returns[opt_id] = exp_ret

    # Initialize optimizer with no delta neutrality for demo
    optimizer = PortfolioOptimizer(
        risk_aversion=0.1,  # Very low risk aversion
        delta_neutrality_weight=0.0,  # No delta neutrality requirement
        confidence_level=0.95,
        max_delta=500.0,
        max_vega=3000.0,
        max_gamma=50.0
    )

    # Create reasonable return covariance matrix (diagonal with moderate variance)
    n_opts = len(option_data)
    return_covariance = np.eye(n_opts) * 0.04  # 20% return volatility

    # Optimize portfolio
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
    print(f"\n  Expected Performance:")
    print(f"    E[Return]:  {opt_result.expected_return:.4f}")
    print(f"    Sharpe:     {opt_result.expected_sharpe:.3f}")
    print(f"    CVaR_95:    ${opt_result.expected_cvar:,.2f}")
    print(f"    VaR_95:     ${opt_result.var_95:,.2f}")

    print(f"\n  Top 3 Positions:")
    sorted_positions = sorted(opt_result.positions.items(),
                             key=lambda x: abs(x[1]), reverse=True)[:3]
    for opt_id, qty in sorted_positions:
        print(f"    {opt_id}: {qty:+.2f} contracts")

    # ========================================================================
    # LAYER 5: RISK CONTROL
    # ========================================================================
    print("\n[LAYER 5: RISK CONTROL]")
    print("-" * 80)

    risk_limits = BitcoinRiskLimits(
        base_max_var=100000.0,
        base_max_cvar=150000.0,
        base_max_vega=3000.0,
        base_max_delta=500.0
    )

    risk_controller = BitcoinRiskController(
        risk_limits=risk_limits,
        portfolio_value=1000000.0
    )

    # Check each proposed position
    total_approved = 0
    total_rejected = 0

    for opt_id, qty in opt_result.positions.items():
        opt_info = option_data[opt_id]

        order = OrderProposal(
            symbol='BTC',
            option_type=opt_info['type'],
            strike=opt_info['strike'],
            expiry=opt_info['expiry'],
            quantity=int(abs(qty)),
            direction='buy' if qty > 0 else 'sell',
            underlying_price=current_price,
            volatility=realized_vol,
            risk_free_rate=r
        )

        risk_check = risk_controller.check_order(order)

        if risk_check.status.value == 'approved':
            total_approved += 1
        else:
            total_rejected += 1

    print(f"✓ Risk Check Complete")
    print(f"  Approved: {total_approved}/{len(opt_result.positions)}")
    print(f"  Rejected: {total_rejected}/{len(opt_result.positions)}")

    # ========================================================================
    # LAYER 6: EXECUTION (Simulated)
    # ========================================================================
    print("\n[LAYER 6: EXECUTION]")
    print("-" * 80)

    # Simulate execution
    executed_positions = {}
    total_cost = 0.0

    for opt_id, qty in opt_result.positions.items():
        # Simple execution: assume filled at mid price
        fill_price = option_data[opt_id]['price']
        cost = abs(qty) * fill_price

        executed_positions[opt_id] = qty
        total_cost += cost

    print(f"✓ Execution Complete")
    print(f"  Executed: {len(executed_positions)} positions")
    print(f"  Total Cost: ${total_cost:,.2f}")

    # ========================================================================
    # LAYER 7: MONITORING
    # ========================================================================
    print("\n[LAYER 7: REAL-TIME MONITORING]")
    print("-" * 80)

    monitor = RealTimeMonitor(initial_capital=1000000.0)

    # Update monitor with current state
    snapshot = monitor.update(
        positions=executed_positions,
        option_prices=option_data,
        greeks=opt_result.portfolio_greeks.to_dict(),
        forecast_time=10.5,
        pricing_time=25.3,
        risk_check_time=5.2
    )

    print(f"✓ Monitoring Update Complete")
    print(f"\n{monitor.generate_report()}")

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "="*80)
    print(" PIPELINE SUMMARY")
    print("="*80)

    print(f"\n✓ Complete pipeline executed successfully!")
    print(f"\nKey Metrics:")
    print(f"  • Forecasted Vol:      {realized_vol:.2%}")
    print(f"  • Options Priced:      {len(option_data)}")
    print(f"  • Portfolio Delta:     {opt_result.portfolio_greeks.delta:,.2f}")
    print(f"  • Portfolio Vega:      {opt_result.portfolio_greeks.vega:,.2f}")
    print(f"  • Expected Sharpe:     {opt_result.expected_sharpe:.3f}")
    print(f"  • Expected CVaR:       ${opt_result.expected_cvar:,.2f}")

    # Handle case when no positions
    if len(opt_result.positions) > 0:
        approval_rate = total_approved / len(opt_result.positions)
        print(f"  • Risk Approval Rate:  {total_approved}/{len(opt_result.positions)} ({approval_rate:.1%})")
    else:
        print(f"  • Risk Approval Rate:  N/A (no positions)")

    print(f"  • Total Investment:    ${total_cost:,.2f}")

    print(f"\n✓ Monitoring system active - tracking {len(executed_positions)} positions")

    print("\n" + "="*80)
    print(" DEMO COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()
