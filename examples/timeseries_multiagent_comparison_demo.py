"""
Time Series Driven Multi-Agent Option Pricing vs Traditional Methods

Complete demonstration comparing:
1. Multi-Agent pricing (agents use LSTM + GARCH forecasts)
2. Traditional Black-Scholes pricing

This shows how market prices emerge from agent interactions based on
time series forecasts, and how they compare to analytical solutions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models.multi_agent.timeseries_driven_agents import (
    TSMarketMaker, TSInformedTrader, TSArbitrageur
)
from models.multi_agent.ts_market_environment import TSMarketEnvironment


def setup_market() -> TSMarketEnvironment:
    """Set up market environment with agents and options."""
    print("\n" + "="*80)
    print("SETTING UP MARKET ENVIRONMENT")
    print("="*80)

    # Create market
    market = TSMarketEnvironment(
        initial_spot=100.0,
        spot_drift=0.0001,
        spot_vol=0.02,
        risk_free_rate=0.05
    )
    print(f"\nMarket initialized:")
    print(f"  Initial spot: ${market.initial_spot}")
    print(f"  Drift: {market.spot_drift*100:.3f}% daily")
    print(f"  Volatility: {market.spot_vol*100:.1f}% daily")
    print(f"  Risk-free rate: {market.risk_free_rate*100:.1f}%")

    # Create agents
    print(f"\nCreating agents...")

    # Market Makers (provide liquidity)
    mm1 = TSMarketMaker("MM1", initial_cash=2000000, spread_multiplier=0.8)
    mm2 = TSMarketMaker("MM2", initial_cash=2000000, spread_multiplier=1.2)
    market.add_agent(mm1)
    market.add_agent(mm2)

    # Informed Traders (directional based on forecasts)
    trader1 = TSInformedTrader("Trader1", initial_cash=1000000, conviction_threshold=0.02)
    trader2 = TSInformedTrader("Trader2", initial_cash=1000000, conviction_threshold=0.03)
    market.add_agent(trader1)
    market.add_agent(trader2)

    # Arbitrageurs (exploit mispricings)
    arb1 = TSArbitrageur("Arb1", initial_cash=1500000, arbitrage_threshold=0.03)
    market.add_agent(arb1)

    # Add options
    print(f"\nCreating option contracts...")
    current_spot = market.spot_price

    # Different strikes (OTM, ATM, ITM)
    strikes = [
        current_spot * 0.95,  # ITM
        current_spot * 1.00,  # ATM
        current_spot * 1.05   # OTM
    ]

    # Different expiries
    expiries = [
        30/252,   # 1 month
        90/252,   # 3 months
    ]

    for strike in strikes:
        for expiry in expiries:
            market.add_option(strike, expiry)

    print(f"\nMarket setup complete!")
    print(f"  Total agents: {len(market.agents)}")
    print(f"  Total options: {len(market.available_options)}")

    return market


def run_simulation(market: TSMarketEnvironment, num_periods: int = 200) -> pd.DataFrame:
    """Run the multi-agent simulation."""
    print("\n" + "="*80)
    print("RUNNING MULTI-AGENT SIMULATION")
    print("="*80)

    results = market.simulate(num_periods=num_periods, verbose=True)

    return results


def analyze_results(results: pd.DataFrame, market: TSMarketEnvironment):
    """Analyze and compare results."""
    print("\n" + "="*80)
    print("ANALYSIS: MULTI-AGENT VS BLACK-SCHOLES")
    print("="*80)

    # Summary statistics
    stats = market.get_summary_statistics(results)

    print(f"\nSimulation Statistics:")
    print(f"  Total periods: {stats['total_periods']}")
    print(f"  Total trades: {stats['total_trades']}")
    print(f"  Avg trades/period: {stats['avg_trades_per_period']:.2f}")

    if 'mean_price_diff' in stats:
        print(f"\nPrice Comparison (Multi-Agent vs Black-Scholes):")
        print(f"  Mean difference: ${stats['mean_price_diff']:.4f}")
        print(f"  Std difference: ${stats['std_price_diff']:.4f}")
        print(f"  MAE: ${stats['mae']:.4f}")
        print(f"  RMSE: ${stats['rmse']:.4f}")

    if 'mean_price_diff_pct' in stats:
        print(f"  Mean difference %: {stats['mean_price_diff_pct']:.2f}%")
        print(f"  Std difference %: {stats['std_price_diff_pct']:.2f}%")

    print(f"\nAgent Performance:")
    for agent_id, perf in stats['agent_performance'].items():
        print(f"  {agent_id} ({perf['role']}):")
        print(f"    Final P&L: ${perf['final_pnl']:,.2f}")
        print(f"    Positions: {perf['num_positions']}")
        print(f"    Cash: ${perf['cash']:,.2f}")

    return stats


def create_visualizations(results: pd.DataFrame, market: TSMarketEnvironment):
    """Create comparison visualizations."""
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)

    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))

    # 1. Spot price evolution
    ax1 = plt.subplot(3, 2, 1)
    ax1.plot(market.spot_history, linewidth=2, color='darkblue')
    ax1.set_title('Underlying Spot Price Evolution', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Time Period')
    ax1.set_ylabel('Spot Price ($)')
    ax1.grid(True, alpha=0.3)

    # 2. Price comparison for ATM option
    ax2 = plt.subplot(3, 2, 2)
    atm_strike = min(market.available_options, key=lambda x: abs(x[0] - market.initial_spot))
    atm_data = results[
        (results['strike'] == atm_strike[0]) &
        (results['expiry'] == atm_strike[1])
    ].copy()

    if len(atm_data) > 0:
        ax2.plot(atm_data['timestamp'], atm_data['market_price'],
                label='Multi-Agent Price', linewidth=2, color='green')
        ax2.plot(atm_data['timestamp'], atm_data['bs_price'],
                label='Black-Scholes Price', linewidth=2, color='red', linestyle='--')
        ax2.set_title(f'ATM Option Price Comparison (K={atm_strike[0]:.2f})',
                     fontsize=12, fontweight='bold')
        ax2.set_xlabel('Time Period')
        ax2.set_ylabel('Option Price ($)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    # 3. Price difference distribution
    ax3 = plt.subplot(3, 2, 3)
    valid_diffs = results['price_diff'].dropna()
    if len(valid_diffs) > 0:
        ax3.hist(valid_diffs, bins=30, edgecolor='black', alpha=0.7, color='purple')
        ax3.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Difference')
        ax3.set_title('Price Difference Distribution (Multi-Agent - BS)',
                     fontsize=12, fontweight='bold')
        ax3.set_xlabel('Price Difference ($)')
        ax3.set_ylabel('Frequency')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

    # 4. Price difference over time
    ax4 = plt.subplot(3, 2, 4)
    if len(atm_data) > 0:
        ax4.plot(atm_data['timestamp'], atm_data['price_diff'],
                linewidth=2, color='orange')
        ax4.axhline(0, color='red', linestyle='--', linewidth=1)
        ax4.set_title('Price Difference Over Time (ATM Option)',
                     fontsize=12, fontweight='bold')
        ax4.set_xlabel('Time Period')
        ax4.set_ylabel('Price Difference ($)')
        ax4.grid(True, alpha=0.3)

    # 5. Agent P&L evolution
    ax5 = plt.subplot(3, 2, 5)
    for agent_id, agent in market.agents.items():
        if agent.pnl_history:
            ax5.plot(agent.pnl_history, label=f"{agent_id} ({agent.role.value})",
                    linewidth=2)
    ax5.set_title('Agent P&L Evolution', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Time Period')
    ax5.set_ylabel('P&L ($)')
    ax5.legend(loc='best', fontsize=8)
    ax5.grid(True, alpha=0.3)
    ax5.axhline(0, color='black', linestyle='-', linewidth=0.5)

    # 6. Trading activity
    ax6 = plt.subplot(3, 2, 6)
    trades_per_period = results.groupby('timestamp')['num_trades'].first()
    ax6.bar(trades_per_period.index, trades_per_period.values,
           color='steelblue', alpha=0.7)
    ax6.set_title('Trading Activity Over Time', fontsize=12, fontweight='bold')
    ax6.set_xlabel('Time Period')
    ax6.set_ylabel('Number of Trades')
    ax6.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('timeseries_multiagent_comparison.png', dpi=300, bbox_inches='tight')
    print("Visualization saved to: timeseries_multiagent_comparison.png")

    return fig


def run_comparison_by_moneyness(results: pd.DataFrame):
    """Compare pricing accuracy across different moneyness levels."""
    print("\n" + "="*80)
    print("COMPARISON BY MONEYNESS")
    print("="*80)

    # Calculate moneyness for each option
    results_with_moneyness = results.copy()
    results_with_moneyness['moneyness'] = results_with_moneyness['strike'] / results_with_moneyness['spot_price']

    # Categorize by moneyness
    results_with_moneyness['moneyness_category'] = pd.cut(
        results_with_moneyness['moneyness'],
        bins=[0, 0.97, 1.03, 2.0],
        labels=['ITM', 'ATM', 'OTM']
    )

    # Group statistics
    moneyness_stats = results_with_moneyness.groupby('moneyness_category').agg({
        'price_diff': ['mean', 'std', 'count'],
        'price_diff_pct': ['mean', 'std']
    }).round(4)

    print("\nPrice Difference by Moneyness:")
    print(moneyness_stats)

    return moneyness_stats


def main():
    """Run complete demonstration."""
    print("\n")
    print("="*80)
    print("TIME SERIES DRIVEN MULTI-AGENT OPTION PRICING")
    print("vs")
    print("TRADITIONAL BLACK-SCHOLES PRICING")
    print("="*80)
    print("\nThis demonstration shows how option prices emerge from the")
    print("interaction of agents using time series forecasts (LSTM + GARCH),")
    print("and compares these market prices with Black-Scholes analytical prices.")
    print("="*80)

    # Setup market
    market = setup_market()

    # Run simulation
    results = run_simulation(market, num_periods=200)

    # Analyze results
    stats = analyze_results(results, market)

    # Comparison by moneyness
    moneyness_stats = run_comparison_by_moneyness(results)

    # Create visualizations
    fig = create_visualizations(results, market)

    # Final summary
    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)

    print("\n1. Price Discovery:")
    print("   - Agents use LSTM for price prediction and GARCH for volatility")
    print("   - Market prices emerge from agent interactions")
    print("   - Order matching creates equilibrium prices")

    if 'mean_price_diff_pct' in stats:
        print(f"\n2. Pricing Accuracy:")
        diff_pct = abs(stats['mean_price_diff_pct'])
        if diff_pct < 5:
            print(f"   - Multi-agent prices closely match BS ({diff_pct:.2f}% difference)")
            print("   - Agents' time series forecasts lead to efficient pricing")
        elif diff_pct < 10:
            print(f"   - Moderate difference from BS ({diff_pct:.2f}%)")
            print("   - Market microstructure effects are visible")
        else:
            print(f"   - Significant divergence from BS ({diff_pct:.2f}%)")
            print("   - Agent forecasts may be capturing different information")

    print("\n3. Agent Behavior:")
    print("   - Market Makers: Provide liquidity with forecast-adjusted spreads")
    print("   - Informed Traders: Take directional positions based on predictions")
    print("   - Arbitrageurs: Exploit mispricings between market and theory")

    print("\n4. Advantages of Multi-Agent Approach:")
    print("   - Incorporates time series forecasting")
    print("   - Models market microstructure")
    print("   - Captures agent heterogeneity")
    print("   - Endogenous price formation")

    print("\n" + "="*80)
    print("DEMONSTRATION COMPLETE")
    print("="*80)

    print("\nResults saved:")
    print("  - Visualization: timeseries_multiagent_comparison.png")
    print("  - Data: Available in 'results' DataFrame")

    # Save results
    results.to_csv('timeseries_multiagent_results.csv', index=False)
    print("  - CSV: timeseries_multiagent_results.csv")

    plt.show()


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)

    main()
