"""
Bitcoin Options Trading Evaluation - Quick Start Demo

This script demonstrates how to:
1. Fetch Bitcoin options data (real or synthetic)
2. Train LSTM + GARCH forecasting models
3. Run multi-agent trading simulation
4. Backtest trading strategies
5. Calculate profitability metrics

Usage:
    python examples/bitcoin_trading_demo.py
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from evaluation_modules.bitcoin_trading_evaluation import (
    BitcoinTradingEvaluator,
    BitcoinTradingConfig
)
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')


def demo_synthetic_data():
    """
    Demo 1: Quick evaluation using synthetic data
    Good for testing without API access
    """
    print("\n" + "=" * 80)
    print("DEMO 1: Synthetic Bitcoin Data Evaluation")
    print("=" * 80)

    config = BitcoinTradingConfig(
        use_real_data=False,  # Use synthetic data
        use_behavior_features=True,  # Use enhanced LSTM
        lookback_periods=60,
        forecast_horizon=5,
        num_epochs=30,  # Reduced for demo
        num_periods=100,  # Reduced for demo
        initial_capital=100000.0,
        transaction_cost=0.0005,
        edge_threshold=0.02,
        use_multiagent=True
    )

    evaluator = BitcoinTradingEvaluator(config)
    results = evaluator.run_full_evaluation()

    print("\n" + "=" * 80)
    print("DEMO 1 COMPLETE")
    print("=" * 80)

    return results


def demo_real_data():
    """
    Demo 2: Evaluation using real Deribit API data
    Requires internet connection
    """
    print("\n" + "=" * 80)
    print("DEMO 2: Real Bitcoin Data from Deribit")
    print("=" * 80)

    config = BitcoinTradingConfig(
        use_real_data=True,  # Use Deribit API
        use_behavior_features=True,
        lookback_periods=60,
        forecast_horizon=5,
        num_epochs=30,
        num_periods=100,
        initial_capital=100000.0,
        use_multiagent=True
    )

    evaluator = BitcoinTradingEvaluator(config)

    try:
        results = evaluator.run_full_evaluation()

        print("\n" + "=" * 80)
        print("DEMO 2 COMPLETE")
        print("=" * 80)

        return results

    except Exception as e:
        print(f"\n⚠ Error fetching real data: {e}")
        print("→ Falling back to synthetic data demo")
        return demo_synthetic_data()


def demo_comparison():
    """
    Demo 3: Compare Pure LSTM vs Behavior-Enhanced LSTM
    Shows the benefit of market microstructure features
    """
    print("\n" + "=" * 80)
    print("DEMO 3: Pure LSTM vs Behavior-Enhanced LSTM Comparison")
    print("=" * 80)

    results_comparison = {}

    # Test 1: Pure LSTM
    print("\n[Test 1/2] Running with Pure LSTM...")
    config_pure = BitcoinTradingConfig(
        use_real_data=False,
        use_behavior_features=False,  # Pure LSTM
        num_epochs=30,
        num_periods=100,
        use_multiagent=True
    )

    evaluator_pure = BitcoinTradingEvaluator(config_pure)
    results_pure = evaluator_pure.run_full_evaluation()
    results_comparison['pure_lstm'] = results_pure

    # Test 2: Behavior-Enhanced LSTM
    print("\n[Test 2/2] Running with Behavior-Enhanced LSTM...")
    config_behavior = BitcoinTradingConfig(
        use_real_data=False,
        use_behavior_features=True,  # Behavior-enhanced
        num_epochs=30,
        num_periods=100,
        use_multiagent=True
    )

    evaluator_behavior = BitcoinTradingEvaluator(config_behavior)
    results_behavior = evaluator_behavior.run_full_evaluation()
    results_comparison['behavior_lstm'] = results_behavior

    # Compare results
    print("\n" + "=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)

    if 'mae' in results_pure['forecast_results'] and 'mae' in results_behavior['forecast_results']:
        mae_pure = results_pure['forecast_results']['mae']
        mae_behavior = results_behavior['forecast_results']['mae']
        improvement = ((mae_pure - mae_behavior) / mae_pure) * 100

        print(f"\nForecasting Accuracy:")
        print(f"  Pure LSTM MAE: ${mae_pure:.2f}")
        print(f"  Behavior-Enhanced MAE: ${mae_behavior:.2f}")
        print(f"  → Improvement: {improvement:+.2f}%")

    if 'agent_pnl' in results_pure['trading_results'] and 'agent_pnl' in results_behavior['trading_results']:
        total_pnl_pure = sum(pnl['pnl'] for pnl in results_pure['trading_results']['agent_pnl'].values())
        total_pnl_behavior = sum(pnl['pnl'] for pnl in results_behavior['trading_results']['agent_pnl'].values())

        print(f"\nTrading Performance:")
        print(f"  Pure LSTM P&L: ${total_pnl_pure:,.2f}")
        print(f"  Behavior-Enhanced P&L: ${total_pnl_behavior:,.2f}")
        print(f"  → Difference: ${total_pnl_behavior - total_pnl_pure:+,.2f}")

    print("\n" + "=" * 80)
    print("DEMO 3 COMPLETE")
    print("=" * 80)

    return results_comparison


def demo_profitability_analysis():
    """
    Demo 4: Focused profitability analysis
    Shows detailed P&L breakdown and risk metrics
    """
    print("\n" + "=" * 80)
    print("DEMO 4: Profitability & Risk Analysis")
    print("=" * 80)

    config = BitcoinTradingConfig(
        use_real_data=False,
        use_behavior_features=True,
        num_epochs=50,
        num_periods=200,  # Longer simulation
        initial_capital=100000.0,
        transaction_cost=0.0005,
        edge_threshold=0.02,
        use_multiagent=True
    )

    evaluator = BitcoinTradingEvaluator(config)
    results = evaluator.run_full_evaluation()

    # Detailed profitability analysis
    print("\n" + "=" * 80)
    print("DETAILED PROFITABILITY ANALYSIS")
    print("=" * 80)

    if 'agent_pnl' in results['trading_results']:
        print("\n[Multi-Agent Trading Results]")
        print("-" * 80)

        total_pnl = 0
        for agent_id, pnl_data in results['trading_results']['agent_pnl'].items():
            print(f"\n{agent_id}:")
            print(f"  Initial Capital: ${config.initial_capital:,.2f}")
            print(f"  Final Capital:   ${pnl_data['final_capital']:,.2f}")
            print(f"  P&L:            ${pnl_data['pnl']:+,.2f} ({pnl_data['pnl_pct']:+.2f}%)")

            # Annualized return (assuming 200 periods = 1 year for crypto)
            annualized_return = pnl_data['pnl_pct']
            print(f"  Annualized:      {annualized_return:+.2f}%")

            total_pnl += pnl_data['pnl']

        print(f"\n{'Total Portfolio P&L:':<20} ${total_pnl:+,.2f}")

        # Calculate overall return
        total_initial = config.initial_capital * 2.0  # MM + IT + ARB
        overall_return = (total_pnl / total_initial) * 100
        print(f"{'Overall Return:':<20} {overall_return:+.2f}%")

    if 'backtest_results' in results and 'comparison_df' in results['backtest_results']:
        print("\n[Model Comparison Results]")
        print("-" * 80)

        df = results['backtest_results']['comparison_df']

        print("\nRanking by Total P&L:")
        ranked = df.sort_values('Total_PnL', ascending=False)
        for idx, row in ranked.iterrows():
            print(f"  {idx+1}. {row['Model']:<20} P&L: ${row['Total_PnL']:+,.2f}  "
                  f"Sharpe: {row['Sharpe_Ratio']:.2f}  "
                  f"Win Rate: {row['Win_Rate']:.1f}%")

        print("\nRisk Metrics:")
        for idx, row in df.iterrows():
            print(f"  {row['Model']}:")
            print(f"    • Max Drawdown: ${row['Max_Drawdown']:,.2f}")
            print(f"    • VaR (95%):    ${row['VaR_95']:,.2f}")
            print(f"    • Sharpe Ratio: {row['Sharpe_Ratio']:.2f}")

    print("\n" + "=" * 80)
    print("DEMO 4 COMPLETE")
    print("=" * 80)

    return results


def main():
    """
    Main demo runner
    """
    print("\n" + "█" * 80)
    print("BITCOIN OPTIONS TRADING EVALUATION - DEMO SUITE")
    print("█" * 80)

    print("\nAvailable Demos:")
    print("  1. Quick Synthetic Data Demo (fastest)")
    print("  2. Real Deribit API Data Demo (requires internet)")
    print("  3. Model Comparison Demo (Pure vs Behavior-Enhanced)")
    print("  4. Detailed Profitability Analysis")
    print("  5. Run All Demos")

    choice = input("\nSelect demo (1-5) or press Enter for default [1]: ").strip()

    if not choice:
        choice = "1"

    if choice == "1":
        results = demo_synthetic_data()

    elif choice == "2":
        results = demo_real_data()

    elif choice == "3":
        results = demo_comparison()

    elif choice == "4":
        results = demo_profitability_analysis()

    elif choice == "5":
        print("\nRunning all demos...")
        demo_synthetic_data()
        demo_comparison()
        demo_profitability_analysis()
        # Uncomment if you want to test real API:
        # demo_real_data()

    else:
        print(f"\n⚠ Invalid choice '{choice}'. Running default demo.")
        results = demo_synthetic_data()

    print("\n" + "█" * 80)
    print("ALL DEMOS COMPLETE")
    print("█" * 80)
    print("\n✓ Check the generated report files for detailed results")
    print("✓ Modify BitcoinTradingConfig parameters to customize evaluation")
    print("\n")


if __name__ == "__main__":
    main()
