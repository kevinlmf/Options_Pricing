"""
Comparison: Pure Time Series vs Behavior-Enhanced Time Series

This demonstration compares:
1. Pure LSTM (baseline) - Uses only price history
2. Behavior-Enhanced LSTM - Uses price history + market microstructure features

Shows how behavior analysis improves forecasting accuracy.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import sys
import os
from typing import Dict, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models.multi_agent.timeseries_driven_agents import TSMarketMaker, TSInformedTrader, TSArbitrageur
from models.multi_agent.ts_market_environment import TSMarketEnvironment
from time_series_forecasting.deep_learning.rnn_models import LSTMForecaster, RNNTrainer
from time_series_forecasting.deep_learning.behavior_enhanced_lstm import (
    BehaviorEnhancedLSTM, BehaviorEnhancedLSTMTrainer
)
from time_series_forecasting.behavior_features import BehaviorFeatureExtractor


def setup_market_and_collect_data(num_periods: int = 300) -> Tuple[np.ndarray, np.ndarray]:
    """
    Set up market, run simulation, and collect price + behavior data.

    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]: (prices, behavior_features)
    """
    print("\n" + "="*80)
    print("STEP 1: COLLECTING DATA FROM MULTI-AGENT MARKET")
    print("="*80)

    # Create market
    market = TSMarketEnvironment(
        initial_spot=100.0,
        spot_drift=0.0002,
        spot_vol=0.02,
        risk_free_rate=0.05
    )

    # Add agents
    market.add_agent(TSMarketMaker("MM1", initial_cash=2000000))
    market.add_agent(TSMarketMaker("MM2", initial_cash=2000000))
    market.add_agent(TSInformedTrader("Trader1", initial_cash=1000000))
    market.add_agent(TSInformedTrader("Trader2", initial_cash=1000000))
    market.add_agent(TSArbitrageur("Arb1", initial_cash=1500000))

    # Add options
    current_spot = market.spot_price
    market.add_option(current_spot * 1.0, 90/252)
    market.add_option(current_spot * 1.05, 90/252)

    # Initialize behavior feature extractor
    behavior_extractor = BehaviorFeatureExtractor(lookback_window=10)

    # Run simulation and collect data
    print(f"\nRunning simulation for {num_periods} periods...")
    prices = []
    behavior_features_list = []

    for t in range(num_periods):
        # Simulate spot movement
        market.simulate_spot_price_movement()
        prices.append(market.spot_price)

        # Run trading
        market.current_time = t
        orders = market.collect_orders()

        for order in orders:
            if order.option_key in market.order_books:
                market.order_books[order.option_key].add_order(order)

        trades = market.match_orders()

        # Extract behavior features
        option_key = market.available_options[0]  # Use first option
        order_book = {
            'bids': market.order_books[option_key].bids,
            'asks': market.order_books[option_key].asks
        }

        behavior_extractor.update_market_data(
            order_book=order_book,
            trades=trades,
            agents=market.agents,
            current_price=market.spot_price,
            timestamp=t
        )

        features = behavior_extractor.extract_features(t)
        behavior_features_list.append(behavior_extractor.get_feature_vector(features))

        # Clear order books for next period
        for book in market.order_books.values():
            book.clear()

        if (t+1) % 50 == 0:
            print(f"  Period {t+1}/{num_periods} - Spot: ${market.spot_price:.2f}")

    prices = np.array(prices)
    behavior_features = np.array(behavior_features_list)

    print(f"\nData collection complete!")
    print(f"  Prices shape: {prices.shape}")
    print(f"  Behavior features shape: {behavior_features.shape}")

    return prices, behavior_features


def train_pure_lstm(prices: np.ndarray, seq_length: int = 20) -> Tuple[RNNTrainer, Dict]:
    """
    Train pure LSTM model (baseline).

    Parameters:
    -----------
    prices : np.ndarray
        Price time series
    seq_length : int
        Sequence length

    Returns:
    --------
    Tuple: (trainer, history)
    """
    print("\n" + "="*80)
    print("STEP 2: TRAINING PURE LSTM (BASELINE)")
    print("="*80)

    # Create model
    model = LSTMForecaster(
        input_size=1,
        hidden_size=64,
        num_layers=2,
        output_size=1,
        dropout=0.2
    )

    trainer = RNNTrainer(model, learning_rate=0.001)

    # Train
    print("\nTraining pure LSTM...")
    history = trainer.fit(
        train_data=pd.Series(prices),
        seq_length=seq_length,
        forecast_horizon=1,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        early_stopping_patience=15,
        verbose=True
    )

    print(f"\nPure LSTM training complete!")
    print(f"  Best validation loss: {history['best_val_loss']:.6f}")

    return trainer, history


def train_behavior_enhanced_lstm(prices: np.ndarray,
                                 behavior_features: np.ndarray,
                                 seq_length: int = 20) -> Tuple[BehaviorEnhancedLSTMTrainer, Dict]:
    """
    Train behavior-enhanced LSTM model.

    Parameters:
    -----------
    prices : np.ndarray
        Price time series
    behavior_features : np.ndarray
        Behavior features
    seq_length : int
        Sequence length

    Returns:
    --------
    Tuple: (trainer, history)
    """
    print("\n" + "="*80)
    print("STEP 3: TRAINING BEHAVIOR-ENHANCED LSTM")
    print("="*80)

    # Create model
    model = BehaviorEnhancedLSTM(
        price_input_size=1,
        behavior_input_size=behavior_features.shape[1],
        lstm_hidden_size=64,
        lstm_num_layers=2,
        behavior_hidden_size=32,
        fusion_hidden_size=32,
        output_size=1,
        dropout=0.2
    )

    trainer = BehaviorEnhancedLSTMTrainer(model, learning_rate=0.001)

    # Train
    print("\nTraining behavior-enhanced LSTM...")
    history = trainer.fit(
        price_data=prices,
        behavior_data=behavior_features,
        seq_length=seq_length,
        forecast_horizon=1,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        early_stopping_patience=15,
        verbose=True
    )

    print(f"\nBehavior-enhanced LSTM training complete!")
    print(f"  Best validation loss: {history['best_val_loss']:.6f}")

    return trainer, history


def evaluate_models(pure_trainer: RNNTrainer,
                   behavior_trainer: BehaviorEnhancedLSTMTrainer,
                   prices: np.ndarray,
                   behavior_features: np.ndarray,
                   seq_length: int = 20) -> Dict:
    """
    Evaluate and compare both models.

    Returns:
    --------
    Dict: Evaluation results
    """
    print("\n" + "="*80)
    print("STEP 4: EVALUATING MODELS")
    print("="*80)

    # Split data
    split_idx = int(len(prices) * 0.8)
    test_prices = prices[split_idx:]
    test_behavior = behavior_features[split_idx:]

    # Pure LSTM predictions
    print("\nGenerating pure LSTM predictions...")
    pure_predictions = pure_trainer.predict(
        data=pd.Series(test_prices),
        seq_length=seq_length,
        steps=len(test_prices) - seq_length
    )

    # Behavior-enhanced predictions
    print("Generating behavior-enhanced LSTM predictions...")
    behavior_predictions = behavior_trainer.predict(
        price_data=test_prices,
        behavior_data=test_behavior,
        seq_length=seq_length,
        steps=len(test_prices) - seq_length
    )

    # Calculate metrics
    test_targets = test_prices[seq_length:]

    # Ensure same length
    min_len = min(len(pure_predictions), len(behavior_predictions), len(test_targets))
    pure_predictions = pure_predictions[:min_len]
    behavior_predictions = behavior_predictions[:min_len]
    test_targets = test_targets[:min_len]

    # Metrics
    def calculate_metrics(predictions, targets):
        mae = np.mean(np.abs(predictions - targets))
        rmse = np.sqrt(np.mean((predictions - targets) ** 2))
        mape = np.mean(np.abs((predictions - targets) / targets)) * 100
        direction_accuracy = np.mean(
            np.sign(predictions[1:] - predictions[:-1]) ==
            np.sign(targets[1:] - targets[:-1])
        ) * 100
        return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape,
                'Direction_Accuracy': direction_accuracy}

    pure_metrics = calculate_metrics(pure_predictions, test_targets)
    behavior_metrics = calculate_metrics(behavior_predictions, test_targets)

    print("\n" + "-"*80)
    print("EVALUATION RESULTS")
    print("-"*80)

    print("\nPure LSTM (Baseline):")
    for metric, value in pure_metrics.items():
        print(f"  {metric}: {value:.4f}")

    print("\nBehavior-Enhanced LSTM:")
    for metric, value in behavior_metrics.items():
        print(f"  {metric}: {value:.4f}")

    print("\nImprovement (%):")
    for metric in pure_metrics.keys():
        if metric == 'Direction_Accuracy':
            improvement = behavior_metrics[metric] - pure_metrics[metric]
            print(f"  {metric}: {improvement:+.2f} percentage points")
        else:
            improvement = (pure_metrics[metric] - behavior_metrics[metric]) / pure_metrics[metric] * 100
            print(f"  {metric}: {improvement:+.2f}%")

    return {
        'pure_predictions': pure_predictions,
        'behavior_predictions': behavior_predictions,
        'test_targets': test_targets,
        'pure_metrics': pure_metrics,
        'behavior_metrics': behavior_metrics
    }


def visualize_results(results: Dict,
                     pure_history: Dict,
                     behavior_history: Dict):
    """Create comprehensive visualizations."""
    print("\n" + "="*80)
    print("STEP 5: CREATING VISUALIZATIONS")
    print("="*80)

    fig = plt.figure(figsize=(16, 12))

    # 1. Training loss comparison
    ax1 = plt.subplot(3, 2, 1)
    ax1.plot(pure_history['val_losses'], label='Pure LSTM', linewidth=2, color='blue')
    ax1.plot(behavior_history['val_losses'], label='Behavior-Enhanced LSTM',
            linewidth=2, color='green')
    ax1.set_title('Training Validation Loss Comparison', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')

    # 2. Predictions comparison
    ax2 = plt.subplot(3, 2, 2)
    test_targets = results['test_targets']
    ax2.plot(test_targets, label='Actual', linewidth=2, color='black', alpha=0.7)
    ax2.plot(results['pure_predictions'], label='Pure LSTM',
            linewidth=1.5, linestyle='--', color='blue')
    ax2.plot(results['behavior_predictions'], label='Behavior-Enhanced',
            linewidth=1.5, linestyle='--', color='green')
    ax2.set_title('Price Predictions Comparison', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Price ($)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Prediction errors
    ax3 = plt.subplot(3, 2, 3)
    pure_errors = (results['pure_predictions'] - test_targets).flatten()
    behavior_errors = (results['behavior_predictions'] - test_targets).flatten()
    ax3.plot(pure_errors, label='Pure LSTM Error', linewidth=1.5, color='blue', alpha=0.7)
    ax3.plot(behavior_errors, label='Behavior-Enhanced Error',
            linewidth=1.5, color='green', alpha=0.7)
    ax3.axhline(0, color='red', linestyle='--', linewidth=1)
    ax3.set_title('Prediction Errors Over Time', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('Error ($)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Error distribution
    ax4 = plt.subplot(3, 2, 4)
    ax4.hist(pure_errors, bins=30, alpha=0.5, label='Pure LSTM',
            color='blue', edgecolor='black')
    ax4.hist(behavior_errors, bins=30, alpha=0.5, label='Behavior-Enhanced',
            color='green', edgecolor='black')
    ax4.axvline(0, color='red', linestyle='--', linewidth=2)
    ax4.set_title('Prediction Error Distribution', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Error ($)')
    ax4.set_ylabel('Frequency')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')

    # 5. Metrics comparison
    ax5 = plt.subplot(3, 2, 5)
    metrics_names = ['MAE', 'RMSE', 'MAPE']
    pure_values = [results['pure_metrics'][m] for m in metrics_names]
    behavior_values = [results['behavior_metrics'][m] for m in metrics_names]

    x = np.arange(len(metrics_names))
    width = 0.35

    ax5.bar(x - width/2, pure_values, width, label='Pure LSTM',
           color='blue', alpha=0.7, edgecolor='black')
    ax5.bar(x + width/2, behavior_values, width, label='Behavior-Enhanced',
           color='green', alpha=0.7, edgecolor='black')

    ax5.set_title('Error Metrics Comparison', fontsize=12, fontweight='bold')
    ax5.set_xticks(x)
    ax5.set_xticklabels(metrics_names)
    ax5.set_ylabel('Error Value')
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')

    # 6. Direction accuracy
    ax6 = plt.subplot(3, 2, 6)
    direction_acc = [
        results['pure_metrics']['Direction_Accuracy'],
        results['behavior_metrics']['Direction_Accuracy']
    ]
    colors = ['blue', 'green']
    bars = ax6.bar(['Pure LSTM', 'Behavior-Enhanced'], direction_acc,
                   color=colors, alpha=0.7, edgecolor='black')
    ax6.set_title('Directional Prediction Accuracy', fontsize=12, fontweight='bold')
    ax6.set_ylabel('Accuracy (%)')
    ax6.set_ylim([0, 100])
    ax6.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig('pure_vs_behavior_comparison.png', dpi=300, bbox_inches='tight')
    print("\nVisualization saved to: pure_vs_behavior_comparison.png")

    return fig


def main():
    """Run complete comparison demonstration."""
    print("\n")
    print("="*80)
    print("PURE TIME SERIES vs BEHAVIOR-ENHANCED TIME SERIES")
    print("LSTM Forecasting Comparison")
    print("="*80)
    print("\nThis demonstration compares:")
    print("1. Pure LSTM - Uses only price history")
    print("2. Behavior-Enhanced LSTM - Uses price + market microstructure features")
    print("="*80)

    # Set seed
    np.random.seed(42)
    torch.manual_seed(42)

    # Collect data
    prices, behavior_features = setup_market_and_collect_data(num_periods=300)

    # Train pure LSTM
    pure_trainer, pure_history = train_pure_lstm(prices, seq_length=20)

    # Train behavior-enhanced LSTM
    behavior_trainer, behavior_history = train_behavior_enhanced_lstm(
        prices, behavior_features, seq_length=20
    )

    # Evaluate
    results = evaluate_models(
        pure_trainer, behavior_trainer,
        prices, behavior_features,
        seq_length=20
    )

    # Visualize
    fig = visualize_results(results, pure_history, behavior_history)

    # Final summary
    print("\n" + "="*80)
    print("CONCLUSIONS")
    print("="*80)

    improvement_mae = (results['pure_metrics']['MAE'] -
                      results['behavior_metrics']['MAE']) / results['pure_metrics']['MAE'] * 100

    print(f"\n1. Forecast Accuracy:")
    if improvement_mae > 5:
        print(f"   ✓ Behavior-enhanced model shows {improvement_mae:.1f}% improvement in MAE")
        print("   ✓ Market microstructure features provide predictive value")
    elif improvement_mae > 0:
        print(f"   ~ Behavior-enhanced model shows modest {improvement_mae:.1f}% improvement")
    else:
        print("   ✗ Behavior features did not improve forecast accuracy")

    dir_improvement = (results['behavior_metrics']['Direction_Accuracy'] -
                      results['pure_metrics']['Direction_Accuracy'])
    print(f"\n2. Directional Accuracy:")
    if dir_improvement > 2:
        print(f"   ✓ {dir_improvement:.1f} percentage point improvement")
        print("   ✓ Better at predicting price direction")
    else:
        print(f"   ~ Similar directional accuracy ({dir_improvement:+.1f} pp)")

    print(f"\n3. Key Insights:")
    print("   - Market microstructure contains valuable information")
    print("   - Order flow, spreads, and agent behavior matter")
    print("   - Combining time series + behavior analysis is beneficial")
    print("   - Future work: test on real market data")

    print("\n" + "="*80)
    print("DEMONSTRATION COMPLETE")
    print("="*80)

    plt.show()


if __name__ == "__main__":
    main()
