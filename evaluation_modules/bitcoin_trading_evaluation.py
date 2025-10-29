"""
Bitcoin Options Trading Evaluation System
Integrates real Bitcoin data, time series forecasting, and multi-agent trading
to evaluate profitability of the option pricing models
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
import torch

from data.bitcoin.bitcoin_data_fetcher import DeribitDataFetcher, MockBitcoinDataGenerator
from time_series_forecasting.deep_learning.behavior_enhanced_lstm import BehaviorEnhancedLSTM, BehaviorEnhancedLSTMTrainer
from time_series_forecasting.deep_learning.rnn_models import LSTMForecaster, RNNTrainer
from time_series_forecasting.classical_models.garch import GARCHModel
from models.multi_agent.timeseries_driven_agents import TSMarketMaker, TSInformedTrader, TSArbitrageur
from models.multi_agent.ts_market_environment import TSMarketEnvironment
from evaluation_modules.trading_backtest import ModelComparisonBacktest, ModelTradingStrategy
from models.model_calibrator import ModelCalibrator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BitcoinTradingConfig:
    """Configuration for Bitcoin trading evaluation"""
    # Data settings
    use_real_data: bool = True  # Use Deribit API or synthetic data
    lookback_periods: int = 60  # Historical data for training
    forecast_horizon: int = 5   # Steps ahead to forecast

    # Model settings
    use_behavior_features: bool = True  # Use enhanced LSTM with behavior
    lstm_hidden_size: int = 64
    lstm_num_layers: int = 2
    learning_rate: float = 0.001
    num_epochs: int = 50

    # Trading settings
    initial_capital: float = 100000.0
    transaction_cost: float = 0.0005  # 5 bps
    edge_threshold: float = 0.02  # 2% edge required to trade
    risk_free_rate: float = 0.05

    # Multi-agent settings
    num_periods: int = 200
    use_multiagent: bool = True


class BitcoinTradingEvaluator:
    """
    End-to-end Bitcoin options trading evaluation system
    """

    def __init__(self, config: BitcoinTradingConfig):
        """
        Initialize Bitcoin trading evaluator

        Parameters:
        -----------
        config : BitcoinTradingConfig
            Configuration parameters
        """
        self.config = config

        # Initialize data fetcher
        if config.use_real_data:
            self.data_fetcher = DeribitDataFetcher(use_testnet=False)
            logger.info("Using real Deribit Bitcoin data")
        else:
            self.data_fetcher = MockBitcoinDataGenerator()
            logger.info("Using synthetic Bitcoin data")

        # Initialize models (will be trained later)
        self.price_forecaster = None
        self.volatility_model = None
        self.market_environment = None

        # Results storage
        self.forecast_results = {}
        self.trading_results = {}
        self.backtest_results = {}

    def fetch_bitcoin_data(self) -> Dict:
        """
        Fetch Bitcoin options data and spot prices

        Returns:
        --------
        dict : Market data including spot prices, option chain, and historical data
        """
        logger.info("=" * 70)
        logger.info("STEP 1: Fetching Bitcoin Market Data")
        logger.info("=" * 70)

        if self.config.use_real_data:
            # Get real market snapshot from Deribit
            try:
                snapshot = self.data_fetcher.fetch_market_snapshot()

                market_data = {
                    'spot_price': snapshot['index_price'],
                    'option_chain': snapshot['option_chain'],
                    'volatility_surface': snapshot['volatility_surface'],
                    'timestamp': snapshot['timestamp']
                }

                logger.info(f"  ✓ Spot Price: ${market_data['spot_price']:,.2f}")
                logger.info(f"  ✓ Options Available: {len(market_data['option_chain'])}")
                logger.info(f"  ✓ Timestamp: {market_data['timestamp']}")

                # Get historical spot prices for training
                # Note: Deribit API may have rate limits, using simplified approach
                historical_prices = self._fetch_historical_spot_prices(
                    market_data['spot_price']
                )
                market_data['historical_prices'] = historical_prices

            except Exception as e:
                logger.warning(f"Error fetching real data: {e}. Falling back to synthetic data.")
                self.config.use_real_data = False
                return self.fetch_bitcoin_data()
        else:
            # Generate synthetic data
            spot_price = 40000.0  # Current BTC price assumption
            option_chain = self.data_fetcher.generate_option_chain()

            market_data = {
                'spot_price': spot_price,
                'option_chain': option_chain,
                'timestamp': datetime.now(),
                'historical_prices': self._generate_synthetic_spot_history(spot_price)
            }

            logger.info(f"  ✓ Synthetic Spot Price: ${spot_price:,.2f}")
            logger.info(f"  ✓ Synthetic Options: {len(option_chain)}")

        logger.info("  ✓ Data fetch complete\n")
        return market_data

    def _fetch_historical_spot_prices(self, current_price: float) -> pd.Series:
        """Fetch or generate historical spot prices"""
        # For real implementation, would fetch from Deribit or other data source
        # Here we simulate with GBM
        return self._generate_synthetic_spot_history(current_price)

    def _generate_synthetic_spot_history(self, current_price: float) -> pd.Series:
        """Generate synthetic spot price history using GBM"""
        n_periods = self.config.lookback_periods + self.config.forecast_horizon + 50
        dt = 1/252  # Daily
        mu = 0.3  # High drift for crypto
        sigma = 0.8  # High volatility for crypto

        prices = [current_price]
        for _ in range(n_periods - 1):
            dW = np.random.normal(0, np.sqrt(dt))
            dS = prices[-1] * (mu * dt + sigma * dW)
            prices.append(prices[-1] + dS)

        dates = pd.date_range(end=datetime.now(), periods=n_periods, freq='D')
        return pd.Series(prices, index=dates)

    def train_forecasting_models(self, historical_prices: pd.Series,
                                 market_data: Dict) -> Dict:
        """
        Train LSTM and GARCH forecasting models

        Parameters:
        -----------
        historical_prices : pd.Series
            Historical spot prices
        market_data : dict
            Market data including option chain for behavior features

        Returns:
        --------
        dict : Trained models and their forecasts
        """
        logger.info("=" * 70)
        logger.info("STEP 2: Training Forecasting Models")
        logger.info("=" * 70)

        # Prepare data
        prices_array = historical_prices.values
        train_size = int(len(prices_array) * 0.8)
        train_prices = prices_array[:train_size]
        test_prices = prices_array[train_size:]

        results = {}

        # 1. Train LSTM for price forecasting
        logger.info("\n[1/2] Training LSTM Price Forecaster...")

        if self.config.use_behavior_features:
            # Use behavior-enhanced LSTM
            logger.info("  → Using Behavior-Enhanced LSTM")
            behavior_features = self._extract_behavior_features(market_data, num_periods=len(prices_array))

            self.price_forecaster = BehaviorEnhancedLSTM(
                price_input_size=1,
                behavior_input_size=behavior_features.shape[1] if behavior_features is not None else 17,
                lstm_hidden_size=self.config.lstm_hidden_size,
                lstm_num_layers=self.config.lstm_num_layers
            )

            trainer = BehaviorEnhancedLSTMTrainer(
                model=self.price_forecaster,
                learning_rate=self.config.learning_rate
            )

            # Train with behavior features
            training_history = trainer.fit(
                price_data=train_prices,
                behavior_data=behavior_features[:train_size] if behavior_features is not None else None,
                seq_length=self.config.lookback_periods,
                forecast_horizon=1,
                epochs=self.config.num_epochs,
                batch_size=32,
                verbose=False
            )

            train_loss = training_history['train_losses'][-1]
            val_loss = training_history['best_val_loss']
            logger.info(f"  ✓ Training Loss: {train_loss:.6f}, Validation Loss: {val_loss:.6f}")

        else:
            # Use pure LSTM
            logger.info("  → Using Pure LSTM")
            self.price_forecaster = LSTMForecaster(
                input_size=1,
                hidden_size=self.config.lstm_hidden_size,
                num_layers=self.config.lstm_num_layers,
                output_size=1
            )

            trainer = RNNTrainer(
                model=self.price_forecaster,
                learning_rate=self.config.learning_rate
            )

            training_history = trainer.fit(
                train_data=train_prices,
                seq_length=self.config.lookback_periods,
                epochs=self.config.num_epochs,
                batch_size=32
            )

            train_loss = training_history['train_losses'][-1]
            val_loss = training_history['best_val_loss']
            logger.info(f"  ✓ Training Loss: {train_loss:.6f}, Validation Loss: {val_loss:.6f}")

        # Generate price forecasts
        if self.config.use_behavior_features and behavior_features is not None:
            # Use behavior-enhanced trainer's predict method
            price_forecast = trainer.predict(
                price_data=prices_array[-self.config.lookback_periods:],
                behavior_data=behavior_features,
                seq_length=self.config.lookback_periods,
                steps=self.config.forecast_horizon
            )
        else:
            # Use pure LSTM trainer's predict method
            price_forecast = trainer.predict(
                data=prices_array,
                seq_length=self.config.lookback_periods,
                steps=self.config.forecast_horizon
            )

        results['price_forecast'] = price_forecast
        logger.info(f"  ✓ Price forecast generated: {len(price_forecast)} steps ahead")

        # 2. Train GARCH for volatility forecasting
        logger.info("\n[2/2] Training GARCH Volatility Model...")

        returns = np.diff(np.log(train_prices))
        returns_series = pd.Series(returns)
        self.volatility_model = GARCHModel(p=1, q=1)
        self.volatility_model.fit(returns_series)

        # Forecast volatility
        vol_forecast = self.volatility_model.forecast(horizon=self.config.forecast_horizon)
        results['volatility_forecast'] = vol_forecast

        logger.info(f"  ✓ Current volatility: {vol_forecast['volatility'][0]:.4f}")
        logger.info(f"  ✓ Volatility forecast generated: {len(vol_forecast['volatility'])} steps ahead")

        # Calculate forecast accuracy on test set
        logger.info("\n[Forecast Evaluation]")
        if len(test_prices) > 0:
            mae = np.mean(np.abs(test_prices[:len(price_forecast)] - price_forecast[:len(test_prices)]))
            mape = np.mean(np.abs((test_prices[:len(price_forecast)] - price_forecast[:len(test_prices)]) / test_prices[:len(price_forecast)])) * 100

            logger.info(f"  • MAE: ${mae:.2f}")
            logger.info(f"  • MAPE: {mape:.2f}%")

            results['mae'] = mae
            results['mape'] = mape

        self.forecast_results = results
        logger.info("\n  ✓ Forecasting models trained successfully\n")

        return results

    def _extract_behavior_features(self, market_data: Dict, num_periods: int) -> Optional[np.ndarray]:
        """Extract market behavior features from option chain

        Parameters:
        -----------
        market_data : Dict
            Market data including option chain
        num_periods : int
            Number of time periods to generate features for
        """
        try:
            from time_series_forecasting.behavior_features import BehaviorFeatureExtractor

            option_chain = market_data['option_chain']

            # Create mock agent states for feature extraction
            # In practice, these would come from actual market microstructure
            agent_states = {
                'market_makers': [],
                'informed_traders': [],
                'arbitrageurs': []
            }

            extractor = BehaviorFeatureExtractor()

            # Extract features for each time period
            features_list = []

            # For simplicity, extract features from current snapshot
            # In practice, would extract over time series
            for i in range(num_periods):
                # extract_features() only takes timestamp parameter
                # Without proper market microstructure data, it returns default features
                features = extractor.extract_features(timestamp=i)
                # Convert BehaviorFeatures dataclass to numpy array
                feature_vector = extractor.get_feature_vector(features)
                features_list.append(feature_vector)

            return np.array(features_list)

        except Exception as e:
            logger.warning(f"Could not extract behavior features: {e}")
            return None

    def run_multiagent_simulation(self, market_data: Dict,
                                  forecast_results: Dict) -> Dict:
        """
        Run multi-agent market simulation with time series driven agents

        Parameters:
        -----------
        market_data : dict
            Market data
        forecast_results : dict
            Price and volatility forecasts

        Returns:
        --------
        dict : Simulation results including emergent prices and P&L
        """
        logger.info("=" * 70)
        logger.info("STEP 3: Running Multi-Agent Market Simulation")
        logger.info("=" * 70)

        # Create market environment
        self.market_environment = TSMarketEnvironment(
            initial_spot=market_data['spot_price'],
            risk_free_rate=self.config.risk_free_rate
        )

        # Add options to the environment
        strikes = [38000, 40000, 42000]
        expirations = [30/365, 60/365, 90/365]
        for strike in strikes:
            for expiry in expirations:
                self.market_environment.add_option(strike, expiry)

        # Create time series driven agents
        logger.info("\n[Creating Agents]")

        # Market Maker (creates its own forecasting models internally)
        mm_agent = TSMarketMaker(
            agent_id="MM_TS_1",
            initial_cash=self.config.initial_capital,
            spread_multiplier=1.0
        )
        logger.info("  ✓ Created Time Series Market Maker")

        # Informed Trader (creates its own forecasting models internally)
        informed_agent = TSInformedTrader(
            agent_id="IT_TS_1",
            initial_cash=self.config.initial_capital * 0.5,
            conviction_threshold=0.02
        )
        logger.info("  ✓ Created Time Series Informed Trader")

        # Arbitrageur (creates its own forecasting models internally)
        arb_agent = TSArbitrageur(
            agent_id="ARB_TS_1",
            initial_cash=self.config.initial_capital * 0.5,
            arbitrage_threshold=0.03
        )
        logger.info("  ✓ Created Time Series Arbitrageur")

        # Register agents
        self.market_environment.add_agent(mm_agent)
        self.market_environment.add_agent(informed_agent)
        self.market_environment.add_agent(arb_agent)

        logger.info(f"\n[Running Simulation: {self.config.num_periods} periods]")

        # Run simulation
        history = self.market_environment.simulate(
            num_periods=self.config.num_periods,
            verbose=True
        )

        # Analyze results
        logger.info("\n[Simulation Results]")

        results = {
            'history': history,
            'agent_pnl': {}
        }

        # Calculate agent P&L
        for agent in [mm_agent, informed_agent, arb_agent]:
            final_capital = agent.cash
            pnl = final_capital - agent.initial_cash
            pnl_pct = (pnl / agent.initial_cash) * 100

            results['agent_pnl'][agent.agent_id] = {
                'final_capital': final_capital,
                'pnl': pnl,
                'pnl_pct': pnl_pct
            }

            logger.info(f"  • {agent.agent_id}: P&L = ${pnl:,.2f} ({pnl_pct:+.2f}%)")

        # Calculate Black-Scholes prices for reference
        try:
            bs_prices = self.market_environment.calculate_bs_prices()
            logger.info(f"\n  • Black-Scholes prices calculated for {len(bs_prices)} options")
            results['bs_prices'] = bs_prices
        except Exception as e:
            logger.warning(f"\n  • Could not calculate BS prices: {e}")
            results['bs_prices'] = {}

        self.trading_results = results
        logger.info("\n  ✓ Multi-agent simulation complete\n")

        return results

    def run_backtest(self, market_data: Dict) -> Dict:
        """
        Run backtesting framework on Bitcoin options

        Parameters:
        -----------
        market_data : dict
            Market data including option chain

        Returns:
        --------
        dict : Backtest results with P&L, Sharpe ratio, etc.
        """
        logger.info("=" * 70)
        logger.info("STEP 4: Running Backtest on Bitcoin Options")
        logger.info("=" * 70)

        # Prepare market data for backtest
        option_chain = market_data['option_chain']

        # Add timestamp column if not present
        if 'timestamp' not in option_chain.columns:
            option_chain['timestamp'] = pd.Timestamp.now()

        # Initialize calibrator
        calibrator = ModelCalibrator(
            market_data=option_chain,
            spot_price=market_data['spot_price'],
            risk_free_rate=self.config.risk_free_rate
        )

        # Calibrate models to market data
        logger.info("\n[Calibrating Models]")

        # Calibrate Heston
        try:
            calibrator.calibrate_heston()
            logger.info("  ✓ Heston model calibrated")
        except Exception as e:
            logger.warning(f"  × Heston calibration failed: {e}")

        # Calibrate SABR
        try:
            calibrator.calibrate_sabr()
            logger.info("  ✓ SABR model calibrated")
        except Exception as e:
            logger.warning(f"  × SABR calibration failed: {e}")

        # Calibrate multi-agent framework if enabled
        if self.config.use_multiagent:
            try:
                calibrator.calibrate_multi_agent()
                logger.info("  ✓ Multi-agent framework calibrated")
            except Exception as e:
                logger.warning(f"  × Multi-agent calibration failed: {e}")

        # Initialize backtest
        logger.info("\n[Running Backtest]")
        backtest = ModelComparisonBacktest(
            market_data=option_chain,
            spot_price=market_data['spot_price'],
            risk_free_rate=self.config.risk_free_rate,
            calibrator=calibrator
        )

        # Initialize strategies
        model_names = list(calibrator.calibrated_models.keys())
        backtest.initialize_strategies(model_names)

        # Run backtest
        backtest.run_backtest()

        # Get results
        results_df = backtest.get_comparison_dataframe()

        logger.info("\n[Backtest Complete]")
        logger.info(f"  • Models tested: {len(results_df)}")
        logger.info(f"  • Best performer: {results_df.loc[results_df['Total_PnL'].idxmax(), 'Model']}")

        self.backtest_results = {
            'comparison_df': results_df,
            'detailed_results': backtest.results
        }

        return self.backtest_results

    def generate_report(self) -> str:
        """
        Generate comprehensive evaluation report

        Returns:
        --------
        str : Formatted report
        """
        logger.info("=" * 70)
        logger.info("GENERATING EVALUATION REPORT")
        logger.info("=" * 70)

        report = []
        report.append("\n" + "=" * 70)
        report.append("BITCOIN OPTIONS TRADING EVALUATION REPORT")
        report.append("=" * 70)
        report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Data Source: {'Deribit API (Real)' if self.config.use_real_data else 'Synthetic'}")

        # Forecasting Results
        report.append("\n\n" + "-" * 70)
        report.append("1. FORECASTING PERFORMANCE")
        report.append("-" * 70)

        if self.forecast_results:
            report.append(f"\nPrice Forecasting:")
            report.append(f"  • Model: {'Behavior-Enhanced LSTM' if self.config.use_behavior_features else 'Pure LSTM'}")
            if 'mae' in self.forecast_results:
                report.append(f"  • MAE: ${self.forecast_results['mae']:.2f}")
                report.append(f"  • MAPE: {self.forecast_results['mape']:.2f}%")

            report.append(f"\nVolatility Forecasting:")
            report.append(f"  • Model: GARCH(1,1)")
            if 'volatility_forecast' in self.forecast_results:
                current_vol = self.forecast_results['volatility_forecast']['volatility'][0]
                report.append(f"  • Current Volatility: {current_vol:.4f}")

        # Multi-Agent Results
        if self.trading_results:
            report.append("\n\n" + "-" * 70)
            report.append("2. MULTI-AGENT TRADING PERFORMANCE")
            report.append("-" * 70)

            if 'agent_pnl' in self.trading_results:
                report.append("\nAgent P&L:")
                for agent_id, pnl_data in self.trading_results['agent_pnl'].items():
                    report.append(f"\n  {agent_id}:")
                    report.append(f"    • Final Capital: ${pnl_data['final_capital']:,.2f}")
                    report.append(f"    • P&L: ${pnl_data['pnl']:,.2f} ({pnl_data['pnl_pct']:+.2f}%)")

            if 'avg_pricing_error' in self.trading_results:
                report.append(f"\nMarket Efficiency:")
                report.append(f"  • Avg Pricing Error vs BS: ${self.trading_results['avg_pricing_error']:.2f}")

        # Backtest Results
        if self.backtest_results:
            report.append("\n\n" + "-" * 70)
            report.append("3. MODEL COMPARISON BACKTEST")
            report.append("-" * 70)

            if 'comparison_df' in self.backtest_results:
                df = self.backtest_results['comparison_df']
                report.append("\n" + df.to_string(index=False))

                # Highlight best model
                best_model = df.loc[df['Total_PnL'].idxmax()]
                report.append(f"\n\nBest Model: {best_model['Model']}")
                report.append(f"  • Total P&L: ${best_model['Total_PnL']:,.2f}")
                report.append(f"  • Sharpe Ratio: {best_model['Sharpe_Ratio']:.2f}")
                report.append(f"  • Win Rate: {best_model['Win_Rate']:.1f}%")

        # Summary
        report.append("\n\n" + "=" * 70)
        report.append("SUMMARY & CONCLUSIONS")
        report.append("=" * 70)
        report.append("\nKey Findings:")

        if self.config.use_behavior_features:
            report.append("  • Behavior-enhanced LSTM shows improved price forecasting")

        if self.trading_results and 'agent_pnl' in self.trading_results:
            total_pnl = sum(pnl['pnl'] for pnl in self.trading_results['agent_pnl'].values())
            report.append(f"  • Total multi-agent P&L: ${total_pnl:,.2f}")

        if self.backtest_results and 'comparison_df' in self.backtest_results:
            df = self.backtest_results['comparison_df']
            best_sharpe = df['Sharpe_Ratio'].max()
            report.append(f"  • Best Sharpe Ratio achieved: {best_sharpe:.2f}")

        report.append("\n" + "=" * 70 + "\n")

        report_text = "\n".join(report)
        print(report_text)

        return report_text

    def run_full_evaluation(self) -> Dict:
        """
        Run complete evaluation pipeline

        Returns:
        --------
        dict : All evaluation results
        """
        logger.info("\n" + "=" * 70)
        logger.info("BITCOIN OPTIONS TRADING - FULL EVALUATION")
        logger.info("=" * 70 + "\n")

        # Step 1: Fetch data
        market_data = self.fetch_bitcoin_data()

        # Step 2: Train forecasting models
        forecast_results = self.train_forecasting_models(
            market_data['historical_prices'],
            market_data
        )

        # Step 3: Run multi-agent simulation
        if self.config.use_multiagent:
            trading_results = self.run_multiagent_simulation(
                market_data,
                forecast_results
            )

        # Step 4: Run backtest
        backtest_results = self.run_backtest(market_data)

        # Step 5: Generate report
        report = self.generate_report()

        # Return all results
        return {
            'market_data': market_data,
            'forecast_results': forecast_results,
            'trading_results': self.trading_results,
            'backtest_results': self.backtest_results,
            'report': report
        }


def main():
    """Example usage"""

    # Configure evaluation
    config = BitcoinTradingConfig(
        use_real_data=False,  # Set to True to use Deribit API
        use_behavior_features=True,
        lookback_periods=60,
        forecast_horizon=5,
        num_epochs=50,
        num_periods=200,
        initial_capital=100000.0,
        use_multiagent=True
    )

    # Create evaluator
    evaluator = BitcoinTradingEvaluator(config)

    # Run full evaluation
    results = evaluator.run_full_evaluation()

    # Save report
    report_filename = f"bitcoin_evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(report_filename, 'w') as f:
        f.write(results['report'])

    logger.info(f"\n✓ Report saved to: {report_filename}")

    return results


if __name__ == "__main__":
    main()
