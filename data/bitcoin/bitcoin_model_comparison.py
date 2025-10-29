#!/usr/bin/env python3
"""
Bitcoin Options Model Comparison

Complete pipeline for comparing option pricing models on Bitcoin options:
- Heston Stochastic Volatility
- SABR Model
- Local Volatility (Dupire)
- Multi-Agent Framework

Evaluates models based on trading P&L, Sharpe ratio, and risk metrics
"""

import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging

# Import project modules
import sys
from pathlib import Path
# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from bitcoin.bitcoin_data_fetcher import DeribitDataFetcher, MockBitcoinDataGenerator
from models.model_calibrator import ModelCalibrator
from evaluation_modules.trading_backtest import ModelComparisonBacktest
from evaluation_modules.visualization import ModelComparisonVisualizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BitcoinModelComparisonPipeline:
    """
    Complete pipeline for Bitcoin options model comparison
    """

    def __init__(self,
                 use_real_data: bool = False,
                 spot_price: float = 40000,
                 risk_free_rate: float = 0.05):
        """
        Initialize pipeline

        Parameters:
        -----------
        use_real_data : bool
            Use real Deribit data (requires API access) or synthetic data
        spot_price : float
            Initial BTC spot price (used for synthetic data)
        risk_free_rate : float
            Risk-free interest rate
        """
        self.use_real_data = use_real_data
        self.spot_price = spot_price
        self.risk_free_rate = risk_free_rate

        self.market_data = None
        self.calibrator = None
        self.backtest = None
        self.results = None

    def fetch_market_data(self) -> pd.DataFrame:
        """
        Fetch or generate market data

        Returns:
        --------
        pd.DataFrame : Market option data
        """
        logger.info("=" * 80)
        logger.info("STEP 1: Fetching Market Data")
        logger.info("=" * 80)

        if self.use_real_data:
            logger.info("Fetching real data from Deribit API...")
            try:
                fetcher = DeribitDataFetcher(use_testnet=False)

                # Get current BTC price
                self.spot_price = fetcher.get_btc_index_price()
                logger.info(f"Current BTC price: ${self.spot_price:.2f}")

                # Get option chain
                option_chain = fetcher.get_option_chain(currency="BTC")

                # Add timestamp column
                option_chain['timestamp'] = pd.Timestamp.now()

                self.market_data = option_chain

                logger.info(f"Fetched {len(option_chain)} options from Deribit")

            except Exception as e:
                logger.error(f"Failed to fetch real data: {e}")
                logger.info("Falling back to synthetic data...")
                self.use_real_data = False

        if not self.use_real_data:
            logger.info("Generating synthetic Bitcoin options data...")
            generator = MockBitcoinDataGenerator(
                spot_price=self.spot_price,
                volatility=0.8  # High volatility for crypto
            )

            # Generate option chain (reduced for speed)
            maturities = [0.25, 0.5]  # 3M, 6M (reduced from 4 to 2)
            option_chain = generator.generate_option_chain(
                n_strikes=9,  # Reduced from 15 to 9
                maturities=maturities
            )

            # Simulate time series (3 snapshots for faster backtest)
            time_series = []
            for i in range(3):
                snapshot = option_chain.copy()
                snapshot['timestamp'] = pd.Timestamp.now() + timedelta(days=i)

                # Add some price dynamics
                price_shock = np.random.normal(0, 0.02)  # 2% daily vol
                snapshot['underlying_price'] = self.spot_price * (1 + price_shock * i)
                snapshot['mark_price'] = snapshot['mark_price'] * (1 + price_shock * i * 0.5)

                time_series.append(snapshot)

            self.market_data = pd.concat(time_series, ignore_index=True)

            logger.info(f"Generated {len(self.market_data)} synthetic option data points")

        logger.info(f"Market data shape: {self.market_data.shape}")
        logger.info(f"Columns: {list(self.market_data.columns)}")

        return self.market_data

    def calibrate_models(self):
        """
        Calibrate all models to market data
        """
        logger.info("\n" + "=" * 80)
        logger.info("STEP 2: Calibrating All Models")
        logger.info("=" * 80)

        # Use initial snapshot for calibration
        initial_data = self.market_data[
            self.market_data['timestamp'] == self.market_data['timestamp'].min()
        ].copy()

        logger.info(f"Calibrating on {len(initial_data)} options")

        # Initialize calibrator
        self.calibrator = ModelCalibrator(
            market_data=initial_data,
            spot_price=self.spot_price,
            risk_free_rate=self.risk_free_rate
        )

        # Calibrate all models
        calibration_results = self.calibrator.calibrate_all_models()

        # Print calibration summary
        logger.info("\n" + "-" * 60)
        logger.info("Calibration Summary:")
        logger.info("-" * 60)

        for model_name, params in calibration_results.items():
            if params:
                logger.info(f"\n{model_name.upper()}:")
                if isinstance(params, dict):
                    for key, value in params.items():
                        if key not in ['framework', 'model', 'by_expiry']:
                            logger.info(f"  {key}: {value}")

        return calibration_results

    def run_backtest(self):
        """
        Run trading backtest with all models
        """
        logger.info("\n" + "=" * 80)
        logger.info("STEP 3: Running Trading Backtest")
        logger.info("=" * 80)

        # Initialize backtest
        self.backtest = ModelComparisonBacktest(
            market_data=self.market_data,
            spot_price=self.spot_price,
            risk_free_rate=self.risk_free_rate,
            calibrator=self.calibrator
        )

        # Initialize strategies for all calibrated models
        model_names = ['heston', 'sabr', 'local_volatility', 'multi_agent']
        self.backtest.initialize_strategies(model_names)

        # Run backtest
        self.backtest.run_backtest()

        # Store results
        self.results = self.backtest.results

        return self.results

    def generate_report(self, save_dir: str = "./bitcoin_comparison_results"):
        """
        Generate comprehensive comparison report

        Parameters:
        -----------
        save_dir : str
            Directory to save results
        """
        logger.info("\n" + "=" * 80)
        logger.info("STEP 4: Generating Comprehensive Report")
        logger.info("=" * 80)

        # Create visualizations
        visualizer = ModelComparisonVisualizer(self.results, save_dir=save_dir)
        visualizer.generate_report()

        logger.info(f"\nAll results saved to: {save_dir}/")

    def run_complete_pipeline(self, save_dir: str = "./bitcoin_comparison_results"):
        """
        Run complete pipeline from data fetching to report generation

        Parameters:
        -----------
        save_dir : str
            Directory to save results
        """
        logger.info("\n" + "=" * 80)
        logger.info("BITCOIN OPTIONS MODEL COMPARISON PIPELINE")
        logger.info("=" * 80)
        logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Use real data: {self.use_real_data}")
        logger.info(f"Initial spot price: ${self.spot_price:.2f}")
        logger.info(f"Risk-free rate: {self.risk_free_rate:.2%}")
        logger.info("=" * 80)

        try:
            # Step 1: Fetch data
            self.fetch_market_data()

            # Step 2: Calibrate models
            self.calibrate_models()

            # Step 3: Run backtest
            self.run_backtest()

            # Step 4: Generate report
            self.generate_report(save_dir=save_dir)

            logger.info("\n" + "=" * 80)
            logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
            logger.info("=" * 80)
            logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

            # Print final summary
            self._print_final_summary()

            return self.results

        except Exception as e:
            logger.error(f"Pipeline failed with error: {e}")
            import traceback
            traceback.print_exc()
            raise

    def _print_final_summary(self):
        """Print final comparison summary"""
        logger.info("\n" + "=" * 80)
        logger.info("FINAL PERFORMANCE RANKING")
        logger.info("=" * 80)

        # Create comparison DataFrame
        comparison_data = []
        for model_name, result in self.results.items():
            comparison_data.append({
                'Model': model_name.upper(),
                'Total_PnL': result.total_pnl,
                'Sharpe': result.sharpe_ratio,
                'Max_DD': result.max_drawdown,
                'Win_Rate': result.win_rate
            })

        df = pd.DataFrame(comparison_data)

        # Rank by Sharpe ratio
        df_ranked = df.sort_values('Sharpe', ascending=False)

        logger.info("\nRanked by Sharpe Ratio:")
        for i, row in df_ranked.iterrows():
            logger.info(
                f"{row['Model']:20s} | "
                f"Sharpe: {row['Sharpe']:6.2f} | "
                f"PnL: ${row['Total_PnL']:8.2f} | "
                f"Win Rate: {row['Win_Rate']:5.1f}%"
            )

        logger.info("\n" + "=" * 80)


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Bitcoin Options Model Comparison Pipeline'
    )
    parser.add_argument(
        '--real-data',
        action='store_true',
        help='Use real Deribit data (requires API access)'
    )
    parser.add_argument(
        '--spot-price',
        type=float,
        default=40000,
        help='Initial BTC spot price for synthetic data (default: 40000)'
    )
    parser.add_argument(
        '--risk-free-rate',
        type=float,
        default=0.05,
        help='Risk-free interest rate (default: 0.05)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./bitcoin_comparison_results',
        help='Output directory for results (default: ./bitcoin_comparison_results)'
    )

    args = parser.parse_args()

    # Initialize pipeline
    pipeline = BitcoinModelComparisonPipeline(
        use_real_data=args.real_data,
        spot_price=args.spot_price,
        risk_free_rate=args.risk_free_rate
    )

    # Run complete pipeline
    try:
        results = pipeline.run_complete_pipeline(save_dir=args.output_dir)

        logger.info("\n✅ Pipeline completed successfully!")
        logger.info(f"📊 Results saved to: {args.output_dir}/")
        logger.info(f"📈 Check 'model_comparison_report.md' for detailed analysis")

        return 0

    except Exception as e:
        logger.error(f"\n❌ Pipeline failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
